"""
mc_robustness.py
================
Monte Carlo parameter robustness test for walk-forward optimised strategies.

PURPOSE
-------
A walk-forward optimisation selects the best parameters for each training
window independently. The question this module answers is:

    "If the optimiser had landed on slightly different parameters in each
     training window, would the stitched OOS performance remain acceptable?"

This is NOT a test of whether the strategy works — it tests whether the
walk-forward APPROACH is stable, i.e. whether the results depend critically
on hitting exact parameter values or whether performance holds across a
neighbourhood of plausible alternatives.

METHODOLOGY
-----------
1. Run normal walk-forward optimisation → best_params per window (done externally)
2. For each window, build a perturbation grid: ±PERTURB_PCT around each
   best parameter, keeping filter_mode fixed
3. Repeat N_SAMPLES times:
       a. Sample one param set per window independently (random.choice)
       b. Run full stitched OOS using those params — no retraining
       c. Compute metrics on the stitched equity curve
4. Analyse the distribution of CAGR, Sharpe, MaxDD, CalMAR across samples

Each sample represents "one possible way the optimisation could have gone".
Carry states differ between universes because different params in window N
produce different open positions at the end of window N, which then feed
into window N+1. This is correct and realistic behaviour.

KEY DESIGN DECISIONS
--------------------
- filter_mode is held fixed (not perturbed) — we test parameter sensitivity
  within the winning filter mode, not whether a different mode should have won
- Perturbation uses ±20% of the base value, not grid-step neighbours — this
  tests general parameter robustness, not whether our pre-chosen grid was
  well-spaced
- Integer params (fast, slow) are rounded; slow-fast >= 75 constraint enforced
- Parallelised at the universe level (universes are independent); windows
  within a universe are sequential due to carry state dependency
- Seeds are set per sample for full reproducibility

INTERPRETATION
--------------
Focus on tail metrics, not mean:
- p05 CAGR > 0          : strategy profitable in 95% of alternate realities
- p05 Sharpe > 0        : risk-adjusted positive in 95% of cases
- p05 MaxDD > -0.35     : drawdown bounded even in adverse param realisations
- p(CAGR < 0) < 0.10   : less than 10% chance of losing scenario

If these hold, the strategy is ROBUST — results do not depend on hitting
exact parameter values. If p05 CAGR is negative or p(loss) > 20%, the
strategy is FRAGILE — the backtest results may be a lucky parameter pick.

USAGE
-----
    from mc_robustness import run_monte_carlo_robustness, analyze_robustness

    # After normal walk-forward run:
    results_df = run_monte_carlo_robustness(
        best_params=wf_best_params,      # dict: window_id -> param_dict
        windows=wf_windows,              # list of window dicts from walk_forward
        df=df,                           # full price DataFrame
        cash_df=CASH,                    # full cash DataFrame
        vol_window=VOL_WINDOW,
        selected_mode=chosen_mode,
        n_samples=1000,
        n_jobs=N_JOBS,
        perturb_pct=0.20,
        seed=42
    )

    baseline = compute_metrics(wf_equity)
    analyze_robustness(results_df, baseline)

DEPENDENCIES
------------
Requires from main strategy module:
    run_strategy_with_trades
    compute_metrics
    compute_fund_breadth_signal   (only if fund filter used)
    
    
    
Now also includes bootstrapping of data

Reshuffle blocks of returns (120 or 250 days) to create many synthetic price 
histories with the same short-term autocorrelation structure but a different 
sequence of regimes. Re-run the full walk-forward procedure 
(including re-optimisation) on each synthetic history. 
Asks: "does the walk-forward procedure reliably find good parameters 
on histories that look like ours but aren't exactly ours?" 
This is testing the procedure, not just the parameters.
"""

import itertools
import logging
import random 
import time
import os
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# These must be imported from your main strategy module.
# Adjust the import path to match your project structure.
# ---------------------------------------------------------------------------
from strategy_test_library import (
                                run_strategy_with_trades,
                                compute_metrics,
                                compute_fund_breadth_signal,
                                walk_forward
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Parameters that are perturbed. filter_mode, fund_idx, fund_params,
# target_vol are held fixed — only continuous/integer strategy params vary.
PERTURB_PARAMS  = ["X", "Y", "fast", "slow", "stop_loss"]

# Parameters that must be rounded to integers after perturbation
INTEGER_PARAMS  = {"fast", "slow"}

# Minimum separation between slow and fast MA to prevent excessive crossovers
MIN_SLOW_FAST_GAP = 75

# Minimum values to prevent degenerate params after downward perturbation
MIN_VALUES = {
    "X":         0.02,
    "Y":         0.01,
    "fast":      10,
    "slow":      50,
    "stop_loss": 0.01,
}


# ---------------------------------------------------------------------------
# Step 1 — Build perturbation grid for a single window
# ---------------------------------------------------------------------------

def build_perturbation_grid(base_params, pct=0.20):
    """
    Build the cartesian product of ±pct perturbations around base_params.

    For each parameter in PERTURB_PARAMS, three candidate values are
    generated: base*(1-pct), base, base*(1+pct). Integer params are rounded.
    Combinations that violate slow-fast >= MIN_SLOW_FAST_GAP are excluded.
    Values below MIN_VALUES are clipped.

    Parameters
    ----------
    base_params : dict   — best params for one window (output from walk_forward)
    pct         : float  — perturbation fraction, default 0.20 (±20%)

    Returns
    -------
    list[dict]  — all valid perturbed parameter dicts for this window.
                  Each dict contains all keys from base_params, with
                  PERTURB_PARAMS values replaced by the perturbed values
                  and all other keys (filter_mode, fund_idx etc.) preserved.
    """
    grid = {}

    for name in PERTURB_PARAMS:
        val = base_params[name]
        lo  = max(MIN_VALUES.get(name, 0.0), val * (1 - pct))
        hi  = val * (1 + pct)

        if name in INTEGER_PARAMS:
            lo  = max(MIN_VALUES.get(name, 1), round(lo))
            hi  = round(hi)
            mid = int(val)
            variants = sorted(set([lo, mid, hi]))
        else:
            variants = [lo, val, hi]

        grid[name] = variants

    combos = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), vals))

        # Preserve non-perturbed keys unchanged
        for key in base_params:
            if key not in PERTURB_PARAMS:
                p[key] = base_params[key]

        # Enforce MA separation constraint
        if p.get("filter_mode") == "ma":
            if p["slow"] - p["fast"] < MIN_SLOW_FAST_GAP:
                continue

        combos.append(p)

    logging.debug(
        "build_perturbation_grid: %d valid combinations (pct=%.0f%%)",
        len(combos), pct * 100
    )
    return combos


# ---------------------------------------------------------------------------
# Step 2 — Build perturbation grids for all windows
# ---------------------------------------------------------------------------

def build_all_perturbation_grids(best_params, pct=0.20):
    """
    Build perturbation grids for every window.

    Parameters
    ----------
    best_params : dict  — {window_id: param_dict} from walk_forward
    pct         : float — perturbation fraction

    Returns
    -------
    dict  — {window_id: list[param_dict]}
    """
    return {
        w_id: build_perturbation_grid(params, pct=pct)
        for w_id, params in best_params.items()
    }


# ---------------------------------------------------------------------------
# Step 3 — Sample one universe (one param set per window)
# ---------------------------------------------------------------------------

def sample_universe(window_variants, rng):
    """
    Sample one parameter set per window to form a single 'universe'.

    Parameters
    ----------
    window_variants : dict   — {window_id: list[param_dict]}
    rng             : random.Random instance for reproducibility

    Returns
    -------
    dict  — {window_id: param_dict}
    """
    return {
        w_id: rng.choice(variants)
        for w_id, variants in window_variants.items()
    }


# ---------------------------------------------------------------------------
# Step 4 — Run one universe: stitch OOS equity curve with perturbed params
# ---------------------------------------------------------------------------

def run_universe(universe, windows, df, cash_df, vol_window,
                 selected_mode, funds_df=None, price_col="Zamkniecie"):
    """
    Stitch the full OOS equity curve using perturbed params — no retraining.

    This mirrors the stitching logic in walk_forward exactly. Carry states
    are propagated between windows, so different parameter choices in window N
    correctly affect the carry state entering window N+1.

    Parameters
    ----------
    universe     : dict         — {window_id: param_dict} from sample_universe
    windows      : list[dict]   — window definitions from walk_forward, each
                                  containing: warmup_start, train_end (=test_start),
                                  test_end, warmup_df
    df           : DataFrame    — full price series
    cash_df      : DataFrame    — full cash series
    vol_window   : int          — rolling vol window
    selected_mode: str          — 'full', 'vol_entry', or 'vol_dynamic'
    funds_df     : DataFrame|None — fund panel (only needed if fund filter used)

    Returns
    -------
    equity : pd.Series   — stitched OOS equity curve (starts at 1.0)
    trades : list[dict]  — all trades across windows
    """
    equity_parts = []
    all_trades   = []
    carry_state  = None
    prev_equity  = 1.0

    for w_id, w in enumerate(windows):
        params = universe[w_id]

        # --- Prepare OOS slice with warmup prepended ---
        test_start   = w["test_start"]
        test_end     = w["test_end"]
        warmup_start = w["warmup_start"]

        oos_slice = df.loc[
            (df.index >= test_start) &
            (df.index <  test_end)
        ].copy()

        warmup_slice = df.loc[
            (df.index >= warmup_start) &
            (df.index <  test_start)
        ].copy()

        cash_slice = cash_df.reindex(
            pd.date_range(warmup_start, test_end, freq="B"),
            method="ffill"
        ).reindex(
            pd.concat([warmup_slice, oos_slice]).index,
            method="ffill"
        )

        # --- Fund signal (if applicable) ---
        fund_signal = None
        if params.get("filter_mode") == "fund" and funds_df is not None:
            fund_slice = funds_df.loc[
                (funds_df.index >= warmup_start) &
                (funds_df.index <  test_end)
            ]
            full_signal = compute_fund_breadth_signal(
                fund_slice, **params["fund_params"]
            )
            fund_signal = full_signal.loc[full_signal.index >= test_start]

        # --- Run strategy with fixed params ---
        result = run_strategy_with_trades(
            oos_slice,
            X             = params["X"],
            Y             = params["Y"],
            fast          = int(params["fast"]),
            slow          = int(params["slow"]),
            stop_loss     = params["stop_loss"],
            target_vol    = params.get("target_vol"),
            position_mode = selected_mode,
            vol_window    = vol_window,
            use_momentum  = (params.get("filter_mode") == "mom"),
            cash_df       = cash_slice,
            initial_state = carry_state,
            warmup_df     = warmup_slice if len(warmup_slice) > 0 else None,
            fund_signal   = fund_signal,
            price_col     = price_col
        )

        if result is None or result[0] is None:
            # Strategy returned None (insufficient data) — skip window
            logging.debug("run_universe: window %d returned None, skipping", w_id)
            continue

        result_df, _, trades_df, end_state = result

        # --- Chain-link equity ---
        eq = result_df["equity"]
        eq = eq / eq.iloc[0] * prev_equity

        equity_parts.append(eq)
        if trades_df is not None and len(trades_df) > 0:
            all_trades.extend(trades_df.to_dict("records"))

        prev_equity = eq.iloc[-1]
        carry_state = end_state

    if not equity_parts:
        return None, []

    return pd.concat(equity_parts), all_trades


# ---------------------------------------------------------------------------
# Step 5 — Single sample runner (for parallelisation)
# ---------------------------------------------------------------------------

def _run_single_sample(seed, window_variants, windows, df, cash_df,
                        vol_window, selected_mode, funds_df, price_col):
    """
    Run one Monte Carlo sample with a fixed seed. Designed to be called
    via joblib.Parallel — all arguments must be picklable.

    Parameters
    ----------
    seed : int  — random seed for this sample (ensures reproducibility)

    Returns
    -------
    dict  — metrics for this sample, plus the seed for traceability
    """
    rng = random.Random(seed)
    universe = sample_universe(window_variants, rng)

    equity, trades = run_universe(
        universe, windows, df, cash_df,
        vol_window, selected_mode, funds_df, 
        price_col=price_col
    )

    if equity is None:
        return None

    metrics         = compute_metrics(equity)
    metrics         = {k: float(v) for k, v in metrics.items()}
    metrics["seed"] = seed
    return metrics


# ---------------------------------------------------------------------------
# Step 6 — Main Monte Carlo loop
# ---------------------------------------------------------------------------

def run_monte_carlo_robustness(
    best_params,
    windows,
    df,
    cash_df,
    vol_window,
    selected_mode,
    funds_df    = None,
    n_samples   = 1000,
    n_jobs      = 1,
    perturb_pct = 0.20,
    seed        = 42,
    price_col="Zamkniecie"
):
    """
    Run the full Monte Carlo robustness test.

    Parameters
    ----------
    best_params   : dict        — {window_id: param_dict} from walk_forward.
                                  Each param_dict must contain: X, Y, fast,
                                  slow, stop_loss, filter_mode, and optionally
                                  fund_idx, fund_params, target_vol.
    windows       : list[dict]  — window definitions. Each dict must contain:
                                  warmup_start, test_start, test_end.
                                  Typically extracted from walk_forward internals
                                  or reconstructed from wf_results DataFrame.
    df            : DataFrame   — full price series (same as passed to walk_forward)
    cash_df       : DataFrame   — full cash series
    vol_window    : int         — rolling volatility window (bars)
    selected_mode : str         — position sizing mode: 'full', 'vol_entry',
                                  or 'vol_dynamic'
    funds_df      : DataFrame   — fund NAV panel (only needed if any window
                                  uses fund filter). Pass None if MA/mom only.
    n_samples     : int         — number of Monte Carlo universes to simulate.
                                  1000 gives stable p05/p95 estimates.
                                  2000 for publication-quality results.
    n_jobs        : int         — number of parallel jobs (joblib). Use N_JOBS
                                  from main script. Set to 1 for debugging.
    perturb_pct   : float       — perturbation fraction, default 0.20 (±20%).
                                  Larger values test wider robustness but may
                                  produce many degenerate param combinations.
    seed          : int         — master random seed. Sample i uses seed+i,
                                  ensuring full reproducibility.

    Returns
    -------
    pd.DataFrame  — one row per sample, columns: CAGR, Vol, Sharpe, MaxDD,
                    CalMAR, seed. Failed samples (None) are dropped.
    """
    t_start = time.time()

    logging.info("=" * 80)
    logging.info("MONTE CARLO ROBUSTNESS TEST")
    logging.info("=" * 80)
    logging.info(
        "Config: n_samples=%d | perturb_pct=%.0f%% | n_jobs=%d | seed=%d",
        n_samples, perturb_pct * 100, n_jobs, seed
    )
    _cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    logging.info(
        "Parallel computing: %d logical cores detected, using %d jobs.",
        _cpu_count, N_JOBS
    )
    # --- Build perturbation grids --- 
    window_variants = build_all_perturbation_grids(best_params, pct=perturb_pct)

    for w_id, variants in window_variants.items():
        logging.info(
            "Window %d (%s filter): %d perturbation variants",
            w_id,
            best_params[w_id].get("filter_mode", "?"),
            len(variants)
        )

    # --- Time a single sample before full run ---
    logging.info("Timing single sample...")
    t0 = time.time()
    _run_single_sample(
        seed=seed,
        window_variants=window_variants,
        windows=windows,
        df=df,
        cash_df=cash_df,
        vol_window=vol_window,
        selected_mode=selected_mode,
        funds_df=funds_df,
        price_col     = price_col
    )
    single_ms = (time.time() - t0) * 1000
    est_total_s = single_ms * n_samples / max(n_jobs, 1) / 1000
    logging.info(
        "Single sample: %.0fms -> estimated total: %.0fs (~%.1f min) on %d jobs",
        single_ms, est_total_s, est_total_s / 60, n_jobs
    )

    # --- Run Monte Carlo ---
    try:
        raw_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_run_single_sample)(
                seed          = seed + i,
                window_variants = window_variants,
                windows       = windows,
                df            = df,
                cash_df       = cash_df,
                vol_window    = vol_window,
                selected_mode = selected_mode,
                funds_df      = funds_df,
                price_col     = price_col
            )
            for i in range(n_samples)
        )
        logging.info("Monte Carlo completed using multiprocessing backend.")
    except Exception as e:
        logging.warning(
            "Parallel execution failed (%s) — falling back to sequential.", e
        )
        raw_results = [
            _run_single_sample(
                seed          = seed + i,
                window_variants = window_variants,
                windows       = windows,
                df            = df,
                cash_df       = cash_df,
                vol_window    = vol_window,
                selected_mode = selected_mode,
                funds_df      = funds_df,
            )
            for i in range(n_samples)
        ]

    # --- Collect results ---
    valid = [r for r in raw_results if r is not None]
    n_failed = n_samples - len(valid)
    if n_failed > 0:
        logging.warning("%d samples returned None and were dropped.", n_failed)

    results_df = pd.DataFrame(valid)

    elapsed = time.time() - t_start
    logging.info(
        "Monte Carlo finished: %d valid samples in %.1fs (%.1f min)",
        len(valid), elapsed, elapsed / 60
    )

    return results_df


# ---------------------------------------------------------------------------
# Step 7 — Analysis and reporting
# ---------------------------------------------------------------------------

def analyze_robustness(results_df, baseline_metrics, thresholds=None):
    """
    Print a formatted robustness report comparing the Monte Carlo distribution
    to the baseline (actual walk-forward result).

    Parameters
    ----------
    results_df       : pd.DataFrame  — output from run_monte_carlo_robustness
    baseline_metrics : dict          — metrics from the actual WF run.
                                       Keys: CAGR, Vol, Sharpe, MaxDD, CalMAR
    thresholds       : dict|None     — custom pass/fail thresholds. If None,
                                       uses conservative defaults suitable for
                                       a single-index Polish equity strategy:
                                         CAGR  p05 > 0.00
                                         Sharpe p05 > 0.00
                                         MaxDD  p05 > -0.35

    Returns
    -------
    dict  — summary statistics per metric, plus overall verdict
    """
    if thresholds is None:
        thresholds = {
            "CAGR":   {"p05_min": 0.00,  "label": "p05 CAGR > 0%"},
            "Sharpe": {"p05_min": 0.00,  "label": "p05 Sharpe > 0"},
            "MaxDD":  {"p05_min": -0.35, "label": "p05 MaxDD > -35%"},
        }

    metrics_to_show = ["CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR"]
    summary = {}

    logging.info("=" * 80)
    logging.info("MONTE CARLO ROBUSTNESS REPORT  (n=%d samples)", len(results_df))
    logging.info("=" * 80)

    header = f"{'Metric':<10} {'Baseline':>10} {'Mean':>10} {'p05':>10} "
    header += f"{'p25':>10} {'Median':>10} {'p75':>10} {'p95':>10} {'P(worse)':>10}"
    logging.info(header)
    logging.info("-" * 90)

    for col in metrics_to_show:
        if col not in results_df.columns:
            continue

        base  = baseline_metrics.get(col, float("nan"))
        mean  = results_df[col].mean()
        p05   = results_df[col].quantile(0.05)
        p25   = results_df[col].quantile(0.25)
        med   = results_df[col].median()
        p75   = results_df[col].quantile(0.75)
        p95   = results_df[col].quantile(0.95)

        # "worse than baseline" — for MaxDD, worse = more negative
        if col == "MaxDD":
            p_worse = (results_df[col] < base).mean()
        else:
            p_worse = (results_df[col] < base).mean()

        summary[col] = {
            "baseline": base,
            "mean": mean, "p05": p05, "p25": p25,
            "median": med, "p75": p75, "p95": p95,
            "p_worse_than_baseline": p_worse,
        }

        # Format percentages for CAGR, MaxDD; 2dp for Sharpe/CalMAR
        fmt = "{:.1%}" if col in ("CAGR", "MaxDD", "Vol") else "{:.2f}"
        row = (
            f"{col:<10} "
            f"{fmt.format(base):>10} "
            f"{fmt.format(mean):>10} "
            f"{fmt.format(p05):>10} "
            f"{fmt.format(p25):>10} "
            f"{fmt.format(med):>10} "
            f"{fmt.format(p75):>10} "
            f"{fmt.format(p95):>10} "
            f"{p_worse:>9.1%}"
        )
        logging.info(row)

    logging.info("-" * 90)

    # --- Pass/fail verdict ---
    passes = []
    fails  = []
    for metric, cfg in thresholds.items():
        if metric not in summary:
            continue
        p05_val = summary[metric]["p05"]
        if p05_val >= cfg["p05_min"]:
            passes.append(cfg["label"])
        else:
            fails.append(
                f"{cfg['label']}  [actual p05={p05_val:.3f}]"
            )

    # Additional check: P(CAGR < 0)
    p_loss = (results_df["CAGR"] < 0).mean()
    summary["p_loss"] = p_loss
    loss_label = f"P(CAGR < 0) = {p_loss:.1%}"
    if p_loss < 0.10:
        passes.append(loss_label)
    else:
        fails.append(f"P(CAGR < 0) < 10%  [actual {p_loss:.1%}]")

    # Check: baseline should not be a top-decile outlier in its own distribution
    for metric in ["CAGR", "Sharpe"]:
        if metric not in summary:
            continue
        p_worse = summary[metric]["p_worse_than_baseline"]
        if p_worse > 0.85:
            fails.append(
                f"P({metric} worse than baseline) = {p_worse:.1%}  "
                f"[baseline is top-decile outlier — suggests overfitting]"
            )
        else:
            passes.append(
                f"P({metric} worse than baseline) = {p_worse:.1%}  [baseline not an outlier]"
            )

    # Check: median CAGR should be meaningfully positive
    median_cagr = summary["CAGR"]["median"]
    if median_cagr < 0.02:
        fails.append(
            f"Median CAGR = {median_cagr:.1%}  [typical universe performs poorly, threshold 2%]"
        )
    else:
        passes.append(f"Median CAGR = {median_cagr:.1%}  [> 2% threshold]")

    # Check: result should not be highly parameter-sensitive
    cagr_range  = summary["CAGR"]["p95"] - summary["CAGR"]["p05"]
    cagr_median = summary["CAGR"]["median"]
    if cagr_median > 0 and cagr_range / cagr_median > 3.0:
        fails.append(
            f"CAGR p95-p05 range = {cagr_range:.1%}, median = {cagr_median:.1%}  "
            f"[ratio {cagr_range/cagr_median:.1f}x > 3.0 — highly parameter-sensitive]"
        )
    elif cagr_median > 0:
        passes.append(
            f"CAGR p95-p05 range/median = {cagr_range/cagr_median:.1f}x  [< 3.0 threshold]"
        )

    verdict = "ROBUST" if not fails else "FRAGILE"

    logging.info("")
    logging.info("VERDICT: %s", verdict)
    if passes:
        for p in passes:
            logging.info("  PASS  %s", p)
    if fails:
        for f in fails:
            logging.info("  FAIL  %s", f)
    logging.info("=" * 80)

    summary["verdict"] = verdict
    return summary

# ---------------------------------------------------------------------------
# Utility — extract window definitions from walk_forward wf_results DataFrame
# ---------------------------------------------------------------------------

def extract_windows_from_wf_results(wf_results, train_years=8):
    """
    Reconstruct window definitions from the wf_results DataFrame produced
    by walk_forward. This avoids having to modify walk_forward to return
    window metadata separately.

    Parameters
    ----------
    wf_results   : pd.DataFrame  — per-window results from walk_forward
    train_years  : int           — training window length in years (default 8)

    Returns
    -------
    list[dict]  — one dict per window with keys:
                  warmup_start, test_start, test_end, window_id
    """
    windows = []
    for i, row in wf_results.iterrows():
        test_start   = pd.Timestamp(row["TestStart"])
        test_end     = pd.Timestamp(row["TestEnd"])
        train_start  = pd.Timestamp(row["TrainStart"])
        # Warmup begins at train_start — same as the WF loop uses
        warmup_start = train_start
        windows.append({
            "window_id":    i,
            "warmup_start": warmup_start,
            "test_start":   test_start,
            "test_end":     test_end,
        })
    return windows


def extract_best_params_from_wf_results(wf_results):
    """
    Extract best_params dict from the wf_results DataFrame.

    Parameters
    ----------
    wf_results : pd.DataFrame — per-window results from walk_forward

    Returns
    -------
    dict  — {window_index: param_dict}
    """
    best_params = {}
    for i, row in wf_results.iterrows():
        best_params[i] = {
            "filter_mode": row["filter_mode"],
            "fund_idx":    row.get("fund_idx"),
            "fund_params": row.get("fund_params"),
            "X":           float(row["X"]),
            "Y":           float(row["Y"]),
            "fast":        int(row["fast"]),
            "slow":        int(row["slow"]),
            "stop_loss":   float(row["stop_loss"]),
            "target_vol":  row.get("target_vol"),
        }
    return best_params


#=====================================================================
# DATA HISTORY BOOTSTRAP
#=====================================================================


def block_bootstrap_history(df, price_col, cash_col, block_size=250, seed=None):
    """
    Reshuffle (index_return, cash_return) pairs in blocks.
    Preserves within-block equity/rate correlation.
    Rate jumps at block boundaries are accepted as necessary artefact.
    
    Returns synthetic df with reconstructed price and cash columns,
    same index as input.
    """
    rng = np.random.default_rng(seed)
    
    # Work in return space for both series
    idx_ret  = df[price_col].pct_change().dropna()
    cash_ret = df[cash_col].pct_change().dropna()
    
    # Align — both series may have different start dates
    aligned = pd.concat([idx_ret, cash_ret], axis=1).dropna()
    aligned.columns = ["idx_ret", "cash_ret"]
    
    n = len(aligned)
    # Build non-overlapping blocks from aligned returns
    block_starts = list(range(0, n - block_size + 1, block_size))
    sampled_idx  = rng.choice(len(block_starts), 
                               size=len(block_starts), 
                               replace=True)
    
    reshuffled = pd.concat(
        [aligned.iloc[block_starts[i] : block_starts[i] + block_size]
         for i in sampled_idx]
    ).iloc[:n]
    reshuffled.index = aligned.index
    
    # Reconstruct price series from reshuffled returns, starting at 1.0
    synthetic_price = (1 + reshuffled["idx_ret"]).cumprod()
    synthetic_cash  = (1 + reshuffled["cash_ret"]).cumprod()
    
    # Build output df preserving all original columns
    out = df.loc[aligned.index].copy()
    out[price_col] = synthetic_price.values
    out[cash_col]  = synthetic_cash.values
 
    logging.debug(
    "block_bootstrap_history: combined=%d, aligned=%d, out=%d",
    len(df), len(aligned), len(out)
)
    
    return out

def _bootstrap_single_sample(
    i, combined, df, cash_df, price_col, cash_price_col, block_size, wf_kwargs
):
    """
    Single bootstrap sample — designed for joblib.Parallel dispatch.
    Fully self-contained, no shared mutable state.
    """
    try:
        synthetic = block_bootstrap_history(
            combined, price_col, "cash_price",
            block_size=block_size, seed=i
        )

        n_synth = len(synthetic)
        synthetic_df = df.iloc[:n_synth].copy()
        synthetic_df[price_col] = synthetic[price_col].values
        synthetic_cash = cash_df.reindex(synthetic_df.index).ffill()
        synthetic_cash[cash_price_col] = synthetic["cash_price"].values
        
        logging.debug(
            "Bootstrap %d: synthetic_df=%d, synthetic_cash=%d, "
            "synthetic_df.index[0]=%s, synthetic_df.index[-1]=%s",
            i, len(synthetic_df), len(synthetic_cash),
            synthetic_df.index[0], synthetic_df.index[-1]
            )   
        
        
        equity, wf_res, _ = walk_forward(
            synthetic_df, cash_df=synthetic_cash, **wf_kwargs
        )

        if equity is None or equity.empty:
            return None

        m = compute_metrics(equity)
        return {
            "sample":  i,
            "CAGR":    m["CAGR"],
            "Sharpe":  m["Sharpe"],
            "MaxDD":   m["MaxDD"],
            "CalMAR":  m["CalMAR"],
            "Sortino": m.get("Sortino", np.nan)
        }

    except Exception as e:
        import traceback
        logging.warning("Bootstrap sample %d failed: %s\n%s", i, e, traceback.format_exc())
        return None


def run_block_bootstrap_robustness(
    df, cash_df,
    price_col="Zamkniecie",
    cash_price_col="Zamkniecie",
    n_samples=500,
    block_size=250,
    **wf_kwargs
):
    """
    Run full walk-forward re-optimisation on n_samples block-bootstrapped
    synthetic histories. Bootstrap samples are evaluated in parallel using
    the same three-tier fallback as walk_forward (loky -> threading -> sequential).

    Each sample independently reshuffles (index_return, cash_return) blocks,
    runs the full walk-forward optimisation, and returns OOS metrics.
    The distribution of results answers: would the walk-forward procedure
    perform well on a different plausible market history?
    """
    _cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    logging.info("=" * 80)
    logging.info("BLOCK BOOTSTRAP ROBUSTNESS TEST")
    logging.info("=" * 80)
    logging.info(
        "Config: n_samples=%d | block_size=%d days | n_jobs=%d",
        n_samples, block_size, N_JOBS
    )

    # Merge price and cash into single df for joint reshuffling
    combined = df[[price_col]].copy()
    combined["cash_price"] = cash_df[cash_price_col].reindex(
        df.index).ffill()

    # Estimate runtime from single sample
    logging.info("Timing single sample...")
    t0 = time.time()
    _bootstrap_single_sample(
        0, combined, df, cash_df, price_col, cash_price_col,
        block_size, wf_kwargs
    )
    single_time = time.time() - t0
    estimated  = single_time * n_samples / N_JOBS
    logging.info(
        "Single sample: %.0fms -> estimated total: %.0fs (~%.1f min) on %d jobs",
        single_time * 1000, estimated, estimated / 60, N_JOBS
    )

    # Three-tier parallel fallback — mirrors walk_forward parallelisation
    results_list = None
    for backend, n_jobs, label in [
        ("loky",      N_JOBS, "multiprocessing"),
        ("threading", N_JOBS, "threading"),
        (None,        1,      "sequential"),
    ]:
        try:
            if backend is None:
                results_list = [
                    _bootstrap_single_sample(
                        i, combined, df, cash_df,
                        price_col, cash_price_col,
                        block_size, wf_kwargs
                    )
                    for i in range(n_samples)
                ]
            else:
                results_list = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_bootstrap_single_sample)(
                        i, combined, df, cash_df,
                        price_col, cash_price_col,
                        block_size, wf_kwargs
                    )
                    for i in range(n_samples)
                )
            logging.info(
                "Block bootstrap completed using %s backend (%d jobs).",
                label, n_jobs
            )
            break

        except Exception as e:
            logging.warning(
                "Bootstrap backend '%s' failed: %s — trying next option.",
                label, e
            )
            results_list = None

    if results_list is None:
        logging.error("All bootstrap backends failed.")
        return pd.DataFrame()

    # Filter failed samples
    results_df = pd.DataFrame(
        [r for r in results_list if r is not None]
    )
    n_valid   = len(results_df)
    n_failed  = n_samples - n_valid
    if n_failed > 0:
        logging.warning(
            "%d/%d bootstrap samples failed and were excluded.", 
            n_failed, n_samples
        )

    if results_df.empty:
        logging.error("No valid bootstrap samples — cannot report.")
        return results_df

    # Summary report
    logging.info("=" * 80)
    logging.info(
        "BLOCK BOOTSTRAP ROBUSTNESS REPORT  (n=%d valid samples, block=%d days)",
        n_valid, block_size
    )
    logging.info("=" * 80)
    logging.info(
        "%-10s %8s %8s %8s %8s %8s %8s %8s",
        "Metric", "Mean", "p05", "p25", "Median", "p75", "p95", "P(>0%)"
    )
    logging.info("-" * 80)
    for col in ["CAGR", "Sharpe", "MaxDD", "CalMAR", "Sortino"]:
        s = results_df[col]
        logging.info(
            "%-10s %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.2f%% %8.1f%%",
            col,
            s.mean()*100, s.quantile(0.05)*100, s.quantile(0.25)*100,
            s.median()*100, s.quantile(0.75)*100, s.quantile(0.95)*100,
            (s > 0).mean()*100
        )
    logging.info("-" * 80)

    # Key verdict thresholds
    p_cagr_pos  = (results_df["CAGR"] > 0.00).mean()
    p_cagr_3pct = (results_df["CAGR"] > 0.03).mean()
    p_maxdd_ok  = (results_df["MaxDD"] > -0.35).mean()
    p_beats_bh  = (results_df["Sharpe"] > 0.25).mean()  # baseline B&H Sharpe

    logging.info("P(CAGR > 0%%):    %5.1f%%", p_cagr_pos  * 100)
    logging.info("P(CAGR > 3%%):    %5.1f%%", p_cagr_3pct * 100)
    logging.info("P(MaxDD > -35%%): %5.1f%%", p_maxdd_ok  * 100)
    logging.info("P(Sharpe > B&H):  %5.1f%%", p_beats_bh  * 100)
    logging.info("=" * 80)

    return results_df


# ---------------------------------------------------------------------------
# Bootstrap analysis and reporting
# ---------------------------------------------------------------------------

def analyze_bootstrap(results_df, baseline_metrics, thresholds=None):
    """
    Print a formatted report for the block bootstrap robustness test.

    The bootstrap test asks a different question from the Monte Carlo test:
    not "are these parameters on a robust plateau?" but "does the walk-forward
    procedure reliably find a good strategy on alternative plausible histories?"

    Because block reshuffling suppresses multi-year trends, the bootstrap
    is structurally biased against trend-following. Thresholds are therefore
    set lower than the MC thresholds — a borderline pass here is meaningful.

    Parameters
    ----------
    results_df       : pd.DataFrame  — output from run_block_bootstrap_robustness.
                                       One row per synthetic history, columns:
                                       CAGR, Vol, Sharpe, MaxDD, CalMAR.
    baseline_metrics : dict          — metrics from the actual WF run on real data.
                                       Keys: CAGR, Vol, Sharpe, MaxDD, CalMAR.
    thresholds       : dict|None     — custom pass/fail thresholds. If None,
                                       uses defaults appropriate for a single-index
                                       Polish equity trend-following strategy:
                                         CAGR  p05 > -0.01  (tolerates mild losses
                                                              on worst synthetic histories)
                                         Sharpe p05 > -0.10  (risk-adjusted near-zero)
                                         MaxDD  p05 > -0.40  (wider than MC — bootstrap
                                                              can produce concentrated
                                                              drawdown periods)
                                         P(CAGR < 0) < 0.20  (strategy loss-making on
                                                              fewer than 1 in 5 histories)

    Returns
    -------
    dict  — summary statistics per metric, plus overall verdict
    """
    if thresholds is None:
        thresholds = {
            "CAGR":   {"p05_min": -0.01, "label": "p05 CAGR > -1%"},
            "Sharpe": {"p05_min": -0.10, "label": "p05 Sharpe > -0.10"},
            "MaxDD":  {"p05_min": -0.40, "label": "p05 MaxDD > -40%"},
            "p_loss": {"max":      0.20,  "label": "P(CAGR < 0) < 20%"},
        }

    metrics_to_show = ["CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR"]
    summary = {}

    logging.info("=" * 80)
    logging.info("BLOCK BOOTSTRAP ROBUSTNESS REPORT  (n=%d samples)", len(results_df))
    logging.info(
        "Note: block reshuffling suppresses multi-year trends — "
        "thresholds are set lower than MC to account for this structural bias."
    )
    logging.info("=" * 80)

    header = f"{'Metric':<10} {'Baseline':>10} {'Mean':>10} {'p05':>10} "
    header += f"{'p25':>10} {'Median':>10} {'p75':>10} {'p95':>10} {'P(CAGR<0)':>10}"
    logging.info(header)
    logging.info("-" * 90)

    p_loss = (results_df["CAGR"] < 0).mean() if "CAGR" in results_df.columns else float("nan")

    for col in metrics_to_show:
        if col not in results_df.columns:
            continue

        base = baseline_metrics.get(col, float("nan"))
        mean = results_df[col].mean()
        p05  = results_df[col].quantile(0.05)
        p25  = results_df[col].quantile(0.25)
        med  = results_df[col].median()
        p75  = results_df[col].quantile(0.75)
        p95  = results_df[col].quantile(0.95)

        summary[col] = {
            "baseline": base,
            "mean":     mean,
            "p05":      p05,
            "p25":      p25,
            "median":   med,
            "p75":      p75,
            "p95":      p95,
        }

        # P(CAGR<0) shown only on the CAGR row, blank on others
        p_loss_display = f"{p_loss:.1%}" if col == "CAGR" else ""

        if col in ("CAGR", "Sharpe", "CalMAR"):
            fmt = lambda v: f"{v:.2f}" if col != "CAGR" else f"{v:.1%}"
        elif col == "MaxDD":
            fmt = lambda v: f"{v:.1%}"
        else:
            fmt = lambda v: f"{v:.1%}"

        row = (
            f"{col:<10} {fmt(base):>10} {fmt(mean):>10} {fmt(p05):>10} "
            f"{fmt(p25):>10} {fmt(med):>10} {fmt(p75):>10} {fmt(p95):>10} "
            f"{p_loss_display:>10}"
        )
        logging.info(row)

    summary["p_loss"] = p_loss
    logging.info("-" * 90)

    # ------------------------------------------------------------------
    # Pass / fail checks
    # ------------------------------------------------------------------
    passes = []
    fails  = []

    # p05 threshold checks
    for metric, cfg in thresholds.items():
        if metric == "p_loss":
            continue
        if metric not in summary:
            continue
        p05   = summary[metric]["p05"]
        limit = cfg["p05_min"]
        if p05 < limit:
            fails.append(
                f"p05 {metric} = {p05:.2%}  < {limit:.2%}  [{cfg['label']}]"
            )
        else:
            passes.append(
                f"p05 {metric} = {p05:.2%}  >= {limit:.2%}  [{cfg['label']}]"
            )

    # P(loss) check
    p_loss_limit = thresholds["p_loss"]["max"]
    if p_loss > p_loss_limit:
        fails.append(
            f"P(CAGR < 0) = {p_loss:.1%}  > {p_loss_limit:.0%}  "
            f"[{thresholds['p_loss']['label']}]"
        )
    else:
        passes.append(
            f"P(CAGR < 0) = {p_loss:.1%}  <= {p_loss_limit:.0%}  "
            f"[{thresholds['p_loss']['label']}]"
        )

    # Median CAGR check — lower bar than MC given structural bias
    median_cagr = summary["CAGR"]["median"]
    if median_cagr < 0.01:
        fails.append(
            f"Median CAGR = {median_cagr:.1%}  "
            f"[procedure fails to find positive strategy on typical synthetic history]"
        )
    else:
        passes.append(
            f"Median CAGR = {median_cagr:.1%}  [> 1% threshold]"
        )

    # Baseline vs median — is real history unusually good?
    base_cagr   = baseline_metrics.get("CAGR", float("nan"))
    
    p95_cagr    = summary["CAGR"]["p95"]
    if base_cagr > p95_cagr:
        fails.append(
            f"Baseline CAGR {base_cagr:.1%} exceeds p95 of bootstrap distribution "
            f"({p95_cagr:.1%}) — real history may be unusually favourable for this strategy"
        )
    else:
        passes.append(
            f"Baseline CAGR {base_cagr:.1%} within bootstrap distribution "
            f"(p95 = {p95_cagr:.1%}) — real history not an outlier"
        )

    verdict = "ROBUST" if not fails else "FRAGILE"

    logging.info("")
    logging.info("VERDICT: %s", verdict)
    if passes:
        for p in passes:
            logging.info("  PASS  %s", p)
    if fails:
        for f in fails:
            logging.info("  FAIL  %s", f)
    logging.info("=" * 80)

    summary["verdict"] = verdict
    return summary