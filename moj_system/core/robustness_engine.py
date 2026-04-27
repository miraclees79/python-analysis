"""
mc_robustness.py
================
Monte Carlo parameter robustness test for walk-forward optimised strategies.

ATR STOP MODE SUPPORT
---------------------
When use_atr_stop=True is active in the run configuration:

  - PERTURB_PARAMS replaces "X" with "N_atr" so the normalised ATR
    multiplier is perturbed rather than the fixed stop fraction.
  - MIN_VALUES adds a "N_atr" floor (default 0.02, same as X floor).
  - build_perturbation_grid handles either key name transparently.
  - extract_best_params_from_wf_results reads both X and N_atr columns.
  - run_universe passes use_atr_stop and atr_window through to
    run_strategy_with_trades.

All other logic (window sampling, equity stitching, carry state, parallel
execution, analysis/reporting) is unchanged.
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

from moj_system.core.strategy_engine import (
    run_strategy_with_trades,
    compute_metrics,
    walk_forward
)
# Hack for legacy code compatibility
def compute_fund_breadth_signal(*args, **kwargs):
    return None
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Parameters perturbed in fixed-stop mode
PERTURB_PARAMS_FIXED = ["X", "Y", "fast", "slow", "stop_loss"]

# Parameters perturbed in ATR-stop mode (N_atr replaces X)
PERTURB_PARAMS_ATR   = ["N_atr", "Y", "fast", "slow", "stop_loss"]

INTEGER_PARAMS  = {"fast", "slow"}

MIN_SLOW_FAST_GAP = 75

MIN_VALUES = {
    "X":         0.02,
    "N_atr":     0.02,   # minimum normalised ATR multiplier (same floor as X)
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

    Automatically selects PERTURB_PARAMS_ATR when base_params contains
    "N_atr" (i.e. use_atr_stop=True was active), otherwise uses
    PERTURB_PARAMS_FIXED. This means the caller does not need to pass a
    mode flag — the param dict self-describes which mode is active.

    Parameters
    ----------
    base_params : dict   — best params for one window
    pct         : float  — perturbation fraction (default 0.20 = ±20%)

    Returns
    -------
    list[dict]  — all valid perturbed parameter dicts for this window.
    """
    # Determine which stop parameter is active
    use_atr = "N_atr" in base_params and base_params.get("use_atr_stop", False)
    perturb_params = PERTURB_PARAMS_ATR if use_atr else PERTURB_PARAMS_FIXED

    grid = {}

    for name in perturb_params:
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
            if key not in perturb_params:
                p[key] = base_params[key]

        # Enforce MA separation constraint
        if p.get("filter_mode") == "ma":
            if p["slow"] - p["fast"] < MIN_SLOW_FAST_GAP:
                continue

        # In fixed mode: absolute stop must be less than trailing stop fraction
        if not use_atr and "X" in p and "stop_loss" in p:
            if p["stop_loss"] >= p["X"]:
                continue

        combos.append(p)

    logging.debug(
        "build_perturbation_grid: %d valid combinations (pct=%.0f%%, mode=%s)",
        len(combos), pct * 100, "ATR" if use_atr else "fixed"
    )
    return combos


# ---------------------------------------------------------------------------
# Step 2 — Build perturbation grids for all windows
# ---------------------------------------------------------------------------

def build_all_perturbation_grids(best_params, pct=0.20):
    return {
        w_id: build_perturbation_grid(params, pct=pct)
        for w_id, params in best_params.items()
    }


# ---------------------------------------------------------------------------
# Step 3 — Sample one universe
# ---------------------------------------------------------------------------

def sample_universe(window_variants, rng):
    return {
        w_id: rng.choice(variants)
        for w_id, variants in window_variants.items()
    }


# ---------------------------------------------------------------------------
# Step 4 — Run one universe
# ---------------------------------------------------------------------------

def run_universe(universe, windows, df, cash_df, vol_window,
                 selected_mode, funds_df=None, price_col="Zamkniecie"):
    """
    Stitch the full OOS equity curve using perturbed params — no retraining.

    ATR parameters (use_atr_stop, N_atr, atr_window) are read from each
    window's param dict and forwarded to run_strategy_with_trades, so
    ATR mode is handled transparently without additional arguments.
    """
    equity_parts = []
    all_trades   = []
    carry_state  = None
    prev_equity  = 1.0

    for w_id, w in enumerate(windows):
        params = universe[w_id]

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

        result = run_strategy_with_trades(
            oos_slice,
            X             = params.get("X", 0.10),
            Y             = params["Y"],
            fast          = int(params["fast"]),
            slow          = int(params["slow"]),
            stop_loss     = params["stop_loss"],
            target_vol    = params.get("target_vol"),
            position_mode = selected_mode,
            vol_window    = vol_window,
            filter_mode  = params.get("filter_mode"),
            cash_df       = cash_slice,
            initial_state = carry_state,
            warmup_df     = warmup_slice if len(warmup_slice) > 0 else None,
            fund_signal   = fund_signal,
            price_col     = price_col,
            # ATR stop parameters — forwarded from the perturbed param dict
            use_atr_stop  = params.get("use_atr_stop", False),
            N_atr         = params.get("N_atr", 0.10),
            atr_window    = params.get("atr_window", 20),
        )

        if result is None or result[0] is None:
            logging.debug("run_universe: window %d returned None, skipping", w_id)
            continue

        result_df, _, trades_df, end_state = result

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
# Step 5 — Single sample runner
# ---------------------------------------------------------------------------

def _run_single_sample(seed, window_variants, windows, df, cash_df,
                        vol_window, selected_mode, funds_df, price_col):
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

    ATR stop mode is detected automatically from best_params — no
    additional arguments are required from the caller.
    """
    t_start = time.time()

    logging.info("=" * 80)
    logging.info("MONTE CARLO ROBUSTNESS TEST")
    logging.info("=" * 80)

    # Detect stop mode from first window's params for logging
    first_params = next(iter(best_params.values()))
    stop_mode = "ATR" if first_params.get("use_atr_stop", False) else "fixed %"
    logging.info(
        "Config: n_samples=%d | perturb_pct=%.0f%% | n_jobs=%d | seed=%d | stop_mode=%s",
        n_samples, perturb_pct * 100, n_jobs, seed, stop_mode
    )

    window_variants = build_all_perturbation_grids(best_params, pct=perturb_pct)

    for w_id, variants in window_variants.items():
        logging.info(
            "Window %d (%s filter, %s stop): %d perturbation variants",
            w_id,
            best_params[w_id].get("filter_mode", "?"),
            stop_mode,
            len(variants)
        )

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
        price_col=price_col
    )
    single_ms = (time.time() - t0) * 1000
    est_total_s = 2 * single_ms * n_samples / max(n_jobs, 1) / 1000
    logging.info(
        "Single sample: %.0fms -> estimated total (x2): %.0fs (~%.1f min) on %d jobs",
        single_ms, est_total_s, est_total_s / 60, n_jobs
    )

    try:
        raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
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

    p_loss = (results_df["CAGR"] < 0).mean()
    summary["p_loss"] = p_loss
    loss_label = f"P(CAGR < 0) = {p_loss:.1%}"
    if p_loss < 0.10:
        passes.append(loss_label)
    else:
        fails.append(f"P(CAGR < 0) < 10%  [actual {p_loss:.1%}]")

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

    median_cagr = summary["CAGR"]["median"]
    if median_cagr < 0.02:
        fails.append(
            f"Median CAGR = {median_cagr:.1%}  [typical universe performs poorly, threshold 2%]"
        )
    else:
        passes.append(f"Median CAGR = {median_cagr:.1%}  [> 2% threshold]")

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
# Utility — extract window definitions from wf_results DataFrame
# ---------------------------------------------------------------------------

def extract_windows_from_wf_results(wf_results, train_years=8):
    windows = []
    for i, row in wf_results.iterrows():
        test_start   = pd.Timestamp(row["TestStart"])
        test_end     = pd.Timestamp(row["TestEnd"])
        train_start  = pd.Timestamp(row["TrainStart"])
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

    Reads both X and N_atr columns (whichever are present) and includes
    use_atr_stop and atr_window so run_universe can forward them correctly
    to run_strategy_with_trades.
    """
    best_params = {}
    for i, row in wf_results.iterrows():
        use_atr = bool(row.get("use_atr_stop", False))
        p = {
            "filter_mode":  row["filter_mode"],
            "fund_idx":     row.get("fund_idx"),
            "fund_params":  row.get("fund_params"),
            "X":            float(row["X"]) if "X" in row and not use_atr else 0.10,
            "Y":            float(row["Y"]),
            "fast":         int(row["fast"]),
            "slow":         int(row["slow"]),
            "stop_loss":    float(row["stop_loss"]),
            "target_vol":   row.get("target_vol"),
            "use_atr_stop": use_atr,
            "N_atr":        float(row["N_atr"]) if "N_atr" in row else 0.10,
            "atr_window":   int(row["atr_window"]) if "atr_window" in row else 20,
        }
        best_params[i] = p
    return best_params


#=====================================================================
# DATA HISTORY BOOTSTRAP
#=====================================================================


def block_bootstrap_history(df, price_col, cash_col, block_size=250, seed=None):
    """
    Reshuffle (index_return, cash_return) pairs in blocks.
    """
    rng = np.random.default_rng(seed)
    
    idx_ret  = df[price_col].pct_change().dropna()
    cash_ret = df[cash_col].pct_change().dropna()

    aligned = pd.concat([idx_ret, cash_ret], axis=1).dropna()
    aligned.columns = ["idx_ret", "cash_ret"]

    n = len(aligned)
    block_starts = list(range(0, n - block_size + 1, block_size))
    n_blocks_needed = int(np.ceil(n / block_size))
    sampled_idx  = rng.choice(len(block_starts),
                           size=n_blocks_needed,
                           replace=True)

    reshuffled = pd.concat(
        [aligned.iloc[block_starts[i] : block_starts[i] + block_size]
         for i in sampled_idx]
        )

    reshuffled = reshuffled.iloc[:n].reset_index(drop=True)

    synthetic_price = (1 + reshuffled["idx_ret"]).cumprod()
    synthetic_cash  = (1 + reshuffled["cash_ret"]).cumprod()

    out = df.loc[aligned.index].copy().reset_index(drop=True)
    out[price_col] = synthetic_price.values
    out[cash_col]  = synthetic_cash.values
    out.index = aligned.index
    return out

def _bootstrap_single_sample(
    i, combined, df, cash_df, price_col, cash_price_col, block_size, wf_kwargs
):
    """
    Single bootstrap sample — designed for joblib.Parallel dispatch.
    ATR parameters are forwarded via wf_kwargs transparently.
    """
    wf_kwargs_inner = {**wf_kwargs, "n_jobs": 1}
    
    try:
        synthetic = block_bootstrap_history(
            combined, price_col, "cash_price",
            block_size=block_size, seed=i
        )

        n_synth = len(synthetic)
        synthetic_df = combined.iloc[:n_synth, [combined.columns.get_loc(price_col)]].copy()
        synthetic_df[price_col] = synthetic[price_col].values
        synthetic_cash = combined.iloc[:n_synth][["cash_price"]].copy()
        synthetic_cash = synthetic_cash.rename(columns={"cash_price": cash_price_col})
        synthetic_cash[cash_price_col] = synthetic["cash_price"].values
        
        logging.debug(
            "Bootstrap %d: synthetic_df=%d, synthetic_cash=%d, "
            "synthetic_df.index[0]=%s, synthetic_df.index[-1]=%s",
            i, len(synthetic_df), len(synthetic_cash),
            synthetic_df.index[0], synthetic_df.index[-1]
            )   
        
        equity, wf_res, _ = walk_forward(
            synthetic_df, cash_df=synthetic_cash, **wf_kwargs_inner
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
    synthetic histories. ATR parameters are forwarded via wf_kwargs.
    """
    _cpu_count = os.cpu_count() or 1
    n_jobs= max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count
    
    logging.info("=" * 80)
    logging.info("BLOCK BOOTSTRAP ROBUSTNESS TEST")
    logging.info("=" * 80)
    logging.info(
        "Config: n_samples=%d | block_size=%d days | n_jobs=%d",
        n_samples, block_size, n_jobs
    )

    common_idx = df.index.intersection(cash_df.index)
    combined = df.loc[common_idx, [price_col]].copy()
    combined["cash_price"] = cash_df.loc[common_idx, cash_price_col]
    
    logging.info("Timing single sample...")
    t0 = time.time()
    _bootstrap_single_sample(
        0, combined, df, cash_df, price_col, cash_price_col,
        block_size, wf_kwargs
    )
    single_time = time.time() - t0
    estimated  = single_time * n_samples / n_jobs
    logging.info(
        "Single sample: %.0fms -> estimated total: %.0fs (~%.1f min) on %d jobs",
        single_time * 1000, estimated, estimated / 60, n_jobs
    )
    logging.info(
        "Single sample: %.0fms -> realistic estimation (x2): %.0fs (~%.1f min) on %d jobs",
        single_time * 1000, 2*estimated, 2*estimated / 60, n_jobs
    )

    valid     = []
    failed    = 0
    log_every = max(1, n_samples // 20)
    success   = False

    N_OUTER_JOBS = n_jobs
    for backend, n_jobs, label in [
            ("loky",      N_OUTER_JOBS, "multiprocessing"),
            ("threading", N_OUTER_JOBS, "threading"),
            (None,        1,            "sequential"),
            ]:
        try:
            if backend is None:
                source = (
                    _bootstrap_single_sample(
                        i, combined, df, cash_df,
                        price_col, cash_price_col,
                        block_size, wf_kwargs
                    )
                    for i in range(n_samples)
                )
            else:
                source = Parallel(n_jobs=n_jobs, backend=backend,
                                  return_as="generator")(
                    delayed(_bootstrap_single_sample)(
                        i, combined, df, cash_df,
                        price_col, cash_price_col,
                        block_size, wf_kwargs
                    )
                    for i in range(n_samples)
                )

            for idx, r in enumerate(source):
                if r is not None:
                    valid.append(r)
                else:
                    failed += 1
                completed = idx + 1
                if completed % log_every == 0 or completed == n_samples:
                    pct = completed / n_samples * 100
                    bar = ("=" * (completed * 20 // n_samples)).ljust(20, "-")
                    logging.info(
                        "Bootstrap progress: [%s] %d/%d (%.0f%%) - %d valid, %d failed",
                        bar, completed, n_samples, pct, len(valid), failed
                    )

            logging.info(
                "Block bootstrap completed using %s backend (%d jobs).",
                label, n_jobs
            )
            success = True
            break

        except Exception as e:
            logging.warning(
                "Bootstrap backend '%s' failed: %s — trying next option.",
                label, e
            )
            valid   = []
            failed  = 0

    if not success:
        logging.error("All bootstrap backends failed.")
        return pd.DataFrame()

    results_df = pd.DataFrame(valid)
    n_valid   = len(results_df)
    n_failed = failed
    
    if n_failed > 0:
        logging.warning(
            "%d/%d bootstrap samples failed and were excluded.", 
            n_failed, n_samples
        )

    if results_df.empty:
        logging.error("No valid bootstrap samples — cannot report.")
        return results_df

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

    return results_df


# ---------------------------------------------------------------------------
# Bootstrap analysis and reporting
# ---------------------------------------------------------------------------

def analyze_bootstrap(results_df, baseline_metrics, thresholds=None):
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

    passes = []
    fails  = []

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