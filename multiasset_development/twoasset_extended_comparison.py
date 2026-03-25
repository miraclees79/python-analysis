"""
twoasset_extended_comparison.py
================================
Apples-to-apples comparison of the 2-asset (WIG+TBSP+MMF) signal strategy
against the global equity sweep results, using the extended TBSP backcast
and extended MMF (WIBOR1M chain-link) to achieve OOS start 2011-01-04.

PURPOSE
-------
The sweep evaluated Mode A and Mode B on OOS 2011-01-04 to 2026-03-20.
The production 2-asset strategy starts OOS ~2016 using standard TBSP data
(available from 2006). This script re-runs the 2-asset strategy with:
  - Extended TBSP (tbsp_extended_combined.csv from Drive, starting ~1999)
  - Extended MMF (WIBOR1M chain-link back to 1994)
  - Window lengths: 6+1, 7+1, 7+2, 8+1, 8+2, 9+1, 9+2
  - OOS trimmed to 2011-01-04 to match sweep period

MC/BOOTSTRAP ROBUSTNESS (optional)
-----------------------------------
When RUN_MC=True, each window configuration runs:
  - MC parameter perturbation (n=N_MC samples) for WIG and TBSP separately
  - Verdict (ROBUST / FRAGILE) and all failure reasons logged per config
When RUN_BOOTSTRAP=True, each window configuration additionally runs:
  - Block bootstrap (n=N_BOOTSTRAP samples, block=250 days) for WIG and TBSP
  - Verdict and failure reasons logged per config
MC and bootstrap can be enabled independently.

OUTPUT
------
Prints a comparison table of 2-asset strategy metrics vs sweep Mode B results.
If MC/bootstrap is enabled, appends a robustness section to the table with
per-config verdicts and failure reason detail.

USAGE
-----
  python twoasset_extended_comparison.py

Requires:
  - strategy_test_library.py and multiasset_library.py on PYTHONPATH
    (i.e. run from multiasset_freeze_0_4/ or equivalent)
  - Extended TBSP CSV: pass path via TBSP_EXTENDED_PATH env var, or
    set TBSP_EXTENDED_PATH below. If not set, falls back to standard ^tbsp.
  - Standard stooq downloads for WIG, MMF, PL10Y, DE10Y.
"""

import os
import sys
import logging
import tempfile
import datetime as dt

import numpy as np
import pandas as pd

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Imports from freeze ──────────────────────────────────────────────────────

from strategy_test_library import (
    download_csv, load_csv,
    walk_forward, compute_metrics, compute_buy_and_hold,
)

from global_equity_library import build_mmf_extended
from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    build_spread_prefilter,
    build_yield_momentum_prefilter,
)
from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    run_block_bootstrap_robustness,
    analyze_bootstrap,
    EQUITY_THRESHOLDS_MC,
    BOND_THRESHOLDS_MC,
    EQUITY_THRESHOLDS_BOOTSTRAP,
    BOND_THRESHOLDS_BOOTSTRAP,
)

from price_series_builder import build_and_upload

# ============================================================
# USER SETTINGS
# ============================================================

# Path to tbsp_extended_combined.csv (downloaded from Drive).
# Set to None to fall back to standard ^tbsp from stooq.
TBSP_EXTENDED_PATH = os.getenv(
    "TBSP_EXTENDED_PATH",
    None   # <- paste local path here if not using env var
)

# OOS period to match the sweep
OOS_START = "2011-01-04"
OOS_END   = "2026-03-23"   # update to today's date when running

# Window configurations to evaluate
WINDOW_CONFIGS = [    (6, 1), (7, 1), (7, 2), (8, 1), (8, 2), (9, 1), (9, 2),]


#WINDOW_CONFIGS = [  (8, 2), (9, 1), (9, 2),]


# Strategy settings — must match production
POSITION_MODE        = "full"
VOL_WINDOW           = 20
FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]
COOLDOWN_DAYS        = 10
ANNUAL_CAP           = 12
ALLOC_OBJECTIVE      = "calmar"
YIELD_PREFILTER_BP   = 50.0

# Equity grids
X_GRID_EQ = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ   = [50, 75, 100]
SLOW_EQ   = [150, 200, 250]
TV_EQ     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ = [126, 252]

# --- Trailing stop mode ---
# USE_ATR_STOP = False : fixed percentage trailing stop (current default)
#     X_GRID_EQ is used; stop fires when price < (1 - X) * peak
# USE_ATR_STOP = True  : ATR-scaled Chandelier exit
#     N_ATR_GRID is used; stop fires when price < peak - N * ATR
#     ATR = rolling mean of |daily price change| over ATR_WINDOW bars
#     Recommended N_ATR range for WIG: 3–6 (wider = more room to breathe)
#     ATR_WINDOW: 20 matches VOL_WINDOW; increase to 40–60 for a slower ATR
#
# The bond walk-forward (Phase 3) uses a separate flag USE_ATR_STOP_BD
# so equity and bond stops can be tuned independently.
# Set USE_ATR_STOP_BD = USE_ATR_STOP to keep them in sync.
#
USE_ATR_STOP    = True          # Equity trailing stop mode
ATR_WINDOW      = 20             # Rolling window for ATR estimate (days)
N_ATR_GRID      = [3.0, 4.0, 5.0, 6.0, 7.0]   # Multiplier grid for IS search

USE_ATR_STOP_BD = False          # Bond trailing stop mode (can differ from equity)
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [2.0, 3.0, 4.0, 5.0, 6.0]

# Bond grids
X_GRID_BD = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD = [0.001]
FAST_BD   = [50, 100, 150]
SLOW_BD   = [200, 300, 400, 500]
TV_BD     = [0.08, 0.10, 0.12]
SL_BD     = [0.01, 0.02, 0.03]

# Parallelism
_cpu = os.cpu_count() or 1
N_JOBS = max(1, _cpu - 1) if _cpu > 3 and sys.platform == "win32" else _cpu

FAST_MODE = True

# ---------------------------------------------------------------------------
# DATA FLOOR DATES
# ---------------------------------------------------------------------------
# WIG (Mode A only): daily continuous trading started 1994-10-03.
#   Earlier data has multi-day gaps that distort the breakout trough
#   calculation and MA windows.  Clipped after download in Phase 2.
WIG_DATA_FLOOR = "1995-01-02"
 
# MMF extension: chain-link WIBOR 1M backwards from first MMF NAV to this
#   date. Ensures IS windows starting before 1999 have realistic cash returns
#   rather than ret_mmf=0 on signal-off days. Applied in Phase 2.
MMF_FLOOR = "1995-01-02"
DATA_START = "1990-01-01"  # hard floor for all series



# ── MC / Bootstrap robustness settings ──────────────────────────────────────
#
# RUN_MC=True  : run MC parameter perturbation for each window config.
#                Adds verdict + failure reasons to comparison table.
#                Runtime: ~14 min per config on 12-core machine at N_MC=1000.
#                Smoke test: set N_MC=10.
#
# RUN_BOOTSTRAP=True : run block bootstrap for each window config.
#                Runtime: ~3h per config on 12-core machine at N_BOOTSTRAP=500.
#                Requires RUN_MC=True (bootstrap reuses the WF results).
#
# Both can be disabled independently. When both are False the script
# produces only OOS portfolio metrics (original behaviour).
#
RUN_MC         = True   # MC parameter perturbation per config
N_MC           = 500    # MC samples (set to 10 for smoke test) 1000 base
RUN_BOOTSTRAP  = False   # Block bootstrap per config (requires RUN_MC=True)
N_BOOTSTRAP    = 100     # Bootstrap samples (set to 10 for smoke test) 500 base


# Sweep Mode B results for comparison (from global_equity_sweep_results.csv)
# Both-ROBUST configs only, sorted by CalMAR
SWEEP_MODEB = [
    {"config": "B 6+1", "cagr": 0.0539, "sharpe": 0.723, "maxdd": -0.130, "calmar": 0.415, "wig_mc": "ROBUST", "tbsp_mc": "ROBUST"},
    {"config": "B 7+2", "cagr": 0.0613, "sharpe": 0.663, "maxdd": -0.159, "calmar": 0.385, "wig_mc": "ROBUST", "tbsp_mc": "ROBUST"},
    {"config": "B 6+2", "cagr": 0.0616, "sharpe": 0.657, "maxdd": -0.177, "calmar": 0.348, "wig_mc": "ROBUST", "tbsp_mc": "ROBUST"},
    {"config": "B 8+2", "cagr": 0.0488, "sharpe": 0.656, "maxdd": -0.153, "calmar": 0.318, "wig_mc": "ROBUST", "tbsp_mc": "ROBUST"},
    {"config": "B 9+2", "cagr": 0.0632, "sharpe": 0.579, "maxdd": -0.225, "calmar": 0.280, "wig_mc": "ROBUST", "tbsp_mc": "ROBUST"},
    {"config": "B 9+1", "cagr": 0.0598, "sharpe": 0.616, "maxdd": -0.213, "calmar": 0.280, "wig_mc": "FRAGILE","tbsp_mc": "ROBUST"},
]


# ============================================================
# PHASE 1 — DOWNLOAD DATA
# ============================================================

def download_all(tmp_dir):
    log.info("=" * 70)
    log.info("DOWNLOADING DATA")
    log.info("=" * 70)

    def _get(ticker, label, mandatory=True):
        url  = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        path = os.path.join(tmp_dir, f"{label}.csv")
        ok   = download_csv(url, path)
        if not ok:
            if mandatory:
                log.error("FAIL: %s", label); sys.exit(1)
            return None
        df = load_csv(path)
        if df is None and mandatory:
            log.error("FAIL load_csv: %s", label); sys.exit(1)
        if df is not None:
            log.info("OK  %-10s  %5d rows  %s to %s",
                     label, len(df),
                     df.index.min().date(), df.index.max().date())
        return df

    WIG   = _get("wig",       "WIG")
    WIG = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    MMF   = _get("2720.n",    "MMF")
    W1M   = _get("plopln1m",  "WIBOR1M", mandatory=False)
    PL10Y = _get("10yply.b",  "PL10Y")
    DE10Y = _get("10ydey.b",  "DE10Y")

    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()

    TBSP = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "tbsp_extended_full.csv",
        combined_filename = "tbsp_extended_combined.csv",
        extension_ticker  = "^tbsp",
        extension_source  = "stooq",
    )

    if TBSP is None:
        logging.error("FAIL: TBSP build/update failed.")
        sys.exit(1)
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "TBSP", len(TBSP),
                 TBSP.index.min().date(), TBSP.index.max().date())
    for col in ["Najwyzszy", "Najnizszy"]:
        if col not in TBSP.columns:
            TBSP[col] = TBSP["Zamkniecie"]

    return WIG, TBSP, MMF, W1M, PL10Y, DE10Y


# ============================================================
# PHASE 2 — DERIVED SERIES
# ============================================================

def build_derived(WIG, TBSP, MMF, W1M, PL10Y, DE10Y):
    log.info("=" * 70)
    log.info("BUILDING DERIVED SERIES")
    log.info("=" * 70)

    # Extended MMF
    if W1M is not None:
        MMF_EXT = build_mmf_extended(MMF, W1M, floor_date="MMF_FLOOR")
        log.info("MMF extended back to %s", MMF_EXT.index.min().date())
    else:
        MMF_EXT = MMF
        log.warning("WIBOR1M unavailable — using standard MMF (starts ~1999)")

    # Bond entry gate (spread + yield pre-filter)
    spread_pf = build_spread_prefilter(PL10Y, DE10Y)
    yield_pf  = build_yield_momentum_prefilter(
        PL10Y, rise_threshold_bp=YIELD_PREFILTER_BP, lookback_days=63
    )
    tbsp_idx = TBSP.index
    spread_gate = spread_pf.reindex(tbsp_idx, method="ffill").fillna(1).astype(int)
    yield_gate  = yield_pf.reindex(tbsp_idx,  method="ffill").fillna(1).astype(int)
    bond_gate   = (spread_gate & yield_gate).astype(int)
    bond_gate.name = "bond_entry_gate"

    log.info(
        "Bond entry gate: %.1f%% days pass  (spread %.1f%% | yield %.1f%%)",
        bond_gate.mean()*100, spread_gate.mean()*100, yield_gate.mean()*100,
    )

    # Return series
    ret_eq  = WIG["Zamkniecie"].pct_change().dropna()
    ret_bd  = TBSP["Zamkniecie"].pct_change().dropna()
    ret_mmf = MMF_EXT["Zamkniecie"].pct_change().dropna()

    return MMF_EXT, bond_gate, ret_eq, ret_bd, ret_mmf


# ============================================================
# ROBUSTNESS HELPERS
# ============================================================

def _derive_mc_failures(summary, thresholds, baseline_metrics):
    """
    Re-derive failure reasons from an analyze_robustness summary dict.

    The passes/fails lists are local to analyze_robustness and not returned.
    This function reconstructs them from the returned summary dict so callers
    can log and display failure detail without modifying analyze_robustness.

    Parameters
    ----------
    summary          : dict  — returned by analyze_robustness
    thresholds       : dict  — same thresholds dict passed to analyze_robustness
    baseline_metrics : dict  — compute_metrics output for the baseline equity curve

    Returns
    -------
    list[str]  — human-readable failure reason strings (empty if ROBUST)
    """
    fails = []

    # F1: p05 threshold checks (CAGR, Sharpe, MaxDD)
    for metric, cfg in thresholds.items():
        sub = summary.get(metric)
        if sub is None:
            continue
        p05_val = sub["p05"]
        if p05_val < cfg["p05_min"]:
            fails.append(
                f"{cfg['label']}  [actual p05={p05_val:.3f}]"
            )

    # F2: P(CAGR < 0) >= 10%
    p_loss = summary.get("p_loss", 0.0)
    if p_loss >= 0.10:
        fails.append(f"P(CAGR < 0) < 10%  [actual {p_loss:.1%}]")

    # F3: baseline is top-decile outlier (overfitting flag)
    for metric in ["CAGR", "Sharpe"]:
        sub = summary.get(metric)
        if sub is None:
            continue
        p_worse = sub.get("p_worse_than_baseline", 0.0)
        if p_worse > 0.85:
            fails.append(
                f"P({metric} worse than baseline) = {p_worse:.1%}  "
                f"[baseline top-decile outlier — overfitting]"
            )

    # F4: median CAGR < 2%
    cagr_sub = summary.get("CAGR", {})
    median_cagr = cagr_sub.get("median", 0.0)
    if median_cagr < 0.02:
        fails.append(
            f"Median CAGR = {median_cagr:.1%}  [< 2% threshold]"
        )

    # F5: CAGR p95-p05 range / median > 3.0
    p95 = cagr_sub.get("p95", 0.0)
    p05 = cagr_sub.get("p05", 0.0)
    cagr_range = p95 - p05
    if median_cagr > 0 and cagr_range / median_cagr > 3.0:
        fails.append(
            f"CAGR p95-p05 range/median = {cagr_range/median_cagr:.1f}x  "
            f"[> 3.0 — highly parameter-sensitive]"
        )

    return fails


def _derive_bootstrap_failures(summary, thresholds, baseline_metrics):
    """
    Re-derive failure reasons from an analyze_bootstrap summary dict.

    Parameters
    ----------
    summary          : dict  — returned by analyze_bootstrap
    thresholds       : dict  — same thresholds dict passed to analyze_bootstrap
    baseline_metrics : dict  — compute_metrics output for the baseline equity curve

    Returns
    -------
    list[str]  — human-readable failure reason strings (empty if ROBUST)
    """
    fails = []

    # B1: p05 threshold checks (CAGR, Sharpe, MaxDD)
    for metric, cfg in thresholds.items():
        if metric == "p_loss":
            continue
        sub = summary.get(metric)
        if sub is None:
            continue
        p05_val = sub["p05"]
        limit   = cfg["p05_min"]
        if p05_val < limit:
            fails.append(
                f"p05 {metric} = {p05_val:.2%}  < {limit:.2%}  [{cfg['label']}]"
            )

    # B2: P(CAGR < 0) >= 20%
    p_loss       = summary.get("p_loss", 0.0)
    p_loss_limit = thresholds.get("p_loss", {}).get("max", 0.20)
    if p_loss > p_loss_limit:
        fails.append(
            f"P(CAGR < 0) = {p_loss:.1%}  > {p_loss_limit:.0%}  "
            f"[{thresholds.get('p_loss', {}).get('label', 'P(CAGR < 0) threshold')}]"
        )

    # B3: median CAGR < 1%
    cagr_sub    = summary.get("CAGR", {})
    median_cagr = cagr_sub.get("median", 0.0)
    if median_cagr < 0.01:
        fails.append(
            f"Median CAGR = {median_cagr:.1%}  "
            f"[procedure fails to find positive strategy on typical synthetic history]"
        )

    # B4: baseline CAGR > p95 of bootstrap distribution
    base_cagr = baseline_metrics.get("CAGR", float("nan"))
    p95_cagr  = cagr_sub.get("p95", float("nan"))
    if not np.isnan(base_cagr) and not np.isnan(p95_cagr) and base_cagr > p95_cagr:
        fails.append(
            f"Baseline CAGR {base_cagr:.1%} > p95 of bootstrap dist "
            f"({p95_cagr:.1%}) — real history unusually favourable"
        )

    return fails


def run_robustness_for_config(
    label,
    wf_equity_eq, wf_results_eq, wf_equity_bd, wf_results_bd,
    WIG, TBSP, MMF_EXT, bond_gate,
    train_years, test_years,
):
    """
    Run MC (and optionally bootstrap) robustness for one window config.

    Called once per (train_years, test_years) combination after the
    walk-forward results are available from run_config().

    Returns a dict with keys:
        wig_mc_verdict, wig_mc_failures   : str, list[str]
        tbsp_mc_verdict, tbsp_mc_failures : str, list[str]
        wig_bb_verdict, wig_bb_failures   : str | None, list[str] | None
        tbsp_bb_verdict, tbsp_bb_failures : str | None, list[str] | None
    """
    result = {
        "wig_mc_verdict":   None,
        "wig_mc_failures":  [],
        "tbsp_mc_verdict":  None,
        "tbsp_mc_failures": [],
        "wig_bb_verdict":   None,
        "wig_bb_failures":  None,
        "tbsp_bb_verdict":  None,
        "tbsp_bb_failures": None,
    }

    if not RUN_MC:
        return result

    # ── MC: WIG ─────────────────────────────────────────────────────────────
    log.info("  [MC WIG %s] n=%d", label, N_MC)
    windows_eq     = extract_windows_from_wf_results(wf_results_eq, train_years=train_years)
    best_params_eq = extract_best_params_from_wf_results(wf_results_eq)

    mc_eq = run_monte_carlo_robustness(
        best_params   = best_params_eq,
        windows       = windows_eq,
        df            = WIG,
        cash_df       = MMF_EXT,
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,
        n_samples     = N_MC,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie",
    )
    baseline_eq      = compute_metrics(wf_equity_eq)
    summary_mc_eq    = analyze_robustness(mc_eq, baseline_eq, thresholds=EQUITY_THRESHOLDS_MC)
    result["wig_mc_verdict"]  = summary_mc_eq["verdict"]
    result["wig_mc_failures"] = _derive_mc_failures(
        summary_mc_eq, EQUITY_THRESHOLDS_MC, baseline_eq
    )

    # ── MC: TBSP ────────────────────────────────────────────────────────────
    log.info("  [MC TBSP %s] n=%d", label, N_MC)
    windows_bd     = extract_windows_from_wf_results(wf_results_bd, train_years=train_years)
    best_params_bd = extract_best_params_from_wf_results(wf_results_bd)

    mc_bd = run_monte_carlo_robustness(
        best_params   = best_params_bd,
        windows       = windows_bd,
        df            = TBSP,
        cash_df       = MMF_EXT,
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,
        n_samples     = N_MC,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie",
    )
    baseline_bd      = compute_metrics(wf_equity_bd)
    summary_mc_bd    = analyze_robustness(mc_bd, baseline_bd, thresholds=BOND_THRESHOLDS_MC)
    result["tbsp_mc_verdict"]  = summary_mc_bd["verdict"]
    result["tbsp_mc_failures"] = _derive_mc_failures(
        summary_mc_bd, BOND_THRESHOLDS_MC, baseline_bd
    )

    if not RUN_BOOTSTRAP:
        return result

    # ── Bootstrap: WIG ──────────────────────────────────────────────────────
    log.info("  [Bootstrap WIG %s] n=%d", label, N_BOOTSTRAP)
    bb_eq = run_block_bootstrap_robustness(
        df                    = WIG,
        cash_df               = MMF_EXT,
        price_col             = "Zamkniecie",
        cash_price_col        = "Zamkniecie",
        n_samples             = N_BOOTSTRAP,
        block_size            = 250,
        train_years           = train_years,
        test_years            = test_years,
        vol_window            = VOL_WINDOW,
        funds_df              = None,
        fund_params_grid      = None,
        selected_mode         = POSITION_MODE,
        filter_modes_override = FORCE_FILTER_MODE_EQ,
        X_grid                = X_GRID_EQ,
        Y_grid                = Y_GRID_EQ,
        fast_grid             = FAST_EQ,
        slow_grid             = SLOW_EQ,
        tv_grid               = TV_EQ,
        sl_grid               = SL_EQ,
        mom_lookback_grid     = MOM_LB_EQ,
        use_atr_stop          = USE_ATR_STOP,
        N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
        atr_window            = ATR_WINDOW,
    )
    summary_bb_eq    = analyze_bootstrap(bb_eq, baseline_eq, thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)
    result["wig_bb_verdict"]  = summary_bb_eq["verdict"]
    result["wig_bb_failures"] = _derive_bootstrap_failures(
        summary_bb_eq, EQUITY_THRESHOLDS_BOOTSTRAP, baseline_eq
    )

    # ── Bootstrap: TBSP ─────────────────────────────────────────────────────
    log.info("  [Bootstrap TBSP %s] n=%d", label, N_BOOTSTRAP)
    bb_bd = run_block_bootstrap_robustness(
        df                    = TBSP,
        cash_df               = MMF_EXT,
        price_col             = "Zamkniecie",
        cash_price_col        = "Zamkniecie",
        n_samples             = N_BOOTSTRAP,
        block_size            = 250,
        train_years           = train_years,
        test_years            = test_years,
        vol_window            = VOL_WINDOW,
        funds_df              = None,
        fund_params_grid      = None,
        selected_mode         = POSITION_MODE,
        filter_modes_override = FORCE_FILTER_MODE_BD,
        X_grid                = X_GRID_BD,
        Y_grid                = Y_GRID_BD,
        fast_grid             = FAST_BD,
        slow_grid             = SLOW_BD,
        tv_grid               = TV_BD,
        sl_grid               = SL_BD,
        mom_lookback_grid     = [252],
        use_atr_stop          = USE_ATR_STOP_BD,
        N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window            = ATR_WINDOW_BD,
        entry_gate_series     = bond_gate,
    )
    summary_bb_bd    = analyze_bootstrap(bb_bd, baseline_bd, thresholds=BOND_THRESHOLDS_BOOTSTRAP)
    result["tbsp_bb_verdict"]  = summary_bb_bd["verdict"]
    result["tbsp_bb_failures"] = _derive_bootstrap_failures(
        summary_bb_bd, BOND_THRESHOLDS_BOOTSTRAP, baseline_bd
    )

    return result


# ============================================================
# PHASE 3 — RUN ONE WINDOW CONFIGURATION
# ============================================================

def run_config(train_years, test_years, WIG, TBSP, MMF_EXT,
               bond_gate, ret_eq, ret_bd, ret_mmf):

    label = f"{train_years}+{test_years}"
    log.info("")
    log.info("--- Config %s ---", label)

    # Equity walk-forward
    wf_eq_eq, wf_res_eq, wf_tr_eq = walk_forward(
        df                    = WIG,
        cash_df               = MMF_EXT,
        train_years           = train_years,
        test_years            = test_years,
        vol_window            = VOL_WINDOW,
        selected_mode         = POSITION_MODE,
        filter_modes_override = FORCE_FILTER_MODE_EQ,
        X_grid=X_GRID_EQ, Y_grid=Y_GRID_EQ,
        fast_grid=FAST_EQ, slow_grid=SLOW_EQ,
        tv_grid=TV_EQ, sl_grid=SL_EQ,
        mom_lookback_grid=MOM_LB_EQ,
        objective="calmar",
        n_jobs                = N_JOBS,
        fast_mode             = FAST_MODE,
        use_atr_stop          = USE_ATR_STOP,
        N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
        atr_window            = ATR_WINDOW,
    )
    if wf_eq_eq.empty:
        log.warning("Empty equity WF for %s", label)
        return None

    # Bond walk-forward
    wf_eq_bd, wf_res_bd, wf_tr_bd = walk_forward(
        df                    = TBSP,
        cash_df               = MMF_EXT,
        train_years           = train_years,
        test_years            = test_years,
        vol_window            = VOL_WINDOW,
        selected_mode         = POSITION_MODE,
        filter_modes_override = FORCE_FILTER_MODE_BD,
        X_grid=X_GRID_BD, Y_grid=Y_GRID_BD,
        fast_grid=FAST_BD, slow_grid=SLOW_BD,
        tv_grid=TV_BD, sl_grid=SL_BD,
        mom_lookback_grid=[252],
        objective="calmar",
        n_jobs=N_JOBS,
        entry_gate_series=bond_gate,
        fast_mode             = FAST_MODE,
        use_atr_stop          = USE_ATR_STOP_BD,
        N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window            = ATR_WINDOW_BD,
    )
    if wf_eq_bd.empty:
        log.warning("Empty bond WF for %s", label)
        return None

    m_eq = compute_metrics(wf_eq_eq)
    m_bd = compute_metrics(wf_eq_bd)
    log.info("  WIG  OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
             m_eq["CAGR"]*100, m_eq["Sharpe"], m_eq["MaxDD"]*100, m_eq["CalMAR"])
    log.info("  TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
             m_bd["CAGR"]*100, m_bd["Sharpe"], m_bd["MaxDD"]*100, m_bd["CalMAR"])

    # Build signals
    sig_eq = build_signal_series(wf_eq_eq, wf_tr_eq)
    sig_bd = build_signal_series(wf_eq_bd, wf_tr_bd)

    # Trim to common OOS period
    oos_s = max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min())
    oos_e = min(wf_res_eq["TestEnd"].max(),   wf_res_bd["TestEnd"].max())

    def _trim(s, a, b):
        return s.loc[(s.index >= a) & (s.index <= b)]

    sig_eq_oos = _trim(sig_eq, oos_s, oos_e)
    sig_bd_oos = _trim(sig_bd, oos_s, oos_e)

    # Allocation walk-forward
    port_eq, weights, realloc_log, alloc_df = allocation_walk_forward(
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_full  = sig_eq,
        sig_bond_full    = sig_bd,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        wf_results_eq    = wf_res_eq,
        wf_results_bd    = wf_res_bd,
        step             = 0.10,
        objective        = ALLOC_OBJECTIVE,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )

    if port_eq.empty:
        log.warning("Empty portfolio for %s", label)
        return None

    # Trim portfolio to sweep OOS period for fair comparison
    oos_target_s = pd.Timestamp(OOS_START)
    oos_target_e = pd.Timestamp(OOS_END)
    port_trimmed = port_eq.loc[
        (port_eq.index >= oos_target_s) &
        (port_eq.index <= oos_target_e)
    ]

    if port_trimmed.empty:
        log.warning(
            "Config %s: portfolio has no data in target OOS %s–%s. "
            "Actual OOS: %s–%s. Extended TBSP needed for this window.",
            label, OOS_START, OOS_END,
            port_eq.index.min().date(), port_eq.index.max().date(),
        )
        return None

    port_trimmed = port_trimmed / port_trimmed.iloc[0]
    m_port = {k: float(v) for k, v in compute_metrics(port_trimmed).items()}

    # Mean weights from alloc_df (over windows that overlap target OOS)
    alloc_oos = alloc_df[
        pd.to_datetime(alloc_df["TestStart"]) >= oos_target_s
    ]
    w_eq_mean  = alloc_oos["w_equity"].mean() if "w_equity" in alloc_oos else float("nan")
    w_bd_mean  = alloc_oos["w_bond"].mean()   if "w_bond"   in alloc_oos else float("nan")
    w_mmf_mean = alloc_oos["w_mmf"].mean()    if "w_mmf"    in alloc_oos else float("nan")

    log.info(
        "  Portfolio [%s–%s]: CAGR=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%  CalMAR=%.3f  "
        "Realloc=%d  w_eq=%.0f%%  w_bd=%.0f%%",
        port_trimmed.index.min().date(), port_trimmed.index.max().date(),
        m_port["CAGR"]*100, m_port["Sharpe"], m_port["MaxDD"]*100, m_port["CalMAR"],
        len(realloc_log), w_eq_mean*100, w_bd_mean*100,
    )

    # ── Robustness (optional) ──────────────────────────────────────────────
    robustness = run_robustness_for_config(
        label        = label,
        wf_equity_eq = wf_eq_eq,
        wf_results_eq= wf_res_eq,
        wf_equity_bd = wf_eq_bd,
        wf_results_bd= wf_res_bd,
        WIG          = WIG,
        TBSP         = TBSP,
        MMF_EXT      = MMF_EXT,
        bond_gate    = bond_gate,
        train_years  = train_years,
        test_years   = test_years,
    )

    return {
        "config":              label,
        "oos_start":           str(port_trimmed.index.min().date()),
        "oos_end":             str(port_trimmed.index.max().date()),
        "n_realloc":           len(realloc_log),
        "cagr":                m_port["CAGR"],
        "vol":                 m_port["Vol"],
        "sharpe":              m_port["Sharpe"],
        "maxdd":               m_port["MaxDD"],
        "calmar":              m_port["CalMAR"],
        "sortino":             m_port["Sortino"],
        "w_eq_mean":           w_eq_mean,
        "w_bd_mean":           w_bd_mean,
        "w_mmf_mean":          w_mmf_mean,
        "wig_oos_cagr":        m_eq["CAGR"],
        "tbsp_oos_cagr":       m_bd["CAGR"],
        # Robustness results (all None when RUN_MC=False)
        "wig_mc_verdict":      robustness["wig_mc_verdict"],
        "wig_mc_failures":     robustness["wig_mc_failures"],
        "tbsp_mc_verdict":     robustness["tbsp_mc_verdict"],
        "tbsp_mc_failures":    robustness["tbsp_mc_failures"],
        "wig_bb_verdict":      robustness["wig_bb_verdict"],
        "wig_bb_failures":     robustness["wig_bb_failures"],
        "tbsp_bb_verdict":     robustness["tbsp_bb_verdict"],
        "tbsp_bb_failures":    robustness["tbsp_bb_failures"],
    }


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

def _fmt_verdict(verdict):
    """Format a verdict string for the comparison table."""
    if verdict is None:
        return "N/A"
    return verdict   # "ROBUST" or "FRAGILE"


def print_comparison(results, bh_wig_cagr, bh_tbsp_cagr,
                     bh_wig_sharpe, bh_tbsp_sharpe):
    sep  = "=" * 110
    sep2 = "-" * 110

    print()
    print(sep)
    print("  2-ASSET (WIG+TBSP) STRATEGY  vs  SWEEP MODE B  vs  BUY-AND-HOLD")
    print(f"  OOS: {OOS_START} to {OOS_END}  |  Extended TBSP + MMF")
    stop_mode = "ATR-scaled" if USE_ATR_STOP else "fixed %"
    print(f"  Equity trailing stop: {stop_mode}"
          + (f"  (ATR window={ATR_WINDOW})" if USE_ATR_STOP else ""))
    if RUN_MC:
        print(f"  MC: n={N_MC} samples per asset per config"
              + (f"  |  Bootstrap: n={N_BOOTSTRAP}" if RUN_BOOTSTRAP else ""))
    print(sep)
    print()

    # B&H reference
    print("  BUY-AND-HOLD BENCHMARKS (same OOS period):")
    print(f"    WIG  B&H: CAGR={bh_wig_cagr*100:.2f}%  Sharpe={bh_wig_sharpe:.3f}")
    print(f"    TBSP B&H: CAGR={bh_tbsp_cagr*100:.2f}%  Sharpe={bh_tbsp_sharpe:.3f}")
    static_70_30 = 0.70 * bh_wig_cagr + 0.30 * bh_tbsp_cagr
    print(f"    70/30 static mix (approx): CAGR≈{static_70_30*100:.2f}%")
    print()

    # ── OOS Performance table ──────────────────────────────────────────────
    hdr = (
        f"  {'Config':<10} {'CAGR':>7} {'Sharpe':>8} {'MaxDD':>8} "
        f"{'CalMAR':>8} {'Sortino':>8} {'Realloc':>8} "
        f"{'w_EQ':>6} {'w_BD':>6} {'Source'}"
    )
    print(hdr)
    print(sep2)

    print("  -- 2-asset signal strategy (this script) --")
    for r in sorted(results, key=lambda x: -x["calmar"]):
        print(
            f"  {'2A '+r['config']:<10} "
            f"{r['cagr']*100:>6.2f}% "
            f"{r['sharpe']:>8.3f} "
            f"{r['maxdd']*100:>7.2f}% "
            f"{r['calmar']:>8.3f} "
            f"{r['sortino']:>8.3f} "
            f"{r['n_realloc']:>8d} "
            f"{r['w_eq_mean']*100:>5.0f}% "
            f"{r['w_bd_mean']*100:>5.0f}% "
            f"[OOS from {r['oos_start']}]"
        )

    print()
    print("  -- Sweep Mode B (WIG+MSCI_World+TBSP, MSCI_World≈0) --")
    for r in SWEEP_MODEB:
        wig_flag = "✓" if r["wig_mc"] == "ROBUST" else "✗"
        print(
            f"  {'Sw '+r['config']:<10} "
            f"{r['cagr']*100:>6.2f}% "
            f"{r['sharpe']:>8.3f} "
            f"{r['maxdd']*100:>7.2f}% "
            f"{r['calmar']:>8.3f} "
            f"{'N/A':>8} "
            f"{'N/A':>8} "
            f"{'N/A':>5}  "
            f"{'N/A':>4}  "
            f"WIG_MC={wig_flag} TBSP_MC=✓"
        )

    print()
    print("  -- B&H references --")
    for label, cagr, sharpe in [
        ("WIG B&H",  bh_wig_cagr,  bh_wig_sharpe),
        ("TBSP B&H", bh_tbsp_cagr, bh_tbsp_sharpe),
    ]:
        print(
            f"  {label:<10} "
            f"{cagr*100:>6.2f}% "
            f"{sharpe:>8.3f} "
            f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
        )

    # ── Robustness table (only when MC was run) ────────────────────────────
    if RUN_MC:
        print()
        print(sep2)
        print("  ROBUSTNESS VERDICTS")
        print(sep2)

        # Header
        bb_cols = "  BB-WIG   BB-TBSP" if RUN_BOOTSTRAP else ""
        print(
            f"  {'Config':<10}  {'MC-WIG':<9}  {'MC-TBSP':<9}{bb_cols}  "
            f"{'Both ROBUST'}"
        )
        print("  " + "-" * 75)

        for r in sorted(results, key=lambda x: x["config"]):
            wig_mc  = _fmt_verdict(r["wig_mc_verdict"])
            tbsp_mc = _fmt_verdict(r["tbsp_mc_verdict"])
            both_mc = (wig_mc == "ROBUST" and tbsp_mc == "ROBUST")

            if RUN_BOOTSTRAP:
                wig_bb  = _fmt_verdict(r["wig_bb_verdict"])
                tbsp_bb = _fmt_verdict(r["tbsp_bb_verdict"])
                both_bb = (r["wig_bb_verdict"] == "ROBUST" and
                           r["tbsp_bb_verdict"] == "ROBUST")
                both_all = both_mc and both_bb
                bb_str = f"  {wig_bb:<9}  {tbsp_bb:<9}"
            else:
                bb_str   = ""
                both_all = both_mc

            flag = "*** BOTH ROBUST ***" if both_all else ""
            print(
                f"  {'2A '+r['config']:<10}  {wig_mc:<9}  {tbsp_mc:<9}"
                f"{bb_str}  {flag}"
            )

        # ── Failure reason detail ──────────────────────────────────────────
        print()
        print(sep2)
        print("  FAILURE DETAIL  (FRAGILE configs only)")
        print(sep2)

        any_failures = False
        for r in sorted(results, key=lambda x: x["config"]):
            config_label = f"2A {r['config']}"
            has_failure  = False

            # MC failures
            if r["wig_mc_verdict"] == "FRAGILE" and r["wig_mc_failures"]:
                has_failure  = True
                any_failures = True
                print(f"\n  {config_label}  MC-WIG  FRAGILE:")
                for f in r["wig_mc_failures"]:
                    print(f"    - {f}")

            if r["tbsp_mc_verdict"] == "FRAGILE" and r["tbsp_mc_failures"]:
                has_failure  = True
                any_failures = True
                print(f"\n  {config_label}  MC-TBSP  FRAGILE:")
                for f in r["tbsp_mc_failures"]:
                    print(f"    - {f}")

            # Bootstrap failures
            if RUN_BOOTSTRAP:
                if r["wig_bb_verdict"] == "FRAGILE" and r["wig_bb_failures"]:
                    has_failure  = True
                    any_failures = True
                    print(f"\n  {config_label}  BB-WIG  FRAGILE:")
                    for f in r["wig_bb_failures"]:
                        print(f"    - {f}")

                if r["tbsp_bb_verdict"] == "FRAGILE" and r["tbsp_bb_failures"]:
                    has_failure  = True
                    any_failures = True
                    print(f"\n  {config_label}  BB-TBSP  FRAGILE:")
                    for f in r["tbsp_bb_failures"]:
                        print(f"    - {f}")

        if not any_failures:
            print("  (none — all configs passed all checks)")

    # ── Key findings ───────────────────────────────────────────────────────
    print()
    print(sep)
    print("  KEY FINDINGS:")
    best_2a = max(results, key=lambda x: x["calmar"]) if results else None
    if best_2a:
        print(f"  Best 2-asset config: {best_2a['config']}  "
              f"CalMAR={best_2a['calmar']:.3f}  CAGR={best_2a['cagr']*100:.2f}%")
        if RUN_MC:
            wig_v  = _fmt_verdict(best_2a["wig_mc_verdict"])
            tbsp_v = _fmt_verdict(best_2a["tbsp_mc_verdict"])
            print(f"    MC robustness: WIG={wig_v}  TBSP={tbsp_v}")
            if RUN_BOOTSTRAP:
                wig_b  = _fmt_verdict(best_2a["wig_bb_verdict"])
                tbsp_b = _fmt_verdict(best_2a["tbsp_bb_verdict"])
                print(f"    Bootstrap:     WIG={wig_b}  TBSP={tbsp_b}")

    best_B = max(SWEEP_MODEB, key=lambda x: x["calmar"])
    print(f"  Best sweep Mode B:   {best_B['config']}  "
          f"CalMAR={best_B['calmar']:.3f}  CAGR={best_B['cagr']*100:.2f}%")
    if best_2a:
        calmar_diff = best_2a["calmar"] - best_B["calmar"]
        cagr_diff   = best_2a["cagr"]   - best_B["cagr"]
        print(f"  2-asset vs Mode B (best vs best): "
              f"CalMAR {calmar_diff:+.3f}  CAGR {cagr_diff*100:+.2f}pp")

    if RUN_MC:
        robust_configs = [
            r["config"] for r in results
            if r["wig_mc_verdict"] == "ROBUST" and r["tbsp_mc_verdict"] == "ROBUST"
            and (not RUN_BOOTSTRAP or (
                r["wig_bb_verdict"] == "ROBUST" and r["tbsp_bb_verdict"] == "ROBUST"
            ))
        ]
        label = "MC+Bootstrap" if RUN_BOOTSTRAP else "MC"
        if robust_configs:
            print(f"  {label}-robust configs: {', '.join(robust_configs)}")
        else:
            print(f"  No configs passed all {label} checks.")

    print(sep)


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=" * 70)
    log.info("2-ASSET EXTENDED COMPARISON  START: %s", dt.datetime.now())
    log.info("OOS target: %s to %s", OOS_START, OOS_END)
    log.info("Window configs: %s", WINDOW_CONFIGS)
    log.info("Trailing stop: %s%s",
             "ATR-scaled" if USE_ATR_STOP else "fixed %",
             f" (window={ATR_WINDOW}, grid={N_ATR_GRID})" if USE_ATR_STOP else "")
    log.info("RUN_MC=%s (n=%d)  RUN_BOOTSTRAP=%s (n=%d)",
             RUN_MC, N_MC, RUN_BOOTSTRAP, N_BOOTSTRAP)
    if TBSP_EXTENDED_PATH:
        log.info("Extended TBSP: %s", TBSP_EXTENDED_PATH)
    else:
        log.warning("No extended TBSP path set — OOS will start later than 2011")
    log.info("=" * 70)

    tmp_dir = tempfile.mkdtemp()

    # Download
    WIG, TBSP, MMF, W1M, PL10Y, DE10Y = download_all(tmp_dir)

    # Derived series
    MMF_EXT, bond_gate, ret_eq, ret_bd, ret_mmf = build_derived(
        WIG, TBSP, MMF, W1M, PL10Y, DE10Y
    )

    # B&H benchmarks on target OOS period
    wig_bh = WIG["Zamkniecie"].loc[
        (WIG.index >= pd.Timestamp(OOS_START)) &
        (WIG.index <= pd.Timestamp(OOS_END))
    ]
    tbsp_bh = TBSP["Zamkniecie"].loc[
        (TBSP.index >= pd.Timestamp(OOS_START)) &
        (TBSP.index <= pd.Timestamp(OOS_END))
    ]
    bh_wig_eq  = wig_bh  / wig_bh.iloc[0]
    bh_tbsp_eq = tbsp_bh / tbsp_bh.iloc[0]
    m_wig_bh   = compute_metrics(bh_wig_eq)
    m_tbsp_bh  = compute_metrics(bh_tbsp_eq)

    log.info(
        "B&H WIG  [%s–%s]: CAGR=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%",
        OOS_START, OOS_END,
        m_wig_bh["CAGR"]*100, m_wig_bh["Sharpe"], m_wig_bh["MaxDD"]*100,
    )
    log.info(
        "B&H TBSP [%s–%s]: CAGR=%.2f%%  Sharpe=%.3f  MaxDD=%.2f%%",
        OOS_START, OOS_END,
        m_tbsp_bh["CAGR"]*100, m_tbsp_bh["Sharpe"], m_tbsp_bh["MaxDD"]*100,
    )

    # Run all window configs
    results = []
    for train_years, test_years in WINDOW_CONFIGS:
        r = run_config(
            train_years, test_years,
            WIG, TBSP, MMF_EXT,
            bond_gate, ret_eq, ret_bd, ret_mmf,
        )
        if r is not None:
            results.append(r)

    # Print comparison
    print_comparison(
        results,
        bh_wig_cagr   = m_wig_bh["CAGR"],
        bh_tbsp_cagr  = m_tbsp_bh["CAGR"],
        bh_wig_sharpe = m_wig_bh["Sharpe"],
        bh_tbsp_sharpe= m_tbsp_bh["Sharpe"],
    )

    # Save results CSV
    if results:
        out_path = "twoasset_comparison_results.csv"
        # Strip list columns before saving to CSV
        csv_rows = []
        for r in results:
            row = {k: v for k, v in r.items()
                   if not isinstance(v, list)}
            csv_rows.append(row)
        pd.DataFrame(csv_rows).to_csv(out_path, index=False)
        log.info("Results saved to %s", out_path)

    log.info("DONE: %s", dt.datetime.now())


if __name__ == "__main__":
    main()