"""
multiasset_daily_runfile.py
===========================
Daily runner for the multi-asset pension fund strategy (WIG + TBSP + MMF).

Runs Phases 1–5 only (data download, per-asset walk-forward, allocation
walk-forward, reporting).  Robustness phases (7 & 8) are disabled to
keep the GitHub Actions run within the 6-hour job timeout on a 2-core
runner.  Full robustness can be triggered by running
multiasset_runfile.py manually or via workflow_dispatch on the full
workflow.

After Phase 5, calls multiasset_daily_output.build_daily_outputs() to
produce the four daily artefacts:
  outputs/multiasset_signal_status.txt
  outputs/multiasset_signal_log.csv
  outputs/multiasset_equity_chart.png
  outputs/multiasset_signal_snapshot.json

These are then uploaded to Google Drive by the GitHub Actions workflow.

Based on multiasset_freeze_0_1/multiasset_runfile.py.
All parameter settings are identical to the frozen v0.1 configuration.
"""

import os
import sys
import logging
import tempfile
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

from strategy_test_library import (
    download_csv,
    load_csv,
    walk_forward,
    compute_metrics,
    compute_buy_and_hold,
    analyze_trades,
    print_backtest_report,
)

from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    print_multiasset_report,
)

from multiasset_daily_output import build_daily_outputs


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE = "MultiassetDailyRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(console_handler)

logging.info("=" * 80)
logging.info("MULTI-ASSET DAILY RUNFILE START: %s", dt.datetime.now())
logging.info("=" * 80)


# ============================================================
# USER SETTINGS  (must match multiasset_freeze_0_1 frozen values)
# ============================================================

POSITION_MODE = "full"

TRAIN_YEARS_EQ = 8
TEST_YEARS_EQ  = 2
TRAIN_YEARS_BD = 8
TEST_YEARS_BD  = 2

VOL_WINDOW = 20

_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

logging.info(
    "Parallel computing: %d logical cores detected, using %d jobs.",
    _cpu_count, N_JOBS
)

COOLDOWN_DAYS = 10
ANNUAL_CAP    = 12
ALLOC_OBJECTIVE = "calmar"

FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]

# Parameter grids — identical to freeze_0_1
X_GRID_EQ   = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ   = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ     = [50, 75, 100]
SLOW_EQ     = [150, 200, 250]
TV_EQ       = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ       = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ   = [126, 252]

X_GRID_BD   = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD   = [0.001]
FAST_BD     = [50, 100, 150]
SLOW_BD     = [200, 300, 400, 500]
TV_BD       = [0.08, 0.10, 0.12]
SL_BD       = [0.01, 0.02, 0.03]

SPREAD_PREFILTER_HARD_EXIT = False

# Robustness phases are OFF for daily run
RUN_MONTE_CARLO_PARAM_SINGLE      = False
RUN_BLOCK_BOOTSTRAP_SINGLE        = False
RUN_ROBUSTNESS_ALLOCATION_WEIGHT  = False

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOAD DATA")
logging.info("=" * 80)

tmp_dir = tempfile.mkdtemp()

csv_wig  = os.path.join(tmp_dir, "wig.csv")
download_csv("https://stooq.pl/q/d/l/?s=wig&i=d", csv_wig)
WIG = load_csv(csv_wig)
if WIG is None:
    logging.error("Failed to load WIG data. Exiting.")
    sys.exit(1)
logging.info("WIG loaded: %d rows  (%s to %s)",
             len(WIG), WIG.index.min().date(), WIG.index.max().date())

csv_tbsp = os.path.join(tmp_dir, "tbsp.csv")
download_csv("https://stooq.pl/q/d/l/?s=tbsp&i=d", csv_tbsp)
TBSP = load_csv(csv_tbsp)
if TBSP is None:
    logging.error("Failed to load TBSP data. Exiting.")
    sys.exit(1)
logging.info("TBSP loaded: %d rows  (%s to %s)",
             len(TBSP), TBSP.index.min().date(), TBSP.index.max().date())

csv_mmf  = os.path.join(tmp_dir, "mmf.csv")
download_csv("https://stooq.pl/q/d/l/?s=2720.n&i=d", csv_mmf)
MMF = load_csv(csv_mmf)
if MMF is None:
    logging.error("Failed to load MMF data. Exiting.")
    sys.exit(1)
logging.info("MMF  loaded: %d rows  (%s to %s)",
             len(MMF), MMF.index.min().date(), MMF.index.max().date())

# Yield curve data for bond spread pre-filter
csv_pl10y  = os.path.join(tmp_dir, "pl10y.csv")
csv_de10y  = os.path.join(tmp_dir, "de10y.csv")
download_csv("https://stooq.pl/q/d/l/?s=10yply.b&i=d", csv_pl10y)
download_csv("https://stooq.pl/q/d/l/?s=10ydey.b&i=d", csv_de10y)
PL10Y = load_csv(csv_pl10y)
DE10Y = load_csv(csv_de10y)

if PL10Y is None or DE10Y is None:
    logging.warning("Yield curve data unavailable — spread pre-filter disabled.")
    spread_prefilter = None
else:
    logging.info("PL10Y loaded: %d rows  (%s to %s)",
                 len(PL10Y), PL10Y.index.min().date(), PL10Y.index.max().date())
    logging.info("DE10Y loaded: %d rows  (%s to %s)",
                 len(DE10Y), DE10Y.index.min().date(), DE10Y.index.max().date())

def _daily_returns(df: pd.DataFrame, col: str = "Zamkniecie") -> pd.Series:
    return df[col].pct_change().rename(col)

ret_eq  = _daily_returns(WIG).dropna()
ret_bd  = _daily_returns(TBSP).dropna()
ret_mmf = _daily_returns(MMF).dropna()

logging.info(
    "Return series: equity %d rows | bond %d rows | mmf %d rows",
    len(ret_eq), len(ret_bd), len(ret_mmf)
)


# ============================================================
# PHASE 2 — EQUITY SIGNAL WALK-FORWARD (WIG)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 2: EQUITY SIGNAL WALK-FORWARD  (WIG, train=%dy test=%dy)",
             TRAIN_YEARS_EQ, TEST_YEARS_EQ)
logging.info("=" * 80)

wf_equity_eq, wf_results_eq, wf_trades_eq = walk_forward(
    df                    = WIG,
    cash_df               = MMF,
    train_years           = TRAIN_YEARS_EQ,
    test_years            = TEST_YEARS_EQ,
    vol_window            = VOL_WINDOW,
    selected_mode         = POSITION_MODE,
    filter_modes_override = FORCE_FILTER_MODE_EQ,
    X_grid                = X_GRID_EQ,
    Y_grid                = Y_GRID_EQ,
    fast_grid             = FAST_EQ,
    slow_grid             = SLOW_EQ,
    tv_grid               = TV_EQ,
    sl_grid               = SL_EQ,
    mom_lookback_grid     = MOM_LB_EQ,
    n_jobs                = N_JOBS,
)

if wf_equity_eq.empty:
    logging.error("Equity walk-forward returned empty equity curve. Exiting.")
    sys.exit(1)

metrics_eq = compute_metrics(wf_equity_eq)
logging.info(
    "EQUITY SIGNAL OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
    metrics_eq["CAGR"] * 100, metrics_eq["Sharpe"],
    metrics_eq["MaxDD"] * 100, metrics_eq["CalMAR"]
)

if not wf_trades_eq.empty and "Exit Reason" in wf_trades_eq.columns:
    closed_eq = wf_trades_eq[~wf_trades_eq["Exit Reason"].isin(BOUNDARY_EXITS)].copy()
else:
    closed_eq = pd.DataFrame()

sig_equity = build_signal_series(wf_equity_eq, wf_trades_eq)
logging.info(
    "Equity signal: %.1f%% of OOS days in position (%d of %d)",
    sig_equity.mean() * 100, sig_equity.sum(), len(sig_equity)
)


# ============================================================
# PHASE 3 — BOND SIGNAL WALK-FORWARD (TBSP)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 3: BOND SIGNAL WALK-FORWARD  (TBSP, train=%dy test=%dy)",
             TRAIN_YEARS_BD, TEST_YEARS_BD)
logging.info("=" * 80)

wf_equity_bd, wf_results_bd, wf_trades_bd = walk_forward(
    df                    = TBSP,
    cash_df               = MMF,
    train_years           = TRAIN_YEARS_BD,
    test_years            = TEST_YEARS_BD,
    vol_window            = VOL_WINDOW,
    selected_mode         = POSITION_MODE,
    filter_modes_override = FORCE_FILTER_MODE_BD,
    X_grid                = X_GRID_BD,
    Y_grid                = Y_GRID_BD,
    fast_grid             = FAST_BD,
    slow_grid             = SLOW_BD,
    tv_grid               = TV_BD,
    sl_grid               = SL_BD,
    mom_lookback_grid     = [252],
    n_jobs                = N_JOBS,
)

if wf_equity_bd.empty:
    logging.error("Bond walk-forward returned empty equity curve. Exiting.")
    sys.exit(1)

metrics_bd = compute_metrics(wf_equity_bd)
logging.info(
    "BOND SIGNAL OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
    metrics_bd["CAGR"] * 100, metrics_bd["Sharpe"],
    metrics_bd["MaxDD"] * 100, metrics_bd["CalMAR"]
)

if not wf_trades_bd.empty and "Exit Reason" in wf_trades_bd.columns:
    closed_bd = wf_trades_bd[~wf_trades_bd["Exit Reason"].isin(BOUNDARY_EXITS)].copy()
else:
    closed_bd = pd.DataFrame()

sig_bond_raw = build_signal_series(wf_equity_bd, wf_trades_bd)

# --- Bond spread pre-filter (optional; matches freeze_0_1 logic) ---
if PL10Y is not None and DE10Y is not None:
    try:
        spread        = PL10Y["Zamkniecie"] - DE10Y["Zamkniecie"]
        spread_change = spread.rolling(63).apply(lambda x: x.iloc[-1] - x.iloc[0])
        prefilter_ok  = (spread_change <= 0.75)
        common_idx    = sig_bond_raw.index.intersection(prefilter_ok.index)
        if SPREAD_PREFILTER_HARD_EXIT:
            sig_bond = (sig_bond_raw.loc[common_idx] & prefilter_ok.loc[common_idx]).astype(int)
        else:
            sig_bond = sig_bond_raw.copy()
        spread_prefilter_aligned = prefilter_ok.reindex(sig_bond.index, fill_value=True)
        logging.info("Bond spread pre-filter computed (hard_exit=%s)", SPREAD_PREFILTER_HARD_EXIT)
    except Exception as exc:
        logging.warning("Bond spread pre-filter failed: %s — using raw bond signal.", exc)
        sig_bond = sig_bond_raw
        spread_prefilter_aligned = None
else:
    sig_bond = sig_bond_raw
    spread_prefilter_aligned = None

logging.info(
    "Bond signal: %.1f%% of OOS days in position (%d of %d)",
    sig_bond.mean() * 100, sig_bond.sum(), len(sig_bond)
)


# ============================================================
# PHASE 4 — ALLOCATION WALK-FORWARD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: ALLOCATION WALK-FORWARD  (both-signals-on weight optimisation)")
logging.info("=" * 80)

oos_start_eq = wf_results_eq["TestStart"].min()
oos_start_bd = wf_results_bd["TestStart"].min()
oos_start    = max(oos_start_eq, oos_start_bd)

oos_end_eq = wf_results_eq["TestEnd"].max()
oos_end_bd = wf_results_bd["TestEnd"].max()
oos_end    = min(oos_end_eq, oos_end_bd)

logging.info("Common OOS period: %s to %s", oos_start.date(), oos_end.date())

def _trim(s, start, end):
    return s.loc[(s.index >= start) & (s.index <= end)]

sig_eq_oos  = _trim(sig_equity, oos_start, oos_end)
sig_bd_oos  = _trim(sig_bond,   oos_start, oos_end)
ret_eq_oos  = _trim(ret_eq,     oos_start, oos_end)
ret_bd_oos  = _trim(ret_bd,     oos_start, oos_end)
ret_mmf_oos = _trim(ret_mmf,    oos_start, oos_end)

portfolio_equity, weights_series, reallocation_log, alloc_results_df = (
    allocation_walk_forward(
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_full  = sig_equity,
        sig_bond_full    = sig_bond,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        wf_results_eq    = wf_results_eq,
        wf_results_bd    = wf_results_bd,
        step             = 0.10,
        objective        = ALLOC_OBJECTIVE,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )
)

if portfolio_equity.empty:
    logging.error("Portfolio equity curve is empty after allocation walk-forward.")
    sys.exit(1)


# ============================================================
# PHASE 5 — REPORTING
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: REPORTING")
logging.info("=" * 80)

portfolio_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}

bh_equity_eq, bh_metrics_eq = compute_buy_and_hold(
    WIG, price_col="Zamkniecie", start=oos_start, end=oos_end
)
bh_equity_bd, bh_metrics_bd = compute_buy_and_hold(
    TBSP, price_col="Zamkniecie", start=oos_start, end=oos_end
)

print_multiasset_report(
    portfolio_metrics  = portfolio_metrics,
    bh_equity_metrics  = bh_metrics_eq,
    bh_bond_metrics    = bh_metrics_bd,
    alloc_results_df   = alloc_results_df,
    reallocation_log   = reallocation_log,
    sig_equity         = sig_eq_oos,
    sig_bond           = sig_bd_oos,
    oos_start          = oos_start,
    oos_end            = oos_end,
    sig_bond_raw       = sig_bond_raw,
    spread_prefilter   = spread_prefilter_aligned,
)

logging.info(
    "PORTFOLIO OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
    portfolio_metrics["CAGR"] * 100,
    portfolio_metrics["Sharpe"],
    portfolio_metrics["MaxDD"] * 100,
    portfolio_metrics["CalMAR"],
)


# ============================================================
# PHASE 6 — DAILY OUTPUT (signal, log, chart, snapshot)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: DAILY OUTPUT")
logging.info("=" * 80)

result = build_daily_outputs(
    wf_equity_eq      = wf_equity_eq,
    wf_trades_eq      = wf_trades_eq,
    wf_results_eq     = wf_results_eq,
    wf_equity_bd      = wf_equity_bd,
    wf_trades_bd      = wf_trades_bd,
    wf_results_bd     = wf_results_bd,
    portfolio_equity  = portfolio_equity,
    portfolio_metrics = portfolio_metrics,
    weights_series    = weights_series,
    reallocation_log  = reallocation_log,
    bh_eq_equity      = bh_equity_eq,
    bh_eq_metrics     = bh_metrics_eq,
    bh_bd_equity      = bh_equity_bd,
    bh_bd_metrics     = bh_metrics_bd,
    WIG               = WIG,
    TBSP              = TBSP,
    sig_eq_oos        = sig_eq_oos,
    sig_bd_oos        = sig_bd_oos,
    output_dir        = OUTPUT_DIR,
    # Drive credentials: passed via env vars by GitHub Actions workflow.
    # build_daily_outputs will pre-fetch multiasset_signal_log.csv from
    # Drive before writing today's row (ensures correct ENTER/EXIT/HOLD
    # determination on a fresh, stateless runner).
    gdrive_folder_id   = os.getenv("GDRIVE_FOLDER_ID"),
    gdrive_credentials = "/tmp/credentials.json"
    if os.path.exists("/tmp/credentials.json") else None,
)

logging.info(
    "Daily output complete — action=%s  eq=%s  bd=%s",
    result["action"],
    result["signal_equity"],
    result["signal_bond"],
)

logging.info("=" * 80)
logging.info("MULTI-ASSET DAILY RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("=" * 80)
