"""
multiasset_daily_runfile.py
===========================
Daily runner for the multi-asset pension fund strategy (WIG + TBSP + MMF).

Runs Phases 1–6 only (data download, per-asset walk-forward, allocation
walk-forward, reporting, daily output).  Robustness phases (7 & 8) are
disabled to keep the GitHub Actions run within the 6-hour job timeout.

After Phase 6, calls multiasset_daily_output.build_daily_outputs() to
produce the four daily artefacts:
  outputs/multiasset_signal_status.txt
  outputs/multiasset_signal_log.csv
  outputs/multiasset_equity_chart.png
  outputs/multiasset_signal_snapshot.json

Based on multiasset_runfile.py.  All parameter settings are identical to
the frozen v0.4+ configuration.  7+2 walk-forward windows.
"""

import os
import sys
import logging
import tempfile
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from strategy_test_library import (
    load_csv,
    load_stooq_local,       # consolidated loader — replaces per-runfile _stooq()
    get_n_jobs,             # canonical N_JOBS calculation
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
    build_standard_two_asset_data,  # consolidated setup — replaces inline block
    build_spread_prefilter,
    build_yield_momentum_prefilter,
)
from multiasset_daily_output import build_daily_outputs
from global_equity_library import build_mmf_extended
from price_series_builder import build_and_upload
from stooq_hybrid_updater import run_update


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE = "MultiassetRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logging.info("=" * 80)
logging.info("MULTI-ASSET DAILY RUNFILE START: %s", dt.datetime.now())
logging.info("=" * 80)


# ============================================================
# USER SETTINGS
# ============================================================

POSITION_MODE = "full"

TRAIN_YEARS_EQ = 7
TEST_YEARS_EQ  = 2
TRAIN_YEARS_BD = 7
TEST_YEARS_BD  = 2

VOL_WINDOW = 20

N_JOBS = get_n_jobs()
logging.info("Parallel computing: %d jobs.", N_JOBS)

COOLDOWN_DAYS = 10
ANNUAL_CAP    = 12
ALLOC_OBJECTIVE = "calmar"

FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]

X_GRID_EQ   = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ   = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ     = [50, 75, 100]
SLOW_EQ     = [150, 200, 250]
TV_EQ       = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ       = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ   = [126, 252]

USE_ATR_STOP    = True
ATR_WINDOW      = 20
N_ATR_GRID      = [0.08, 0.10, 0.12, 0.15, 0.20]

USE_ATR_STOP_BD = False
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [0.05, 0.08, 0.10, 0.15]

X_GRID_BD   = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD   = [0.001]
FAST_BD     = [50, 100, 150]
SLOW_BD     = [200, 300, 400, 500]
TV_BD       = [0.08, 0.10, 0.12]
SL_BD       = [0.01, 0.02, 0.03]

YIELD_PREFILTER_THRESHOLD_BP = 50.0
CLOSE_COL  = "Zamkniecie"
FAST_MODE  = True
WIG_DATA_FLOOR = "1995-01-02"
MMF_FLOOR      = "1995-01-02"
DATA_START     = "1990-01-01"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOAD DATA")
logging.info("=" * 80)

run_update(get_funds=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# All series loaded via consolidated load_stooq_local() — no per-runfile _stooq()
WIG   = load_stooq_local("wig",       "WIG",     DATA_DIR, DATA_START)
WIG   = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
MMF   = load_stooq_local("fund_2720", "MMF",     DATA_DIR, DATA_START)
W1M   = load_stooq_local("wibor1m",  "WIBOR1M", DATA_DIR, DATA_START, mandatory=False)
PL10Y = load_stooq_local("pl10y",    "PL10Y",   DATA_DIR, DATA_START)
DE10Y = load_stooq_local("de10y",    "DE10Y",   DATA_DIR, DATA_START)

folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()

TBSP = build_and_upload(
    folder_id         = folder_id,
    raw_filename      = "tbsp_extended_full.csv",
    combined_filename = "tbsp_extended_combined.csv",
    extension_ticker  = "tbsp",
    extension_source  = "stooq",
)
if TBSP is None:
    TBSP = load_stooq_local("tbsp", "TBSP", DATA_DIR, DATA_START)
    if TBSP is None:
        logging.error("FAIL: TBSP build/update failed.")
        sys.exit(1)
logging.info("OK  %-14s  %5d rows  %s to %s",
             "TBSP", len(TBSP), TBSP.index.min().date(), TBSP.index.max().date())
for col in ["Najwyzszy", "Najnizszy"]:
    if col not in TBSP.columns:
        TBSP[col] = TBSP[CLOSE_COL]


# ============================================================
# PHASE 2 — DERIVED SERIES
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 2: DERIVED SERIES")
logging.info("=" * 80)

# Consolidated into build_standard_two_asset_data() — replaces ~50 lines of setup
derived = build_standard_two_asset_data(
    wig                  = WIG,
    tbsp                 = TBSP,
    mmf                  = MMF,
    wibor1m              = W1M,
    pl10y                = PL10Y,
    de10y                = DE10Y,
    mmf_floor            = MMF_FLOOR,
    yield_prefilter_bp   = YIELD_PREFILTER_THRESHOLD_BP,
)
MMF_EXT       = derived["mmf_ext"]
bond_entry_gate = derived["bond_gate"]
ret_eq        = derived["ret_eq"]
ret_bd        = derived["ret_bd"]
ret_mmf       = derived["ret_mmf"]
spread_prefilter = derived["spread_pf"]
yield_prefilter  = derived["yield_pf"]


# ============================================================
# PHASE 3 — EQUITY SIGNAL WALK-FORWARD (WIG)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 3: EQUITY SIGNAL WALK-FORWARD  (WIG, train=%dy test=%dy)",
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
    fast_mode             = FAST_MODE,
    use_atr_stop          = USE_ATR_STOP,
    N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
    atr_window            = ATR_WINDOW,
)

if wf_equity_eq.empty:
    logging.error("Equity walk-forward returned empty equity curve. Exiting.")
    sys.exit(1)

metrics_eq = compute_metrics(wf_equity_eq)
logging.info("EQUITY SIGNAL OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
             metrics_eq["CAGR"]*100, metrics_eq["Sharpe"],
             metrics_eq["MaxDD"]*100, metrics_eq["CalMAR"])

sig_equity = build_signal_series(wf_equity_eq, wf_trades_eq)
logging.info("Equity signal: %.1f%% of OOS days in position", sig_equity.mean()*100)


# ============================================================
# PHASE 4 — BOND SIGNAL WALK-FORWARD (TBSP)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: BOND SIGNAL WALK-FORWARD  (TBSP, train=%dy test=%dy)",
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
    entry_gate_series     = bond_entry_gate,
    fast_mode             = FAST_MODE,
    use_atr_stop          = USE_ATR_STOP_BD,
    N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
    atr_window            = ATR_WINDOW_BD,
)

if wf_equity_bd.empty:
    logging.error("Bond walk-forward returned empty equity curve. Exiting.")
    sys.exit(1)

metrics_bd = compute_metrics(wf_equity_bd)
logging.info("BOND SIGNAL OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
             metrics_bd["CAGR"]*100, metrics_bd["Sharpe"],
             metrics_bd["MaxDD"]*100, metrics_bd["CalMAR"])

sig_bond = build_signal_series(wf_equity_bd, wf_trades_bd)
logging.info("Bond signal: %.1f%% of OOS days in position", sig_bond.mean()*100)

sig_bond_raw = build_signal_series(wf_equity_bd, wf_trades_bd)  # pre-filter reference copy

spread_prefilter_aligned = (
    spread_prefilter.reindex(sig_bond.index, method="ffill").fillna(1).astype(int)
)
yield_prefilter_aligned = (
    yield_prefilter.reindex(sig_bond.index, method="ffill").fillna(1).astype(int)
)
sig_bond_post_spread = sig_bond.copy()


# ============================================================
# PHASE 5 — ALLOCATION WALK-FORWARD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: ALLOCATION WALK-FORWARD  (both-signals-on weight optimisation)")
logging.info("=" * 80)

oos_start = max(wf_results_eq["TestStart"].min(), wf_results_bd["TestStart"].min())
oos_end   = min(wf_results_eq["TestEnd"].max(),   wf_results_bd["TestEnd"].max())
logging.info("Common OOS period: %s to %s", oos_start.date(), oos_end.date())

def _trim(s, start, end):
    return s.loc[(s.index >= start) & (s.index <= end)]

sig_eq_oos  = _trim(sig_equity, oos_start, oos_end)
sig_bd_oos  = _trim(sig_bond,   oos_start, oos_end)

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

portfolio_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}
logging.info("Portfolio OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f  Reallocs=%d",
             portfolio_metrics["CAGR"]*100, portfolio_metrics["Sharpe"],
             portfolio_metrics["MaxDD"]*100, portfolio_metrics["CalMAR"],
             len(reallocation_log))


# ============================================================
# PHASE 6 — REPORTING
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: REPORTING")
logging.info("=" * 80)

bh_equity_eq, bh_metrics_eq = compute_buy_and_hold(
    WIG, price_col=CLOSE_COL, start=oos_start, end=oos_end
)
bh_equity_bd, bh_metrics_bd = compute_buy_and_hold(
    TBSP, price_col=CLOSE_COL, start=oos_start, end=oos_end
)

print_multiasset_report(
    portfolio_metrics    = portfolio_metrics,
    bh_equity_metrics    = bh_metrics_eq,
    bh_bond_metrics      = bh_metrics_bd,
    alloc_results_df     = alloc_results_df,
    reallocation_log     = reallocation_log,
    sig_equity           = sig_eq_oos,
    sig_bond             = sig_bd_oos,
    oos_start            = oos_start,
    oos_end              = oos_end,
    sig_bond_raw         = sig_bond_raw,
    spread_prefilter     = spread_prefilter_aligned,
    yield_prefilter      = yield_prefilter,
    sig_bond_post_spread = sig_bond_post_spread,
)


# ============================================================
# PHASE 7 — DAILY OUTPUT
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 7: DAILY OUTPUT")
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
    gdrive_folder_id  = os.getenv("GDRIVE_FOLDER_ID"),
    gdrive_credentials= "/tmp/credentials.json"
                        if os.path.exists("/tmp/credentials.json") else None,
)

logging.info("Daily output complete — action=%s  eq=%s  bd=%s",
             result["action"], result["signal_equity"], result["signal_bond"])

logging.info("=" * 80)
logging.info("MULTI-ASSET DAILY RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("=" * 80)
