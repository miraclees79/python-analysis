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

Based on multiasset_freeze_0_3/multiasset_runfile.py.
All parameter settings are identical to the frozen v0.4 configuration.
9+1 years
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
    allocation_weight_robustness,
    print_allocation_robustness_report,
    build_spread_prefilter,
    build_yield_price_proxy,
    build_yield_momentum_prefilter,
)

from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    run_block_bootstrap_robustness,
    analyze_bootstrap,
    BOND_THRESHOLDS_MC,
    BOND_THRESHOLDS_BOOTSTRAP,
    EQUITY_THRESHOLDS_MC,
    EQUITY_THRESHOLDS_BOOTSTRAP,
)
from multiasset_daily_output import build_daily_outputs

from price_series_builder import build_and_upload, load_combined_from_drive
from global_equity_library import build_mmf_extended
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
logging.info("MULTI-ASSET RUNFILE START: %s", dt.datetime.now())
logging.info("=" * 80)


# ============================================================
# USER SETTINGS
# ============================================================

# --- Position sizing mode ---
# 'full'       : always 100% in the signal asset (recommended for first run)
# 'vol_entry'  : size at entry using volatility targeting
# 'vol_dynamic': size updated daily
POSITION_MODE = "full"

# --- Walk-forward window lengths (years) ---
TRAIN_YEARS_EQ = 7     # equity signal training window
TEST_YEARS_EQ  = 2     # equity signal test window
TRAIN_YEARS_BD = 7     # bond signal training window
TEST_YEARS_BD  = 2     # bond signal test window

# --- Volatility window (days) ---
VOL_WINDOW = 20

# --- Parallelism ---
_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

logging.info(
    "Parallel computing : %d logical cores detected, using %d jobs.",
    _cpu_count, N_JOBS
)

# --- Reallocation gate ---
COOLDOWN_DAYS = 10
ANNUAL_CAP    = 12

# --- Allocation optimisation objective ---
# Applied in Phase 4 only.  Per-asset signal optimisation always uses CalMAR.
ALLOC_OBJECTIVE = "calmar"

# --- Force filter mode (set None for automatic, or e.g. ["ma"] to restrict) ---
FORCE_FILTER_MODE_EQ = None    # equity signal
FORCE_FILTER_MODE_BD = ["ma"]  # bond: MA-only always (no momentum / fund breadth)

# --- WIG signal parameter grids (same family as WIG20TR) ---
X_GRID_EQ   = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ   = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ     = [50, 75, 100]
SLOW_EQ     = [150, 200, 250]
TV_EQ       = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ       = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ   = [126, 252]

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
N_ATR_GRID      = [0.08, 0.10, 0.12, 0.15, 0.20]   # Normalised ATR grid (same scale as X_GRID_EQ)
 
USE_ATR_STOP_BD = False          # Bond trailing stop mode (can differ from equity)
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [0.05, 0.08, 0.10, 0.15]          # Normalised ATR grid (same scale as X_GRID_BD)


# --- Bond signal source ---
# True  = use PL10Y yield-derived constant-maturity bond price as signal source
#         (longer history back to 1999, duration-weighted price proxy)
# False = use TBSP price index as signal source (current behaviour, from 2006)
# Execution is always on TBSP regardless of this setting.
USE_YIELD_SIGNAL = False  # default preserves current behaviour

# --- TBSP signal parameter grids ---
# Breakout disabled: Y_GRID_BD = [0.001] (price always > near-zero trough).
# Trailing stop X kept — still caps drawdown within a position.
# MA grid: wider slow MAs to capture multi-year bond trends.
X_GRID_BD   = [0.05, 0.08, 0.10, 0.15]         # trailing stop from peak
Y_GRID_BD   = [0.001]                           # effectively no breakout threshold
FAST_BD     = [50, 100, 150]
SLOW_BD     = [200, 300, 400, 500]
TV_BD       = [0.08, 0.10, 0.12]
SL_BD       = [0.01, 0.02, 0.03]               # tight absolute stops for bonds

# Validate: slow - fast >= 75 is enforced inside walk_forward automatically

# --- Yield price proxy signal parameter grids ---
# Proxy is significantly more volatile than TBSP price index.
# Stops must be wide enough to survive normal yield oscillations
# without firing on noise.
#
# Reference volatility: yield proxy daily vol ~0.4-0.6% 
# (vs TBSP ~0.15-0.25%)
# → stops need to be roughly 2-3x wider than TBSP equivalents
#
X_GRID_YLD  = [0.10, 0.15, 0.20, 0.25, 0.30]  # trailing stop from peak
Y_GRID_YLD  = [0.001]                           # breakout still disabled
FAST_YLD    = [50, 100, 150]                    # unchanged — scale-independent
SLOW_YLD    = [200, 300, 400, 500]              # unchanged — scale-independent
TV_YLD      = [0.08, 0.10, 0.12]               # unchanged
SL_YLD      = [0.05, 0.08, 0.10, 0.15]         # 4-5x wider than TBSP SL_BD


# Monte Carlo robustness checks 
RUN_MONTE_CARLO_PARAM_SINGLE = True # MC parameter robustness for single asset strategies
ITERATIONS_MC_PARAM_SINGLE = 1000 # 1000 is the true test variant, 10 for smoke test

RUN_BLOCK_BOOTSTRAP_SINGLE = False # Run bootstrap robustness test for single asset strategies
ITERATIONS_BOOTSTRAP_SINGLE = 500 # 500 is the true test variant, 10 for smoke test
# ATTENTION - # ~3-6h on 12-core machine — run overnight

RUN_ROBUSTNESS_ALLOCATION_WEIGHT = True # Run deterministic perturbations on allocation weights


# --- Bond Spread pre-filter mode ---
# True  = Option A: hard gate — pre-filter breach forces immediate exit
#                   from any open bond position (AND gate on daily sig_bond)
# False = Option B: entry gate only — pre-filter breach blocks new bond
#                   entries but does not force exit from existing positions
SPREAD_PREFILTER_HARD_EXIT = False

# --- Yield momentum pre-filter ---
# Blocks new bond entries when PL 10Y yield has risen sharply.
# Complements the spread pre-filter (which responds to PL-DE spread widening).
# The two filters address different regimes:
#   Spread filter : catches external stress (PL risk premium widening vs DE)
#   Yield filter  : catches domestic rate-rising cycles (NBP hiking, inflation)
#
# True  = Option A: hard exit — rising yield forces exit from open bond position
# False = Option B: entry gate only — blocks new entries, won't exit existing
YIELD_PREFILTER_HARD_EXIT   = False   # Option B default

# Threshold: block entry if 63-day yield change exceeds this many basis points
# 50bp ~= one NBP hike cycle move over a quarter
# 75bp ~= more aggressive — only blocks sustained hiking
YIELD_PREFILTER_THRESHOLD_BP = 50.0

CLOSE_COL = "Zamkniecie"   # stooq close column name used throughout
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

GDRIVE_FOLDER_ID_DEFAULT = None
# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOAD DATA")
logging.info("=" * 80)

tmp_dir = tempfile.mkdtemp()

def download_all(tmp_dir):
    logging.info("=" * 70)
    logging.info("DOWNLOADING DATA")
    logging.info("=" * 70)
    
    run_update(get_funds=False) # load data from GDrive to /data subfolder

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    def _stooq(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame | None:
        """
        Load a stooq series from /data subfolder, clip to DATA_START.
        """
    
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
        df = load_csv(path)
        if df is None:
            if mandatory:
                logging.error("FAIL: load_csv returned None for %s — exiting.", label)
                sys.exit(1)
                return None
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info(
            "OK  : %-14s  %5d rows  %s to %s",
            label, len(df), df.index.min().date(), df.index.max().date(),
            )
        return df



    WIG   = _stooq("wig",       "WIG")
    WIG = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    MMF   = _stooq("fund_2720",    "MMF")
    W1M   = _stooq("wibor1m",  "WIBOR1M", mandatory=False)
    PL10Y = _stooq("pl10y",  "PL10Y")
    DE10Y = _stooq("de10y",  "DE10Y")

    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()

    TBSP = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "tbsp_extended_full.csv",
        combined_filename = "tbsp_extended_combined.csv",
        extension_ticker  = "tbsp",
        extension_source  = "stooq",
    )

    if TBSP is None:
        TBSP = _stooq("tbsp",  "TBSP")
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

WIG, TBSP, MMF, W1M, PL10Y, DE10Y = download_all(tmp_dir)

if PL10Y is None or DE10Y is None:
    logging.error("Failed to load yield data for spread pre-filter. Exiting.")
    sys.exit(1)
logging.info("PL10Y loaded: %d rows  (%s to %s)",
             len(PL10Y), PL10Y.index.min().date(), PL10Y.index.max().date())
logging.info("DE10Y loaded: %d rows  (%s to %s)",
             len(DE10Y), DE10Y.index.min().date(), DE10Y.index.max().date())

if W1M is not None:
    MMF_EXT = build_mmf_extended(MMF, W1M, floor_date="1994-10-03")
    logging.info("MMF extended back to %s", MMF_EXT.index.min().date())
else:
    MMF_EXT = MMF
    logging.warning("WIBOR1M unavailable — using standard MMF (starts ~1999)")

# --- Compute daily return series from price columns ---
# stooq CSVs use 'Zamkniecie' as the close price column.
def _daily_returns(df: pd.DataFrame, col: str = "Zamkniecie") -> pd.Series:
    return df[col].pct_change().rename(col)

ret_eq  = _daily_returns(WIG)
ret_bd  = _daily_returns(TBSP)
ret_mmf = _daily_returns(MMF)

# Drop the first NaN row from each
ret_eq  = ret_eq.dropna()
ret_bd  = ret_bd.dropna()
ret_mmf = ret_mmf.dropna()


# After line 306 (ret_mmf = ret_mmf.dropna())

# --- Bond entry gate (spread + yield) — built once, passed into walk_forward ---
# Applied inside training windows to match the constraints active in production.
spread_prefilter = build_spread_prefilter(PL10Y, DE10Y)
yield_prefilter  = build_yield_momentum_prefilter(
    PL10Y,
    rise_threshold_bp = YIELD_PREFILTER_THRESHOLD_BP,
    lookback_days     = 63,
)
# Replace the bond_entry_gate block entirely:

# Align both filters to TBSP's actual date index
tbsp_index = TBSP.index

spread_gate = (
    spread_prefilter
    .reindex(tbsp_index, method="ffill")
    .fillna(1)
    .astype(int)
)

yield_gate = (
    yield_prefilter
    .reindex(tbsp_index, method="ffill")
    .fillna(1)
    .astype(int)
)

bond_entry_gate = (spread_gate & yield_gate).astype(int)
bond_entry_gate.name = "bond_entry_gate"

logging.info(
    "Bond entry gate built: %.1f%% of days pass  "
    "(spread: %.1f%% pass | yield: %.1f%% pass)",
    bond_entry_gate.mean() * 100,
    spread_gate.mean() * 100,
    yield_gate.mean() * 100,
)


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
    fast_mode=FAST_MODE,         # (or fast_mode=True in daily runfile)
    use_atr_stop          = USE_ATR_STOP,
    N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
    atr_window            = ATR_WINDOW,
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

# Per-asset report (single-asset, equity only)
BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}
if not wf_trades_eq.empty and "Exit Reason" in wf_trades_eq.columns:
    closed_eq = wf_trades_eq[~wf_trades_eq["Exit Reason"].isin(BOUNDARY_EXITS)].copy()
else:
    closed_eq = pd.DataFrame()

trade_stats_eq = analyze_trades(closed_eq) if not closed_eq.empty else None

print_backtest_report(
    metrics              = metrics_eq,
    trades               = wf_trades_eq,
    trade_stats          = trade_stats_eq,
    wf_results           = wf_results_eq,
    position_mode        = POSITION_MODE,
    filter_modes_override= FORCE_FILTER_MODE_EQ,
)

# Build daily binary signal from trade log
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

# --- Select bond signal source ---
if USE_YIELD_SIGNAL:
    logging.info("Bond signal source: PL10Y yield-derived price proxy")
    bond_signal_source = build_yield_price_proxy(PL10Y)

    # Wider grids — yield proxy has higher volatility than TBSP price index
    X_GRID_BD_RUN  = X_GRID_YLD
    SL_GRID_BD_RUN = SL_YLD
else:
    logging.info("Bond signal source: TBSP price index")
    bond_signal_source = TBSP

    X_GRID_BD_RUN  = X_GRID_BD
    SL_GRID_BD_RUN = SL_BD

    


# TBSP OOS window must start no earlier than equity OOS start
# (they share the same combined OOS period for portfolio simulation).
# walk_forward handles this naturally via TBSP's own data start date.

wf_equity_bd, wf_results_bd, wf_trades_bd = walk_forward(
    df                    = bond_signal_source,
    cash_df               = MMF,
    train_years           = TRAIN_YEARS_BD,
    test_years            = TEST_YEARS_BD,
    vol_window            = VOL_WINDOW,
    selected_mode         = POSITION_MODE,
    filter_modes_override = FORCE_FILTER_MODE_BD,   # ["ma"] always
    X_grid                = X_GRID_BD,
    Y_grid                = Y_GRID_BD,              # [0.001] = no breakout
    fast_grid             = FAST_BD,
    slow_grid             = SLOW_BD,
    tv_grid               = TV_BD,
    sl_grid               = SL_BD,
    mom_lookback_grid     = [252],                  # not used (filter_mode=ma)
    n_jobs                = N_JOBS,
    entry_gate_series     = bond_entry_gate,
    fast_mode=FAST_MODE,         # (or fast_mode=True in daily runfile)
    use_atr_stop          = USE_ATR_STOP_BD,
    N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
    atr_window            = ATR_WINDOW_BD,
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

trade_stats_bd = analyze_trades(closed_bd) if not closed_bd.empty else None

print_backtest_report(
    metrics              = metrics_bd,
    trades               = wf_trades_bd,
    trade_stats          = trade_stats_bd,
    wf_results           = wf_results_bd,
    position_mode        = POSITION_MODE,
    filter_modes_override= FORCE_FILTER_MODE_BD,
)

sig_bond = build_signal_series(wf_equity_bd, wf_trades_bd)
logging.info(
    "Bond signal: %.1f%% of OOS days in position (%d of %d)",
    sig_bond.mean() * 100, sig_bond.sum(), len(sig_bond)
)

# --- Post-walk_forward filter diagnostics (gate was applied inside walk_forward) ---
# sig_bond from build_signal_series already reflects the entry gate.
# Reconstruct the pre-gate counterfactual for reporting purposes only.

sig_bond_raw = build_signal_series(wf_equity_bd, wf_trades_bd)   # same as sig_bond
# (they are identical — the gate was applied inside wf, so the trade log
#  already reflects gated entries. sig_bond_raw here is kept for report compatibility.)

spread_prefilter_aligned = (
    spread_prefilter
    .reindex(sig_bond.index, method="ffill")
    .fillna(1)
    .astype(int)
)

yield_prefilter_aligned = (
    yield_prefilter
    .reindex(sig_bond.index, method="ffill")
    .fillna(1)
    .astype(int)
)

sig_bond_post_spread = sig_bond.copy()   # kept for report signature compatibility

logging.info(
    "Bond signal (gate-consistent): %.1f%% of OOS days in position  "
    "| spread filter ON %.1f%% | yield filter ON %.1f%%",
    sig_bond.mean() * 100,
    spread_prefilter_aligned.mean() * 100,
    yield_prefilter_aligned.mean() * 100,
)

# ============================================================
# PHASE 4 — ALLOCATION WALK-FORWARD  (both-signals-on weights)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: ALLOCATION WALK-FORWARD  (both-signals-on weight optimisation)")
logging.info("=" * 80)

# Align the common OOS period across both assets
oos_start_eq = wf_results_eq["TestStart"].min()
oos_start_bd = wf_results_bd["TestStart"].min()
oos_start    = max(oos_start_eq, oos_start_bd)   # must have both signals

oos_end_eq = wf_results_eq["TestEnd"].max()
oos_end_bd = wf_results_bd["TestEnd"].max()
oos_end    = min(oos_end_eq, oos_end_bd)

logging.info(
    "Common OOS period: %s to %s", oos_start.date(), oos_end.date()
)

# Trim signals and returns to common OOS period
def _trim(s, start, end):
    return s.loc[(s.index >= start) & (s.index <= end)]

sig_eq_oos  = _trim(sig_equity, oos_start, oos_end)
sig_bd_oos  = _trim(sig_bond,   oos_start, oos_end)
ret_eq_oos  = _trim(ret_eq,     oos_start, oos_end)
ret_bd_oos  = _trim(ret_bd,     oos_start, oos_end)
ret_mmf_oos = _trim(ret_mmf,    oos_start, oos_end)

portfolio_equity, weights_series, reallocation_log, alloc_results_df = (
    allocation_walk_forward(
        equity_returns   = ret_eq,          # full history — training uses this
        bond_returns     = ret_bd,          # full history — training uses this
        mmf_returns      = ret_mmf,         # full history — training uses this
        sig_equity_full  = sig_equity,      # full OOS signal — used for training slices
        sig_bond_full    = sig_bond,        # full OOS signal — used for training slices
        sig_equity_oos   = sig_eq_oos,      # trimmed to common period — used for test
        sig_bond_oos     = sig_bd_oos,      # trimmed to common period — used for test
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
    sig_bond_raw     = sig_bond_raw,
    spread_prefilter = spread_prefilter_aligned,
    yield_prefilter  = yield_prefilter,
    sig_bond_post_spread = sig_bond_post_spread 
)


# ============================================================
# PHASE 6 — DAILY OUTPUT (signal, log, chart, snapshot)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: DAILY OUTPUT")
logging.info("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


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
