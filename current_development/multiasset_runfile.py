"""
multiasset_runfile.py
=====================
Multi-asset pension fund strategy — WIG equity + TBSP bond + MMF.

Sequential optimisation in four phases:
  Phase 1 — Download WIG (equity proxy), TBSP (bond proxy), MMF data.
  Phase 2 — Walk-forward optimisation of the WIG signal.
  Phase 3 — Walk-forward optimisation of the TBSP signal (MA-only, no breakout).
  Phase 4 — Walk-forward optimisation of the both-signals-on allocation weights
            followed by combined portfolio simulation with reallocation gate 
  Phase 5 -  reporting
  Phase 6 - plots
  Phase 7 - single asset strategy robustness tests
  Phase 8 - multi asset allocation robustness test

Each phase reuses machinery from strategy_test_library unchanged.
Phase 4–5 use the new multiasset_library.

DESIGN NOTES
------------
- WIG signal : breakout + MA/momentum filter — same framework as WIG20TR,
               re-optimised on the WIG index (longer history since 1991).
- TBSP signal: MA crossover only.  Breakout disabled by setting Y_grid=[0.001]
               (price is always above a near-zero trough threshold).
               No breakout means the entry condition collapses to: filter ON.
               Exit: filter turns off OR absolute stop triggered.
               Trailing stop X kept in grid — it still provides a drawdown cap.
- MMF        : money-market fund (code 2720.n on stooq) used as:
                 (a) cash return when signals are off
                 (b) residual allocation when signals are on but < 100% invested
- Both-on weights: grid-searched each training window; 45 combinations at 10%
                   steps.  MMF absorbs the remainder (equity + bond <= 100%).
- Reallocation gate: 10-day cooldown (primary) + annual cap of 12 (safety).
- OOS start constrained by TBSP history (available from end-2006 on stooq).
  Practical OOS start with 8+2 windows: end-2014.

ADJUSTABLE SETTINGS — see "USER SETTINGS" block below.
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
    allocation_weight_robustness,
    print_allocation_robustness_report,
)
from multiasset_daily_output import build_daily_outputs
from global_equity_library import build_mmf_extended
from price_series_builder import build_and_upload
from stooq_hybrid_updater import run_update

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
    EQUITY_THRESHOLDS_BOOTSTRAP,)

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

RUN_BLOCK_BOOTSTRAP_SINGLE = True # Run bootstrap robustness test for single asset strategies
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
# PHASE 6 — PLOTS
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: PLOTTING")
logging.info("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

# --- Equity curves ---
ax = axes[0]
portfolio_equity.plot(ax=ax, label="Portfolio",      color="steelblue",  linewidth=2)
bh_equity_eq.plot(   ax=ax, label="B&H WIG",        color="darkorange",  linewidth=1, linestyle="--")
bh_equity_bd.plot(   ax=ax, label="B&H TBSP",       color="seagreen",    linewidth=1, linestyle="--")
ax.set_title("Portfolio vs Buy & Hold  (OOS equity curves, normalised to 1.0)")
ax.set_ylabel("Equity (normalised)")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Drawdown ---
ax = axes[1]
dd_port = portfolio_equity / portfolio_equity.cummax() - 1
dd_eq   = bh_equity_eq    / bh_equity_eq.cummax()    - 1
dd_bd   = bh_equity_bd    / bh_equity_bd.cummax()    - 1
dd_port.plot(ax=ax, label="Portfolio",  color="steelblue",  linewidth=1.5)
dd_eq.plot(  ax=ax, label="B&H WIG",   color="darkorange",  linewidth=1, linestyle="--", alpha=0.7)
dd_bd.plot(  ax=ax, label="B&H TBSP",  color="seagreen",    linewidth=1, linestyle="--", alpha=0.7)
ax.set_title("Drawdown")
ax.set_ylabel("Drawdown")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Signal and allocation state ---
ax = axes[2]
sig_eq_oos.plot(ax=ax, label="Equity signal", color="darkorange", linewidth=0.8, alpha=0.7)
sig_bd_oos.plot(ax=ax, label="Bond signal",   color="seagreen",   linewidth=0.8, alpha=0.7)

# Mark reallocation dates
if reallocation_log:
    realloc_dates = [r["Date"] for r in reallocation_log]
    ax.vlines(realloc_dates, ymin=0, ymax=1.05,
              color="grey", linewidth=0.5, alpha=0.5, label="Reallocation")

ax.set_title("Asset Signals and Reallocation Events")
ax.set_ylabel("Signal (0=off, 1=on)")
ax.set_ylim(-0.05, 1.15)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "multiasset_equity.png"
plt.savefig(plot_path, dpi=150)
plt.close()
logging.info("Plot saved to %s", plot_path)




# ============================================================
# PHASE 7 — LEVEL 1 ROBUSTNESS: Monte Carlo perturbation of paramenters in single asset strategies
# ============================================================
logging.info("=" * 80)
logging.info("PHASE 7: SINGLE ASSET STRATEGIES ROBUSTNESS L1 (parameters perturbation AND/OR bootstrap history)")
logging.info("=" * 80)


if RUN_MONTE_CARLO_PARAM_SINGLE:
    logging.info("=" * 80)
    logging.info("Monte Carlo robustness check calculation for single asset strategies")
    logging.info("=" * 80)
    
 

    
    logging.info("=" * 80)
    logging.info("Equity component - WIG")
    logging.info("=" * 80)

    # Extract from existing wf_results DataFrame
    windows_eq     = extract_windows_from_wf_results(wf_results_eq, train_years=8)
    best_params_eq = extract_best_params_from_wf_results(wf_results_eq)

    # Run
    mc_results_eq = run_monte_carlo_robustness(
        best_params   = best_params_eq,
        windows       = windows_eq,
        df            = WIG,
        cash_df       = MMF,
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,        # or FUNDS if fund filter enabled
        n_samples     = ITERATIONS_MC_PARAM_SINGLE,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie"
    )

    # Report
    baseline_eq = compute_metrics(wf_equity_eq)
    analyze_robustness(mc_results_eq, baseline_eq,thresholds=EQUITY_THRESHOLDS_MC)


    logging.info("=" * 80)
    logging.info("Bond component - TBSP")
    logging.info("=" * 80)

    # Extract from existing wf_results DataFrame
    windows_bd     = extract_windows_from_wf_results(wf_results_bd, train_years=8)
    best_params_bd = extract_best_params_from_wf_results(wf_results_bd)

    # Run
    mc_results_bd= run_monte_carlo_robustness(
        best_params   = best_params_bd,
        windows       = windows_bd,
        df            = TBSP,
        cash_df       = MMF,
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,        # or FUNDS if fund filter enabled
        n_samples     = ITERATIONS_MC_PARAM_SINGLE,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie"
    )

    # Report
    baseline_bd = compute_metrics(wf_equity_bd)
    analyze_robustness(mc_results_bd, baseline_bd,thresholds=BOND_THRESHOLDS_MC)

else:
    logging.info("=" * 80)
    logging.info("Monte Carlo robustness check calculation for single asset strategies skipped by user choice")
    logging.info("=" * 80)




# -------------------------------------------------------
# PHASE 7 LEVEL 1 Part 2 - Block Bootstrap Robustness Check for single asset strategies
# -------------------------------------------------------
if RUN_BLOCK_BOOTSTRAP_SINGLE:
    logging.info("=" * 80)
    logging.info("Block bootstrap robustness check for single asset strategies")
    logging.info("=" * 80)

    logging.info("=" * 80)
    logging.info("Equity component - WIG")
    logging.info("=" * 80)


    bb_results_eq = run_block_bootstrap_robustness(
        df               = WIG,
        cash_df          = MMF,
        price_col        = "Zamkniecie",
        cash_price_col   = "Zamkniecie",
        n_samples        = ITERATIONS_BOOTSTRAP_SINGLE,
        block_size       = 250,
        # --- wf_kwargs: mirrors the walk_forward call above ---
        train_years              = TRAIN_YEARS_EQ,
        test_years               = TEST_YEARS_EQ,
        vol_window               = VOL_WINDOW,
        funds_df                 = None,
        fund_params_grid         = None,
        selected_mode            = POSITION_MODE,
        filter_modes_override    = FORCE_FILTER_MODE_EQ,
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

    baseline_eq = compute_metrics(wf_equity_eq)
    analyze_bootstrap(bb_results_eq, baseline_eq, thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

    logging.info("=" * 80)
    logging.info("Bond component - TBSP")
    logging.info("=" * 80)


    bb_results_bd = run_block_bootstrap_robustness(
        df               = TBSP,
        cash_df          = MMF,
        price_col        = "Zamkniecie",
        cash_price_col   = "Zamkniecie",
        n_samples        = ITERATIONS_BOOTSTRAP_SINGLE,
        block_size       = 250,
        # --- wf_kwargs: mirrors the walk_forward call above ---
        train_years              = TRAIN_YEARS_BD,
        test_years               = TEST_YEARS_BD,
        vol_window               = VOL_WINDOW,
        funds_df                 = None,
        fund_params_grid         = None,
        selected_mode            = POSITION_MODE,
        filter_modes_override    = FORCE_FILTER_MODE_BD,
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
        )

    baseline_bd = compute_metrics(wf_equity_bd)
    analyze_bootstrap(bb_results_bd, baseline_bd,thresholds=BOND_THRESHOLDS_BOOTSTRAP)

else:
    logging.info("=" * 80)
    logging.info("Block bootstrap robustness check skipped by user choice")
    logging.info("=" * 80)






# ============================================================
# PHASE 8 — LEVEL 2 ROBUSTNESS: ALLOCATION WEIGHT PERTURBATION
# ============================================================
logging.info("=" * 80)
logging.info("PHASE 8: ALLOCATION WEIGHT ROBUSTNESS  (Level 2 — equity weight perturbation)")
logging.info("=" * 80)


if RUN_ROBUSTNESS_ALLOCATION_WEIGHT:

    robustness_df = allocation_weight_robustness(
        alloc_results_df = alloc_results_df,
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        baseline_metrics = portfolio_metrics,
        perturb_steps    = [-0.2, -0.1, 0.0, 0.1, 0.2],
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )

    print_allocation_robustness_report(robustness_df)

else:
    logging.info("=" * 80)
    logging.info("Allocation weight robustness check - equity weight perturbation skipped by user choice")
    logging.info("=" * 80)


logging.info("=" * 80)
logging.info("MULTI-ASSET RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("=" * 80)
