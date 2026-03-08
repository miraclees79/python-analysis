"""
multiasset_runfile.py
=====================
Multi-asset pension fund strategy — WIG equity + TBSP bond + MMF.

Sequential optimisation in four phases:
  Phase 1 — Download WIG (equity proxy), TBSP (bond proxy), MMF data.
  Phase 2 — Walk-forward optimisation of the WIG signal.
  Phase 3 — Walk-forward optimisation of the TBSP signal (MA-only, no breakout).
  Phase 4 — Walk-forward optimisation of the both-signals-on allocation weights.
  Phase 5 — Combined portfolio simulation with reallocation gate + reporting.

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
TRAIN_YEARS_EQ = 8     # equity signal training window
TEST_YEARS_EQ  = 2     # equity signal test window
TRAIN_YEARS_BD = 8     # bond signal training window
TEST_YEARS_BD  = 2     # bond signal test window

# --- Volatility window (days) ---
VOL_WINDOW = 20

# --- Parallelism ---
_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

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


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOAD DATA")
logging.info("=" * 80)

tmp_dir = tempfile.mkdtemp()

# --- Equity: WIG total return index ---
csv_wig  = os.path.join(tmp_dir, "wig.csv")
download_csv("https://stooq.pl/q/d/l/?s=wig&i=d", csv_wig)
WIG = load_csv(csv_wig)
if WIG is None:
    logging.error("Failed to load WIG data. Exiting.")
    sys.exit(1)
logging.info("WIG loaded: %d rows  (%s to %s)",
             len(WIG), WIG.index.min().date(), WIG.index.max().date())

# --- Bond: TBSP Polish government bond total return index ---
csv_tbsp = os.path.join(tmp_dir, "tbsp.csv")
download_csv("https://stooq.pl/q/d/l/?s=tbsp&i=d", csv_tbsp)
TBSP = load_csv(csv_tbsp)
if TBSP is None:
    logging.error("Failed to load TBSP data. Exiting.")
    sys.exit(1)
logging.info("TBSP loaded: %d rows  (%s to %s)",
             len(TBSP), TBSP.index.min().date(), TBSP.index.max().date())

# --- MMF: Goldman Sachs Konserwatywny (money-market proxy, code 2720) ---
csv_mmf  = os.path.join(tmp_dir, "mmf.csv")
download_csv("https://stooq.pl/q/d/l/?s=2720.n&i=d", csv_mmf)
MMF = load_csv(csv_mmf)
if MMF is None:
    logging.error("Failed to load MMF data. Exiting.")
    sys.exit(1)
logging.info("MMF  loaded: %d rows  (%s to %s)",
             len(MMF), MMF.index.min().date(), MMF.index.max().date())

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

# TBSP OOS window must start no earlier than equity OOS start
# (they share the same combined OOS period for portfolio simulation).
# walk_forward handles this naturally via TBSP's own data start date.

wf_equity_bd, wf_results_bd, wf_trades_bd = walk_forward(
    df                    = TBSP,
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
        equity_returns  = ret_eq_oos,
        bond_returns    = ret_bd_oos,
        mmf_returns     = ret_mmf_oos,
        sig_equity_oos  = sig_eq_oos,
        sig_bond_oos    = sig_bd_oos,
        wf_results_eq   = wf_results_eq,
        wf_results_bd   = wf_results_bd,
        step            = 0.10,
        objective       = ALLOC_OBJECTIVE,
        cooldown_days   = COOLDOWN_DAYS,
        annual_cap      = ANNUAL_CAP,
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
)


# ============================================================
# PHASE 5 — PLOTS
# ============================================================

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

logging.info("=" * 80)
logging.info("MULTI-ASSET RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("=" * 80)
