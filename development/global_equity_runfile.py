"""
global_equity_runfile.py
========================
Global equity multi-market portfolio strategy.

Two portfolio modes, selected via PORTFOLIO_MODE in USER SETTINGS:

  "global_equity"  (Mode A — detailed foreign indices)
  -------------------------------------------------------
  Assets  : WIG (Polish broad market, 1991+), S&P 500, STOXX 600,
            Nikkei 225, TBSP
  Residual: MMF
  OOS start: ~2016 (TBSP-constrained), 9-year training window

  WIG is used as the single Polish equity proxy because its history
  back to 1991 allows 9 full training windows before TBSP becomes
  the common OOS constraint.  WIG20TR and the mid/small blend are
  too short (start 2005/2010) to be useful here.

  "msci_world"  (Mode B — MSCI World + two Polish equities)
  -----------------------------------------------------------
  Assets  : WIG20TR (Polish large cap), PL_MID (50:50 MWIG40TR+SWIG80TR),
            MSCI World combined (WSJ history + URTH extension), TBSP
  Residual: MMF
  OOS start: ~2019 (MSCI World + PL_MID constrained), 9-year training window

  MSCI World is read from the pre-built combined CSV on Google Drive
  (built once by wsj_msci_world.py).

PHASES
------
  Phase 1  — Download and validate all data
  Phase 2  — Per-asset signal walk-forward (one call per asset)
  Phase 3  — Bond signal walk-forward (TBSP, shared across modes)
  Phase 4  — Allocation walk-forward (N-asset, no annual cap)
  Phase 5  — Reporting
  Phase 6  — Plots
  [Phase 7/8 — robustness, to be added]

DESIGN NOTES
------------
- walk_forward (strategy_test_library) is called unchanged per asset.
- All equity signals use breakout + MA/momentum filter, same grids as
  existing WIG/WIG20TR optimisation.
- Bond signal (TBSP): MA-only, no breakout (Y_grid=[0.001]).
- FX treatment: controlled by FX_HEDGED flag.  Default True (hedged).
  When False, foreign asset returns are multiplied by the same-day
  PLN/foreign currency return.
- No annual reallocation cap (annual_cap=999).  Cooldown gate active.
- MMF absorbs residual allocation when signals are off or weights < 100%.

ADJUSTABLE SETTINGS — see USER SETTINGS block below.
"""

import os
import sys
import logging
import tempfile
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
from multiasset_library import (build_signal_series,)


from global_equity_library import (
    download_yfinance,
    build_return_series,
    build_blend,
    build_price_df_from_returns,
    allocation_walk_forward_n,
    print_global_equity_report,
    CLOSE_COL,
    DATA_START,
)

from wsj_msci_world import load_combined_from_drive


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE = "GlobalEquityRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logging.info("=" * 80)
logging.info("GLOBAL EQUITY RUNFILE START: %s", dt.datetime.now())
logging.info("=" * 80)


# ============================================================
# USER SETTINGS  <- edit these to switch modes and adjust behaviour
# ============================================================

# ---------------------------------------------------------------------------
# PORTFOLIO MODE  (the primary switch)
# ---------------------------------------------------------------------------
# "global_equity" : Mode A — WIG + S&P500 + STOXX600 + Nikkei + TBSP + MMF
#                   OOS start ~2016, 9+1 walk-forward, longest history
#
# "msci_world"    : Mode B — WIG20TR + PL_MID + MSCI_World + TBSP + MMF
#                   OOS start ~2019, 9+1 walk-forward, two Polish equities
PORTFOLIO_MODE = "msci_world"   # <- "global_equity" or "msci_world"

# ---------------------------------------------------------------------------
# GOOGLE DRIVE (required for msci_world mode; ignored in global_equity mode)
# ---------------------------------------------------------------------------
# Same folder ID used by all other strategy scripts.
# Env var GDRIVE_FOLDER_ID takes precedence over this hardcoded value.
GDRIVE_FOLDER_ID_DEFAULT = ""   # <- paste your folder ID here

# ---------------------------------------------------------------------------
# FX TREATMENT
# ---------------------------------------------------------------------------
# True  : funds are PLN-hedged — use local index returns directly
# False : funds are unhedged  — compound local returns with daily FX return
FX_HEDGED = True

# ---------------------------------------------------------------------------
# POSITION SIZING
# ---------------------------------------------------------------------------
POSITION_MODE = "full"   # "full" | "vol_entry" | "vol_dynamic"

# ---------------------------------------------------------------------------
# WALK-FORWARD WINDOWS
# ---------------------------------------------------------------------------
TRAIN_YEARS = 9    # training window length (years) — same for all assets
TEST_YEARS  = 1    # test window length (years)      — same for all assets
VOL_WINDOW  = 20   # rolling volatility window (days)

# ---------------------------------------------------------------------------
# REALLOCATION GATE
# ---------------------------------------------------------------------------
COOLDOWN_DAYS = 10    # minimum calendar days between reallocations
ANNUAL_CAP    = 999   # effectively disabled (no hard annual limit)

# ---------------------------------------------------------------------------
# ALLOCATION OBJECTIVE
# ---------------------------------------------------------------------------
ALLOC_OBJECTIVE = "calmar"   # "calmar" | "sharpe" | "cagr"

# ---------------------------------------------------------------------------
# ALLOCATION WEIGHT GRID STEP
# ---------------------------------------------------------------------------
ALLOC_STEP = 0.10   # 10% increments

# ---------------------------------------------------------------------------
# FILTER MODES
# ---------------------------------------------------------------------------
# None = automatic selection per window
# ["ma"] = force MA crossover filter only
# ["ma", "mom"] = allow MA or momentum
FORCE_FILTER_MODE_EQ = None    # equity signals
FORCE_FILTER_MODE_BD = ["ma"]  # bond: MA-only always

# ---------------------------------------------------------------------------
# EQUITY SIGNAL PARAMETER GRIDS  (shared across all equity assets)
# ---------------------------------------------------------------------------
X_GRID_EQ = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ   = [50, 75, 100]
SLOW_EQ   = [150, 200, 250]
TV_EQ     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ = [126, 252]

# ---------------------------------------------------------------------------
# BOND SIGNAL PARAMETER GRIDS  (TBSP, MA-only, no breakout)
# ---------------------------------------------------------------------------
X_GRID_BD = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD = [0.001]   # near-zero: effectively disables breakout entry condition
FAST_BD   = [50, 100, 150]
SLOW_BD   = [200, 300, 400, 500]
TV_BD     = [0.08, 0.10, 0.12]
SL_BD     = [0.01, 0.02, 0.03]

# ---------------------------------------------------------------------------
# PARALLELISM
# ---------------------------------------------------------------------------
_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

# ---------------------------------------------------------------------------
# MID/SMALL BLEND WEIGHTS  (Mode B only)
# ---------------------------------------------------------------------------
MWIG_WEIGHT = 0.50   # weight of MWIG40TR in the PL_MID blend
SWIG_WEIGHT = 0.50   # weight of SWIG80TR in the PL_MID blend


# ============================================================
# VALIDATE MODE SETTING
# ============================================================

_VALID_MODES = ("global_equity", "msci_world")
if PORTFOLIO_MODE not in _VALID_MODES:
    raise ValueError(
        f"PORTFOLIO_MODE must be one of {_VALID_MODES}, got '{PORTFOLIO_MODE}'"
    )

logging.info("=" * 80)
logging.info("PORTFOLIO MODE  : %s", PORTFOLIO_MODE)
logging.info("FX TREATMENT    : %s", "hedged (local returns)" if FX_HEDGED else "unhedged (FX-adjusted)")
logging.info("TRAIN / TEST    : %d + %d years", TRAIN_YEARS, TEST_YEARS)
logging.info("PARALLELISM     : %d logical cores, using %d jobs", _cpu_count, N_JOBS)
logging.info("=" * 80)


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DATA DOWNLOAD")
logging.info("=" * 80)

tmp_dir = tempfile.gettempdir()

def _stooq(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame | None:
    """Download a stooq series, clip to DATA_START, exit if mandatory and missing."""
    url  = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
    path = os.path.join(tmp_dir, f"ge_{label}.csv")
    ok   = download_csv(url, path)
    if not ok:
        if mandatory:
            logging.error("FAIL: %s (%s) — exiting.", label, ticker)
            sys.exit(1)
        logging.warning("WARN: %s (%s) — not available, skipping.", label, ticker)
        return None
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

# --- Shared across both modes ---
TBSP = _stooq("^tbsp",   "TBSP")
MMF  = _stooq("2720.n",  "MMF")

# --- FX rates (downloaded always; applied only when FX_HEDGED=False) ---
USDPLN = _stooq("usdpln", "USDPLN")
EURPLN = _stooq("eurpln", "EURPLN")
JPYPLN = _stooq("jpypln", "JPYPLN")

# --- Mode-specific downloads ---
if PORTFOLIO_MODE == "global_equity":
    logging.info("--- Mode A: global_equity ---")

    WIG     = _stooq("wig",   "WIG")       # Polish broad market, 1991+
    SPX     = _stooq("^spx",  "SP500")     # S&P 500
    NKX     = _stooq("nkx",   "Nikkei225")

    logging.info("Downloading STOXX 600 from yfinance ...")
    STOXX600 = download_yfinance("^STOXX", start=DATA_START)
    if STOXX600 is None:
        logging.error("FAIL: STOXX 600 — yfinance download returned None. Exiting.")
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "STOXX600(yf)", len(STOXX600),
        STOXX600.index.min().date(), STOXX600.index.max().date(),
    )

    # Mode A does not use these
    WIG20TR  = None
    MWIG40TR = None
    SWIG80TR = None
    MSCIW    = None

elif PORTFOLIO_MODE == "msci_world":
    logging.info("--- Mode B: msci_world ---")

    WIG20TR  = _stooq("wig20tr",  "WIG20TR")
    MWIG40TR = _stooq("mwig40tr", "MWIG40TR")
    SWIG80TR = _stooq("swig80tr", "SWIG80TR")

    # MSCI World combined series from Google Drive
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()
    if not folder_id:
        logging.error(
            "FAIL: GDRIVE_FOLDER_ID not set.\n"
            "  Set GDRIVE_FOLDER_ID_DEFAULT in USER SETTINGS or the env var.\n"
            "  Build the combined file first with: python wsj_msci_world.py"
        )
        sys.exit(1)
    MSCIW = load_combined_from_drive(folder_id=folder_id)
    if MSCIW is None:
        logging.error(
            "FAIL: msci_world_combined.csv not found in Drive folder %s.\n"
            "  Run wsj_msci_world.py to build it, then retry.",
            folder_id,
        )
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "MSCI_World", len(MSCIW),
        MSCIW.index.min().date(), MSCIW.index.max().date(),
    )

    # Mode B does not use these
    WIG      = None
    SPX      = None
    NKX      = None
    STOXX600 = None


# ============================================================
# PHASE 2 — RETURN SERIES CONSTRUCTION
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 2: RETURN SERIES CONSTRUCTION")
logging.info("=" * 80)

# --- FX series (close price, PLN per foreign unit) ---
fx_usd = USDPLN[CLOSE_COL] if USDPLN is not None else None
fx_eur = EURPLN[CLOSE_COL] if EURPLN is not None else None
fx_jpy = JPYPLN[CLOSE_COL] if JPYPLN is not None else None

# --- Shared ---
ret_tbsp = build_return_series(TBSP, hedged=True)   # PLN bond index — no FX
ret_mmf  = build_return_series(MMF,  hedged=True)   # PLN money market — no FX

logging.info(
    "TBSP: %d return days  %s to %s",
    len(ret_tbsp.dropna()), ret_tbsp.dropna().index.min().date(),
    ret_tbsp.dropna().index.max().date(),
)
logging.info(
    "MMF : %d return days  %s to %s",
    len(ret_mmf.dropna()), ret_mmf.dropna().index.min().date(),
    ret_mmf.dropna().index.max().date(),
)

# --- Mode-specific returns ---
if PORTFOLIO_MODE == "global_equity":
    ret_WIG     = build_return_series(WIG,     hedged=True)   # PLN — no FX
    ret_SPX     = build_return_series(SPX,     fx_series=fx_usd, hedged=FX_HEDGED)
    ret_STOXX   = build_return_series(STOXX600, fx_series=fx_eur, hedged=FX_HEDGED)
    ret_NKX     = build_return_series(NKX,     fx_series=fx_jpy, hedged=FX_HEDGED)

    for label, ret in [
        ("WIG",      ret_WIG),
        ("SP500",    ret_SPX),
        ("STOXX600", ret_STOXX),
        ("Nikkei225",ret_NKX),
    ]:
        r = ret.dropna()
        logging.info(
            "%-12s  %5d days  %s to %s  mean=%.4f%%  ann_vol=%.2f%%",
            label, len(r), r.index.min().date(), r.index.max().date(),
            r.mean() * 100, r.std() * (252 ** 0.5) * 100,
        )

    # Build price DataFrames for walk_forward from return series
    # (WIG, SPX, NKX are already in stooq format from load_csv;
    #  STOXX600 from yfinance is already stooq-format via download_yfinance)
    # For FX_HEDGED=False, returns differ from raw prices — rebuild price df
    if not FX_HEDGED:
        logging.info("FX_HEDGED=False: rebuilding price DFs from PLN return series.")
        price_WIG     = build_price_df_from_returns(ret_WIG,   "WIG_PLN")
        price_SPX     = build_price_df_from_returns(ret_SPX,   "SP500_PLN")
        price_STOXX   = build_price_df_from_returns(ret_STOXX, "STOXX600_PLN")
        price_NKX     = build_price_df_from_returns(ret_NKX,   "Nikkei225_PLN")
    else:
        price_WIG     = WIG
        price_SPX     = SPX
        price_STOXX   = STOXX600
        price_NKX     = NKX

elif PORTFOLIO_MODE == "msci_world":
    # PL_MID blend: returns in PLN (both components are PLN-native)
    PL_MID = build_blend(MWIG40TR, SWIG80TR, MWIG_WEIGHT, SWIG_WEIGHT, "PL_MID")
    ret_WIG20TR = build_return_series(WIG20TR, hedged=True)
    ret_PL_MID  = build_return_series(PL_MID,  hedged=True)
    ret_MSCIW   = build_return_series(MSCIW,   fx_series=fx_usd, hedged=FX_HEDGED)

    for label, ret in [
        ("WIG20TR",    ret_WIG20TR),
        ("PL_MID",     ret_PL_MID),
        ("MSCI_World", ret_MSCIW),
    ]:
        r = ret.dropna()
        logging.info(
            "%-12s  %5d days  %s to %s  mean=%.4f%%  ann_vol=%.2f%%",
            label, len(r), r.index.min().date(), r.index.max().date(),
            r.mean() * 100, r.std() * (252 ** 0.5) * 100,
        )

    if not FX_HEDGED:
        logging.info("FX_HEDGED=False: rebuilding price DFs from PLN return series.")
        price_WIG20TR = build_price_df_from_returns(ret_WIG20TR, "WIG20TR_PLN")
        price_PL_MID  = build_price_df_from_returns(ret_PL_MID,  "PL_MID_PLN")
        price_MSCIW   = build_price_df_from_returns(ret_MSCIW,   "MSCIW_PLN")
    else:
        price_WIG20TR = WIG20TR
        price_PL_MID  = PL_MID
        price_MSCIW   = MSCIW


# ============================================================
# PHASE 3 — PER-ASSET EQUITY SIGNAL WALK-FORWARD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 3: PER-ASSET EQUITY SIGNAL WALK-FORWARD")
logging.info("=" * 80)

def _run_equity_wf(price_df: pd.DataFrame, label: str):
    """Run walk_forward for one equity asset using shared equity grids."""
    logging.info("  [%s] walk-forward starting ...", label)
    wf_eq, wf_res, wf_tr = walk_forward(
        df                    = price_df,
        cash_df               = MMF,
        train_years           = TRAIN_YEARS,
        test_years            = TEST_YEARS,
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
        objective             = "calmar",
        n_jobs                = N_JOBS,
    )
    m = compute_metrics(wf_eq)
    logging.info(
        "  [%s] OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
        label,
        m.get("CAGR", 0) * 100, m.get("Sharpe", 0),
        m.get("MaxDD", 0) * 100, m.get("CalMAR", 0),
    )
    sig = build_signal_series(wf_eq, wf_tr)
    logging.info(
        "  [%s] signal: %.1f%% of OOS days in position",
        label, sig.mean() * 100,
    )
    return wf_eq, wf_res, wf_tr, sig


if PORTFOLIO_MODE == "global_equity":
    wf_eq_WIG,    wf_res_WIG,    wf_tr_WIG,    sig_WIG    = _run_equity_wf(price_WIG,   "WIG")
    wf_eq_SPX,    wf_res_SPX,    wf_tr_SPX,    sig_SPX    = _run_equity_wf(price_SPX,   "SP500")
    wf_eq_STOXX,  wf_res_STOXX,  wf_tr_STOXX,  sig_STOXX  = _run_equity_wf(price_STOXX, "STOXX600")
    wf_eq_NKX,    wf_res_NKX,    wf_tr_NKX,    sig_NKX    = _run_equity_wf(price_NKX,   "Nikkei225")

elif PORTFOLIO_MODE == "msci_world":
    wf_eq_WIG20,  wf_res_WIG20,  wf_tr_WIG20,  sig_WIG20  = _run_equity_wf(price_WIG20TR, "WIG20TR")
    wf_eq_PLMID,  wf_res_PLMID,  wf_tr_PLMID,  sig_PLMID  = _run_equity_wf(price_PL_MID,  "PL_MID")
    wf_eq_MSCIW,  wf_res_MSCIW,  wf_tr_MSCIW,  sig_MSCIW  = _run_equity_wf(price_MSCIW,   "MSCI_World")


# ============================================================
# PHASE 4 — BOND SIGNAL WALK-FORWARD (TBSP)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: BOND SIGNAL WALK-FORWARD  (TBSP, train=%dy test=%dy)",
             TRAIN_YEARS, TEST_YEARS)
logging.info("=" * 80)

wf_eq_TBSP, wf_res_TBSP, wf_tr_TBSP = walk_forward(
    df                    = TBSP,
    cash_df               = MMF,
    train_years           = TRAIN_YEARS,
    test_years            = TEST_YEARS,
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
    objective             = "calmar",
    n_jobs                = N_JOBS,
)

m_tbsp = compute_metrics(wf_eq_TBSP)
logging.info(
    "TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
    m_tbsp.get("CAGR", 0) * 100, m_tbsp.get("Sharpe", 0),
    m_tbsp.get("MaxDD", 0) * 100, m_tbsp.get("CalMAR", 0),
)

sig_TBSP = build_signal_series(wf_eq_TBSP, wf_tr_TBSP)
logging.info(
    "TBSP signal: %.1f%% of OOS days in position",
    sig_TBSP.mean() * 100,
)


# ============================================================
# PHASE 5 — ALLOCATION WALK-FORWARD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: ALLOCATION WALK-FORWARD  (N-asset, annual_cap=%d)", ANNUAL_CAP)
logging.info("=" * 80)

# --- Assemble asset dicts based on mode ---
# The allocation walk-forward uses TBSP's window schedule as the reference
# (TBSP is the most common binding constraint on OOS start across both modes).

if PORTFOLIO_MODE == "global_equity":
    asset_keys = ["WIG", "SP500", "STOXX600", "Nikkei225", "TBSP"]

    returns_dict = {
        "WIG":       ret_WIG.dropna(),
        "SP500":     ret_SPX.dropna(),
        "STOXX600":  ret_STOXX.dropna(),
        "Nikkei225": ret_NKX.dropna(),
        "TBSP":      ret_tbsp.dropna(),
    }
    signals_full_dict = {
        "WIG":       sig_WIG,
        "SP500":     sig_SPX,
        "STOXX600":  sig_STOXX,
        "Nikkei225": sig_NKX,
        "TBSP":      sig_TBSP,
    }
    # OOS signals: trimmed to the common OOS period (computed inside allocation_wf)
    signals_oos_dict = dict(signals_full_dict)   # same object; trimming done inside

    # Reference walk-forward results for window schedule = TBSP
    wf_results_ref = wf_res_TBSP

elif PORTFOLIO_MODE == "msci_world":
    asset_keys = ["WIG20TR", "PL_MID", "MSCI_World", "TBSP"]

    returns_dict = {
        "WIG20TR":    ret_WIG20TR.dropna(),
        "PL_MID":     ret_PL_MID.dropna(),
        "MSCI_World": ret_MSCIW.dropna(),
        "TBSP":       ret_tbsp.dropna(),
    }
    signals_full_dict = {
        "WIG20TR":    sig_WIG20,
        "PL_MID":     sig_PLMID,
        "MSCI_World": sig_MSCIW,
        "TBSP":       sig_TBSP,
    }
    signals_oos_dict = dict(signals_full_dict)

    # Reference walk-forward results for window schedule:
    # Use the TBSP walk-forward, which produces the latest OOS start
    # (the common OOS constraint is MSCI_World start ~2019, but
    #  allocation_walk_forward_n clips to the intersection internally).
    wf_results_ref = wf_res_TBSP

portfolio_equity, weights_series, reallocation_log, alloc_results_df = (
    allocation_walk_forward_n(
        returns_dict       = returns_dict,
        signals_full_dict  = signals_full_dict,
        signals_oos_dict   = signals_oos_dict,
        mmf_returns        = ret_mmf.dropna(),
        wf_results_ref     = wf_results_ref,
        asset_keys         = asset_keys,
        step               = ALLOC_STEP,
        objective          = ALLOC_OBJECTIVE,
        cooldown_days      = COOLDOWN_DAYS,
        annual_cap         = ANNUAL_CAP,
        train_years        = TRAIN_YEARS,
    )
)

if portfolio_equity is None or portfolio_equity.empty:
    logging.error("Portfolio equity curve is empty after allocation walk-forward. Exiting.")
    sys.exit(1)

portfolio_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}
logging.info(
    "PORTFOLIO OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
    portfolio_metrics.get("CAGR", 0) * 100,
    portfolio_metrics.get("Sharpe", 0),
    portfolio_metrics.get("MaxDD", 0) * 100,
    portfolio_metrics.get("CalMAR", 0),
)


# ============================================================
# PHASE 6 — REPORTING
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: REPORTING")
logging.info("=" * 80)

oos_start = portfolio_equity.index.min()
oos_end   = portfolio_equity.index.max()

# Build B&H metrics for each asset over the common OOS period
bh_metrics_dict = {}
for key, ret in returns_dict.items():
    ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
    if ret_oos.empty:
        bh_metrics_dict[key] = {}
        continue
    bh_equity = (1 + ret_oos).cumprod()
    bh_equity = bh_equity / bh_equity.iloc[0]
    bh_metrics_dict[key] = {k: float(v) for k, v in compute_metrics(bh_equity).items()}

# Trim OOS signals to common portfolio period
signals_oos_trimmed = {
    k: s.loc[(s.index >= oos_start) & (s.index <= oos_end)]
    for k, s in signals_oos_dict.items()
}

print_global_equity_report(
    portfolio_metrics  = portfolio_metrics,
    bh_metrics_dict    = bh_metrics_dict,
    alloc_results_df   = alloc_results_df,
    reallocation_log   = reallocation_log,
    signals_oos_dict   = signals_oos_trimmed,
    oos_start          = oos_start,
    oos_end            = oos_end,
    portfolio_mode     = PORTFOLIO_MODE,
    fx_hedged          = FX_HEDGED,
)


# ============================================================
# PHASE 7 — CHARTS
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 7: CHARTS")
logging.info("=" * 80)

CHART_PATH = f"global_equity_{PORTFOLIO_MODE}_chart.png"

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(
    f"Global Equity Portfolio  [{PORTFOLIO_MODE}]  "
    f"{'Hedged' if FX_HEDGED else 'Unhedged'}  "
    f"{oos_start.date()} to {oos_end.date()}",
    fontsize=12,
)

# Panel 1 — portfolio equity vs individual asset B&H
ax1 = axes[0]
ax1.set_title("Portfolio equity (OOS) vs buy-and-hold per asset")
ax1.set_yscale("log")

port_norm = portfolio_equity / portfolio_equity.iloc[0] * 100
ax1.plot(port_norm.index, port_norm.values, color="black", linewidth=2.0,
         label="Portfolio", zorder=10)

colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
for i, (key, ret) in enumerate(returns_dict.items()):
    ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
    if ret_oos.empty:
        continue
    bh = (1 + ret_oos).cumprod() * 100
    ax1.plot(bh.index, bh.values, color=colors[i % len(colors)],
             linewidth=0.9, alpha=0.7, label=key)

ax1.legend(fontsize=8, ncol=3)
ax1.set_ylabel("Cumulative return (log, base=100)")
ax1.grid(True, alpha=0.3)

# Panel 2 — per-asset signal occupancy over time (stacked binary)
ax2 = axes[1]
ax2.set_title("Per-asset signal state (1=in position)")

y_offset = 0
yticks, ylabels = [], []
for key, sig in signals_oos_trimmed.items():
    if sig.empty:
        continue
    ax2.fill_between(sig.index, y_offset, y_offset + sig.values * 0.8,
                     alpha=0.6, label=key, step="post")
    yticks.append(y_offset + 0.4)
    ylabels.append(key)
    y_offset += 1.0

ax2.set_yticks(yticks)
ax2.set_yticklabels(ylabels, fontsize=8)
ax2.set_ylabel("Signal (in=shaded)")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=120, bbox_inches="tight")
plt.close()
logging.info("Chart saved: %s", CHART_PATH)

logging.info("=" * 80)
logging.info("GLOBAL EQUITY RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("Mode: %s  |  FX: %s", PORTFOLIO_MODE, "hedged" if FX_HEDGED else "unhedged")
logging.info("OOS: %s to %s  (%d reallocations)",
             oos_start.date(), oos_end.date(), len(reallocation_log))
logging.info("=" * 80)
