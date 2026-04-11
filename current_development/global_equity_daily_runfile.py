"""
global_equity_daily_runfile.py
================================
Daily runner for the global equity portfolio strategy.

Runs Phases 1–7 (data download, per-asset walk-forward, bond walk-forward,
allocation walk-forward, reporting, daily output).  Robustness phases are
disabled to keep the GitHub Actions run within the 6-hour timeout.

After Phase 7, four daily artefacts are written to OUTPUT_DIR and then
uploaded to Google Drive by the workflow:
  global_equity_signal_status.txt
  global_equity_signal_log.csv
  global_equity_chart.png
  global_equity_signal_snapshot.json

Based on global_equity_runfile.py.
All parameter settings are identical — edit USER SETTINGS here to match.
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
    load_stooq_local,
    load_csv,
    walk_forward,
    compute_metrics,
    compute_buy_and_hold,
    analyze_trades,
    print_backtest_report,
    
)
from multiasset_library import build_signal_series

from global_equity_library import (
    download_yfinance,
    build_return_series,
    build_blend,
    build_price_df_from_returns,
    build_mmf_extended,
    allocation_walk_forward_n,
    print_global_equity_report,
    CLOSE_COL,
    DATA_START,
)

from price_series_builder import build_and_upload, load_combined_from_drive
from global_equity_daily_output import build_daily_outputs
from msci_world_synthetic import (
    build_full_msci_world_extended,
    load_synthetic_from_drive,
)
from stooq_hybrid_updater import run_update
from global_equity_daily_output import build_daily_outputs


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
TRAIN_YEARS = 7    # training window length (years) — same for all assets
TEST_YEARS  = 2    # test window length (years)      — same for all assets
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


USE_ATR_STOP    = True          # Equity trailing stop mode
ATR_WINDOW      = 20             # Rolling window for ATR estimate (days)
N_ATR_GRID      = [0.08, 0.10, 0.12, 0.15, 0.20]   # Normalised ATR grid (same scale as X_GRID_EQ)

USE_ATR_STOP_BD = False          # Bond trailing stop mode (can differ from equity)
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [0.05, 0.08, 0.10, 0.15]          # Normalised ATR grid (same scale as X_GRID_BD)

FAST_MODE = True

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

# ---------------------------------------------------------------------------
# DATA FLOOR DATES
# ---------------------------------------------------------------------------
# WIG (Mode A only): daily continuous trading started 1994-10-03.
#   Earlier data has multi-day gaps that distort the breakout trough
#   calculation and MA windows.  Clipped after download in Phase 2.
WIG_DATA_FLOOR = "1994-10-03"

# MMF extension: chain-link WIBOR 1M backwards from first MMF NAV to this
#   date. Ensures IS windows starting before 1999 have realistic cash returns
#   rather than ret_mmf=0 on signal-off days. Applied in Phase 2.
MMF_FLOOR = "1994-10-03"

# ---------------------------------------------------------------------------
# DAILY OUTPUT
# ---------------------------------------------------------------------------
OUTPUT_DIR = "global_equity_outputs"   # local directory for daily artefacts


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

run_update(get_funds=False) # load data from GDrive to /data subfolder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# TBSP backcast (stooq ^tbsp extension, generic Date/Close format)
TBSP = build_and_upload(
      folder_id          = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()
          or GDRIVE_FOLDER_ID_DEFAULT.strip()),
      raw_filename       = "tbsp_extended_full.csv",
      combined_filename  = "tbsp_extended_combined.csv",
      extension_ticker   = "tbsp",
      extension_source   = "stooq",
  )

MMF  = load_stooq_local("fund_2720",  "MMF", DATA_DIR, DATA_START)

# WIBOR 1M — used to extend MMF backwards to MMF_FLOOR
# Not mandatory: if unavailable the MMF runs from its natural start (~1999)
WIBOR1M = load_stooq_local("wibor1m", "WIBOR1M", DATA_DIR, DATA_START, mandatory=False)
if WIBOR1M is None:
    logging.warning(
        "WIBOR1M (plopln1m) unavailable — MMF will not be extended. "
        "IS windows before 1999 will use ret_mmf=0 for cash returns."
    )

# --- Extended MMF (both modes) ---
# Chain-link WIBOR 1M backwards from first real MMF observation to MMF_FLOOR.
# This gives realistic ~18-25% p.a. cash returns for 1994-1999 IS windows
# in Mode A (global_equity), where WIG IS data goes back to 1994.
if WIBOR1M is not None:
    MMF_EXT = build_mmf_extended(MMF, WIBOR1M, floor_date=MMF_FLOOR)
    logging.info(
        "MMF extended: %s to %s (%d rows total; original MMF from %s)",
        MMF_EXT.index.min().date(), MMF_EXT.index.max().date(),
        len(MMF_EXT), MMF.index.min().date(),
    )
else:
    MMF_EXT = MMF   # no extension available; falls back to original series

# --- FX rates (downloaded always; applied only when FX_HEDGED=False) ---
USDPLN = load_stooq_local("usdpln", "USDPLN", DATA_DIR, DATA_START)
EURPLN = load_stooq_local("eurpln", "EURPLN", DATA_DIR, DATA_START)
JPYPLN = load_stooq_local("jpypln", "JPYPLN", DATA_DIR, DATA_START)

# --- Mode-specific downloads ---
if PORTFOLIO_MODE == "global_equity":
    logging.info("--- Mode A: global_equity ---")

    WIG   = load_stooq_local("wig",       "WIG",     DATA_DIR, DATA_START)
    WIG   = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    SPX     = load_stooq_local("sp500",  "SP500", DATA_DIR, DATA_START)     # S&P 500
    NKX     = load_stooq_local("nk225",   "Nikkei225", DATA_DIR, DATA_START)

    # STOXX 600: load from Drive (historical base + yfinance extension).
    # build_stoxx600() auto-selects first-build vs update mode:
    #   First build: reads stoxx600.csv from Drive, extends with yfinance
    #   Update:      reads stoxx600_combined.csv from Drive, extends forward
    _stoxx_folder = (
        os.environ.get("GDRIVE_FOLDER_ID", "").strip()
        or GDRIVE_FOLDER_ID_DEFAULT.strip()
    )
    if not _stoxx_folder:
        logging.error(
            "FAIL: GDRIVE_FOLDER_ID not set — cannot load STOXX 600. "
            "Set GDRIVE_FOLDER_ID_DEFAULT or the env var."
        )
        sys.exit(1)
    # STOXX 600 (yfinance ^STOXX extension, generic Date/Close format)
    STOXX600 = build_and_upload(
        folder_id          = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()  or GDRIVE_FOLDER_ID_DEFAULT.strip()),
        raw_filename       = "stoxx600.csv",
        combined_filename  = "stoxx600_combined.csv",
        extension_ticker   = "^STOXX",
    )
    if STOXX600 is None:
        logging.error(
            "FAIL: STOXX 600 build/update failed. "
            "Ensure stoxx600.csv is uploaded to Drive folder %s.",
            _stoxx_folder,
        )
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "STOXX600", len(STOXX600),
        STOXX600.index.min().date(), STOXX600.index.max().date(),
    )

    # Mode A does not use these
    WIG20TR  = None
    MWIG40TR = None
    SWIG80TR = None
    MSCIW    = None

elif PORTFOLIO_MODE == "msci_world":
    logging.info("--- Mode B: msci_world ---")

    WIG     = load_stooq_local("wig",       "WIG",     DATA_DIR, DATA_START)       # Polish broad market, 1991+WIG   = load_stooq_local("wig",       "WIG",     DATA_DIR, DATA_START)
    WIG   = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    
    # MSCI World combined series from Google Drive
    MSCIW_wsj = build_and_upload(
        folder_id         = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()  
                             or GDRIVE_FOLDER_ID_DEFAULT.strip()),
        raw_filename      = "msci_world_wsj_raw.csv",
        combined_filename = "msci_world_combined.csv",
        extension_ticker  = "URTH",
        )
    if MSCIW_wsj is None:
        logging.error("FAIL: MSCI World (WSJ+URTH) build failed — exiting.")
        sys.exit(1)

    # Prepend synthetic backcast (1990–2009) from Drive
    # If msci_world_synthetic.csv is not on Drive yet, MSCIW_wsj is used as-is
    # and Mode B OOS start remains ~2019. Run msci_world_synthetic.build_and_upload_synthetic()
    # once to generate it.
    MSCIW = build_full_msci_world_extended(
        wsj_combined_df = MSCIW_wsj,
        folder_id       = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()  
                           or GDRIVE_FOLDER_ID_DEFAULT.strip()),
        )
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "MSCI_World", len(MSCIW),
        MSCIW.index.min().date(), MSCIW.index.max().date(),
        )

    # Mode B does not use these
    
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

# --- WIG floor (Mode A only) ---
# Clip WIG to the daily-continuous-trading era. Earlier data has multi-day
# gaps that distort the breakout trough and MA signal calculations.
if PORTFOLIO_MODE == "global_equity" and WIG is not None:
    _wig_before = len(WIG)
    WIG = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    logging.info(
        "WIG clipped to %s: %d → %d rows",
        WIG_DATA_FLOOR, _wig_before, len(WIG),
    )

# --- Extended MMF (both modes) ---
# Chain-link WIBOR 1M backwards from first real MMF observation to MMF_FLOOR.
# This gives realistic ~18-25% p.a. cash returns for 1994-1999 IS windows
# in Mode A (global_equity), where WIG IS data goes back to 1994.
if WIBOR1M is not None:
    MMF_EXT = build_mmf_extended(MMF, WIBOR1M, floor_date=MMF_FLOOR)
    logging.info(
        "MMF extended: %s to %s (%d rows total; original MMF from %s)",
        MMF_EXT.index.min().date(), MMF_EXT.index.max().date(),
        len(MMF_EXT), MMF.index.min().date(),
    )
else:
    MMF_EXT = MMF   # no extension available; falls back to original series

# --- Shared returns ---
ret_tbsp = build_return_series(TBSP,    hedged=True)   # PLN bond index — no FX
ret_mmf  = build_return_series(MMF_EXT, hedged=True)   # PLN money market (extended)

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

def _run_equity_wf(price_df, label):
    logging.info("  [%s] walk-forward starting ...", label)
    wf_eq, wf_res, wf_tr = walk_forward(
        df=price_df, cash_df=MMF,
        train_years=TRAIN_YEARS, test_years=TEST_YEARS,
        vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
        filter_modes_override=FORCE_FILTER_MODE_EQ,
        X_grid=X_GRID_EQ, Y_grid=Y_GRID_EQ, fast_grid=FAST_EQ,
        slow_grid=SLOW_EQ, tv_grid=TV_EQ, sl_grid=SL_EQ,
        mom_lookback_grid=MOM_LB_EQ, objective="calmar", n_jobs=N_JOBS,
        fast_mode             = FAST_MODE,
        use_atr_stop          = USE_ATR_STOP,
        N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
        atr_window            = ATR_WINDOW,
    )
    m = compute_metrics(wf_eq)
    logging.info("  [%s] CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
                 label, m.get("CAGR",0)*100, m.get("Sharpe",0),
                 m.get("MaxDD",0)*100, m.get("CalMAR",0))
    sig = build_signal_series(wf_eq, wf_tr)
    logging.info("  [%s] signal: %.1f%% of OOS days in position", label, sig.mean()*100)
    return wf_eq, wf_res, wf_tr, sig

if PORTFOLIO_MODE == "global_equity":
    wf_eq_WIG,   wf_res_WIG,   wf_tr_WIG,   sig_WIG   = _run_equity_wf(price_WIG,   "WIG")
    wf_eq_SPX,   wf_res_SPX,   wf_tr_SPX,   sig_SPX   = _run_equity_wf(price_SPX,   "SP500")
    wf_eq_STOXX, wf_res_STOXX, wf_tr_STOXX, sig_STOXX = _run_equity_wf(price_STOXX, "STOXX600")
    wf_eq_NKX,   wf_res_NKX,   wf_tr_NKX,   sig_NKX   = _run_equity_wf(price_NKX,   "Nikkei225")
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
    df=TBSP, cash_df=MMF,
    train_years=TRAIN_YEARS, test_years=TEST_YEARS,
    vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
    filter_modes_override=FORCE_FILTER_MODE_BD,
    X_grid=X_GRID_BD, Y_grid=Y_GRID_BD, fast_grid=FAST_BD,
    slow_grid=SLOW_BD, tv_grid=TV_BD, sl_grid=SL_BD,
    mom_lookback_grid=[252], objective="calmar", n_jobs=N_JOBS,    fast_mode=FAST_MODE,
    use_atr_stop          = USE_ATR_STOP_BD,
    N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
    atr_window            = ATR_WINDOW,
)
m_tbsp = compute_metrics(wf_eq_TBSP)
logging.info("TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
             m_tbsp.get("CAGR",0)*100, m_tbsp.get("Sharpe",0),
             m_tbsp.get("MaxDD",0)*100, m_tbsp.get("CalMAR",0))
sig_TBSP = build_signal_series(wf_eq_TBSP, wf_tr_TBSP)
logging.info("TBSP signal: %.1f%% of OOS days in position", sig_TBSP.mean()*100)


# ============================================================
# PHASE 5 — ALLOCATION WALK-FORWARD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: ALLOCATION WALK-FORWARD  (N-asset, annual_cap=%d)", ANNUAL_CAP)
logging.info("=" * 80)

if PORTFOLIO_MODE == "global_equity":
    asset_keys        = ["WIG", "SP500", "STOXX600", "Nikkei225", "TBSP"]
    returns_dict      = {"WIG": ret_WIG.dropna(), "SP500": ret_SPX.dropna(),
                         "STOXX600": ret_STOXX.dropna(), "Nikkei225": ret_NKX.dropna(),
                         "TBSP": ret_tbsp.dropna()}
    signals_full_dict = {"WIG": sig_WIG, "SP500": sig_SPX,
                         "STOXX600": sig_STOXX, "Nikkei225": sig_NKX, "TBSP": sig_TBSP}
elif PORTFOLIO_MODE == "msci_world":
    asset_keys        = ["WIG20TR", "PL_MID", "MSCI_World", "TBSP"]
    returns_dict      = {"WIG20TR": ret_WIG20TR.dropna(), "PL_MID": ret_PL_MID.dropna(),
                         "MSCI_World": ret_MSCIW.dropna(), "TBSP": ret_tbsp.dropna()}
    signals_full_dict = {"WIG20TR": sig_WIG20, "PL_MID": sig_PLMID,
                         "MSCI_World": sig_MSCIW, "TBSP": sig_TBSP}

signals_oos_dict = dict(signals_full_dict)
wf_results_ref   = wf_res_TBSP

portfolio_equity, weights_series, reallocation_log, alloc_results_df = (
    allocation_walk_forward_n(
        returns_dict      = returns_dict,
        signals_full_dict = signals_full_dict,
        signals_oos_dict  = signals_oos_dict,
        mmf_returns       = ret_mmf.dropna(),
        wf_results_ref    = wf_results_ref,
        asset_keys        = asset_keys,
        step              = ALLOC_STEP,
        objective         = ALLOC_OBJECTIVE,
        cooldown_days     = COOLDOWN_DAYS,
        annual_cap        = ANNUAL_CAP,
        train_years       = TRAIN_YEARS,
    )
)

if portfolio_equity is None or portfolio_equity.empty:
    logging.error("Portfolio equity curve is empty — exiting.")
    sys.exit(1)

portfolio_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}
logging.info("PORTFOLIO OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
             portfolio_metrics.get("CAGR",0)*100, portfolio_metrics.get("Sharpe",0),
             portfolio_metrics.get("MaxDD",0)*100, portfolio_metrics.get("CalMAR",0))


# ============================================================
# PHASE 6 — REPORTING
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: REPORTING")
logging.info("=" * 80)

oos_start = portfolio_equity.index.min()
oos_end   = portfolio_equity.index.max()

bh_metrics_dict = {}
for key, ret in returns_dict.items():
    ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
    if ret_oos.empty:
        bh_metrics_dict[key] = {}
        continue
    bh_eq = (1 + ret_oos).cumprod()
    bh_eq = bh_eq / bh_eq.iloc[0]
    bh_metrics_dict[key] = {k: float(v) for k, v in compute_metrics(bh_eq).items()}

signals_oos_trimmed = {
    k: s.loc[(s.index >= oos_start) & (s.index <= oos_end)]
    for k, s in signals_oos_dict.items()
}

print_global_equity_report(
    portfolio_metrics = portfolio_metrics,
    bh_metrics_dict   = bh_metrics_dict,
    alloc_results_df  = alloc_results_df,
    reallocation_log  = reallocation_log,
    signals_oos_dict  = signals_oos_trimmed,
    oos_start         = oos_start,
    oos_end           = oos_end,
    portfolio_mode    = PORTFOLIO_MODE,
    fx_hedged         = FX_HEDGED,
)



# ============================================================
# PHASE 9 — DAILY OUTPUT  (status, log, chart, snapshot)
# Phases 7-8 (per-asset MC robustness + allocation weight perturbation)
# are skipped in the daily runner — run in global_equity_runfile.py only.
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 7: DAILY OUTPUT")
logging.info("=" * 80)

daily_result = build_daily_outputs(
    wf_results_dict    = {k: wf_results_ref for k in asset_keys},
    portfolio_equity   = portfolio_equity,
    portfolio_metrics  = portfolio_metrics,
    weights_series     = weights_series,
    reallocation_log   = reallocation_log,
    bh_metrics_dict    = bh_metrics_dict,
    returns_dict       = returns_dict,
    signals_oos_dict   = signals_oos_trimmed,
    asset_keys         = asset_keys,
    portfolio_mode     = PORTFOLIO_MODE,
    fx_hedged          = FX_HEDGED,
    output_dir         = OUTPUT_DIR,
    run_date           = dt.date.today(),
    gdrive_folder_id   = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()
                          or GDRIVE_FOLDER_ID_DEFAULT.strip() or None),
    gdrive_credentials = (os.path.join(tempfile.gettempdir(), "credentials.json")
                          if os.path.exists(
                              os.path.join(tempfile.gettempdir(), "credentials.json"))
                          else None),
)

logging.info("Daily output complete — action=%s", daily_result["action"])
for k in asset_keys:
    logging.info("  %-14s  signal=%s  weight=%.0f%%",
                 k, daily_result["signals"].get(k, "OUT"),
                 daily_result["weights"].get(k, 0.0) * 100)

logging.info("=" * 80)
logging.info("GLOBAL EQUITY DAILY RUNFILE COMPLETE: %s", dt.datetime.now())
logging.info("Mode: %s  |  FX: %s", PORTFOLIO_MODE, "hedged" if FX_HEDGED else "unhedged")
logging.info("OOS: %s to %s  (%d reallocations)",
             oos_start.date(), oos_end.date(), len(reallocation_log))
logging.info("=" * 80)
