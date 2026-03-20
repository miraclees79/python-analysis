"""
global_equity_window_sweep.py
==============================
Walk-forward window configuration sweep for the global equity portfolio.

PURPOSE
-------
Evaluates all combinations of:
  - Portfolio mode : "global_equity" (Mode A), "msci_world" (Mode B)
  - Train years    : 6, 7, 8, 9, 10, 12
  - Test years     : 1, 2

That is 2 × 6 × 2 = 24 configurations in total.

For each configuration the script runs:
  1. Per-asset signal walk-forward
  2. TBSP bond signal walk-forward
  3. N-asset allocation walk-forward
  4. Monte Carlo parameter robustness (n=N_MC_SAMPLES per asset)
  5. N-asset allocation weight perturbation (±10pp, ±20pp per asset)
  6. Buy-and-hold benchmarks for comparison

MODE B ASSET LIST (revised 20 Mar 2026)
----------------------------------------
Mode B was updated from WIG20TR + PL_MID + MSCI_World + TBSP to
WIG + MSCI_World + TBSP.  WIG is used in both modes; its walk-forward
result is pre-computed once and shared.

CACHING STRATEGY
-----------------
WIG and TBSP walk-forwards are purely functions of (train_years, test_years)
and do not vary by portfolio mode.  They are pre-computed once for every
(train_years, test_years) combination before the main sweep loop, then
looked up from a dict.  MSCI World walk-forwards are similarly pre-computed
for Mode B configs.

This eliminates:
  - 12 redundant WIG  walk-forward runs  (was run once per config, not per key)
  - 24 redundant TBSP walk-forward runs  (discover_oos_starts() first pass
    also ran TBSP 24 times; both eliminated)
  Total saving: ~40% of walk-forward runtime.

discover_oos_starts() has been removed entirely.  OOS start dates are
derived directly from the pre-computed tbsp_wf_cache.

OOS PERIOD ALIGNMENT
--------------------
Within each portfolio mode the OOS period is fixed to the *latest possible*
common start across all (train, test) combinations so that all configs per
mode are evaluated on exactly the same date range.

OUTPUT
------
  global_equity_sweep_results.csv  — full results table (one row per config)
  global_equity_sweep_report.txt   — human-readable formatted summary
  global_equity_sweep.log          — full run log
"""

import os
import sys
import csv
import logging
import tempfile
import datetime as dt
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from strategy_test_library import (
    download_csv,
    load_csv,
    walk_forward,
    compute_metrics,
)
from multiasset_library import build_signal_series

from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    BOND_THRESHOLDS_MC,
    EQUITY_THRESHOLDS_MC,
)

from global_equity_library import (
    build_return_series,
    build_blend,
    build_price_df_from_returns,
    build_mmf_extended,
    allocation_walk_forward_n,
    allocation_weight_robustness_n,
    CLOSE_COL,
    DATA_START,
)

from price_series_builder import build_and_upload, load_combined_from_drive
from msci_world_synthetic import (
    build_full_msci_world_extended,
    load_synthetic_from_drive,
)


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE = "global_equity_sweep.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
root_logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
root_logger.addHandler(_ch)


# ============================================================
# USER SETTINGS
# ============================================================

GDRIVE_FOLDER_ID_DEFAULT = ""   # <- paste your folder ID here

FX_HEDGED       = True
POSITION_MODE   = "full"
VOL_WINDOW      = 20
COOLDOWN_DAYS   = 10
ANNUAL_CAP      = 999
ALLOC_OBJECTIVE = "calmar"
ALLOC_STEP      = 0.10

FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]

_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

N_MC_SAMPLES  = 500
PERTURB_STEPS = [-0.2, -0.1, 0.0, 0.1, 0.2]
ALLOC_ROB_MIN_WEIGHT = 0.05

WIG_DATA_FLOOR = "1994-10-03"
MMF_FLOOR      = "1994-10-03"

TRAIN_YEARS_LIST = [6, 7, 8, 9, 10, 12]
TEST_YEARS_LIST  = [1, 2]
MODES_LIST       = ["global_equity", "msci_world"]

X_GRID_EQ = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ   = [50, 75, 100]
SLOW_EQ   = [150, 200, 250]
TV_EQ     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ = [126, 252]

X_GRID_BD = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD = [0.001]
FAST_BD   = [50, 100, 150]
SLOW_BD   = [200, 300, 400, 500]
TV_BD     = [0.08, 0.10, 0.12]
SL_BD     = [0.01, 0.02, 0.03]

CSV_OUTPUT = "global_equity_sweep_results.csv"
TXT_REPORT = "global_equity_sweep_report.txt"


# ============================================================
# PHASE 1 — DOWNLOAD ALL DATA ONCE
# ============================================================

def download_all_data() -> dict:
    logging.info("=" * 70)
    logging.info("DOWNLOADING ALL DATA")
    logging.info("=" * 70)

    tmp_dir = tempfile.gettempdir()
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()

    def _stooq(ticker, label, mandatory=True):
        url  = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        path = os.path.join(tmp_dir, f"sweep_{label}.csv")
        ok   = download_csv(url, path)
        if not ok:
            if mandatory:
                logging.error("FAIL mandatory: %s (%s)", label, ticker)
                sys.exit(1)
            logging.warning("WARN optional: %s (%s) not available", label, ticker)
            return None
        df = load_csv(path)
        if df is None:
            if mandatory:
                logging.error("FAIL load_csv: %s", label)
                sys.exit(1)
            return None
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info("OK  %-14s  %5d rows  %s to %s",
                     label, len(df), df.index.min().date(), df.index.max().date())
        return df

    raw = {}

    # TBSP extended (Drive CSV + stooq extension)
    raw["TBSP"] = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "tbsp_extended_full.csv",
        combined_filename = "tbsp_extended_combined.csv",
        extension_ticker  = "^tbsp",
        extension_source  = "stooq",
    )
    if raw["TBSP"] is None:
        logging.error("FAIL: TBSP build/update failed.")
        sys.exit(1)
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "TBSP", len(raw["TBSP"]),
                 raw["TBSP"].index.min().date(), raw["TBSP"].index.max().date())

    raw["MMF"]     = _stooq("2720.n",   "MMF")
    raw["WIBOR1M"] = _stooq("plopln1m", "WIBOR1M", mandatory=False)

    # FX
    raw["USDPLN"] = _stooq("usdpln", "USDPLN")
    raw["EURPLN"] = _stooq("eurpln", "EURPLN")
    raw["JPYPLN"] = _stooq("jpypln", "JPYPLN")

    # WIG (both modes)
    raw["WIG"] = _stooq("wig", "WIG")

    # Mode A: SP500, STOXX600, Nikkei
    raw["SPX"] = _stooq("^spx", "SP500")
    raw["NKX"] = _stooq("^nkx", "Nikkei225")

    raw["STOXX600"] = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "stoxx600.csv",
        combined_filename = "stoxx600_combined.csv",
        extension_ticker  = "^STOXX",
    )
    if raw["STOXX600"] is None:
        logging.error("FAIL: STOXX 600 build/update failed.")
        sys.exit(1)
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "STOXX600", len(raw["STOXX600"]),
                 raw["STOXX600"].index.min().date(), raw["STOXX600"].index.max().date())

    # Mode B: MSCI World (synthetic 1990 + WSJ 2010 + URTH)
    msciw_wsj = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "msci_world_wsj_raw.csv",
        combined_filename = "msci_world_combined.csv",
        extension_ticker  = "URTH",
    )
    if msciw_wsj is None:
        logging.error("FAIL: MSCI World (WSJ+URTH) build failed.")
        sys.exit(1)
    raw["MSCIW"] = build_full_msci_world_extended(
        wsj_combined_df = msciw_wsj,
        folder_id       = folder_id,
    )
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "MSCI_World", len(raw["MSCIW"]),
                 raw["MSCIW"].index.min().date(), raw["MSCIW"].index.max().date())

    logging.info("All data downloaded.")
    return raw


# ============================================================
# BUILD DERIVED SERIES (once per session)
# ============================================================

def build_derived(raw: dict) -> dict:
    fx_usd = raw["USDPLN"][CLOSE_COL] if raw["USDPLN"] is not None else None
    fx_eur = raw["EURPLN"][CLOSE_COL] if raw["EURPLN"] is not None else None
    fx_jpy = raw["JPYPLN"][CLOSE_COL] if raw["JPYPLN"] is not None else None

    # WIG: clip to daily-continuous trading era
    wig = raw["WIG"].copy()
    wig = wig.loc[wig.index >= pd.Timestamp(WIG_DATA_FLOOR)]

    # Extended MMF
    mmf_ext = (build_mmf_extended(raw["MMF"], raw["WIBOR1M"], floor_date=MMF_FLOOR)
               if raw["WIBOR1M"] is not None else raw["MMF"])

    ret_tbsp = build_return_series(raw["TBSP"], hedged=True)
    ret_mmf  = build_return_series(mmf_ext,     hedged=True)

    # Mode A returns
    ret_WIG   = build_return_series(wig,            hedged=True)
    ret_SPX   = build_return_series(raw["SPX"],     fx_series=fx_usd, hedged=FX_HEDGED)
    ret_STOXX = build_return_series(raw["STOXX600"], fx_series=fx_eur, hedged=FX_HEDGED)
    ret_NKX   = build_return_series(raw["NKX"],     fx_series=fx_jpy, hedged=FX_HEDGED)

    price_WIG   = wig              if FX_HEDGED else build_price_df_from_returns(ret_WIG,   "WIG_PLN")
    price_SPX   = raw["SPX"]       if FX_HEDGED else build_price_df_from_returns(ret_SPX,   "SP500_PLN")
    price_STOXX = raw["STOXX600"]  if FX_HEDGED else build_price_df_from_returns(ret_STOXX, "STOXX600_PLN")
    price_NKX   = raw["NKX"]       if FX_HEDGED else build_price_df_from_returns(ret_NKX,   "Nikkei225_PLN")

    # Mode B returns
    ret_MSCIW   = build_return_series(raw["MSCIW"], fx_series=fx_usd, hedged=FX_HEDGED)
    price_MSCIW = raw["MSCIW"] if FX_HEDGED else build_price_df_from_returns(ret_MSCIW, "MSCIW_PLN")

    return dict(
        ret_tbsp=ret_tbsp, ret_mmf=ret_mmf, mmf_df=raw["MMF"],
        # WIG (shared)
        price_WIG=price_WIG, ret_WIG=ret_WIG,
        # Mode A
        price_SPX=price_SPX, ret_SPX=ret_SPX,
        price_STOXX=price_STOXX, ret_STOXX=ret_STOXX,
        price_NKX=price_NKX, ret_NKX=ret_NKX,
        # Mode B
        price_MSCIW=price_MSCIW, ret_MSCIW=ret_MSCIW,
    )


# ============================================================
# PRE-COMPUTE SHARED WALK-FORWARDS
# ============================================================

def precompute_shared_wf(raw: dict, derived: dict) -> tuple[dict, dict, dict]:
    """
    Pre-compute WIG, TBSP, and MSCI World walk-forwards for every
    (train_years, test_years) key.

    WIG and TBSP do not vary by portfolio mode — computing them once
    per key and reusing across modes eliminates ~40% of total WF runtime
    vs the naive approach of running them fresh inside run_one_config().

    MSCI World is used only in Mode B but is also pre-computed here so
    the per-config function remains a simple cache lookup.

    Returns
    -------
    wig_cache   : {(train, test): (wf_eq, wf_res, wf_tr, sig)}
    tbsp_cache  : {(train, test): (wf_eq, wf_res, wf_tr, sig)}
    msciw_cache : {(train, test): (wf_eq, wf_res, wf_tr, sig)}
    """
    logging.info("=" * 70)
    logging.info("PRE-COMPUTING SHARED WALK-FORWARDS (WIG, TBSP, MSCI World)")
    logging.info("%d train configs × %d test configs = %d keys each",
                 len(TRAIN_YEARS_LIST), len(TEST_YEARS_LIST),
                 len(TRAIN_YEARS_LIST) * len(TEST_YEARS_LIST))
    logging.info("=" * 70)

    wig_cache   = {}
    tbsp_cache  = {}
    msciw_cache = {}
    mmf_df = raw["MMF"]

    for train_years in TRAIN_YEARS_LIST:
        for test_years in TEST_YEARS_LIST:
            key = (train_years, test_years)

            # ── WIG ─────────────────────────────────────────────────────
            logging.info("WIG   WF: train=%dy  test=%dy", train_years, test_years)
            eq, res, tr = walk_forward(
                df                    = derived["price_WIG"],
                cash_df               = mmf_df,
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
                n_jobs=N_JOBS,
            )
            sig = build_signal_series(eq, tr)
            m   = compute_metrics(eq)
            logging.info("  WIG   WF done: CAGR=%.2f%%  signal=%.1f%%",
                         m.get("CAGR", 0) * 100, sig.mean() * 100)
            wig_cache[key] = (eq, res, tr, sig)

            # ── TBSP ────────────────────────────────────────────────────
            logging.info("TBSP  WF: train=%dy  test=%dy", train_years, test_years)
            eq, res, tr = walk_forward(
                df                    = raw["TBSP"],
                cash_df               = mmf_df,
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
            )
            sig = build_signal_series(eq, tr)
            m   = compute_metrics(eq)
            logging.info("  TBSP  WF done: CAGR=%.2f%%  signal=%.1f%%",
                         m.get("CAGR", 0) * 100, sig.mean() * 100)
            tbsp_cache[key] = (eq, res, tr, sig)

            # ── MSCI World ──────────────────────────────────────────────
            logging.info("MSCIW WF: train=%dy  test=%dy", train_years, test_years)
            eq, res, tr = walk_forward(
                df                    = derived["price_MSCIW"],
                cash_df               = mmf_df,
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
                n_jobs=N_JOBS,
            )
            sig = build_signal_series(eq, tr)
            m   = compute_metrics(eq)
            logging.info("  MSCIW WF done: CAGR=%.2f%%  signal=%.1f%%",
                         m.get("CAGR", 0) * 100, sig.mean() * 100)
            msciw_cache[key] = (eq, res, tr, sig)

    logging.info("Pre-computation complete: %d WIG + %d TBSP + %d MSCIW walk-forwards.",
                 len(wig_cache), len(tbsp_cache), len(msciw_cache))
    return wig_cache, tbsp_cache, msciw_cache


# ============================================================
# OOS START DETERMINATION (from cache — no separate first pass)
# ============================================================

def get_common_oos_starts(tbsp_cache: dict, derived: dict) -> dict:
    """
    Derive the common OOS start per mode directly from the pre-computed
    TBSP walk-forward cache.  Replaces the former discover_oos_starts()
    function which ran a redundant TBSP walk-forward first pass.

    Mode A binding constraint: TBSP.
    Mode B binding constraint: max(TBSP OOS start, MSCI World data start
                                   + train_years).
    """
    oos_starts = {"global_equity": [], "msci_world": []}

    msciw_data_start = derived["ret_MSCIW"].dropna().index.min()

    for (train_years, test_years), (_, res, _, _) in tbsp_cache.items():
        tbsp_oos = res["TestStart"].min()

        oos_starts["global_equity"].append(tbsp_oos)

        msciw_oos = msciw_data_start + pd.DateOffset(years=train_years)
        oos_starts["msci_world"].append(max(tbsp_oos, msciw_oos))

    result = {}
    for mode, starts in oos_starts.items():
        result[mode] = max(starts)
        logging.info("Common OOS start [%-15s]: %s", mode, result[mode].date())

    return result


# ============================================================
# SINGLE CONFIGURATION RUN
# ============================================================

def _run_equity_wf(price_df, label, mmf_df):
    """Run a single equity walk-forward and return (eq, res, tr, sig)."""
    eq, res, tr = walk_forward(
        df                    = price_df,
        cash_df               = mmf_df,
        train_years           = _current_train,    # set by run_one_config
        test_years            = _current_test,
        vol_window            = VOL_WINDOW,
        selected_mode         = POSITION_MODE,
        filter_modes_override = FORCE_FILTER_MODE_EQ,
        X_grid=X_GRID_EQ, Y_grid=Y_GRID_EQ,
        fast_grid=FAST_EQ, slow_grid=SLOW_EQ,
        tv_grid=TV_EQ, sl_grid=SL_EQ,
        mom_lookback_grid=MOM_LB_EQ,
        objective="calmar",
        n_jobs=N_JOBS,
    )
    sig = build_signal_series(eq, tr)
    m   = compute_metrics(eq)
    logging.info("  [%s] CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f  signal=%.1f%%",
                 label, m.get("CAGR",0)*100, m.get("Sharpe",0),
                 m.get("MaxDD",0)*100, m.get("CalMAR",0), sig.mean()*100)
    return eq, res, tr, sig

# Module-level mutable for passing train/test into _run_equity_wf cleanly
_current_train = None
_current_test  = None


def run_one_config(
    mode:             str,
    train_years:      int,
    test_years:       int,
    raw:              dict,
    derived:          dict,
    common_oos_start: pd.Timestamp,
    wig_cache:        dict,
    tbsp_cache:       dict,
    msciw_cache:      dict,
) -> dict:
    """
    Run the full pipeline for one (mode, train, test) configuration.

    WIG, TBSP, and MSCI World results are looked up from the pre-computed
    caches instead of being recomputed.  Only the mode-specific per-window
    equity walk-forwards (SP500, STOXX600, Nikkei225 for Mode A) are run
    fresh per config.
    """
    global _current_train, _current_test
    _current_train = train_years
    _current_test  = test_years

    label = f"{mode}  train={train_years}y  test={test_years}y"
    logging.info("")
    logging.info("=" * 70)
    logging.info("CONFIG: %s", label)
    logging.info("=" * 70)
    t0 = time.time()

    key    = (train_years, test_years)
    mmf_df = raw["MMF"]

    # ── Pull cached WIG and TBSP results ──────────────────────────────────
    wf_eq_WIG,  wf_res_WIG,  wf_tr_WIG,  sig_WIG  = wig_cache[key]
    wf_eq_TBSP, wf_res_TBSP, wf_tr_TBSP, sig_TBSP = tbsp_cache[key]

    logging.info("  [WIG]  (cached) CAGR=%.2f%%  signal=%.1f%%",
                 compute_metrics(wf_eq_WIG).get("CAGR",0)*100, sig_WIG.mean()*100)
    logging.info("  [TBSP] (cached) CAGR=%.2f%%  signal=%.1f%%",
                 compute_metrics(wf_eq_TBSP).get("CAGR",0)*100, sig_TBSP.mean()*100)

    # ── Mode-specific walk-forwards ───────────────────────────────────────
    if mode == "global_equity":
        wf_eq_SPX,   wf_res_SPX,   wf_tr_SPX,   sig_SPX   = _run_equity_wf(
            derived["price_SPX"],   "SP500",     mmf_df)
        wf_eq_STOXX, wf_res_STOXX, wf_tr_STOXX, sig_STOXX = _run_equity_wf(
            derived["price_STOXX"], "STOXX600",  mmf_df)
        wf_eq_NKX,   wf_res_NKX,   wf_tr_NKX,   sig_NKX   = _run_equity_wf(
            derived["price_NKX"],   "Nikkei225", mmf_df)

    elif mode == "msci_world":
        # MSCI World also pre-computed
        wf_eq_MSCIW, wf_res_MSCIW, wf_tr_MSCIW, sig_MSCIW = msciw_cache[key]
        logging.info("  [MSCIW] (cached) CAGR=%.2f%%  signal=%.1f%%",
                     compute_metrics(wf_eq_MSCIW).get("CAGR",0)*100, sig_MSCIW.mean()*100)

    # ── Assemble mode-specific dicts ──────────────────────────────────────
    if mode == "global_equity":
        asset_keys   = ["WIG", "SP500", "STOXX600", "Nikkei225", "TBSP"]
        returns_dict = {
            "WIG":       derived["ret_WIG"].dropna(),
            "SP500":     derived["ret_SPX"].dropna(),
            "STOXX600":  derived["ret_STOXX"].dropna(),
            "Nikkei225": derived["ret_NKX"].dropna(),
            "TBSP":      derived["ret_tbsp"].dropna(),
        }
        signals_full = {
            "WIG": sig_WIG, "SP500": sig_SPX, "STOXX600": sig_STOXX,
            "Nikkei225": sig_NKX, "TBSP": sig_TBSP,
        }
        wf_results_ref = wf_res_TBSP
        # Assets for MC robustness: (label, price_df, wf_eq, wf_res, thresholds)
        mc_assets = [
            ("WIG",       derived["price_WIG"],   wf_eq_WIG,   wf_res_WIG,   EQUITY_THRESHOLDS_MC),
            ("SP500",     derived["price_SPX"],   wf_eq_SPX,   wf_res_SPX,   EQUITY_THRESHOLDS_MC),
            ("STOXX600",  derived["price_STOXX"], wf_eq_STOXX, wf_res_STOXX, EQUITY_THRESHOLDS_MC),
            ("Nikkei225", derived["price_NKX"],   wf_eq_NKX,   wf_res_NKX,   EQUITY_THRESHOLDS_MC),
            ("TBSP",      raw["TBSP"],            wf_eq_TBSP,  wf_res_TBSP,  BOND_THRESHOLDS_MC),
        ]

    else:  # msci_world
        asset_keys   = ["WIG", "MSCI_World", "TBSP"]
        returns_dict = {
            "WIG":        derived["ret_WIG"].dropna(),
            "MSCI_World": derived["ret_MSCIW"].dropna(),
            "TBSP":       derived["ret_tbsp"].dropna(),
        }
        signals_full = {
            "WIG": sig_WIG, "MSCI_World": sig_MSCIW, "TBSP": sig_TBSP,
        }
        wf_results_ref = wf_res_TBSP
        mc_assets = [
            ("WIG",        derived["price_WIG"],   wf_eq_WIG,   wf_res_WIG,   EQUITY_THRESHOLDS_MC),
            ("MSCI_World", derived["price_MSCIW"], wf_eq_MSCIW, wf_res_MSCIW, EQUITY_THRESHOLDS_MC),
            ("TBSP",       raw["TBSP"],            wf_eq_TBSP,  wf_res_TBSP,  BOND_THRESHOLDS_MC),
        ]

    # ── Allocation walk-forward ───────────────────────────────────────────
    portfolio_equity, weights_series, realloc_log, alloc_df = allocation_walk_forward_n(
        returns_dict      = returns_dict,
        signals_full_dict = signals_full,
        signals_oos_dict  = dict(signals_full),
        mmf_returns       = derived["ret_mmf"].dropna(),
        wf_results_ref    = wf_results_ref,
        asset_keys        = asset_keys,
        step              = ALLOC_STEP,
        objective         = ALLOC_OBJECTIVE,
        cooldown_days     = COOLDOWN_DAYS,
        annual_cap        = ANNUAL_CAP,
        train_years       = train_years,
    )

    if portfolio_equity is None or portfolio_equity.empty:
        logging.warning("Empty portfolio equity — skipping config.")
        return None

    oos_end = portfolio_equity.index.max()

    # Trim to common OOS start and renormalise
    port_eq_trimmed = portfolio_equity.loc[
        (portfolio_equity.index >= common_oos_start) &
        (portfolio_equity.index <= oos_end)
    ]
    if port_eq_trimmed.empty:
        logging.warning("Portfolio empty after trimming to common OOS start %s",
                        common_oos_start.date())
        return None

    port_eq_trimmed = port_eq_trimmed / port_eq_trimmed.iloc[0]
    port_metrics = {k: float(v) for k, v in compute_metrics(port_eq_trimmed).items()}

    logging.info("  Portfolio [%s–%s]: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.2f",
                 common_oos_start.date(), oos_end.date(),
                 port_metrics.get("CAGR",0)*100, port_metrics.get("Sharpe",0),
                 port_metrics.get("MaxDD",0)*100, port_metrics.get("CalMAR",0))

    # Trim signals to common OOS period
    sigs_trimmed = {
        k: s.loc[(s.index >= common_oos_start) & (s.index <= oos_end)]
        for k, s in signals_full.items()
    }

    # ── B&H benchmarks ────────────────────────────────────────────────────
    bh_metrics = {}
    for key_asset, ret in returns_dict.items():
        r_oos = ret.loc[(ret.index >= common_oos_start) & (ret.index <= oos_end)]
        if r_oos.empty:
            continue
        bh_eq = (1 + r_oos).cumprod()
        bh_eq = bh_eq / bh_eq.iloc[0]
        bh_metrics[key_asset] = {k: float(v) for k, v in compute_metrics(bh_eq).items()}

    # ── MC robustness ─────────────────────────────────────────────────────
    mc_verdicts = {}
    for lbl, price_df, wf_eq, wf_res, thr_mc in mc_assets:
        logging.info("  MC [%s] n=%d ...", lbl, N_MC_SAMPLES)
        try:
            _wins   = extract_windows_from_wf_results(wf_res, train_years=train_years)
            _params = extract_best_params_from_wf_results(wf_res)
            _mc_raw = run_monte_carlo_robustness(
                best_params   = _params,
                windows       = _wins,
                df            = price_df,
                cash_df       = mmf_df,
                vol_window    = VOL_WINDOW,
                selected_mode = POSITION_MODE,
                funds_df      = None,
                n_samples     = N_MC_SAMPLES,
                n_jobs        = N_JOBS,
                perturb_pct   = 0.20,
                seed          = 42,
                price_col     = CLOSE_COL,
            )
            _summary = analyze_robustness(_mc_raw, compute_metrics(wf_eq),
                                          thresholds=thr_mc)
            mc_verdicts[lbl] = {
                "verdict":    _summary.get("verdict", "N/A"),
                "p05_cagr":  _summary.get("CAGR",   {}).get("p05",    float("nan")),
                "p05_maxdd": _summary.get("MaxDD",  {}).get("p05",    float("nan")),
                "p05_sharpe":_summary.get("Sharpe", {}).get("p05",    float("nan")),
                "p_loss":    _summary.get("p_loss", float("nan")),
                "median_cagr":_summary.get("CAGR",  {}).get("median", float("nan")),
                "fail_detail": "; ".join(
                    k for k, v in {
                        "p05_CAGR<0":   _summary.get("CAGR",  {}).get("p05",0) < thr_mc.get("CAGR",  {}).get("p05_min",0),
                        "p05_Sharpe<0": _summary.get("Sharpe",{}).get("p05",0) < thr_mc.get("Sharpe",{}).get("p05_min",0),
                        "p05_MaxDD<thr":_summary.get("MaxDD", {}).get("p05",0) < thr_mc.get("MaxDD", {}).get("p05_min",-1),
                        "p_loss>10%":   _summary.get("p_loss",0) > 0.10,
                        "outlier_CAGR": _summary.get("CAGR",  {}).get("p_worse_than_baseline",0) > 0.85,
                        "low_median":   _summary.get("CAGR",  {}).get("median",1) < 0.02,
                    }.items() if v
                ) or "\u2014",
            }
            logging.info("  MC [%s]: %s", lbl, mc_verdicts[lbl]["verdict"])
        except Exception as exc:
            logging.error("  MC [%s] failed: %s", lbl, exc)
            mc_verdicts[lbl] = {"verdict": "ERROR", "fail_detail": str(exc),
                                "p05_cagr": float("nan")}

    # ── Allocation weight perturbation ────────────────────────────────────
    w_cols = [c for c in alloc_df.columns if c.startswith("w_") and c != "w_mmf"]
    focus_assets = [c[2:] for c in w_cols
                    if alloc_df[c].mean() >= ALLOC_ROB_MIN_WEIGHT
                    and c[2:] in asset_keys]
    alloc_rob = {}

    for focus in focus_assets:
        try:
            rob_df = allocation_weight_robustness_n(
                alloc_results_df = alloc_df,
                returns_dict     = returns_dict,
                mmf_returns      = derived["ret_mmf"],
                signals_oos_dict = sigs_trimmed,
                asset_keys       = asset_keys,
                baseline_metrics = port_metrics,
                perturb_steps    = PERTURB_STEPS,
                focus_asset      = focus,
                cooldown_days    = COOLDOWN_DAYS,
                annual_cap       = ANNUAL_CAP,
            )
            inner = rob_df.loc[rob_df["weight_shift"].abs().between(0.09, 0.11)]
            min_ratio = float(inner["CalMAR_vs_base"].min()) if not inner.empty else float("nan")
            verdict = ("PLATEAU"  if min_ratio >= 0.80
                       else "MODERATE" if min_ratio >= 0.60
                       else "SPIKE"    if not np.isnan(min_ratio)
                       else "N/A")
            alloc_rob[focus] = {"verdict": verdict, "min_calmar_ratio": min_ratio}
            logging.info("  Alloc rob [%s]: %s (ratio %.3f)", focus, verdict, min_ratio)
        except Exception as exc:
            logging.error("  Alloc rob [%s] failed: %s", focus, exc)
            alloc_rob[focus] = {"verdict": "ERROR", "min_calmar_ratio": float("nan")}

    elapsed = time.time() - t0
    logging.info("Config done: %s  (%.0f min)", label, elapsed / 60)

    return {
        "mode":            mode,
        "train_years":     train_years,
        "test_years":      test_years,
        "oos_start":       str(common_oos_start.date()),
        "oos_end":         str(oos_end.date()),
        "n_reallocations": len(realloc_log),
        "port_cagr":       port_metrics.get("CAGR"),
        "port_vol":        port_metrics.get("Vol"),
        "port_sharpe":     port_metrics.get("Sharpe"),
        "port_maxdd":      port_metrics.get("MaxDD"),
        "port_calmar":     port_metrics.get("CalMAR"),
        "port_sortino":    port_metrics.get("Sortino"),
        "bh_metrics":      bh_metrics,
        "mc_verdicts":     mc_verdicts,
        "alloc_rob":       alloc_rob,
        "mean_weights":    {c[2:]: float(alloc_df[c].mean()) for c in w_cols},
    }


# ============================================================
# RESULTS FORMATTING
# ============================================================

def _pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.1f}%"

def _f2(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


def format_report(results: list) -> str:
    lines = []
    sep  = "=" * 130
    sep2 = "-" * 130

    lines += [
        sep,
        "GLOBAL EQUITY WALK-FORWARD WINDOW SWEEP",
        f"Run date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"MC samples per asset: {N_MC_SAMPLES}",
        f"FX treatment: {'hedged' if FX_HEDGED else 'unhedged'}",
        f"Mode B asset list: WIG + MSCI_World + TBSP (revised 20 Mar 2026)",
        sep,
        "",
    ]

    # ── Section 1: Summary ────────────────────────────────────────────────
    lines += [
        "SECTION 1: PORTFOLIO OOS PERFORMANCE SUMMARY",
        "All metrics over the common OOS period per mode.",
        sep2,
    ]
    hdr = (
        f"{'Mode':<15} {'Train':>5} {'Test':>4} {'OOS Start':>10} "
        f"{'CAGR':>7} {'Vol':>6} {'Sharpe':>7} {'MaxDD':>8} {'CalMAR':>8} "
        f"{'Reallocations':>14}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        if r is None:
            continue
        lines.append(
            f"{r['mode']:<15} {r['train_years']:>5} {r['test_years']:>4} "
            f"{r['oos_start']:>10} "
            f"{_pct(r['port_cagr']):>7} {_pct(r['port_vol']):>6} "
            f"{_f2(r['port_sharpe']):>7} {_pct(r['port_maxdd']):>8} "
            f"{_f2(r['port_calmar']):>8} {r['n_reallocations']:>14}"
        )
    lines += ["", ""]

    # ── Section 2: B&H comparison per mode ───────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue
        asset_keys = list(mode_results[0]["bh_metrics"].keys()) if mode_results else []
        lines += [f"SECTION 2: B&H COMPARISON — {mode.upper()}", sep2]
        bh_hdr = f"{'Train':>5} {'Test':>4} | {'Port CAGR':>10} {'Port CalMAR':>12} | "
        bh_hdr += " | ".join(f"{k[:8]:>9} B&H" for k in asset_keys)
        lines.append(bh_hdr)
        lines.append("-" * len(bh_hdr))
        for r in mode_results:
            bh_cagrs = " | ".join(
                f"{_pct(r['bh_metrics'].get(k,{}).get('CAGR',None)):>11}"
                for k in asset_keys
            )
            lines.append(
                f"{r['train_years']:>5} {r['test_years']:>4} | "
                f"{_pct(r['port_cagr']):>10} {_f2(r['port_calmar']):>12} | "
                f"{bh_cagrs}"
            )
        lines += ["", ""]

    # ── Section 3: MC robustness ──────────────────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue
        all_mc_assets = sorted({k for r in mode_results for k in r.get("mc_verdicts", {})})
        lines += [
            f"SECTION 3: MC PARAMETER ROBUSTNESS — {mode.upper()}",
            "  ROBUST = passes all gates  |  FRAGILE = one or more gates failed",
            sep2,
        ]
        hdr3 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:10]:>11}" for a in all_mc_assets)
        lines.append(hdr3)
        lines.append("-" * len(hdr3))
        for r in mode_results:
            cells = []
            for a in all_mc_assets:
                v = r.get("mc_verdicts", {}).get(a, {})
                tag = f"{'R' if v.get('verdict')=='ROBUST' else 'F' if v.get('verdict')=='FRAGILE' else v.get('verdict','?')[:3]}"
                p05 = v.get("p05_cagr", float("nan"))
                cells.append(f"{tag}({_pct(p05)})"[:11].rjust(11))
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | " + " | ".join(cells))
        lines += [""]
        lines.append("  Failure reasons (FRAGILE assets only):")
        for r in mode_results:
            for a in all_mc_assets:
                v = r.get("mc_verdicts", {}).get(a, {})
                if v.get("verdict") == "FRAGILE":
                    lines.append(
                        f"    train={r['train_years']}y test={r['test_years']}y "
                        f"[{a}]: {v.get('fail_detail','')}"
                    )
        lines += ["", ""]

    # ── Section 4: Allocation robustness ──────────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue
        all_alloc = sorted({k for r in mode_results for k in r.get("alloc_rob", {})})
        lines += [
            f"SECTION 4: ALLOCATION WEIGHT ROBUSTNESS — {mode.upper()}",
            "  PLATEAU >= 0.80 CalMAR ratio at +/-10pp  |  SPIKE < 0.60",
            sep2,
        ]
        hdr4 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:10]:>13}" for a in all_alloc)
        lines.append(hdr4)
        lines.append("-" * len(hdr4))
        for r in mode_results:
            cells = []
            for a in all_alloc:
                v = r.get("alloc_rob", {}).get(a, {})
                ratio = v.get("min_calmar_ratio", float("nan"))
                ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
                tag = f"{v.get('verdict','?')[:3]}({ratio_str})"[:13].rjust(13)
                cells.append(tag)
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | " + " | ".join(cells))
        lines += ["", ""]

    # ── Section 5: Mean allocation weights ───────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue
        all_w = sorted({k for r in mode_results for k in r.get("mean_weights", {})})
        lines += [f"SECTION 5: MEAN ALLOCATION WEIGHTS — {mode.upper()}", sep2]
        hdr5 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:8]:>9}" for a in all_w)
        lines.append(hdr5)
        lines.append("-" * len(hdr5))
        for r in mode_results:
            cells = " | ".join(
                f"{_pct(r['mean_weights'].get(a, 0.0)):>9}" for a in all_w
            )
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | {cells}")
        lines += ["", ""]

    lines += [sep, "END OF REPORT"]
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("=" * 70)
    logging.info("GLOBAL EQUITY WINDOW SWEEP — START: %s", dt.datetime.now())
    logging.info("Train years: %s  |  Test years: %s  |  Modes: %s",
                 TRAIN_YEARS_LIST, TEST_YEARS_LIST, MODES_LIST)
    logging.info("MC samples: %d  |  N_JOBS: %d", N_MC_SAMPLES, N_JOBS)
    n_keys    = len(TRAIN_YEARS_LIST) * len(TEST_YEARS_LIST)
    n_configs = len(MODES_LIST) * n_keys
    logging.info("Pre-compute: %d WIG + %d TBSP + %d MSCIW = %d shared WF runs",
                 n_keys, n_keys, n_keys, n_keys * 3)
    logging.info("Per-config:  Mode A = 3 WF runs (SPX/STOXX/NKX)  |  Mode B = 0 additional WF runs")
    logging.info("Total configurations: %d", n_configs)
    logging.info("=" * 70)

    # Step 1: Download everything once
    raw     = download_all_data()
    derived = build_derived(raw)

    # Step 2: Pre-compute shared walk-forwards (WIG, TBSP, MSCI World)
    wig_cache, tbsp_cache, msciw_cache = precompute_shared_wf(raw, derived)

    # Step 3: Derive common OOS start per mode from cache (no separate first pass)
    common_oos_starts = get_common_oos_starts(tbsp_cache, derived)

    # Step 4: Run all configurations
    all_results = []
    config_num  = 0
    for mode in MODES_LIST:
        common_start = common_oos_starts[mode]
        for train_years in TRAIN_YEARS_LIST:
            for test_years in TEST_YEARS_LIST:
                config_num += 1
                logging.info("")
                logging.info(">>> Config %d/%d: %s  train=%dy  test=%dy  common_oos=%s",
                             config_num, n_configs, mode, train_years, test_years,
                             common_start.date())
                result = run_one_config(
                    mode             = mode,
                    train_years      = train_years,
                    test_years       = test_years,
                    raw              = raw,
                    derived          = derived,
                    common_oos_start = common_start,
                    wig_cache        = wig_cache,
                    tbsp_cache       = tbsp_cache,
                    msciw_cache      = msciw_cache,
                )
                all_results.append(result)

    # Step 5: Write CSV
    valid = [r for r in all_results if r is not None]

    csv_cols = [
        "mode", "train_years", "test_years", "oos_start", "oos_end",
        "n_reallocations",
        "port_cagr", "port_vol", "port_sharpe", "port_maxdd",
        "port_calmar", "port_sortino",
    ]
    if valid:
        sample = valid[0]
        for k in sorted(sample["bh_metrics"]):
            csv_cols += [f"bh_{k}_cagr", f"bh_{k}_sharpe"]
        for k in sorted(sample["mc_verdicts"]):
            csv_cols += [f"mc_{k}_verdict", f"mc_{k}_p05cagr", f"mc_{k}_faildetail"]
        for k in sorted(sample["alloc_rob"]):
            csv_cols += [f"alloc_{k}_verdict", f"alloc_{k}_ratio"]
        for k in sorted(sample["mean_weights"]):
            csv_cols.append(f"w_{k}_mean")

    def _row(r):
        row = {c: "" for c in csv_cols}
        for c in ["mode", "train_years", "test_years", "oos_start", "oos_end",
                  "n_reallocations", "port_cagr", "port_vol", "port_sharpe",
                  "port_maxdd", "port_calmar", "port_sortino"]:
            v = r.get(c, "")
            row[c] = f"{v:.6f}" if isinstance(v, float) else v
        for k, m in r.get("bh_metrics", {}).items():
            if f"bh_{k}_cagr" in row:
                row[f"bh_{k}_cagr"]   = f"{m.get('CAGR',''):.6f}"   if "CAGR"   in m else ""
                row[f"bh_{k}_sharpe"] = f"{m.get('Sharpe',''):.4f}" if "Sharpe" in m else ""
        for k, v in r.get("mc_verdicts", {}).items():
            if f"mc_{k}_verdict" in row:
                row[f"mc_{k}_verdict"]    = v.get("verdict", "")
                row[f"mc_{k}_p05cagr"]    = f"{v.get('p05_cagr',''):.4f}" if "p05_cagr" in v else ""
                row[f"mc_{k}_faildetail"] = v.get("fail_detail", "")
        for k, v in r.get("alloc_rob", {}).items():
            if f"alloc_{k}_verdict" in row:
                row[f"alloc_{k}_verdict"] = v.get("verdict", "")
                ratio = v.get("min_calmar_ratio", float("nan"))
                row[f"alloc_{k}_ratio"] = f"{ratio:.4f}" if not np.isnan(ratio) else ""
        for k, w in r.get("mean_weights", {}).items():
            if f"w_{k}_mean" in row:
                row[f"w_{k}_mean"] = f"{w:.4f}"
        return row

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for r in valid:
            writer.writerow(_row(r))
    logging.info("CSV written: %s  (%d rows)", CSV_OUTPUT, len(valid))

    # Step 6: Write text report
    report_text = format_report(valid)
    with open(TXT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info("Text report written: %s", TXT_REPORT)
    print("\n" + report_text)

    logging.info("=" * 70)
    logging.info("SWEEP COMPLETE: %s", dt.datetime.now())
    logging.info("=" * 70)


if __name__ == "__main__":
    main()