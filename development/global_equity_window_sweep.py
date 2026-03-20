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
  4. Monte Carlo parameter robustness (1000 samples per asset)
  5. N-asset allocation weight perturbation (±10pp, ±20pp per asset)
  6. Buy-and-hold benchmarks for comparison

OOS PERIOD ALIGNMENT
--------------------
Within each portfolio mode the OOS period is fixed to the *latest possible*
common start across all (train, test) combinations, so that all 8 configs
per mode are evaluated on exactly the same date range.  This makes
out-of-sample comparisons between window sizes meaningful.

Concretely:
  Mode A  —  OOS start = max OOS start across all 8 combinations for Mode A
  Mode B  —  OOS start = max OOS start across all 8 combinations for Mode B

Once the common OOS start is determined, each combination's equity curves,
signals and metrics are trimmed to [common_oos_start, common_oos_end].

OUTPUT
------
  global_equity_sweep_results.csv  — full results table (one row per config)
  global_equity_sweep_report.txt   — human-readable formatted summary table
  global_equity_sweep.log          — full run log

RUNTIME ESTIMATE
----------------
  ~4-8h on a 12-core machine (16 configs × per-asset WF + 1000 MC samples)
  Run as an overnight batch job.

USER SETTINGS — see the block below before running.
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

# ============================================================
# PATH SETUP — assumes this script lives alongside the other
# development/ scripts so all imports resolve correctly.
# ============================================================
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
from global_equity_daily_output import build_daily_outputs
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

# Google Drive folder — required for both modes:
#   Mode A: STOXX 600 combined CSV
#   Mode B: MSCI World combined CSV
GDRIVE_FOLDER_ID_DEFAULT = ""   # <- paste your folder ID here

# FX treatment (applied equally across all configurations)
FX_HEDGED = True

# Position sizing mode
POSITION_MODE = "full"

# Volatility window (days)
VOL_WINDOW = 20

# Reallocation gate
COOLDOWN_DAYS = 10
ANNUAL_CAP    = 999   # no annual cap

# Allocation optimisation objective
ALLOC_OBJECTIVE = "calmar"
ALLOC_STEP      = 0.10

# Filter modes
FORCE_FILTER_MODE_EQ = None    # equity: auto
FORCE_FILTER_MODE_BD = ["ma"]  # bond: MA-only

# Parallelism
_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

# Monte Carlo iterations per asset per configuration
N_MC_SAMPLES = 500   # set to 10 for a smoke test

# Allocation weight perturbation steps
PERTURB_STEPS = [-0.2, -0.1, 0.0, 0.1, 0.2]

# Minimum mean OOS weight to include an asset in allocation perturbation
ALLOC_ROB_MIN_WEIGHT = 0.05

# MID/SMALL blend weights (Mode B)
MWIG_WEIGHT = 0.50
SWIG_WEIGHT = 0.50

# Data floor dates
WIG_DATA_FLOOR = "1994-10-03"
MMF_FLOOR      = "1994-10-03"

# Sweep dimensions
TRAIN_YEARS_LIST = [6, 7, 8, 9, 10, 12]
TEST_YEARS_LIST  = [1, 2]
MODES_LIST       = ["global_equity", "msci_world"]

# Equity parameter grids (same for all equity assets in both modes)
X_GRID_EQ = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ   = [50, 75, 100]
SLOW_EQ   = [150, 200, 250]
TV_EQ     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ = [126, 252]

# Bond parameter grids (TBSP, MA-only)
X_GRID_BD = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD = [0.001]
FAST_BD   = [50, 100, 150]
SLOW_BD   = [200, 300, 400, 500]
TV_BD     = [0.08, 0.10, 0.12]
SL_BD     = [0.01, 0.02, 0.03]

# Output files
CSV_OUTPUT = "global_equity_sweep_results.csv"
TXT_REPORT = "global_equity_sweep_report.txt"


# ============================================================
# PHASE 1 — DOWNLOAD ALL DATA ONCE
# ============================================================

def download_all_data() -> dict:
    """
    Download all price data needed for both modes.
    Returns a dict of raw DataFrames: keys are asset labels.
    Exits on any mandatory download failure.
    """
    logging.info("=" * 70)
    logging.info("DOWNLOADING ALL DATA")
    logging.info("=" * 70)

    tmp_dir = tempfile.gettempdir()
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()

    def _stooq(ticker: str, label: str, mandatory: bool = True):
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
                     label, len(df),
                     df.index.min().date(), df.index.max().date())
        return df

    raw = {}

    _stoxx_folder = (
    os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    or GDRIVE_FOLDER_ID_DEFAULT.strip())

    # Shared
    raw["TBSP"]   = build_and_upload(
          folder_id          = _stoxx_folder ,
          raw_filename       = "tbsp_extended_full.csv",
          combined_filename  = "tbsp_extended_combined.csv",
          extension_ticker   = "^tbsp",
          extension_source   = "stooq",
      )
    if raw["TBSP"] is None:
        logging.error(
            "FAIL: TBPSbuild/update failed. "
            "Ensure tbsp_extended_full.csv is uploaded to Drive folder %s.",
            _stoxx_folder,
        )
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "TBSP", len(raw["TBSP"]),
        raw["TBSP"].index.min().date(), raw["TBSP"].index.max().date(),
    )
    raw["MMF"]    = _stooq("2720.n",   "MMF")
    raw["WIBOR1M"]= _stooq("plopln1m", "WIBOR1M", mandatory=False)

    # FX
    raw["USDPLN"] = _stooq("usdpln", "USDPLN")
    raw["EURPLN"] = _stooq("eurpln", "EURPLN")
    raw["JPYPLN"] = _stooq("jpypln", "JPYPLN")

    # Mode A equities
    raw["WIG"]    = _stooq("wig",    "WIG")
    raw["SPX"]    = _stooq("^spx",   "SP500")
    raw["NKX"]    = _stooq("^nkx",    "Nikkei225")

    # STOXX 600: load from Drive (historical base + yfinance extension).
    # build_stoxx600() auto-selects first-build vs update mode:
    #   First build: reads stoxx600.csv from Drive, extends with yfinance
    #   Update:      reads stoxx600_combined.csv from Drive, extends forward
    
    if not _stoxx_folder:
        logging.error(
            "FAIL: GDRIVE_FOLDER_ID not set — cannot load STOXX 600. "
            "Set GDRIVE_FOLDER_ID_DEFAULT or the env var."
        )
        sys.exit(1)
    # STOXX 600 (yfinance ^STOXX extension, generic Date/Close format)
    raw["STOXX600"] = build_and_upload(
        folder_id          = _stoxx_folder,
        raw_filename       = "stoxx600.csv",
        combined_filename  = "stoxx600_combined.csv",
        extension_ticker   = "^STOXX",
    )
    if raw["STOXX600"] is None:
        logging.error(
            "FAIL: STOXX 600 build/update failed. "
            "Ensure stoxx600.csv is uploaded to Drive folder %s.",
            _stoxx_folder,
        )
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "STOXX600", len(raw["STOXX600"]),
        raw["STOXX600"].index.min().date(), raw["STOXX600"].index.max().date(),
    )

    # Mode B equities
    raw["WIG"] = _stooq("wig",  "WIG")
    
    # MSCI World combined series from Google Drive
    MSCIW_wsj = build_and_upload(
        folder_id         = _stoxx_folder ,
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
    raw["MSCIW"] = build_full_msci_world_extended(
        wsj_combined_df = MSCIW_wsj,
        folder_id       = (os.environ.get("GDRIVE_FOLDER_ID", "").strip()  
                           or GDRIVE_FOLDER_ID_DEFAULT.strip()),
        )
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "MSCI_World", len(raw["MSCIW"]),
        raw["MSCIW"].index.min().date(), raw["MSCIW"].index.max().date(),
        )

    logging.info("All data downloaded.")
    return raw


# ============================================================
# BUILD DERIVED SERIES (run once per session)
# ============================================================

def build_derived(raw: dict) -> dict:
    """
    Construct return series and price DataFrames from raw data.
    Returns a dict of processed series.
    """
    fx_usd = raw["USDPLN"][CLOSE_COL] if raw["USDPLN"] is not None else None
    fx_eur = raw["EURPLN"][CLOSE_COL] if raw["EURPLN"] is not None else None
    fx_jpy = raw["JPYPLN"][CLOSE_COL] if raw["JPYPLN"] is not None else None

    # WIG: clip to trading-continuous era
    wig = raw["WIG"].copy()
    wig = wig.loc[wig.index >= pd.Timestamp(WIG_DATA_FLOOR)]

    # Extended MMF
    mmf_ext = (build_mmf_extended(raw["MMF"], raw["WIBOR1M"], floor_date=MMF_FLOOR)
               if raw["WIBOR1M"] is not None else raw["MMF"])

    # Shared returns
    ret_tbsp = build_return_series(raw["TBSP"], hedged=True)
    ret_mmf  = build_return_series(mmf_ext,     hedged=True)

    # Mode A returns
    ret_WIG   = build_return_series(wig,          hedged=True)
    ret_SPX   = build_return_series(raw["SPX"],   fx_series=fx_usd, hedged=FX_HEDGED)
    ret_STOXX = build_return_series(raw["STOXX600"], fx_series=fx_eur, hedged=FX_HEDGED)
    ret_NKX   = build_return_series(raw["NKX"],   fx_series=fx_jpy, hedged=FX_HEDGED)

    # Mode A price DFs for walk_forward
    price_WIG   = wig          if FX_HEDGED else build_price_df_from_returns(ret_WIG,   "WIG_PLN")
    price_SPX   = raw["SPX"]   if FX_HEDGED else build_price_df_from_returns(ret_SPX,   "SP500_PLN")
    price_STOXX = raw["STOXX600"] if FX_HEDGED else build_price_df_from_returns(ret_STOXX, "STOXX600_PLN")
    price_NKX   = raw["NKX"]   if FX_HEDGED else build_price_df_from_returns(ret_NKX,   "Nikkei225_PLN")

    # Mode B returns 
    
    ret_MSCIW   = build_return_series(raw["MSCIW"],   fx_series=fx_usd, hedged=FX_HEDGED)

    price_MSCIW  = raw["MSCIW"]   if FX_HEDGED else build_price_df_from_returns(ret_MSCIW,  "MSCIW_PLN")

    return dict(
        # shared
        ret_tbsp=ret_tbsp, ret_mmf=ret_mmf,
        mmf_df=raw["MMF"],
        # mode A prices
        price_WIG=price_WIG, price_SPX=price_SPX,
        price_STOXX=price_STOXX, price_NKX=price_NKX,
        # mode A returns
        ret_WIG=ret_WIG, ret_SPX=ret_SPX,
        ret_STOXX=ret_STOXX, ret_NKX=ret_NKX,
        # mode B prices
        price_MSCIW=price_MSCIW,
        # mode B returns
        ret_MSCIW=ret_MSCIW,
    )


# ============================================================
# SINGLE CONFIGURATION RUN
# ============================================================

def run_one_config(
    mode:        str,
    train_years: int,
    test_years:  int,
    raw:         dict,
    derived:     dict,
    common_oos_start: pd.Timestamp,  # forced OOS start for comparability
) -> dict:
    """
    Run the full pipeline for one (mode, train, test) configuration.

    Returns a result dict with portfolio metrics, robustness verdicts,
    and per-asset B&H comparisons, all trimmed to common_oos_start.
    """
    label = f"{mode}  train={train_years}y  test={test_years}y"
    logging.info("")
    logging.info("=" * 70)
    logging.info("CONFIG: %s", label)
    logging.info("=" * 70)
    t0 = time.time()

    mmf_df   = raw["MMF"]
    ret_mmf  = derived["ret_mmf"]
    ret_tbsp = derived["ret_tbsp"]

    # ── Per-asset equity walk-forwards ────────────────────────────────────
    def _run_wf(price_df, lbl, fm_override, xg, yg, fg, sg, tvg, slg, mlg):
        eq, res, tr = walk_forward(
            df                    = price_df,
            cash_df               = mmf_df,
            train_years           = train_years,
            test_years            = test_years,
            vol_window            = VOL_WINDOW,
            selected_mode         = POSITION_MODE,
            filter_modes_override = fm_override,
            X_grid=xg, Y_grid=yg, fast_grid=fg, slow_grid=sg,
            tv_grid=tvg, sl_grid=slg, mom_lookback_grid=mlg,
            objective="calmar",
            n_jobs=N_JOBS,
        )
        sig = build_signal_series(eq, tr)
        m   = compute_metrics(eq)
        logging.info("  WF [%s]: CAGR=%.2f%% Sharpe=%.2f MaxDD=%.2f%% CalMAR=%.2f  "
                     "signal=%.1f%%",
                     lbl, m.get("CAGR",0)*100, m.get("Sharpe",0),
                     m.get("MaxDD",0)*100, m.get("CalMAR",0), sig.mean()*100)
        return eq, res, tr, sig

    if mode == "global_equity":
        wf_eq_WIG,   wf_res_WIG,   wf_tr_WIG,   sig_WIG   = _run_wf(
            derived["price_WIG"],   "WIG",       FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)
        wf_eq_SPX,   wf_res_SPX,   wf_tr_SPX,   sig_SPX   = _run_wf(
            derived["price_SPX"],   "SP500",     FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)
        wf_eq_STOXX, wf_res_STOXX, wf_tr_STOXX, sig_STOXX = _run_wf(
            derived["price_STOXX"], "STOXX600",  FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)
        wf_eq_NKX,   wf_res_NKX,   wf_tr_NKX,   sig_NKX   = _run_wf(
            derived["price_NKX"],   "Nikkei225", FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)

    elif mode == "msci_world":
        wf_eq_WIG,   wf_res_WIG,   wf_tr_WIG,   sig_WIG   = _run_wf(
            derived["price_WIG"],   "WIG",       FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)
        wf_eq_MSCIW, wf_res_MSCIW, wf_tr_MSCIW, sig_MSCIW = _run_wf(
            derived["price_MSCIW"],  "MSCI_World",FORCE_FILTER_MODE_EQ,
            X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ, TV_EQ, SL_EQ, MOM_LB_EQ)

    # ── TBSP bond walk-forward ────────────────────────────────────────────
    wf_eq_TBSP, wf_res_TBSP, wf_tr_TBSP = walk_forward(
        df=raw["TBSP"], cash_df=mmf_df,
        train_years=train_years, test_years=test_years,
        vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
        filter_modes_override=FORCE_FILTER_MODE_BD,
        X_grid=X_GRID_BD, Y_grid=Y_GRID_BD, fast_grid=FAST_BD,
        slow_grid=SLOW_BD, tv_grid=TV_BD, sl_grid=SL_BD,
        mom_lookback_grid=[252], objective="calmar", n_jobs=N_JOBS,
    )
    sig_TBSP = build_signal_series(wf_eq_TBSP, wf_tr_TBSP)
    m_tbsp   = compute_metrics(wf_eq_TBSP)
    logging.info("  WF [TBSP]: CAGR=%.2f%% signal=%.1f%%",
                 m_tbsp.get("CAGR",0)*100, sig_TBSP.mean()*100)

    # ── Assemble mode-specific dicts ──────────────────────────────────────
    if mode == "global_equity":
        asset_keys = ["WIG", "SP500", "STOXX600", "Nikkei225", "TBSP"]
        returns_dict = {
            "WIG":       derived["ret_WIG"].dropna(),
            "SP500":     derived["ret_SPX"].dropna(),
            "STOXX600":  derived["ret_STOXX"].dropna(),
            "Nikkei225": derived["ret_NKX"].dropna(),
            "TBSP":      ret_tbsp.dropna(),
        }
        signals_full = {
            "WIG":sig_WIG, "SP500":sig_SPX, "STOXX600":sig_STOXX,
            "Nikkei225":sig_NKX, "TBSP":sig_TBSP,
        }
        wf_results_ref = wf_res_TBSP
        mc_assets = [
            ("WIG",       derived["price_WIG"],   wf_eq_WIG,   wf_res_WIG,   EQUITY_THRESHOLDS_MC),
            ("SP500",     derived["price_SPX"],   wf_eq_SPX,   wf_res_SPX,   EQUITY_THRESHOLDS_MC),
            ("STOXX600",  derived["price_STOXX"], wf_eq_STOXX, wf_res_STOXX, EQUITY_THRESHOLDS_MC),
            ("Nikkei225", derived["price_NKX"],   wf_eq_NKX,   wf_res_NKX,   EQUITY_THRESHOLDS_MC),
            ("TBSP",      raw["TBSP"],            wf_eq_TBSP,  wf_res_TBSP,  BOND_THRESHOLDS_MC),
        ]
    else:
        asset_keys = ["WIG", "MSCI_World", "TBSP"]
        returns_dict = {
            "WIG":    derived["ret_WIG"].dropna(),
            
            "MSCI_World": derived["ret_MSCIW"].dropna(),
            "TBSP":       ret_tbsp.dropna(),
        }
        signals_full = {
            "WIG":sig_WIG, 
            "MSCI_World":sig_MSCIW, "TBSP":sig_TBSP,
        }
        wf_results_ref = wf_res_TBSP
        mc_assets = [
            ("WIG",    derived["price_WIG"],  wf_eq_WIG, wf_res_WIG, EQUITY_THRESHOLDS_MC),
            ("MSCI_World", derived["price_MSCIW"],  wf_eq_MSCIW, wf_res_MSCIW, EQUITY_THRESHOLDS_MC),
            ("TBSP",       raw["TBSP"],             wf_eq_TBSP,  wf_res_TBSP,  BOND_THRESHOLDS_MC),
        ]

    # ── Allocation walk-forward ───────────────────────────────────────────
    portfolio_equity, weights_series, realloc_log, alloc_df = allocation_walk_forward_n(
        returns_dict      = returns_dict,
        signals_full_dict = signals_full,
        signals_oos_dict  = dict(signals_full),
        mmf_returns       = ret_mmf.dropna(),
        wf_results_ref    = wf_results_ref,
        asset_keys        = asset_keys,
        step              = ALLOC_STEP,
        objective         = ALLOC_OBJECTIVE,
        cooldown_days     = COOLDOWN_DAYS,
        annual_cap        = ANNUAL_CAP,
        train_years       = train_years,
    )

    if portfolio_equity is None or portfolio_equity.empty:
        logging.warning("Empty portfolio equity — skipping this config.")
        return None

    # ── Trim everything to the common OOS start ───────────────────────────
    # The common OOS start is the latest OOS start across all configs within
    # the same mode.  Trim portfolio equity, signals, and alloc_df accordingly.
    oos_end = portfolio_equity.index.max()

    # Trim portfolio equity
    port_eq_trimmed = portfolio_equity.loc[
        (portfolio_equity.index >= common_oos_start) &
        (portfolio_equity.index <= oos_end)
    ]
    if port_eq_trimmed.empty:
        logging.warning("Portfolio equity is empty after trimming to common OOS start %s — skipping.",
                        common_oos_start.date())
        return None

    # Re-normalise to 1.0 at the common OOS start
    port_eq_trimmed = port_eq_trimmed / port_eq_trimmed.iloc[0]
    port_metrics = {k: float(v) for k, v in compute_metrics(port_eq_trimmed).items()}

    logging.info("  Portfolio [trimmed to %s–%s]: CAGR=%.2f%% Sharpe=%.2f MaxDD=%.2f%% CalMAR=%.2f",
                 common_oos_start.date(), oos_end.date(),
                 port_metrics.get("CAGR",0)*100, port_metrics.get("Sharpe",0),
                 port_metrics.get("MaxDD",0)*100, port_metrics.get("CalMAR",0))

    # Trim signals to common OOS period
    sigs_trimmed = {
        k: s.loc[(s.index >= common_oos_start) & (s.index <= oos_end)]
        for k, s in signals_full.items()
    }

    # ── Buy-and-hold benchmarks (trimmed) ────────────────────────────────
    bh_metrics = {}
    for key, ret in returns_dict.items():
        r_oos = ret.loc[(ret.index >= common_oos_start) & (ret.index <= oos_end)]
        if r_oos.empty:
            continue
        bh_eq = (1 + r_oos).cumprod()
        bh_eq = bh_eq / bh_eq.iloc[0]
        bh_metrics[key] = {k: float(v) for k, v in compute_metrics(bh_eq).items()}

    # ── Monte Carlo robustness (per asset) ───────────────────────────────
    mc_verdicts = {}   # {asset_label: {"verdict": str, "p05_cagr": float, ...}}
    for lbl, price_df, wf_eq, wf_res, thr_mc in mc_assets:
        logging.info("  MC [%s] starting (%d samples) ...", lbl, N_MC_SAMPLES)
        try:
            _wins    = extract_windows_from_wf_results(wf_res, train_years=train_years)
            _params  = extract_best_params_from_wf_results(wf_res)
            _mc_raw  = run_monte_carlo_robustness(
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
                "p05_cagr":  _summary.get("CAGR",{}).get("p05", float("nan")),
                "p05_maxdd": _summary.get("MaxDD",{}).get("p05", float("nan")),
                "p05_sharpe":_summary.get("Sharpe",{}).get("p05", float("nan")),
                "p_loss":    _summary.get("p_loss", float("nan")),
                "median_cagr":_summary.get("CAGR",{}).get("median", float("nan")),
                # key fail reasons (first failing gate only, for compactness)
                "fail_detail": "; ".join(
                    k for k, v in {
                        "p05_CAGR<0":    _summary.get("CAGR",{}).get("p05",0) < thr_mc.get("CAGR",{}).get("p05_min",0),
                        "p05_Sharpe<0":  _summary.get("Sharpe",{}).get("p05",0) < thr_mc.get("Sharpe",{}).get("p05_min",0),
                        "p05_MaxDD<thr": _summary.get("MaxDD",{}).get("p05",0) < thr_mc.get("MaxDD",{}).get("p05_min",-1),
                        "p_loss>10%":    _summary.get("p_loss",0) > 0.10,
                        "outlier_CAGR":  _summary.get("CAGR",{}).get("p_worse_than_baseline",0) > 0.85,
                        "low_median":    _summary.get("CAGR",{}).get("median",1) < 0.02,
                    }.items() if v
                ) or "—",
            }
            logging.info("  MC [%s]: %s", lbl, mc_verdicts[lbl]["verdict"])
        except Exception as exc:
            logging.error("  MC [%s] failed: %s", lbl, exc)
            mc_verdicts[lbl] = {"verdict": "ERROR", "fail_detail": str(exc),
                                 "p05_cagr": float("nan")}

    # ── Allocation weight perturbation ───────────────────────────────────
    w_cols = [c for c in alloc_df.columns if c.startswith("w_") and c != "w_mmf"]
    focus_assets = [c[2:] for c in w_cols
                    if alloc_df[c].mean() >= ALLOC_ROB_MIN_WEIGHT]
    alloc_rob = {}   # {asset: {"verdict": str, "min_calmar_ratio": float}}

    for focus in focus_assets:
        if focus not in asset_keys:
            continue
        try:
            rob_df = allocation_weight_robustness_n(
                alloc_results_df = alloc_df,
                returns_dict     = returns_dict,
                mmf_returns      = ret_mmf,
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
            verdict = ("PLATEAU" if min_ratio >= 0.80
                       else "MODERATE" if min_ratio >= 0.60
                       else "SPIKE" if not np.isnan(min_ratio)
                       else "N/A")
            alloc_rob[focus] = {"verdict": verdict, "min_calmar_ratio": min_ratio}
            logging.info("  Alloc rob [%s]: %s (ratio %.3f)", focus, verdict, min_ratio)
        except Exception as exc:
            logging.error("  Alloc rob [%s] failed: %s", focus, exc)
            alloc_rob[focus] = {"verdict": "ERROR", "min_calmar_ratio": float("nan")}

    elapsed = time.time() - t0
    logging.info("Config done: %s  (%.0f min)", label, elapsed / 60)

    return {
        "mode":              mode,
        "train_years":       train_years,
        "test_years":        test_years,
        "oos_start":         str(common_oos_start.date()),
        "oos_end":           str(oos_end.date()),
        "n_reallocations":   len(realloc_log),
        # Portfolio metrics
        "port_cagr":         port_metrics.get("CAGR"),
        "port_vol":          port_metrics.get("Vol"),
        "port_sharpe":       port_metrics.get("Sharpe"),
        "port_maxdd":        port_metrics.get("MaxDD"),
        "port_calmar":       port_metrics.get("CalMAR"),
        "port_sortino":      port_metrics.get("Sortino"),
        # B&H per asset
        "bh_metrics":        bh_metrics,
        # MC verdicts
        "mc_verdicts":       mc_verdicts,
        # Allocation robustness
        "alloc_rob":         alloc_rob,
        # Mean weights per asset
        "mean_weights":      {c[2:]: float(alloc_df[c].mean()) for c in w_cols},
    }


# ============================================================
# OOS START DISCOVERY  (first pass, no MC)
# ============================================================

def discover_oos_starts(raw: dict, derived: dict) -> dict:
    """
    Run a minimal walk-forward (just TBSP + one equity per mode) for each
    (mode, train, test) combination to find the OOS start date.
    Returns {mode: common_oos_start_timestamp}.
    """
    logging.info("=" * 70)
    logging.info("DISCOVERING OOS START DATES (first pass)")
    logging.info("=" * 70)

    oos_starts_by_mode = {m: [] for m in MODES_LIST}

    for mode in MODES_LIST:
        for train_years in TRAIN_YEARS_LIST:
            for test_years in TEST_YEARS_LIST:
                # Run TBSP WF only — cheapest, always the binding constraint
                eq, res, _ = walk_forward(
                    df=raw["TBSP"], cash_df=raw["MMF"],
                    train_years=train_years, test_years=test_years,
                    vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
                    filter_modes_override=FORCE_FILTER_MODE_BD,
                    X_grid=X_GRID_BD, Y_grid=Y_GRID_BD, fast_grid=FAST_BD,
                    slow_grid=SLOW_BD, tv_grid=TV_BD, sl_grid=SL_BD,
                    mom_lookback_grid=[252], objective="calmar",
                    n_jobs=N_JOBS,  # use max possible cores for speed
                )
                if not eq.empty:
                    # In Mode B, the effective OOS start is constrained further
                    # by MSCI World data start.  The allocation walk-
                    # forward will automatically clip to the intersection, so the
                    # effective OOS start is the allocation portfolio's first date.
                    # Approximate it here as max(TBSP_oos_start, mode_data_floor).
                    tbsp_oos_start = res["TestStart"].min()

                    if mode == "msci_world":
                        # MSCI World starts ~2010, PL_MID starts ~2005
                        # Add 1 training window to the shortest series start
                        msciw_start = (derived["ret_MSCIW"].dropna().index.min()
                                       + pd.DateOffset(years=train_years))
                        
                        effective_start = max(tbsp_oos_start, msciw_start )
                    else:
                        effective_start = tbsp_oos_start

                    oos_starts_by_mode[mode].append(effective_start)
                    logging.info("  %s  train=%dy test=%dy  ->  OOS start %s",
                                 mode, train_years, test_years, effective_start.date())

    # Common OOS start per mode = latest OOS start among all configs
    result = {}
    for mode in MODES_LIST:
        starts = oos_starts_by_mode[mode]
        if starts:
            result[mode] = max(starts)
            logging.info("COMMON OOS START  [%s]: %s", mode, result[mode].date())
        else:
            logging.error("No OOS starts found for mode %s", mode)
            result[mode] = pd.Timestamp("2030-01-01")  # sentinel — will fail gracefully

    return result


# ============================================================
# RESULTS FORMATTING
# ============================================================

def _pct(v):
    """Format a float as percentage string for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.1f}%"

def _f2(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.2f}"


def format_report(results: list) -> str:
    """
    Build a formatted multi-section text report.

    Sections:
      1. Summary table (one row per config)
      2. Per-mode detailed tables with B&H comparisons
      3. MC robustness verdict grid
      4. Allocation weight robustness verdict grid
    """
    lines = []
    sep  = "=" * 130
    sep2 = "-" * 130

    lines += [
        sep,
        "GLOBAL EQUITY WALK-FORWARD WINDOW SWEEP",
        f"Run date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"MC samples per asset: {N_MC_SAMPLES}",
        f"FX treatment: {'hedged' if FX_HEDGED else 'unhedged'}",
        sep,
        "",
    ]

    # ── Section 1: Summary table ──────────────────────────────────────────
    lines += [
        "SECTION 1: PORTFOLIO OOS PERFORMANCE SUMMARY",
        "All metrics are computed over the common OOS period for each mode.",
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

    # ── Section 2: Per-mode B&H comparison ───────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue

        asset_keys = list(mode_results[0]["bh_metrics"].keys()) if mode_results else []
        lines += [
            f"SECTION 2: B&H COMPARISON — {mode.upper()}",
            sep2,
        ]

        # Header: portfolio CAGR vs each asset B&H CAGR
        bh_header = f"{'Train':>5} {'Test':>4} | {'Port CAGR':>10} {'Port CalMAR':>12} | "
        bh_header += " | ".join(f"{k[:8]:>9} B&H" for k in asset_keys)
        lines.append(bh_header)
        lines.append("-" * len(bh_header))

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

    # ── Section 3: MC robustness verdicts ─────────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue

        all_mc_assets = sorted({k for r in mode_results for k in r.get("mc_verdicts", {})})
        lines += [
            f"SECTION 3: MC PARAMETER ROBUSTNESS — {mode.upper()}",
            f"  ROBUST = passes all gates  |  FRAGILE = one or more gates failed",
            sep2,
        ]
        hdr3 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:10]:>11}" for a in all_mc_assets)
        lines.append(hdr3)
        lines.append("-" * len(hdr3))

        for r in mode_results:
            cells = []
            for a in all_mc_assets:
                v = r.get("mc_verdicts", {}).get(a, {})
                verdict = v.get("verdict", "N/A")
                p05 = v.get("p05_cagr", float("nan"))
                tag  = f"{'R' if verdict=='ROBUST' else 'F' if verdict=='FRAGILE' else verdict[:3]}"
                cells.append(f"{tag}({_pct(p05)})"[:11].rjust(11))
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | " + " | ".join(cells))

        lines += ["", ""]

        # Fail details (separate block for readability)
        lines.append(f"  Failure reasons (FRAGILE assets only):")
        for r in mode_results:
            for a in all_mc_assets:
                v = r.get("mc_verdicts", {}).get(a, {})
                if v.get("verdict") == "FRAGILE":
                    lines.append(
                        f"    train={r['train_years']}y test={r['test_years']}y "
                        f"[{a}]: {v.get('fail_detail','')}"
                    )
        lines += ["", ""]

    # ── Section 4: Allocation robustness verdicts ─────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue

        all_alloc_assets = sorted({k for r in mode_results for k in r.get("alloc_rob", {})})
        lines += [
            f"SECTION 4: ALLOCATION WEIGHT ROBUSTNESS — {mode.upper()}",
            f"  PLATEAU >= 0.80 CalMAR ratio at ±10pp  |  SPIKE < 0.60",
            sep2,
        ]
        hdr4 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:10]:>13}" for a in all_alloc_assets)
        lines.append(hdr4)
        lines.append("-" * len(hdr4))

        for r in mode_results:
            cells = []
            for a in all_alloc_assets:
                v = r.get("alloc_rob", {}).get(a, {})
                verdict   = v.get("verdict", "—")
                ratio     = v.get("min_calmar_ratio", float("nan"))
                ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
                tag = f"{verdict[:3]}({ratio_str})"[:13].rjust(13)
                cells.append(tag)
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | " + " | ".join(cells))

        lines += ["", ""]

    # ── Section 5: Mean allocation weights ────────────────────────────────
    for mode in MODES_LIST:
        mode_results = [r for r in results if r and r["mode"] == mode]
        if not mode_results:
            continue

        all_w_assets = sorted({k for r in mode_results for k in r.get("mean_weights", {})})
        lines += [
            f"SECTION 5: MEAN ALLOCATION WEIGHTS (across OOS windows) — {mode.upper()}",
            sep2,
        ]
        hdr5 = f"{'Train':>5} {'Test':>4} | " + " | ".join(f"{a[:8]:>9}" for a in all_w_assets)
        lines.append(hdr5)
        lines.append("-" * len(hdr5))
        for r in mode_results:
            cells = " | ".join(
                f"{_pct(r['mean_weights'].get(a, 0.0)):>9}"
                for a in all_w_assets
            )
            lines.append(f"{r['train_years']:>5} {r['test_years']:>4} | {cells}")
        lines += ["", ""]

    lines.append(sep)
    lines.append("END OF REPORT")
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
    n_configs = len(MODES_LIST) * len(TRAIN_YEARS_LIST) * len(TEST_YEARS_LIST)
    logging.info("Total configurations: %d", n_configs)
    logging.info("=" * 70)

    # Step 1: Download everything once
    raw     = download_all_data()
    derived = build_derived(raw)

    # Step 2: Discover common OOS start per mode
    common_oos_starts = discover_oos_starts(raw, derived)

    # Step 3: Run all configurations
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
                    mode         = mode,
                    train_years  = train_years,
                    test_years   = test_years,
                    raw          = raw,
                    derived      = derived,
                    common_oos_start = common_start,
                )
                all_results.append(result)

    # Step 4: Write CSV
    valid = [r for r in all_results if r is not None]

    csv_cols = [
        "mode", "train_years", "test_years", "oos_start", "oos_end",
        "n_reallocations",
        "port_cagr", "port_vol", "port_sharpe", "port_maxdd",
        "port_calmar", "port_sortino",
    ]
    # Dynamically add B&H CAGR and Sharpe per asset, MC verdicts and p05_cagr per asset,
    # alloc rob verdict per asset, mean weights per asset
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
        for c in ["mode","train_years","test_years","oos_start","oos_end",
                  "n_reallocations","port_cagr","port_vol","port_sharpe",
                  "port_maxdd","port_calmar","port_sortino"]:
            v = r.get(c, "")
            row[c] = f"{v:.6f}" if isinstance(v, float) else v
        for k, m in r.get("bh_metrics", {}).items():
            if f"bh_{k}_cagr" in row:
                row[f"bh_{k}_cagr"]   = f"{m.get('CAGR', ''):.6f}" if "CAGR" in m else ""
                row[f"bh_{k}_sharpe"] = f"{m.get('Sharpe',''):.4f}" if "Sharpe" in m else ""
        for k, v in r.get("mc_verdicts", {}).items():
            if f"mc_{k}_verdict" in row:
                row[f"mc_{k}_verdict"]   = v.get("verdict","")
                row[f"mc_{k}_p05cagr"]   = f"{v.get('p05_cagr',''):.4f}" if "p05_cagr" in v else ""
                row[f"mc_{k}_faildetail"]= v.get("fail_detail","")
        for k, v in r.get("alloc_rob", {}).items():
            if f"alloc_{k}_verdict" in row:
                row[f"alloc_{k}_verdict"] = v.get("verdict","")
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

    # Step 5: Write text report
    report_text = format_report(valid)
    with open(TXT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info("Text report written: %s", TXT_REPORT)

    # Print report to console as well
    print("\n" + report_text)

    logging.info("=" * 70)
    logging.info("SWEEP COMPLETE: %s", dt.datetime.now())
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
