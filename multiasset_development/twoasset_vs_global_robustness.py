"""
twoasset_vs_global_robustness.py
=================================
Full robustness comparison between:
  (A) Production 2-asset strategy (WIG + TBSP + MMF)
  (B) A candidate Mode A or Mode B global equity config

For each strategy the script runs:
  Level 1a: MC parameter perturbation — WIG signal              (n=N_MC)
  Level 1b: MC parameter perturbation — TBSP signal             (n=N_MC)
  Level 1c: MC parameter perturbation — MSCI World / SP500 etc. (n=N_MC, Mode B/A only)
  Level 2a: Block bootstrap — WIG                               (n=N_BOOTSTRAP)
  Level 2b: Block bootstrap — TBSP                              (n=N_BOOTSTRAP)
  Level 2c: Block bootstrap — per-asset for candidate only      (n=N_BOOTSTRAP)
  Level 3:  Allocation weight perturbation (±10pp, ±20pp)

OOS PERIOD
----------
Both strategies are trimmed to the same common OOS window so metrics
are directly comparable. The start date is the LATER of each strategy's
natural OOS start (driven by data availability + train window), unless
FORCE_OOS_START is set explicitly.

DATA
----
Production 2-asset:   WIG, TBSP (extended via Drive), MMF, WIBOR1M, PL10Y, DE10Y
Mode B candidate:     above + MSCI World (Drive combined + synthetic backcast)
Mode A candidate:     above + SP500, STOXX600 (Drive), Nikkei225

USAGE
-----
  # Run with defaults (Mode B, 7+2 windows, n_mc=1000, n_bootstrap=500):
  python twoasset_vs_global_robustness.py

  # Quick smoke test:
  N_MC=10 N_BOOTSTRAP=10 python twoasset_vs_global_robustness.py

  # Override OOS start to match sweep period:
  # Set FORCE_OOS_START = "2011-01-04" in USER SETTINGS below

RUNTIME ESTIMATE (12-core machine)
------------------------------------
  n_mc=1000, n_bootstrap=500: ~8-12h (both strategies × all assets × both tests)
  n_mc=100,  n_bootstrap=50:  ~1-2h
  n_mc=10,   n_bootstrap=10:  ~15min (smoke test)
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

# ── Imports from strategy library ───────────────────────────────────────────

from strategy_test_library import (
    download_csv, load_csv,
    walk_forward, compute_metrics, compute_buy_and_hold,
    analyze_trades, print_backtest_report,
)
from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    print_multiasset_report,
    allocation_weight_robustness,
    print_allocation_robustness_report,
    build_spread_prefilter,
    build_yield_momentum_prefilter,
)
from global_equity_library import (
    build_mmf_extended,
    build_return_series,
    allocation_walk_forward_n,
    allocation_weight_robustness_n,
    print_allocation_robustness_report_n,
    CLOSE_COL,
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
from price_series_builder import build_and_upload, load_combined_from_drive
from msci_world_synthetic import (
    build_full_msci_world_extended,
    load_synthetic_from_drive,
)


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE = "twoasset_vs_global_robustness.log"
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
# USER SETTINGS — edit these before running
# ============================================================

# ── Candidate strategy mode ──────────────────────────────────────────────────
# "msci_world"   : Mode B  — WIG + MSCI_World + TBSP + MMF
# "global_equity": Mode A  — WIG + SP500 + STOXX600 + Nikkei225 + TBSP + MMF
CANDIDATE_MODE = "msci_world"

# ── Walk-forward window lengths ──────────────────────────────────────────────
# Production 2-asset uses these (should match freeze_0_5 settings)
TRAIN_YEARS_EQ = 7
TEST_YEARS_EQ  = 2
TRAIN_YEARS_BD = 7
TEST_YEARS_BD  = 2

# Candidate global equity config to evaluate
# These should match the config selected from global_equity_sweep_results.csv
CANDIDATE_TRAIN_YEARS = 7
CANDIDATE_TEST_YEARS  = 2

# ── OOS period ───────────────────────────────────────────────────────────────
# Set to None to use the natural OOS start (max of both strategies' starts).
# Set to a date string (e.g. "2011-01-04") to force a common start.
# Note: forcing an early start requires extended TBSP and MMF data on Drive.
FORCE_OOS_START = "2008-01-04"   # None = auto-detect

# ── Robustness iterations ────────────────────────────────────────────────────
N_MC        = int(os.getenv("N_MC",        "1000"))  # 10 for smoke test 1000 full
N_BOOTSTRAP = int(os.getenv("N_BOOTSTRAP", "100"))   # 10 for smoke test 500 full

RUN_MC        = True   # MC parameter perturbation
RUN_BOOTSTRAP = True   # Block bootstrap (slow — set False to skip)

# ── Strategy settings — must match production freeze ─────────────────────────
POSITION_MODE        = "full"
VOL_WINDOW           = 20
FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]
COOLDOWN_DAYS        = 10
ANNUAL_CAP           = 12
ALLOC_OBJECTIVE      = "calmar"
YIELD_PREFILTER_BP   = 50.0
FAST_MODE            = True

# ── ATR stop settings ────────────────────────────────────────────────────────
USE_ATR_STOP    = True
ATR_WINDOW      = 20
N_ATR_GRID      = [0.08, 0.10, 0.12, 0.15, 0.20]

USE_ATR_STOP_BD = False
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [0.05, 0.08, 0.10, 0.15]

# ── Parameter grids (2-asset production, unchanged from freeze_0_5) ──────────
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

# ── Candidate equity asset grids (same as sweep) ─────────────────────────────
X_GRID_CAND = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_CAND = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_CAND   = [50, 75, 100]
SLOW_CAND   = [150, 200, 250]
TV_CAND     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_CAND     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_CAND = [126, 252]

# ── Allocation robustness perturbation steps ─────────────────────────────────
ALLOC_PERTURB_STEPS = [-0.2, -0.1, 0.0, 0.1, 0.2]

# ── Data floors ───────────────────────────────────────────────────────────────
WIG_DATA_FLOOR = "1995-01-02"
MMF_FLOOR      = "1995-01-02"
DATA_START     = "1990-01-01"

# ── Parallelism ───────────────────────────────────────────────────────────────
_cpu = os.cpu_count() or 1
N_JOBS = max(1, _cpu - 1) if _cpu > 3 and sys.platform == "win32" else _cpu


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

def download_all_data() -> dict:
    logging.info("=" * 80)
    logging.info("PHASE 1: DOWNLOADING DATA")
    logging.info("=" * 80)

    tmp = tempfile.mkdtemp()
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()

    def _stooq(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame | None:
        """
        Download a stooq series, clip to DATA_START.
        If download fails, fall back to loading CSV from current working directory.
        """
        url         = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        tmp_path    = os.path.join(tmp, f"ge_{label}.csv")
        #local_path  = os.path.join(os.getcwd(), f"{label}.csv")  # fallback filename
        local_path = os.path.join(os.getcwd(), "data", f"{label}.csv")
        # === Attempt online download ===
        ok = download_csv(url, tmp_path)
        if ok:
            logging.info("INFO: %s — downloaded from Stooq", label)
            df = load_csv(tmp_path)
        else:
            logging.warning("WARN: %s — download failed, trying local CSV: %s", label, local_path)

        # === Fallback: try loading local file ===
        if os.path.exists(local_path):
            df = load_csv(local_path)
            if df is not None:
                logging.info("INFO: %s — loaded from local CSV", label)
            else:
                logging.error("FAIL: %s — local CSV exists but load_csv returned None", label)
                if mandatory:
                    sys.exit(1)
                    return None
                else:
                    # No fallback available
                    if mandatory:
                        logging.error("FAIL: %s — neither online nor local CSV available, exiting.", label)
                        sys.exit(1)
                    logging.warning("WARN: %s — optional series missing, skipping.", label)
                    return None

            # === Post-loading logic ===
            df = df.loc[df.index >= pd.Timestamp(DATA_START)]

            logging.info(
                "OK  : %-14s  %5d rows  %s to %s",
                label, len(df), df.index.min().date(), df.index.max().date(),
            )
            return df

    raw = {}

    # Mandatory for both strategies
    raw["WIG"]     = _stooq("wig",       "WIG")
    raw["WIG"]     = raw["WIG"].loc[raw["WIG"].index >= pd.Timestamp(WIG_DATA_FLOOR)]
    raw["MMF"]     = _stooq("2720.n",    "MMF")
    raw["WIBOR1M"] = _stooq("plopln1m",  "WIBOR1M", mandatory=False)
    raw["PL10Y"]   = _stooq("10yply.b",  "PL10Y")
    raw["DE10Y"]   = _stooq("10ydey.b",  "DE10Y")

    # Extended TBSP via Drive
    raw["TBSP"] = build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "tbsp_extended_full.csv",
        combined_filename = "tbsp_extended_combined.csv",
        extension_ticker  = "^tbsp",
        extension_source  = "stooq",
    )
    if raw["TBSP"] is None:
        raw["TBSP"] = _stooq("^tbsp", "TBSP")
        if raw["TBSP"] is None:
            logging.error("FAIL: TBSP unavailable."); sys.exit(1)
    for col in ["Najwyzszy", "Najnizszy"]:
        if col not in raw["TBSP"].columns:
            raw["TBSP"][col] = raw["TBSP"][CLOSE_COL]
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "TBSP", len(raw["TBSP"]),
                 raw["TBSP"].index.min().date(), raw["TBSP"].index.max().date())

    # Candidate-mode-specific data
    if CANDIDATE_MODE == "msci_world":
        logging.info("Loading Mode B assets (MSCI World)...")
        raw["USDPLN"] = _stooq("usdpln", "USDPLN")
        msciw_wsj = build_and_upload(
            folder_id         = folder_id,
            raw_filename      = "msci_world_wsj_raw.csv",
            combined_filename = "msci_world_combined.csv",
            extension_ticker  = "URTH",
        )
        if msciw_wsj is None:
            logging.error("FAIL: MSCI World (WSJ+URTH) unavailable."); sys.exit(1)
        synthetic = load_synthetic_from_drive(folder_id=folder_id)
        raw["MSCIW"] = build_full_msci_world_extended(
            wsj_combined_df = msciw_wsj,
            synthetic_df    = synthetic,
            folder_id       = folder_id,
        )
        if raw["MSCIW"] is None:
            raw["MSCIW"] = msciw_wsj
        logging.info("OK  %-14s  %5d rows  %s to %s",
                     "MSCI_World", len(raw["MSCIW"]),
                     raw["MSCIW"].index.min().date(), raw["MSCIW"].index.max().date())

    elif CANDIDATE_MODE == "global_equity":
        logging.info("Loading Mode A assets (SP500, STOXX600, Nikkei225)...")
        raw["USDPLN"] = _stooq("usdpln", "USDPLN")
        raw["EURPLN"] = _stooq("eurpln", "EURPLN")
        raw["JPYPLN"] = _stooq("jpypln", "JPYPLN")
        raw["SPX"]    = _stooq("^spx",   "SP500")
        raw["NKX"]    = _stooq("^nkx",   "Nikkei225")
        raw["STOXX600"] = build_and_upload(
            folder_id         = folder_id,
            raw_filename      = "stoxx600.csv",
            combined_filename = "stoxx600_combined.csv",
            extension_ticker  = "^STOXX",
        )
        if raw["STOXX600"] is None:
            logging.error("FAIL: STOXX600 unavailable."); sys.exit(1)
        logging.info("OK  %-14s  %5d rows  %s to %s",
                     "STOXX600", len(raw["STOXX600"]),
                     raw["STOXX600"].index.min().date(), raw["STOXX600"].index.max().date())
    else:
        logging.error("Unknown CANDIDATE_MODE: %s", CANDIDATE_MODE); sys.exit(1)

    return raw


# ============================================================
# PHASE 2 — DERIVED SERIES
# ============================================================

def build_derived(raw: dict) -> dict:
    logging.info("=" * 80)
    logging.info("PHASE 2: BUILDING DERIVED SERIES")
    logging.info("=" * 80)

    # Extended MMF
    if raw.get("WIBOR1M") is not None:
        mmf_ext = build_mmf_extended(raw["MMF"], raw["WIBOR1M"], floor_date=MMF_FLOOR)
        logging.info("MMF extended back to %s", mmf_ext.index.min().date())
    else:
        mmf_ext = raw["MMF"]
        logging.warning("WIBOR1M unavailable — using standard MMF (starts ~1999)")

    # Bond entry gate
    spread_pf = build_spread_prefilter(raw["PL10Y"], raw["DE10Y"])
    yield_pf  = build_yield_momentum_prefilter(
        raw["PL10Y"], rise_threshold_bp=YIELD_PREFILTER_BP, lookback_days=63
    )
    tbsp_idx    = raw["TBSP"].index
    spread_gate = spread_pf.reindex(tbsp_idx, method="ffill").fillna(1).astype(int)
    yield_gate  = yield_pf.reindex(tbsp_idx,  method="ffill").fillna(1).astype(int)
    bond_gate   = (spread_gate & yield_gate).astype(int)
    bond_gate.name = "bond_entry_gate"
    logging.info(
        "Bond entry gate: %.1f%% of days pass  (spread %.1f%% | yield %.1f%%)",
        bond_gate.mean()*100, spread_gate.mean()*100, yield_gate.mean()*100,
    )

    # Return series
    ret_eq  = raw["WIG"][CLOSE_COL].pct_change().dropna()
    ret_bd  = raw["TBSP"][CLOSE_COL].pct_change().dropna()
    ret_mmf = mmf_ext[CLOSE_COL].pct_change().dropna()

    derived = dict(
        mmf_ext   = mmf_ext,
        bond_gate = bond_gate,
        ret_eq    = ret_eq,
        ret_bd    = ret_bd,
        ret_mmf   = ret_mmf,
        spread_pf = spread_pf,
        yield_pf  = yield_pf,
    )

    # Candidate-specific return series
    fx_usd = raw.get("USDPLN", pd.DataFrame())[CLOSE_COL] if "USDPLN" in raw else None

    if CANDIDATE_MODE == "msci_world":
        ret_msciw = build_return_series(raw["MSCIW"], fx_series=fx_usd, hedged=True)
        derived["ret_MSCIW"] = ret_msciw
        derived["price_MSCIW"] = raw["MSCIW"]

    elif CANDIDATE_MODE == "global_equity":
        fx_eur = raw.get("EURPLN", pd.DataFrame())[CLOSE_COL] if "EURPLN" in raw else None
        fx_jpy = raw.get("JPYPLN", pd.DataFrame())[CLOSE_COL] if "JPYPLN" in raw else None
        derived["ret_SPX"]    = build_return_series(raw["SPX"],     fx_series=fx_usd, hedged=True)
        derived["ret_STOXX"]  = build_return_series(raw["STOXX600"],fx_series=fx_eur, hedged=True)
        derived["ret_NKX"]    = build_return_series(raw["NKX"],     fx_series=fx_jpy, hedged=True)
        derived["price_SPX"]    = raw["SPX"]
        derived["price_STOXX"]  = raw["STOXX600"]
        derived["price_NKX"]    = raw["NKX"]

    return derived


# ============================================================
# PHASE 3 — 2-ASSET PRODUCTION WALK-FORWARD
# ============================================================

def run_production_strategy(raw: dict, derived: dict) -> dict:
    logging.info("=" * 80)
    logging.info("PHASE 3: 2-ASSET PRODUCTION WALK-FORWARD (WIG + TBSP)")
    logging.info("train=%dy  test=%dy", TRAIN_YEARS_EQ, TEST_YEARS_EQ)
    logging.info("=" * 80)

    mmf_ext   = derived["mmf_ext"]
    bond_gate = derived["bond_gate"]

    # Equity WF
    logging.info("--- WIG walk-forward ---")
    wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
        df=raw["WIG"], cash_df=mmf_ext,
        train_years=TRAIN_YEARS_EQ, test_years=TEST_YEARS_EQ,
        vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
        filter_modes_override=FORCE_FILTER_MODE_EQ,
        X_grid=X_GRID_EQ, Y_grid=Y_GRID_EQ,
        fast_grid=FAST_EQ, slow_grid=SLOW_EQ,
        tv_grid=TV_EQ, sl_grid=SL_EQ,
        mom_lookback_grid=MOM_LB_EQ, objective=ALLOC_OBJECTIVE,
        n_jobs=N_JOBS, fast_mode=FAST_MODE,
        use_atr_stop=USE_ATR_STOP,
        N_atr_grid=N_ATR_GRID if USE_ATR_STOP else None,
        atr_window=ATR_WINDOW,
    )
    if wf_eq.empty:
        logging.error("Equity WF empty."); sys.exit(1)
    m_eq = compute_metrics(wf_eq)
    logging.info("WIG OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 m_eq["CAGR"]*100, m_eq["Sharpe"], m_eq["MaxDD"]*100, m_eq["CalMAR"])

    # Bond WF
    logging.info("--- TBSP walk-forward ---")
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
        df=raw["TBSP"], cash_df=mmf_ext,
        train_years=TRAIN_YEARS_BD, test_years=TEST_YEARS_BD,
        vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
        filter_modes_override=FORCE_FILTER_MODE_BD,
        X_grid=X_GRID_BD, Y_grid=Y_GRID_BD,
        fast_grid=FAST_BD, slow_grid=SLOW_BD,
        tv_grid=TV_BD, sl_grid=SL_BD,
        mom_lookback_grid=[252], objective=ALLOC_OBJECTIVE,
        n_jobs=N_JOBS, fast_mode=FAST_MODE,
        entry_gate_series=bond_gate,
        use_atr_stop=USE_ATR_STOP_BD,
        N_atr_grid=N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window=ATR_WINDOW_BD,
    )
    if wf_bd.empty:
        logging.error("Bond WF empty."); sys.exit(1)
    m_bd = compute_metrics(wf_bd)
    logging.info("TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 m_bd["CAGR"]*100, m_bd["Sharpe"], m_bd["MaxDD"]*100, m_bd["CalMAR"])

    # Signals
    sig_eq = build_signal_series(wf_eq, wf_tr_eq)
    sig_bd = build_signal_series(wf_bd, wf_tr_bd)

    # OOS bounds
    oos_start_eq = wf_res_eq["TestStart"].min()
    oos_start_bd = wf_res_bd["TestStart"].min()
    oos_end_eq   = wf_res_eq["TestEnd"].max()
    oos_end_bd   = wf_res_bd["TestEnd"].max()
    natural_start = max(oos_start_eq, oos_start_bd)
    natural_end   = min(oos_end_eq, oos_end_bd)
    logging.info("Production natural OOS: %s to %s",
                 natural_start.date(), natural_end.date())

    return dict(
        wf_eq=wf_eq, wf_res_eq=wf_res_eq, wf_tr_eq=wf_tr_eq,
        wf_bd=wf_bd, wf_res_bd=wf_res_bd, wf_tr_bd=wf_tr_bd,
        sig_eq=sig_eq, sig_bd=sig_bd,
        m_eq=m_eq, m_bd=m_bd,
        natural_start=natural_start, natural_end=natural_end,
    )


# ============================================================
# PHASE 4 — CANDIDATE GLOBAL EQUITY WALK-FORWARD
# ============================================================

def run_candidate_strategy(raw: dict, derived: dict, prod: dict) -> dict:
    logging.info("=" * 80)
    logging.info("PHASE 4: CANDIDATE %s WALK-FORWARD", CANDIDATE_MODE.upper())
    logging.info("train=%dy  test=%dy", CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS)
    logging.info("=" * 80)

    mmf_ext   = derived["mmf_ext"]
    bond_gate = derived["bond_gate"]
    ret_mmf   = derived["ret_mmf"]

    def _wf_equity_asset(price_df, label, mmf_df):
        logging.info("--- %s walk-forward ---", label)
        eq, res, tr = walk_forward(
            df=price_df, cash_df=mmf_df,
            train_years=CANDIDATE_TRAIN_YEARS,
            test_years=CANDIDATE_TEST_YEARS,
            vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
            filter_modes_override=FORCE_FILTER_MODE_EQ,
            X_grid=X_GRID_CAND, Y_grid=Y_GRID_CAND,
            fast_grid=FAST_CAND, slow_grid=SLOW_CAND,
            tv_grid=TV_CAND, sl_grid=SL_CAND,
            mom_lookback_grid=MOM_LB_CAND, objective=ALLOC_OBJECTIVE,
            n_jobs=N_JOBS, fast_mode=FAST_MODE,
            use_atr_stop=USE_ATR_STOP,
            N_atr_grid=N_ATR_GRID if USE_ATR_STOP else None,
            atr_window=ATR_WINDOW,
        )
        m = compute_metrics(eq) if not eq.empty else {}
        logging.info("%s OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                     label,
                     m.get("CAGR",0)*100, m.get("Sharpe",0),
                     m.get("MaxDD",0)*100, m.get("CalMAR",0))
        return eq, res, tr, m

    # WIG is always in the candidate
    wf_wig, wf_res_wig, wf_tr_wig, m_wig = _wf_equity_asset(raw["WIG"], "WIG", mmf_ext)

    # TBSP bond — same as production (reuse prod results for bond to avoid redundancy)
    # Bond WF is independent of equity mode, so we can re-run to get candidate's OOS schedule
    logging.info("--- TBSP walk-forward (candidate schedule) ---")
    wf_tbsp, wf_res_tbsp, wf_tr_tbsp = walk_forward(
        df=raw["TBSP"], cash_df=mmf_ext,
        train_years=CANDIDATE_TRAIN_YEARS,
        test_years=CANDIDATE_TEST_YEARS,
        vol_window=VOL_WINDOW, selected_mode=POSITION_MODE,
        filter_modes_override=FORCE_FILTER_MODE_BD,
        X_grid=X_GRID_BD, Y_grid=Y_GRID_BD,
        fast_grid=FAST_BD, slow_grid=SLOW_BD,
        tv_grid=TV_BD, sl_grid=SL_BD,
        mom_lookback_grid=[252], objective=ALLOC_OBJECTIVE,
        n_jobs=N_JOBS, fast_mode=FAST_MODE,
        entry_gate_series=bond_gate,
        use_atr_stop=USE_ATR_STOP_BD,
        N_atr_grid=N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window=ATR_WINDOW_BD,
    )
    m_tbsp = compute_metrics(wf_tbsp) if not wf_tbsp.empty else {}
    logging.info("TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 m_tbsp.get("CAGR",0)*100, m_tbsp.get("Sharpe",0),
                 m_tbsp.get("MaxDD",0)*100, m_tbsp.get("CalMAR",0))

    # Mode-specific assets
    cand_wf_results = {}  # {label: (wf_eq, wf_res, wf_tr, price_df, ret_series, m)}

    if CANDIDATE_MODE == "msci_world":
        wf_mw, wf_res_mw, wf_tr_mw, m_mw = _wf_equity_asset(
            raw["MSCIW"], "MSCI_World", mmf_ext
        )
        cand_wf_results["MSCI_World"] = (wf_mw, wf_res_mw, wf_tr_mw,
                                          raw["MSCIW"], derived["ret_MSCIW"], m_mw)

    elif CANDIDATE_MODE == "global_equity":
        for label, price_key, ret_key in [
            ("SP500",     "price_SPX",   "ret_SPX"),
            ("STOXX600",  "price_STOXX", "ret_STOXX"),
            ("Nikkei225", "price_NKX",   "ret_NKX"),
        ]:
            wf_a, wf_res_a, wf_tr_a, m_a = _wf_equity_asset(
                derived[price_key], label, mmf_ext
            )
            cand_wf_results[label] = (wf_a, wf_res_a, wf_tr_a,
                                       derived[price_key], derived[ret_key], m_a)

    # Signals
    sig_wig  = build_signal_series(wf_wig,  wf_tr_wig)
    sig_tbsp = build_signal_series(wf_tbsp, wf_tr_tbsp)
    cand_signals = {
        "WIG":  sig_wig,
        "TBSP": sig_tbsp,
    }
    for label, (wf_a, _, wf_tr_a, _, _, _) in cand_wf_results.items():
        cand_signals[label] = build_signal_series(wf_a, wf_tr_a)

    # Build returns_dict for allocation_walk_forward_n
    ret_eq_series = derived["ret_eq"]
    ret_bd_series = derived["ret_bd"]

    returns_dict = {
        "WIG":  ret_eq_series.dropna(),
        "TBSP": ret_bd_series.dropna(),
    }
    for label, (_, _, _, _, ret_s, _) in cand_wf_results.items():
        returns_dict[label] = ret_s.dropna()

    # Asset key ordering: WIG first, foreign equities, TBSP last
    if CANDIDATE_MODE == "msci_world":
        asset_keys = ["WIG", "MSCI_World", "TBSP"]
    else:
        asset_keys = ["WIG", "SP500", "STOXX600", "Nikkei225", "TBSP"]

    # Natural OOS bounds
    oos_start_cand = wf_res_tbsp["TestStart"].min()
    for label, (_, wf_res_a, _, _, _, _) in cand_wf_results.items():
        oos_start_cand = max(oos_start_cand, wf_res_a["TestStart"].min())
    oos_end_cand = wf_res_tbsp["TestEnd"].max()
    logging.info("Candidate natural OOS start: %s", oos_start_cand.date())

    return dict(
        wf_wig=wf_wig, wf_res_wig=wf_res_wig, wf_tr_wig=wf_tr_wig,
        wf_tbsp=wf_tbsp, wf_res_tbsp=wf_res_tbsp, wf_tr_tbsp=wf_tr_tbsp,
        cand_wf_results=cand_wf_results,
        cand_signals=cand_signals,
        returns_dict=returns_dict,
        asset_keys=asset_keys,
        m_wig=m_wig, m_tbsp=m_tbsp,
        natural_start=oos_start_cand,
        natural_end=oos_end_cand,
    )


# ============================================================
# PHASE 5 — DETERMINE COMMON OOS PERIOD
# ============================================================

def determine_common_oos(prod: dict, cand: dict) -> tuple:
    if FORCE_OOS_START is not None:
        oos_start = pd.Timestamp(FORCE_OOS_START)
        logging.info("OOS start FORCED to %s", oos_start.date())
    else:
        oos_start = max(prod["natural_start"], cand["natural_start"])
        logging.info("OOS start AUTO-DETECTED: %s", oos_start.date())

    oos_end = min(prod["natural_end"], cand["natural_end"])
    logging.info(
        "Common OOS period: %s to %s  (%.1f years)",
        oos_start.date(), oos_end.date(),
        (oos_end - oos_start).days / 365.25,
    )
    return oos_start, oos_end


# ============================================================
# PHASE 6 — ALLOCATION WALK-FORWARDS
# ============================================================

def run_production_allocation(raw, derived, prod, oos_start, oos_end):
    logging.info("--- Production allocation walk-forward ---")
    ret_eq  = derived["ret_eq"]
    ret_bd  = derived["ret_bd"]
    ret_mmf = derived["ret_mmf"]

    def _trim(s, a, b): return s.loc[(s.index >= a) & (s.index <= b)]

    sig_eq_oos = _trim(prod["sig_eq"], oos_start, oos_end)
    sig_bd_oos = _trim(prod["sig_bd"], oos_start, oos_end)

    portfolio_eq, weights_s, realloc_log, alloc_df = allocation_walk_forward(
        equity_returns  = ret_eq,
        bond_returns    = ret_bd,
        mmf_returns     = ret_mmf,
        sig_equity_full = prod["sig_eq"],
        sig_bond_full   = prod["sig_bd"],
        sig_equity_oos  = sig_eq_oos,
        sig_bond_oos    = sig_bd_oos,
        wf_results_eq   = prod["wf_res_eq"],
        wf_results_bd   = prod["wf_res_bd"],
        step            = 0.10,
        objective       = ALLOC_OBJECTIVE,
        cooldown_days   = COOLDOWN_DAYS,
        annual_cap      = ANNUAL_CAP,
    )

    if portfolio_eq.empty:
        logging.error("Production portfolio equity curve is empty."); sys.exit(1)

    # Trim and renormalise to common OOS
    port = portfolio_eq.loc[
        (portfolio_eq.index >= oos_start) & (portfolio_eq.index <= oos_end)
    ]
    port = port / port.iloc[0]
    m = {k: float(v) for k, v in compute_metrics(port).items()}
    logging.info(
        "Production OOS [%s–%s]: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f  Reallocations=%d",
        oos_start.date(), oos_end.date(),
        m["CAGR"]*100, m["Sharpe"], m["MaxDD"]*100, m["CalMAR"], len(realloc_log),
    )
    return dict(
        portfolio_eq=port, weights_s=weights_s, realloc_log=realloc_log,
        alloc_df=alloc_df, metrics=m,
        sig_eq_oos=sig_eq_oos, sig_bd_oos=sig_bd_oos,
    )


def run_candidate_allocation(raw, derived, cand, oos_start, oos_end):
    logging.info("--- Candidate allocation walk-forward ---")
    ret_mmf    = derived["ret_mmf"]
    asset_keys = cand["asset_keys"]
    returns_dict = cand["returns_dict"]

    # Trim signals to common OOS period
    def _trim(s, a, b): return s.loc[(s.index >= a) & (s.index <= b)]

    signals_oos = {k: _trim(s, oos_start, oos_end)
                   for k, s in cand["cand_signals"].items()}

    portfolio_eq, weights_s, realloc_log, alloc_df = allocation_walk_forward_n(
        returns_dict      = returns_dict,
        signals_full_dict = cand["cand_signals"],
        signals_oos_dict  = signals_oos,
        mmf_returns       = ret_mmf,
        wf_results_ref    = cand["wf_res_tbsp"],
        asset_keys        = asset_keys,
        step              = 0.10,
        objective         = ALLOC_OBJECTIVE,
        cooldown_days     = COOLDOWN_DAYS,
        annual_cap        = 999,  # global equity uses no annual cap
        train_years       = CANDIDATE_TRAIN_YEARS,
    )

    if portfolio_eq is None or portfolio_eq.empty:
        logging.error("Candidate portfolio equity curve is empty."); sys.exit(1)

    port = portfolio_eq.loc[
        (portfolio_eq.index >= oos_start) & (portfolio_eq.index <= oos_end)
    ]
    if port.empty:
        logging.error("Candidate portfolio empty after trimming to common OOS."); sys.exit(1)
    port = port / port.iloc[0]
    m = {k: float(v) for k, v in compute_metrics(port).items()}
    logging.info(
        "Candidate OOS [%s–%s]: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f  Reallocations=%d",
        oos_start.date(), oos_end.date(),
        m["CAGR"]*100, m["Sharpe"], m["MaxDD"]*100, m["CalMAR"], len(realloc_log),
    )
    return dict(
        portfolio_eq=port, weights_s=weights_s, realloc_log=realloc_log,
        alloc_df=alloc_df, metrics=m,
        signals_oos=signals_oos,
    )


# ============================================================
# PHASE 7 — MC ROBUSTNESS
# ============================================================

def _run_mc(label, wf_equity, wf_results, price_df, cash_df, thresholds, train_years):
    logging.info("=" * 60)
    logging.info("MC: %s  (n=%d)", label, N_MC)
    logging.info("=" * 60)
    windows     = extract_windows_from_wf_results(wf_results, train_years=train_years)
    best_params = extract_best_params_from_wf_results(wf_results)
    mc_raw = run_monte_carlo_robustness(
        best_params   = best_params,
        windows       = windows,
        df            = price_df,
        cash_df       = cash_df,
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,
        n_samples     = N_MC,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = CLOSE_COL,
    )
    baseline = compute_metrics(wf_equity)
    summary  = analyze_robustness(mc_raw, baseline, thresholds=thresholds)
    return summary


def run_mc_all(raw, derived, prod, cand):
    if not RUN_MC:
        logging.info("MC skipped (RUN_MC=False)")
        return {}, {}

    logging.info("=" * 80)
    logging.info("PHASE 7: MC PARAMETER ROBUSTNESS")
    logging.info("=" * 80)

    mmf_ext = derived["mmf_ext"]

    prod_mc = {}
    prod_mc["WIG"] = _run_mc(
        "Production WIG", prod["wf_eq"], prod["wf_res_eq"],
        raw["WIG"], mmf_ext, EQUITY_THRESHOLDS_MC, TRAIN_YEARS_EQ,
    )
    prod_mc["TBSP"] = _run_mc(
        "Production TBSP", prod["wf_bd"], prod["wf_res_bd"],
        raw["TBSP"], mmf_ext, BOND_THRESHOLDS_MC, TRAIN_YEARS_BD,
    )

    cand_mc = {}
    cand_mc["WIG"] = _run_mc(
        "Candidate WIG", cand["wf_wig"], cand["wf_res_wig"],
        raw["WIG"], mmf_ext, EQUITY_THRESHOLDS_MC, CANDIDATE_TRAIN_YEARS,
    )
    cand_mc["TBSP"] = _run_mc(
        "Candidate TBSP", cand["wf_tbsp"], cand["wf_res_tbsp"],
        raw["TBSP"], mmf_ext, BOND_THRESHOLDS_MC, CANDIDATE_TRAIN_YEARS,
    )
    for label, (wf_a, wf_res_a, _, price_df, _, _) in cand["cand_wf_results"].items():
        cand_mc[label] = _run_mc(
            f"Candidate {label}", wf_a, wf_res_a,
            price_df, mmf_ext, EQUITY_THRESHOLDS_MC, CANDIDATE_TRAIN_YEARS,
        )

    return prod_mc, cand_mc


# ============================================================
# PHASE 8 — BLOCK BOOTSTRAP ROBUSTNESS
# ============================================================

def _run_bootstrap(label, price_df, cash_df, thresholds,
                   train_years, test_years,
                   filter_modes_override,
                   x_grid, y_grid, fast_grid, slow_grid,
                   tv_grid, sl_grid, mom_lb_grid,
                   wf_equity, entry_gate_series=None,
                   use_atr_stop=False, N_atr_grid=None, atr_window=20):
    logging.info("=" * 60)
    logging.info("Bootstrap: %s  (n=%d)", label, N_BOOTSTRAP)
    logging.info("=" * 60)
    bb_raw = run_block_bootstrap_robustness(
        df                    = price_df,
        cash_df               = cash_df,
        price_col             = CLOSE_COL,
        cash_price_col        = CLOSE_COL,
        n_samples             = N_BOOTSTRAP,
        block_size            = 250,
        train_years           = train_years,
        test_years            = test_years,
        vol_window            = VOL_WINDOW,
        funds_df              = None,
        fund_params_grid      = None,
        selected_mode         = POSITION_MODE,
        filter_modes_override = filter_modes_override,
        X_grid                = x_grid,
        Y_grid                = y_grid,
        fast_grid             = fast_grid,
        slow_grid             = slow_grid,
        tv_grid               = tv_grid,
        sl_grid               = sl_grid,
        mom_lookback_grid     = mom_lb_grid,
        entry_gate_series     = entry_gate_series,
        use_atr_stop          = use_atr_stop,
        N_atr_grid            = N_atr_grid,
        atr_window            = atr_window,
    )
    baseline = compute_metrics(wf_equity)
    summary  = analyze_bootstrap(bb_raw, baseline, thresholds=thresholds)
    return summary


def run_bootstrap_all(raw, derived, prod, cand):
    if not RUN_BOOTSTRAP:
        logging.info("Bootstrap skipped (RUN_BOOTSTRAP=False)")
        return {}, {}

    logging.info("=" * 80)
    logging.info("PHASE 8: BLOCK BOOTSTRAP ROBUSTNESS")
    logging.info("=" * 80)

    mmf_ext   = derived["mmf_ext"]
    bond_gate = derived["bond_gate"]

    prod_bb = {}
    prod_bb["WIG"] = _run_bootstrap(
        "Production WIG", raw["WIG"], mmf_ext,
        EQUITY_THRESHOLDS_BOOTSTRAP,
        TRAIN_YEARS_EQ, TEST_YEARS_EQ,
        FORCE_FILTER_MODE_EQ,
        X_GRID_EQ, Y_GRID_EQ, FAST_EQ, SLOW_EQ,
        TV_EQ, SL_EQ, MOM_LB_EQ,
        prod["wf_eq"],
        use_atr_stop=USE_ATR_STOP,
        N_atr_grid=N_ATR_GRID if USE_ATR_STOP else None,
        atr_window=ATR_WINDOW,
    )
    prod_bb["TBSP"] = _run_bootstrap(
        "Production TBSP", raw["TBSP"], mmf_ext,
        BOND_THRESHOLDS_BOOTSTRAP,
        TRAIN_YEARS_BD, TEST_YEARS_BD,
        FORCE_FILTER_MODE_BD,
        X_GRID_BD, Y_GRID_BD, FAST_BD, SLOW_BD,
        TV_BD, SL_BD, [252],
        prod["wf_bd"],
        entry_gate_series=bond_gate,
        use_atr_stop=USE_ATR_STOP_BD,
        N_atr_grid=N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window=ATR_WINDOW_BD,
    )

    cand_bb = {}
    cand_bb["WIG"] = _run_bootstrap(
        "Candidate WIG", raw["WIG"], mmf_ext,
        EQUITY_THRESHOLDS_BOOTSTRAP,
        CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS,
        FORCE_FILTER_MODE_EQ,
        X_GRID_CAND, Y_GRID_CAND, FAST_CAND, SLOW_CAND,
        TV_CAND, SL_CAND, MOM_LB_CAND,
        cand["wf_wig"],
        use_atr_stop=USE_ATR_STOP,
        N_atr_grid=N_ATR_GRID if USE_ATR_STOP else None,
        atr_window=ATR_WINDOW,
    )
    cand_bb["TBSP"] = _run_bootstrap(
        "Candidate TBSP", raw["TBSP"], mmf_ext,
        BOND_THRESHOLDS_BOOTSTRAP,
        CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS,
        FORCE_FILTER_MODE_BD,
        X_GRID_BD, Y_GRID_BD, FAST_BD, SLOW_BD,
        TV_BD, SL_BD, [252],
        cand["wf_tbsp"],
        entry_gate_series=bond_gate,
        use_atr_stop=USE_ATR_STOP_BD,
        N_atr_grid=N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window=ATR_WINDOW_BD,
    )
    for label, (wf_a, _, _, price_df, _, _) in cand["cand_wf_results"].items():
        cand_bb[label] = _run_bootstrap(
            f"Candidate {label}", price_df, mmf_ext,
            EQUITY_THRESHOLDS_BOOTSTRAP,
            CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS,
            FORCE_FILTER_MODE_EQ,
            X_GRID_CAND, Y_GRID_CAND, FAST_CAND, SLOW_CAND,
            TV_CAND, SL_CAND, MOM_LB_CAND,
            wf_a,
            use_atr_stop=USE_ATR_STOP,
            N_atr_grid=N_ATR_GRID if USE_ATR_STOP else None,
            atr_window=ATR_WINDOW,
        )

    return prod_bb, cand_bb


# ============================================================
# PHASE 9 — ALLOCATION WEIGHT ROBUSTNESS
# ============================================================

def run_alloc_robustness(derived, prod_alloc, cand_alloc, cand):
    logging.info("=" * 80)
    logging.info("PHASE 9: ALLOCATION WEIGHT ROBUSTNESS")
    logging.info("=" * 80)

    ret_eq  = derived["ret_eq"]
    ret_bd  = derived["ret_bd"]
    ret_mmf = derived["ret_mmf"]

    # Production (2-asset): perturb equity weight
    logging.info("--- Production allocation robustness ---")
    prod_rob = allocation_weight_robustness(
        alloc_results_df = prod_alloc["alloc_df"],
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_oos   = prod_alloc["sig_eq_oos"],
        sig_bond_oos     = prod_alloc["sig_bd_oos"],
        baseline_metrics = prod_alloc["metrics"],
        perturb_steps    = ALLOC_PERTURB_STEPS,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )
    print_allocation_robustness_report(prod_rob)

    # Candidate (N-asset): perturb WIG weight (typically the dominant allocation)
    logging.info("--- Candidate allocation robustness (WIG focus) ---")
    cand_rob_wig = allocation_weight_robustness_n(
        alloc_results_df = cand_alloc["alloc_df"],
        returns_dict     = cand["returns_dict"],
        mmf_returns      = ret_mmf,
        signals_oos_dict = cand_alloc["signals_oos"],
        asset_keys       = cand["asset_keys"],
        baseline_metrics = cand_alloc["metrics"],
        perturb_steps    = ALLOC_PERTURB_STEPS,
        focus_asset      = "WIG",
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = 999,
    )
    print_allocation_robustness_report_n(cand_rob_wig, focus_asset="WIG")

    # Perturb TBSP weight too (relevant for Mode B where TBSP often has large weight)
    logging.info("--- Candidate allocation robustness (TBSP focus) ---")
    cand_rob_tbsp = allocation_weight_robustness_n(
        alloc_results_df = cand_alloc["alloc_df"],
        returns_dict     = cand["returns_dict"],
        mmf_returns      = ret_mmf,
        signals_oos_dict = cand_alloc["signals_oos"],
        asset_keys       = cand["asset_keys"],
        baseline_metrics = cand_alloc["metrics"],
        perturb_steps    = ALLOC_PERTURB_STEPS,
        focus_asset      = "TBSP",
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = 999,
    )
    print_allocation_robustness_report_n(cand_rob_tbsp, focus_asset="TBSP")

    return prod_rob, cand_rob_wig, cand_rob_tbsp


# ============================================================
# PHASE 10 — SUMMARY REPORT
# ============================================================

def _verdict(summary: dict) -> str:
    if not summary:
        return "skipped"
    return summary.get("verdict", "N/A")


def _calmar_ratio_at_10pp(rob_df) -> str:
    if rob_df is None or rob_df.empty:
        return "N/A"
    col = "equity_shift" if "equity_shift" in rob_df.columns else "weight_shift"
    inner = rob_df.loc[rob_df[col].abs().between(0.09, 0.11)]
    if inner.empty:
        return "N/A"
    r = inner["CalMAR_vs_base"].min()
    if pd.isna(r):
        return "N/A"
    verdict = "PLATEAU" if r >= 0.80 else "MODERATE" if r >= 0.60 else "SPIKE"
    return f"{verdict} ({r:.3f})"


def print_summary(
    oos_start, oos_end,
    prod_alloc, cand_alloc,
    prod_mc, cand_mc,
    prod_bb, cand_bb,
    prod_rob, cand_rob_wig, cand_rob_tbsp,
    raw, derived,
):
    sep  = "=" * 90
    sep2 = "-" * 90

    logging.info("")
    logging.info(sep)
    logging.info("ROBUSTNESS COMPARISON SUMMARY")
    logging.info("OOS: %s to %s  (%.1f years)",
                 oos_start.date(), oos_end.date(),
                 (oos_end - oos_start).days / 365.25)
    logging.info("Production:  2-asset  (WIG+TBSP+MMF)  train=%dy  test=%dy",
                 TRAIN_YEARS_EQ, TEST_YEARS_EQ)
    logging.info("Candidate:   %s  train=%dy  test=%dy",
                 CANDIDATE_MODE, CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS)
    logging.info(sep)

    # ── Portfolio performance ─────────────────────────────────────────────────
    def _m(m, key): return m.get(key, float("nan"))

    logging.info("")
    logging.info("PORTFOLIO OOS PERFORMANCE (common OOS period, renormalised to 1.0)")
    logging.info(sep2)
    logging.info("%-14s  %7s  %7s  %7s  %8s  %8s  %10s",
                 "Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR", "Reallocations")
    logging.info(sep2)

    pm  = prod_alloc["metrics"]
    cm  = cand_alloc["metrics"]
    logging.info("%-14s  %6.2f%%  %6.2f%%  %7.3f  %7.2f%%  %8.3f  %10d",
                 "Production", _m(pm,"CAGR")*100, _m(pm,"Vol")*100,
                 _m(pm,"Sharpe"), _m(pm,"MaxDD")*100, _m(pm,"CalMAR"),
                 len(prod_alloc["realloc_log"]))
    logging.info("%-14s  %6.2f%%  %6.2f%%  %7.3f  %7.2f%%  %8.3f  %10d",
                 "Candidate", _m(cm,"CAGR")*100, _m(cm,"Vol")*100,
                 _m(cm,"Sharpe"), _m(cm,"MaxDD")*100, _m(cm,"CalMAR"),
                 len(cand_alloc["realloc_log"]))

    # B&H benchmarks
    mmf_ext = derived["mmf_ext"]
    bh_wig,  bh_m_wig  = compute_buy_and_hold(raw["WIG"],  CLOSE_COL, oos_start, oos_end)
    bh_tbsp, bh_m_tbsp = compute_buy_and_hold(raw["TBSP"], CLOSE_COL, oos_start, oos_end)
    logging.info("%-14s  %6.2f%%  %6.2f%%  %7.3f  %7.2f%%  %8.3f  %10s",
                 "B&H WIG", bh_m_wig["CAGR"]*100, bh_m_wig["Vol"]*100,
                 bh_m_wig["Sharpe"], bh_m_wig["MaxDD"]*100, bh_m_wig["CalMAR"], "—")
    logging.info("%-14s  %6.2f%%  %6.2f%%  %7.3f  %7.2f%%  %8.3f  %10s",
                 "B&H TBSP", bh_m_tbsp["CAGR"]*100, bh_m_tbsp["Vol"]*100,
                 bh_m_tbsp["Sharpe"], bh_m_tbsp["MaxDD"]*100, bh_m_tbsp["CalMAR"], "—")

    logging.info("")
    logging.info("CAGR delta (Candidate – Production): %+.2f pp",
                 (_m(cm,"CAGR") - _m(pm,"CAGR"))*100)
    logging.info("CalMAR delta (Candidate – Production): %+.3f",
                 _m(cm,"CalMAR") - _m(pm,"CalMAR"))
    logging.info("MaxDD delta (Candidate – Production): %+.2f pp",
                 (_m(cm,"MaxDD") - _m(pm,"MaxDD"))*100)

    # ── MC verdicts ───────────────────────────────────────────────────────────
    logging.info("")
    logging.info("MC PARAMETER ROBUSTNESS VERDICTS  (n=%d)", N_MC if RUN_MC else 0)
    logging.info(sep2)
    all_assets = sorted(set(list(prod_mc.keys()) + list(cand_mc.keys())))
    logging.info("%-16s  %-10s  %-10s", "Asset", "Production", "Candidate")
    logging.info(sep2)
    for asset in all_assets:
        pv = _verdict(prod_mc.get(asset, {}))
        cv = _verdict(cand_mc.get(asset, {}))
        logging.info("%-16s  %-10s  %-10s", asset, pv, cv)

    prod_mc_pass = all(v.get("verdict") == "ROBUST" for v in prod_mc.values()) if prod_mc else False
    cand_mc_pass = all(v.get("verdict") == "ROBUST" for v in cand_mc.values()) if cand_mc else False
    logging.info(sep2)
    logging.info("%-16s  %-10s  %-10s",
                 "All ROBUST?",
                 "YES" if prod_mc_pass else "NO",
                 "YES" if cand_mc_pass else "NO")

    # ── Bootstrap verdicts ────────────────────────────────────────────────────
    logging.info("")
    logging.info("BLOCK BOOTSTRAP VERDICTS  (n=%d, block=250d)", N_BOOTSTRAP if RUN_BOOTSTRAP else 0)
    logging.info(sep2)
    all_bb = sorted(set(list(prod_bb.keys()) + list(cand_bb.keys())))
    logging.info("%-16s  %-10s  %-10s", "Asset", "Production", "Candidate")
    logging.info(sep2)
    for asset in all_bb:
        pv = _verdict(prod_bb.get(asset, {}))
        cv = _verdict(cand_bb.get(asset, {}))
        logging.info("%-16s  %-10s  %-10s", asset, pv, cv)

    prod_bb_pass = all(v.get("verdict") == "ROBUST" for v in prod_bb.values()) if prod_bb else False
    cand_bb_pass = all(v.get("verdict") == "ROBUST" for v in cand_bb.values()) if cand_bb else False
    logging.info(sep2)
    logging.info("%-16s  %-10s  %-10s",
                 "All ROBUST?",
                 "YES" if prod_bb_pass else "NO",
                 "YES" if cand_bb_pass else "NO")

    # ── Allocation weight robustness ──────────────────────────────────────────
    logging.info("")
    logging.info("ALLOCATION WEIGHT ROBUSTNESS  (CalMAR ratio at ±10pp)")
    logging.info(sep2)
    logging.info("%-28s  %-14s  %-14s",
                 "Focus asset / strategy", "Production", "Candidate")
    logging.info(sep2)
    logging.info("%-28s  %-14s  %-14s",
                 "Equity/WIG weight",
                 _calmar_ratio_at_10pp(prod_rob),
                 _calmar_ratio_at_10pp(cand_rob_wig))
    logging.info("%-28s  %-14s  %-14s",
                 "TBSP weight",
                 "N/A (bond = residual)",
                 _calmar_ratio_at_10pp(cand_rob_tbsp))

    # ── Overall assessment ────────────────────────────────────────────────────
    logging.info("")
    logging.info("OVERALL ASSESSMENT")
    logging.info(sep2)

    prod_overall = prod_mc_pass and prod_bb_pass
    cand_overall = cand_mc_pass and cand_bb_pass

    logging.info("Production  2-asset: MC=%s  Bootstrap=%s  → %s",
                 "PASS" if prod_mc_pass else "FAIL",
                 "PASS" if prod_bb_pass else "FAIL",
                 "FULL PASS" if prod_overall else "PARTIAL/FAIL")
    logging.info("Candidate   %s: MC=%s  Bootstrap=%s  → %s",
                 CANDIDATE_MODE,
                 "PASS" if cand_mc_pass else "FAIL",
                 "PASS" if cand_bb_pass else "FAIL",
                 "FULL PASS" if cand_overall else "PARTIAL/FAIL")

    if cand_mc_pass and not prod_mc_pass:
        logging.info("NOTE: Candidate passes MC where production does not — "
                     "may indicate improved signal robustness.")
    if prod_mc_pass and not cand_mc_pass:
        logging.info("NOTE: Production passes MC where candidate does not — "
                     "candidate signal(s) are fragile.")

    logging.info(sep)


# ============================================================
# PHASE 11 — PLOT
# ============================================================

def save_comparison_plot(oos_start, oos_end, prod_alloc, cand_alloc,
                          raw, derived, bh_wig=None, bh_tbsp=None):
    logging.info("Saving comparison chart...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    title = (
        f"Robustness Comparison: 2-Asset vs {CANDIDATE_MODE}  "
        f"[{oos_start.date()} – {oos_end.date()}]"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    prod_eq = prod_alloc["portfolio_eq"]
    cand_eq = cand_alloc["portfolio_eq"]

    if bh_wig is None:
        bh_wig, _ = compute_buy_and_hold(raw["WIG"], CLOSE_COL, oos_start, oos_end)
    if bh_tbsp is None:
        bh_tbsp, _ = compute_buy_and_hold(raw["TBSP"], CLOSE_COL, oos_start, oos_end)

    # Panel 1: equity curves
    ax = axes[0]
    prod_eq.plot(ax=ax, label="2-Asset Production", color="steelblue", linewidth=2)
    cand_eq.plot(ax=ax, label=f"Candidate ({CANDIDATE_MODE})", color="darkgreen",
                 linewidth=2, linestyle="--")
    bh_wig.plot(ax=ax, label="B&H WIG", color="darkorange",
                linewidth=1, linestyle=":", alpha=0.7)
    bh_tbsp.plot(ax=ax, label="B&H TBSP", color="sienna",
                 linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title("Portfolio Equity (OOS, renormalised to 1.0)")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: drawdown comparison
    ax = axes[1]
    dd_prod = prod_eq / prod_eq.cummax() - 1
    dd_cand = cand_eq / cand_eq.cummax() - 1
    dd_wig  = bh_wig  / bh_wig.cummax()  - 1
    dd_prod.plot(ax=ax, label="2-Asset Production", color="steelblue", linewidth=1.5)
    dd_cand.plot(ax=ax, label=f"Candidate ({CANDIDATE_MODE})", color="darkgreen",
                 linewidth=1.5, linestyle="--")
    dd_wig.plot(ax=ax, label="B&H WIG", color="darkorange",
                linewidth=1, linestyle=":", alpha=0.6)
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 3: allocation weights over time (production)
    ax = axes[2]
    ws = prod_alloc["weights_s"]
    if ws is not None and not ws.empty:
        w_eq  = ws.apply(lambda d: d.get("equity", 0) if isinstance(d, dict) else 0)
        w_bd  = ws.apply(lambda d: d.get("bond",   0) if isinstance(d, dict) else 0)
        w_mmf = ws.apply(lambda d: d.get("mmf",    0) if isinstance(d, dict) else 0)
        ax.stackplot(ws.index, w_eq, w_bd, w_mmf,
                     labels=["Equity (WIG)", "Bond (TBSP)", "MMF"],
                     colors=["steelblue", "seagreen", "lightgrey"], alpha=0.8)
        ax.set_title("Production 2-Asset Allocation Weights Over Time")
        ax.set_ylabel("Weight")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No weights_series available", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plot_path = "twoasset_vs_global_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info("Chart saved to %s", plot_path)


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("=" * 80)
    logging.info("TWOASSET VS GLOBAL EQUITY ROBUSTNESS COMPARISON")
    logging.info("START: %s", dt.datetime.now())
    logging.info("Candidate mode: %s  (train=%dy  test=%dy)",
                 CANDIDATE_MODE, CANDIDATE_TRAIN_YEARS, CANDIDATE_TEST_YEARS)
    logging.info("Production: 2-asset  (train=%dy  test=%dy)",
                 TRAIN_YEARS_EQ, TEST_YEARS_EQ)
    logging.info("FORCE_OOS_START: %s", FORCE_OOS_START)
    logging.info("N_MC=%d  N_BOOTSTRAP=%d  RUN_MC=%s  RUN_BOOTSTRAP=%s",
                 N_MC, N_BOOTSTRAP, RUN_MC, RUN_BOOTSTRAP)
    logging.info("N_JOBS=%d", N_JOBS)
    logging.info("=" * 80)

    raw     = download_all_data()
    derived = build_derived(raw)
    prod    = run_production_strategy(raw, derived)
    cand    = run_candidate_strategy(raw, derived, prod)

    oos_start, oos_end = determine_common_oos(prod, cand)

    prod_alloc = run_production_allocation(raw, derived, prod, oos_start, oos_end)
    cand_alloc = run_candidate_allocation(raw, derived, cand, oos_start, oos_end)

    prod_mc, cand_mc = run_mc_all(raw, derived, prod, cand)
    prod_bb, cand_bb = run_bootstrap_all(raw, derived, prod, cand)

    prod_rob, cand_rob_wig, cand_rob_tbsp = run_alloc_robustness(
        derived, prod_alloc, cand_alloc, cand
    )

    print_summary(
        oos_start, oos_end,
        prod_alloc, cand_alloc,
        prod_mc, cand_mc,
        prod_bb, cand_bb,
        prod_rob, cand_rob_wig, cand_rob_tbsp,
        raw, derived,
    )

    save_comparison_plot(oos_start, oos_end, prod_alloc, cand_alloc, raw, derived)

    logging.info("=" * 80)
    logging.info("DONE: %s", dt.datetime.now())
    logging.info("Log: %s", os.path.abspath(LOG_FILE))
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
