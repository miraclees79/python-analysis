"""
twoasset_robustness.py
======================
Full robustness evaluation for the extended 2-asset (WIG+TBSP+MMF) strategy
using extended TBSP (Drive backcast) and extended MMF (WIBOR1M chain-link).

Evaluates two window configurations:
  - 7+2  (best config from twoasset_extended_comparison.py)
  - 9+1  (existing production baseline)

Robustness layers per config:
  Level 1a: MC parameter perturbation — WIG signal     (n=1000)
  Level 1b: MC parameter perturbation — TBSP signal    (n=1000)
  Level 2a: Block bootstrap — WIG                      (n=500)
  Level 2b: Block bootstrap — TBSP                     (n=500)
  Level 3:  Allocation weight perturbation (equity ±10pp, ±20pp)

All results reported against the stitched OOS equity curve on the
extended OOS period (2011-01-04 onward with extended TBSP).

USAGE
-----
  # From multiasset_freeze_0_4/ (so imports resolve):
  TBSP_EXTENDED_PATH=/path/to/tbsp_extended_combined.csv \\
  GDRIVE_FOLDER_ID=your_folder_id \\
  python twoasset_robustness.py

  # To skip slow bootstrap (MC only):
  RUN_BOOTSTRAP=0 python twoasset_robustness.py

  # To run only one config:
  CONFIGS=7+2 python twoasset_robustness.py
  CONFIGS=9+1 python twoasset_robustness.py
"""

import os
import sys
import logging
import tempfile
import datetime as dt

import numpy as np
import pandas as pd

# ── Logging ─────────────────────────────────────────────────────────────────

LOG_FILE = "twoasset_robustness.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_fh  = logging.FileHandler(LOG_FILE, mode="w")
_fh.setFormatter(_fmt)
root_logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
root_logger.addHandler(_ch)

# ── Imports ──────────────────────────────────────────────────────────────────

from strategy_test_library import (
    download_csv, load_csv,
    walk_forward, compute_metrics,
)
from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    allocation_weight_robustness,
    print_allocation_robustness_report,
    build_spread_prefilter,
    build_yield_momentum_prefilter,
    )
from global_equity_library import build_mmf_extended
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

# ── Drive import (for extended TBSP) ─────────────────────────────────────────

try:
    from price_series_builder import build_and_upload
    _PRICE_BUILDER_AVAILABLE = True
except ImportError:
    _PRICE_BUILDER_AVAILABLE = False
    logging.warning(
        "price_series_builder not found — will use TBSP_EXTENDED_PATH local file. "
        "Set TBSP_EXTENDED_PATH env var to tbsp_extended_combined.csv."
    )

# ============================================================
# USER SETTINGS
# ============================================================

# Which configs to run (subset of WINDOW_CONFIGS)
# Set env var CONFIGS="7+2" or "9+1" or "7+2,9+1"
_configs_env = os.getenv("CONFIGS", "7+2,9+1")
WINDOW_CONFIGS = []
for _c in _configs_env.split(","):
    _c = _c.strip()
    if "+" in _c:
        _tr, _te = _c.split("+")
        WINDOW_CONFIGS.append((int(_tr), int(_te)))

if not WINDOW_CONFIGS:
    WINDOW_CONFIGS = [(7, 2), (9, 1)]

# Google Drive folder (for extended TBSP via build_and_upload)
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "").strip()

# Local fallback for extended TBSP (if Drive not available)
TBSP_EXTENDED_PATH = os.getenv("TBSP_EXTENDED_PATH", None)

# Robustness iterations
N_MC         = int(os.getenv("N_MC",  "1000"))   # set to 10 for smoke test
N_BOOTSTRAP  = int(os.getenv("N_BOOTSTRAP", "500"))

FAST_MODE = True

# Whether to run bootstrap (slow: ~3-6h per config on 12-core machine)
RUN_BOOTSTRAP = os.getenv("RUN_BOOTSTRAP", "1") != "0"

# Strategy settings — must match production freeze
POSITION_MODE        = "full"
VOL_WINDOW           = 20
FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]
COOLDOWN_DAYS        = 10
ANNUAL_CAP           = 12
ALLOC_OBJECTIVE      = "calmar"
YIELD_PREFILTER_BP   = 50.0

# Equity grids
X_GRID_EQ = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ   = [50, 75, 100]
SLOW_EQ   = [150, 200, 250]
TV_EQ     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ = [126, 252]

# Bond grids
X_GRID_BD = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD = [0.001]
FAST_BD   = [50, 100, 150]
SLOW_BD   = [200, 300, 400, 500]
TV_BD     = [0.08, 0.10, 0.12]
SL_BD     = [0.01, 0.02, 0.03]

# Allocation robustness perturbation steps
ALLOC_PERTURB_STEPS = [-0.2, -0.1, 0.0, 0.1, 0.2]

# Parallelism
_cpu = os.cpu_count() or 1
N_JOBS = max(1, _cpu - 1) if _cpu > 3 and sys.platform == "win32" else _cpu


# ============================================================
# DATA LOADING
# ============================================================

def load_all_data():
    logging.info("=" * 80)
    logging.info("LOADING DATA")
    logging.info("=" * 80)

    tmp_dir = tempfile.mkdtemp()

    def _stooq(ticker, label, mandatory=True):
        url  = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
        path = os.path.join(tmp_dir, f"{label}.csv")
        ok   = download_csv(url, path)
        if not ok:
            if mandatory:
                logging.error("FAIL: %s", label); sys.exit(1)
            return None
        df = load_csv(path)
        if df is None and mandatory:
            logging.error("FAIL load_csv: %s", label); sys.exit(1)
        if df is not None:
            logging.info("OK  %-10s  %5d rows  %s to %s",
                         label, len(df),
                         df.index.min().date(), df.index.max().date())
        return df

    WIG   = _stooq("wig",      "WIG")
    MMF   = _stooq("2720.n",   "MMF")
    W1M   = _stooq("plopln1m", "WIBOR1M", mandatory=False)
    PL10Y = _stooq("10yply.b", "PL10Y")
    DE10Y = _stooq("10ydey.b", "DE10Y")

    # Extended TBSP — try Drive first, then local file, then stooq fallback
    TBSP = None

    if _PRICE_BUILDER_AVAILABLE and GDRIVE_FOLDER_ID:
        logging.info("Loading extended TBSP via price_series_builder (Drive)...")
        TBSP = build_and_upload(
            folder_id         = GDRIVE_FOLDER_ID,
            raw_filename      = "tbsp_extended_full.csv",
            combined_filename = "tbsp_extended_combined.csv",
            extension_ticker  = "^tbsp",
            extension_source  = "stooq",
        )
        if TBSP is not None:
            for col in ["Najwyzszy", "Najnizszy"]:
                if col not in TBSP.columns:
                    TBSP[col] = TBSP["Zamkniecie"]
            logging.info("OK  %-10s  %5d rows  %s to %s  (Drive extended)",
                         "TBSP_EXT", len(TBSP),
                         TBSP.index.min().date(), TBSP.index.max().date())

    if TBSP is None and TBSP_EXTENDED_PATH and os.path.exists(TBSP_EXTENDED_PATH):
        logging.info("Loading extended TBSP from local file: %s", TBSP_EXTENDED_PATH)
        TBSP = _load_local_tbsp(TBSP_EXTENDED_PATH)

    if TBSP is None:
        logging.warning(
            "Extended TBSP not available — using standard ^tbsp. "
            "OOS will start ~2015, not 2011. Set GDRIVE_FOLDER_ID or TBSP_EXTENDED_PATH."
        )
        TBSP = _stooq("^tbsp", "TBSP")

    return WIG, TBSP, MMF, W1M, PL10Y, DE10Y


def _load_local_tbsp(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip()
        date_col = next(
            (c for c in df.columns if c.lower() in ["data", "date"]), None
        )
        if date_col is None:
            raise ValueError(f"No date column. Columns: {list(df.columns)}")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        df.index.name = "Data"
        if "Zamkniecie" not in df.columns:
            close_col = next(
                (c for c in df.columns if c.lower() in ["close", "price"]), None
            )
            if close_col is None:
                raise ValueError(f"No close column. Columns: {list(df.columns)}")
            df = df.rename(columns={close_col: "Zamkniecie"})
        for col in ["Najwyzszy", "Najnizszy"]:
            if col not in df.columns:
                df[col] = df["Zamkniecie"]
        logging.info("OK  %-10s  %5d rows  %s to %s  (local extended)",
                     "TBSP_EXT", len(df),
                     df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        logging.error("Failed to load local TBSP: %s", exc)
        return None


def build_derived(WIG, TBSP, MMF, W1M, PL10Y, DE10Y):
    logging.info("=" * 80)
    logging.info("BUILDING DERIVED SERIES")
    logging.info("=" * 80)

    # Extended MMF
    if W1M is not None:
        MMF_EXT = build_mmf_extended(MMF, W1M, floor_date="1994-10-03")
        logging.info("MMF extended back to %s", MMF_EXT.index.min().date())
    else:
        MMF_EXT = MMF
        logging.warning("WIBOR1M unavailable — using standard MMF (starts ~1999)")

    # Bond entry gate
    spread_pf = build_spread_prefilter(PL10Y, DE10Y)
    yield_pf  = build_yield_momentum_prefilter(
        PL10Y, rise_threshold_bp=YIELD_PREFILTER_BP, lookback_days=63
    )
    tbsp_idx    = TBSP.index
    spread_gate = spread_pf.reindex(tbsp_idx, method="ffill").fillna(1).astype(int)
    yield_gate  = yield_pf.reindex(tbsp_idx,  method="ffill").fillna(1).astype(int)
    bond_gate   = (spread_gate & yield_gate).astype(int)
    bond_gate.name = "bond_entry_gate"
    logging.info("Bond entry gate: %.1f%% of days pass", bond_gate.mean() * 100)

    ret_eq  = WIG["Zamkniecie"].pct_change().dropna()
    ret_bd  = TBSP["Zamkniecie"].pct_change().dropna()
    ret_mmf = MMF_EXT["Zamkniecie"].pct_change().dropna()

    return MMF_EXT, bond_gate, ret_eq, ret_bd, ret_mmf


# ============================================================
# WALK-FORWARD (reused across robustness layers)
# ============================================================

def run_walk_forward(train_years, test_years,
                     WIG, TBSP, MMF_EXT, bond_gate):
    """
    Run equity and bond walk-forwards. Returns all outputs needed
    by the robustness layers.
    """
    logging.info("=" * 80)
    logging.info("WALK-FORWARD  train=%dy  test=%dy", train_years, test_years)
    logging.info("=" * 80)

    wf_eq_eq, wf_res_eq, wf_tr_eq = walk_forward(
        df                    = WIG,
        cash_df               = MMF_EXT,
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
        fast_mode=FAST_MODE
    )

    wf_eq_bd, wf_res_bd, wf_tr_bd = walk_forward(
        df                    = TBSP,
        cash_df               = MMF_EXT,
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
        entry_gate_series=bond_gate,
        fast_mode=FAST_MODE,
    )

    if wf_eq_eq.empty or wf_eq_bd.empty:
        logging.error("Walk-forward returned empty equity curve — aborting config.")
        return None

    m_eq = compute_metrics(wf_eq_eq)
    m_bd = compute_metrics(wf_eq_bd)
    logging.info("WIG  OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 m_eq["CAGR"]*100, m_eq["Sharpe"], m_eq["MaxDD"]*100, m_eq["CalMAR"])
    logging.info("TBSP OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 m_bd["CAGR"]*100, m_bd["Sharpe"], m_bd["MaxDD"]*100, m_bd["CalMAR"])

    return (wf_eq_eq, wf_res_eq, wf_tr_eq,
            wf_eq_bd, wf_res_bd, wf_tr_bd,
            m_eq, m_bd)


# ============================================================
# ALLOCATION WALK-FORWARD + PORTFOLIO METRICS
# ============================================================

def run_allocation(wf_eq_eq, wf_eq_bd,
                   wf_res_eq, wf_res_bd,
                   wf_tr_eq, wf_tr_bd,
                   ret_eq, ret_bd, ret_mmf):

    sig_eq = build_signal_series(wf_eq_eq, wf_tr_eq)
    sig_bd = build_signal_series(wf_eq_bd, wf_tr_bd)

    oos_start = max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min())
    oos_end   = min(wf_res_eq["TestEnd"].max(),   wf_res_bd["TestEnd"].max())

    def _trim(s, a, b):
        return s.loc[(s.index >= a) & (s.index <= b)]

    sig_eq_oos = _trim(sig_eq, oos_start, oos_end)
    sig_bd_oos = _trim(sig_bd, oos_start, oos_end)

    portfolio_eq, weights_s, realloc_log, alloc_df = allocation_walk_forward(
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_full  = sig_eq,
        sig_bond_full    = sig_bd,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        wf_results_eq    = wf_res_eq,
        wf_results_bd    = wf_res_bd,
        step             = 0.10,
        objective        = ALLOC_OBJECTIVE,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )

    port_metrics = {k: float(v) for k, v in compute_metrics(portfolio_eq).items()}
    logging.info(
        "Portfolio OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f  "
        "Reallocations=%d",
        port_metrics["CAGR"]*100, port_metrics["Sharpe"],
        port_metrics["MaxDD"]*100, port_metrics["CalMAR"],
        len(realloc_log),
    )

    return portfolio_eq, port_metrics, sig_eq_oos, sig_bd_oos, alloc_df


# ============================================================
# LEVEL 1: MC PARAMETER ROBUSTNESS
# ============================================================

def run_mc_robustness(label, wf_equity, wf_results, price_df, cash_df,
                      thresholds, train_years):
    logging.info("=" * 80)
    logging.info("LEVEL 1 MC — %s  (n=%d)", label, N_MC)
    logging.info("=" * 80)

    windows     = extract_windows_from_wf_results(wf_results, train_years=train_years)
    best_params = extract_best_params_from_wf_results(wf_results)

    mc_results = run_monte_carlo_robustness(
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
        price_col     = "Zamkniecie",
    )

    baseline = compute_metrics(wf_equity)
    summary  = analyze_robustness(mc_results, baseline, thresholds=thresholds)
    return summary


# ============================================================
# LEVEL 2: BLOCK BOOTSTRAP
# ============================================================

def run_bootstrap_robustness(label, price_df, cash_df,
                              thresholds, train_years, test_years,
                              filter_modes_override,
                              x_grid, y_grid, fast_grid, slow_grid,
                              tv_grid, sl_grid, mom_lb_grid,
                              wf_equity,
                              entry_gate_series=None):
    logging.info("=" * 80)
    logging.info("LEVEL 2 BOOTSTRAP — %s  (n=%d)", label, N_BOOTSTRAP)
    logging.info("=" * 80)

    bb_results = run_block_bootstrap_robustness(
        df                    = price_df,
        cash_df               = cash_df,
        price_col             = "Zamkniecie",
        cash_price_col        = "Zamkniecie",
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
    )

    baseline = compute_metrics(wf_equity)
    summary  = analyze_bootstrap(bb_results, baseline, thresholds=thresholds)
    return summary


# ============================================================
# LEVEL 3: ALLOCATION WEIGHT PERTURBATION
# ============================================================

def run_alloc_robustness(alloc_df, ret_eq, ret_bd, ret_mmf,
                          sig_eq_oos, sig_bd_oos, port_metrics):
    logging.info("=" * 80)
    logging.info("LEVEL 3 ALLOCATION WEIGHT PERTURBATION")
    logging.info("=" * 80)

    rob_df = allocation_weight_robustness(
        alloc_results_df = alloc_df,
        equity_returns   = ret_eq,
        bond_returns     = ret_bd,
        mmf_returns      = ret_mmf,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        baseline_metrics = port_metrics,
        perturb_steps    = ALLOC_PERTURB_STEPS,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )
    print_allocation_robustness_report(rob_df)
    return rob_df


# ============================================================
# SUMMARY REPORT
# ============================================================

def print_summary(config_label, port_metrics,
                  mc_eq_summary, mc_bd_summary,
                  bb_eq_summary, bb_bd_summary,
                  alloc_rob_df):

    sep = "=" * 80

    logging.info("")
    logging.info(sep)
    logging.info("ROBUSTNESS SUMMARY — %s", config_label)
    logging.info(sep)

    logging.info("Portfolio OOS: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%  CalMAR=%.3f",
                 port_metrics["CAGR"]*100, port_metrics["Sharpe"],
                 port_metrics["MaxDD"]*100, port_metrics["CalMAR"])

    logging.info("-" * 80)
    logging.info("MC verdicts:        WIG=%s   TBSP=%s",
                 mc_eq_summary.get("verdict", "N/A"),
                 mc_bd_summary.get("verdict", "N/A"))

    if bb_eq_summary is not None and bb_bd_summary is not None:
        logging.info("Bootstrap verdicts: WIG=%s   TBSP=%s",
                     bb_eq_summary.get("verdict", "N/A"),
                     bb_bd_summary.get("verdict", "N/A"))
    else:
        logging.info("Bootstrap verdicts: skipped")

    if alloc_rob_df is not None and not alloc_rob_df.empty:
        inner = alloc_rob_df.loc[
            alloc_rob_df["equity_shift"].abs().between(0.09, 0.11)
        ]
        if not inner.empty:
            ratio = inner["CalMAR_vs_base"].min()
            verdict = ("PLATEAU" if ratio >= 0.80
                       else "MODERATE" if ratio >= 0.60
                       else "SPIKE")
            logging.info("Alloc weight:       %s  (min CalMAR ratio at ±10pp = %.3f)",
                         verdict, ratio)

    # Overall verdict
    mc_pass = (mc_eq_summary.get("verdict") == "ROBUST" and
               mc_bd_summary.get("verdict") == "ROBUST")
    bb_pass = (bb_eq_summary is None or bb_bd_summary is None or
               (bb_eq_summary.get("verdict") == "ROBUST" and
                bb_bd_summary.get("verdict") == "ROBUST"))
    alloc_pass = True
    if alloc_rob_df is not None and not alloc_rob_df.empty:
        inner = alloc_rob_df.loc[
            alloc_rob_df["equity_shift"].abs().between(0.09, 0.11)
        ]
        if not inner.empty:
            alloc_pass = inner["CalMAR_vs_base"].min() >= 0.60

    overall = "FULL PASS" if (mc_pass and bb_pass and alloc_pass) else "PARTIAL"
    logging.info("")
    logging.info("OVERALL: %s  (MC=%s  Bootstrap=%s  Alloc=%s)",
                 overall,
                 "PASS" if mc_pass   else "FAIL",
                 "PASS" if bb_pass   else "FAIL",
                 "PASS" if alloc_pass else "FAIL")
    logging.info(sep)


# ============================================================
# MAIN
# ============================================================

def main():
    logging.info("=" * 80)
    logging.info("2-ASSET EXTENDED ROBUSTNESS  START: %s", dt.datetime.now())
    logging.info("Configs: %s", [(t, te) for t, te in WINDOW_CONFIGS])
    logging.info("N_MC=%d  N_BOOTSTRAP=%d  RUN_BOOTSTRAP=%s",
                 N_MC, N_BOOTSTRAP, RUN_BOOTSTRAP)
    logging.info("N_JOBS=%d", N_JOBS)
    logging.info("=" * 80)

    # Load data once
    WIG, TBSP, MMF, W1M, PL10Y, DE10Y = load_all_data()
    MMF_EXT, bond_gate, ret_eq, ret_bd, ret_mmf = build_derived(
        WIG, TBSP, MMF, W1M, PL10Y, DE10Y
    )

    all_summaries = {}

    for train_years, test_years in WINDOW_CONFIGS:
        config_label = f"{train_years}+{test_years}"
        logging.info("")
        logging.info("*" * 80)
        logging.info("*  CONFIG: %s", config_label)
        logging.info("*" * 80)

        # Walk-forward
        wf_out = run_walk_forward(
            train_years, test_years,
            WIG, TBSP, MMF_EXT, bond_gate,
        )
        if wf_out is None:
            logging.error("Skipping config %s — walk-forward failed.", config_label)
            continue

        (wf_eq_eq, wf_res_eq, wf_tr_eq,
         wf_eq_bd, wf_res_bd, wf_tr_bd,
         m_eq, m_bd) = wf_out

        # Allocation
        portfolio_eq, port_metrics, sig_eq_oos, sig_bd_oos, alloc_df = (
            run_allocation(
                wf_eq_eq, wf_eq_bd,
                wf_res_eq, wf_res_bd,
                wf_tr_eq, wf_tr_bd,
                ret_eq, ret_bd, ret_mmf,
            )
        )

        # Level 1a: MC — WIG
        mc_eq_summary = run_mc_robustness(
            label       = f"WIG [{config_label}]",
            wf_equity   = wf_eq_eq,
            wf_results  = wf_res_eq,
            price_df    = WIG,
            cash_df     = MMF_EXT,
            thresholds  = EQUITY_THRESHOLDS_MC,
            train_years = train_years,
        )

        # Level 1b: MC — TBSP
        mc_bd_summary = run_mc_robustness(
            label       = f"TBSP [{config_label}]",
            wf_equity   = wf_eq_bd,
            wf_results  = wf_res_bd,
            price_df    = TBSP,
            cash_df     = MMF_EXT,
            thresholds  = BOND_THRESHOLDS_MC,
            train_years = train_years,
        )

        # Level 2a: Bootstrap — WIG
        bb_eq_summary = None
        bb_bd_summary = None

        if RUN_BOOTSTRAP:
            bb_eq_summary = run_bootstrap_robustness(
                label                 = f"WIG [{config_label}]",
                price_df              = WIG,
                cash_df               = MMF_EXT,
                thresholds            = EQUITY_THRESHOLDS_BOOTSTRAP,
                train_years           = train_years,
                test_years            = test_years,
                filter_modes_override = FORCE_FILTER_MODE_EQ,
                x_grid=X_GRID_EQ, y_grid=Y_GRID_EQ,
                fast_grid=FAST_EQ, slow_grid=SLOW_EQ,
                tv_grid=TV_EQ, sl_grid=SL_EQ,
                mom_lb_grid=MOM_LB_EQ,
                wf_equity             = wf_eq_eq,
            )

            # Level 2b: Bootstrap — TBSP
            # Note: entry_gate_series passed so bootstrap uses same gate as production.
            # bond_gate is on the TBSP index — reindex to full date range inside bootstrap.
            bb_bd_summary = run_bootstrap_robustness(
                label                 = f"TBSP [{config_label}]",
                price_df              = TBSP,
                cash_df               = MMF_EXT,
                thresholds            = BOND_THRESHOLDS_BOOTSTRAP,
                train_years           = train_years,
                test_years            = test_years,
                filter_modes_override = FORCE_FILTER_MODE_BD,
                x_grid=X_GRID_BD, y_grid=Y_GRID_BD,
                fast_grid=FAST_BD, slow_grid=SLOW_BD,
                tv_grid=TV_BD, sl_grid=SL_BD,
                mom_lb_grid=[252],
                wf_equity             = wf_eq_bd,
                entry_gate_series     = bond_gate,
            )

        # Level 3: Allocation weight perturbation
        alloc_rob_df = run_alloc_robustness(
            alloc_df, ret_eq, ret_bd, ret_mmf,
            sig_eq_oos, sig_bd_oos, port_metrics,
        )

        # Summary
        print_summary(
            config_label   = config_label,
            port_metrics   = port_metrics,
            mc_eq_summary  = mc_eq_summary,
            mc_bd_summary  = mc_bd_summary,
            bb_eq_summary  = bb_eq_summary,
            bb_bd_summary  = bb_bd_summary,
            alloc_rob_df   = alloc_rob_df,
        )

        all_summaries[config_label] = {
            "port_metrics":    port_metrics,
            "mc_wig_verdict":  mc_eq_summary.get("verdict"),
            "mc_tbsp_verdict": mc_bd_summary.get("verdict"),
            "bb_wig_verdict":  bb_eq_summary.get("verdict") if bb_eq_summary else "skipped",
            "bb_tbsp_verdict": bb_bd_summary.get("verdict") if bb_bd_summary else "skipped",
        }

    # Final cross-config summary
    if len(all_summaries) > 1:
        logging.info("")
        logging.info("=" * 80)
        logging.info("CROSS-CONFIG SUMMARY")
        logging.info("=" * 80)
        logging.info(
            "%-8s  %7s  %7s  %8s  %8s  %-9s  %-9s  %-11s  %-11s",
            "Config", "CAGR", "Sharpe", "MaxDD", "CalMAR",
            "MC_WIG", "MC_TBSP", "Boot_WIG", "Boot_TBSP",
        )
        logging.info("-" * 90)
        for cfg, s in all_summaries.items():
            m = s["port_metrics"]
            logging.info(
                "%-8s  %6.2f%%  %7.3f  %7.2f%%  %8.3f  %-9s  %-9s  %-11s  %-11s",
                cfg,
                m["CAGR"]*100, m["Sharpe"], m["MaxDD"]*100, m["CalMAR"],
                s["mc_wig_verdict"],  s["mc_tbsp_verdict"],
                s["bb_wig_verdict"],  s["bb_tbsp_verdict"],
            )
        logging.info("=" * 80)

    logging.info("DONE: %s", dt.datetime.now())
    logging.info("Full log: %s", os.path.abspath(LOG_FILE))


if __name__ == "__main__":
    main()
