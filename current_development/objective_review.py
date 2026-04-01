"""
objective_review.py
===================
Annual objective function review for the multi-asset pension fund strategy
(WIG equity + TBSP bond + MMF).  Implements the process defined in Section 12
of the strategy documentation.

WHAT IT DOES
------------
Runs 13 experiment configurations covering all 5 objective functions applied
to each of the three optimisers independently:

  Group A — equity signal objective varies, bond=calmar, alloc=calmar  (5 runs)
  Group B — bond signal objective varies,  equity=calmar, alloc=calmar (4 runs, calmar/calmar/calmar already in A)
  Group C — alloc objective varies,        equity=calmar, bond=calmar  (4 runs, same)

Data is downloaded once.  Walk-forward results are cached so that:
  - calmar/calmar WIG and TBSP runs are computed once and reused across groups B and C.
  - calmar/calmar WIG run is reused for all of group B.
  - calmar/calmar TBSP run is reused for all of group A.

For signal optimisers (groups A and B), MC parameter perturbation is run on
the signal component whose objective varied.  For group C (allocation only),
no per-signal MC is needed — the allocation optimiser has no separate MC
framework.

OUTPUT
------
  objective_review_results.csv       — one row per experiment, all metrics
  objective_review_summary.log       — full console/log output
  objective_review_decision_table.txt — formatted decision table for journal

SWITCHING CRITERIA (from Section 12.3)
---------------------------------------
A challenger is flagged SWITCH_CANDIDATE if ALL of the following hold:
  (a) portfolio CAGR   > incumbent + 0.005  (0.5pp)
  (b) portfolio CalMAR > incumbent + 0.05
  (c) portfolio MaxDD  > incumbent - 0.02   (no worse than 2pp deeper)
  (d) MC verdict on the varied signal = ROBUST (where applicable)
  (e) year_win_count  >= 3  (wins in >= 3 of the last 5 full calendar years)

All 5 must be satisfied simultaneously.  Near-misses (4 of 5) are flagged
separately.

USAGE
-----
  cd multiasset_freeze_0_4
  python objective_review.py

  # Smoke test with n_mc=10:
  python objective_review.py --n_mc 10

  # Skip MC entirely (OOS metrics only):
  python objective_review.py --no_mc

RUNTIME ESTIMATE (12-core machine)
------------------------------------
  n_mc=1000 : ~3h  (13 runs × ~14min WIG walk-forward + ~3min TBSP + ~14min MC each)
  n_mc=100  : ~45min
  n_mc=10   : ~8min (smoke test)
"""

import os
import sys
import csv
import logging
import tempfile
import datetime as dt
from copy import deepcopy

import pandas as pd
import numpy as np

# ── imports from freeze_0_4 package (assumes PYTHONPATH=multiasset_freeze_0_4) ──
from strategy_test_library import (
    download_csv,
    load_csv,
    walk_forward,
    compute_metrics,
    analyze_trades,
)
from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    build_spread_prefilter,
    build_yield_momentum_prefilter,
)
from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    EQUITY_THRESHOLDS_MC,
    BOND_THRESHOLDS_MC,
)

from price_series_builder import build_and_upload, load_combined_from_drive
from global_equity_library import build_mmf_extended
from stooq_hybrid_updater import run_update
# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "objective_review_summary.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_fh  = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setFormatter(_fmt)
_ch  = logging.StreamHandler()
_ch.setFormatter(_fmt)
root_logger.addHandler(_fh)
root_logger.addHandler(_ch)


# ============================================================
# FROZEN v0.4 SETTINGS  (must match multiasset_freeze_0_4 exactly)
# ============================================================

POSITION_MODE    = "full"
TRAIN_YEARS_EQ   = 9
TEST_YEARS_EQ    = 1
TRAIN_YEARS_BD   = 9
TEST_YEARS_BD    = 1
VOL_WINDOW       = 20
COOLDOWN_DAYS    = 10
ANNUAL_CAP       = 12
BOUNDARY_EXITS   = {"CARRY", "SAMPLE_END"}

FORCE_FILTER_MODE_EQ = None
FORCE_FILTER_MODE_BD = ["ma"]

X_GRID_EQ  = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID_EQ  = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_EQ    = [50, 75, 100]
SLOW_EQ    = [150, 200, 250]
TV_EQ      = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_EQ      = [0.05, 0.08, 0.10, 0.15]
MOM_LB_EQ  = [126, 252]

X_GRID_BD  = [0.05, 0.08, 0.10, 0.15]
Y_GRID_BD  = [0.001]
FAST_BD    = [50, 100, 150]
SLOW_BD    = [200, 300, 400, 500]
TV_BD      = [0.08, 0.10, 0.12]
SL_BD      = [0.01, 0.02, 0.03]

YIELD_PREFILTER_THRESHOLD_BP = 50.0

# Last 5 full calendar years used for per-year switching criterion
# Updated each review — should be the 5 years ending the previous Dec 31.
LAST_5_YEARS = [2021, 2022, 2023, 2024, 2025]

# Switching thresholds (Section 12.3)
SWITCH_CAGR_MARGIN   =  0.005   # challenger must beat incumbent by >= 0.5pp
SWITCH_CALMAR_MARGIN =  0.05    # challenger must beat incumbent by >= 0.05
SWITCH_MAXDD_SLACK   = -0.02    # challenger MaxDD must be >= incumbent - 2pp
SWITCH_YEAR_MIN      = 3        # must win in >= N of the LAST_5_YEARS

# All five objective functions to evaluate
ALL_OBJECTIVES = ["calmar", "sharpe", "sortino", "calmar_sharpe", "calmar_sortino"]
INCUMBENT      = "calmar"

WIG_DATA_FLOOR = "1995-01-02"
 
# MMF extension: chain-link WIBOR 1M backwards from first MMF NAV to this
#   date. Ensures IS windows starting before 1999 have realistic cash returns
#   rather than ret_mmf=0 on signal-off days. Applied in Phase 2.
MMF_FLOOR = "1995-01-02"
DATA_START = "1990-01-01"  # hard floor for all series


# ============================================================
# EXPERIMENT MATRIX
# ============================================================
# Each row: (eq_obj, bd_obj, alloc_obj, mc_component)
# mc_component: "equity" | "bond" | None
# The calmar/calmar/calmar row is always the first (row index 0 = incumbent).

def build_experiment_matrix():
    rows = []
    # Incumbent
    rows.append({"eq_obj": "calmar", "bd_obj": "calmar", "alloc_obj": "calmar",
                 "mc_component": "equity",   # run MC on equity for incumbent too
                 "group": "incumbent"})
    # Group A — equity objective varies
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": obj, "bd_obj": "calmar", "alloc_obj": "calmar",
                     "mc_component": "equity",
                     "group": "A_eq"})
    # Group B — bond objective varies
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": "calmar", "bd_obj": obj, "alloc_obj": "calmar",
                     "mc_component": "bond",
                     "group": "B_bd"})
    # Group C — alloc objective varies
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": "calmar", "bd_obj": "calmar", "alloc_obj": obj,
                     "mc_component": None,
                     "group": "C_alloc"})
    return rows


# ============================================================
# DATA DOWNLOAD  (once)
# ============================================================

def download_data():
    logging.info("=" * 80)
    logging.info("DOWNLOADING DATA")
    logging.info("=" * 80)
    

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

    if W1M is not None:
        MMF_EXT = build_mmf_extended(MMF, W1M, floor_date="1994-10-03")
        logging.info("MMF extended back to %s", MMF_EXT.index.min().date())
    else:
        MMF_EXT = MMF

        logging.warning("WIBOR1M unavailable — using standard MMF (starts ~1999)")
    logging.info("WIG %d rows | TBSP %d rows | MMF %d rows | PL10Y %d rows | DE10Y %d rows",
                 len(WIG), len(TBSP), len(MMF), len(PL10Y), len(DE10Y))

    # Build bond entry gate (identical to multiasset_runfile.py)
    spread_prefilter = build_spread_prefilter(PL10Y, DE10Y)
    yield_prefilter  = build_yield_momentum_prefilter(
        PL10Y,
        rise_threshold_bp=YIELD_PREFILTER_THRESHOLD_BP,
        lookback_days=63,
    )
    tbsp_index   = TBSP.index
    spread_gate  = spread_prefilter.reindex(tbsp_index, method="ffill").fillna(1).astype(int)
    yield_gate   = yield_prefilter.reindex(tbsp_index, method="ffill").fillna(1).astype(int)
    bond_entry_gate = (spread_gate & yield_gate).astype(int)
    bond_entry_gate.name = "bond_entry_gate"
    logging.info("Bond entry gate: %.1f%% of days pass", bond_entry_gate.mean() * 100)

    # Daily returns
    ret_eq  = WIG["Zamkniecie"].pct_change().dropna().rename("ret_eq")
    ret_bd  = TBSP["Zamkniecie"].pct_change().dropna().rename("ret_bd")
    ret_mmf = MMF["Zamkniecie"].pct_change().dropna().rename("ret_mmf")

    return dict(WIG=WIG, TBSP=TBSP, MMF=MMF,
                ret_eq=ret_eq, ret_bd=ret_bd, ret_mmf=ret_mmf,
                bond_entry_gate=bond_entry_gate)


# ============================================================
# WALK-FORWARD HELPERS
# ============================================================

def _cpu_jobs():
    n = os.cpu_count() or 1
    return max(1, n - 1) if n > 3 and sys.platform == "win32" else n


def run_equity_wf(data, eq_obj):
    """Run WIG walk-forward with the given objective. Returns (wf_equity, wf_results, wf_trades)."""
    logging.info("  [WIG walk-forward] objective=%s", eq_obj)
    return walk_forward(
        df                    = data["WIG"],
        cash_df               = data["MMF"],
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
        objective             = eq_obj,
        n_jobs                = _cpu_jobs(),
    )


def run_bond_wf(data, bd_obj):
    """Run TBSP walk-forward with the given objective. Returns (wf_equity, wf_results, wf_trades)."""
    logging.info("  [TBSP walk-forward] objective=%s", bd_obj)
    return walk_forward(
        df                    = data["TBSP"],
        cash_df               = data["MMF"],
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
        objective             = bd_obj,
        n_jobs                = _cpu_jobs(),
        entry_gate_series     = data["bond_entry_gate"],
    )


def run_allocation(data, wf_equity_eq, wf_results_eq, wf_equity_bd, wf_results_bd,
                   sig_equity, sig_bond, alloc_obj):
    """Run the allocation walk-forward. Returns (portfolio_equity, weights_series, realloc_log, alloc_results_df)."""
    logging.info("  [Allocation walk-forward] objective=%s", alloc_obj)

    oos_start_eq = wf_results_eq["TestStart"].min()
    oos_start_bd = wf_results_bd["TestStart"].min()
    oos_start    = max(oos_start_eq, oos_start_bd)
    oos_end_eq   = wf_results_eq["TestEnd"].max()
    oos_end_bd   = wf_results_bd["TestEnd"].max()
    oos_end      = min(oos_end_eq, oos_end_bd)

    def _trim(s, a, b):
        return s.loc[(s.index >= a) & (s.index <= b)]

    sig_eq_oos = _trim(sig_equity, oos_start, oos_end)
    sig_bd_oos = _trim(sig_bond,   oos_start, oos_end)

    return allocation_walk_forward(
        equity_returns   = data["ret_eq"],
        bond_returns     = data["ret_bd"],
        mmf_returns      = data["ret_mmf"],
        sig_equity_full  = sig_equity,
        sig_bond_full    = sig_bond,
        sig_equity_oos   = sig_eq_oos,
        sig_bond_oos     = sig_bd_oos,
        wf_results_eq    = wf_results_eq,
        wf_results_bd    = wf_results_bd,
        step             = 0.10,
        objective        = alloc_obj,
        cooldown_days    = COOLDOWN_DAYS,
        annual_cap       = ANNUAL_CAP,
    )


# ============================================================
# MC HELPER
# ============================================================

def run_mc(data, wf_results, asset_df_key, n_mc, thresholds, train_years):
    """Run MC perturbation on a signal walk-forward result. Returns analyze_robustness summary dict."""
    windows     = extract_windows_from_wf_results(wf_results, train_years=train_years)
    best_params = extract_best_params_from_wf_results(wf_results)
    mc_results  = run_monte_carlo_robustness(
        best_params   = best_params,
        windows       = windows,
        df            = data[asset_df_key],
        cash_df       = data["MMF"],
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,
        n_samples     = n_mc,
        n_jobs        = _cpu_jobs(),
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie",
    )
    return mc_results   # caller passes to analyze_robustness with its baseline


# ============================================================
# PER-YEAR PORTFOLIO RETURN HELPER
# ============================================================

def annual_cagr_by_year(portfolio_equity):
    """
    Compute calendar-year CAGR for each full year in the portfolio equity curve.
    Returns dict {year: cagr_float}.
    """
    annual = {}
    df = portfolio_equity.copy()
    df.index = pd.to_datetime(df.index)
    for year in df.index.year.unique():
        yr = df[df.index.year == year]
        if len(yr) < 50:
            continue   # skip partial years
        start_val = yr.iloc[0]
        end_val   = yr.iloc[-1]
        days      = (yr.index[-1] - yr.index[0]).days
        if days < 1 or start_val <= 0:
            continue
        cagr = (end_val / start_val) ** (365.25 / days) - 1
        annual[year] = cagr
    return annual


def count_year_wins(cand_annual, incumb_annual, years):
    """Count how many of `years` the challenger beats the incumbent."""
    wins = 0
    for y in years:
        c = cand_annual.get(y)
        i = incumb_annual.get(y)
        if c is not None and i is not None and c > i:
            wins += 1
    return wins


# ============================================================
# SWITCHING CRITERIA EVALUATION
# ============================================================

def evaluate_switching_criteria(row, incumbent_row):
    """
    Apply the 5-gate switching criteria from Section 12.3.
    Returns (all_met: bool, n_met: int, detail: dict).
    """
    cagr_ok  = (row["portfolio_cagr"]  - incumbent_row["portfolio_cagr"])  >= SWITCH_CAGR_MARGIN
    calmar_ok= (row["portfolio_calmar"] - incumbent_row["portfolio_calmar"]) >= SWITCH_CALMAR_MARGIN
    maxdd_ok = row["portfolio_maxdd"] >= (incumbent_row["portfolio_maxdd"]  + SWITCH_MAXDD_SLACK)
    mc_ok    = (row["mc_verdict"] == "ROBUST") if row["mc_verdict"] is not None else True  # alloc group: no MC
    year_ok  = row["year_win_count"] >= SWITCH_YEAR_MIN

    gates  = {"cagr": cagr_ok, "calmar": calmar_ok, "maxdd": maxdd_ok,
              "mc_robust": mc_ok, "year_wins": year_ok}
    n_met  = sum(gates.values())
    all_met= n_met == 5
    return all_met, n_met, gates


# ============================================================
# MAIN REVIEW LOOP
# ============================================================

def run_review(n_mc, run_mc_flag):
    logging.info("=" * 80)
    logging.info("ANNUAL OBJECTIVE FUNCTION REVIEW  —  %s", dt.datetime.now().strftime("%Y-%m-%d"))
    logging.info("n_mc=%d  |  run_mc=%s  |  last_5_years=%s",
                 n_mc, run_mc_flag, LAST_5_YEARS)
    logging.info("=" * 80)

    # ── 1. Download data once ──
    data = download_data()

    # ── 2. Build experiment matrix ──
    experiments = build_experiment_matrix()
    logging.info("Experiment matrix: %d configurations", len(experiments))

    # ── 3. Walk-forward cache: {(eq_obj, bd_obj): (wf_eq_bundle, wf_bd_bundle)} ──
    # A bundle = (wf_equity, wf_results, wf_trades, sig_series)
    wf_cache_eq = {}   # keyed by eq_obj
    wf_cache_bd = {}   # keyed by bd_obj

    results = []       # list of dicts — one per experiment
    incumbent_row = None

    for exp_idx, exp in enumerate(experiments):
        eq_obj    = exp["eq_obj"]
        bd_obj    = exp["bd_obj"]
        alloc_obj = exp["alloc_obj"]
        mc_comp   = exp["mc_component"]
        group     = exp["group"]

        logging.info("")
        logging.info("=" * 80)
        logging.info("EXPERIMENT %d/%d  |  group=%s  |  eq=%s  bd=%s  alloc=%s",
                     exp_idx + 1, len(experiments), group, eq_obj, bd_obj, alloc_obj)
        logging.info("=" * 80)

        # ── 3a. Equity walk-forward (cached) ──
        if eq_obj not in wf_cache_eq:
            wf_eq, wf_res_eq, wf_tr_eq = run_equity_wf(data, eq_obj)
            sig_eq = build_signal_series(wf_eq, wf_tr_eq)
            wf_cache_eq[eq_obj] = (wf_eq, wf_res_eq, wf_tr_eq, sig_eq)
        wf_eq, wf_res_eq, wf_tr_eq, sig_eq = wf_cache_eq[eq_obj]

        # ── 3b. Bond walk-forward (cached) ──
        if bd_obj not in wf_cache_bd:
            wf_bd, wf_res_bd, wf_tr_bd = run_bond_wf(data, bd_obj)
            sig_bd = build_signal_series(wf_bd, wf_tr_bd)
            wf_cache_bd[bd_obj] = (wf_bd, wf_res_bd, wf_tr_bd, sig_bd)
        wf_bd, wf_res_bd, wf_tr_bd, sig_bd = wf_cache_bd[bd_obj]

        # ── 3c. Individual signal metrics ──
        m_eq = compute_metrics(wf_eq)
        m_bd = compute_metrics(wf_bd)

        # ── 3d. Allocation walk-forward ──
        port_eq, weights_s, realloc_log, alloc_df = run_allocation(
            data, wf_eq, wf_res_eq, wf_bd, wf_res_bd,
            sig_eq, sig_bd, alloc_obj,
        )
        m_port = compute_metrics(port_eq) if not port_eq.empty else {}
        n_realloc = len(realloc_log)

        # ── 3e. Per-year portfolio CAGR ──
        annual_cagrs = annual_cagr_by_year(port_eq) if not port_eq.empty else {}
        year_wins    = None   # filled after incumbent established

        # ── 3f. MC on varied signal component ──
        mc_verdict  = None
        mc_p05_cagr = None
        mc_p05_maxdd= None
        mc_p_worse  = None

        if run_mc_flag and mc_comp is not None:
            logging.info("  [MC] component=%s  n=%d", mc_comp, n_mc)
            if mc_comp == "equity":
                mc_raw = run_mc(data, wf_res_eq, "WIG",  n_mc, EQUITY_THRESHOLDS_MC, TRAIN_YEARS_EQ)
                mc_baseline = m_eq
                mc_thresh   = EQUITY_THRESHOLDS_MC
            else:
                mc_raw = run_mc(data, wf_res_bd, "TBSP", n_mc, BOND_THRESHOLDS_MC,   TRAIN_YEARS_BD)
                mc_baseline = m_bd
                mc_thresh   = BOND_THRESHOLDS_MC

            mc_summary = analyze_robustness(mc_raw, mc_baseline, thresholds=mc_thresh)
            mc_verdict  = mc_summary.get("verdict")
            mc_p05_cagr = mc_summary.get("CAGR", {}).get("p05")
            mc_p05_maxdd= mc_summary.get("MaxDD", {}).get("p05")
            mc_p_worse  = mc_summary.get("CAGR", {}).get("p_worse")

        # ── 3g. Assemble result row ──
        row = {
            "group":           group,
            "eq_obj":          eq_obj,
            "bd_obj":          bd_obj,
            "alloc_obj":       alloc_obj,
            "mc_component":    mc_comp,
            # Equity signal
            "eq_cagr":         m_eq.get("CAGR"),
            "eq_sharpe":       m_eq.get("Sharpe"),
            "eq_calmar":       m_eq.get("CalMAR"),
            "eq_maxdd":        m_eq.get("MaxDD"),
            # Bond signal
            "bd_cagr":         m_bd.get("CAGR"),
            "bd_sharpe":       m_bd.get("Sharpe"),
            "bd_calmar":       m_bd.get("CalMAR"),
            "bd_maxdd":        m_bd.get("MaxDD"),
            # Portfolio
            "portfolio_cagr":  m_port.get("CAGR"),
            "portfolio_sharpe":m_port.get("Sharpe"),
            "portfolio_calmar":m_port.get("CalMAR"),
            "portfolio_maxdd": m_port.get("MaxDD"),
            "portfolio_sortino":m_port.get("Sortino"),
            "n_reallocations": n_realloc,
            # MC
            "mc_verdict":      mc_verdict,
            "mc_p05_cagr":     mc_p05_cagr,
            "mc_p05_maxdd":    mc_p05_maxdd,
            "mc_p_worse":      mc_p_worse,
            # Per-year wins computed after incumbent established
            "year_win_count":  None,
            "year_win_detail": None,
            "annual_cagrs":    annual_cagrs,   # temporary, stripped from CSV
        }

        results.append(row)

        # Store incumbent (first row = calmar/calmar/calmar)
        if exp_idx == 0:
            incumbent_row = row
            logging.info("  Incumbent portfolio: CAGR=%.2f%%  CalMAR=%.2f  MaxDD=%.2f%%  Reallocations=%d",
                         (row["portfolio_cagr"] or 0) * 100,
                         row["portfolio_calmar"] or 0,
                         (row["portfolio_maxdd"] or 0) * 100,
                         n_realloc)

    # ── 4. Fill year_win_count now that incumbent is known ──
    incumb_annual = incumbent_row["annual_cagrs"]
    for row in results:
        wins = count_year_wins(row["annual_cagrs"], incumb_annual, LAST_5_YEARS)
        row["year_win_count"] = wins
        detail_parts = []
        for y in LAST_5_YEARS:
            c = row["annual_cagrs"].get(y)
            i = incumb_annual.get(y)
            if c is not None and i is not None:
                marker = "W" if c > i else "L"
                detail_parts.append(f"{y}:{marker}({c*100:+.1f}%vs{i*100:+.1f}%)")
        row["year_win_detail"] = " | ".join(detail_parts)

    # ── 5. Apply switching criteria ──
    for row in results:
        if row["group"] == "incumbent":
            row["switch_all_met"]  = False
            row["switch_n_met"]    = 5   # incumbent trivially meets all its own criteria
            row["switch_detail"]   = "INCUMBENT"
            continue
        all_met, n_met, gates = evaluate_switching_criteria(row, incumbent_row)
        row["switch_all_met"]  = all_met
        row["switch_n_met"]    = n_met
        row["switch_detail"]   = " | ".join(
            f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()
        )

    # ── 6. Write CSV ──
    csv_path = "objective_review_results.csv"
    csv_cols = [
        "group", "eq_obj", "bd_obj", "alloc_obj", "mc_component",
        "eq_cagr", "eq_sharpe", "eq_calmar", "eq_maxdd",
        "bd_cagr", "bd_sharpe", "bd_calmar", "bd_maxdd",
        "portfolio_cagr", "portfolio_sharpe", "portfolio_calmar",
        "portfolio_maxdd", "portfolio_sortino", "n_reallocations",
        "mc_verdict", "mc_p05_cagr", "mc_p05_maxdd", "mc_p_worse",
        "year_win_count", "year_win_detail",
        "switch_all_met", "switch_n_met", "switch_detail",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            # Format floats for readability
            out = {}
            for k in csv_cols:
                v = row.get(k)
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = v
            writer.writerow(out)
    logging.info("Results written to %s", csv_path)

    # ── 7. Print decision table ──
    print_decision_table(results, incumbent_row)

    return results


# ============================================================
# DECISION TABLE PRINTER
# ============================================================

def _pct(v, decimals=2):
    if v is None: return "N/A"
    return f"{v * 100:.{decimals}f}%"

def _f(v, decimals=2):
    if v is None: return "N/A"
    return f"{v:.{decimals}f}"


def print_decision_table(results, incumbent_row):
    """Print a formatted review decision table to the log and a separate .txt file."""

    header = (
        f"{'GROUP':<12} {'EQ_OBJ':<16} {'BD_OBJ':<16} {'ALLOC_OBJ':<16} "
        f"{'PORT_CAGR':>10} {'PORT_CALMAR':>12} {'PORT_MAXDD':>11} {'PORT_SHARPE':>12} "
        f"{'N_REALLOC':>10} {'MC_VERDICT':>11} {'YR_WINS':>8} "
        f"{'N_MET':>6} {'SWITCH':>8}"
    )
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "ANNUAL OBJECTIVE FUNCTION REVIEW — DECISION TABLE",
        f"Run date  : {dt.datetime.now().strftime('%Y-%m-%d')}",
        f"Incumbent : calmar / calmar / calmar",
        f"Last 5 yrs: {LAST_5_YEARS}",
        f"Switch bar: CAGR +{SWITCH_CAGR_MARGIN*100:.1f}pp | CalMAR +{SWITCH_CALMAR_MARGIN:.2f} | MaxDD slack {SWITCH_MAXDD_SLACK*100:.1f}pp | MC=ROBUST | >= {SWITCH_YEAR_MIN} yr wins",
        "=" * len(header),
        "",
        header,
        sep,
    ]

    for row in results:
        switch_flag = ("*** SWITCH CANDIDATE ***" if row.get("switch_all_met")
                       else f"({row.get('switch_n_met', 0)}/5 gates)")
        mc_v = row["mc_verdict"] if row["mc_verdict"] is not None else "(alloc)"
        line = (
            f"{row['group']:<12} {row['eq_obj']:<16} {row['bd_obj']:<16} {row['alloc_obj']:<16} "
            f"{_pct(row['portfolio_cagr']):>10} {_f(row['portfolio_calmar']):>12} "
            f"{_pct(row['portfolio_maxdd']):>11} {_f(row['portfolio_sharpe']):>12} "
            f"{str(row['n_reallocations']):>10} {str(mc_v):>11} "
            f"{str(row.get('year_win_count', 'N/A')):>8} "
            f"{str(row.get('switch_n_met', '')):>6} {switch_flag}"
        )
        lines.append(line)
        # Indented switching gate detail for non-incumbent rows
        if row["group"] != "incumbent" and row.get("switch_detail"):
            lines.append(f"{'':>14}  gates: {row['switch_detail']}")
        if row.get("year_win_detail"):
            lines.append(f"{'':>14}  years: {row['year_win_detail']}")
        lines.append("")

    lines += [sep, ""]

    # Candidates summary
    candidates = [r for r in results if r.get("switch_all_met")]
    near_misses = [r for r in results if r.get("switch_n_met") == 4 and not r.get("switch_all_met")]

    if candidates:
        lines.append("SWITCH CANDIDATES (all 5 gates passed):")
        for r in candidates:
            lines.append(f"  eq={r['eq_obj']}  bd={r['bd_obj']}  alloc={r['alloc_obj']}")
    else:
        lines.append("NO SWITCH CANDIDATES — retain CalMAR for all optimisers.")

    if near_misses:
        lines.append("")
        lines.append("NEAR-MISSES (4/5 gates — document in journal):")
        for r in near_misses:
            lines.append(
                f"  eq={r['eq_obj']}  bd={r['bd_obj']}  alloc={r['alloc_obj']}"
                f"  | failing gate: {r['switch_detail']}"
            )

    lines.append("")
    lines.append("=" * len(header))

    full_text = "\n".join(lines)

    # Log each line
    for line in lines:
        logging.info(line)

    # Write to file
    txt_path = "objective_review_decision_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")
    logging.info("Decision table written to %s", txt_path)


# ============================================================
# RUN SETTINGS  — edit here before running from Spyder / IDE
# ============================================================

# Number of Monte Carlo samples per signal component.
#   1000 — full production run (~3h on 12-core machine)
#   100  — quick indicative run (~45min)
#   10   — smoke test to verify the script runs end-to-end (~8min)
N_MC = 1000

# Set to False to skip MC entirely and collect OOS walk-forward metrics only.
# Useful for a fast first pass or when MC results are already known.
RUN_MC = True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_review(n_mc=N_MC, run_mc_flag=RUN_MC)
    logging.info("Review complete.")
