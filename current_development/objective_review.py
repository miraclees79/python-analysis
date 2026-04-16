"""
objective_review.py
===================
Annual objective function review for the multi-asset pension fund strategy.
Implements the process defined in Section 12 of the strategy documentation.

Runs 13 experiment configurations covering all 5 objective functions applied
to each of the three optimisers independently:

  Group A — equity signal objective varies, bond=calmar, alloc=calmar  (5 runs)
  Group B — bond signal objective varies,  equity=calmar, alloc=calmar (4 runs)
  Group C — alloc objective varies,        equity=calmar, bond=calmar  (4 runs)

Performance utilities (annual_cagr_by_year, count_year_wins) are now in
strategy_test_library and imported from there rather than defined locally.
build_standard_two_asset_data() is used for the shared data setup phase.
get_n_jobs() replaces the duplicated cpu-count block.

USAGE
-----
  cd <project dir>
  python objective_review.py

  # Smoke test:
  python objective_review.py   # set N_MC = 10 in RUN SETTINGS below
"""

import os
import sys
import csv
import logging
import tempfile
import datetime as dt

import pandas as pd
import numpy as np

from strategy_test_library import (
    load_csv,
    load_stooq_local,           # replaces per-runfile _stooq()
    get_n_jobs,                 # replaces duplicated cpu-count block
    walk_forward,
    compute_metrics,
    analyze_trades,
    annual_cagr_by_year,        # moved from objective_review.py
    count_year_wins,            # moved from objective_review.py
)
from multiasset_library import (
    build_signal_series,
    allocation_walk_forward,
    build_standard_two_asset_data,   # replaces inline build_derived() block
)
from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    EQUITY_THRESHOLDS_MC,
    BOND_THRESHOLDS_MC,
)
from price_series_builder import build_and_upload
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
# FROZEN v0.4 SETTINGS  (must match multiasset production freeze exactly)
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

WIG_DATA_FLOOR = "1995-01-02"
MMF_FLOOR      = "1995-01-02"
DATA_START     = "1990-01-01"

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

FAST_MODE = True

LAST_5_YEARS = [2021, 2022, 2023, 2024, 2025]

SWITCH_CAGR_MARGIN   =  0.005
SWITCH_CALMAR_MARGIN =  0.05
SWITCH_MAXDD_SLACK   = -0.02
SWITCH_YEAR_MIN      = 3

ALL_OBJECTIVES = ["calmar", "sharpe", "sortino", "calmar_sharpe", "calmar_sortino"]
INCUMBENT      = "calmar"


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

def build_experiment_matrix():
    rows = []
    rows.append({"eq_obj": "calmar", "bd_obj": "calmar", "alloc_obj": "calmar",
                 "mc_component": "equity", "group": "incumbent"})
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": obj, "bd_obj": "calmar", "alloc_obj": "calmar",
                     "mc_component": "equity", "group": "A_eq"})
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": "calmar", "bd_obj": obj, "alloc_obj": "calmar",
                     "mc_component": "bond", "group": "B_bd"})
    for obj in ALL_OBJECTIVES:
        if obj == "calmar":
            continue
        rows.append({"eq_obj": "calmar", "bd_obj": "calmar", "alloc_obj": obj,
                     "mc_component": None, "group": "C_alloc"})
    return rows


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data() -> dict:
    logging.info("=" * 80)
    logging.info("DOWNLOADING DATA")
    logging.info("=" * 80)

    run_update(get_funds=False)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # load_stooq_local replaces the local _stooq() helper
    WIG    = load_stooq_local("wig",       "WIG",     DATA_DIR, DATA_START)
    WIG    = WIG.loc[WIG.index >= pd.Timestamp(WIG_DATA_FLOOR)]
    MMF    = load_stooq_local("fund_2720", "MMF",     DATA_DIR, DATA_START)
    W1M    = load_stooq_local("wibor1m",  "WIBOR1M", DATA_DIR, DATA_START, mandatory=False)
    PL10Y  = load_stooq_local("pl10y",    "PL10Y",   DATA_DIR, DATA_START)
    DE10Y  = load_stooq_local("de10y",    "DE10Y",   DATA_DIR, DATA_START)

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
            logging.error("FAIL: TBSP unavailable."); sys.exit(1)
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "TBSP", len(TBSP), TBSP.index.min().date(), TBSP.index.max().date())
    for col in ["Najwyzszy", "Najnizszy"]:
        if col not in TBSP.columns:
            TBSP[col] = TBSP["Zamkniecie"]

    # build_standard_two_asset_data replaces the ~50-line inline setup block
    derived = build_standard_two_asset_data(
        wig                = WIG,
        tbsp               = TBSP,
        mmf                = MMF,
        wibor1m            = W1M,
        pl10y              = PL10Y,
        de10y              = DE10Y,
        mmf_floor          = MMF_FLOOR,
        yield_prefilter_bp = YIELD_PREFILTER_THRESHOLD_BP,
    )

    return dict(
        WIG          = WIG,
        TBSP         = TBSP,
        MMF          = MMF,
        mmf_ext      = derived["mmf_ext"],
        bond_entry_gate = derived["bond_gate"],
        ret_eq       = derived["ret_eq"],
        ret_bd       = derived["ret_bd"],
        ret_mmf      = derived["ret_mmf"],
    )


# ============================================================
# WALK-FORWARD HELPERS
# ============================================================

def run_equity_wf(data: dict, eq_obj: str):
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
        n_jobs                = get_n_jobs(),
        fast_mode             = FAST_MODE,
        use_atr_stop          = USE_ATR_STOP,
        N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
        atr_window            = ATR_WINDOW,
        
    )


def run_bond_wf(data: dict, bd_obj: str):
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
        n_jobs                = get_n_jobs(),
        entry_gate_series     = data["bond_entry_gate"],
        fast_mode=FAST_MODE,
        use_atr_stop          = USE_ATR_STOP_BD,
        N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
        atr_window            = ATR_WINDOW,
    )


def run_allocation(data: dict, wf_equity_eq, wf_results_eq, wf_equity_bd,
                   wf_results_bd, sig_equity, sig_bond, alloc_obj: str):
    logging.info("  [Allocation walk-forward] objective=%s", alloc_obj)

    oos_start = max(wf_results_eq["TestStart"].min(), wf_results_bd["TestStart"].min())
    oos_end   = min(wf_results_eq["TestEnd"].max(),   wf_results_bd["TestEnd"].max())

    def _trim(s, a, b):
        return s.loc[(s.index >= a) & (s.index <= b)]

    return allocation_walk_forward(
        equity_returns   = data["ret_eq"],
        bond_returns     = data["ret_bd"],
        mmf_returns      = data["ret_mmf"],
        sig_equity_full  = sig_equity,
        sig_bond_full    = sig_bond,
        sig_equity_oos   = _trim(sig_equity, oos_start, oos_end),
        sig_bond_oos     = _trim(sig_bond,   oos_start, oos_end),
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

def run_mc(data: dict, wf_results, asset_df_key: str, n_mc: int,
           thresholds: dict, train_years: int) -> pd.DataFrame:
    windows     = extract_windows_from_wf_results(wf_results, train_years=train_years)
    best_params = extract_best_params_from_wf_results(wf_results)
    return run_monte_carlo_robustness(
        best_params   = best_params,
        windows       = windows,
        df            = data[asset_df_key],
        cash_df       = data["MMF"],
        vol_window    = VOL_WINDOW,
        selected_mode = POSITION_MODE,
        funds_df      = None,
        n_samples     = n_mc,
        n_jobs        = get_n_jobs(),
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie",
    )


# ============================================================
# SWITCHING CRITERIA EVALUATION
# ============================================================

def evaluate_switching_criteria(row: dict, incumbent_row: dict) -> tuple:
    """Apply the 5-gate switching criteria from Section 12.3."""
    cagr_ok   = (row["portfolio_cagr"]  - incumbent_row["portfolio_cagr"])  >= SWITCH_CAGR_MARGIN
    calmar_ok = (row["portfolio_calmar"] - incumbent_row["portfolio_calmar"]) >= SWITCH_CALMAR_MARGIN
    maxdd_ok  = row["portfolio_maxdd"] >= (incumbent_row["portfolio_maxdd"]  + SWITCH_MAXDD_SLACK)
    mc_ok     = (row["mc_verdict"] == "ROBUST") if row["mc_verdict"] is not None else True
    year_ok   = row["year_win_count"] >= SWITCH_YEAR_MIN

    gates  = {"cagr": cagr_ok, "calmar": calmar_ok, "maxdd": maxdd_ok,
              "mc_robust": mc_ok, "year_wins": year_ok}
    n_met  = sum(gates.values())
    all_met = n_met == 5
    return all_met, n_met, gates


# ============================================================
# MAIN REVIEW LOOP
# ============================================================

def run_review(n_mc: int, run_mc_flag: bool) -> list:
    logging.info("=" * 80)
    logging.info("ANNUAL OBJECTIVE FUNCTION REVIEW  —  %s",
                 dt.datetime.now().strftime("%Y-%m-%d"))
    logging.info("n_mc=%d  |  run_mc=%s  |  last_5_years=%s",
                 n_mc, run_mc_flag, LAST_5_YEARS)
    logging.info("=" * 80)

    data        = download_data()
    experiments = build_experiment_matrix()
    logging.info("Experiment matrix: %d configurations", len(experiments))

    wf_cache_eq = {}
    wf_cache_bd = {}
    results     = []
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

        if eq_obj not in wf_cache_eq:
            wf_eq, wf_res_eq, wf_tr_eq = run_equity_wf(data, eq_obj)
            sig_eq = build_signal_series(wf_eq, wf_tr_eq)
            wf_cache_eq[eq_obj] = (wf_eq, wf_res_eq, wf_tr_eq, sig_eq)
        wf_eq, wf_res_eq, wf_tr_eq, sig_eq = wf_cache_eq[eq_obj]

        if bd_obj not in wf_cache_bd:
            wf_bd, wf_res_bd, wf_tr_bd = run_bond_wf(data, bd_obj)
            sig_bd = build_signal_series(wf_bd, wf_tr_bd)
            wf_cache_bd[bd_obj] = (wf_bd, wf_res_bd, wf_tr_bd, sig_bd)
        wf_bd, wf_res_bd, wf_tr_bd, sig_bd = wf_cache_bd[bd_obj]

        m_eq = compute_metrics(wf_eq)
        m_bd = compute_metrics(wf_bd)

        port_eq, weights_s, realloc_log, alloc_df = run_allocation(
            data, wf_eq, wf_res_eq, wf_bd, wf_res_bd, sig_eq, sig_bd, alloc_obj
        )
        m_port    = compute_metrics(port_eq) if not port_eq.empty else {}
        n_realloc = len(realloc_log)

        # annual_cagr_by_year and count_year_wins now come from strategy_test_library
        annual_cagrs = annual_cagr_by_year(port_eq) if not port_eq.empty else {}

        mc_verdict  = None
        mc_p05_cagr = None
        mc_p05_maxdd= None
        mc_p_worse  = None

        if run_mc_flag and mc_comp is not None:
            logging.info("  [MC] component=%s  n=%d", mc_comp, n_mc)
            if mc_comp == "equity":
                mc_raw  = run_mc(data, wf_res_eq, "WIG",  n_mc, EQUITY_THRESHOLDS_MC, TRAIN_YEARS_EQ)
                mc_base = m_eq
                mc_thr  = EQUITY_THRESHOLDS_MC
            else:
                mc_raw  = run_mc(data, wf_res_bd, "TBSP", n_mc, BOND_THRESHOLDS_MC,   TRAIN_YEARS_BD)
                mc_base = m_bd
                mc_thr  = BOND_THRESHOLDS_MC

            mc_summary  = analyze_robustness(mc_raw, mc_base, thresholds=mc_thr)
            mc_verdict  = mc_summary.get("verdict")
            mc_p05_cagr = mc_summary.get("CAGR",  {}).get("p05")
            mc_p05_maxdd= mc_summary.get("MaxDD", {}).get("p05")
            mc_p_worse  = mc_summary.get("CAGR",  {}).get("p_worse")

        row = {
            "group":             group,
            "eq_obj":            eq_obj,
            "bd_obj":            bd_obj,
            "alloc_obj":         alloc_obj,
            "mc_component":      mc_comp,
            "eq_cagr":           m_eq.get("CAGR"),
            "eq_sharpe":         m_eq.get("Sharpe"),
            "eq_calmar":         m_eq.get("CalMAR"),
            "eq_maxdd":          m_eq.get("MaxDD"),
            "bd_cagr":           m_bd.get("CAGR"),
            "bd_sharpe":         m_bd.get("Sharpe"),
            "bd_calmar":         m_bd.get("CalMAR"),
            "bd_maxdd":          m_bd.get("MaxDD"),
            "portfolio_cagr":    m_port.get("CAGR"),
            "portfolio_sharpe":  m_port.get("Sharpe"),
            "portfolio_calmar":  m_port.get("CalMAR"),
            "portfolio_maxdd":   m_port.get("MaxDD"),
            "portfolio_sortino": m_port.get("Sortino"),
            "n_reallocations":   n_realloc,
            "mc_verdict":        mc_verdict,
            "mc_p05_cagr":       mc_p05_cagr,
            "mc_p05_maxdd":      mc_p05_maxdd,
            "mc_p_worse":        mc_p_worse,
            "year_win_count":    None,
            "year_win_detail":   None,
            "annual_cagrs":      annual_cagrs,   # stripped from CSV output
        }
        results.append(row)

        if exp_idx == 0:
            incumbent_row = row
            logging.info("  Incumbent portfolio: CAGR=%.2f%%  CalMAR=%.2f  "
                         "MaxDD=%.2f%%  Reallocations=%d",
                         (row["portfolio_cagr"] or 0)*100,
                         row["portfolio_calmar"] or 0,
                         (row["portfolio_maxdd"] or 0)*100, n_realloc)

    # Fill year_win_count now that incumbent is known
    incumb_annual = incumbent_row["annual_cagrs"]
    for row in results:
        # count_year_wins now comes from strategy_test_library
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

    # Apply switching criteria
    for row in results:
        if row["group"] == "incumbent":
            row["switch_all_met"] = False
            row["switch_n_met"]   = 5
            row["switch_detail"]  = "INCUMBENT"
            continue
        all_met, n_met, gates = evaluate_switching_criteria(row, incumbent_row)
        row["switch_all_met"] = all_met
        row["switch_n_met"]   = n_met
        row["switch_detail"]  = " | ".join(
            f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()
        )

    # Write CSV
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
            out = {}
            for k in csv_cols:
                v = row.get(k)
                out[k] = f"{v:.6f}" if isinstance(v, float) else v
            writer.writerow(out)
    logging.info("Results written to %s", csv_path)

    print_decision_table(results, incumbent_row)
    return results


# ============================================================
# DECISION TABLE PRINTER
# ============================================================

def _pct(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def _f(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def print_decision_table(results: list, incumbent_row: dict) -> None:
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
        f"Switch bar: CAGR +{SWITCH_CAGR_MARGIN*100:.1f}pp | CalMAR +{SWITCH_CALMAR_MARGIN:.2f} | "
        f"MaxDD slack {SWITCH_MAXDD_SLACK*100:.1f}pp | MC=ROBUST | >= {SWITCH_YEAR_MIN} yr wins",
        "=" * len(header), "",
        header, sep,
    ]

    for row in results:
        switch_flag = ("*** SWITCH CANDIDATE ***" if row.get("switch_all_met")
                       else f"({row.get('switch_n_met', 0)}/5 gates)")
        mc_v = row["mc_verdict"] if row["mc_verdict"] is not None else "(alloc)"
        line = (
            f"{row['group']:<12} {row['eq_obj']:<16} {row['bd_obj']:<16} "
            f"{row['alloc_obj']:<16} "
            f"{_pct(row['portfolio_cagr']):>10} {_f(row['portfolio_calmar']):>12} "
            f"{_pct(row['portfolio_maxdd']):>11} {_f(row['portfolio_sharpe']):>12} "
            f"{str(row['n_reallocations']):>10} {str(mc_v):>11} "
            f"{str(row.get('year_win_count', 'N/A')):>8} "
            f"{str(row.get('switch_n_met', '')):>6} {switch_flag}"
        )
        lines.append(line)
        if row["group"] != "incumbent" and row.get("switch_detail"):
            lines.append(f"{'':>14}  gates: {row['switch_detail']}")
        if row.get("year_win_detail"):
            lines.append(f"{'':>14}  years: {row['year_win_detail']}")
        lines.append("")

    lines += [sep, ""]

    candidates  = [r for r in results if r.get("switch_all_met")]
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

    lines += ["", "=" * len(header)]
    full_text = "\n".join(lines)

    for line in lines:
        logging.info(line)

    txt_path = "objective_review_decision_table.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")
    logging.info("Decision table written to %s", txt_path)


# ============================================================
# RUN SETTINGS — edit before running
# ============================================================

N_MC    = 1000   # 10 for smoke test, 1000 for production
RUN_MC  = True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_review(n_mc=N_MC, run_mc_flag=RUN_MC)
    logging.info("Review complete.")
