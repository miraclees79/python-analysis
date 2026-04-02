"""
asset_config_sweep.py
=====================
Systematic sweep over (asset × train/test window) configurations to identify
which combinations are worth promoting to full bootstrap analysis and daily
monitoring.

For each (asset, train_years, test_years) configuration the script runs:
  1. Walk-forward optimisation (WF)
  2. MC parameter perturbation robustness (n_mc samples)
  3. Regime decomposition (ADX trend, momentum, volatility)

Results are collected into two output files written to sweep_outputs/:

  sweep_results_YYYYMMDD_HHMM.csv   — one row per (asset, train, test).
                                       Also written as sweep_results_latest.csv.
                                       Primary artefact for candidate selection.

  sweep_windows_YYYYMMDD_HHMM.csv   — one row per WF window within each config
                                       (asset, train, test, window_idx, ...).
                                       Useful for inspecting parameter stability
                                       and filter_mode consistency.

-----------------------------------------------------------------------
CANDIDATE SELECTION GUIDE
-----------------------------------------------------------------------
Open sweep_results_latest.csv in a spreadsheet.

Primary gates (all must pass for candidate=True):
  mc_verdict        == "ROBUST"
  oos_calmar        >= 0.30
  oos_maxdd         >= -40.0   (stored as %, e.g. -35.2)
  mc_cagr_p05       >= 0.0     (positive in 95% of MC universes)

Secondary indicators (no hard gate — use to rank candidates):
  cagr_vs_bh        : strategy CAGR minus B&H CAGR (pp); prefer > 3pp
  mc_cagr_median    : median universe CAGR; prefer > 5%
  filter_consistency: % of windows with dominant filter; prefer > 60%
  param_change_count: number of window-to-window parameter changes; lower = more stable
  adx_uptrend_strat_cagr vs adx_uptrend_bh_cagr: does strategy capture uptrends?
  adx_downtrend_strat_cagr: how much does strategy lose in downtrends?
  window_cagr_min   : worst single OOS window; prefer > -10%

After candidate selection, run bootstrap on chosen configs:
  run_block_bootstrap_robustness() from mc_robustness.py
  with n_samples=500 on each selected config.

-----------------------------------------------------------------------
ASSETS
-----------------------------------------------------------------------
Stooq-native (no Drive needed):
  WIG20TR, MWIG40TR, SWIG80TR, SP500, NASDAQ100, Nikkei225

Drive-backed (require GDRIVE_FOLDER_ID env var or GDRIVE_FOLDER_ID_DEFAULT):
  STOXX600, MSCI_World
  Skipped with a warning if folder ID is not available.

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
  # List available assets and window configs, then exit
  python asset_config_sweep.py --list

  # Smoke test — n_mc=20, no regime, fast verification
  python asset_config_sweep.py --mode smoke

  # Full sweep — n_mc=1000, with regime decomp
  python asset_config_sweep.py --mode full

  # Custom n_mc
  python asset_config_sweep.py --mode full --n_mc 500

  # Targeted re-run (specific assets only)
  python asset_config_sweep.py --assets WIG20TR SP500 --mode full

  # Skip regime decomp (saves ~20% runtime)
  python asset_config_sweep.py --mode full --no_regime

-----------------------------------------------------------------------
RUNTIME ESTIMATE
-----------------------------------------------------------------------
Full sweep (8 assets × 5 window configs = 40 configs):
  WF per config            : ~2-4 min on 4-core machine
  MC n=1000 per config     : ~8-15 min on 4-core machine
  Regime decomp per config : ~30 sec
  Total                    : ~6-8h on a 4-core machine
  Recommended: run overnight.

Smoke test (n_mc=20): ~30 min total.  Use to verify all assets load
and the pipeline runs end-to-end before committing to a full run.
"""

import argparse
import datetime as dt
import logging
import os
import sys
import tempfile
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy machinery — same imports as strat_test_runfile_with_MC.py
# ---------------------------------------------------------------------------
from strategy_test_library import (
    load_stooq_local,       # consolidated loader — replaces per-runfile _stooq()
    load_csv,
    walk_forward,
    compute_metrics,
    compute_buy_and_hold,
    prepare_regime_inputs,
    run_regime_decomposition,
)
from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    EQUITY_THRESHOLDS_MC,
)
from global_equity_library import build_mmf_extended
from price_series_builder import build_and_upload
from stooq_hybrid_updater import run_update

# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "sweep_run.log"
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
# CONFIGURATION
# ============================================================

# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------
GDRIVE_FOLDER_ID_DEFAULT = ""   # fallback if GDRIVE_FOLDER_ID env var not set

ASSET_REGISTRY = {
    "WIG20TR": {
        "source": "stooq",
        "ticker": "wig20tr",
    },
    "MWIG40TR": {
        "source": "stooq",
        "ticker": "mwig40tr",
    },
    "SWIG80TR": {
        "source": "stooq",
        "ticker": "swig80tr",
    },
    "SP500": {
        "source": "stooq",
        "ticker": "sp500",
    },
    "NASDAQ100": {
        "source": "stooq",
        "ticker": "ndq100",
    },
    "Nikkei225": {
        "source": "stooq",
        "ticker": "nk225",
    },
    "STOXX600": {
        "source": "drive",
        "drive_kwargs": {
            "raw_filename":      "stoxx600.csv",
            "combined_filename": "stoxx600_combined.csv",
            "extension_ticker":  "^STOXX",
            "extension_source":  "yfinance",
        },
    },
    "MSCI_World": {
        "source": "drive",
        "drive_kwargs": {
            "raw_filename":      "msci_world_wsj_raw.csv",
            "combined_filename": "msci_world_combined.csv",
            "extension_ticker":  "URTH",
            "extension_source":  "yfinance",
            "raw_csv_format":    "wsj",
        },
    },
}

# Default asset order — all 8 assets.  CLI --assets overrides this.
DEFAULT_ASSETS = [
    "WIG20TR", "MWIG40TR", "SWIG80TR",
    "SP500", "NASDAQ100", "Nikkei225",
    "STOXX600", "MSCI_World",
]

# ---------------------------------------------------------------------------
# Window configurations  (train_years, test_years)
# ---------------------------------------------------------------------------
WINDOW_CONFIGS = [
    (6, 1),
    (6, 2),
    (7, 1),
    (7, 2),
    (8, 1),
    (8, 2),
    (9, 1),
    (9, 2),
]

# ---------------------------------------------------------------------------
# Shared strategy settings — identical to strat_test_runfile_with_MC.py
# ---------------------------------------------------------------------------
POSITION_MODE     = "full"
VOL_WINDOW        = 20
FORCE_FILTER_MODE = ["ma", "mom", "mom_blend"]
OBJECTIVE         = "calmar"

X_GRID      = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID      = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_GRID   = [50, 75, 100]
SLOW_GRID   = [150, 200, 250]
TV_GRID     = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_GRID     = [0.05, 0.08, 0.10, 0.15]
MOM_LB_GRID = [252]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAST_MODE  = True

# ---------------------------------------------------------------------------
# STOP MODE CONFIGURATIONS  ← ADD THIS BLOCK after the existing grids
# ---------------------------------------------------------------------------

N_ATR_GRID     = [0.08, 0.10, 0.12, 0.15, 0.20]   # same scale as X_GRID
ATR_WINDOW     = 20

STOP_MODE_CONFIGS = [
    {
        "stop_mode":    "fixed",
        "use_atr_stop": False,
        "atr_window":   ATR_WINDOW,
        "N_atr_grid":   None,
    },
    {
        "stop_mode":    "atr",
        "use_atr_stop": True,
        "atr_window":   ATR_WINDOW,
        "N_atr_grid":   N_ATR_GRID,
    },
]

# ---------------------------------------------------------------------------
# Data floor dates
# ---------------------------------------------------------------------------
WIG_DATA_FLOOR = "1995-01-02"
MMF_FLOOR      = "1995-01-02"
DATA_START     = "1990-01-01"


# ---------------------------------------------------------------------------
# Candidate selection thresholds
# All four primary gates must pass for candidate=True.
# ---------------------------------------------------------------------------
CANDIDATE_MIN_CALMAR    = 0.30
CANDIDATE_MIN_MAXDD_PCT = -40.0  # oos_maxdd stored as %; floor is -40%
CANDIDATE_MC_VERDICT    = "ROBUST"
CANDIDATE_MIN_P05_CAGR  = 0.00   # MC p05 CAGR >= 0


# ============================================================
# HELPERS — DATA LOADING
# ============================================================

def _cpu_jobs() -> int:
    n = os.cpu_count() or 1
    return max(1, n - 1) if n > 3 and sys.platform == "win32" else n





def _load_asset(asset_name: str, tmp_dir: str) -> pd.DataFrame | None:
    cfg = ASSET_REGISTRY.get(asset_name)
    if cfg is None:
        logging.error("Unknown asset '%s'. Not in ASSET_REGISTRY.", asset_name)
        return None

    if cfg["source"] == "stooq":
        return load_stooq_local(cfg["ticker"], asset_name, DATA_DIR, DATA_START)

    if cfg["source"] == "drive":
        folder_id = (
            os.environ.get("GDRIVE_FOLDER_ID", "").strip()
            or GDRIVE_FOLDER_ID_DEFAULT.strip()
        )
        if not folder_id:
            logging.warning(
                "SKIP: %s — GDRIVE_FOLDER_ID not set. "
                "Set env var or GDRIVE_FOLDER_ID_DEFAULT to include Drive-backed assets.",
                asset_name,
            )
            return None
        kw = dict(cfg["drive_kwargs"])
        kw["folder_id"] = folder_id
        df = build_and_upload(**kw)
        if df is None:
            logging.warning(
                "SKIP: %s — build_and_upload returned None. "
                "Ensure the raw CSV is uploaded to Drive folder %s.",
                asset_name, folder_id,
            )
        return df

    logging.error("Unknown source '%s' for asset '%s'.", cfg["source"], asset_name)
    return None


def _load_cash(tmp_dir: str) -> pd.DataFrame | None:
    """Load MMF and extend backwards with WIBOR 1M if available."""
    mmf = load_stooq_local("fund_2720",  "MMF", DATA_DIR, DATA_START)
    if mmf is None:
        logging.error("Failed to load MMF — cannot run sweep without cash series.")
        return None
    wibor1m =  load_stooq_local("wibor1m", "WIBOR1M", DATA_DIR, DATA_START, mandatory=False)
    if wibor1m is not None:
        mmf = build_mmf_extended(mmf, wibor1m, floor_date=MMF_FLOOR)
        logging.info(
            "MMF extended back to %s (%d rows total).",
            mmf.index.min().date(), len(mmf),
        )
    else:
        logging.warning(
            "WIBOR1M unavailable — MMF runs from %s. "
            "IS windows before that date use ret_mmf=0 for cash returns.",
            mmf.index.min().date(),
        )
    return mmf


# ============================================================
# PER-WINDOW SUMMARY
# ============================================================

def _extract_window_rows(
    asset_name:  str,
    train_years: int,
    test_years:  int,
    wf_results:  pd.DataFrame,
    stop_mode:   str = "fixed",       # ← NEW PARAMETER
) -> list[dict]:
    rows = []
    prev_params = None

    use_atr = stop_mode == "atr"
    stop_col = "N_atr" if use_atr else "X"   # ← pick correct column

    for idx, row in wf_results.iterrows():
        cur_params = {
            "filter_mode": str(row.get("filter_mode", "")),
            "stop_param":  float(row.get(stop_col, np.nan)),   # ← unified name
            "Y":           float(row.get("Y",          np.nan)),
            "fast":        int(row.get("fast",          0)),
            "slow":        int(row.get("slow",          0)),
            "stop_loss":   float(row.get("stop_loss",   np.nan)),
        }

        param_changed = False
        if prev_params is not None:
            param_changed = any(cur_params[k] != prev_params[k] for k in cur_params)

        rows.append({
            "asset":         asset_name,
            "train_years":   train_years,
            "test_years":    test_years,
            "stop_mode":     stop_mode,              # ← NEW
            "window_idx":    idx,
            "train_start":   str(pd.Timestamp(row["TrainStart"]).date()),
            "test_start":    str(pd.Timestamp(row["TestStart"]).date()),
            "test_end":      str(pd.Timestamp(row["TestEnd"]).date()),
            "win_cagr":      round(float(row.get("CAGR",   np.nan)) * 100, 2),
            "win_sharpe":    round(float(row.get("Sharpe", np.nan)),        3),
            "win_maxdd":     round(float(row.get("MaxDD",  np.nan)) * 100,  2),
            "win_calmar":    round(float(row.get("CalMAR", np.nan)),         3),
            "filter_mode":   cur_params["filter_mode"],
            "stop_param":    cur_params["stop_param"],  # ← replaces "X"
            "Y":             cur_params["Y"],
            "fast":          cur_params["fast"],
            "slow":          cur_params["slow"],
            "stop_loss":     cur_params["stop_loss"],
            "param_changed": param_changed,
        })
        prev_params = cur_params

    return rows


# ============================================================
# REGIME DECOMPOSITION SUMMARY EXTRACTOR
# ============================================================

def _extract_regime_stats(regime_results: dict) -> dict:
    """
    Flatten regime decomposition output (from run_regime_decomposition) into
    scalar columns for the summary CSV.

    ADX trend labeller (uptrend / downtrend / sideways):
      adx_{reg}_strat_cagr, adx_{reg}_bh_cagr, adx_{reg}_strat_sharpe,
      adx_{reg}_pct_time

    Volatility labeller (high_vol / normal_vol / low_vol):
      vol_{reg}_strat_cagr, vol_{reg}_bh_cagr, vol_{reg}_pct_time
    """
    out = {}
    if not regime_results:
        return out

    # ADX trend
    adx_stats = regime_results.get("ADX trend", {}).get("stats")
    if adx_stats is not None and not adx_stats.empty:
        for reg in ["uptrend", "downtrend", "sideways"]:
            if reg in adx_stats.index:
                r = adx_stats.loc[reg]
                out[f"adx_{reg}_strat_cagr"]  = round(float(r.get("strat_cagr",   np.nan)), 2)
                out[f"adx_{reg}_bh_cagr"]      = round(float(r.get("bh_cagr",      np.nan)), 2)
                out[f"adx_{reg}_strat_sharpe"] = round(float(r.get("strat_sharpe", np.nan)), 3)
                out[f"adx_{reg}_pct_time"]      = round(float(r.get("pct_time",     np.nan)), 1)
            else:
                for col in ["strat_cagr", "bh_cagr", "strat_sharpe", "pct_time"]:
                    out[f"adx_{reg}_{col}"] = np.nan

    # Volatility
    vol_stats = regime_results.get("Volatility", {}).get("stats")
    if vol_stats is not None and not vol_stats.empty:
        for reg in ["high_vol", "normal_vol", "low_vol"]:
            if reg in vol_stats.index:
                r = vol_stats.loc[reg]
                out[f"vol_{reg}_strat_cagr"] = round(float(r.get("strat_cagr", np.nan)), 2)
                out[f"vol_{reg}_bh_cagr"]    = round(float(r.get("bh_cagr",    np.nan)), 2)
                out[f"vol_{reg}_pct_time"]   = round(float(r.get("pct_time",   np.nan)), 1)
            else:
                for col in ["strat_cagr", "bh_cagr", "pct_time"]:
                    out[f"vol_{reg}_{col}"] = np.nan

    return out


# ============================================================
# CANDIDATE FLAG
# ============================================================

def _is_candidate(row: dict) -> bool:
    """Return True if all four primary candidate gates are satisfied."""
    if row.get("error_msg"):
        return False

    calmar   = row.get("oos_calmar",   np.nan)
    maxdd    = row.get("oos_maxdd",    np.nan)   # stored as %, e.g. -35.2
    verdict  = row.get("mc_verdict",   "")
    p05_cagr = row.get("mc_cagr_p05", np.nan)   # stored as fraction, e.g. 0.05

    if np.isnan(calmar) or np.isnan(maxdd):
        return False
    if calmar < CANDIDATE_MIN_CALMAR:
        return False
    if maxdd < CANDIDATE_MIN_MAXDD_PCT:
        return False
    if verdict not in (CANDIDATE_MC_VERDICT, "SKIPPED"):
        return False
    # p05_cagr is stored as fraction (e.g. 0.05 = 5%) from analyze_robustness
    if not np.isnan(p05_cagr) and p05_cagr < CANDIDATE_MIN_P05_CAGR:
        return False

    return True


# ============================================================
# SINGLE CONFIGURATION RUN
# ============================================================

def run_config(
    asset_name:   str,
    df:           pd.DataFrame,
    cash_df:      pd.DataFrame,
    train_years:  int,
    test_years:   int,
    stop_cfg:     dict,              # ← NEW: replaces implicit fixed-stop
    n_mc:         int,
    run_regime:   bool,
    n_jobs:       int,
) -> tuple[dict, list[dict]]:

    stop_mode    = stop_cfg["stop_mode"]
    use_atr_stop = stop_cfg["use_atr_stop"]
    atr_window   = stop_cfg["atr_window"]
    N_atr_grid   = stop_cfg["N_atr_grid"]

    label = f"{asset_name} [{train_years}+{test_years}] [{stop_mode}]"  # ← include stop_mode
    logging.info("")
    logging.info("=" * 70)
    logging.info("CONFIG: %s", label)
    logging.info("=" * 70)

    summary = {
        "asset":       asset_name,
        "train_years": train_years,
        "test_years":  test_years,
        "stop_mode":   stop_mode,    # ← NEW column
        "error_msg":   "",
        "candidate":   False,
    }

    # ── Walk-forward ─────────────────────────────────────────────────────
    try:
        wf_equity, wf_results, wf_trades = walk_forward(
            df                    = df,
            cash_df               = cash_df,
            train_years           = train_years,
            test_years            = test_years,
            vol_window            = VOL_WINDOW,
            selected_mode         = POSITION_MODE,
            filter_modes_override = FORCE_FILTER_MODE,
            X_grid                = X_GRID,
            Y_grid                = Y_GRID,
            fast_grid             = FAST_GRID,
            slow_grid             = SLOW_GRID,
            tv_grid               = TV_GRID,
            sl_grid               = SL_GRID,
            mom_lookback_grid     = MOM_LB_GRID,
            objective             = OBJECTIVE,
            n_jobs                = n_jobs,
            fast_mode             = True,
            use_atr_stop          = use_atr_stop,      # ← NEW
            N_atr_grid            = N_atr_grid,         # ← NEW (None for fixed)
            atr_window            = atr_window,         # ← NEW
        )
    except Exception as exc:
        logging.error("%s: walk-forward raised exception — %s", label, exc)
        summary["error_msg"] = f"wf_exception: {exc}"
        return summary, []

    # ... rest of run_config is identical, just pass stop_mode to
    # _extract_window_rows and carry it in summary ...

    if wf_equity.empty or wf_results.empty:
        logging.warning("%s: walk-forward produced no results.", label)
        summary["error_msg"] = "wf_empty"
        return summary, []

    oos_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    summary.update({
        "oos_cagr":     round(oos_metrics["CAGR"]    * 100, 2),
        "oos_vol":      round(oos_metrics["Vol"]     * 100, 2),
        "oos_sharpe":   round(oos_metrics["Sharpe"],         3),
        "oos_sortino":  round(oos_metrics["Sortino"],        3),
        "oos_maxdd":    round(oos_metrics["MaxDD"]   * 100,  2),
        "oos_calmar":   round(oos_metrics["CalMAR"],         3),
        "n_wf_windows": len(wf_results),
        "oos_start":    str(wf_results["TestStart"].min().date()),
        "oos_end":      str(wf_results["TestEnd"].max().date()),
    })

    oos_start = wf_results["TestStart"].min()
    oos_end   = wf_results["TestEnd"].max()
    bh_equity = None

    try:
        bh_equity, bh_metrics = compute_buy_and_hold(
            df, price_col="Zamkniecie", start=oos_start, end=oos_end
        )
        summary.update({
            "bh_cagr":    round(bh_metrics["CAGR"]    * 100, 2),
            "bh_sharpe":  round(bh_metrics["Sharpe"],         3),
            "bh_maxdd":   round(bh_metrics["MaxDD"]   * 100,  2),
            "bh_calmar":  round(bh_metrics["CalMAR"],         3),
            "cagr_vs_bh": round(
                (oos_metrics["CAGR"] - bh_metrics["CAGR"]) * 100, 2
            ),
        })
    except Exception as exc:
        logging.warning("%s: B&H computation failed (non-fatal) — %s", label, exc)
        for col in ("bh_cagr", "bh_sharpe", "bh_maxdd", "bh_calmar", "cagr_vs_bh"):
            summary[col] = np.nan

    # ── Per-window rows ───────────────────────────────────────────────────
    window_rows = _extract_window_rows(
        asset_name, train_years, test_years, wf_results,
        stop_mode=stop_mode,         # ← pass stop_mode
    )

    if window_rows:
        win_cagrs = [w["win_cagr"] for w in window_rows if not np.isnan(w["win_cagr"])]
        summary["window_cagr_mean"]    = round(np.mean(win_cagrs),  2) if win_cagrs else np.nan
        summary["window_cagr_std"]     = round(np.std(win_cagrs),   2) if len(win_cagrs) > 1 else np.nan
        summary["window_cagr_min"]     = round(np.min(win_cagrs),   2) if win_cagrs else np.nan
        summary["window_cagr_max"]     = round(np.max(win_cagrs),   2) if win_cagrs else np.nan
        summary["param_change_count"]  = sum(1 for w in window_rows if w["param_changed"])

        filter_modes = [w["filter_mode"] for w in window_rows if w["filter_mode"]]
        if filter_modes:
            dominant_mode, dominant_count = Counter(filter_modes).most_common(1)[0]
            summary["dominant_filter"]    = dominant_mode
            summary["filter_consistency"] = round(
                dominant_count / len(filter_modes) * 100, 1
            )
        else:
            summary["dominant_filter"]    = "unknown"
            summary["filter_consistency"] = np.nan
    else:
        for col in ("window_cagr_mean", "window_cagr_std", "window_cagr_min",
                    "window_cagr_max", "param_change_count",
                    "dominant_filter", "filter_consistency"):
            summary[col] = np.nan

    # ── MC robustness ─────────────────────────────────────────────────────
    # (MC columns setup, _set_mc_nan, and the MC run block are all unchanged)
    # mc_results from extract_best_params_from_wf_results already handles
    # ATR mode via the use_atr_stop flag stored in wf_results — no change needed

    MC_METRICS  = ["cagr", "sharpe", "maxdd", "calmar"]
    MC_STATS    = ["p05", "p25", "median", "p75", "p95", "p_worse"]

    def _set_mc_nan():
        for m in MC_METRICS:
            for s in MC_STATS:
                summary[f"mc_{m}_{s}"] = np.nan
        summary["mc_p_loss"]    = np.nan
        summary["mc_n_samples"] = 0

    if n_mc > 0:
        try:
            windows     = extract_windows_from_wf_results(wf_results, train_years=train_years)
            best_params = extract_best_params_from_wf_results(wf_results)

            mc_df = run_monte_carlo_robustness(
                best_params   = best_params,
                windows       = windows,
                df            = df,
                cash_df       = cash_df,
                vol_window    = VOL_WINDOW,
                selected_mode = POSITION_MODE,
                funds_df      = None,
                n_samples     = n_mc,
                n_jobs        = n_jobs,
                perturb_pct   = 0.20,
                seed          = 42,
                price_col     = "Zamkniecie",
            )

            mc_sum = analyze_robustness(
                mc_df,
                oos_metrics,
                thresholds=EQUITY_THRESHOLDS_MC,
            )

            summary["mc_verdict"]   = mc_sum.get("verdict", "UNKNOWN")
            summary["mc_n_samples"] = len(mc_df)
            summary["mc_p_loss"]    = round(float(mc_sum.get("p_loss", np.nan)), 4)

            METRIC_KEY_MAP = {"cagr": "CAGR", "sharpe": "Sharpe", "maxdd": "MaxDD", "calmar": "CalMAR"}
            STAT_KEY_MAP   = {"p05": "p05", "p25": "p25", "median": "median",
                              "p75": "p75", "p95": "p95", "p_worse": "p_worse_than_baseline"}
            for m in MC_METRICS:
                ms = mc_sum.get(METRIC_KEY_MAP[m], {})
                for s in MC_STATS:
                    summary[f"mc_{m}_{s}"] = round(float(ms.get(STAT_KEY_MAP[s], np.nan)), 4)

            logging.info(
                "  MC (%d samples): verdict=%s  p05_CAGR=%.2f%%  p_loss=%.1f%%",
                len(mc_df), summary["mc_verdict"],
                summary["mc_cagr_p05"] * 100, summary["mc_p_loss"] * 100,
            )

        except Exception as exc:
            logging.error("%s: MC robustness raised exception — %s", label, exc)
            summary["mc_verdict"] = "ERROR"
            summary["error_msg"]  = (
                (summary["error_msg"] + " | " if summary["error_msg"] else "")
                + f"mc_exception: {exc}"
            )
            _set_mc_nan()
    else:
        summary["mc_verdict"] = "SKIPPED"
        _set_mc_nan()

    # ── Regime decomposition ─────────────────────────────────────────────
    # unchanged — regime is independent of stop mode
    if run_regime:
        if bh_equity is not None:
            try:
                inputs = prepare_regime_inputs(df, wf_results, wf_equity, bh_equity)
                regime_results = run_regime_decomposition(
                    **{k: inputs[k] for k in [
                        "close", "high", "low",
                        "daily_returns_strat", "daily_returns_bh",
                        "equity_strat", "equity_bh",
                    ]}
                )
                summary.update(_extract_regime_stats(regime_results))
            except Exception as exc:
                logging.warning("%s: regime decomposition failed (non-fatal) — %s", label, exc)

    summary["candidate"] = _is_candidate(summary)

    logging.info(
        "RESULT  %-36s  CAGR=%+5.1f%%  CalMAR=%5.2f  MaxDD=%5.1f%%  "
        "MC=%-8s  candidate=%s",
        label,
        summary.get("oos_cagr",   np.nan),
        summary.get("oos_calmar", np.nan),
        summary.get("oos_maxdd",  np.nan),
        summary.get("mc_verdict", "N/A"),
        summary["candidate"],
    )
    return summary, window_rows
# ============================================================
# MAIN SWEEP LOOP
# ============================================================

def run_sweep(
    assets:     list[str],
    n_mc:       int,
    run_regime: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    tmp_dir = tempfile.mkdtemp()
    n_jobs  = _cpu_jobs()
    # Total now: assets × window configs × stop modes
    total   = len(assets) * len(WINDOW_CONFIGS) * len(STOP_MODE_CONFIGS)

    logging.info("=" * 80)
    logging.info("ASSET CONFIG SWEEP  START: %s", dt.datetime.now())
    logging.info("  Assets       : %s", assets)
    logging.info("  Window configs: %s", WINDOW_CONFIGS)
    logging.info("  Stop modes   : %s", [c["stop_mode"] for c in STOP_MODE_CONFIGS])
    logging.info("  n_mc         : %d", n_mc)
    logging.info("  run_regime   : %s", run_regime)
    logging.info("  n_jobs       : %d", n_jobs)
    logging.info("  Total configs: %d", total)
    logging.info("=" * 80)

    logging.info("Loading MMF cash series ...")
    cash_df = _load_cash(tmp_dir)
    if cash_df is None:
        logging.error("Cannot load MMF — aborting sweep.")
        return pd.DataFrame(), pd.DataFrame()

    all_summaries = []
    all_windows   = []
    run_count     = 0

    tmp_dir = tempfile.gettempdir()
    run_update(get_funds=False)



    for asset_name in assets:
        logging.info("")
        logging.info("▌ ASSET: %s", asset_name)

        df = _load_asset(asset_name, tmp_dir)
        if df is None:
            logging.warning(
                "  Skipping all configs for %s — data unavailable.",
                asset_name,
            )
            for train_y, test_y in WINDOW_CONFIGS:
                for stop_cfg in STOP_MODE_CONFIGS:
                    all_summaries.append({
                        "asset":       asset_name,
                        "train_years": train_y,
                        "test_years":  test_y,
                        "stop_mode":   stop_cfg["stop_mode"],
                        "error_msg":   "data_unavailable",
                        "candidate":   False,
                    })
            continue

        for train_years, test_years in WINDOW_CONFIGS:
            for stop_cfg in STOP_MODE_CONFIGS:   # ← NEW inner loop
                run_count += 1
                logging.info(
                    "  [%d/%d]  %s  [%d+%d]  [%s] ...",
                    run_count, total, asset_name,
                    train_years, test_years, stop_cfg["stop_mode"],
                )
                summary, win_rows = run_config(
                    asset_name  = asset_name,
                    df          = df,
                    cash_df     = cash_df,
                    train_years = train_years,
                    test_years  = test_years,
                    stop_cfg    = stop_cfg,      # ← pass stop config
                    n_mc        = n_mc,
                    run_regime  = run_regime,
                    n_jobs      = n_jobs,
                )
                all_summaries.append(summary)
                all_windows.extend(win_rows)

    summary_df = pd.DataFrame(all_summaries)
    windows_df = pd.DataFrame(all_windows)

    if not summary_df.empty and "oos_calmar" in summary_df.columns:
        summary_df = summary_df.sort_values(
            ["candidate", "oos_calmar"], ascending=[False, False]
        ).reset_index(drop=True)

    return summary_df, windows_df

# ============================================================
# REPORTING AND SAVING
# ============================================================

def _print_sweep_summary(summary_df: pd.DataFrame) -> None:
    """Log a compact candidate selection table."""
    if summary_df.empty:
        logging.warning("No results to summarise.")
        return

    DISPLAY_COLS = [
        "asset", "train_years", "test_years",
        "oos_cagr", "oos_calmar", "oos_maxdd", "oos_sharpe",
        "bh_cagr", "cagr_vs_bh",
        "mc_verdict", "mc_cagr_p05", "mc_cagr_median", "mc_p_loss",
        "dominant_filter", "filter_consistency",
        "window_cagr_min", "param_change_count",
        "candidate", "error_msg",
    ]
    present = [c for c in DISPLAY_COLS if c in summary_df.columns]

    sep = "=" * 120
    logging.info("\n%s\n  SWEEP SUMMARY\n%s", sep, sep)
    logging.info("\n%s", summary_df[present].to_string(index=False))

    candidates = summary_df[summary_df.get("candidate", False) == True]
    logging.info("\n%s", sep)
    logging.info(
        "  %d / %d CONFIGURATIONS PASSED ALL CANDIDATE CRITERIA",
        len(candidates), len(summary_df),
    )
    logging.info(
        "  Criteria: MC=%s  CalMAR≥%.2f  MaxDD≥%.1f%%  MC_p05_CAGR≥%.1f%%",
        CANDIDATE_MC_VERDICT, CANDIDATE_MIN_CALMAR,
        CANDIDATE_MIN_MAXDD_PCT, CANDIDATE_MIN_P05_CAGR * 100,
    )
    logging.info("%s", sep)

    if not candidates.empty:
        for stop_mode, grp in candidates.groupby("stop_mode"):
            logging.info("  Stop mode: %s", stop_mode)
            for _, r in grp.iterrows():
                p05_pct = (
                    round(r.get("mc_cagr_p05", np.nan) * 100, 1)
                    if not np.isnan(r.get("mc_cagr_p05", np.nan)) else np.nan
                )
                logging.info(
                    "    ✓  %-12s [%d+%d]  CAGR=%+5.1f%%  CalMAR=%5.2f  MaxDD=%5.1f%%  "
                    "MC=%-8s  p05=%.1f%%  filter=%s(%.0f%%)",
                    r["asset"], r["train_years"], r["test_years"],
                    r.get("oos_cagr",          np.nan),
                    r.get("oos_calmar",         np.nan),
                    r.get("oos_maxdd",          np.nan),
                    r.get("mc_verdict",         "N/A"),
                    p05_pct,
                    r.get("dominant_filter",    "?"),
                    r.get("filter_consistency", np.nan),
            )
    else:
        logging.info("  No configurations passed all criteria.")
        logging.info(
            "  Consider relaxing CANDIDATE_MIN_CALMAR (currently %.2f) "
            "or CANDIDATE_MIN_P05_CAGR (currently %.2f) in the config block.",
            CANDIDATE_MIN_CALMAR, CANDIDATE_MIN_P05_CAGR,
        )

    logging.info("%s", sep)


def _save_results(
    summary_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    timestamp:  str,
) -> None:
    """Write timestamped and latest CSV copies to sweep_outputs/."""
    os.makedirs("sweep_outputs", exist_ok=True)

    for df, stem in [(summary_df, "sweep_results"), (windows_df, "sweep_windows")]:
        if df is None or df.empty:
            logging.warning("%s is empty — not writing CSV.", stem)
            continue
        ts_path     = f"sweep_outputs/{stem}_{timestamp}.csv"
        latest_path = f"sweep_outputs/{stem}_latest.csv"
        df.to_csv(ts_path,     index=False, sep=";", encoding="utf-8")
        df.to_csv(latest_path, index=False, sep=";", encoding="utf-8")
        logging.info(
            "Saved %s: %s  (%d rows) [also -> %s]",
            stem, ts_path, len(df), latest_path,
        )


# ============================================================
# ENTRY POINT
# ============================================================

def _list_configs() -> None:
    print("\nAvailable assets:")
    for name, cfg in ASSET_REGISTRY.items():
        src    = cfg["source"]
        detail = cfg.get("ticker") or cfg.get("drive_kwargs", {}).get("combined_filename", "")
        note   = "  [requires GDRIVE_FOLDER_ID]" if src == "drive" else ""
        print(f"  {name:<14} ({src}: {detail}){note}")
    print("\nWindow configurations (train_years + test_years):")
    for t, v in WINDOW_CONFIGS:
        print(f"  {t}+{v}")
    print(f"\nDefault sweep assets: {DEFAULT_ASSETS}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep asset × window configs for strategy candidate selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="full",
        help=(
            "smoke = n_mc=20, no regime (fast verification);\n"
            "full  = n_mc=1000, with regime decomp (production run)"
        ),
    )
    parser.add_argument(
        "--n_mc", type=int, default=None,
        help="Override n_mc (takes precedence over --mode default).",
    )
    parser.add_argument(
        "--assets", nargs="+", default=None,
        help=(
            "Asset keys to include. Must match ASSET_REGISTRY keys. "
            "Default: all 8 assets. E.g. --assets WIG20TR SP500"
        ),
    )
    parser.add_argument(
        "--no_regime", action="store_true",
        help="Skip regime decomposition (~20%% faster).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available assets and window configs, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        _list_configs()
        sys.exit(0)

    # Resolve run settings
    if args.mode == "smoke":
        n_mc       = args.n_mc if args.n_mc is not None else 20
        run_regime = False   # always skip in smoke mode
    else:
        n_mc       = args.n_mc if args.n_mc is not None else 1000
        run_regime = not args.no_regime

    assets = args.assets if args.assets else DEFAULT_ASSETS

    unknown = [a for a in assets if a not in ASSET_REGISTRY]
    if unknown:
        logging.error(
            "Unknown asset(s): %s\nAvailable: %s",
            unknown, list(ASSET_REGISTRY.keys()),
        )
        sys.exit(1)

    logging.info(
        "Settings: mode=%s  n_mc=%d  run_regime=%s  assets=%s",
        args.mode, n_mc, run_regime, assets,
    )

    # Run
    summary_df, windows_df = run_sweep(
        assets     = assets,
        n_mc       = n_mc,
        run_regime = run_regime,
    )

    if summary_df.empty:
        logging.error("Sweep produced no results.")
        sys.exit(1)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    _save_results(summary_df, windows_df, timestamp)
    _print_sweep_summary(summary_df)

    logging.info("SWEEP COMPLETE: %s", dt.datetime.now())
    logging.info("Results in sweep_outputs/")
    logging.info("Full log in %s", LOG_FILE)


if __name__ == "__main__":
    main()
