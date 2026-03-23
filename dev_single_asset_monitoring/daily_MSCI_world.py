"""
daily_runfile_generic.py
========================
Generic daily monitoring runfile for any single equity index.
Mirrors daily_runfile_WIG20TR.py in structure — same walk-forward,
same daily_output module, same Drive upload pattern — but driven by
the ASSET_CONFIG block at the top of this file.

DEPLOYMENT CHECKLIST
--------------------
1. Set ASSET_NAME to the asset label (must be a key in ASSET_REGISTRY).
2. Set OUTPUT_PREFIX to a unique short string (lowercase, no spaces).
   This prefix is prepended to all output filenames and log file names,
   ensuring co-existence with other strategies in the same Drive folder.
3. Set TRAIN_YEARS / TEST_YEARS from sweep results.
4. Optionally override FORCE_FILTER_MODE if the sweep showed a dominant
   filter (e.g. ["ma"] for bond-like or index-agnostic series).
5. Copy daily_generic_asset.yml, rename it, and set OUTPUT_PREFIX in
   the workflow's env block to match the value set here.

OUTPUT FILES
------------
All written to outputs/ directory:
  {OUTPUT_PREFIX}_signal_status.txt
  {OUTPUT_PREFIX}_signal_log.csv
  {OUTPUT_PREFIX}_equity_chart.png
  {OUTPUT_PREFIX}_signal_snapshot.json
  {OUTPUT_PREFIX}_DailyRun.log

NOTES
-----
- daily_output.py (from WIG20TR) writes hardcoded filenames internally.
  This script writes to an isolated subdirectory (outputs/{OUTPUT_PREFIX}/)
  and copies the four files to outputs/{PREFIX}_*.ext for Drive upload.
  daily_output.py is never modified.
- The Drive log pre-fetch (before action determination) is preserved:
  build_daily_outputs() downloads {PREFIX}_signal_log.csv from Drive
  before writing today's row, ensuring correct ENTER/EXIT/HOLD on
  fresh GitHub Actions runners where outputs/ is always empty.
  The Drive filename must match the prefixed name — the GH Actions
  workflow is responsible for uploading the prefixed files.
"""

import os
import sys
import shutil
import logging
import tempfile
import datetime as dt

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategy_test_library import (
    download_csv,
    load_csv,
    walk_forward,
    compute_metrics,
    analyze_trades,
    compute_buy_and_hold,
    prepare_regime_inputs,
    run_regime_decomposition,
    print_backtest_report,
)
from daily_output import build_daily_outputs
from global_equity_library import build_mmf_extended
from price_series_builder import build_and_upload


# ============================================================
# ██████████████████████████████████████████████████████████
#  ASSET CONFIG BLOCK — edit this for each deployed asset
# ██████████████████████████████████████████████████████████
# ============================================================

ASSET_NAME    = "MSCI_World"
# Must match a key in ASSET_REGISTRY below.

OUTPUT_PREFIX = "msci_world"
# Unique lowercase identifier for output file naming.
# Determines log file name, output filenames, and Drive filenames.
# Examples: "sp500", "mwig40tr", "nikkei225", "stoxx600", "msci_world"

TRAIN_YEARS   = 8
TEST_YEARS    = 1
# Set from sweep candidate results.

FORCE_FILTER_MODE = ["ma", "mom"]
# Override to ["ma"] if sweep showed ma-only was consistently dominant
# for this asset across window configs.

# ============================================================
# ASSET REGISTRY
# ============================================================
# Identical to asset_config_sweep.py — keep in sync when adding assets.

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
        "ticker": "^spx",
    },
    "NASDAQ100": {
        "source": "stooq",
        "ticker": "^ndq",
    },
    "Nikkei225": {
        "source": "stooq",
        "ticker": "^nkx",
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

# ============================================================
# STRATEGY CONFIGURATION
# ============================================================
# Identical defaults to strat_test_runfile_with_MC.py.
# Override per-asset only if the sweep warrants it.

POSITION_MODE  = "full"
VOL_WINDOW     = 20
OBJECTIVE      = "calmar"

X_GRID         = [0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID         = [0.02, 0.03, 0.05, 0.07, 0.10]
FAST_GRID      = [50, 75, 100]
SLOW_GRID      = [150, 200, 250]
TV_GRID        = [0.08, 0.10, 0.12, 0.15, 0.20]
SL_GRID        = [0.05, 0.08, 0.10, 0.15]
MOM_LB_GRID    = [252]

MMF_FLOOR      = "1994-10-03"
DATA_START     = "1990-01-01"
BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}

OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = f"{OUTPUT_PREFIX}_DailyRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_fh  = logging.FileHandler(LOG_FILE, mode="w")
_fh.setFormatter(_fmt)
_ch  = logging.StreamHandler()
_ch.setFormatter(_fmt)
root_logger.addHandler(_fh)
root_logger.addHandler(_ch)

logging.info("=" * 80)
logging.info("%s DAILY RUNFILE START: %s", ASSET_NAME.upper(), dt.datetime.now())
logging.info("  OUTPUT_PREFIX : %s", OUTPUT_PREFIX)
logging.info("  TRAIN_YEARS   : %d", TRAIN_YEARS)
logging.info("  TEST_YEARS    : %d", TEST_YEARS)
logging.info("  FILTER_MODE   : %s", FORCE_FILTER_MODE)
logging.info("=" * 80)


# ============================================================
# PARALLELISM
# ============================================================

_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count
logging.info(
    "Parallel grid search: %d logical cores detected, using %d jobs.",
    _cpu_count, N_JOBS,
)


# ============================================================
# PHASE 1 — DATA DOWNLOAD
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOAD DATA")
logging.info("=" * 80)

tmp_dir = tempfile.gettempdir()


def _stooq(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame | None:
    url  = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
    path = os.path.join(tmp_dir, f"{OUTPUT_PREFIX}_{label}.csv")
    ok   = download_csv(url, path)
    if not ok:
        if mandatory:
            logging.error("FAIL: %s (%s) — exiting.", label, ticker)
            sys.exit(1)
        logging.warning("WARN: %s — not available, skipping.", label)
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


# --- Load target index ---
cfg = ASSET_REGISTRY.get(ASSET_NAME)
if cfg is None:
    logging.error("ASSET_NAME '%s' not found in ASSET_REGISTRY.", ASSET_NAME)
    sys.exit(1)

if cfg["source"] == "stooq":
    INDEX_DF = _stooq(cfg["ticker"], ASSET_NAME, mandatory=True)

elif cfg["source"] == "drive":
    folder_id = (
        os.environ.get("GDRIVE_FOLDER_ID", "").strip()
        or GDRIVE_FOLDER_ID_DEFAULT.strip()
    )
    if not folder_id:
        logging.error(
            "GDRIVE_FOLDER_ID not set — cannot load Drive-backed asset '%s'.",
            ASSET_NAME,
        )
        sys.exit(1)
    kw = dict(cfg["drive_kwargs"])
    kw["folder_id"] = folder_id
    INDEX_DF = build_and_upload(**kw)
    if INDEX_DF is None:
        logging.error("Failed to build/load '%s' from Drive.", ASSET_NAME)
        sys.exit(1)
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        ASSET_NAME, len(INDEX_DF),
        INDEX_DF.index.min().date(), INDEX_DF.index.max().date(),
    )
else:
    logging.error("Unknown source '%s' for asset '%s'.", cfg["source"], ASSET_NAME)
    sys.exit(1)

df = INDEX_DF

# --- Load cash (MMF + WIBOR extension) ---
MMF = _stooq("2720.n", "MMF", mandatory=True)

WIBOR1M = _stooq("plopln1m", "WIBOR1M", mandatory=False)
if WIBOR1M is not None:
    CASH = build_mmf_extended(MMF, WIBOR1M, floor_date=MMF_FLOOR)
    logging.info(
        "MMF extended: %s to %s (%d rows total; original MMF from %s)",
        CASH.index.min().date(), CASH.index.max().date(),
        len(CASH), MMF.index.min().date(),
    )
else:
    logging.warning(
        "WIBOR1M unavailable — MMF not extended. "
        "IS windows before %s use ret_mmf=0.",
        MMF.index.min().date(),
    )
    CASH = MMF


# ============================================================
# PHASE 2 — WALK-FORWARD OPTIMISATION
# ============================================================

logging.info("=" * 80)
logging.info(
    "PHASE 2: WALK-FORWARD  (%s, train=%dy test=%dy)",
    ASSET_NAME, TRAIN_YEARS, TEST_YEARS,
)
logging.info("=" * 80)

wf_equity, wf_results, wf_trades = walk_forward(
    df                    = df,
    cash_df               = CASH,
    train_years           = TRAIN_YEARS,
    test_years            = TEST_YEARS,
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
    n_jobs                = N_JOBS,
    fast_mode             = True,
)

if wf_equity.empty or wf_results.empty:
    logging.error("Walk-forward produced no results — exiting.")
    sys.exit(1)

# ── OOS metrics ──────────────────────────────────────────────────────────
wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
logging.info("=" * 80)
logging.info("Stitched OOS Metrics:")
logging.info(
    "CAGR: %.2f%%  Vol: %.2f%%  Sharpe: %.2f  MaxDD: %.2f%%  CalMAR: %.2f",
    wf_metrics["CAGR"]   * 100,
    wf_metrics["Vol"]    * 100,
    wf_metrics["Sharpe"],
    wf_metrics["MaxDD"]  * 100,
    wf_metrics["CalMAR"],
)
logging.info("=" * 80)

# ── Trade statistics ─────────────────────────────────────────────────────
if not wf_trades.empty and "Exit Reason" in wf_trades.columns:
    wf_trades_closed = wf_trades[
        ~wf_trades["Exit Reason"].isin(BOUNDARY_EXITS)
    ].copy()
else:
    wf_trades_closed = pd.DataFrame()

trade_stats = (
    analyze_trades(wf_trades_closed, boundary_exits=BOUNDARY_EXITS)
    if not wf_trades_closed.empty else None
)

print_backtest_report(
    metrics              = wf_metrics,
    trades               = wf_trades,
    trade_stats          = trade_stats,
    wf_results           = wf_results,
    position_mode        = POSITION_MODE,
    filter_modes_override= FORCE_FILTER_MODE,
)

# ── Buy & Hold ────────────────────────────────────────────────────────────
oos_start = wf_results["TestStart"].min()
oos_end   = wf_results["TestEnd"].max()

bh_equity, bh_metrics = compute_buy_and_hold(
    df, price_col="Zamkniecie", start=oos_start, end=oos_end
)
logging.info(
    "B&H (%s to %s):  CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%",
    oos_start.date(), oos_end.date(),
    bh_metrics["CAGR"] * 100, bh_metrics["Sharpe"], bh_metrics["MaxDD"] * 100,
)


# ============================================================
# PHASE 3 — REGIME DECOMPOSITION (informational)
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 3: REGIME DECOMPOSITION")
logging.info("=" * 80)

try:
    inputs = prepare_regime_inputs(df, wf_results, wf_equity, bh_equity)
    run_regime_decomposition(**{k: inputs[k] for k in [
        "close", "high", "low",
        "daily_returns_strat", "daily_returns_bh",
        "equity_strat", "equity_bh",
    ]})
except Exception as exc:
    logging.warning("Regime decomposition failed (non-fatal): %s", exc)


# ============================================================
# PHASE 4 — STATIC EQUITY CHART
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: EQUITY CHART")
logging.info("=" * 80)

if wf_equity is not None and not wf_equity.empty:
    wf_equity.index = pd.to_datetime(wf_equity.index)
    if not wf_trades.empty:
        wf_trades["EntryDate"] = pd.to_datetime(wf_trades["EntryDate"])
        wf_trades["ExitDate"]  = pd.to_datetime(wf_trades["ExitDate"])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(wf_equity.index, wf_equity.values,
            label="OOS Equity Curve", color="blue", linewidth=2)

    bh_al = bh_equity.loc[
        (bh_equity.index >= wf_equity.index.min()) &
        (bh_equity.index <= wf_equity.index.max())
    ]
    ax.plot(bh_al.index, bh_al.values,
            label=f"Buy & Hold {ASSET_NAME}", color="grey",
            linewidth=1.5, linestyle="--")

    for _, row in wf_results.iterrows():
        ax.axvspan(row["TestStart"], row["TestEnd"], color="grey", alpha=0.07)

    if not wf_trades.empty:
        for _, trade in wf_trades.iterrows():
            color = "green" if trade["Return"] > 0 else "red"
            ax.axvspan(trade["EntryDate"], trade["ExitDate"], color=color, alpha=0.15)
            trade_eq = wf_equity.loc[
                (wf_equity.index >= trade["EntryDate"]) &
                (wf_equity.index <= trade["ExitDate"])
            ]
            if trade_eq.empty:
                continue
            y_pos = (trade_eq.max() * 1.02 if trade["Return"] > 0
                     else trade_eq.min() * 0.98)
            ax.text(
                trade["ExitDate"], y_pos,
                f"{trade['Return']*100:.1f}%",
                color=color, fontsize=8, ha="left", va="bottom", rotation=45,
            )

    ax.text(
        0.01, 0.97,
        (
            f"Strategy:   CAGR {wf_metrics['CAGR']*100:.1f}% | "
            f"Vol {wf_metrics['Vol']*100:.1f}% | "
            f"MaxDD {wf_metrics['MaxDD']*100:.1f}%\n"
            f"Buy & Hold: CAGR {bh_metrics['CAGR']*100:.1f}% | "
            f"Vol {bh_metrics['Vol']*100:.1f}% | "
            f"MaxDD {bh_metrics['MaxDD']*100:.1f}%"
        ),
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title(
        f"{ASSET_NAME} Walk-Forward OOS Equity  "
        f"[train={TRAIN_YEARS}y  test={TEST_YEARS}y  filter={FORCE_FILTER_MODE}]",
        fontsize=13,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    static_chart_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_equity_chart_static.png")
    plt.savefig(static_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Static chart saved to %s", static_chart_path)


# ============================================================
# PHASE 5 — DAILY OUTPUT
# ============================================================
#
# daily_output.py writes to a fixed set of filenames:
#   signal_status.txt, signal_log.csv, equity_chart.png, signal_snapshot.json
#
# We isolate each asset's output to a subdirectory (outputs/{PREFIX}/)
# to avoid clobbering WIG20TR or other assets, then copy the four files
# to outputs/{PREFIX}_*.ext for Drive upload.
#
# The Drive log pre-fetch inside build_daily_outputs() looks for
# "signal_log.csv" in the Drive folder.  Since we upload the file as
# "{PREFIX}_signal_log.csv", we pass gdrive_log_filename_override so
# build_daily_outputs() fetches the right file.  If daily_output.py
# does not support that override, the workaround is to run each asset
# in its own Drive folder — documented below.
#
# IMPORTANT: set GDRIVE_FOLDER_ID env var before running so the Drive
# pre-fetch can locate the correct signal_log.csv for this asset.
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: DAILY OUTPUT")
logging.info("=" * 80)

PREFIX_OUTPUT_DIR = os.path.join(OUTPUT_DIR, OUTPUT_PREFIX)
os.makedirs(PREFIX_OUTPUT_DIR, exist_ok=True)

# daily_output._fetch_log_from_drive looks for "signal_log.csv" by default.
# The file on Drive is named "{PREFIX}_signal_log.csv".
# We temporarily monkey-patch the constant so the fetch targets the correct file.
# This is the minimal surgical change — no modifications to daily_output.py.
import daily_output as _do


# daily_output._fetch_log_from_drive looks for "signal_log.csv" by default.
# The file on Drive is named "{PREFIX}_signal_log.csv".



#_orig_log_file = _do.LOG_FILE  # save
#_do.LOG_FILE   = f"{OUTPUT_PREFIX}_signal_log.csv"  # patch

outputs = build_daily_outputs(
    wf_equity   = wf_equity,
    wf_trades   = wf_trades,
    wf_metrics  = wf_metrics,
    wf_results  = wf_results,
    bh_equity   = bh_equity,
    bh_metrics  = bh_metrics,
    df          = df,
    output_dir  = PREFIX_OUTPUT_DIR,
    price_col   = "Zamkniecie",
    logfile_name = f"{OUTPUT_PREFIX}_signal_log.csv",
    gdrive_folder_id   = os.getenv("GDRIVE_FOLDER_ID"),
    gdrive_credentials = "/tmp/credentials.json"
                         if os.path.exists("/tmp/credentials.json") else None,
)

#_do.LOG_FILE = _orig_log_file  # restore

logging.info(
    "Daily output complete — action=%s  signal=%s",
    outputs["action"], outputs["signal"],
)

# Copy outputs to prefixed names in outputs/ for Drive upload by GH Actions.
# daily_output.py writes fixed names inside PREFIX_OUTPUT_DIR;
# we rename to prefixed names in the parent outputs/ directory.
FILE_MAP = {
    "signal_status.txt":    f"{OUTPUT_PREFIX}_signal_status.txt",
    f"{OUTPUT_PREFIX}_signal_log.csv":       f"{OUTPUT_PREFIX}_signal_log.csv",
    "equity_chart.png":     f"{OUTPUT_PREFIX}_equity_chart.png",
    "signal_snapshot.json": f"{OUTPUT_PREFIX}_signal_snapshot.json",
}

for orig_name, prefixed_name in FILE_MAP.items():
    src  = os.path.join(PREFIX_OUTPUT_DIR, orig_name)
    dest = os.path.join(OUTPUT_DIR, prefixed_name)
    if os.path.exists(src):
        shutil.copy2(src, dest)
        logging.info("Copied %s -> %s", os.path.basename(src), prefixed_name)
    else:
        logging.warning("Expected output not found: %s", src)


# ============================================================
# DONE
# ============================================================

logging.info("=" * 80)
logging.info(
    "%s DAILY RUNFILE COMPLETE: %s",
    ASSET_NAME.upper(), dt.datetime.now(),
)
logging.info("  Action : %s", outputs["action"])
logging.info("  Signal : %s", outputs["signal"])
logging.info("  Outputs: %s/", OUTPUT_DIR)
logging.info("=" * 80)
