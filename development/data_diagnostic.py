"""
data_diagnostic.py
==================
Step 1 data validation for the global equity framework.

Downloads all required price and FX series for both portfolio modes,
checks date coverage, data continuity, and plots PLN-adjusted cumulative
returns since 1990 for all assets.

Run this script standalone before assembling the full runfile:
    python data_diagnostic.py

Outputs
-------
  global_equity_data_diagnostic.log   — detailed download and validation log
  global_equity_diagnostic.png        — multi-panel cumulative return chart

What this script validates
--------------------------
  1. All stooq tickers download successfully and have data back to >= 1990
     (or the earliest available date).
  2. STOXX 600 (yfinance) — actual available history start date.
  3. URTH / MSCI World (yfinance) — available history start date.
     Reports whether a longer-history proxy is available on stooq.
  4. All FX series (USDPLN, EURPLN, JPYPLN) download from stooq.
  5. The MWIG40TR + SWIG80TR 50:50 blend constructs without gaps.
  6. PLN-adjusted returns (both hedged and unhedged) are computed for
     all foreign assets — the comparison between the two is plotted.
  7. Calendar alignment: all series are aligned to the WIG20TR calendar;
     gaps and forward-fill counts are reported.
  8. Data floor enforcement at 1990-01-01.

The script does NOT run walk_forward — it is purely a data layer check.
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
import matplotlib.dates as mdates

# ── path setup: assume script lives alongside strategy_test_library.py ────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_test_library import download_csv, load_csv
from global_equity_library import (
    download_yfinance,
    yfinance_to_stooq,
    build_return_series,
    build_blend,
    build_price_df_from_returns,
    CLOSE_COL,
    DATA_START,
    FX_TICKERS,
    YFINANCE_TICKERS,
)

# ============================================================
# LOGGING
# ============================================================

LOG_FILE = "global_equity_data_diagnostic.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(ch)

logging.info("=" * 80)
logging.info("GLOBAL EQUITY DATA DIAGNOSTIC  START: %s", dt.datetime.now())
logging.info("=" * 80)

# ============================================================
# USER SETTINGS
# ============================================================

# Diagnostic-only FX switch: plot both hedged and unhedged for comparison
SHOW_BOTH_FX = True

# Output chart path
CHART_PATH = "global_equity_diagnostic.png"

# ============================================================
# PHASE 1 — DOWNLOAD ALL SERIES
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 1: DOWNLOADING ALL SERIES")
logging.info("=" * 80)

tmp_dir = tempfile.gettempdir()

# ── Helper: stooq download + load ────────────────────────────────────────────
def _stooq(ticker_code: str, label: str) -> pd.DataFrame | None:
    url  = f"https://stooq.pl/q/d/l/?s={ticker_code}&i=d"
    path = os.path.join(tmp_dir, f"diag_{label}.csv")
    ok   = download_csv(url, path)
    if not ok:
        logging.error("FAIL: %s (%s) — stooq download failed.", label, ticker_code)
        return None
    df = load_csv(path)
    if df is None:
        logging.error("FAIL: %s — load_csv returned None.", label)
        return None
    # Apply data floor
    df = df.loc[df.index >= pd.Timestamp(DATA_START)]
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        label, len(df), df.index.min().date(), df.index.max().date(),
    )
    return df

# ── Stooq: PLN-native equity indices ─────────────────────────────────────────
WIG20TR  = _stooq("wig20tr",  "WIG20TR")
MWIG40TR = _stooq("mwig40tr", "MWIG40TR")
SWIG80TR = _stooq("swig80tr", "SWIG80TR")

# ── Stooq: Foreign equity indices ────────────────────────────────────────────
SPX     = _stooq("^spx",  "SP500")
NKX     = _stooq("^nkx",   "Nikkei225")

# ── Stooq: TBSP + MMF ────────────────────────────────────────────────────────
TBSP    = _stooq("^tbsp",  "TBSP")
MMF     = _stooq("2720.n", "MMF")

# ── Stooq: FX rates ──────────────────────────────────────────────────────────
USDPLN  = _stooq("usdpln", "USDPLN")
EURPLN  = _stooq("eurpln", "EURPLN")
JPYPLN  = _stooq("jpypln", "JPYPLN")

# ── yfinance: STOXX 600 ───────────────────────────────────────────────────────
STOXX600 = download_yfinance("^STOXX", start=DATA_START)
if STOXX600 is not None:
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "STOXX600(yf)", len(STOXX600),
        STOXX600.index.min().date(), STOXX600.index.max().date(),
    )
else:
    logging.error("FAIL: STOXX600 — yfinance download returned None.")

# ── yfinance: MSCI World proxy (URTH ETF) ────────────────────────────────────
URTH = download_yfinance("URTH", start=DATA_START)
if URTH is not None:
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        "URTH(yf)", len(URTH),
        URTH.index.min().date(), URTH.index.max().date(),
    )
else:
    logging.warning("WARN: URTH not available. Trying iShares IWDA.L as fallback.")
    URTH = download_yfinance("IWDA.L", start=DATA_START)
    if URTH is not None:
        logging.info(
            "OK  : %-14s  %5d rows  %s to %s",
            "IWDA.L(yf)", len(URTH),
            URTH.index.min().date(), URTH.index.max().date(),
        )
    else:
        logging.error("FAIL: Both URTH and IWDA.L unavailable.")

# ── stooq: MSCI World check (stooq may carry ^MSCIW or similar) ───────────────
MSCIW_STOOQ = _stooq("^msciw", "MSCIW_stooq")
if MSCIW_STOOQ is not None:
    logging.info("NOTE: MSCI World index found on stooq — longer history available.")
else:
    logging.info(
        "NOTE: MSCI World not found on stooq (^msciw). "
        "URTH/IWDA.L will be used for Mode B (msci_world)."
    )

# ============================================================
# PHASE 2 — SERIES SUMMARY
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 2: SERIES COVERAGE SUMMARY")
logging.info("=" * 80)

series_map = {
    "WIG20TR":   WIG20TR,
    "MWIG40TR":  MWIG40TR,
    "SWIG80TR":  SWIG80TR,
    "SP500":     SPX,
    "Nikkei225": NKX,
    "STOXX600":  STOXX600,
    "MSCI_World":URTH,
    "TBSP":      TBSP,
    "MMF":       MMF,
    "USDPLN":    USDPLN,
    "EURPLN":    EURPLN,
    "JPYPLN":    JPYPLN,
}

logging.info("%-14s  %-6s  %-12s  %-12s  %-8s", "Series", "Rows", "Start", "End", "NaN_close")
logging.info("-" * 70)

for label, df in series_map.items():
    if df is None:
        logging.info("%-14s  MISSING", label)
        continue
    n_rows   = len(df)
    start    = df.index.min().date()
    end      = df.index.max().date()
    n_nan    = df[CLOSE_COL].isna().sum() if CLOSE_COL in df.columns else "N/A"
    logging.info("%-14s  %-6d  %-12s  %-12s  %-8s", label, n_rows, start, end, n_nan)

# ============================================================
# PHASE 3 — BUILD BLEND AND PLN RETURN SERIES
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 3: BLEND + RETURN SERIES CONSTRUCTION")
logging.info("=" * 80)

# ── PL mid/small blend ───────────────────────────────────────────────────────
PL_MID = None
if MWIG40TR is not None and SWIG80TR is not None:
    PL_MID = build_blend(MWIG40TR, SWIG80TR, w_a=0.5, w_b=0.5, label="PL_MID")
else:
    logging.error("Cannot build PL_MID blend — one or both input series missing.")

# ── Return series: hedged (default) ──────────────────────────────────────────
# PLN-native: no FX adjustment regardless of switch
def _ret_pln(df, label, fx_series=None, hedged=True):
    if df is None:
        logging.warning("Return series '%s': source DataFrame is None — skipping.", label)
        return None
    r = build_return_series(df, fx_series=fx_series, hedged=hedged)
    logging.info(
        "Return series '%-16s': %5d rows  %s to %s  "
        "mean_daily=%.4f%%  ann_vol=%.2f%%",
        label, len(r.dropna()),
        r.dropna().index.min().date(), r.dropna().index.max().date(),
        r.mean() * 100, r.std() * np.sqrt(252) * 100,
    )
    return r

ret_WIG20TR  = _ret_pln(WIG20TR,  "WIG20TR")
ret_PL_MID   = _ret_pln(PL_MID,   "PL_MID(blend)")
ret_TBSP     = _ret_pln(TBSP,     "TBSP")
ret_MMF      = _ret_pln(MMF,      "MMF")

# Foreign assets — hedged (local return only)
ret_STOXX_H  = _ret_pln(STOXX600, "STOXX600_hedged",  fx_series=None, hedged=True)
ret_SPX_H    = _ret_pln(SPX,      "SP500_hedged",     fx_series=None, hedged=True)
ret_NKX_H    = _ret_pln(NKX,      "Nikkei_hedged",    fx_series=None, hedged=True)
ret_WORLD_H  = _ret_pln(URTH,     "MSCI_World_hedged",fx_series=None, hedged=True)

# Foreign assets — unhedged (FX-adjusted)
fx_usd = USDPLN[CLOSE_COL].rename("FX_USD") if USDPLN is not None else None
fx_eur = EURPLN[CLOSE_COL].rename("FX_EUR") if EURPLN is not None else None
fx_jpy = JPYPLN[CLOSE_COL].rename("FX_JPY") if JPYPLN is not None else None

ret_STOXX_U  = _ret_pln(STOXX600, "STOXX600_unhedged", fx_series=fx_eur, hedged=False)
ret_SPX_U    = _ret_pln(SPX,      "SP500_unhedged",    fx_series=fx_usd, hedged=False)
ret_NKX_U    = _ret_pln(NKX,      "Nikkei_unhedged",   fx_series=fx_jpy, hedged=False)
ret_WORLD_U  = _ret_pln(URTH,     "MSCI_World_unhedged",fx_series=fx_usd, hedged=False)

# ============================================================
# PHASE 4 — CALENDAR ALIGNMENT CHECK
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 4: CALENDAR ALIGNMENT")
logging.info("=" * 80)

# Reference: WIG20TR trading calendar (Polish market)
if WIG20TR is not None:
    ref_idx = WIG20TR.index
    logging.info("Reference calendar: WIG20TR  (%d trading days)", len(ref_idx))

    check_map = {
        "MWIG40TR":   MWIG40TR,
        "SWIG80TR":   SWIG80TR,
        "SP500":      SPX,
        "Nikkei225":  NKX,
        "STOXX600":   STOXX600,
        "MSCI_World": URTH,
        "TBSP":       TBSP,
        "MMF":        MMF,
    }

    for label, df in check_map.items():
        if df is None:
            continue
        # Days in WIG20TR calendar but not in this series (will be forward-filled)
        overlap = ref_idx.intersection(df.index)
        gap_days = len(ref_idx) - len(overlap)
        coverage_start = max(ref_idx.min(), df.index.min())
        coverage_end   = min(ref_idx.max(), df.index.max())
        logging.info(
            "  %-14s  overlap=%d  ffill_days=%d  common_range=%s to %s",
            label, len(overlap), gap_days,
            coverage_start.date(), coverage_end.date(),
        )
else:
    logging.warning("WIG20TR missing — calendar alignment check skipped.")

# ============================================================
# PHASE 5 — CUMULATIVE RETURN CHARTS
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 5: GENERATING DIAGNOSTIC CHARTS")
logging.info("=" * 80)

def _cum(ret: pd.Series | None, start: str = "1990-01-01") -> pd.Series | None:
    """Cumulative return index starting at 100, trimmed to start date."""
    if ret is None:
        return None
    r = ret.dropna()
    r = r.loc[r.index >= pd.Timestamp(start)]
    if r.empty:
        return None
    return 100.0 * (1 + r).cumprod()

fig, axes = plt.subplots(3, 1, figsize=(14, 18))
fig.suptitle(
    "Global Equity Framework — Data Diagnostic\n"
    f"Generated: {dt.date.today()}",
    fontsize=13, y=0.98,
)

# ── Panel 1: PLN-native equity indices ───────────────────────────────────────
ax1 = axes[0]
ax1.set_title("PLN-native equity indices (hedged, cumulative return base=100)")
ax1.set_yscale("log")

for label, ret in [
    ("WIG20TR",     ret_WIG20TR),
    ("PL_MID 50:50",ret_PL_MID),
    ("TBSP",        ret_TBSP),
    ("MMF",         ret_MMF),
]:
    c = _cum(ret)
    if c is not None:
        ax1.plot(c.index, c.values, label=label, linewidth=1.2)

ax1.legend(fontsize=9)
ax1.set_ylabel("Cumulative return (log)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.grid(True, alpha=0.3)

# ── Panel 2: Foreign equity — hedged vs unhedged ─────────────────────────────
ax2 = axes[1]
ax2.set_title("Foreign equity indices — hedged (solid) vs unhedged/FX-adjusted (dashed)")
ax2.set_yscale("log")

styles = [("STOXX600", ret_STOXX_H, ret_STOXX_U, "C0"),
          ("S&P500",   ret_SPX_H,   ret_SPX_U,   "C1"),
          ("Nikkei225",ret_NKX_H,   ret_NKX_U,   "C2"),
          ("MSCI_World",ret_WORLD_H, ret_WORLD_U, "C3")]

for label, r_h, r_u, color in styles:
    c_h = _cum(r_h)
    c_u = _cum(r_u)
    if c_h is not None:
        ax2.plot(c_h.index, c_h.values, color=color, linewidth=1.2,
                 linestyle="-", label=f"{label} hedged")
    if c_u is not None and SHOW_BOTH_FX:
        ax2.plot(c_u.index, c_u.values, color=color, linewidth=0.9,
                 linestyle="--", label=f"{label} unhedged")

ax2.legend(fontsize=8, ncol=2)
ax2.set_ylabel("Cumulative return (log)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.grid(True, alpha=0.3)

# ── Panel 3: FX rates (PLN per unit of foreign currency) ─────────────────────
ax3 = axes[2]
ax3.set_title("FX rates — PLN per unit of foreign currency (stooq)")

for label, df, color in [
    ("USD/PLN", USDPLN, "C1"),
    ("EUR/PLN", EURPLN, "C2"),
    ("JPY/PLN (×100)", JPYPLN, "C3"),
]:
    if df is None:
        continue
    series = df[CLOSE_COL].dropna()
    series = series.loc[series.index >= pd.Timestamp(DATA_START)]
    if label.startswith("JPY"):
        series = series * 100   # scale JPY for readability on same axis
    ax3.plot(series.index, series.values, label=label, color=color, linewidth=1.0)

ax3.legend(fontsize=9)
ax3.set_ylabel("PLN per foreign unit")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel("Year")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(CHART_PATH, dpi=120, bbox_inches="tight")
plt.close()
logging.info("Chart saved: %s", CHART_PATH)

# ============================================================
# PHASE 6 — DATA SUFFICIENCY SUMMARY
# ============================================================

logging.info("=" * 80)
logging.info("PHASE 6: DATA SUFFICIENCY FOR WALK-FORWARD")
logging.info("=" * 80)

TRAIN_YEARS = 9
TEST_YEARS  = 1
WINDOW      = TRAIN_YEARS + TEST_YEARS

logging.info(
    "Walk-forward window: %d year train + %d year test = %d year total",
    TRAIN_YEARS, TEST_YEARS, WINDOW,
)
logging.info(
    "With %d-year windows, OOS first year = data_start + %d years",
    WINDOW, TRAIN_YEARS,
)

check_returns = {
    "WIG20TR":   ret_WIG20TR,
    "PL_MID":    ret_PL_MID,
    "SP500":     ret_SPX_H,
    "STOXX600":  ret_STOXX_H,
    "Nikkei225": ret_NKX_H,
    "MSCI_World":ret_WORLD_H,
    "TBSP":      ret_TBSP,
    "MMF":       ret_MMF,
}

logging.info(
    "%-14s  %-12s  %-12s  %-8s  %-16s",
    "Asset", "Start", "End", "Years", "OOS_first_year",
)
logging.info("-" * 70)

for label, ret in check_returns.items():
    if ret is None:
        logging.info("%-14s  MISSING", label)
        continue
    r = ret.dropna()
    if r.empty:
        logging.info("%-14s  EMPTY", label)
        continue
    start = r.index.min()
    end   = r.index.max()
    years = (end - start).days / 365.25
    oos_first = start + pd.DateOffset(years=TRAIN_YEARS)
    logging.info(
        "%-14s  %-12s  %-12s  %-8.1f  %-16s",
        label, start.date(), end.date(), years, oos_first.date(),
    )

# Common OOS start = max of all individual OOS firsts
oos_firsts = []
for label, ret in check_returns.items():
    if ret is None:
        continue
    r = ret.dropna()
    if r.empty:
        continue
    oos_firsts.append(r.index.min() + pd.DateOffset(years=TRAIN_YEARS))

if oos_firsts:
    common_oos_start = max(oos_firsts)
    common_oos_end   = min(ret.dropna().index.max()
                           for ret in check_returns.values() if ret is not None)
    oos_years = (common_oos_end - common_oos_start).days / 365.25
    logging.info("-" * 70)
    logging.info("COMMON OOS START  : %s", common_oos_start.date())
    logging.info("COMMON OOS END    : %s", common_oos_end.date())
    logging.info("COMMON OOS LENGTH : %.1f years", oos_years)

logging.info("=" * 80)
logging.info("GLOBAL EQUITY DATA DIAGNOSTIC  COMPLETE: %s", dt.datetime.now())
logging.info("Log file: %s", os.path.abspath(LOG_FILE))
logging.info("Chart:    %s", os.path.abspath(CHART_PATH))
logging.info("=" * 80)
