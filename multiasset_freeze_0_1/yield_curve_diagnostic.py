"""
yield_curve_diagnostic.py
=========================
Diagnostic script to explore Polish yield curve signals as potential
pre-filters for the TBSP bond signal.

Downloads:
  - PL 10Y government bond yield  (10YPLY.B)
  - DE 10Y government bond yield  (10YDEY.B)
  - WIBOR 1Y                      (PLOPLN1Y)
  - TBSP index                    (^TBSP)

Computes:
  - PL yield curve slope: PL10Y - WIBOR1Y
  - PL-DE spread: PL10Y - DE10Y
  - 3-month rolling change in PL-DE spread

Produces a multi-panel plot covering the full available history with:
  Panel 1: TBSP price index (log scale)
  Panel 2: PL yield curve slope (PL10Y - WIBOR1Y), zero line marked
  Panel 3: PL-DE spread level and 3M rolling change
  Panel 4: Pre-filter state (AND of slope>0 and spread_change<threshold)
           overlaid with TBSP returns coloured by sign

Key episodes annotated: 2008, 2015 (rate cuts), 2022 (hiking cycle),
2023 (disinflation rally).

Output: yield_curve_diagnostic.png
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
import matplotlib.patches as mpatches

# ── reuse existing download/load infrastructure ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "development"))
from strategy_test_library import download_csv, load_csv

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("yield_curve_diagnostic.log", mode="w"),
    ]
)

# ── settings ──────────────────────────────────────────────────────────────────
# Spread change threshold: if 3M change in PL-DE spread exceeds this (bp),
# stress filter triggers
SPREAD_CHANGE_THRESHOLD_BP = 75.0   # basis points

# Slope smoothing window (trading days) to reduce noise on slope signal
SLOPE_SMOOTH_DAYS = 20

# Rolling window for spread change (approx 3 months of trading days)
SPREAD_CHANGE_WINDOW = 63

# ── download ──────────────────────────────────────────────────────────────────
TICKERS = {
    "PL10Y":  "10YPLY.B",
    "DE10Y":  "10YDEY.B",
    "WIBOR1Y": "PLOPLN1Y",
    "TBSP":   "^TBSP",
}

BASE_URL = "https://stooq.pl/q/d/l/?s={ticker}&i=d"

tmp_dir = tempfile.gettempdir()
raw = {}

logging.info("=" * 70)
logging.info("DOWNLOADING YIELD AND PRICE DATA")
logging.info("=" * 70)

for name, ticker in TICKERS.items():
    url  = BASE_URL.format(ticker=ticker.lower())
    path = os.path.join(tmp_dir, f"{name}.csv")
    ok   = download_csv(url, path)
    if not ok:
        logging.error("Failed to download %s (%s) — aborting.", name, ticker)
        sys.exit(1)
    df = load_csv(path)
    if df is None:
        logging.error("Failed to load %s — aborting.", name)
        sys.exit(1)
    # Extract close price column (second column after date index)
    close_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    raw[name] = df[close_col].rename(name)
    logging.info("%s loaded: %d rows  (%s to %s)",
                 name, len(raw[name]),
                 raw[name].index.min().date(),
                 raw[name].index.max().date())

# ── align to common index (forward-fill yields for non-trading days) ──────────
logging.info("=" * 70)
logging.info("COMPUTING SIGNALS")
logging.info("=" * 70)

# Combine on union of all dates, forward-fill yield series (yields don't
# have daily fixings on all calendar days)
combined = pd.DataFrame(raw)
combined = combined.sort_index()
combined = combined.ffill()   # forward-fill gaps in yield series
combined = combined.dropna()  # drop any leading NaNs

logging.info("Common aligned series: %d rows  (%s to %s)",
             len(combined),
             combined.index.min().date(),
             combined.index.max().date())

# ── derived series ────────────────────────────────────────────────────────────

# Yield curve slope (in percentage points, same units as raw yields)
combined["slope_raw"]    = combined["PL10Y"] - combined["WIBOR1Y"]
combined["slope_smooth"] = combined["slope_raw"].rolling(SLOPE_SMOOTH_DAYS).mean()

# PL-DE spread (in percentage points → convert to basis points for display)
combined["spread_pp"]    = combined["PL10Y"] - combined["DE10Y"]
combined["spread_bp"]    = combined["spread_pp"] * 100

# 3-month rolling change in spread (bp)
combined["spread_3m_chg_bp"] = (
    combined["spread_bp"] - combined["spread_bp"].shift(SPREAD_CHANGE_WINDOW)
)

# ── pre-filter signals ────────────────────────────────────────────────────────

# Slope filter: smoothed slope > 0 (curve not inverted)
combined["filter_slope"] = (combined["slope_smooth"] > 0).astype(int)

# Spread stress filter: 3M spread change < threshold
combined["filter_spread"] = (
    combined["spread_3m_chg_bp"] < SPREAD_CHANGE_THRESHOLD_BP
).astype(int)

# Combined AND pre-filter
combined["filter_combined"] = (
    (combined["filter_slope"] == 1) & (combined["filter_spread"] == 1)
).astype(int)

# TBSP daily returns
combined["tbsp_ret"] = combined["TBSP"].pct_change()

# ── summary statistics ────────────────────────────────────────────────────────
logging.info("-" * 70)
logging.info("PRE-FILTER SUMMARY")
logging.info("-" * 70)

n_total = len(combined.dropna())

for col, label in [
    ("filter_slope",    "Slope filter ON (curve not inverted, smoothed)"),
    ("filter_spread",   f"Spread filter ON (3M change < {SPREAD_CHANGE_THRESHOLD_BP}bp)"),
    ("filter_combined", "Combined AND filter ON"),
]:
    s = combined[col].dropna()
    pct = s.mean() * 100
    logging.info("  %-55s : %.1f%% of days", label, pct)

logging.info("-" * 70)

# TBSP return stats conditional on filter state
tbsp_valid = combined[["filter_combined", "tbsp_ret"]].dropna()
in_filter  = tbsp_valid[tbsp_valid["filter_combined"] == 1]["tbsp_ret"]
out_filter = tbsp_valid[tbsp_valid["filter_combined"] == 0]["tbsp_ret"]

def annualised_return(daily_rets):
    return ((1 + daily_rets).prod() ** (252 / len(daily_rets)) - 1) * 100

logging.info("TBSP RETURNS CONDITIONAL ON COMBINED PRE-FILTER:")
logging.info("  Filter ON  (%d days): Ann. return = %.2f%%  |  Daily mean = %.4f%%  |  Daily std = %.4f%%",
             len(in_filter),
             annualised_return(in_filter),
             in_filter.mean() * 100,
             in_filter.std() * 100)
logging.info("  Filter OFF (%d days): Ann. return = %.2f%%  |  Daily mean = %.4f%%  |  Daily std = %.4f%%",
             len(out_filter),
             annualised_return(out_filter),
             out_filter.mean() * 100,
             out_filter.std() * 100)

# Also show by individual filter
for col, label in [("filter_slope", "Slope filter"), ("filter_spread", "Spread filter")]:
    s = combined[[col, "tbsp_ret"]].dropna()
    on  = s[s[col] == 1]["tbsp_ret"]
    off = s[s[col] == 0]["tbsp_ret"]
    logging.info("%s ON  (%d days): Ann. return = %.2f%%", label, len(on),  annualised_return(on))
    logging.info("%s OFF (%d days): Ann. return = %.2f%%", label, len(off), annualised_return(off))

logging.info("-" * 70)

# Key episodes: what did the filters do?
EPISODES = {
    "2008 crisis":          ("2007-06-01", "2009-06-01"),
    "2015 NBP rate cuts":   ("2014-06-01", "2016-06-01"),
    "2021-22 hiking cycle": ("2021-06-01", "2023-03-01"),
    "2023 disinflation":    ("2023-01-01", "2024-06-01"),
}

logging.info("PRE-FILTER STATE DURING KEY EPISODES:")
for ep_name, (start, end) in EPISODES.items():
    ep = combined.loc[start:end, ["filter_slope", "filter_spread",
                                   "filter_combined", "tbsp_ret"]].dropna()
    if len(ep) == 0:
        continue
    tbsp_ep = annualised_return(ep["tbsp_ret"])
    in_ep   = ep[ep["filter_combined"] == 1]["tbsp_ret"]
    out_ep  = ep[ep["filter_combined"] == 0]["tbsp_ret"]
    slope_on_pct  = ep["filter_slope"].mean() * 100
    spread_on_pct = ep["filter_spread"].mean() * 100
    combined_on_pct = ep["filter_combined"].mean() * 100
    logging.info(
        "  %s (%s to %s):",
        ep_name, start, end
    )
    logging.info(
        "    TBSP ann. return over episode: %.2f%%", tbsp_ep
    )
    logging.info(
        "    Slope filter ON: %.0f%% | Spread filter ON: %.0f%% | Combined ON: %.0f%%",
        slope_on_pct, spread_on_pct, combined_on_pct
    )
    if len(in_ep) > 5:
        logging.info(
            "    TBSP ann. return when combined ON:  %.2f%%  (%d days)",
            annualised_return(in_ep), len(in_ep)
        )
    if len(out_ep) > 5:
        logging.info(
            "    TBSP ann. return when combined OFF: %.2f%%  (%d days)",
            annualised_return(out_ep), len(out_ep)
        )

# ── helper: shade regions where condition is True ────────────────────────────
def _shade_regions(ax, index, condition, color, alpha, label):
    """Shade contiguous regions where boolean Series condition is True."""
    in_region = False
    start_idx = None
    labeled   = False
    for date, val in zip(index, condition):
        if val and not in_region:
            in_region = True
            start_idx = date
        elif not val and in_region:
            in_region = False
            kw = {"label": label} if not labeled else {}
            ax.axvspan(start_idx, date, color=color, alpha=alpha, **kw)
            labeled = True
    if in_region:
        kw = {"label": label} if not labeled else {}
        ax.axvspan(start_idx, index[-1], color=color, alpha=alpha, **kw)


# ── plot ──────────────────────────────────────────────────────────────────────
logging.info("=" * 70)
logging.info("GENERATING PLOT")
logging.info("=" * 70)

BLUE   = "#2E75B6"
RED    = "#C00000"
GREEN  = "#375623"
ORANGE = "#E36C09"
GREY   = "#666666"
LGREY  = "#EEEEEE"

plot_data = combined.dropna(subset=["slope_smooth", "spread_3m_chg_bp",
                                     "filter_combined", "TBSP"])

fig, axes = plt.subplots(5, 1, figsize=(16, 20),
                          gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5, 1]})
fig.suptitle("Polish Yield Curve — Pre-filter Diagnostic\n"
             f"Slope smooth={SLOPE_SMOOTH_DAYS}d | "
             f"Spread change threshold={SPREAD_CHANGE_THRESHOLD_BP}bp (3M)",
             fontsize=13, fontweight="bold", y=0.98)

# ── Panel 1: TBSP price ───────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(plot_data.index, plot_data["TBSP"], color=BLUE, linewidth=1.2)
# Shade periods when combined filter is OFF
filter_off = plot_data["filter_combined"] == 0
_shade_regions(ax1, plot_data.index, filter_off, color=RED, alpha=0.15,
               label="Filter OFF")
ax1.set_ylabel("TBSP Index", fontsize=10)
ax1.set_title("Panel 1: TBSP Price (shaded = combined pre-filter OFF)", fontsize=10)
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# ── Panel 2: Yield curve slope ────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(plot_data.index, plot_data["slope_raw"],
         color=GREY, linewidth=0.6, alpha=0.5, label="Slope raw")
ax2.plot(plot_data.index, plot_data["slope_smooth"],
         color=BLUE, linewidth=1.2, label=f"Slope {SLOPE_SMOOTH_DAYS}d smooth")
ax2.axhline(0, color=RED, linewidth=1.0, linestyle="--", label="Zero (inversion)")
ax2.fill_between(plot_data.index, plot_data["slope_smooth"], 0,
                 where=plot_data["slope_smooth"] < 0,
                 color=RED, alpha=0.2, label="Inverted")
ax2.fill_between(plot_data.index, plot_data["slope_smooth"], 0,
                 where=plot_data["slope_smooth"] >= 0,
                 color=GREEN, alpha=0.1, label="Normal")
ax2.set_ylabel("PL10Y - WIBOR1Y (%)", fontsize=10)
ax2.set_title("Panel 2: PL Yield Curve Slope (PL10Y - WIBOR1Y)", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, ncol=3)

# ── Panel 3: PL-DE spread ─────────────────────────────────────────────────────
ax3 = axes[2]
ax3_twin = ax3.twinx()
ax3.plot(plot_data.index, plot_data["spread_bp"],
         color=BLUE, linewidth=1.2, label="PL-DE spread (bp)")
ax3_twin.plot(plot_data.index, plot_data["spread_3m_chg_bp"],
              color=ORANGE, linewidth=1.0, alpha=0.8, label="3M change (bp)")
ax3_twin.axhline(SPREAD_CHANGE_THRESHOLD_BP, color=RED, linewidth=1.0,
                 linestyle="--", label=f"Threshold ({SPREAD_CHANGE_THRESHOLD_BP}bp)")
ax3_twin.fill_between(plot_data.index, plot_data["spread_3m_chg_bp"],
                      SPREAD_CHANGE_THRESHOLD_BP,
                      where=plot_data["spread_3m_chg_bp"] > SPREAD_CHANGE_THRESHOLD_BP,
                      color=RED, alpha=0.2)
ax3.set_ylabel("PL-DE Spread (bp)", fontsize=10)
ax3_twin.set_ylabel("3M Spread Change (bp)", fontsize=10)
ax3.set_title("Panel 3: PL-DE 10Y Spread", fontsize=10)
ax3.grid(True, alpha=0.3)
lines3a, labels3a = ax3.get_legend_handles_labels()
lines3b, labels3b = ax3_twin.get_legend_handles_labels()
ax3.legend(lines3a + lines3b, labels3a + labels3b, fontsize=9, ncol=3)

# ── Panel 4: Filter states ────────────────────────────────────────────────────
ax4 = axes[3]
ax4.step(plot_data.index, plot_data["filter_slope"],
         color=BLUE, linewidth=1.0, where="post", label="Slope filter")
ax4.step(plot_data.index, plot_data["filter_spread"] * 0.95,
         color=ORANGE, linewidth=1.0, where="post", label="Spread filter")
ax4.step(plot_data.index, plot_data["filter_combined"] * 0.9,
         color=GREEN, linewidth=1.5, where="post", label="Combined (AND)")
ax4.set_ylim(-0.1, 1.2)
ax4.set_yticks([0, 1])
ax4.set_yticklabels(["OFF", "ON"])
ax4.set_ylabel("Filter State", fontsize=10)
ax4.set_title("Panel 4: Pre-filter States", fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9, ncol=3)

# ── Panel 5: TBSP forward returns by filter state ────────────────────────────
ax5 = axes[4]
# 63-day (3M) forward TBSP return
fwd_ret_63 = combined["TBSP"].pct_change(63).shift(-63) * 100
plot_fwd = pd.DataFrame({
    "fwd_ret": fwd_ret_63,
    "filter":  combined["filter_combined"]
}).dropna()

# Rolling mean of forward return by filter state
roll_on  = plot_fwd[plot_fwd["filter"] == 1]["fwd_ret"].rolling(126).mean()
roll_off = plot_fwd[plot_fwd["filter"] == 0]["fwd_ret"].rolling(126).mean()
ax5.plot(roll_on.index,  roll_on,  color=GREEN, linewidth=1.2,
         label="Filter ON — 63d fwd return (126d rolling mean)")
ax5.plot(roll_off.index, roll_off, color=RED, linewidth=1.2,
         label="Filter OFF — 63d fwd return (126d rolling mean)")
ax5.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
ax5.set_ylabel("63d Fwd Return (%)", fontsize=10)
ax5.set_title("Panel 5: TBSP 63-Day Forward Return by Filter State (126d rolling mean)",
              fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# ── episode annotations on all panels ────────────────────────────────────────
EPISODE_COLORS = ["#8B4513", "#4B0082", "#8B0000", "#006400"]
for ax in axes:
    for (ep_name, (start, end)), col in zip(EPISODES.items(), EPISODE_COLORS):
        try:
            xs = pd.Timestamp(start)
            xe = pd.Timestamp(end)
            ax.axvspan(xs, xe, alpha=0.04, color=col)
        except Exception:
            pass

# Episode labels on top panel only
for (ep_name, (start, end)), col in zip(EPISODES.items(), EPISODE_COLORS):
    try:
        xmid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ypos = axes[0].get_ylim()[1] * 0.92
        axes[0].text(xmid, ypos, ep_name, fontsize=7, color=col,
                     ha="center", va="top", rotation=90, alpha=0.7)
    except Exception:
        pass

for ax in axes:
    ax.set_xlim(plot_data.index.min(), plot_data.index.max())

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "yield_curve_diagnostic.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
logging.info("Plot saved to %s", out_path)
logging.info("=" * 70)
logging.info("DONE")
logging.info("=" * 70)