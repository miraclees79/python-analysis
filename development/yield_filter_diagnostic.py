"""
yield_filter_diagnostic.py
==========================
Standalone diagnostic for the yield momentum pre-filter calibration.

Computes the PL10Y 63-day yield change series and evaluates how the
filter would have behaved at different thresholds and persistence
settings across the full available history (1999-2026).

Outputs:
  - Threshold comparison table (raw filter, no persistence)
  - Persistence sensitivity table at focus threshold
  - Blocked episode table at focus threshold + focus persistence
  - 4-panel plot: yield, 63d change, raw vs smoothed filter, TBSP with shading
  - yield_filter_diagnostic.png

Run from /development folder.
"""

import os
import logging
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── settings ──────────────────────────────────────────────────────────────────
LOOKBACK_DAYS    = 63           # 3-month rolling window
THRESHOLDS_BP    = [25, 50, 75, 100, 150]
FOCUS_THRESHOLD  = 50           # highlighted in episode table and plot
PERSIST_DAYS     = [1, 5, 10, 15]   # persistence sensitivity comparison
FOCUS_PERSIST    = 10           # highlighted in episode table and plot


# ── helpers ───────────────────────────────────────────────────────────────────
def download_csv(url, path):
    import urllib.request
    urllib.request.urlretrieve(url, path)


def load_csv(path):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.strip() for c in df.columns]
        return df.sort_index()
    except Exception as e:
        logging.error("load_csv failed: %s", e)
        return None


def cagr(rets):
    if len(rets) == 0:
        return np.nan
    return (1 + rets).prod() ** (252 / len(rets)) - 1


def maxdd(rets):
    if len(rets) == 0:
        return np.nan
    eq = (1 + rets).cumprod()
    return float((eq / eq.cummax() - 1).min())


def apply_persistence(raw_filter: pd.Series, min_days: int) -> pd.Series:
    """
    Filter only turns OFF (0) when threshold has been breached for
    min_days consecutive days. Eliminates short-duration chatter.
    min_days=1 is equivalent to the raw filter (no smoothing).
    """
    if min_days <= 1:
        return raw_filter.copy()
    return raw_filter.rolling(min_days).min().fillna(1).astype(int)


def extract_blocks(filt: pd.Series):
    """
    Return list of (start, end) tuples for contiguous OFF (0) periods.
    """
    blocks = []
    in_block = False
    block_start = None
    for date, val in filt.items():
        if val == 0 and not in_block:
            block_start = date
            in_block = True
        elif val == 1 and in_block:
            blocks.append((block_start, date))
            in_block = False
    if in_block:
        blocks.append((block_start, filt.index[-1]))
    return blocks


# ── download ──────────────────────────────────────────────────────────────────
logging.info("Downloading PL10Y and TBSP...")
tmp_dir = tempfile.mkdtemp()

csv_pl10y = os.path.join(tmp_dir, "pl10y.csv")
csv_tbsp  = os.path.join(tmp_dir, "tbsp.csv")
download_csv("https://stooq.pl/q/d/l/?s=10yply.b&i=d", csv_pl10y)
download_csv("https://stooq.pl/q/d/l/?s=^tbsp&i=d",     csv_tbsp)

pl10y = load_csv(csv_pl10y)
tbsp  = load_csv(csv_tbsp)

# ── build series ──────────────────────────────────────────────────────────────
# Yield change on full PL10Y history (back to 1999)
ytm         = pl10y.iloc[:, 1].copy().ffill()
ytm_bp      = ytm * 100
ytm_chg     = ytm_bp - ytm_bp.shift(LOOKBACK_DAYS)

# TBSP daily returns
tbsp_price  = tbsp.iloc[:, 1].copy()
tbsp_ret    = tbsp_price.pct_change()

# Two aligned datasets:
#   yield_only — full PL10Y history for filter statistics and episode tables
#   with_tbsp  — intersected with TBSP for conditional return statistics
yield_only = pd.DataFrame({
    "ytm":     ytm,
    "ytm_chg": ytm_chg,
}).dropna()

with_tbsp = pd.DataFrame({
    "ytm":      ytm,
    "ytm_chg":  ytm_chg,
    "tbsp_ret": tbsp_ret,
}).dropna()

logging.info(
    "PL10Y yield series : %d rows  (%s to %s)",
    len(yield_only),
    yield_only.index.min().date(),
    yield_only.index.max().date()
)
logging.info(
    "TBSP overlap       : %d rows  (%s to %s)",
    len(with_tbsp),
    with_tbsp.index.min().date(),
    with_tbsp.index.max().date()
)

# ── Table 1: threshold comparison (raw filter, no persistence) ────────────────
logging.info("=" * 72)
logging.info("TABLE 1 — THRESHOLD COMPARISON  (lookback=%dd, no persistence)", LOOKBACK_DAYS)
logging.info("=" * 72)
logging.info(
    "%-8s  %-7s  %-10s  %-10s  %-8s  %-10s  %-10s  %-8s",
    "Thresh", "ON%", "ON_CAGR%", "OFF_CAGR%", "Diff_pp",
    "ON_MaxDD%", "OFF_MaxDD%", "N_blocks"
)
logging.info("-" * 72)

thresh_results = {}
for thresh in THRESHOLDS_BP:
    raw_filt = (yield_only["ytm_chg"] < thresh).astype(int)
    blocks   = extract_blocks(raw_filt)

    # TBSP conditional returns over intersected period
    filt_tbsp = raw_filt.reindex(with_tbsp.index).fillna(1).astype(int)
    on_ret    = with_tbsp["tbsp_ret"][filt_tbsp == 1]
    off_ret   = with_tbsp["tbsp_ret"][filt_tbsp == 0]

    on_cagr_  = cagr(on_ret)  * 100
    off_cagr_ = cagr(off_ret) * 100
    on_mdd_   = maxdd(on_ret) * 100
    off_mdd_  = maxdd(off_ret)* 100
    pct_on_   = raw_filt.mean() * 100

    thresh_results[thresh] = dict(
        pct_on=pct_on_, on_cagr=on_cagr_, off_cagr=off_cagr_,
        diff=on_cagr_ - off_cagr_, on_mdd=on_mdd_, off_mdd=off_mdd_,
        n_blocks=len(blocks), raw_filt=raw_filt
    )

    logging.info(
        "%-8s  %-7.1f  %-10.2f  %-10.2f  %-8.2f  %-10.2f  %-10.2f  %-8d",
        f"{thresh}bp", pct_on_, on_cagr_, off_cagr_,
        on_cagr_ - off_cagr_, on_mdd_, off_mdd_, len(blocks)
    )

# ── Table 2: persistence sensitivity at focus threshold ───────────────────────
logging.info("=" * 72)
logging.info(
    "TABLE 2 — PERSISTENCE SENSITIVITY  (threshold=%dbp, lookback=%dd)",
    FOCUS_THRESHOLD, LOOKBACK_DAYS
)
logging.info("=" * 72)
logging.info(
    "%-12s  %-7s  %-10s  %-10s  %-8s  %-10s  %-8s",
    "Persist", "ON%", "ON_CAGR%", "OFF_CAGR%", "Diff_pp",
    "OFF_MaxDD%", "N_blocks"
)
logging.info("-" * 72)

raw_focus = (yield_only["ytm_chg"] < FOCUS_THRESHOLD).astype(int)
persist_results = {}

for min_days in PERSIST_DAYS:
    smooth_filt  = apply_persistence(raw_focus, min_days)
    blocks       = extract_blocks(smooth_filt)

    filt_tbsp    = smooth_filt.reindex(with_tbsp.index).fillna(1).astype(int)
    on_ret       = with_tbsp["tbsp_ret"][filt_tbsp == 1]
    off_ret      = with_tbsp["tbsp_ret"][filt_tbsp == 0]

    on_cagr_     = cagr(on_ret)  * 100
    off_cagr_    = cagr(off_ret) * 100
    off_mdd_     = maxdd(off_ret) * 100
    pct_on_      = smooth_filt.mean() * 100

    persist_results[min_days] = dict(
        pct_on=pct_on_, on_cagr=on_cagr_, off_cagr=off_cagr_,
        diff=on_cagr_ - off_cagr_, off_mdd=off_mdd_,
        n_blocks=len(blocks), smooth_filt=smooth_filt
    )

    marker = " <- focus" if min_days == FOCUS_PERSIST else ""
    logging.info(
        "%-12s  %-7.1f  %-10.2f  %-10.2f  %-8.2f  %-10.2f  %-8d%s",
        f"{min_days}d", pct_on_, on_cagr_, off_cagr_,
        on_cagr_ - off_cagr_, off_mdd_, len(blocks), marker
    )

# ── Table 3: episode table at focus threshold + focus persistence ─────────────
logging.info("=" * 72)
logging.info(
    "TABLE 3 — BLOCKED EPISODES  (threshold=%dbp, persistence=%dd)",
    FOCUS_THRESHOLD, FOCUS_PERSIST
)
logging.info("=" * 72)

focus_filt  = persist_results[FOCUS_PERSIST]["smooth_filt"]
focus_blocks = extract_blocks(focus_filt)

logging.info("Total blocked periods: %d", len(focus_blocks))
logging.info(
    "%-12s  %-12s  %-8s  %-12s  %-30s",
    "Start", "End", "Days", "TBSP_ret%", "Note"
)
logging.info("-" * 72)

for start, end in focus_blocks:
    mask       = (with_tbsp.index >= start) & (with_tbsp.index < end)
    period_ret = (1 + with_tbsp["tbsp_ret"][mask]).prod() - 1 if mask.sum() > 0 else np.nan
    days       = (end - start).days

    # Auto-label significant episodes
    note = ""
    if period_ret < -0.05:
        note = "*** Major stress"
    elif period_ret < -0.01:
        note = "* Moderate stress"
    elif period_ret > 0.01:
        note = "(TBSP positive — false block)"

    logging.info(
        "%-12s  %-12s  %-8d  %-12.2f  %-30s",
        str(start.date()), str(end.date()), days,
        period_ret * 100 if not np.isnan(period_ret) else np.nan,
        note
    )

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle(
    f"PL10Y Yield Momentum Pre-filter Diagnostic\n"
    f"Threshold={FOCUS_THRESHOLD}bp  |  Lookback={LOOKBACK_DAYS}d  "
    f"|  Persistence={FOCUS_PERSIST}d",
    fontsize=12
)

# Panel 1 — PL10Y yield level
ax1 = axes[0]
ax1.plot(yield_only.index, yield_only["ytm"], color="steelblue", linewidth=0.8)
ax1.set_ylabel("PL 10Y Yield (%)")
ax1.set_title("PL 10Y Yield Level")
ax1.grid(True, alpha=0.3)

# Panel 2 — 63-day yield change with threshold line
ax2 = axes[1]
ax2.plot(yield_only.index, yield_only["ytm_chg"], color="darkorange",
         linewidth=0.8, label=f"{LOOKBACK_DAYS}d yield change (bp)")
ax2.axhline(FOCUS_THRESHOLD, color="red", linewidth=1.0,
            linestyle="--", label=f"Threshold {FOCUS_THRESHOLD}bp")
ax2.axhline(0, color="grey", linewidth=0.5, linestyle=":")
ax2.set_ylabel(f"{LOOKBACK_DAYS}d Change (bp)")
ax2.set_title(f"{LOOKBACK_DAYS}-Day Yield Change vs Threshold")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3 — raw vs persistence-smoothed filter state
ax3 = axes[2]
raw_filt_focus = thresh_results[FOCUS_THRESHOLD]["raw_filt"]
smooth_filt_focus = persist_results[FOCUS_PERSIST]["smooth_filt"]

ax3.fill_between(yield_only.index,
                 raw_filt_focus.reindex(yield_only.index).fillna(1),
                 alpha=0.3, color="red", label=f"Raw filter OFF (>{FOCUS_THRESHOLD}bp)")
ax3.fill_between(yield_only.index,
                 smooth_filt_focus.reindex(yield_only.index).fillna(1),
                 alpha=0.5, color="navy", label=f"Smoothed filter OFF (persist={FOCUS_PERSIST}d)")
ax3.set_ylabel("Filter state (1=ON)")
ax3.set_title(f"Raw vs Persistence-Smoothed Filter  ({FOCUS_THRESHOLD}bp threshold)")
ax3.set_ylim(-0.1, 1.3)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4 — TBSP with smoothed blocked periods shaded
ax4 = axes[3]
tbsp_eq = (1 + with_tbsp["tbsp_ret"]).cumprod()
ax4.plot(with_tbsp.index, tbsp_eq, color="steelblue", linewidth=0.8, label="TBSP")

for start, end in focus_blocks:
    ax4.axvspan(start, end, alpha=0.25, color="red")

ax4.legend(
    handles=[
        plt.Line2D([0], [0], color="steelblue", linewidth=1, label="TBSP cumulative return"),
        Patch(facecolor="red", alpha=0.25,
              label=f"Filter OFF (>{FOCUS_THRESHOLD}bp, persist={FOCUS_PERSIST}d)")
    ],
    fontsize=8
)
ax4.set_ylabel("Cumulative Return")
ax4.set_title("TBSP Performance with Smoothed Blocked Periods")
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.xaxis.set_major_locator(mdates.YearLocator(2))

plt.tight_layout()
plt.savefig("yield_filter_diagnostic.png", dpi=150, bbox_inches="tight")
logging.info("Plot saved to yield_filter_diagnostic.png")
plt.show()