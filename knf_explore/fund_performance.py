# fund_performance.py
#
# Computes risk-adjusted performance measures for matched KNF-stooq funds,
# ranked within each KNF category.
#
# Input sources (from Drive, in priority order):
#   1. knf_stooq_confirmed.csv   — manually reviewed match list (authoritative)
#      Columns: subfundId, stooq_id, knf_name, stooq_name
#      (produced by manual review of knf_stooq_review.csv + knf_stooq_matches.csv)
#   2. knf_stooq_matches.csv     — raw matcher output (fallback until review complete)
#      Only tier == "exact" rows used from this file to keep quality high.
#
# For each fund the script fetches daily NAV from stooq's CSV endpoint:
#   https://stooq.pl/q/d/l/?s={stooq_id}.n&i=d
#
# Performance measures computed over 1Y / 3Y / 5Y windows and full history:
#   ann_return      — CAGR
#   ann_vol         — annualised daily return std dev
#   sharpe          — ann_return / ann_vol  (no risk-free rate subtracted;
#                     add WIBOR later when integrating benchmark data)
#   max_drawdown    — peak-to-trough decline
#   calmar          — ann_return / abs(max_drawdown)
#   data_years      — years of history available for that window
#
# Outputs (to Drive):
#   fund_performance_YYYYMMDD.csv   — full table, all funds, all windows
#   fund_ranking_YYYYMMDD.csv       — within-category rank on Sharpe (1Y primary)
#
# Required env vars:
#   GOOGLE_CREDENTIALS   — service account JSON
#   GDRIVE_FOLDER_ID     — Drive folder containing input and output files
#
# Optional env vars:
#   STOOQ_DELAY_MIN      — min seconds between stooq requests (default 0.5)
#   STOOQ_DELAY_MAX      — max seconds between stooq requests (default 1.5)
#   MIN_HISTORY_DAYS     — minimum days of NAV required to include a fund (default 252)
#
# Dependencies:
#   pip install pandas numpy requests google-auth google-api-python-client

import os
import io
import time
import random
import logging
import datetime
import requests
import numpy as np
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.oauth2 import service_account

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GDRIVE_FOLDER    = os.getenv("GDRIVE_FOLDER_ID", "")
CREDS_PATH       = "/tmp/credentials.json"
OUTPUT_DIR       = "/tmp/fund_perf"

STOOQ_NAV_URL    = "https://stooq.pl/q/d/l/?s={stooq_id}.n&i=d"
STOOQ_DELAY_MIN  = float(os.getenv("STOOQ_DELAY_MIN", "0.5"))
STOOQ_DELAY_MAX  = float(os.getenv("STOOQ_DELAY_MAX", "1.5"))
MIN_HISTORY_DAYS = int(os.getenv("MIN_HISTORY_DAYS", "252"))

CONFIRMED_FILE   = "knf_stooq_confirmed.csv"
MATCHES_FILE     = "knf_stooq_matches.csv"

WINDOWS = {
    "1y":   252,
    "3y":   756,
    "5y":  1260,
    "all":  None,   # None = use full available history
}

TRADING_DAYS_PER_YEAR = 252

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

def get_drive_service():
    creds_json = os.getenv("GOOGLE_CREDENTIALS", "")
    if creds_json:
        with open(CREDS_PATH, "w") as f:
            f.write(creds_json)
    creds = service_account.Credentials.from_service_account_file(
        CREDS_PATH,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def download_csv_from_drive(service, folder_id: str, filename: str) -> pd.DataFrame | None:
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id,name)").execute()
    items = res.get("files", [])
    if not items:
        log.warning("File not found on Drive: %s", filename)
        return None
    file_id = items[0]["id"]
    buf = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    try:
        return pd.read_csv(buf, sep=";", encoding="utf-8")
    except Exception:
        buf.seek(0)
        return pd.read_csv(buf, encoding="utf-8")


def upload_csv_to_drive(service, folder_id: str, local_path: str):
    filename = os.path.basename(local_path)
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=query, fields="files(id)").execute()
    items = res.get("files", [])
    with open(local_path, "rb") as f:
        buf = io.BytesIO(f.read())
    media = MediaIoBaseUpload(buf, mimetype="text/csv", resumable=False)
    if items:
        service.files().update(fileId=items[0]["id"], media_body=media).execute()
    else:
        meta = {"name": filename, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()
    log.info("Uploaded %s to Drive", filename)


# ---------------------------------------------------------------------------
# Load match list
# ---------------------------------------------------------------------------

def load_match_list(service) -> pd.DataFrame:
    """
    Tries knf_stooq_confirmed.csv first (manually reviewed, authoritative).
    Falls back to knf_stooq_matches.csv filtered to exact/partial matches only.

    Returns DataFrame with columns:
        subfundId, stooq_id, knf_name, stooq_name, tfi_name, category,
        classification, riskLevel, managementAndOtherFeesRate, maxEntryFeeRate,
        hasBenchmark, benchmark
    """
    confirmed = download_csv_from_drive(service, GDRIVE_FOLDER, CONFIRMED_FILE)
    if confirmed is not None and not confirmed.empty:
        log.info("Using confirmed match list: %d funds", len(confirmed))
        # Confirmed file must have at least: subfundId, stooq_id, knf_name, stooq_name
        # Category metadata joined in from matches file if missing
        missing_meta = [c for c in ("category", "classification", "tfi_name")
                        if c not in confirmed.columns]
        if missing_meta:
            log.info("  Confirmed file missing %s — joining from matches file", missing_meta)
            matches = download_csv_from_drive(service, GDRIVE_FOLDER, MATCHES_FILE)
            if matches is not None:
                meta_cols = ["subfundId"] + missing_meta
                available = [c for c in meta_cols if c in matches.columns]
                confirmed = confirmed.merge(
                    matches[available].drop_duplicates("subfundId"),
                    on="subfundId", how="left",
                )
        return confirmed

    # Fallback: raw matches, restrict to high-confidence tiers
    log.info(
        "%s not found — falling back to %s (exact/partial tiers only)",
        CONFIRMED_FILE, MATCHES_FILE,
    )
    matches = download_csv_from_drive(service, GDRIVE_FOLDER, MATCHES_FILE)
    if matches is None or matches.empty:
        raise RuntimeError(
            f"Neither {CONFIRMED_FILE} nor {MATCHES_FILE} found on Drive."
        )

    high_confidence = matches[
        matches["match_tier"].isin(["exact", "partial"])
    ].copy()

    n_all  = len(matches)
    n_kept = len(high_confidence)
    n_dropped = n_all - n_kept
    log.info(
        "  Raw matches: %d total, keeping %d exact/partial, "
        "dropping %d fuzzy/historical/none",
        n_all, n_kept, n_dropped,
    )

    # Rename to canonical column names expected downstream
    rename = {
        "name":       "knf_name",
        "stooq_title": "stooq_name",
    }
    high_confidence = high_confidence.rename(
        columns={k: v for k, v in rename.items() if k in high_confidence.columns}
    )
    return high_confidence


# ---------------------------------------------------------------------------
# Stooq NAV fetch
# ---------------------------------------------------------------------------

def fetch_nav(stooq_id) -> pd.Series | None:
    """
    Downloads daily NAV series for a stooq fund ID.
    Returns a pd.Series indexed by date (datetime), values = Close price.
    Returns None on failure (404, empty, parse error).
    """
    url = STOOQ_NAV_URL.format(stooq_id=int(stooq_id))
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 404 or not resp.content.strip():
            log.warning("  stooq %s: empty/404", stooq_id)
            return None
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            log.warning("  stooq %s: no Close column or empty CSV", stooq_id)
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        series = df["Close"].dropna()
        if len(series) < MIN_HISTORY_DAYS:
            log.warning(
                "  stooq %s: only %d days of history (min %d) — skipping",
                stooq_id, len(series), MIN_HISTORY_DAYS,
            )
            return None
        return series
    except Exception as exc:
        log.warning("  stooq %s: fetch error %s", stooq_id, exc)
        return None


# ---------------------------------------------------------------------------
# Performance calculations
# ---------------------------------------------------------------------------

def ann_return(returns: pd.Series) -> float:
    """CAGR from a series of daily log returns."""
    n = len(returns)
    if n < 2:
        return float("nan")
    total = returns.sum()   # sum of log returns = log of total return
    years = n / TRADING_DAYS_PER_YEAR
    return float(np.exp(total / years) - 1)


def ann_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.25 = -25%)."""
    if len(prices) < 2:
        return float("nan")
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())


def compute_metrics(prices: pd.Series, window_days: int | None) -> dict:
    """Compute all metrics for a given price series and window."""
    if window_days is not None and len(prices) >= window_days:
        prices = prices.iloc[-window_days:]
    elif window_days is not None:
        # Not enough history for this window
        return {
            "ann_return":    float("nan"),
            "ann_vol":       float("nan"),
            "sharpe":        float("nan"),
            "max_drawdown":  float("nan"),
            "calmar":        float("nan"),
            "data_years":    round(len(prices) / TRADING_DAYS_PER_YEAR, 1),
            "n_days":        len(prices),
            "insufficient":  True,
        }

    log_ret = np.log(prices / prices.shift(1)).dropna()
    ar = ann_return(log_ret)
    av = ann_volatility(log_ret)
    md = max_drawdown(prices)
    sharpe = ar / av if av and not np.isnan(av) and av > 0 else float("nan")
    calmar = ar / abs(md) if md and not np.isnan(md) and md < 0 else float("nan")

    return {
        "ann_return":   round(ar, 4),
        "ann_vol":      round(av, 4),
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(md, 4),
        "calmar":       round(calmar, 4),
        "data_years":   round(len(prices) / TRADING_DAYS_PER_YEAR, 1),
        "n_days":       len(prices),
        "insufficient": False,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_performance_table(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each matched fund: fetch NAV, compute metrics across all windows.
    Returns a flat DataFrame with one row per (fund, window).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    n_total = len(match_df)

    for i, fund in match_df.iterrows():
        stooq_id = fund.get("stooq_id")
        knf_name = fund.get("knf_name", fund.get("name", ""))
        category = fund.get("category", "")
        tfi      = fund.get("tfi_name", "")

        log.info(
            "[%d/%d]  %-12s  %s",
            i + 1, n_total, str(stooq_id), str(knf_name)[:60],
        )

        prices = fetch_nav(stooq_id)
        if prices is None:
            # Record as unfetched so it's visible in output
            for window_name in WINDOWS:
                rows.append({
                    "subfundId":              fund.get("subfundId"),
                    "stooq_id":               stooq_id,
                    "knf_name":               knf_name,
                    "tfi_name":               tfi,
                    "category":               category,
                    "classification":         fund.get("classification", ""),
                    "riskLevel":              fund.get("riskLevel"),
                    "managementAndOtherFeesRate": fund.get("managementAndOtherFeesRate"),
                    "maxEntryFeeRate":         fund.get("maxEntryFeeRate"),
                    "hasBenchmark":            fund.get("hasBenchmark"),
                    "benchmark":               fund.get("benchmark", ""),
                    "window":                  window_name,
                    "ann_return":              float("nan"),
                    "ann_vol":                 float("nan"),
                    "sharpe":                  float("nan"),
                    "max_drawdown":            float("nan"),
                    "calmar":                  float("nan"),
                    "data_years":              float("nan"),
                    "n_days":                  0,
                    "fetch_ok":                False,
                })
            time.sleep(random.uniform(STOOQ_DELAY_MIN, STOOQ_DELAY_MAX))
            continue

        log.info("  %d days of NAV (%.1f years)", len(prices),
                 len(prices) / TRADING_DAYS_PER_YEAR)

        for window_name, window_days in WINDOWS.items():
            metrics = compute_metrics(prices, window_days)
            rows.append({
                "subfundId":              fund.get("subfundId"),
                "stooq_id":               stooq_id,
                "knf_name":               knf_name,
                "tfi_name":               tfi,
                "category":               category,
                "classification":         fund.get("classification", ""),
                "riskLevel":              fund.get("riskLevel"),
                "managementAndOtherFeesRate": fund.get("managementAndOtherFeesRate"),
                "maxEntryFeeRate":         fund.get("maxEntryFeeRate"),
                "hasBenchmark":            fund.get("hasBenchmark"),
                "benchmark":               fund.get("benchmark", ""),
                "window":                  window_name,
                **metrics,
                "fetch_ok":                True,
            })

        time.sleep(random.uniform(STOOQ_DELAY_MIN, STOOQ_DELAY_MAX))

    return pd.DataFrame(rows)


def build_ranking(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (category, window), rank funds by Sharpe descending.
    Returns a wide-format table: one row per fund, columns for each window's
    Sharpe, return, drawdown and rank.
    """
    ranking_rows = []

    for (category, window), grp in perf_df.groupby(["category", "window"]):
        valid = grp[grp["fetch_ok"] & ~grp["sharpe"].isna()].copy()
        if valid.empty:
            continue
        valid = valid.sort_values("sharpe", ascending=False)
        valid["rank_sharpe"]  = range(1, len(valid) + 1)
        valid["rank_calmar"]  = valid["calmar"].rank(ascending=False, method="min").astype("Int64")
        valid["rank_return"]  = valid["ann_return"].rank(ascending=False, method="min").astype("Int64")
        ranking_rows.append(valid)

    if not ranking_rows:
        return pd.DataFrame()

    return pd.concat(ranking_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y%m%d")

    log.info("Fund performance comparison — start  (%s)", today)
    log.info("GDRIVE_FOLDER    : %s", GDRIVE_FOLDER)
    log.info("MIN_HISTORY_DAYS : %d", MIN_HISTORY_DAYS)
    log.info("Windows          : %s", list(WINDOWS.keys()))

    service  = get_drive_service()
    match_df = load_match_list(service)
    log.info("Match list loaded: %d funds across %d categories",
             len(match_df),
             match_df["category"].nunique() if "category" in match_df.columns else 0)

    perf_df = build_performance_table(match_df)
    log.info("Performance table built: %d rows (%d funds × %d windows)",
             len(perf_df), len(match_df), len(WINDOWS))

    # Summary by category/window
    log.info("\n--- Sharpe summary (1Y, top 3 per category) ---")
    rank_df = build_ranking(perf_df)
    for cat, grp in rank_df[rank_df["window"] == "1y"].groupby("category"):
        top = grp.nsmallest(3, "rank_sharpe")[
            ["rank_sharpe", "knf_name", "sharpe", "ann_return", "max_drawdown"]
        ]
        log.info("  %s", cat)
        for _, r in top.iterrows():
            log.info(
                "    #%d  sharpe=%5.2f  ret=%+.1f%%  dd=%.1f%%  %s",
                r["rank_sharpe"],
                r["sharpe"],
                r["ann_return"] * 100,
                r["max_drawdown"] * 100,
                str(r["knf_name"])[:55],
            )

    def save(df, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s  (%d rows)", filename, len(df))
        upload_csv_to_drive(service, GDRIVE_FOLDER, path)

    save(perf_df,  f"fund_performance_{today}.csv")
    save(rank_df,  f"fund_ranking_{today}.csv")

    log.info("Fund performance comparison — complete")
