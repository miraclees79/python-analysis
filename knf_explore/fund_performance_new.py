# fund_performance.py
#
# Computes risk-adjusted performance measures for matched KNF-stooq funds,
# ranked within each KNF category.
#
# All return metrics are NET-OF-ONGOING-FEES (stooq NAV already reflects daily
# fee deduction from assets) and GROSS-OF-ENTRY-FEES (maxEntryFeeRate is not
# reflected in NAV and is carried as metadata only).
#
# Input sources (from Drive, in priority order):
#   1. knf_stooq_confirmed.csv   — manually reviewed match list (authoritative)
#      Columns: subfundId, stooq_id, knf_name, stooq_name
#   2. knf_stooq_matches.csv     — raw matcher output (fallback; exact/partial only)
#
# NAV series fetched from stooq via download_csv(), mirroring Download-oceny_v3-FI.py:
#   CSV_BASE_URL env var, rotating User-Agent, retry with exponential backoff.
#
# ── Absolute metrics (per fund, per window) ──────────────────────────────────
#   ann_return      CAGR
#   ann_vol         annualised daily return std dev
#   sharpe          ann_return / ann_vol  (no risk-free rate; relative ranking only)
#   sortino         ann_return / annualised downside deviation (negative days only)
#   max_drawdown    peak-to-trough decline (negative)
#   calmar          ann_return / |max_drawdown|
#
# ── Category-relative metrics (OLS vs equal-weighted category benchmark) ─────
#   alpha           annualised Jensen alpha vs equal-weighted category peers
#   beta            OLS beta to category benchmark
#   tracking_error  annualised residual std dev
#   ir              information ratio = alpha / tracking_error
#   pct_ir_positive % of rolling 1Y windows where IR > 0  (consistency signal)
#   n_ir_windows    number of rolling windows used for pct_ir_positive
#   category_funds  number of funds in category with data for that window
#
# Category benchmark is the equal-weighted average of daily log returns across
# all funds in the same category that have data on each day. Funds with fewer
# than MIN_CATEGORY_PEERS peers are flagged (category_peers_ok = False) but
# still computed.
#
# Outputs (to Drive):
#   fund_performance_YYYYMMDD.csv   — full metrics table (all funds, all windows)
#   fund_ranking_YYYYMMDD.csv       — within-category ranking with all metrics
#
# Required env vars:
#   GOOGLE_CREDENTIALS   service account JSON
#   GDRIVE_FOLDER_ID     Drive folder ID (input and output)
#
# Optional env vars:
#   CSV_BASE_URL         stooq CSV template, {} = stooq ID
#                        (default: https://stooq.pl/q/d/l/?s={}.n&i=d)
#   STOOQ_DELAY_MIN      min seconds between requests (default 0.5)
#   STOOQ_DELAY_MAX      max seconds between requests (default 1.5)
#   STOOQ_MAX_RETRIES    retries on transient errors (default 3)
#   STOOQ_BACKOFF        initial backoff seconds, doubles per retry (default 2.0)
#   MIN_HISTORY_DAYS     minimum NAV days to include a fund (default 252)
#   MIN_CATEGORY_PEERS   minimum peers for reliable category benchmark (default 3)
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
from numpy.linalg import lstsq

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

GDRIVE_FOLDER      = os.getenv("GDRIVE_FOLDER_ID", "")
CREDS_PATH         = "/tmp/credentials.json"
OUTPUT_DIR         = "/tmp/fund_perf"

# Mirrors the CSV_BASE_URL pattern from Download-oceny_v3-FI.py.
# The placeholder {} is replaced with the stooq numeric ID.
CSV_BASE_URL       = os.getenv("CSV_BASE_URL", "https://stooq.pl/q/d/l/?s={}.n&i=d")

STOOQ_DELAY_MIN    = float(os.getenv("STOOQ_DELAY_MIN",   "0.5"))
STOOQ_DELAY_MAX    = float(os.getenv("STOOQ_DELAY_MAX",   "1.5"))
STOOQ_MAX_RETRIES  = int(os.getenv("STOOQ_MAX_RETRIES",   "3"))
STOOQ_BACKOFF      = float(os.getenv("STOOQ_BACKOFF",     "2.0"))  # doubled per retry
MIN_HISTORY_DAYS   = int(os.getenv("MIN_HISTORY_DAYS",    "252"))
MIN_CATEGORY_PEERS = int(os.getenv("MIN_CATEGORY_PEERS",  "3"))

CONFIRMED_FILE     = "knf_stooq_confirmed.csv"
MATCHES_FILE       = "knf_stooq_matches.csv"

# KNF API — mirroring knf_explore.py
KNF_API_BASE    = "https://wybieramfundusze-api.knf.gov.pl"
KNF_API_RETRIES = int(os.getenv("KNF_API_RETRIES", "3"))
KNF_API_BACKOFF = float(os.getenv("KNF_API_BACKOFF", "2.0"))

KNF_SESSION = requests.Session()
KNF_SESSION.headers.update({"Accept": "application/json"})

# Windows: name -> number of trading days (None = full history)
WINDOWS = {
    "1y":   252,
    "3y":   756,
    "5y":  1260,
    "all":  None,
}

# Rolling IR: 1Y window, step = 1Q
ROLLING_IR_WINDOW = 252
ROLLING_IR_STEP   = 63

TDAYS = 252  # trading days per year

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
    downloader = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf, sep=None, engine="python", encoding="utf-8")


def upload_csv_to_drive(service, folder_id: str, local_path: str,
                        max_retries: int = 3):
    """
    Upload a local CSV to Drive, retrying with a fresh service object on any
    connection error. The original service may have a stale SSL connection after
    long-running NAV downloads — rebuilding it on failure is the safest fix.
    """
    filename = os.path.basename(local_path)
    query    = f"name='{filename}' and '{folder_id}' in parents and trashed=false"

    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            # Rebuild service on every attempt after the first to avoid stale
            # SSL connections that form during long computation phases.
            svc = get_drive_service() if attempt > 1 else service

            res   = svc.files().list(q=query, fields="files(id)").execute()
            items = res.get("files", [])

            with open(local_path, "rb") as f:
                buf = io.BytesIO(f.read())
            media = MediaIoBaseUpload(buf, mimetype="text/csv", resumable=False)

            if items:
                svc.files().update(fileId=items[0]["id"], media_body=media).execute()
            else:
                meta = {"name": filename, "parents": [folder_id]}
                svc.files().create(body=meta, media_body=media, fields="id").execute()

            log.info("Uploaded %s to Drive", filename)
            return

        except Exception as exc:
            log.warning("Upload attempt %d/%d failed for %s: %s",
                        attempt, max_retries, filename, exc)
            if attempt < max_retries:
                log.info("  Retrying in %.0fs with a fresh Drive connection...", delay)
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ---------------------------------------------------------------------------
# Load match list
# ---------------------------------------------------------------------------

def load_match_list(service) -> pd.DataFrame:
    """
    Tries knf_stooq_confirmed.csv first (manually reviewed, authoritative).
    Falls back to knf_stooq_matches.csv filtered to exact/partial tiers only.

    Returns DataFrame with columns:
        subfundId, stooq_id, knf_name, stooq_name, tfi_name, category,
        classification, riskLevel, managementAndOtherFeesRate, maxEntryFeeRate,
        hasBenchmark, benchmark
    """
    confirmed = download_csv_from_drive(service, GDRIVE_FOLDER, CONFIRMED_FILE)
    if confirmed is not None and not confirmed.empty:
        log.info("Using confirmed match list: %d funds", len(confirmed))
        log.info("  Confirmed columns: %s", list(confirmed.columns))

        META_COLS = (
            "subfundId", "category", "classification", "tfi_name",
            "riskLevel", "managementAndOtherFeesRate", "maxEntryFeeRate",
            "hasBenchmark", "benchmark",
        )
        missing_meta = [c for c in META_COLS if c not in confirmed.columns]
        if missing_meta:
            log.info("  Joining missing columns %s from matches file", missing_meta)
            matches = download_csv_from_drive(service, GDRIVE_FOLDER, MATCHES_FILE)
            if matches is not None:
                log.info("  Matches columns: %s", list(matches.columns))
                join_cols = ["stooq_id"] + [c for c in missing_meta if c in matches.columns]
                confirmed = confirmed.merge(
                    matches[join_cols].drop_duplicates("stooq_id"),
                    on="stooq_id", how="left",
                )
                log.info("  After join: %d rows", len(confirmed))
        return confirmed

    # Fallback: raw matches, high-confidence tiers only
    log.info("%s not found — falling back to %s (exact/partial tiers only)",
             CONFIRMED_FILE, MATCHES_FILE)
    matches = download_csv_from_drive(service, GDRIVE_FOLDER, MATCHES_FILE)
    if matches is None or matches.empty:
        raise RuntimeError(
            f"Neither {CONFIRMED_FILE} nor {MATCHES_FILE} found on Drive."
        )
    high_confidence = matches[
        matches["match_tier"].isin(["exact", "partial"])
    ].copy()
    log.info("  Raw matches: %d total -> %d kept (exact/partial), %d dropped",
             len(matches), len(high_confidence), len(matches) - len(high_confidence))
    return high_confidence.rename(columns={
        "name":        "knf_name",
        "stooq_title": "stooq_name",
    })


# ---------------------------------------------------------------------------
# stooq NAV download  (mirrors Download-oceny_v3-FI.py download_csv pattern)
# ---------------------------------------------------------------------------

def download_csv(stooq_id) -> pd.Series | None:
    """
    Downloads daily NAV CSV for a stooq numeric fund ID.
    Returns a pd.Series indexed by date (datetime64, ascending), values = NAV.
    Returns None on any unrecoverable failure.
    """
    url   = CSV_BASE_URL.format(int(stooq_id))
    delay = STOOQ_BACKOFF

    for attempt in range(1, STOOQ_MAX_RETRIES + 1):
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            log.debug("  GET %s  (attempt %d)", url, attempt)
            resp = requests.get(url, headers=headers, timeout=15)

            if resp.status_code == 404:
                log.warning("  stooq %s: 404 — not found", stooq_id)
                return None

            resp.raise_for_status()

            if not resp.content.strip():
                log.warning("  stooq %s: empty response (attempt %d)", stooq_id, attempt)
                if attempt < STOOQ_MAX_RETRIES:
                    time.sleep(delay); delay *= 2; continue
                return None

            df = pd.read_csv(io.StringIO(resp.text))

            if df.empty or "Zamkniecie" not in df.columns:
                log.warning("  stooq %s: missing Zamkniecie column or empty CSV", stooq_id)
                return None

            df["Data"] = pd.to_datetime(df["Data"])
            series = (
                df.sort_values("Data")
                  .set_index("Data")["Zamkniecie"]
                  .dropna()
            )

            if len(series) < MIN_HISTORY_DAYS:
                log.warning("  stooq %s: only %d days (min %d) — skipping",
                            stooq_id, len(series), MIN_HISTORY_DAYS)
                return None

            return series

        except requests.HTTPError as exc:
            log.warning("  stooq %s: HTTP %s (attempt %d)", stooq_id, exc, attempt)
        except Exception as exc:
            log.warning("  stooq %s: %s (attempt %d)", stooq_id, exc, attempt)

        if attempt < STOOQ_MAX_RETRIES:
            time.sleep(delay); delay *= 2

    log.warning("  stooq %s: all %d attempts failed", stooq_id, STOOQ_MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Absolute performance metrics
# ---------------------------------------------------------------------------

def _ann_return(log_ret: pd.Series) -> float:
    n = len(log_ret)
    if n < 2:
        return float("nan")
    return float(np.exp(log_ret.sum() / (n / TDAYS)) - 1)


def _ann_vol(log_ret: pd.Series) -> float:
    return float(log_ret.std() * np.sqrt(TDAYS)) if len(log_ret) > 1 else float("nan")


def _max_drawdown(prices: pd.Series) -> float:
    if len(prices) < 2:
        return float("nan")
    dd = (prices - prices.cummax()) / prices.cummax()
    return float(dd.min())


def _sortino(log_ret: pd.Series) -> float:
    ar  = _ann_return(log_ret)
    neg = log_ret[log_ret < 0]
    if len(neg) < 2:
        return float("nan")
    dd = neg.std() * np.sqrt(TDAYS)
    return ar / dd if dd > 0 else float("nan")


def compute_absolute_metrics(prices: pd.Series, window_days: int | None) -> dict:
    """
    Slice prices to window, compute all absolute metrics.
    Returns dict including private keys '_log_ret' and '_prices' for
    downstream category-relative calculations (stripped before output).
    """
    if window_days is not None and len(prices) >= window_days:
        prices = prices.iloc[-window_days:]
    elif window_days is not None:
        # Insufficient history for this window — still pass log_ret for benchmark
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return {
            "ann_return":   float("nan"),
            "ann_vol":      float("nan"),
            "sharpe":       float("nan"),
            "sortino":      float("nan"),
            "max_drawdown": float("nan"),
            "calmar":       float("nan"),
            "data_years":   round(len(prices) / TDAYS, 1),
            "n_days":       len(prices),
            "insufficient": True,
            "_log_ret":     log_ret,
        }

    log_ret = np.log(prices / prices.shift(1)).dropna()
    ar  = _ann_return(log_ret)
    av  = _ann_vol(log_ret)
    md  = _max_drawdown(prices)
    sh  = ar / av  if (av  > 0 and not np.isnan(av))  else float("nan")
    so  = _sortino(log_ret)
    cal = ar / abs(md) if (md < 0 and not np.isnan(md)) else float("nan")

    return {
        "ann_return":   round(ar,  4),
        "ann_vol":      round(av,  4),
        "sharpe":       round(sh,  4),
        "sortino":      round(so,  4),
        "max_drawdown": round(md,  4),
        "calmar":       round(cal, 4),
        "data_years":   round(len(prices) / TDAYS, 1),
        "n_days":       len(prices),
        "insufficient": False,
        "_log_ret":     log_ret,
    }


# ---------------------------------------------------------------------------
# Category-relative metrics
# ---------------------------------------------------------------------------

def compute_relative_metrics(
    fund_log_ret: pd.Series,
    category_benchmark: pd.Series,
) -> dict:
    """
    OLS regression of fund daily log returns against equal-weighted category
    benchmark (itself the mean of daily log returns across all category peers).

    alpha  — annualised Jensen alpha (net-of-fees, since NAV is net-of-fees)
    beta   — sensitivity to category average
    tracking_error — annualised std dev of residuals
    ir     — information ratio = alpha / tracking_error
    """
    common = fund_log_ret.index.intersection(category_benchmark.index)
    if len(common) < 30:
        nan = float("nan")
        return dict(alpha=nan, beta=nan, tracking_error=nan, ir=nan)

    y = fund_log_ret.loc[common].values
    x = category_benchmark.loc[common].values
    X = np.column_stack([np.ones(len(x)), x])

    coeffs, _, _, _ = lstsq(X, y, rcond=None)
    alpha_daily, beta = coeffs

    residuals = y - (alpha_daily + beta * x)
    te_daily  = residuals.std()
    alpha_ann = float(np.exp(alpha_daily * TDAYS) - 1)
    te_ann    = float(te_daily * np.sqrt(TDAYS))
    ir        = alpha_ann / te_ann if te_ann > 0 else float("nan")

    return dict(
        alpha          = round(alpha_ann, 4),
        beta           = round(float(beta), 4),
        tracking_error = round(te_ann, 4),
        ir             = round(ir, 4),
    )


def compute_rolling_ir(
    fund_log_ret: pd.Series,
    category_benchmark: pd.Series,
) -> dict:
    """
    Rolls a 1Y (252-day) OLS window in steps of 1Q (63 days) over the full
    overlapping history. Reports the fraction of windows where IR > 0 as a
    consistency/persistence signal.
    """
    common = fund_log_ret.index.intersection(category_benchmark.index)
    if len(common) < ROLLING_IR_WINDOW:
        return dict(pct_ir_positive=float("nan"), n_ir_windows=0)

    y_all = fund_log_ret.loc[common].values
    x_all = category_benchmark.loc[common].values
    n     = len(common)

    ir_vals = []
    for start in range(0, n - ROLLING_IR_WINDOW + 1, ROLLING_IR_STEP):
        y = y_all[start : start + ROLLING_IR_WINDOW]
        x = x_all[start : start + ROLLING_IR_WINDOW]
        X = np.column_stack([np.ones(ROLLING_IR_WINDOW), x])
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        a_d   = coeffs[0]
        resid = y - (coeffs[0] + coeffs[1] * x)
        te    = resid.std() * np.sqrt(TDAYS)
        a_ann = float(np.exp(a_d * TDAYS) - 1)
        ir_vals.append(a_ann / te if te > 0 else float("nan"))

    valid   = [v for v in ir_vals if not np.isnan(v)]
    pct_pos = sum(1 for v in valid if v > 0) / len(valid) if valid else float("nan")

    return dict(
        pct_ir_positive = round(pct_pos, 3),
        n_ir_windows    = len(valid),
    )


# ---------------------------------------------------------------------------
# Phase 1a: fetch KNF subfund detail + KID metadata for all confirmed funds
# ---------------------------------------------------------------------------

def _knf_get(path: str, params: dict = None) -> dict | None:
    """Single KNF API GET with retry/backoff. Returns parsed JSON or None."""
    url   = KNF_API_BASE + path
    delay = KNF_API_BACKOFF
    for attempt in range(1, KNF_API_RETRIES + 1):
        try:
            r = KNF_SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            log.warning("KNF GET %s (attempt %d): %s", path, attempt, exc)
            if attempt < KNF_API_RETRIES:
                time.sleep(delay)
                delay *= 2
    return None


def fetch_knf_metadata(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every subfundId in match_df, independently fetches:
      - subfund detail: classification, category, tfi_name, fromDate
      - KID detail:     riskLevel, managementAndOtherFeesRate, maxEntryFeeRate,
                        hasBenchmark, benchmark

    This is independent of the matching pipeline so that funds added manually
    to knf_stooq_confirmed.csv (from the unmatched subsample) also receive
    full metadata — they were never in knf_stooq_matches.csv.

    Returns a DataFrame indexed by subfundId with all metadata columns.
    Missing columns are filled with None. Existing non-null values in
    match_df are preserved; only gaps are filled.
    """
    subfund_ids = match_df["subfundId"].dropna().astype(int).tolist()
    n = len(subfund_ids)
    log.info("Fetching KNF subfund detail for %d funds...", n)

    # ── Step A: subfund detail (classification, category, tfi via fund chain) ──
    # Build companyId → tfi_name lookup once
    company_raw = _knf_get("/v1/companies", {
        "includeInactive": "false", "sort": "name,asc", "size": 500, "page": 0,
    })
    company_by_id = {}
    if company_raw:
        for c in company_raw.get("content", []):
            if c.get("companyId") is not None:
                company_by_id[c["companyId"]] = c["name"]
    log.info("  Companies loaded: %d", len(company_by_id))

    # Build fundId → companyId lookup
    fund_raw = _knf_get("/v1/funds", {
        "includeInactive": "false", "includeLiquidating": "false",
        "sort": "name,asc", "size": 2000, "page": 0,
    })
    company_by_fund = {}
    if fund_raw:
        for f in fund_raw.get("content", []):
            fid = f.get("fundId")
            cid = f.get("companyId")
            if fid and cid:
                company_by_fund[fid] = cid
    log.info("  Funds loaded: %d", len(company_by_fund))

    detail_rows = {}
    for i, sfid in enumerate(subfund_ids):
        detail = _knf_get(f"/v1/subfunds/{sfid}")
        if detail:
            fund_id = detail.get("fundId")
            cid     = company_by_fund.get(fund_id)
            tfi     = company_by_id.get(cid, "") if cid else ""
            detail_rows[sfid] = {
                "subfundId":      sfid,
                "classification": detail.get("classification"),
                "category":       detail.get("category"),
                "fromDate":       detail.get("fromDate"),
                "tfi_name":       tfi,
            }
        if (i + 1) % 50 == 0:
            log.info("  subfund detail: %d / %d", i + 1, n)
        time.sleep(random.uniform(0.1, 0.2))

    log.info("Subfund detail fetched: %d / %d", len(detail_rows), n)

    # ── Step B: KID data (fees, risk level, benchmark) ──
    log.info("Fetching KID data for %d funds...", n)
    uuid_map = {}
    for i, sfid in enumerate(subfund_ids):
        body = _knf_get("/v1/key-information/units", {
            "subfundId":       sfid,
            "isPrimary":       "true",
            "includeInactive": "false",
            "sort":            "publicationDate,desc",
            "size":            1,
            "page":            0,
        })
        if body:
            content = body.get("content", [])
            if content:
                uuid_map[sfid] = content[0]["documentUuid"]
        if (i + 1) % 50 == 0:
            log.info("  KID UUID: %d / %d  (%d found)", i + 1, n, len(uuid_map))
        time.sleep(random.uniform(0.1, 0.2))

    log.info("KID UUIDs found: %d / %d", len(uuid_map), n)

    kid_rows = {}
    for i, (sfid, uuid) in enumerate(uuid_map.items()):
        kid = _knf_get(f"/v1/key-information/units/{uuid}")
        if kid:
            kid_rows[sfid] = {
                "subfundId":                  sfid,
                "riskLevel":                  kid.get("riskLevel"),
                "maxEntryFeeRate":            kid.get("maxEntryFeeRate"),
                "maxExitFeeRate":             kid.get("maxExitFeeRate"),
                "managementAndOtherFeesRate": kid.get("managementAndOtherFeesRate"),
                "hasBenchmark":               kid.get("hasBenchmark"),
                "benchmark":                  kid.get("benchmark"),
            }
        if (i + 1) % 50 == 0:
            log.info("  KID detail: %d / %d", i + 1, len(uuid_map))
        time.sleep(random.uniform(0.1, 0.2))

    log.info("KID detail fetched: %d / %d", len(kid_rows), n)

    # ── Merge into a single metadata DataFrame ──
    meta_df = pd.DataFrame(list(detail_rows.values())) if detail_rows else pd.DataFrame()
    kid_df  = pd.DataFrame(list(kid_rows.values()))    if kid_rows  else pd.DataFrame()

    if not meta_df.empty and not kid_df.empty:
        meta_df = meta_df.merge(kid_df, on="subfundId", how="left")
    elif not kid_df.empty:
        meta_df = kid_df

    if meta_df.empty:
        log.warning("fetch_knf_metadata returned no data — metadata gaps will remain.")
        return match_df

    # ── Fill gaps in match_df (preserve existing non-null values) ──
    META_COLS = [
        "classification", "category", "fromDate", "tfi_name",
        "riskLevel", "maxEntryFeeRate", "maxExitFeeRate",
        "managementAndOtherFeesRate", "hasBenchmark", "benchmark",
    ]
    meta_df["subfundId"] = meta_df["subfundId"].astype(str)
    match_df = match_df.copy()
    match_df["subfundId"] = match_df["subfundId"].astype(str)

    # Add missing columns to match_df first
    for col in META_COLS:
        if col not in match_df.columns:
            match_df[col] = None

    merged = match_df.merge(
        meta_df[["subfundId"] + [c for c in META_COLS if c in meta_df.columns]],
        on="subfundId", how="left", suffixes=("", "_knf"),
    )

    # For each metadata column: use _knf value only where original is null
    for col in META_COLS:
        knf_col = col + "_knf"
        if knf_col in merged.columns:
            merged[col] = merged[col].where(
                merged[col].notna() & (merged[col] != ""),
                merged[knf_col],
            )
            merged.drop(columns=[knf_col], inplace=True)

    n_filled = len(merged[merged["category"].notna()]) - len(match_df[match_df.get("category", pd.Series()).notna()]) if "category" in match_df.columns else 0
    log.info("Metadata refresh complete. category non-null: %d / %d",
             merged["category"].notna().sum(), len(merged))

    return merged


# ---------------------------------------------------------------------------
# Phase 1: fetch all NAV series
# ---------------------------------------------------------------------------

def fetch_all_nav(match_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Downloads NAV for every fund in match_df.
    Returns {str(stooq_id): prices_series}.
    Funds that fail to download are absent from the dict.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nav = {}
    n   = len(match_df)

    for i, (_, fund) in enumerate(match_df.iterrows()):
        sid      = fund.get("stooq_id")
        knf_name = fund.get("knf_name", fund.get("name", ""))
        log.info("[%d/%d]  stooq=%-6s  %s", i + 1, n, sid, str(knf_name)[:60])

        prices = download_csv(sid)
        if prices is not None:
            log.info("  OK — %d days (%.1fy)", len(prices), len(prices) / TDAYS)
            nav[str(sid)] = prices
        time.sleep(random.uniform(STOOQ_DELAY_MIN, STOOQ_DELAY_MAX))

    log.info("NAV fetched: %d / %d funds", len(nav), n)
    return nav


# ---------------------------------------------------------------------------
# Phase 2: compute metrics (absolute + relative) for all funds × windows
# ---------------------------------------------------------------------------

def build_performance_table(
    match_df: pd.DataFrame,
    nav:      dict[str, pd.Series],
) -> pd.DataFrame:
    """
    For every (fund, window):
      - absolute metrics from NAV slice
      - category-relative metrics vs equal-weighted benchmark built on the fly
        from all funds in the same category that have data for that window
    """
    rows = []

    for window_name, window_days in WINDOWS.items():
        log.info("--- Window: %s ---", window_name)

        # Build per-category log-return DataFrames for this window
        # {category: DataFrame(index=date, columns=stooq_id_str)}
        cat_ret_frames: dict[str, pd.DataFrame] = {}

        for _, fund in match_df.iterrows():
            sid      = str(fund.get("stooq_id"))
            category = str(fund.get("category", ""))
            prices   = nav.get(sid)
            if prices is None:
                continue

            p_w = prices.iloc[-window_days:] if window_days and len(prices) >= window_days else prices
            lr  = np.log(p_w / p_w.shift(1)).dropna()

            if category not in cat_ret_frames:
                cat_ret_frames[category] = pd.DataFrame()
            cat_ret_frames[category][sid] = lr

        # Equal-weighted daily benchmark per category
        cat_benchmarks: dict[str, pd.Series] = {
            cat: frame.mean(axis=1)
            for cat, frame in cat_ret_frames.items()
        }
        log.info("  Benchmarks built for %d categories", len(cat_benchmarks))

        # Compute metrics for each fund
        for _, fund in match_df.iterrows():
            sid      = str(fund.get("stooq_id"))
            category = str(fund.get("category", ""))
            prices   = nav.get(sid)

            base_row = {
                "subfundId":                  fund.get("subfundId"),
                "stooq_id":                   fund.get("stooq_id"),
                "knf_name":                   fund.get("knf_name", fund.get("name", "")),
                "tfi_name":                   fund.get("tfi_name", ""),
                "category":                   category,
                "classification":             fund.get("classification", ""),
                "riskLevel":                  fund.get("riskLevel"),
                "managementAndOtherFeesRate": fund.get("managementAndOtherFeesRate"),
                "maxEntryFeeRate":            fund.get("maxEntryFeeRate"),
                "hasBenchmark":               fund.get("hasBenchmark"),
                "benchmark":                  fund.get("benchmark", ""),
                "window":                     window_name,
            }

            n_peers  = cat_ret_frames.get(category, pd.DataFrame()).shape[1]
            peers_ok = n_peers >= MIN_CATEGORY_PEERS

            if prices is None:
                nan = float("nan")
                rows.append({
                    **base_row,
                    "ann_return": nan, "ann_vol": nan, "sharpe": nan,
                    "sortino": nan, "max_drawdown": nan, "calmar": nan,
                    "data_years": nan, "n_days": 0,
                    "insufficient": True, "fetch_ok": False,
                    "alpha": nan, "beta": nan,
                    "tracking_error": nan, "ir": nan,
                    "pct_ir_positive": nan, "n_ir_windows": 0,
                    "category_funds": n_peers, "category_peers_ok": peers_ok,
                })
                continue

            abs_m   = compute_absolute_metrics(prices, window_days)
            log_ret = abs_m.pop("_log_ret")

            bench  = cat_benchmarks.get(category, pd.Series(dtype=float))
            rel_m  = compute_relative_metrics(log_ret, bench)
            roll_m = compute_rolling_ir(log_ret, bench)

            rows.append({
                **base_row,
                **abs_m,
                "fetch_ok":          True,
                **rel_m,
                **roll_m,
                "category_funds":    n_peers,
                "category_peers_ok": peers_ok,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 3: within-category ranking
# ---------------------------------------------------------------------------

def build_ranking(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (category, window), rank funds by IR (primary).
    Funds without a valid IR (single-fund category, insufficient history)
    are ranked last. Secondary ranks by Sharpe, Sortino, Calmar, Return
    are also added for reference.
    """
    ranking_rows = []

    for (category, window), grp in perf_df.groupby(["category", "window"]):
        grp = grp.copy()

        has_ir   = grp["fetch_ok"] & ~grp["ir"].isna()
        ranked   = grp[has_ir].sort_values("ir", ascending=False).copy()
        unranked = grp[~has_ir].copy()

        ranked["rank_ir"] = range(1, len(ranked) + 1)
        for col, src in [
            ("rank_sharpe",  "sharpe"),
            ("rank_sortino", "sortino"),
            ("rank_calmar",  "calmar"),
            ("rank_return",  "ann_return"),
        ]:
            ranked[col] = ranked[src].rank(
                ascending=False, method="min", na_option="bottom"
            ).astype("Int64")

        offset = len(ranked) + 1
        for col in ("rank_ir", "rank_sharpe", "rank_sortino", "rank_calmar", "rank_return"):
            unranked[col] = pd.array(
                [offset + i for i in range(len(unranked))], dtype="Int64"
            )

        ranking_rows.append(pd.concat([ranked, unranked], ignore_index=True))

    return pd.concat(ranking_rows, ignore_index=True) if ranking_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y%m%d")

    log.info("Fund performance comparison — start  (%s)", today)
    log.info("GDRIVE_FOLDER      : %s", GDRIVE_FOLDER)
    log.info("CSV_BASE_URL       : %s", CSV_BASE_URL)
    log.info("STOOQ_MAX_RETRIES  : %d  BACKOFF: %.1fs", STOOQ_MAX_RETRIES, STOOQ_BACKOFF)
    log.info("MIN_HISTORY_DAYS   : %d", MIN_HISTORY_DAYS)
    log.info("MIN_CATEGORY_PEERS : %d", MIN_CATEGORY_PEERS)
    log.info("Windows            : %s", list(WINDOWS.keys()))

    service  = get_drive_service()
    match_df = load_match_list(service)
    log.info("Match list loaded: %d funds, %d categories",
             len(match_df),
             match_df["category"].nunique() if "category" in match_df.columns else 0)

    # Phase 1a — refresh KNF metadata for all funds (fills gaps for manually
    # added funds that were never in knf_stooq_matches.csv)
    log.info("=== Phase 1a: Fetching KNF metadata ===")
    match_df = fetch_knf_metadata(match_df)
    log.info("Match list after metadata refresh: %d funds, %d categories",
             len(match_df),
             match_df["category"].nunique() if "category" in match_df.columns else 0)

    # Phase 1 — download all NAV series up front
    log.info("=== Phase 1: Downloading NAV series ===")
    nav = fetch_all_nav(match_df)

    # Phase 2 — compute absolute + category-relative metrics
    log.info("=== Phase 2: Computing metrics ===")
    perf_df = build_performance_table(match_df, nav)
    log.info("Performance table: %d rows (%d funds x %d windows)",
             len(perf_df), len(match_df), len(WINDOWS))

    # Phase 3 — rank within category
    log.info("=== Phase 3: Ranking ===")
    rank_df = build_ranking(perf_df)

    # Summary: top 3 per category by IR, 3Y window
    log.info("\n--- IR summary (3Y window, top 3 per category) ---")
    w3 = rank_df[
        (rank_df["window"] == "3y") &
        rank_df["fetch_ok"] &
        ~rank_df["ir"].isna()
    ]
    for cat, grp in w3.groupby("category"):
        top = grp.nsmallest(3, "rank_ir")[
            ["rank_ir", "knf_name", "ir", "alpha", "sharpe", "sortino", "ann_return"]
        ]
        log.info("  %-40s  (%d funds with valid IR)", cat, len(grp))
        for _, r in top.iterrows():
            log.info(
                "    #%d  IR=%+.3f  alpha=%+.1f%%  sharpe=%.2f"
                "  sortino=%.2f  ret=%+.1f%%  %s",
                r["rank_ir"], r["ir"], r["alpha"] * 100,
                r["sharpe"], r["sortino"], r["ann_return"] * 100,
                str(r["knf_name"])[:50],
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save(df: pd.DataFrame, filename: str):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s  (%d rows)", filename, len(df))
        upload_csv_to_drive(service, GDRIVE_FOLDER, path)

    save(perf_df, f"fund_performance_{today}.csv")
    save(rank_df, f"fund_ranking_{today}.csv")

    log.info("Fund performance comparison — complete")