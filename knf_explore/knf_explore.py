# knf_explore.py
#
# Exploration script for the KNF "Wybieram fundusze" API.
# https://wybieramfundusze-api.knf.gov.pl
#
# What this script does:
#   1. Fetches the full list of active subfunds (isPrimary unit class)
#   2. For each subfund, records: subfundId, fundId, name, type,
#      classification, category, fromDate, isLiquidating
#   3. Samples NAV history for a subset of subfunds to assess historical depth
#   4. Fetches KID data (primary unit) to extract benchmark strings,
#      fee rates, and risk levels
#   5. Writes four CSV reports to the output directory, then uploads
#      them all to a Google Drive folder
#
# Required GitHub Actions secrets:
#   GOOGLE_CREDENTIALS   — service account JSON (raw, pasted as-is)
#   GDRIVE_FOLDER_ID     — target Drive folder ID
#
# Optional environment variables (with defaults):
#   KNF_PAGE_SIZE        — items per API page (default 200, max ~500 in practice)
#   HISTORY_SAMPLE_N     — how many subfunds to probe for NAV history (default 50)
#   HISTORY_FROM_DATE    — earliest date for history depth probe (default 2000-01-01)

import os
import time
import random
import logging
import requests
import pandas as pd
from io import BytesIO
from datetime import date

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL       = "https://wybieramfundusze-api.knf.gov.pl"
PAGE_SIZE      = int(os.getenv("KNF_PAGE_SIZE",      "200"))
HISTORY_N      = int(os.getenv("HISTORY_SAMPLE_N",   "50"))
HISTORY_FROM   = os.getenv("HISTORY_FROM_DATE",      "2000-01-01")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR",              "/tmp/knf_outputs")
GDRIVE_FOLDER  = os.getenv("GDRIVE_FOLDER_ID",        "")
CREDS_PATH     = "/tmp/credentials.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# Generic paged fetch
# ---------------------------------------------------------------------------

def fetch_all_pages(path: str, params: dict) -> list[dict]:
    """Fetch every page of a paged KNF endpoint and return all items."""
    params = {**params, "size": PAGE_SIZE, "page": 0}
    items  = []
    page   = 0

    while True:
        params["page"] = page
        url = f"{BASE_URL}{path}"

        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
        except Exception as exc:
            log.error("GET %s page %d failed: %s", path, page, exc)
            break

        content     = body.get("content", [])
        page_info   = body.get("page", {})
        total_pages = page_info.get("totalPages", 1)

        items.extend(content)
        log.info("  %s  page %d/%d  (+%d items, total so far %d)",
                 path, page + 1, total_pages, len(content), len(items))

        if page + 1 >= total_pages:
            break
        page += 1
        time.sleep(random.uniform(0.2, 0.5))   # polite rate limiting

    return items


# ---------------------------------------------------------------------------
# STEP 1 — Active subfunds
# ---------------------------------------------------------------------------

def fetch_subfunds() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1: Fetching active subfunds")
    log.info("=" * 60)

    raw = fetch_all_pages("/v1/subfunds", {
        "includeInactive":    "false",
        "includeLiquidating": "false",
        "sort":               "name,asc",
    })

    if not raw:
        log.warning("No subfunds returned.")
        return pd.DataFrame()

    rows = []
    for s in raw:
        rows.append({
            "subfundId":      s.get("subfundId"),
            "fundId":         s.get("fundId"),
            "nationalCode":   s.get("nationalCode"),
            "name":           s.get("name"),
            "fullName":       s.get("fullName"),
            "type":           s.get("type"),           # FIO / SFIO / FIZ
            "classification": s.get("classification"), # e.g. "akcji"
            "category":       s.get("category"),       # e.g. "akcji polskich"
            "fromDate":       s.get("fromDate"),
            "isLiquidating":  s.get("isLiquidating", False),
            "isActive":       s.get("isActive", True),
        })

    df = pd.DataFrame(rows)
    log.info("Total active subfunds fetched: %d", len(df))

    # Summary by category
    log.info("\n--- Subfund count by category ---")
    cat_counts = (
        df.groupby(["classification", "category"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    for _, row in cat_counts.iterrows():
        log.info("  %-20s | %-40s | %d",
                 row["classification"], row["category"], row["count"])

    return df


# ---------------------------------------------------------------------------
# STEP 2 — NAV history depth probe
# ---------------------------------------------------------------------------

def probe_nav_history(subfund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample HISTORY_N subfunds (spread across categories) and query their
    full NAV history back to HISTORY_FROM. Record the earliest available
    date and total observation count for each sampled subfund.
    """
    log.info("=" * 60)
    log.info("STEP 2: Probing NAV history depth (sample n=%d)", HISTORY_N)
    log.info("=" * 60)

    if subfund_df.empty:
        return pd.DataFrame()

    # Stratified sample: pick funds from each category proportionally
    categories = subfund_df["category"].unique()
    per_cat    = max(1, HISTORY_N // max(len(categories), 1))
    pieces     = []
    for cat in categories:
        group = subfund_df[subfund_df["category"] == cat]
        pieces.append(group.sample(min(len(group), per_cat), random_state=42))
    sampled = pd.concat(pieces).head(HISTORY_N).reset_index(drop=True)

    log.info("Sampling %d subfunds for NAV history probe", len(sampled))

    results = []

    for _, row in sampled.iterrows():
        sfid  = row["subfundId"]
        name  = row["name"]
        cat   = row["category"]

        # Fetch the oldest available page (sort date ascending, page 0)
        params = {
            "subfundId":  sfid,
            "isPrimary":  "true",
            "fromDate":   HISTORY_FROM,
            "sort":       "date,asc",
            "size":       1,
            "page":       0,
        }
        try:
            r = SESSION.get(f"{BASE_URL}/v1/valuations/units",
                            params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
        except Exception as exc:
            log.warning("  subfund %d (%s): history request failed: %s", sfid, name, exc)
            results.append({
                "subfundId":       sfid,
                "name":            name,
                "category":        cat,
                "earliest_date":   None,
                "total_obs":       None,
                "error":           str(exc),
            })
            time.sleep(0.3)
            continue

        page_info  = body.get("page", {})
        total_obs  = page_info.get("totalElements", 0)
        content    = body.get("content", [])
        earliest   = content[0]["date"] if content else None

        results.append({
            "subfundId":     sfid,
            "name":          name,
            "category":      cat,
            "earliest_date": earliest,
            "total_obs":     total_obs,
            "error":         None,
        })
        log.info("  subfund %d  %-45s  earliest=%s  obs=%s",
                 sfid, name[:45], earliest, total_obs)
        time.sleep(random.uniform(0.2, 0.5))

    if not results:
        log.warning("No history probe results collected.")
        return pd.DataFrame(columns=[
            "subfundId", "name", "category",
            "earliest_date", "total_obs", "error"
        ])

    df_hist = pd.DataFrame(results)

    # Summary statistics
    valid = df_hist[df_hist["earliest_date"].notna()].copy()
    if not valid.empty:
        valid["earliest_date"] = pd.to_datetime(valid["earliest_date"])
        today = pd.Timestamp(date.today())
        valid["years_history"] = (today - valid["earliest_date"]).dt.days / 365.25

        log.info("\n--- NAV history depth summary ---")
        log.info("  Median years of history : %.1f", valid["years_history"].median())
        log.info("  >= 10 years             : %d / %d",
                 (valid["years_history"] >= 10).sum(), len(valid))
        log.info("  >= 15 years             : %d / %d",
                 (valid["years_history"] >= 15).sum(), len(valid))
        log.info("  >= 20 years             : %d / %d",
                 (valid["years_history"] >= 20).sum(), len(valid))

        by_cat = (
            valid.groupby("category")["years_history"]
                 .agg(["median", "min", "count"])
                 .reset_index()
                 .sort_values("median", ascending=False)
        )
        log.info("\n--- History depth by category (sampled) ---")
        for _, r in by_cat.iterrows():
            log.info("  %-40s  median=%.1fy  min=%.1fy  n=%d",
                     r["category"], r["median"], r["min"], r["count"])

    return df_hist


# ---------------------------------------------------------------------------
# STEP 3 — KID data: benchmarks, fees, risk levels
# ---------------------------------------------------------------------------

def fetch_kid_data(subfund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch the current (active) primary-unit KID for every subfund and
    extract: benchmark string, risk level, management fee, entry fee.
    Paginates through /v1/key-information/units with isPrimary=true.
    """
    log.info("=" * 60)
    log.info("STEP 3: Fetching KID data (benchmarks, fees, risk levels)")
    log.info("=" * 60)

    raw = fetch_all_pages("/v1/key-information/units", {
        "isPrimary":      "true",
        "includeInactive": "false",
        "sort":            "publicationDate,desc",
    })

    if not raw:
        log.warning("No KID records returned.")
        return pd.DataFrame()

    rows = []
    for k in raw:
        rows.append({
            "documentUuid":             k.get("documentUuid"),
            "subfundId":                k.get("subfundId"),
            "fundId":                   k.get("fundId"),
            "unit":                     k.get("unit"),
            "currency":                 k.get("currency"),
            "isPrimary":                k.get("isPrimary"),
            "publicationDate":          k.get("publicationDate"),
            "riskLevel":                k.get("riskLevel"),
            "maxEntryFeeRate":          k.get("maxEntryFeeRate"),
            "maxExitFeeRate":           k.get("maxExitFeeRate"),
            "managementAndOtherFeesRate": k.get("managementAndOtherFeesRate"),
            "hasSuccessFee":            k.get("hasSuccessFee"),
            "hasBenchmark":             k.get("hasBenchmark"),
            "benchmark":                k.get("benchmark"),
        })

    df_kid = pd.DataFrame(rows)
    log.info("Total KID records fetched: %d", len(df_kid))

    # Keep only the most recent KID per subfund (API already sorts desc,
    # but there may be multiple per subfund if history is included)
    df_kid = (
        df_kid
        .sort_values("publicationDate", ascending=False)
        .drop_duplicates(subset="subfundId", keep="first")
    )
    log.info("After dedup (latest KID per subfund): %d", len(df_kid))

    # Merge with subfund names and categories
    if not subfund_df.empty:
        df_kid = df_kid.merge(
            subfund_df[["subfundId", "name", "classification", "category"]],
            on="subfundId",
            how="left",
        )

    # Benchmark analysis
    has_bm  = df_kid["hasBenchmark"] == True
    log.info("\n--- Benchmark coverage ---")
    log.info("  Subfunds with KID data:    %d", len(df_kid))
    log.info("  Has benchmark (hasBenchmark=True): %d  (%.0f%%)",
             has_bm.sum(), 100 * has_bm.mean() if len(df_kid) else 0)
    log.info("  Non-null benchmark string: %d",
             df_kid["benchmark"].notna().sum())

    # Benchmark string frequency
    bm_counts = (
        df_kid[df_kid["benchmark"].notna()]
        .groupby("benchmark")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    log.info("\n--- Most common benchmark strings (top 30) ---")
    for _, r in bm_counts.head(30).iterrows():
        log.info("  %3d  %s", r["count"], r["benchmark"])

    # Benchmark coverage by category
    if "category" in df_kid.columns:
        bm_by_cat = (
            df_kid.groupby("category")["hasBenchmark"]
                  .agg(total="count",
                       has_bm=lambda x: (x == True).sum())
                  .reset_index()
        )
        bm_by_cat["pct"] = (
            100 * bm_by_cat["has_bm"] / bm_by_cat["total"]
        ).round(1)
        log.info("\n--- Benchmark coverage by category ---")
        for _, r in bm_by_cat.sort_values("pct", ascending=False).iterrows():
            log.info("  %-40s  %d/%d  (%.0f%%)",
                     r["category"], r["has_bm"], r["total"], r["pct"])

    return df_kid


# ---------------------------------------------------------------------------
# STEP 4 — Write CSV reports
# ---------------------------------------------------------------------------

def write_reports(subfund_df, hist_df, kid_df) -> list[str]:
    """Write exploration results to CSV files. Returns list of written paths."""

    written = []

    def save(df, filename, label):
        if df is None or df.empty:
            log.warning("Skipping %s — empty DataFrame", filename)
            return
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s: %d rows  →  %s", label, len(df), path)
        written.append(path)

    save(subfund_df, "knf_subfunds.csv",        "Active subfunds")
    save(hist_df,    "knf_history_probe.csv",   "NAV history probe")
    save(kid_df,     "knf_kid_data.csv",        "KID data")

    # Summary table: subfund + KID joined, ready for Google Sheets review
    if not subfund_df.empty and not kid_df.empty:
        summary_cols = [
            "subfundId", "name", "classification", "category", "type",
            "fromDate", "riskLevel", "managementAndOtherFeesRate",
            "maxEntryFeeRate", "hasBenchmark", "benchmark",
        ]
        # kid_df already has name/classification/category from merge
        summary = kid_df[[c for c in summary_cols if c in kid_df.columns]].copy()
        save(summary, "knf_summary.csv", "Summary (subfund+KID)")

    return written


# ---------------------------------------------------------------------------
# STEP 5 — Upload to Google Drive
# ---------------------------------------------------------------------------

def upload_to_drive(file_paths: list[str]):
    if not GDRIVE_FOLDER:
        log.warning("GDRIVE_FOLDER_ID not set — skipping Drive upload.")
        return
    if not os.path.exists(CREDS_PATH):
        log.warning("Credentials not found at %s — skipping Drive upload.", CREDS_PATH)
        return

    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds   = Credentials.from_service_account_file(
                      CREDS_PATH,
                      scopes=["https://www.googleapis.com/auth/drive.file"])
        service = build("drive", "v3", credentials=creds)

        for path in file_paths:
            filename = os.path.basename(path)
            mimetype = "text/csv"

            q = (f"name='{filename}' and "
                 f"'{GDRIVE_FOLDER}' in parents and trashed=false")
            existing = (service.files()
                               .list(q=q, fields="files(id)")
                               .execute()
                               .get("files", []))

            media = MediaFileUpload(path, mimetype=mimetype, resumable=True)

            if existing:
                fid = existing[0]["id"]
                service.files().update(fileId=fid, media_body=media).execute()
                log.info("Drive UPDATED: %s (id=%s)", filename, fid)
            else:
                meta = {"name": filename, "parents": [GDRIVE_FOLDER]}
                service.files().create(
                    body=meta, media_body=media, fields="id"
                ).execute()
                log.info("Drive CREATED: %s", filename)

    except Exception as exc:
        log.error("Drive upload failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    log.info("KNF API exploration — start")
    log.info("BASE_URL      : %s", BASE_URL)
    log.info("PAGE_SIZE     : %d", PAGE_SIZE)
    log.info("HISTORY_N     : %d", HISTORY_N)
    log.info("HISTORY_FROM  : %s", HISTORY_FROM)
    log.info("OUTPUT_DIR    : %s", OUTPUT_DIR)

    subfund_df = fetch_subfunds()
    hist_df    = probe_nav_history(subfund_df)
    kid_df     = fetch_kid_data(subfund_df)
    written    = write_reports(subfund_df, hist_df, kid_df)
    upload_to_drive(written)

    log.info("KNF API exploration — complete. Files written: %d", len(written))