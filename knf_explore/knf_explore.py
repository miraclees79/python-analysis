# knf_explore.py
#
# Exploration script for the KNF "Wybieram fundusze" API.
# https://wybieramfundusze-api.knf.gov.pl
#
# Findings from first run:
#   - List endpoints return lightweight records (id, name, type, _links only)
#   - classification, category, fromDate live in the detail endpoint
#   - KID list returns minimal fields; riskLevel, benchmark etc. are in detail
#   - This script therefore: pages the list to get all IDs, then fetches each
#     detail record individually with polite rate limiting
#
# What this script produces:
#   knf_subfunds.csv       — full active subfund list with classification/category
#   knf_history_probe.csv  — NAV history depth sample (earliest date, obs count)
#   knf_kid_data.csv       — KID detail: benchmark, fees, risk level per subfund
#   knf_summary.csv        — joined table (subfund + KID), ready for Sheets review
#
# Required GitHub Actions secrets:
#   GOOGLE_CREDENTIALS   — service account JSON (raw, pasted as-is)
#   GDRIVE_FOLDER_ID     — target Drive folder ID
#
# Optional env vars (with defaults):
#   KNF_PAGE_SIZE        — items per list-endpoint page   (default 200)
#   HISTORY_SAMPLE_N     — subfunds to probe for NAV depth (default 50)
#   HISTORY_FROM_DATE    — earliest NAV date to query      (default 2000-01-01)

import os
import time
import random
import logging
import requests
import pandas as pd
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

BASE_URL      = "https://wybieramfundusze-api.knf.gov.pl"
PAGE_SIZE     = int(os.getenv("KNF_PAGE_SIZE",     "200"))
HISTORY_N     = int(os.getenv("HISTORY_SAMPLE_N",  "50"))
HISTORY_FROM  = os.getenv("HISTORY_FROM_DATE",     "2000-01-01")
OUTPUT_DIR    = os.getenv("OUTPUT_DIR",             "/tmp/knf_outputs")
GDRIVE_FOLDER = os.getenv("GDRIVE_FOLDER_ID",       "")
CREDS_PATH    = "/tmp/credentials.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_json(url, params=None):
    """Single GET with error handling. Returns parsed JSON or None."""
    try:
        r = SESSION.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("GET %s failed: %s", url, exc)
        return None


def fetch_all_ids(path, params):
    """
    Page through a list endpoint and return every item.
    List endpoints return lightweight records (id + name + type + _links).
    """
    params = {**params, "size": PAGE_SIZE, "page": 0}
    items  = []
    page   = 0

    while True:
        params["page"] = page
        body = get_json(f"{BASE_URL}{path}", params)
        if body is None:
            log.error("Page fetch failed at page %d for %s", page, path)
            break

        content     = body.get("content", [])
        page_info   = body.get("page", {})
        total_pages = page_info.get("totalPages", 1)

        items.extend(content)
        log.info("  %s  page %d/%d  (+%d, total %d)",
                 path, page + 1, total_pages, len(content), len(items))

        if page + 1 >= total_pages:
            break
        page += 1
        time.sleep(random.uniform(0.15, 0.35))

    return items


# ---------------------------------------------------------------------------
# STEP 1 — Active subfunds (list -> detail hydration)
# ---------------------------------------------------------------------------

def fetch_subfunds():
    log.info("=" * 60)
    log.info("STEP 1: Fetching active subfunds (list + detail)")
    log.info("=" * 60)

    # 1a — lightweight list to collect all subfundIds
    raw = fetch_all_ids("/v1/subfunds", {
        "includeInactive":    "false",
        "includeLiquidating": "false",
        "sort":               "name,asc",
    })

    if not raw:
        log.warning("No subfunds returned from list endpoint.")
        return pd.DataFrame()

    log.info("List endpoint: %d subfunds. Fetching detail records...", len(raw))

    # 1b — fetch detail for each subfundId
    rows = []
    for i, s in enumerate(raw):
        sfid   = s["subfundId"]
        detail = get_json(f"{BASE_URL}/v1/subfunds/{sfid}") or s

        rows.append({
            "subfundId":      detail.get("subfundId"),
            "fundId":         detail.get("fundId"),
            "nationalCode":   detail.get("nationalCode"),
            "name":           detail.get("name"),
            "fullName":       detail.get("fullName"),
            "type":           detail.get("type"),
            "classification": detail.get("classification"),
            "category":       detail.get("category"),
            "fromDate":       detail.get("fromDate"),
            "isLiquidating":  detail.get("isLiquidating", False),
            "isActive":       detail.get("isActive", True),
        })

        if (i + 1) % 100 == 0:
            log.info("  ... %d / %d details fetched", i + 1, len(raw))

        time.sleep(random.uniform(0.1, 0.2))

    df = pd.DataFrame(rows)
    log.info("Subfunds hydrated: %d", len(df))

    # Category summary
    log.info("\n--- Subfund count by category ---")
    cat_counts = (
        df.groupby(["classification", "category"], dropna=False)
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    for _, row in cat_counts.iterrows():
        log.info("  %-20s | %-45s | %d",
                 row["classification"], row["category"], row["count"])

    return df


# ---------------------------------------------------------------------------
# STEP 2a — Known long-running funds: deep history probe without isPrimary
# ---------------------------------------------------------------------------

# These are confirmed long-running funds with stooq history since the 1990s.
# We test them without isPrimary to see whether the KNF API holds pre-2024 NAV.
# subfundId -> (name, stooq_code, stooq_history_since)
KNOWN_LONG_FUNDS = {
    196067: ("Goldman Sachs Akcji",       "2718", "1999"),
    197630: ("Skarbiec Akcji Polskich",   "3872", "1997"),
}

def probe_known_funds() -> pd.DataFrame:
    """
    For each known long-running fund, query the full valuation history
    WITHOUT the isPrimary filter to see the maximum depth available.
    Also lists all available unit classes so we understand what isPrimary
    was hiding.
    """
    log.info("=" * 60)
    log.info("STEP 2a: Deep history probe for known long-running funds")
    log.info("=" * 60)

    results = []

    for sfid, (name, stooq_code, stooq_since) in KNOWN_LONG_FUNDS.items():
        log.info("  Probing subfundId=%d  %s  (stooq: %s, hist since %s)",
                 sfid, name, stooq_code, stooq_since)

        # --- What unit classes exist for this subfund? ---
        units_body = get_json(f"{BASE_URL}/v1/valuations/units", {
            "subfundId": sfid,
            "fromDate":  HISTORY_FROM,
            "sort":      "date,asc",
            "size":      200,   # get enough to see all unit types
            "page":      0,
        })

        if units_body is None:
            log.warning("    Request failed for subfundId=%d", sfid)
            results.append({
                "subfundId": sfid, "name": name,
                "stooq_code": stooq_code, "stooq_since": stooq_since,
                "unit_class": None, "isPrimary": None,
                "earliest_date": None, "total_obs": None, "error": "request failed",
            })
            continue

        content   = units_body.get("content", [])
        page_info = units_body.get("page", {})
        total_obs = page_info.get("totalElements", 0)

        # Collect distinct unit classes seen in this page
        unit_classes = {}
        for rec in content:
            uc = rec.get("unit")
            if uc not in unit_classes:
                unit_classes[uc] = {
                    "isPrimary":   rec.get("isPrimary"),
                    "isArticle83": rec.get("isArticle83"),
                    "currency":    rec.get("currency"),
                    "first_date":  rec.get("date"),
                }

        log.info("    Unit classes found on first page: %s",
                 {k: v["isPrimary"] for k, v in unit_classes.items()})
        log.info("    Total valuation records (all units, from %s): %d",
                 HISTORY_FROM, total_obs)

        # Earliest date across all units on this first page
        earliest_all = content[0]["date"] if content else None
        log.info("    Earliest date (any unit, page 0): %s", earliest_all)

        # --- Now probe each unit class individually ---
        for uc, meta in unit_classes.items():
            uc_body = get_json(f"{BASE_URL}/v1/valuations/units", {
                "subfundId": sfid,
                "unit":      uc,
                "fromDate":  HISTORY_FROM,
                "sort":      "date,asc",
                "size":      1,
                "page":      0,
            })
            if uc_body is None:
                continue

            uc_page  = uc_body.get("page", {})
            uc_total = uc_page.get("totalElements", 0)
            uc_cont  = uc_body.get("content", [])
            uc_early = uc_cont[0]["date"] if uc_cont else None

            log.info("    unit=%-4s  isPrimary=%-5s  earliest=%-12s  obs=%d",
                     uc, meta["isPrimary"], uc_early or "none", uc_total)

            results.append({
                "subfundId":    sfid,
                "name":         name,
                "stooq_code":   stooq_code,
                "stooq_since":  stooq_since,
                "unit_class":   uc,
                "isPrimary":    meta["isPrimary"],
                "isArticle83":  meta["isArticle83"],
                "currency":     meta["currency"],
                "earliest_date": uc_early,
                "total_obs":    uc_total,
                "error":        None,
            })
            time.sleep(random.uniform(0.1, 0.2))

        time.sleep(random.uniform(0.2, 0.4))

    df = pd.DataFrame(results)

    # Summary
    if not df.empty and df["earliest_date"].notna().any():
        valid = df[df["earliest_date"].notna()].copy()
        valid["earliest_date"] = pd.to_datetime(valid["earliest_date"])
        today = pd.Timestamp(date.today())
        valid["years_history"] = (today - valid["earliest_date"]).dt.days / 365.25

        log.info("\n--- Known funds history summary ---")
        for _, r in valid.iterrows():
            log.info("  %-45s  unit=%-4s  primary=%-5s  earliest=%s  yrs=%.1f  obs=%d",
                     r["name"][:45], r["unit_class"], r["isPrimary"],
                     r["earliest_date"].date(), r["years_history"], r["total_obs"])

    return df


# ---------------------------------------------------------------------------
# STEP 2b — Broader category sample without isPrimary filter
# ---------------------------------------------------------------------------

def probe_nav_history(subfund_df):
    """
    Sample HISTORY_N subfunds spread across the core investable categories,
    query NAV history back to HISTORY_FROM WITHOUT the isPrimary filter.
    Previous run with isPrimary showed universal 2024-01-01 floor — this run
    checks whether dropping the filter reveals older data.

    Core categories sampled (PPK, lifecycle, alternatywne excluded):
        akcji polskich, akcji zagranicznych, akcji pozostałe,
        obligacji skarbowych, obligacji korporacyjnych,
        obligacji uniwersalne, obligacji pozostałe,
        mieszane ograniczonej alokacji, mieszane aktywnej alokacji,
        mieszane pozostałe, absolutnej stopy zwrotu, rynku surowców
    """
    log.info("=" * 60)
    log.info("STEP 2b: Category history probe — no isPrimary filter (n=%d)", HISTORY_N)
    log.info("=" * 60)

    empty_result = pd.DataFrame(columns=[
        "subfundId", "name", "category",
        "earliest_date", "total_obs", "error"
    ])

    if subfund_df.empty:
        log.warning("Subfund DataFrame is empty — skipping history probe.")
        return empty_result

    # Exclude categories that are out of scope for the assessment pipeline
    EXCLUDE_GROUPS = {"PPK", "alternatywne"}
    EXCLUDE_CATEGORY_PATTERNS = ["cyklu życia", "zdefiniowanej daty"]

    def is_excluded(row):
        if row["classification"] in EXCLUDE_GROUPS:
            return True
        cat = row["category"] or ""
        return any(p in cat for p in EXCLUDE_CATEGORY_PATTERNS)

    core = subfund_df[~subfund_df.apply(is_excluded, axis=1)].copy()
    log.info("  Core investable subfunds (after exclusions): %d / %d",
             len(core), len(subfund_df))

    # Stratified sample across core categories
    pieces = []
    categories = core["category"].dropna().unique()
    per_cat    = max(2, HISTORY_N // max(len(categories), 1))
    for cat in categories:
        grp = core[core["category"] == cat]
        pieces.append(grp.sample(min(len(grp), per_cat), random_state=42))

    if not pieces:
        log.warning("No subfunds available after exclusions.")
        return empty_result

    sampled = pd.concat(pieces).head(HISTORY_N).reset_index(drop=True)
    log.info("Sampling %d subfunds across %d categories",
             len(sampled), len(categories))

    results = []
    for _, row in sampled.iterrows():
        sfid = row["subfundId"]
        name = row["name"]
        cat  = row.get("category") or "unclassified"

        # No isPrimary filter — get oldest record across all unit classes
        body = get_json(f"{BASE_URL}/v1/valuations/units", {
            "subfundId": sfid,
            "fromDate":  HISTORY_FROM,
            "sort":      "date,asc",
            "size":      1,
            "page":      0,
        })

        if body is None:
            results.append({"subfundId": sfid, "name": name, "category": cat,
                             "earliest_date": None, "total_obs": None,
                             "error": "request failed"})
            time.sleep(0.3)
            continue

        page_info = body.get("page", {})
        total_obs = page_info.get("totalElements", 0)
        content   = body.get("content", [])
        earliest  = content[0]["date"] if content else None

        results.append({"subfundId": sfid, "name": name, "category": cat,
                        "earliest_date": earliest, "total_obs": total_obs,
                        "error": None})

        log.info("  %6d  %-45s  earliest=%-12s  obs=%s",
                 sfid, name[:45], earliest or "none", total_obs)
        time.sleep(random.uniform(0.15, 0.3))

    if not results:
        return empty_result

    df_hist = pd.DataFrame(results)

    valid = df_hist[df_hist["earliest_date"].notna()].copy()
    if not valid.empty:
        valid["earliest_date"] = pd.to_datetime(valid["earliest_date"])
        today = pd.Timestamp(date.today())
        valid["years_history"] = (today - valid["earliest_date"]).dt.days / 365.25

        log.info("\n--- NAV history depth summary (no isPrimary filter) ---")
        log.info("  Median years : %.1f", valid["years_history"].median())
        log.info("  >= 5 years   : %d / %d",
                 (valid["years_history"] >= 5).sum(), len(valid))
        log.info("  >= 10 years  : %d / %d",
                 (valid["years_history"] >= 10).sum(), len(valid))
        log.info("  >= 15 years  : %d / %d",
                 (valid["years_history"] >= 15).sum(), len(valid))
        log.info("  >= 20 years  : %d / %d",
                 (valid["years_history"] >= 20).sum(), len(valid))

        by_cat = (
            valid.groupby("category")["years_history"]
                 .agg(median="median", min="min", count="count")
                 .reset_index()
                 .sort_values("median", ascending=False)
        )
        log.info("\n--- History by category (no isPrimary filter) ---")
        for _, r in by_cat.iterrows():
            log.info("  %-45s  median=%.1fy  min=%.1fy  n=%d",
                     r["category"], r["median"], r["min"], r["count"])

    return df_hist


# ---------------------------------------------------------------------------
# STEP 3 — KID data via detail endpoint (benchmark, fees, risk)
# ---------------------------------------------------------------------------

def fetch_kid_data(subfund_df):
    """
    For each subfund: fetch the latest primary-unit KID UUID from the list
    endpoint (one call per subfund), then fetch its detail record for the
    full fields (benchmark, riskLevel, fees).
    """
    log.info("=" * 60)
    log.info("STEP 3: Fetching KID detail data")
    log.info("=" * 60)

    if subfund_df.empty:
        log.warning("Subfund DataFrame empty — skipping KID fetch.")
        return pd.DataFrame()

    subfund_ids = subfund_df["subfundId"].dropna().astype(int).tolist()
    log.info("Fetching latest KID UUID for %d subfunds...", len(subfund_ids))

    # 3a — get latest KID document UUID per subfund
    uuid_map = {}
    for i, sfid in enumerate(subfund_ids):
        body = get_json(f"{BASE_URL}/v1/key-information/units", {
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

        if (i + 1) % 100 == 0:
            log.info("  ... KID list: %d / %d  (%d UUIDs found)",
                     i + 1, len(subfund_ids), len(uuid_map))
        time.sleep(random.uniform(0.1, 0.2))

    log.info("KID UUIDs found: %d / %d subfunds", len(uuid_map), len(subfund_ids))

    # 3b — fetch detail for each UUID
    rows = []
    for i, (sfid, uuid) in enumerate(uuid_map.items()):
        detail = get_json(f"{BASE_URL}/v1/key-information/units/{uuid}")
        if detail is None:
            rows.append({"subfundId": sfid, "documentUuid": uuid})
            time.sleep(0.3)
            continue

        rows.append({
            "subfundId":                  sfid,
            "documentUuid":               uuid,
            "unit":                       detail.get("unit"),
            "currency":                   detail.get("currency"),
            "isPrimary":                  detail.get("isPrimary"),
            "publicationDate":            detail.get("publicationDate"),
            "riskLevel":                  detail.get("riskLevel"),
            "maxEntryFeeRate":            detail.get("maxEntryFeeRate"),
            "maxExitFeeRate":             detail.get("maxExitFeeRate"),
            "managementAndOtherFeesRate": detail.get("managementAndOtherFeesRate"),
            "hasSuccessFee":              detail.get("hasSuccessFee"),
            "hasBenchmark":               detail.get("hasBenchmark"),
            "benchmark":                  detail.get("benchmark"),
        })

        if (i + 1) % 100 == 0:
            log.info("  ... KID detail: %d / %d", i + 1, len(uuid_map))
        time.sleep(random.uniform(0.1, 0.2))

    df_kid = pd.DataFrame(rows)
    log.info("KID detail records: %d", len(df_kid))

    # Merge with subfund names and categories
    if not subfund_df.empty and not df_kid.empty:
        df_kid = df_kid.merge(
            subfund_df[["subfundId", "name", "classification", "category"]],
            on="subfundId", how="left",
        )

    if df_kid.empty:
        return df_kid

    # --- Analysis ---
    has_bm = df_kid["hasBenchmark"] == True
    log.info("\n--- Benchmark coverage ---")
    log.info("  Subfunds with KID data     : %d", len(df_kid))
    log.info("  hasBenchmark = True        : %d  (%.0f%%)",
             has_bm.sum(), 100 * has_bm.mean())
    log.info("  Non-null benchmark string  : %d",
             df_kid["benchmark"].notna().sum())

    bm_counts = (
        df_kid[df_kid["benchmark"].notna()]
        .groupby("benchmark").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    log.info("\n--- Most common benchmark strings (top 40) ---")
    for _, r in bm_counts.head(40).iterrows():
        log.info("  %3d  %s", r["count"], r["benchmark"])

    if "category" in df_kid.columns:
        bm_by_cat = (
            df_kid.groupby("category", dropna=False)["hasBenchmark"]
                  .agg(total="count",
                       has_bm=lambda x: (x == True).sum())
                  .reset_index()
        )
        bm_by_cat["pct"] = (
            100 * bm_by_cat["has_bm"] / bm_by_cat["total"]
        ).round(1)
        log.info("\n--- Benchmark coverage by category ---")
        for _, r in bm_by_cat.sort_values("pct", ascending=False).iterrows():
            log.info("  %-45s  %d/%d  (%.0f%%)",
                     r["category"], r["has_bm"], r["total"], r["pct"])

    return df_kid


# ---------------------------------------------------------------------------
# STEP 4 — Write CSV reports
# ---------------------------------------------------------------------------

def write_reports(subfund_df, known_df, hist_df, kid_df):
    written = []

    def save(df, filename, label):
        if df is None or df.empty:
            log.warning("Skipping %s — empty.", filename)
            return
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %-30s %d rows → %s", label, len(df), path)
        written.append(path)

    save(subfund_df, "knf_subfunds.csv",           "Active subfunds")
    save(known_df,   "knf_known_funds_probe.csv",  "Known funds deep probe")
    save(hist_df,    "knf_history_probe.csv",      "NAV history probe")
    save(kid_df,     "knf_kid_data.csv",           "KID data")

    if not subfund_df.empty and not kid_df.empty:
        summary_cols = [
            "subfundId", "name", "classification", "category", "type",
            "fromDate", "riskLevel", "managementAndOtherFeesRate",
            "maxEntryFeeRate", "hasBenchmark", "benchmark",
        ]
        summary = kid_df[[c for c in summary_cols if c in kid_df.columns]].copy()
        save(summary, "knf_summary.csv", "Summary (subfund+KID)")

    return written


# ---------------------------------------------------------------------------
# STEP 5 — Upload to Google Drive
# ---------------------------------------------------------------------------

def upload_to_drive(file_paths):
    if not GDRIVE_FOLDER:
        log.warning("GDRIVE_FOLDER_ID not set — skipping Drive upload.")
        return
    if not os.path.exists(CREDS_PATH):
        log.warning("Credentials not at %s — skipping.", CREDS_PATH)
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
            q = (f"name='{filename}' and "
                 f"'{GDRIVE_FOLDER}' in parents and trashed=false")
            existing = (service.files()
                               .list(q=q, fields="files(id)")
                               .execute()
                               .get("files", []))
            media = MediaFileUpload(path, mimetype="text/csv", resumable=True)
            if existing:
                fid = existing[0]["id"]
                service.files().update(fileId=fid, media_body=media).execute()
                log.info("Drive UPDATED : %s (id=%s)", filename, fid)
            else:
                meta = {"name": filename, "parents": [GDRIVE_FOLDER]}
                service.files().create(
                    body=meta, media_body=media, fields="id").execute()
                log.info("Drive CREATED : %s", filename)

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
    known_df   = probe_known_funds()
    hist_df    = probe_nav_history(subfund_df)
    kid_df     = fetch_kid_data(subfund_df)
    written    = write_reports(subfund_df, known_df, hist_df, kid_df)
    upload_to_drive(written)

    log.info("KNF API exploration — complete. Files written: %d", len(written))