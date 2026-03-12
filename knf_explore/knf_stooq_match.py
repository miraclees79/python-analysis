# knf_stooq_match.py
#
# Matches KNF subfund names (from knf_summary.csv on Drive) against
# stooq fund names (from nazwy1000..5000.csv on Drive).
#
# Matching strategy — three passes in priority order:
#   1. Exact match (after basic lowercasing + whitespace normalisation)
#   2. Normalised match (strip legal suffixes, TFI prefixes, punctuation)
#   3. Fuzzy match (token-sort ratio >= threshold)
#
# Each KNF subfund gets the best match found, the match tier, and the
# similarity score.  Unmatched records are flagged explicitly.
#
# Outputs (uploaded to same Drive folder):
#   knf_stooq_matches.csv    — full match table, one row per KNF subfund
#   knf_stooq_unmatched.csv  — KNF subfunds with no match above threshold
#   knf_stooq_review.csv     — matches with score < 90 (fuzzy, needs review)
#
# Required secrets / env vars:
#   GOOGLE_CREDENTIALS   — service account JSON
#   GDRIVE_FOLDER_ID     — Drive folder containing all input and output files
#
# Dependencies:
#   pip install pandas rapidfuzz google-auth google-api-python-client

import os
import io
import re
import logging
import unicodedata
import pandas as pd
from rapidfuzz import process, fuzz

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GDRIVE_FOLDER = os.getenv("GDRIVE_FOLDER_ID", "")
CREDS_PATH    = "/tmp/credentials.json"
OUTPUT_DIR    = "/tmp/knf_match"
FUZZY_THRESHOLD = 75   # minimum score to accept a fuzzy match (0-100)

NAZWY_FILES = [
    "nazwy1000.csv",
    "nazwy2000.csv",
    "nazwy3000.csv",
    "nazwy4000.csv",
    "nazwy5000.csv",
]
KNF_SUMMARY_FILE = "knf_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def get_drive_service():
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_file(
        CREDS_PATH, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)


def download_csv_from_drive(service, folder_id: str, filename: str,
                             sep: str = ";") -> pd.DataFrame | None:
    """Download a CSV file by name from a Drive folder into a DataFrame."""
    q = (f"name='{filename}' and '{folder_id}' in parents and trashed=false")
    results = service.files().list(q=q, fields="files(id,name)").execute()
    files   = results.get("files", [])

    if not files:
        log.warning("File not found in Drive: %s", filename)
        return None

    file_id = files[0]["id"]
    from googleapiclient.http import MediaIoBaseDownload
    request    = service.files().get_media(fileId=file_id)
    buf        = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buf.seek(0)
    try:
        df = pd.read_csv(buf, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        buf.seek(0)
        df = pd.read_csv(buf, sep=sep, encoding="cp1250")

    log.info("Downloaded %-30s  %d rows", filename, len(df))
    return df


def upload_csv_to_drive(service, folder_id: str, local_path: str):
    from googleapiclient.http import MediaFileUpload
    filename = os.path.basename(local_path)
    q = (f"name='{filename}' and '{folder_id}' in parents and trashed=false")
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    media    = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    if existing:
        fid = existing[0]["id"]
        service.files().update(fileId=fid, media_body=media).execute()
        log.info("Drive UPDATED : %s (id=%s)", filename, fid)
    else:
        meta = {"name": filename, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()
        log.info("Drive CREATED : %s", filename)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Legal / structural tokens commonly found in fund names that differ
# between KNF (full legal name) and stooq (page title abbreviation)
STRIP_TOKENS = [
    # fund type suffixes
    r"\bfundusz inwestycyjny otwarty\b",
    r"\bfundusz inwestycyjny zamknięty\b",
    r"\bspecjalistyczny fundusz inwestycyjny otwarty\b",
    r"\bfundusz inwestycyjny\b",
    r"\bsfio\b", r"\bfio\b", r"\bfiz\b",
    # structural prefixes/suffixes
    r"\bsubfundusz\b",
    r"\btowarzystwo funduszy inwestycyjnych\b",
    r"\btfi\b",
    # stooq title suffix
    r"- notowania, wykresy, dane historyczne - stooq",
    r"- stooq",
    # common TFI brand prefixes that appear inconsistently
    r"\bgoldman sachs\b",
    r"\bpko\b",
    r"\bpzu\b",
    r"\bsantander\b",
    r"\bpekao\b",
    r"\bskarbiec\b",
    r"\bnn\b",
    r"\bepsilon\b",
    r"\binpzu\b",
]

STRIP_PATTERN = re.compile("|".join(STRIP_TOKENS), re.IGNORECASE)


def to_ascii(s: str) -> str:
    """Fold Polish diacritics to ASCII equivalents."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def normalise(name: str) -> str:
    """
    Normalise a fund name for matching:
      1. Lowercase
      2. Fold diacritics
      3. Strip legal/structural tokens
      4. Collapse whitespace
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = to_ascii(s)
    s = STRIP_PATTERN.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)   # keep only alphanum + space
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

def load_stooq_names(service) -> pd.DataFrame:
    """
    Download all five nazwy CSVs from Drive and concatenate.
    Columns expected: "Title", "Numer kolejny"
    """
    pieces = []
    for fname in NAZWY_FILES:
        df = download_csv_from_drive(service, GDRIVE_FOLDER, fname)
        if df is not None and not df.empty:
            pieces.append(df)

    if not pieces:
        raise RuntimeError("No nazwy files could be downloaded from Drive.")

    stooq = pd.concat(pieces, ignore_index=True)

    # Rename for clarity
    stooq = stooq.rename(columns={
        "Title":         "stooq_title",
        "Numer kolejny": "stooq_id",
    })

    # Drop rows where title is empty or just whitespace
    stooq = stooq[stooq["stooq_title"].notna()].copy()
    stooq["stooq_title"] = stooq["stooq_title"].str.strip()
    stooq = stooq[stooq["stooq_title"] != ""].copy()

    # Add normalised column
    stooq["stooq_norm"] = stooq["stooq_title"].apply(normalise)

    log.info("Stooq names loaded: %d records", len(stooq))
    return stooq.reset_index(drop=True)


def load_knf_summary(service) -> pd.DataFrame:
    """Download knf_summary.csv from Drive."""
    df = download_csv_from_drive(service, GDRIVE_FOLDER, KNF_SUMMARY_FILE)
    if df is None or df.empty:
        raise RuntimeError("knf_summary.csv not found or empty on Drive.")

    # Add normalised column
    df["knf_norm"] = df["name"].apply(normalise)
    log.info("KNF summary loaded: %d records", len(df))
    return df


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_funds(knf_df: pd.DataFrame, stooq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-pass matching:
      Pass 1 — exact match on normalised name
      Pass 2 — exact match on normalised name with TFI brand prefix stripped
               (handles cases like "Goldman Sachs Subfundusz Akcji" vs
               "GS Akcji" — won't catch but at least the branded stripping helps)
      Pass 3 — fuzzy token-sort match using rapidfuzz

    Returns one row per KNF subfund with columns:
      subfundId, name, category, classification,
      match_tier (exact / fuzzy / none),
      match_score (100 for exact, 0-100 for fuzzy),
      stooq_id, stooq_title,
      knf_norm, stooq_norm_matched
    """
    stooq_norms  = stooq_df["stooq_norm"].tolist()
    stooq_index  = stooq_df.index.tolist()

    results = []

    for _, knf_row in knf_df.iterrows():
        knf_norm = knf_row["knf_norm"]

        result = {
            "subfundId":          knf_row.get("subfundId"),
            "name":               knf_row.get("name"),
            "classification":     knf_row.get("classification"),
            "category":           knf_row.get("category"),
            "fromDate":           knf_row.get("fromDate"),
            "riskLevel":          knf_row.get("riskLevel"),
            "managementAndOtherFeesRate": knf_row.get("managementAndOtherFeesRate"),
            "maxEntryFeeRate":    knf_row.get("maxEntryFeeRate"),
            "hasBenchmark":       knf_row.get("hasBenchmark"),
            "benchmark":          knf_row.get("benchmark"),
            "knf_norm":           knf_norm,
            "match_tier":         "none",
            "match_score":        0,
            "stooq_id":           None,
            "stooq_title":        None,
            "stooq_norm_matched": None,
        }

        if not knf_norm:
            results.append(result)
            continue

        # --- Pass 1: exact normalised match ---
        exact_matches = stooq_df[stooq_df["stooq_norm"] == knf_norm]
        if not exact_matches.empty:
            best = exact_matches.iloc[0]
            result.update({
                "match_tier":         "exact",
                "match_score":        100,
                "stooq_id":           best["stooq_id"],
                "stooq_title":        best["stooq_title"],
                "stooq_norm_matched": best["stooq_norm"],
            })
            results.append(result)
            continue

        # --- Pass 2: fuzzy token-sort ratio ---
        # rapidfuzz.process.extractOne returns (match_string, score, index)
        best_match = process.extractOne(
            knf_norm,
            stooq_norms,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )

        if best_match is not None:
            matched_norm, score, idx = best_match
            stooq_row = stooq_df.iloc[idx]
            result.update({
                "match_tier":         "fuzzy",
                "match_score":        round(score, 1),
                "stooq_id":           stooq_row["stooq_id"],
                "stooq_title":        stooq_row["stooq_title"],
                "stooq_norm_matched": matched_norm,
            })

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyse_matches(match_df: pd.DataFrame):
    total   = len(match_df)
    exact   = (match_df["match_tier"] == "exact").sum()
    fuzzy   = (match_df["match_tier"] == "fuzzy").sum()
    none    = (match_df["match_tier"] == "none").sum()
    review  = (
        (match_df["match_tier"] == "fuzzy") &
        (match_df["match_score"] < 90)
    ).sum()

    log.info("\n--- Match summary ---")
    log.info("  Total KNF subfunds   : %d", total)
    log.info("  Exact matches        : %d  (%.0f%%)", exact,  100 * exact  / total)
    log.info("  Fuzzy matches        : %d  (%.0f%%)", fuzzy,  100 * fuzzy  / total)
    log.info("    of which score<90  : %d  (needs review)", review)
    log.info("  No match             : %d  (%.0f%%)", none,   100 * none   / total)

    # By category
    log.info("\n--- Match rate by category ---")
    cat_stats = (
        match_df.groupby("category")
                .apply(lambda g: pd.Series({
                    "total":   len(g),
                    "matched": (g["match_tier"] != "none").sum(),
                    "exact":   (g["match_tier"] == "exact").sum(),
                }), include_groups=False)
                .reset_index()
    )
    cat_stats["pct"] = (100 * cat_stats["matched"] / cat_stats["total"]).round(1)
    for _, r in cat_stats.sort_values("pct", ascending=False).iterrows():
        log.info("  %-45s  %d/%d (%.0f%%)  exact=%d",
                 r["category"], r["matched"], r["total"], r["pct"], r["exact"])

    # Sample of fuzzy matches needing review
    review_df = match_df[
        (match_df["match_tier"] == "fuzzy") &
        (match_df["match_score"] < 90)
    ].sort_values("match_score")

    if not review_df.empty:
        log.info("\n--- Fuzzy matches scoring < 90 (sample, worst first) ---")
        for _, r in review_df.head(20).iterrows():
            log.info("  score=%5.1f  KNF: %-45s  stooq: %s",
                     r["match_score"], str(r["name"])[:45], r["stooq_title"])

    # Sample of unmatched
    unmatched = match_df[match_df["match_tier"] == "none"]
    if not unmatched.empty:
        log.info("\n--- Unmatched KNF subfunds (sample) ---")
        for _, r in unmatched.head(20).iterrows():
            log.info("  category=%-35s  name=%s",
                     str(r["category"])[:35], r["name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("KNF-stooq name matching — start")
    log.info("GDRIVE_FOLDER : %s", GDRIVE_FOLDER)
    log.info("FUZZY_THRESHOLD: %d", FUZZY_THRESHOLD)

    service  = get_drive_service()
    stooq_df = load_stooq_names(service)
    knf_df   = load_knf_summary(service)

    log.info("\nRunning matching (%d KNF × %d stooq)...", len(knf_df), len(stooq_df))
    match_df = match_funds(knf_df, stooq_df)

    analyse_matches(match_df)

    # --- Write outputs ---
    def save(df, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s  (%d rows)", filename, len(df))
        upload_csv_to_drive(service, GDRIVE_FOLDER, path)

    save(match_df, "knf_stooq_matches.csv")

    save(
        match_df[match_df["match_tier"] == "none"],
        "knf_stooq_unmatched.csv",
    )

    save(
        match_df[
            (match_df["match_tier"] == "fuzzy") &
            (match_df["match_score"] < 90)
        ].sort_values("match_score"),
        "knf_stooq_review.csv",
    )

    log.info("KNF-stooq name matching — complete")
