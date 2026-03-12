# knf_stooq_match.py
#
# Matches KNF subfund names against stooq fund titles, constrained by TFI
# (fund management company) to prevent cross-manager mis-matches.
#
# Strategy
# --------
# KNF side:
#   - knf_summary.csv must contain a `tfi_name` column (added by knf_explore.py).
#   - `tfi_name` is normalised → `knf_tfi_norm`.
#
# Stooq side:
#   - Stooq titles follow the pattern:
#       "{TFI brand(s)} {subfund name} - notowania, wykresy..."
#   - A TFI_BRANDS table maps normalised KNF TFI names to the brand token(s)
#     that appear at the start of stooq titles for that manager.
#   - `stooq_tfi_norm` is extracted by scanning for known brand tokens BEFORE
#     the name-stripping step so we can partition on it.
#
# Matching passes (per KNF subfund)
# ----------------------------------
#   1. Build candidate pool: stooq rows whose stooq_tfi_norm matches knf_tfi_norm.
#      If the pool is empty (TFI not in stooq or not in alias table), fall back
#      to the full stooq pool and flag match_tfi_fallback=True.
#   2. Exact match on normalised residual name (name with TFI brand + legal tokens
#      stripped).
#   3. Fuzzy token-sort match on residual name (rapidfuzz), score >= threshold.
#   4. No match.
#
# Outputs (uploaded to Drive):
#   knf_stooq_matches.csv    — full match table, one row per KNF subfund
#   knf_stooq_unmatched.csv  — tier == "none"
#   knf_stooq_review.csv     — fuzzy score < 90 OR tfi_fallback=True
#
# Required env vars:
#   GOOGLE_CREDENTIALS   — service account JSON (written to /tmp/credentials.json)
#   GDRIVE_FOLDER_ID     — Drive folder containing input and output files
#
# Optional env vars:
#   FUZZY_THRESHOLD      — minimum score to accept a fuzzy match (default 75)
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

GDRIVE_FOLDER   = os.getenv("GDRIVE_FOLDER_ID", "")
CREDS_PATH      = "/tmp/credentials.json"
OUTPUT_DIR      = "/tmp/knf_match"
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "75"))

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
# TFI brand alias table
#
# Maps the normalised KNF tfi_name → list of brand token strings that appear
# inside stooq fund titles for that manager.
#
# Each alias is already lowercased with diacritics folded to ASCII.
# Multiple aliases handle historical renames or abbreviations.
# Sorted longest-first at runtime so longer tokens match before shorter ones.
# ---------------------------------------------------------------------------

TFI_BRANDS: dict[str, list[str]] = {
    "goldman sachs tfi sa":             ["goldman sachs"],
    "pko tfi sa":                       ["pko"],
    "pzu tfi sa":                       ["pzu"],
    "santander tfi sa":                 ["santander"],
    "bank pekao sa":                    ["pekao"],
    "pekao tfi sa":                     ["pekao"],
    "skarbiec tfi sa":                  ["skarbiec"],
    "nn investment partners tfi sa":    ["nn"],
    "ipopema tfi sa":                   ["ipopema"],
    "allianz polska tfi sa":            ["allianz"],
    "generali investments tfi sa":      ["generali"],
    "axa tfi sa":                       ["axa"],
    "bnp paribas tfi sa":               ["bnp paribas"],
    "mbank tfi sa":                     ["mbank"],
    "esaliens tfi sa":                  ["esaliens"],
    "investors tfi sa":                 ["investors"],
    "quercus tfi sa":                   ["quercus"],
    "union investment tfi sa":          ["union investment"],
    "amundi tfi sa":                    ["amundi"],
    "fidelity international":           ["fidelity"],
    "superfund tfi sa":                 ["superfund"],
    "rockbridge tfi sa":                ["rockbridge"],
    "opera tfi sa":                     ["opera"],
    "caspar asset management tfi sa":   ["caspar"],
    "noble funds tfi sa":               ["noble funds"],
    "altus tfi sa":                     ["altus"],
    "millennium tfi sa":                ["millennium"],
    "metlife tfi sa":                   ["metlife"],
    "aegon tfi sa":                     ["aegon"],
    "aviva investors poland tfi sa":    ["aviva"],
    "uniqa tfi sa":                     ["uniqa"],
    "bph tfi sa":                       ["bph"],
    "legg mason tfi sa":                ["legg mason"],
    "franklin templeton investments":   ["franklin templeton"],
    "blackrock":                        ["blackrock"],
    "vanguard":                         ["vanguard"],
}

# Flat list sorted by brand token length descending (longest match first)
_SORTED_BRANDS: list[tuple[str, str]] = sorted(
    [
        (norm_tfi, brand)
        for norm_tfi, brands in TFI_BRANDS.items()
        for brand in brands
    ],
    key=lambda x: len(x[1]),
    reverse=True,
)


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
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
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
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    media    = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    if existing:
        fid = existing[0]["id"]
        service.files().update(fileId=fid, media_body=media).execute()
        log.info("Drive UPDATED : %s", filename)
    else:
        meta = {"name": filename, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()
        log.info("Drive CREATED : %s", filename)


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

LEGAL_TOKENS = re.compile("|".join([
    r"\bfundusz inwestycyjny otwarty\b",
    r"\bfundusz inwestycyjny zamkniety\b",
    r"\bspecjalistyczny fundusz inwestycyjny otwarty\b",
    r"\bfundusz inwestycyjny\b",
    r"\bsfio\b", r"\bfio\b", r"\bfiz\b",
    r"\bsubfundusz\b",
    r"\btowarzystwo funduszy inwestycyjnych\b",
    r"\btfi\b",
    r"notowania[, ]+wykresy[, ]+dane historyczne[, ]+stooq",
    r"notowania[, ]+stooq",
    r"\bsa\b",           # strip "S.A." / "SA" from TFI names
    r"- stooq",
]), re.IGNORECASE)

# Regex matching any known brand token (for stripping from residual name)
BRAND_PATTERN = re.compile(
    r"\b(?:" + "|".join(
        re.escape(b) for _, b in _SORTED_BRANDS
    ) + r")\b",
    re.IGNORECASE,
)


def to_ascii(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


def normalise_tfi(name: str) -> str:
    """Normalise a KNF TFI company name for partition key lookup."""
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = to_ascii(s)
    # Strip legal suffixes that don't appear in TFI_BRANDS keys
    s = re.sub(r"\s+s\.?\s*a\.?$", "", s)   # trailing "S.A." / "S. A."
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalise_residual(name: str) -> str:
    """
    Normalise a fund name to its residual (product) tokens:
      1. Lowercase + fold diacritics to ASCII
      2. Strip stooq suffix
      3. Strip all known brand tokens (TFI prefixes)
      4. Strip legal / structural tokens
      5. Collapse punctuation and whitespace
    """
    if not isinstance(name, str):
        return ""
    s = name.lower()
    s = to_ascii(s)
    s = LEGAL_TOKENS.sub(" ", s)
    s = BRAND_PATTERN.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_stooq_tfi(raw_title: str) -> str:
    """
    Identify which TFI this stooq title belongs to.
    Scans _SORTED_BRANDS (longest first) in the normalised title.
    Returns the normalised KNF TFI key if found, else "".
    """
    if not isinstance(raw_title, str):
        return ""
    s = to_ascii(raw_title.lower())
    for norm_tfi, brand in _SORTED_BRANDS:
        if brand in s:
            return norm_tfi
    return ""


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

def load_stooq_names(service) -> pd.DataFrame:
    pieces = []
    for fname in NAZWY_FILES:
        df = download_csv_from_drive(service, GDRIVE_FOLDER, fname)
        if df is not None and not df.empty:
            pieces.append(df)

    if not pieces:
        raise RuntimeError("No nazwy files could be downloaded from Drive.")

    stooq = pd.concat(pieces, ignore_index=True)
    stooq = stooq.rename(columns={
        "Title":         "stooq_title",
        "Numer kolejny": "stooq_id",
    })

    stooq = stooq[stooq["stooq_title"].notna()].copy()
    stooq["stooq_title"] = stooq["stooq_title"].str.strip()
    stooq = stooq[stooq["stooq_title"] != ""].copy()

    # Extract TFI partition key BEFORE stripping brands
    stooq["stooq_tfi_norm"] = stooq["stooq_title"].apply(extract_stooq_tfi)
    # Residual product name (brands + legal tokens stripped)
    stooq["stooq_norm"]     = stooq["stooq_title"].apply(normalise_residual)

    log.info("Stooq names loaded: %d records", len(stooq))
    tfi_cov = (stooq["stooq_tfi_norm"] != "").sum()
    log.info("  Stooq rows with recognised TFI: %d / %d (%.0f%%)",
             tfi_cov, len(stooq), 100 * tfi_cov / max(len(stooq), 1))
    return stooq.reset_index(drop=True)


def load_knf_summary(service) -> pd.DataFrame:
    df = download_csv_from_drive(service, GDRIVE_FOLDER, KNF_SUMMARY_FILE)
    if df is None or df.empty:
        raise RuntimeError("knf_summary.csv not found or empty on Drive.")

    if "tfi_name" not in df.columns:
        log.warning(
            "knf_summary.csv has no 'tfi_name' column.  TFI-constrained "
            "matching is disabled.  Re-run knf_explore.py to add this column."
        )
        df["tfi_name"] = None

    df["knf_norm"]     = df["name"].apply(normalise_residual)
    df["knf_tfi_norm"] = df["tfi_name"].apply(normalise_tfi)

    # Map the normalised tfi name to a TFI_BRANDS key by checking membership
    known_keys = set(TFI_BRANDS.keys())
    df["knf_tfi_in_alias"] = df["knf_tfi_norm"].isin(known_keys)

    log.info("KNF summary loaded: %d records", len(df))
    log.info("  KNF subfunds with TFI alias: %d / %d (%.0f%%)",
             df["knf_tfi_in_alias"].sum(), len(df),
             100 * df["knf_tfi_in_alias"].mean())
    return df


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_funds(knf_df: pd.DataFrame, stooq_df: pd.DataFrame) -> pd.DataFrame:
    """
    TFI-constrained matching.

    Per KNF subfund:
      - Candidate pool = stooq rows with matching stooq_tfi_norm.
        Falls back to full stooq pool if TFI partition is empty/unknown;
        in that case match_tfi_fallback=True.
      - Pass 1: exact match on normalised residual name within pool.
      - Pass 2: fuzzy token-sort within pool (score >= FUZZY_THRESHOLD).
      - Else: tier = "none".
    """
    # Pre-build TFI partitions
    tfi_groups: dict[str, pd.DataFrame] = {}
    for tfi_key, grp in stooq_df.groupby("stooq_tfi_norm"):
        tfi_groups[str(tfi_key)] = grp.reset_index(drop=True)

    results = []

    for _, knf_row in knf_df.iterrows():
        knf_norm     = knf_row["knf_norm"]
        knf_tfi_norm = knf_row.get("knf_tfi_norm", "")

        result = {
            "subfundId":                  knf_row.get("subfundId"),
            "name":                       knf_row.get("name"),
            "tfi_name":                   knf_row.get("tfi_name"),
            "classification":             knf_row.get("classification"),
            "category":                   knf_row.get("category"),
            "fromDate":                   knf_row.get("fromDate"),
            "riskLevel":                  knf_row.get("riskLevel"),
            "managementAndOtherFeesRate": knf_row.get("managementAndOtherFeesRate"),
            "maxEntryFeeRate":            knf_row.get("maxEntryFeeRate"),
            "hasBenchmark":               knf_row.get("hasBenchmark"),
            "benchmark":                  knf_row.get("benchmark"),
            "knf_norm":                   knf_norm,
            "knf_tfi_norm":               knf_tfi_norm,
            "match_tier":                 "none",
            "match_score":                0,
            "match_tfi_fallback":         False,
            "stooq_id":                   None,
            "stooq_title":                None,
            "stooq_norm_matched":         None,
            "stooq_tfi_norm_matched":     None,
        }

        if not knf_norm:
            results.append(result)
            continue

        # --- Select candidate pool ---
        pool = tfi_groups.get(str(knf_tfi_norm)) if knf_tfi_norm else None
        if not pool or pool.empty:
            pool = stooq_df
            result["match_tfi_fallback"] = True

        pool_norms = pool["stooq_norm"].tolist()

        # --- Pass 1: exact residual match ---
        exact = pool[pool["stooq_norm"] == knf_norm]
        if not exact.empty:
            best = exact.iloc[0]
            result.update({
                "match_tier":             "exact",
                "match_score":            100,
                "stooq_id":               best["stooq_id"],
                "stooq_title":            best["stooq_title"],
                "stooq_norm_matched":     best["stooq_norm"],
                "stooq_tfi_norm_matched": best["stooq_tfi_norm"],
            })
            results.append(result)
            continue

        # --- Pass 2: fuzzy token-sort within pool ---
        best_match = process.extractOne(
            knf_norm,
            pool_norms,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )

        if best_match is not None:
            matched_norm, score, idx = best_match
            stooq_row = pool.iloc[idx]
            result.update({
                "match_tier":             "fuzzy",
                "match_score":            round(score, 1),
                "stooq_id":               stooq_row["stooq_id"],
                "stooq_title":            stooq_row["stooq_title"],
                "stooq_norm_matched":     matched_norm,
                "stooq_tfi_norm_matched": stooq_row["stooq_tfi_norm"],
            })

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyse_matches(match_df: pd.DataFrame):
    total    = len(match_df)
    exact    = (match_df["match_tier"] == "exact").sum()
    fuzzy    = (match_df["match_tier"] == "fuzzy").sum()
    none_    = (match_df["match_tier"] == "none").sum()
    fallback = match_df.get("match_tfi_fallback", pd.Series(dtype=bool)).sum()
    review   = (
        (match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)
    ).sum()

    log.info("\n--- Match summary ---")
    log.info("  Total KNF subfunds       : %d", total)
    log.info("  Exact matches            : %d  (%.0f%%)", exact,  100 * exact  / total)
    log.info("  Fuzzy matches            : %d  (%.0f%%)", fuzzy,  100 * fuzzy  / total)
    log.info("    of which score < 90    : %d  (needs review)", review)
    log.info("  No match                 : %d  (%.0f%%)", none_,  100 * none_  / total)
    log.info("  TFI fallback (full pool) : %d", fallback)

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

    review_df = match_df[
        (match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)
    ].sort_values("match_score")

    if not review_df.empty:
        log.info("\n--- Fuzzy matches score < 90 (worst first, up to 20) ---")
        for _, r in review_df.head(20).iterrows():
            log.info("  score=%5.1f  fallback=%-5s  KNF: %-45s  stooq: %s",
                     r["match_score"],
                     str(r.get("match_tfi_fallback", "")),
                     str(r["name"])[:45],
                     r["stooq_title"])

    unmatched = match_df[match_df["match_tier"] == "none"]
    if not unmatched.empty:
        log.info("\n--- Unmatched KNF subfunds (up to 20) ---")
        for _, r in unmatched.head(20).iterrows():
            log.info("  tfi=%-30s  category=%-35s  name=%s",
                     str(r.get("tfi_name", ""))[:30],
                     str(r["category"])[:35],
                     r["name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("KNF-stooq name matching (TFI-constrained) — start")
    log.info("GDRIVE_FOLDER   : %s", GDRIVE_FOLDER)
    log.info("FUZZY_THRESHOLD : %d", FUZZY_THRESHOLD)

    service  = get_drive_service()
    stooq_df = load_stooq_names(service)
    knf_df   = load_knf_summary(service)

    log.info("\nRunning matching (%d KNF × %d stooq)...", len(knf_df), len(stooq_df))
    match_df = match_funds(knf_df, stooq_df)

    analyse_matches(match_df)

    def save(df, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s  (%d rows)", filename, len(df))
        upload_csv_to_drive(service, GDRIVE_FOLDER, path)

    save(match_df, "knf_stooq_matches.csv")
    save(match_df[match_df["match_tier"] == "none"], "knf_stooq_unmatched.csv")

    # Review file: low-confidence fuzzy OR any TFI-fallback match
    review_mask = (
        ((match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)) |
        (match_df.get("match_tfi_fallback", pd.Series(False, index=match_df.index)) == True)
    )
    save(match_df[review_mask].sort_values("match_score"), "knf_stooq_review.csv")

    log.info("KNF-stooq name matching — complete")