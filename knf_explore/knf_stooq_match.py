# knf_stooq_match.py
#
# Matches KNF subfund names against stooq fund titles, constrained by TFI
# (fund management company) to prevent cross-manager mis-matches such as
# "Goldman Sachs Emerytura 2025" matching "PKO Emerytura 2025".
#
# Strategy
# --------
# TFI identification — both sides derived from actual data, no hardcoded table:
#
#   KNF side:
#     knf_summary.csv contains `tfi_name` (official KNF company name, e.g.
#     "Goldman Sachs TFI S.A.").  This is stripped of structural words
#     (TFI, S.A., Towarzystwo Funduszy Inwestycyjnych) to yield a short brand
#     token, e.g. "goldman sachs".  That token becomes `knf_tfi_key`.
#
#   Stooq side:
#     Stooq titles follow: "{brand} {subfund name} - notowania...".
#     We scan each title for ALL known knf_tfi_keys (longest first) to assign
#     `stooq_tfi_key`.  This is data-driven: if a new TFI appears in KNF it
#     is automatically picked up without any code change.
#
# Matching passes (per KNF subfund)
# ----------------------------------
#   1. Candidate pool = stooq rows where stooq_tfi_key == knf_tfi_key.
#      Falls back to full pool if pool is empty; flagged match_tfi_fallback=True.
#   2. Exact match on normalised residual name (brand + legal tokens stripped).
#   3. Fuzzy token-sort match on residual name (rapidfuzz), score >= threshold.
#   4. No match.
#
# Inputs (from Drive):
#   knf_summary.csv       — must include `tfi_name` column (from knf_explore.py)
#   nazwy1000..5000.csv   — stooq fund titles
#
# Outputs (to Drive):
#   knf_stooq_matches.csv   — full match table
#   knf_stooq_unmatched.csv — tier == "none"
#   knf_stooq_review.csv    — fuzzy score < 90 OR tfi_fallback=True
#
# Required env vars:
#   GOOGLE_CREDENTIALS   — service account JSON (written to /tmp/credentials.json)
#   GDRIVE_FOLDER_ID     — Drive folder containing input and output files
#
# Optional env vars:
#   FUZZY_THRESHOLD      — minimum score (default 75)
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
# Drive helpers
# ---------------------------------------------------------------------------

def get_drive_service():
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_file(
        CREDS_PATH, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)


def download_csv_from_drive(service, folder_id, filename, sep=";"):
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    files = service.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if not files:
        log.warning("File not found in Drive: %s", filename)
        return None
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, service.files().get_media(fileId=files[0]["id"]))
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    try:
        df = pd.read_csv(buf, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        buf.seek(0)
        df = pd.read_csv(buf, sep=sep, encoding="cp1250")
    log.info("Downloaded %-30s  %d rows", filename, len(df))
    return df


def upload_csv_to_drive(service, folder_id, local_path):
    from googleapiclient.http import MediaFileUpload
    filename = os.path.basename(local_path)
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=q, fields="files(id)").execute().get("files", [])
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    if existing:
        service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        log.info("Drive UPDATED : %s", filename)
    else:
        service.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media, fields="id",
        ).execute()
        log.info("Drive CREATED : %s", filename)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Stooq title prefix pattern: "XXXX.n - " where XXXX is the numeric fund ID.
# Always present at the start of every stooq page title.
_STOOQ_PREFIX = re.compile(r"^\d+\.n\s*-\s*", re.IGNORECASE)

# Structural words stripped when deriving a TFI brand token from a KNF
# company name.  After removing these, what remains is the brand core:
#   "Goldman Sachs TFI S.A."          -> "goldman sachs"
#   "NN Investment Partners TFI S.A."  -> "nn"
#   "PKO TFI S.A."                     -> "pko"
_TFI_STRUCTURAL = re.compile(
    r"\b(?:"
    r"towarzystwo funduszy inwestycyjnych"
    r"|funduszy inwestycyjnych"
    r"|investment(?:\s+partners)?"
    r"|asset\s+management"
    r"|tfi"
    r"|s\.?\s*a\.?"
    r"|sp\.?\s*z\.?\s*o\.?\s*o\.?"
    r")\b",
    re.IGNORECASE,
)

# Legal / structural tokens stripped from fund *product* names.
# These appear in both KNF names and stooq names after the brand is removed.
_LEGAL = re.compile(
    r"\b(?:"
    r"fundusz inwestycyjny otwarty"
    r"|fundusz inwestycyjny zamkniety"
    r"|specjalistyczny fundusz inwestycyjny otwarty"
    r"|fundusz inwestycyjny"
    r"|sfio|fio|fiz"
    r"|subfundusz"
    r"|parasolowy|parasol"
    r"|towarzystwo funduszy inwestycyjnych"
    r"|tfi"
    r")\b",
    re.IGNORECASE,
)


def to_ascii(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


_STOOQ_SUFFIX = re.compile(r"\s*-\s*stooq\s*$", re.IGNORECASE)


def strip_stooq_prefix(title: str, stooq_id) -> str:
    """
    Remove the leading "XXXX.N - " prefix and the trailing " - Stooq" suffix
    from a stooq page title, leaving only the bare fund name.

    Example:
      "2714.N - Goldman Sachs SFIO Spolek Dywidendowych - Stooq"
      -> "Goldman Sachs SFIO Spolek Dywidendowych"

    The leading prefix is matched case-insensitively using the known stooq_id.
    Falls back to the regex pattern if the id is missing or mismatched.
    """
    if not isinstance(title, str):
        return ""
    # Strip leading "XXXX.N - "
    name = ""
    if stooq_id is not None:
        try:
            exact_prefix = f"{int(stooq_id)}.n - "
            if title.lower().startswith(exact_prefix):
                name = title[len(exact_prefix):]
        except (ValueError, TypeError):
            pass
    if not name:
        name = _STOOQ_PREFIX.sub("", title)
    # Strip trailing " - Stooq"
    name = _STOOQ_SUFFIX.sub("", name)
    return name.strip()


# ---------------------------------------------------------------------------
# Manual overrides for TFI brand tokens.
#
# Used when the general stripping logic cannot derive the correct stooq brand
# token from the KNF company name — typically abbreviations or name changes.
#
# Key:   result of to_ascii(company_name.lower())  (pre-stripping, punctuation intact)
# Value: the exact brand token as it appears in stooq titles (lowercased, ASCII)
#
# To add a new override: run the script once, check the "TFI tokens in KNF but
# absent from stooq" log line, then add the mapping here.
# ---------------------------------------------------------------------------

TFI_BRAND_OVERRIDES: dict[str, str] = {
    # Slash/abbreviation in name — general stripping cannot derive correct token
    "vig/c-quadrat tfi s.a.": "vig cq",
    # TFI name starts with "TFI" — stripping leaves "allianz polska", stooq uses "allianz"
    "tfi allianz polska s.a.": "allianz",
    # stooq uses abbreviated brand name
    # KNF may return "AgioFunds" (one word) or "AGIO Funds" (two words)
    "agiofunds tfi s.a.": "agio",
    "agio funds tfi s.a.": "agio",
    # stooq drops "polska" from the brand
    "amundi polska tfi s.a.": "amundi",
    # stooq drops "investments" from the brand
    "generali investments tfi s.a.": "generali",
    # stooq uses singular "investor" not plural "investors"
    "investors tfi s.a.": "investor",
    # Add further overrides here when stooq uses a different brand name than KNF.
    # Key:   to_ascii(tfi_name.strip().lower())
    # Value: brand token as it appears in stooq titles (lowercased, ASCII)
    # Check stooq titles for the TFI in question before adding — do not guess.
}


def tfi_brand_token(company_name: str) -> str:
    """
    Derive a short brand token from a KNF company name.

    Checks TFI_BRAND_OVERRIDES first (for cases where the general stripping
    logic cannot produce the correct stooq brand token, e.g. abbreviations).
    Otherwise: lowercase → fold diacritics → strip structural words → collapse whitespace.
    """
    if not isinstance(company_name, str) or not company_name.strip():
        return ""
    # Check override table using the raw lowercased ASCII name as key
    raw_key = to_ascii(company_name.strip().lower())
    if raw_key in TFI_BRAND_OVERRIDES:
        return TFI_BRAND_OVERRIDES[raw_key]
    s = raw_key
    s = _TFI_STRUCTURAL.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def residual_name(name_without_prefix: str, brand_token: str = "") -> str:
    """
    Produce a normalised residual product name from a fund name that has
    already had its stooq prefix stripped.

    Strips:
      - Legal / structural tokens (FIO, SFIO, FIZ, Subfundusz, ...)
      - The TFI brand token (e.g. "goldman sachs")
    Leaves only the product-identifying words, lowercased and ASCII-folded.

    Both KNF subfund names and stooq names (post-prefix-strip) go through
    this function so the two sides are normalised identically.
    """
    if not isinstance(name_without_prefix, str):
        return ""
    s = to_ascii(name_without_prefix.lower())
    s = _LEGAL.sub(" ", s)
    if brand_token:
        s = re.sub(r"\b" + re.escape(brand_token) + r"\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# TFIs whose funds are on KNF but not on stooq — excluded from matching.
# Key: to_ascii(tfi_name.strip().lower())
# ---------------------------------------------------------------------------

TFI_NO_STOOQ: set[str] = {
    "pfr tfi s.a.",
    "mtfi s.a.",
    "velofunds tfi s.a.",
}

# ---------------------------------------------------------------------------
# KNF categories excluded from matching — funds in these categories are not
# on stooq regardless of TFI (e.g. PPK workplace pension products).
# Values match the "category" column in knf_summary.csv (substring match).
# ---------------------------------------------------------------------------

CATEGORY_NO_STOOQ_PREFIXES: tuple[str, ...] = (
    "PPK",
)


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

def load_knf_summary(service) -> pd.DataFrame:
    df = download_csv_from_drive(service, GDRIVE_FOLDER, KNF_SUMMARY_FILE)
    if df is None or df.empty:
        raise RuntimeError("knf_summary.csv not found or empty on Drive.")

    if "tfi_name" not in df.columns:
        raise RuntimeError(
            "knf_summary.csv has no 'tfi_name' column. "
            "Re-run knf_explore.py to regenerate it with TFI data."
        )

    # Exclude TFIs whose data is not on stooq
    tfi_key_raw = df["tfi_name"].apply(
        lambda n: to_ascii(n.strip().lower()) if isinstance(n, str) else ""
    )
    excluded_mask = tfi_key_raw.isin(TFI_NO_STOOQ)
    n_excluded = excluded_mask.sum()
    if n_excluded:
        log.info("Excluding %d subfunds from TFIs not on stooq: %s",
                 n_excluded,
                 ", ".join(df.loc[excluded_mask, "tfi_name"].unique()))
        df = df[~excluded_mask].copy()

    # Exclude categories not on stooq (PPK workplace pension products etc.)
    if "category" in df.columns:
        cat_mask = df["category"].apply(
            lambda c: isinstance(c, str) and c.startswith(CATEGORY_NO_STOOQ_PREFIXES)
        )
        n_cat = cat_mask.sum()
        if n_cat:
            log.info("Excluding %d subfunds in categories not on stooq: %s",
                     n_cat,
                     ", ".join(sorted(df.loc[cat_mask, "category"].unique())))
            df = df[~cat_mask].copy()

    # Derive brand token from the official KNF company name
    df["knf_tfi_key"] = df["tfi_name"].apply(tfi_brand_token)
    # Residual product name (brand + legal stripped)
    df["knf_norm"] = df.apply(
        lambda r: residual_name(str(r["name"]), r["knf_tfi_key"]), axis=1
    )

    log.info("KNF summary loaded: %d records (after TFI exclusions)", len(df))
    log.info("  Distinct TFI brand tokens: %d", df["knf_tfi_key"].nunique())
    log.info("  Sample TFI tokens: %s",
             ", ".join(sorted(df["knf_tfi_key"].dropna().unique())[:15]))
    return df


def load_stooq_names(service, knf_tfi_keys: set[str]) -> pd.DataFrame:
    """
    Load and annotate stooq fund titles.

    Each title looks like:  "2710.n - Goldman Sachs Parasol FIO Akcji"
    Processing:
      1. Strip the "XXXX.n - " prefix (using the known stooq_id for an exact match).
      2. Detect the TFI brand token in the stripped name.
      3. Compute the residual product name (brand + legal tokens removed).

    knf_tfi_keys — set of brand tokens derived from KNF company names.
    Passed in so TFI detection is driven by the same data as the KNF side.
    """
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
    stooq = stooq[stooq["stooq_title"].notna() & (stooq["stooq_title"].str.strip() != "")].copy()
    stooq["stooq_title"] = stooq["stooq_title"].str.strip()

    # Step 1: strip "XXXX.n - " prefix using the ID from the same row
    stooq["stooq_name"] = stooq.apply(
        lambda r: strip_stooq_prefix(r["stooq_title"], r["stooq_id"]), axis=1
    )

    # Data quality check: flag rows where prefix strip relied on fallback regex
    # (means the title's numeric prefix didn't match the stooq_id column)
    def prefix_matched(row) -> bool:
        try:
            return row["stooq_title"].lower().startswith(f"{int(row['stooq_id'])}.n")
        except (ValueError, TypeError):
            return False

    mismatch_mask = ~stooq.apply(prefix_matched, axis=1)
    n_mismatch = mismatch_mask.sum()
    if n_mismatch:
        log.warning(
            "  %d stooq rows: title prefix does not match stooq_id — "
            "used regex fallback for these.  Sample:",
            n_mismatch,
        )
        for _, r in stooq[mismatch_mask].head(5).iterrows():
            log.warning("    id=%s  title=%s", r["stooq_id"], r["stooq_title"][:80])

    # Step 2: detect TFI brand token in the stripped name (longest token first)
    sorted_tokens = sorted(knf_tfi_keys - {""}, key=len, reverse=True)

    def detect_tfi(name: str) -> str:
        s = to_ascii(name.lower())
        for token in sorted_tokens:
            if token and token in s:
                return token
        return ""

    stooq["stooq_tfi_key"] = stooq["stooq_name"].apply(detect_tfi)

    # Step 3: residual product name (brand + legal tokens stripped from name)
    stooq["stooq_norm"] = stooq.apply(
        lambda r: residual_name(r["stooq_name"], r["stooq_tfi_key"]), axis=1
    )

    log.info("Stooq names loaded: %d records", len(stooq))
    tfi_cov = (stooq["stooq_tfi_key"] != "").sum()
    log.info("  Stooq rows with recognised TFI token: %d / %d (%.0f%%)",
             tfi_cov, len(stooq), 100 * tfi_cov / max(len(stooq), 1))

    found_in_stooq = set(stooq["stooq_tfi_key"].unique()) - {""}
    missing = sorted(knf_tfi_keys - {""} - found_in_stooq)
    if missing:
        log.info("  TFI tokens in KNF but absent from stooq (%d): %s",
                 len(missing), ", ".join(missing[:20]))

    return stooq.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_funds(knf_df: pd.DataFrame, stooq_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each KNF subfund:
      - Restrict stooq pool to rows with matching stooq_tfi_key.
        Falls back to full pool if empty; flagged match_tfi_fallback=True.
      - Pass 1: exact match on normalised residual name.
      - Pass 2: fuzzy token-sort (score >= FUZZY_THRESHOLD).
      - Else: tier = "none".
    """
    tfi_groups: dict[str, pd.DataFrame] = {}
    for key, grp in stooq_df.groupby("stooq_tfi_key"):
        tfi_groups[str(key)] = grp.reset_index(drop=True)

    results = []

    for _, knf_row in knf_df.iterrows():
        knf_norm    = knf_row["knf_norm"]
        knf_tfi_key = str(knf_row.get("knf_tfi_key", ""))

        result = {
            "subfundId":                  knf_row.get("subfundId"),
            "name":                       knf_row.get("name"),
            "tfi_name":                   knf_row.get("tfi_name"),
            "knf_tfi_key":                knf_tfi_key,
            "classification":             knf_row.get("classification"),
            "category":                   knf_row.get("category"),
            "fromDate":                   knf_row.get("fromDate"),
            "riskLevel":                  knf_row.get("riskLevel"),
            "managementAndOtherFeesRate": knf_row.get("managementAndOtherFeesRate"),
            "maxEntryFeeRate":            knf_row.get("maxEntryFeeRate"),
            "hasBenchmark":               knf_row.get("hasBenchmark"),
            "benchmark":                  knf_row.get("benchmark"),
            "knf_norm":                   knf_norm,
            "match_tier":                 "none",
            "match_score":                0,
            "match_tfi_fallback":         False,
            "stooq_id":                   None,
            "stooq_title":                None,
            "stooq_name":                 None,
            "stooq_tfi_key_matched":      None,
            "stooq_norm_matched":         None,
        }

        if not knf_norm:
            results.append(result)
            continue

        # Select candidate pool
        pool = tfi_groups.get(knf_tfi_key) if knf_tfi_key else None
        if pool is None or pool.empty:
            pool = stooq_df
            result["match_tfi_fallback"] = True

        pool_norms = pool["stooq_norm"].tolist()

        # Pass 1: exact
        exact = pool[pool["stooq_norm"] == knf_norm]
        if not exact.empty:
            best = exact.iloc[0]
            result.update({
                "match_tier":            "exact",
                "match_score":           100,
                "stooq_id":              best["stooq_id"],
                "stooq_title":           best["stooq_title"],
                "stooq_name":            best["stooq_name"],
                "stooq_tfi_key_matched": best["stooq_tfi_key"],
                "stooq_norm_matched":    best["stooq_norm"],
            })
            results.append(result)
            continue

        # Pass 2: fuzzy
        best_match = process.extractOne(
            knf_norm, pool_norms,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if best_match is not None:
            matched_norm, score, idx = best_match
            row = pool.iloc[idx]
            result.update({
                "match_tier":            "fuzzy",
                "match_score":           round(score, 1),
                "stooq_id":              row["stooq_id"],
                "stooq_title":           row["stooq_title"],
                "stooq_name":            row["stooq_name"],
                "stooq_tfi_key_matched": row["stooq_tfi_key"],
                "stooq_norm_matched":    matched_norm,
            })

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_matches(match_df: pd.DataFrame):
    total    = len(match_df)
    exact    = (match_df["match_tier"] == "exact").sum()
    fuzzy    = (match_df["match_tier"] == "fuzzy").sum()
    none_    = (match_df["match_tier"] == "none").sum()
    fallback = match_df["match_tfi_fallback"].sum()
    review   = ((match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)).sum()

    log.info("\n--- Match summary ---")
    log.info("  Total KNF subfunds       : %d", total)
    log.info("  Exact matches            : %d  (%.0f%%)", exact,  100 * exact  / total)
    log.info("  Fuzzy matches            : %d  (%.0f%%)", fuzzy,  100 * fuzzy  / total)
    log.info("    of which score < 90    : %d  (needs review)", review)
    log.info("  No match                 : %d  (%.0f%%)", none_,  100 * none_  / total)
    log.info("  TFI pool fallback        : %d", fallback)

    log.info("\n--- Match rate by category ---")
    cat = (
        match_df.groupby("category")
                .apply(lambda g: pd.Series({
                    "total":   len(g),
                    "matched": (g["match_tier"] != "none").sum(),
                    "exact":   (g["match_tier"] == "exact").sum(),
                }), include_groups=False)
                .reset_index()
    )
    cat["pct"] = (100 * cat["matched"] / cat["total"]).round(1)
    for _, r in cat.sort_values("pct", ascending=False).iterrows():
        log.info("  %-45s  %d/%d (%.0f%%)  exact=%d",
                 r["category"], r["matched"], r["total"], r["pct"], r["exact"])

    review_df = match_df[
        (match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)
    ].sort_values("match_score")
    if not review_df.empty:
        log.info("\n--- Fuzzy score < 90 (worst first, up to 20) ---")
        for _, r in review_df.head(20).iterrows():
            log.info("  score=%5.1f  fallback=%-5s  KNF: %-45s  stooq: %s",
                     r["match_score"], str(r["match_tfi_fallback"]),
                     str(r["name"])[:45], r["stooq_title"])

    unmatched = match_df[match_df["match_tier"] == "none"]
    if not unmatched.empty:
        log.info("\n--- Unmatched (up to 20) ---")
        for _, r in unmatched.head(20).iterrows():
            log.info("  tfi=%-30s  cat=%-35s  name=%s",
                     str(r.get("tfi_name", ""))[:30],
                     str(r["category"])[:35], r["name"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("KNF-stooq name matching (TFI-constrained, API-derived) — start")
    log.info("GDRIVE_FOLDER   : %s", GDRIVE_FOLDER)
    log.info("FUZZY_THRESHOLD : %d", FUZZY_THRESHOLD)

    service = get_drive_service()
    knf_df  = load_knf_summary(service)

    knf_tfi_keys = set(knf_df["knf_tfi_key"].dropna().unique())
    stooq_df = load_stooq_names(service, knf_tfi_keys)

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

    review_mask = (
        ((match_df["match_tier"] == "fuzzy") & (match_df["match_score"] < 90)) |
        (match_df["match_tfi_fallback"] == True)
    )
    save(match_df[review_mask].sort_values("match_score"), "knf_stooq_review.csv")

    log.info("KNF-stooq name matching — complete")