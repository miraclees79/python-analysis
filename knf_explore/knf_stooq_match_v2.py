# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:48:24 2026

@author: adamg
"""

# knf_stooq_match_v2.py
#
# Matches KNF subfund names against stooq fund titles, constrained by TFI
# (fund management company) to prevent cross-manager mis-matches such as
# "Goldman Sachs Emerytura 2025" matching "PKO Emerytura 2025".
#
# STOOQ DATA SOURCE
# -----------------
# Fund names are loaded from .txt files located in the fund_data/ subfolder
# of the current working directory.  Each file follows the stooq bulk-export
# format:
#
#   <TICKER>        <NAME>
#   1122.N          PZU ZRÓWNOWAŻONY
#
# The ticker and name are separated by one or more whitespace characters;
# the ticker always ends with ".N" (case-insensitive).  The stem of each
# filename (without extension) is treated as the stooq fund category, e.g.
# "akcji_polskich.txt" → stooq_category = "akcji_polskich".
#
# Both the KNF category (from knf_summary.csv) and the stooq category
# (derived from the source filename) are written to every output row.
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
#     Stooq titles follow: "{brand} {subfund name}".
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
#   4. History pass (only for unmatched):
#      Fetches /v1/subfunds/{id}/history from the KNF API, extracts all former
#      names (validTo != "9999-12-31"), normalises each, and retries exact then
#      fuzzy against the same TFI-constrained pool.  Recorded as tier
#      "historical_exact" / "historical_fuzzy"; knf_former_name_matched is set.
#   5. No match.
#
# Inputs:
#   knf_summary.csv             — from Google Drive (must include `tfi_name`)
#   fund_data/*.txt             — stooq fund name files (local, see format above)
#
# Outputs (to Google Drive):
#   knf_stooq_matches.csv       — full match table
#   knf_stooq_unmatched.csv     — tier == "none"
#   knf_stooq_review.csv        — fuzzy score < 90 OR tfi_fallback=True
#
# Required env vars:
#   GOOGLE_CREDENTIALS   — service account JSON (written to /tmp/credentials.json)
#   GDRIVE_FOLDER_ID     — Drive folder containing input and output files
#
# Optional env vars:
#   FUZZY_THRESHOLD      — minimum fuzzy score (default 75)
#   FUND_DATA_DIR        — path to folder containing .txt files
#                          (default: ./fund_data relative to this script)
#
# Dependencies:
#   pip install pandas rapidfuzz google-auth google-api-python-client

import os
import io
import re
import time
import logging
import unicodedata
import urllib.request
import urllib.error
import json
from pathlib import Path
import tempfile
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
CREDS_PATH      = os.path.join(tempfile.gettempdir(), "credentials.json")
OUTPUT_DIR      = "/tmp/knf_match"

# Directory containing stooq .txt files.
# Defaults to fund_data/ next to this script.
_SCRIPT_DIR = Path(__file__).resolve().parent
FUND_DATA_DIR = Path(os.getenv("FUND_DATA_DIR", str(_SCRIPT_DIR / "fund_data")))

FUZZY_THRESHOLD  = int(os.getenv("FUZZY_THRESHOLD", "75"))

KNF_API_BASE     = "https://wybieramfundusze-api.knf.gov.pl"
KNF_API_RETRIES  = 3
KNF_API_BACKOFF  = 2.0   # seconds, doubled on each retry

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
# Stooq .txt file loader
# ---------------------------------------------------------------------------

def load_stooq_txt_files(fund_data_dir: Path) -> pd.DataFrame:
    """
    Load all .txt files from fund_data_dir.

    Each file is expected to follow the stooq bulk-export format:

        <TICKER>        <NAME>
        1122.N          PZU ZRÓWNOWAŻONY

    The ticker ends with ".N" (case-insensitive) and is separated from
    the fund name by one or more whitespace characters.  Lines that do
    not match this pattern (e.g. header lines or blank lines) are skipped.

    The filename stem (without extension) becomes the stooq_category for
    every row sourced from that file.

    Returns a DataFrame with columns:
        stooq_id        — numeric part of ticker (int)
        stooq_ticker    — full ticker string, e.g. "1122.N"
        stooq_title     — raw fund name as it appears in the file
        stooq_category  — filename stem (e.g. "akcji_polskich")
    """
    if not fund_data_dir.exists():
        raise FileNotFoundError(
            f"fund_data directory not found: {fund_data_dir}\n"
            "Create the directory and populate it with stooq .txt export files."
        )

    txt_files = sorted(fund_data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {fund_data_dir}. "
            "Populate the directory with stooq fund name export files."
        )

    # Pattern: optional whitespace, digits, ".N" (case-insensitive),
    # one or more whitespace chars, then the fund name (rest of line).
    _LINE_RE = re.compile(
        r"^\s*(\d+)\.n\s+(.+)$",
        re.IGNORECASE,
    )

    pieces = []
    for txt_path in txt_files:
        category = txt_path.stem          # filename without extension
        rows = []
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            log.warning("Could not read %s: %s", txt_path.name, exc)
            continue

        n_skipped = 0
        for line in text.splitlines():
            m = _LINE_RE.match(line)
            if not m:
                n_skipped += 1
                continue
            numeric_id = int(m.group(1))
            name       = m.group(2).strip()
            ticker     = f"{numeric_id}.N"
            rows.append({
                "stooq_id":       numeric_id,
                "stooq_ticker":   ticker,
                "stooq_title":    name,
                "stooq_category": category,
            })

        log.info(
            "Loaded %-40s  %4d funds  (%d lines skipped)",
            txt_path.name, len(rows), n_skipped,
        )
        if rows:
            pieces.append(pd.DataFrame(rows))

    if not pieces:
        raise RuntimeError(
            "No valid fund entries were parsed from the .txt files. "
            "Check that the files follow the '<TICKER>  <NAME>' format."
        )

    df = pd.concat(pieces, ignore_index=True)

    # Deduplicate — same ticker may appear in multiple category files.
    # Keep first occurrence (preserves the category of the first file encountered).
    n_before = len(df)
    df = df.drop_duplicates(subset=["stooq_id"], keep="first")
    n_dupes  = n_before - len(df)
    if n_dupes:
        log.info(
            "Deduplicated %d duplicate stooq IDs across files "
            "(kept first occurrence with its category).",
            n_dupes,
        )

    log.info(
        "Stooq .txt files loaded: %d unique funds from %d files  "
        "(%d categories: %s)",
        len(df),
        len(txt_files),
        df["stooq_category"].nunique(),
        ", ".join(sorted(df["stooq_category"].unique())),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def to_ascii(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")


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

# ---------------------------------------------------------------------------
# Manual overrides for TFI brand tokens
# ---------------------------------------------------------------------------

TFI_BRAND_OVERRIDES: dict[str, str] = {
    "vig/c-quadrat tfi s.a.":        "vig cq",
    "tfi allianz polska s.a.":       "allianz",
    "agiofunds tfi s.a.":            "agio",
    "agio funds tfi s.a.":           "agio",
    "amundi polska tfi s.a.":        "amundi",
    "generali investments tfi s.a.": "generali",
    "investors tfi s.a.":            "investor",
}


def tfi_brand_token(company_name: str) -> str:
    if not isinstance(company_name, str) or not company_name.strip():
        return ""
    raw_key = to_ascii(company_name.strip().lower())
    if raw_key in TFI_BRAND_OVERRIDES:
        return TFI_BRAND_OVERRIDES[raw_key]
    s = raw_key
    s = _TFI_STRUCTURAL.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def residual_name(name: str, brand_token: str = "") -> str:
    if not isinstance(name, str):
        return ""
    s = to_ascii(name.lower())
    s = _LEGAL.sub(" ", s)
    if brand_token:
        s = re.sub(r"\b" + re.escape(brand_token) + r"\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# In-scope TFIs
# ---------------------------------------------------------------------------

TFI_IN_SCOPE: set[str] = {
    "tfi allianz polska s.a.",
    "generali investments tfi s.a.",
    "goldman sachs tfi s.a.",
    "investors tfi s.a.",
    "pko tfi s.a.",
    "tfi pzu sa",
    "quercus tfi s.a.",
    "skarbiec tfi s.a.",
    "uniqa tfi s.a.",
    "vig/c-quadrat tfi s.a.",
}

CATEGORY_NO_STOOQ_PREFIXES: tuple[str, ...] = (
    "PPK",
)

# ---------------------------------------------------------------------------
# Load and prepare KNF summary
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

    tfi_key_raw = df["tfi_name"].apply(
        lambda n: to_ascii(n.strip().lower()) if isinstance(n, str) else ""
    )

    scope_mask = tfi_key_raw.isin(TFI_IN_SCOPE)
    n_out = (~scope_mask).sum()
    log.info(
        "Scope filter: keeping %d subfunds from %d in-scope TFIs, "
        "dropping %d from other TFIs",
        scope_mask.sum(),
        df.loc[scope_mask, "tfi_name"].nunique(),
        n_out,
    )
    df = df[scope_mask].copy()

    found_keys = set(tfi_key_raw[scope_mask].unique())
    missing_keys = TFI_IN_SCOPE - found_keys
    if missing_keys:
        log.warning(
            "TFI_IN_SCOPE keys with no matching rows in knf_summary.csv "
            "(possible spelling mismatch): %s",
            ", ".join(sorted(missing_keys)),
        )

    if "category" in df.columns:
        cat_mask = df["category"].apply(
            lambda c: isinstance(c, str) and c.startswith(CATEGORY_NO_STOOQ_PREFIXES)
        )
        n_cat = cat_mask.sum()
        if n_cat:
            log.info(
                "Excluding %d subfunds in categories not on stooq: %s",
                n_cat,
                ", ".join(sorted(df.loc[cat_mask, "category"].unique())),
            )
            df = df[~cat_mask].copy()

    df["knf_tfi_key"] = df["tfi_name"].apply(tfi_brand_token)
    df["knf_norm"] = df.apply(
        lambda r: residual_name(str(r["name"]), r["knf_tfi_key"]), axis=1
    )

    log.info("KNF summary loaded: %d records (after TFI exclusions)", len(df))
    log.info("  Distinct TFI brand tokens: %d", df["knf_tfi_key"].nunique())
    return df


# ---------------------------------------------------------------------------
# Prepare stooq names DataFrame
# ---------------------------------------------------------------------------

def prepare_stooq_names(raw_stooq: pd.DataFrame, knf_tfi_keys: set[str]) -> pd.DataFrame:
    """
    Annotate the raw stooq DataFrame (produced by load_stooq_txt_files) with:
      - stooq_tfi_key : TFI brand token detected in the fund name
      - stooq_norm    : residual normalised product name

    knf_tfi_keys — set of brand tokens from the KNF side (drives TFI detection).
    """
    stooq = raw_stooq.copy()

    # Detect TFI brand token in the fund name (longest token first to avoid
    # partial matches when one brand contains another as a substring).
    sorted_tokens = sorted(knf_tfi_keys - {""}, key=len, reverse=True)

    def detect_tfi(name: str) -> str:
        s = to_ascii(name.lower())
        for token in sorted_tokens:
            if token and token in s:
                return token
        return ""

    stooq["stooq_tfi_key"] = stooq["stooq_title"].apply(detect_tfi)

    stooq["stooq_norm"] = stooq.apply(
        lambda r: residual_name(r["stooq_title"], r["stooq_tfi_key"]), axis=1
    )

    tfi_cov = (stooq["stooq_tfi_key"] != "").sum()
    log.info(
        "Stooq records annotated: %d total | %d with recognised TFI token (%.0f%%)",
        len(stooq),
        tfi_cov,
        100 * tfi_cov / max(len(stooq), 1),
    )

    found_in_stooq = set(stooq["stooq_tfi_key"].unique()) - {""}
    missing = sorted(knf_tfi_keys - {""} - found_in_stooq)
    if missing:
        log.info(
            "  TFI tokens in KNF but absent from stooq (%d): %s",
            len(missing),
            ", ".join(missing[:20]),
        )

    return stooq.reset_index(drop=True)


# ---------------------------------------------------------------------------
# KNF history fetch
# ---------------------------------------------------------------------------

def fetch_subfund_history(subfund_id: int) -> list[str]:
    """
    Returns a deduplicated list of *former* names for the given subfund,
    i.e. name values from history entries whose validTo is not "9999-12-31".
    Returns [] on any API error.
    """
    url = (
        f"{KNF_API_BASE}/v1/subfunds/{subfund_id}/history"
        "?size=50&sort=validFrom,desc"
    )
    delay = KNF_API_BACKOFF
    for attempt in range(1, KNF_API_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            log.warning(
                "KNF history fetch HTTP %d for subfundId=%s (attempt %d/%d)",
                exc.code, subfund_id, attempt, KNF_API_RETRIES,
            )
        except Exception as exc:
            log.warning(
                "KNF history fetch error for subfundId=%s: %s (attempt %d/%d)",
                subfund_id, exc, attempt, KNF_API_RETRIES,
            )
        if attempt < KNF_API_RETRIES:
            time.sleep(delay)
            delay *= 2
    else:
        return []

    former: list[str] = []
    seen: set[str] = set()
    for entry in payload.get("content", []):
        valid_to = str(entry.get("validTo", ""))
        if valid_to == "9999-12-31":
            continue
        name = entry.get("data", {}).get("name", "")
        if name and name not in seen:
            former.append(name)
            seen.add(name)
    return former


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
      - Pass 3: history — fetch former names from KNF API, retry exact then
                fuzzy for each.  Tier "historical_exact" or "historical_fuzzy".
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
            # KNF fields
            "subfundId":                  knf_row.get("subfundId"),
            "name":                       knf_row.get("name"),
            "tfi_name":                   knf_row.get("tfi_name"),
            "knf_tfi_key":                knf_tfi_key,
            "knf_category":               knf_row.get("category"),
            "classification":             knf_row.get("classification"),
            "fromDate":                   knf_row.get("fromDate"),
            "riskLevel":                  knf_row.get("riskLevel"),
            "managementAndOtherFeesRate": knf_row.get("managementAndOtherFeesRate"),
            "maxEntryFeeRate":            knf_row.get("maxEntryFeeRate"),
            "hasBenchmark":               knf_row.get("hasBenchmark"),
            "benchmark":                  knf_row.get("benchmark"),
            "knf_norm":                   knf_norm,
            # Match metadata
            "match_tier":                 "none",
            "match_score":                0,
            "match_tfi_fallback":         False,
            "knf_former_name_matched":    None,
            # Stooq fields — populated on match
            "stooq_id":                   None,
            "stooq_ticker":               None,
            "stooq_title":                None,
            "stooq_category":             None,   # ← stooq filename category
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

        # ── Pass 1: exact match ──
        exact = pool[pool["stooq_norm"] == knf_norm]
        if not exact.empty:
            best = exact.iloc[0]
            result.update({
                "match_tier":            "exact",
                "match_score":           100,
                "stooq_id":              best["stooq_id"],
                "stooq_ticker":          best["stooq_ticker"],
                "stooq_title":           best["stooq_title"],
                "stooq_category":        best["stooq_category"],
                "stooq_tfi_key_matched": best["stooq_tfi_key"],
                "stooq_norm_matched":    best["stooq_norm"],
            })
            results.append(result)
            continue

        # ── Pass 2: fuzzy match ──
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
                "stooq_ticker":          row["stooq_ticker"],
                "stooq_title":           row["stooq_title"],
                "stooq_category":        row["stooq_category"],
                "stooq_tfi_key_matched": row["stooq_tfi_key"],
                "stooq_norm_matched":    matched_norm,
            })
            results.append(result)
            continue

        # ── Pass 3: KNF history ──
        subfund_id = knf_row.get("subfundId")
        if subfund_id is not None:
            former_names = fetch_subfund_history(int(subfund_id))
            for former_name in former_names:
                former_norm = residual_name(former_name, knf_tfi_key)
                if not former_norm:
                    continue

                # Pass 3a: exact on former name
                exact_h = pool[pool["stooq_norm"] == former_norm]
                if not exact_h.empty:
                    best = exact_h.iloc[0]
                    result.update({
                        "match_tier":              "historical_exact",
                        "match_score":             100,
                        "knf_former_name_matched": former_name,
                        "stooq_id":                best["stooq_id"],
                        "stooq_ticker":            best["stooq_ticker"],
                        "stooq_title":             best["stooq_title"],
                        "stooq_category":          best["stooq_category"],
                        "stooq_tfi_key_matched":   best["stooq_tfi_key"],
                        "stooq_norm_matched":      best["stooq_norm"],
                    })
                    break

                # Pass 3b: fuzzy on former name
                best_h = process.extractOne(
                    former_norm, pool_norms,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=FUZZY_THRESHOLD,
                )
                if best_h is not None:
                    matched_norm, score, idx = best_h
                    row = pool.iloc[idx]
                    result.update({
                        "match_tier":              "historical_fuzzy",
                        "match_score":             round(score, 1),
                        "knf_former_name_matched": former_name,
                        "stooq_id":                row["stooq_id"],
                        "stooq_ticker":            row["stooq_ticker"],
                        "stooq_title":             row["stooq_title"],
                        "stooq_category":          row["stooq_category"],
                        "stooq_tfi_key_matched":   row["stooq_tfi_key"],
                        "stooq_norm_matched":      matched_norm,
                    })
                    break

        results.append(result)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_matches(match_df: pd.DataFrame) -> None:
    total      = len(match_df)
    exact      = (match_df["match_tier"] == "exact").sum()
    fuzzy      = (match_df["match_tier"] == "fuzzy").sum()
    hist_exact = (match_df["match_tier"] == "historical_exact").sum()
    hist_fuzzy = (match_df["match_tier"] == "historical_fuzzy").sum()
    none_      = (match_df["match_tier"] == "none").sum()
    fallback   = match_df["match_tfi_fallback"].sum()
    review     = (
        (match_df["match_tier"].isin(["fuzzy", "historical_fuzzy"])) &
        (match_df["match_score"] < 90)
    ).sum()

    log.info("\n--- Match summary ---")
    log.info("  Total KNF subfunds       : %d", total)
    log.info("  Exact matches            : %d  (%.0f%%)", exact,      100 * exact      / total)
    log.info("  Fuzzy matches            : %d  (%.0f%%)", fuzzy,      100 * fuzzy      / total)
    log.info("  Historical exact         : %d  (%.0f%%)", hist_exact, 100 * hist_exact / total)
    log.info("  Historical fuzzy         : %d  (%.0f%%)", hist_fuzzy, 100 * hist_fuzzy / total)
    log.info("    of which score < 90    : %d  (needs review)", review)
    log.info("  No match                 : %d  (%.0f%%)", none_,      100 * none_      / total)
    log.info("  TFI pool fallback        : %d", fallback)

    log.info("\n--- Match rate by KNF category ---")
    if "knf_category" in match_df.columns:
        cat = (
            match_df.groupby("knf_category")
                    .apply(lambda g: pd.Series({
                        "total":   len(g),
                        "matched": (g["match_tier"] != "none").sum(),
                        "exact":   (g["match_tier"] == "exact").sum(),
                    }), include_groups=False)
                    .reset_index()
        )
        cat["pct"] = (100 * cat["matched"] / cat["total"]).round(1)
        for _, r in cat.sort_values("pct", ascending=False).iterrows():
            log.info(
                "  %-45s  %d/%d (%.0f%%)  exact=%d",
                r["knf_category"], r["matched"], r["total"], r["pct"], r["exact"],
            )

    log.info("\n--- Match rate by stooq category ---")
    matched = match_df[match_df["match_tier"] != "none"]
    if "stooq_category" in matched.columns and not matched.empty:
        sc = matched.groupby("stooq_category").size().sort_values(ascending=False)
        for cat, cnt in sc.items():
            log.info("  %-40s  %d matched funds", cat, cnt)

    review_df = match_df[
        (match_df["match_tier"].isin(["fuzzy", "historical_fuzzy"])) &
        (match_df["match_score"] < 90)
    ].sort_values("match_score")
    if not review_df.empty:
        log.info("\n--- Fuzzy score < 90 (worst first, up to 20) ---")
        for _, r in review_df.head(20).iterrows():
            log.info(
                "  score=%5.1f  fallback=%-5s  KNF: %-45s  stooq: %s",
                r["match_score"], str(r["match_tfi_fallback"]),
                str(r["name"])[:45], r.get("stooq_title", ""),
            )

    unmatched = match_df[match_df["match_tier"] == "none"]
    if not unmatched.empty:
        log.info("\n--- Unmatched (up to 20) ---")
        for _, r in unmatched.head(20).iterrows():
            log.info(
                "  tfi=%-30s  knf_cat=%-35s  name=%s",
                str(r.get("tfi_name", ""))[:30],
                str(r.get("knf_category", ""))[:35],
                r["name"],
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("KNF-stooq name matching (TFI-constrained, local .txt files) — start")
    log.info("GDRIVE_FOLDER   : %s", GDRIVE_FOLDER)
    log.info("FUND_DATA_DIR   : %s", FUND_DATA_DIR)
    log.info("FUZZY_THRESHOLD : %d", FUZZY_THRESHOLD)

    # --- Load stooq data from local .txt files ---
    raw_stooq = load_stooq_txt_files(FUND_DATA_DIR)

    # --- Load KNF summary from Drive ---
    service = get_drive_service()
    knf_df  = load_knf_summary(service)

    # --- Annotate stooq data with TFI tokens and normalised names ---
    knf_tfi_keys = set(knf_df["knf_tfi_key"].dropna().unique())
    stooq_df     = prepare_stooq_names(raw_stooq, knf_tfi_keys)

    # --- Run matching ---
    log.info(
        "\nRunning matching (%d KNF subfunds × %d stooq funds)...",
        len(knf_df), len(stooq_df),
    )
    match_df = match_funds(knf_df, stooq_df)

    analyse_matches(match_df)

    # --- Save and upload outputs ---
    def save(df: pd.DataFrame, filename: str) -> None:
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False, sep=";", encoding="utf-8")
        log.info("Wrote %s  (%d rows)", filename, len(df))
        upload_csv_to_drive(service, GDRIVE_FOLDER, path)

    save(match_df, "knf_stooq_matches.csv")
    save(match_df[match_df["match_tier"] == "none"], "knf_stooq_unmatched.csv")

    review_mask = (
        (match_df["match_tier"].isin(["fuzzy", "historical_fuzzy"])) &
        (match_df["match_score"] < 90)
    ) | (match_df["match_tfi_fallback"] == True)
    save(
        match_df[review_mask].sort_values("match_score"),
        "knf_stooq_review.csv",
    )

    log.info("KNF-stooq name matching — complete")