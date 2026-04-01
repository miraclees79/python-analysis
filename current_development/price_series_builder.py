"""
price_series_builder.py
=======================
Generic builder for combined historical price series stored on Google Drive.

Replaces wsj_msci_world.py and stoxx600.py with a single importable module.
All three use cases share identical pipeline logic:

  MSCI World   : WSJ manual CSV (Date/Open/High/Low/Close) + URTH yfinance extension
  STOXX 600    : manual CSV (Date/Close) + ^STOXX yfinance extension
  TBSP backcast: manual CSV (Date/Close synthetic) + ^TBSP stooq extension
                 (or any other ticker available via yfinance or stooq)

TWO PIPELINE MODES (auto-selected per series)
----------------------------------------------
  First build (combined CSV not yet on Drive):
    raw CSV (Drive) --> parse --> extend with yfinance --> save combined

  Update (combined CSV already on Drive):
    combined CSV (Drive) --> extend with yfinance --> save updated combined

EXTENSION METHOD
----------------
Pure return chain-link — base levels are always preserved exactly.
yfinance (or stooq) levels are never used directly; only daily return
sequences are applied forward from the last base observation.

  anchor_price  = base_series.iloc[-1]
  ext_returns   = extension_ticker.pct_change() for dates > anchor_date
  extension     = anchor_price * (1 + ext_returns).cumprod()
  combined      = concat(base_series, extension)

PUBLIC API
----------
  build_and_upload(
      folder_id,
      raw_filename,
      combined_filename,
      extension_ticker,
      extension_source="yfinance",   # "yfinance" | "stooq"
      raw_csv_format="auto",         # "wsj" | "generic" | "auto"
      credentials_path=CREDENTIALS_PATH,
  ) -> pd.DataFrame | None

  load_combined_from_drive(
      folder_id,
      combined_filename,
      credentials_path=CREDENTIALS_PATH,
  ) -> pd.DataFrame | None

CREDENTIALS
-----------
  Standard service account JSON at os.path.join(tempfile.gettempdir(), "credentials.json")
  Same GDRIVE_FOLDER_ID env var / Drive folder used by all other strategy scripts.

USAGE EXAMPLES
--------------
  # MSCI World (yfinance URTH extension, WSJ raw format)
  from price_series_builder import build_and_upload, load_combined_from_drive

  msci = build_and_upload(
      folder_id          = "your_folder_id",
      raw_filename       = "msci_world_wsj_raw.csv",
      combined_filename  = "msci_world_combined.csv",
      extension_ticker   = "URTH",
  )

  # STOXX 600 (yfinance ^STOXX extension, generic Date/Close format)
  stoxx = build_and_upload(
      folder_id          = "your_folder_id",
      raw_filename       = "stoxx600.csv",
      combined_filename  = "stoxx600_combined.csv",
      extension_ticker   = "^STOXX",
  )

  # TBSP backcast (stooq ^tbsp extension, generic Date/Close format)
  tbsp = build_and_upload(
      folder_id          = "your_folder_id",
      raw_filename       = "tbsp_backcast.csv",
      combined_filename  = "tbsp_extended_combined.csv",
      extension_ticker   = "^tbsp",
      extension_source   = "stooq",
  )

  # Load pre-built combined series (daily runner — no rebuild)
  combined = load_combined_from_drive(
      folder_id         = "your_folder_id",
      combined_filename = "msci_world_combined.csv",
  )
"""

import io
import logging
import os
import sys
import tempfile
import datetime as dt

import numpy as np
import pandas as pd
from pathlib import Path

from strategy_test_library import (
    load_csv)

# Google Drive (required)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False

# yfinance (optional — only required when extension_source="yfinance")
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

# requests (optional — only required when extension_source="stooq")
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ============================================================
# CONSTANTS
# ============================================================

CLOSE_COL        = "Zamkniecie"
DATA_START       = "1990-01-01"
CREDENTIALS_PATH = os.path.join(tempfile.gettempdir(), "credentials.json")

# DATA DIR — used when extension_source="stooq"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def _get_drive_service(credentials_path: str):
    if not _GDRIVE_AVAILABLE:
        raise ImportError(
            "google-api-python-client / google-auth not installed.\n"
            "  pip install google-api-python-client google-auth"
        )
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"credentials.json not found at: {credentials_path}"
        )
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return _gdrive_build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id: str, filename: str) -> str | None:
    q = (
        f"name='{filename}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    results = service.files().list(q=q, fields="files(id,name)").execute()
    files   = results.get("files", [])
    return files[0]["id"] if files else None


def _download_bytes(service, file_id: str) -> bytes:
    request    = service.files().get_media(fileId=file_id)
    buf        = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def _upload_to_drive(
    service,
    folder_id:  str,
    local_path: str,
    filename:   str,
    mimetype:   str = "text/csv",
) -> str:
    existing_id = _find_file_id(service, folder_id, filename)
    media       = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)
    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute()
        logging.info("Drive UPDATED : %s  (id=%s)", filename, existing_id)
        return existing_id
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        result   = service.files().create(
            body=metadata, media_body=media, fields="id"
        ).execute()
        fid = result["id"]
        logging.info("Drive CREATED : %s  (id=%s)", filename, fid)
        return fid


# ============================================================
# RAW CSV PARSERS
# ============================================================

def _parse_wsj_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """
    Parse a WSJ historical-prices CSV.

    Expected format (as of 2025):
        Date,Open,High,Low,Close,Volume
        01/15/25,3741.00,3748.10,3730.22,3741.88,0
    Dates: MM/DD/YY or MM/DD/YYYY.
    """
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
                thousands=",",
                skipinitialspace=True,
            )
            break
        except Exception:
            continue
    else:
        logging.error("_parse_wsj_csv: could not decode CSV.")
        return None

    col_map = {}
    for col in raw.columns:
        c = col.strip().lower()
        if c == "date":
            col_map[col] = "Date"
        elif c in ("close", "price", "last", "adj close"):
            col_map[col] = "Close"
        elif c in ("high", "hi"):
            col_map[col] = "High"
        elif c in ("low", "lo"):
            col_map[col] = "Low"
    raw = raw.rename(columns=col_map)

    if "Date" not in raw.columns or "Close" not in raw.columns:
        logging.error(
            "_parse_wsj_csv: required columns missing. Available: %s",
            list(raw.columns),
        )
        return None

    raw["Date"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip(),
        format="mixed", dayfirst=False, errors="coerce",
    )
    raw = raw.dropna(subset=["Date"])
    if raw.empty:
        logging.error("_parse_wsj_csv: no valid dates after parsing.")
        return None

    out = pd.DataFrame(index=raw["Date"].values)
    out.index      = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Data"
    out[CLOSE_COL]   = pd.to_numeric(raw["Close"].values,  errors="coerce")
    out["Najwyzszy"] = pd.to_numeric(raw["High"].values,   errors="coerce") \
                       if "High" in raw.columns else out[CLOSE_COL]
    out["Najnizszy"] = pd.to_numeric(raw["Low"].values,    errors="coerce") \
                       if "Low"  in raw.columns else out[CLOSE_COL]
    out["Najwyzszy"] = out["Najwyzszy"].where(out["Najwyzszy"] > 0, out[CLOSE_COL])
    out["Najnizszy"] = out["Najnizszy"].where(out["Najnizszy"] > 0, out[CLOSE_COL])

    return _finalise(out)


def _parse_generic_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """
    Parse a generic Date/Close CSV.

    Accepts various separators (comma, semicolon, tab) and decimal
    conventions (including European comma decimals). Column names
    matched case-insensitively; at minimum Date and Close are required.
    """
    raw = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for sep in (",", ";", "\t"):
            try:
                candidate = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    encoding=encoding,
                    sep=sep,
                    thousands=",",
                    decimal=".",
                    skipinitialspace=True,
                )
                if len(candidate.columns) >= 2:
                    # Detect European decimal comma in semicolon-delimited files
                    if sep == ";":
                        close_guess = [
                            c for c in candidate.columns
                            if c.strip().lower() in ("close", "price", "last", "zamkniecie", "TBSP_synth")
                        ]
                        if close_guess:
                            try:
                                first_val = pd.to_numeric(
                                    candidate[close_guess[0]].iloc[0], errors="coerce"
                                )
                                if first_val is not None and first_val > 1_000_000:
                                    candidate = pd.read_csv(
                                        io.BytesIO(raw_bytes),
                                        encoding=encoding,
                                        sep=sep,
                                        thousands=".",
                                        decimal=",",
                                        skipinitialspace=True,
                                    )
                            except Exception:
                                pass
                    raw = candidate
                    break
            except Exception:
                continue
        if raw is not None:
            break

    if raw is None:
        logging.error("_parse_generic_csv: could not parse CSV.")
        return None

    col_map = {}
    for col in raw.columns:
        c = col.strip().lower()
        if c in ("date", "data"):
            col_map[col] = "Date"
        elif c in ("close", "price", "last", "zamkniecie", "adj close"):
            col_map[col] = "Close"
    raw = raw.rename(columns=col_map)

    if "Date" not in raw.columns or "Close" not in raw.columns:
        logging.error(
            "_parse_generic_csv: required columns missing. Available: %s",
            list(raw.columns),
        )
        return None

    raw["Date"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip(),
        format="mixed", dayfirst=False, errors="coerce",
    )
    raw = raw.dropna(subset=["Date"])
    if raw.empty:
        logging.error("_parse_generic_csv: no valid dates after parsing.")
        return None

    close = pd.to_numeric(raw["Close"], errors="coerce")
    close.index = pd.to_datetime(raw["Date"].values).tz_localize(None)
    close = close.dropna()
    close.index.name = "Data"

    out = _to_stooq_df(close)
    return _finalise(out)


def _auto_detect_format(raw_bytes: bytes) -> str:
    """
    Heuristic: if the first non-empty line contains 'Open' or 'Volume'
    it looks like a WSJ export; otherwise treat as generic.
    """
    try:
        header = raw_bytes[:512].decode("utf-8-sig", errors="replace").lower()
    except Exception:
        return "generic"
    if "open" in header and "volume" in header:
        return "wsj"
    return "generic"


# ============================================================
# EXTENSION DATA SOURCES
# ============================================================

def _download_yfinance(ticker: str) -> pd.DataFrame | None:
    """Download daily OHLCV from Yahoo Finance, return stooq-format DataFrame."""
    if not _YFINANCE_AVAILABLE:
        logging.error("yfinance not installed.  pip install yfinance")
        return None

    for yticker in (ticker,):
        logging.info("Downloading %s from yfinance ...", yticker)
        try:
            raw = yf.download(
                yticker,
                start=DATA_START,
                auto_adjust=True,
                progress=False,
                actions=False,
            )
        except Exception as exc:
            logging.error("yfinance download failed for %s: %s", yticker, exc)
            return None

        if raw is None or raw.empty:
            logging.warning("yfinance returned empty data for %s.", yticker)
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(level=1, axis=1)

        close_cands = [c for c in raw.columns if c.lower() == "close"]
        if not close_cands:
            logging.error("No 'Close' column in yfinance output for %s.", yticker)
            return None

        high_cands = [c for c in raw.columns if c.lower() == "high"]
        low_cands  = [c for c in raw.columns if c.lower() == "low"]

        out = pd.DataFrame(index=raw.index)
        out.index      = pd.to_datetime(out.index).tz_localize(None)
        out.index.name = "Data"
        out[CLOSE_COL]   = raw[close_cands[0]]
        out["Najwyzszy"] = raw[high_cands[0]] if high_cands else out[CLOSE_COL]
        out["Najnizszy"] = raw[low_cands[0]]  if low_cands  else out[CLOSE_COL]

        return _finalise(out)

    return None

# MODIFIED TO LOCAL
def _download_stooq(ticker: str, mandatory: bool = True) -> pd.DataFrame | None:
         
        """
        Load a stooq series from /data subfolder, clip to DATA_START.
        """
    
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
        df = load_csv(path)
        if df is None:
            if mandatory:
                logging.error("FAIL: load_csv returned None for %s — exiting.", ticker)
                sys.exit(1)
                return None
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info(
            "OK  : %-14s  %5d rows  %s to %s",
            ticker, len(df), df.index.min().date(), df.index.max().date(),
            )
        return df
   
    


# ============================================================
# CHAIN-LINK EXTENSION
# ============================================================

def _extend_series(
    base_df:   pd.DataFrame,
    ext_df:    pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Extend base_df forward using daily returns from ext_df.

    The base series is preserved exactly. ext_df is used only for dates
    strictly after the last base observation, via pure return chain-link:

        anchor_price = base_df[CLOSE_COL].iloc[-1]
        ext_rets     = ext_df[CLOSE_COL].pct_change() for dates > anchor
        extension    = anchor_price * (1 + ext_rets).cumprod()

    Parameters
    ----------
    base_df : stooq-format DataFrame — authoritative base series
    ext_df  : stooq-format DataFrame — extension source (yfinance or stooq)

    Returns
    -------
    Combined stooq-format DataFrame, or None on failure.
    """
    if base_df is None or base_df.empty:
        logging.error("_extend_series: base_df is empty.")
        return None
    if ext_df is None or ext_df.empty:
        logging.warning(
            "_extend_series: extension data empty — returning base unchanged."
        )
        return base_df.copy()

    base_close  = base_df[CLOSE_COL].sort_index().dropna()
    ext_close   = ext_df[CLOSE_COL].sort_index().dropna()

    anchor_date  = base_close.index.max()
    anchor_price = float(base_close.iloc[-1])

    # Need one ext observation at or before anchor to seed pct_change
    ext_from_anchor = ext_close.loc[ext_close.index >= anchor_date]
    if ext_from_anchor.empty:
        logging.warning(
            "_extend_series: extension has no data at or after %s — "
            "returning base unchanged.", anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    ext_rets_new = (
        ext_from_anchor
        .pct_change()
        .loc[lambda s: s.index > anchor_date]
        .dropna()
    )

    if ext_rets_new.empty:
        logging.info(
            "_extend_series: base already up to date (last: %s).",
            anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    extension = anchor_price * (1 + ext_rets_new).cumprod()
    combined  = pd.concat([base_close, extension]).sort_index()
    combined  = combined[~combined.index.duplicated(keep="last")]

    logging.info(
        "_extend_series: base %d rows (%s to %s) + %d new rows (%s to %s) = %d total",
        len(base_close),
        base_close.index.min().date(), anchor_date.date(),
        len(ext_rets_new),
        ext_rets_new.index.min().date(), ext_rets_new.index.max().date(),
        len(combined),
    )
    return _to_stooq_df(combined)


# ============================================================
# STOOQ-FORMAT HELPERS
# ============================================================

def _to_stooq_df(close_series: pd.Series) -> pd.DataFrame:
    """Wrap a close Series in a stooq-format DataFrame (H=L=C, no intraday)."""
    out = pd.DataFrame({
        CLOSE_COL:   close_series,
        "Najwyzszy": close_series,
        "Najnizszy": close_series,
    })
    out.index.name = "Data"
    return out


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, apply data floor."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[df.index >= pd.Timestamp(DATA_START)]
    df = df.dropna(subset=[CLOSE_COL])
    return df


# ============================================================
# PUBLIC API — LOAD
# ============================================================

def load_combined_from_drive(
    folder_id:        str,
    combined_filename: str,
    credentials_path: str = CREDENTIALS_PATH,
) -> pd.DataFrame | None:
    """
    Download a pre-built combined series CSV from Google Drive.

    Called by the daily runner on every run after the combined series
    has been built at least once.

    Parameters
    ----------
    folder_id         : str — Google Drive folder ID
    combined_filename : str — filename of the combined CSV on Drive
    credentials_path  : str — path to service account JSON

    Returns
    -------
    stooq-format DataFrame, or None if not found / parse error.
    """
    if not _GDRIVE_AVAILABLE:
        logging.error("load_combined_from_drive: google-api packages not available.")
        return None

    try:
        service = _get_drive_service(credentials_path)
    except Exception as exc:
        logging.error("load_combined_from_drive: Drive auth failed — %s", exc)
        return None

    file_id = _find_file_id(service, folder_id, combined_filename)
    if file_id is None:
        logging.warning(
            "load_combined_from_drive: '%s' not found in Drive folder %s.",
            combined_filename, folder_id,
        )
        return None

    raw_bytes = _download_bytes(service, file_id)
    logging.info(
        "load_combined_from_drive: downloaded '%s' (%d bytes).",
        combined_filename, len(raw_bytes),
    )

    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            parse_dates=["Data"],
            index_col="Data",
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Data"
        df = _finalise(df)
        logging.info(
            "load_combined_from_drive: %d rows  %s to %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df
    except Exception as exc:
        logging.error("load_combined_from_drive: parse error — %s", exc)
        return None


# ============================================================
# PUBLIC API — BUILD AND UPLOAD
# ============================================================

def build_and_upload(
    folder_id:          str,
    raw_filename:       str,
    combined_filename:  str,
    extension_ticker:   str,
    extension_source:   str = "yfinance",
    raw_csv_format:     str = "auto",
    credentials_path:   str = CREDENTIALS_PATH,
) -> pd.DataFrame | None:
    """
    Build or update a combined price series and upload to Google Drive.

    Auto-selects between two pipeline modes:

    Update (combined CSV already on Drive):
      1. Download combined CSV from Drive as base
      2. Extend forward with extension_ticker
      3. Save updated combined CSV back to Drive

    First build (combined CSV not on Drive):
      1. Download raw_filename from Drive
      2. Parse with format detector (wsj / generic / auto)
      3. Extend forward with extension_ticker
      4. Save combined CSV to Drive as combined_filename

    Parameters
    ----------
    folder_id          : str — Google Drive folder ID
    raw_filename       : str — filename of the user-uploaded raw CSV on Drive
    combined_filename  : str — filename for the output combined CSV on Drive
    extension_ticker   : str — ticker to use for forward extension
                               yfinance: "URTH", "^STOXX", etc.
                               stooq:    "^tbsp", "wig20tr", etc.
    extension_source   : str — "yfinance" (default) or "stooq"
    raw_csv_format     : str — "wsj" | "generic" | "auto" (default)
                               "auto" detects format from file content
    credentials_path   : str — path to service account JSON

    Returns
    -------
    Combined stooq-format DataFrame, or None on failure.
    """
    logging.info("=" * 70)
    logging.info("PRICE SERIES BUILDER  —  BUILD & UPLOAD")
    logging.info("  Drive folder      : %s", folder_id)
    logging.info("  Raw source        : %s", raw_filename)
    logging.info("  Output file       : %s", combined_filename)
    logging.info("  Extension ticker  : %s (%s)", extension_ticker, extension_source)
    logging.info("  Raw CSV format    : %s", raw_csv_format)
    logging.info("  Credentials       : %s", credentials_path)
    logging.info("=" * 70)

    # Authenticate
    try:
        service = _get_drive_service(credentials_path)
    except Exception as exc:
        logging.error("Drive authentication failed: %s", exc)
        return None

    # ── Determine base series ─────────────────────────────────────────────
    base_df  = None
    mode_str = "unknown"

    existing = load_combined_from_drive(
        folder_id, combined_filename, credentials_path
    )
    if existing is not None:
        base_df  = existing
        mode_str = f"UPDATE  (base = existing {combined_filename})"
        logging.info("Mode: %s  last date = %s", mode_str, base_df.index.max().date())
    else:
        logging.info("Mode: FIRST BUILD  (combined not found on Drive)")
        raw_id = _find_file_id(service, folder_id, raw_filename)
        if raw_id is None:
            logging.error(
                "build_and_upload: '%s' not found in Drive folder %s.\n"
                "  Upload your raw CSV to Drive with exactly that filename.",
                raw_filename, folder_id,
            )
            return None

        raw_bytes = _download_bytes(service, raw_id)
        logging.info(
            "Downloaded raw CSV: '%s'  (%d bytes)", raw_filename, len(raw_bytes)
        )

        # Detect or use specified format
        fmt = raw_csv_format
        if fmt == "auto":
            fmt = _auto_detect_format(raw_bytes)
            logging.info("Auto-detected raw CSV format: %s", fmt)

        if fmt == "wsj":
            base_df = _parse_wsj_csv(raw_bytes)
        else:
            base_df = _parse_generic_csv(raw_bytes)

        if base_df is None:
            logging.error("build_and_upload: failed to parse raw CSV.")
            return None

        mode_str = f"FIRST BUILD  (base = {raw_filename})"
        logging.info(
            "Raw CSV parsed: %d rows  %s to %s",
            len(base_df),
            base_df.index.min().date(), base_df.index.max().date(),
        )

    # ── Download extension data ───────────────────────────────────────────
    if extension_source == "stooq":
        ext_df = _download_stooq(extension_ticker)
    else:
        ext_df = _download_yfinance(extension_ticker)

    if ext_df is None:
        logging.error(
            "build_and_upload: extension download failed for %s (%s). "
            "Returning base series without extension.",
            extension_ticker, extension_source,
        )
        return base_df

    # ── Extend ────────────────────────────────────────────────────────────
    combined = _extend_series(base_df, ext_df)
    if combined is None:
        logging.error("build_and_upload: extension failed.")
        return None

    # ── Save locally and upload ───────────────────────────────────────────
    tmp_path = os.path.join(tempfile.gettempdir(), combined_filename)
    combined.to_csv(tmp_path, date_format="%Y-%m-%d")
    logging.info("Local copy saved: %s  (%d rows)", tmp_path, len(combined))

    try:
        _upload_to_drive(service, folder_id, tmp_path, combined_filename)
    except Exception as exc:
        logging.error("Drive upload failed: %s", exc)
        logging.info("Combined CSV still available locally at: %s", tmp_path)

    logging.info("=" * 70)
    logging.info("BUILD & UPLOAD COMPLETE  [%s]", mode_str)
    logging.info(
        "  %d rows  %s to %s",
        len(combined),
        combined.index.min().date(), combined.index.max().date(),
    )
    logging.info("=" * 70)
    return combined


# ============================================================
# BACKWARD COMPATIBILITY SHIMS
# ============================================================
# Allow existing imports from wsj_msci_world and stoxx600 to keep working
# without modification by importing from this module.

def build_msci_world(folder_id: str, credentials_path: str = CREDENTIALS_PATH):
    """Backward-compatible wrapper for wsj_msci_world.build_and_upload."""
    return build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "msci_world_wsj_raw.csv",
        combined_filename = "msci_world_combined.csv",
        extension_ticker  = "URTH",
        extension_source  = "yfinance",
        raw_csv_format    = "wsj",
        credentials_path  = credentials_path,
    )


def build_stoxx600(folder_id: str, credentials_path: str = CREDENTIALS_PATH):
    """Backward-compatible wrapper for stoxx600.build_and_upload."""
    return build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "stoxx600.csv",
        combined_filename = "stoxx600_combined.csv",
        extension_ticker  = "^STOXX",
        extension_source  = "yfinance",
        raw_csv_format    = "generic",
        credentials_path  = credentials_path,
    )


def build_tbsp_extended(folder_id: str, credentials_path: str = CREDENTIALS_PATH):
    """Build the yield-backcast TBSP extended series."""
    return build_and_upload(
        folder_id         = folder_id,
        raw_filename      = "tbsp_backcast.csv",
        combined_filename = "tbsp_extended_combined.csv",
        extension_ticker  = "^tbsp",
        extension_source  = "stooq",
        raw_csv_format    = "generic",
        credentials_path  = credentials_path,
    )


# ============================================================
# ENTRY POINT  (run standalone to test)
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build or update a combined price series on Google Drive."
    )
    parser.add_argument("--folder_id",         required=True)
    parser.add_argument("--raw_filename",      required=True)
    parser.add_argument("--combined_filename", required=True)
    parser.add_argument("--extension_ticker",  required=True)
    parser.add_argument(
        "--extension_source", default="yfinance",
        choices=["yfinance", "stooq"],
    )
    parser.add_argument(
        "--raw_csv_format", default="auto",
        choices=["auto", "wsj", "generic"],
    )
    parser.add_argument("--credentials_path", default=CREDENTIALS_PATH)
    args = parser.parse_args()

    result = build_and_upload(
        folder_id         = args.folder_id,
        raw_filename      = args.raw_filename,
        combined_filename = args.combined_filename,
        extension_ticker  = args.extension_ticker,
        extension_source  = args.extension_source,
        raw_csv_format    = args.raw_csv_format,
        credentials_path  = args.credentials_path,
    )

    if result is not None:
        print(
            f"\nCombined series ready.\n"
            f"  {len(result)} trading days\n"
            f"  {result.index.min().date()} to {result.index.max().date()}\n"
            f"  Uploaded to Drive as: {args.combined_filename}\n"
        )
        sys.exit(0)
    else:
        print("\nBuild failed. Check logs for details.\n")
        sys.exit(1)
