"""
stoxx600.py
===========
Builds and maintains the combined STOXX 600 price series used by the
global equity framework (Mode A -- global_equity portfolio).

DATA CHAIN
----------
  Base series -- historical STOXX 600 CSV (user-provided):
    File: stoxx600.csv
    Two columns: Date, Close
    Uploaded manually to Google Drive by the user.
    Provides long history not available from yfinance (yfinance only
    has ^STOXX from ~2004).

  Extension -- ^STOXX from Yahoo Finance:
    Downloaded via yfinance.
    Used ONLY to add new trading days after the base CSV's last date.
    No level rescaling -- only daily returns from the extension ticker
    are chain-linked onto the last base price.

EXTENSION METHOD
----------------
  Identical to wsj_msci_world.extend_with_urth:

    anchor_price  = base_series.iloc[-1]           (last base close)
    stoxx_returns = ^STOXX pct_change() for dates strictly after anchor
    extension     = anchor_price * (1 + stoxx_returns).cumprod()
    combined      = concat(base_series, extension)

  Base levels are preserved exactly.  yfinance levels are never used
  directly -- only their daily return sequence matters.

TWO PIPELINE MODES (auto-selected)
------------------------------------
  First build:
    stoxx600.csv (Drive) --> parse --> extend with yfinance --> save combined

  Update (daily runner):
    stoxx600_combined.csv (Drive) --> extend with yfinance --> save updated

HOW TO RUN FROM IDE
-------------------
  1. Upload stoxx600.csv (Date, Close columns) to your Drive folder.
  2. Set GDRIVE_FOLDER_ID in USER SETTINGS below (or set the env var).
  3. Ensure credentials.json is in your system temp directory.
  4. Run:  python stoxx600.py

OUTPUTS (Google Drive)
----------------------
  stoxx600_combined.csv  --  stooq-format, columns:
                             Data (index, YYYY-MM-DD), Zamkniecie,
                             Najwyzszy, Najnizszy

IMPORTABLE API
--------------
  build_and_upload()            Full pipeline (first build or update).
  load_combined_from_drive()    Load pre-built combined CSV from Drive.

CREDENTIALS
-----------
  os.path.join(tempfile.gettempdir(), "credentials.json")
  Same service account JSON used by all other strategy scripts.
  Same GDRIVE_FOLDER_ID env var / Drive folder.
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

# Google Drive (required)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False

# yfinance (required for extension)
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


# ============================================================
# USER SETTINGS  <- edit when running from IDE
# ============================================================

# Google Drive folder ID -- same folder used by all other strategy scripts.
# Env var GDRIVE_FOLDER_ID takes precedence over this hardcoded value.
GDRIVE_FOLDER_ID_DEFAULT = ""   # <- paste your folder ID here

# Filename of the user-provided historical CSV on Google Drive
RAW_DRIVE_FILENAME      = "stoxx600.csv"

# Output filename in Google Drive (read and written by this script)
COMBINED_DRIVE_FILENAME = "stoxx600_combined.csv"

# yfinance ticker for STOXX 600 extension
STOXX_TICKER = "^STOXX"

# Data floor -- rows before this date are discarded
DATA_START = "1990-01-01"

# stooq close column name used throughout the framework
CLOSE_COL = "Zamkniecie"

# Credentials file path (same convention as all other strategy scripts)
CREDENTIALS_PATH = os.path.join(tempfile.gettempdir(), "credentials.json")

# Log file (written when the script runs as __main__)
LOG_FILE = "stoxx600.log"


# ============================================================
# LOGGING
# ============================================================

def _setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(ch)


# ============================================================
# DRIVE HELPERS  (private)
# ============================================================

def _get_folder_id() -> str:
    return (
        os.environ.get("GDRIVE_FOLDER_ID", "").strip()
        or GDRIVE_FOLDER_ID_DEFAULT.strip()
    )


def _get_drive_service():
    if not _GDRIVE_AVAILABLE:
        raise ImportError(
            "google-api-python-client not installed. "
            "pip install google-api-python-client google-auth"
        )
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return _gdrive_build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id: str, filename: str) -> str | None:
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    result = (
        service.files()
        .list(q=q, spaces="drive", fields="files(id, name)")
        .execute()
    )
    files = result.get("files", [])
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
    folder_id: str,
    local_path: str,
    filename:   str,
    mimetype:   str = "text/csv",
) -> str:
    """Upload or update a file in Drive. Returns the Drive file ID."""
    existing_id = _find_file_id(service, folder_id, filename)
    media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)
    if existing_id:
        service.files().update(
            fileId=existing_id, media_body=media
        ).execute()
        logging.info("Drive UPDATED : %s  (id=%s)", filename, existing_id)
        return existing_id
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        result = service.files().create(
            body=metadata, media_body=media, fields="id"
        ).execute()
        fid = result["id"]
        logging.info("Drive CREATED : %s  (id=%s)", filename, fid)
        return fid


# ============================================================
# RAW CSV PARSING
# ============================================================

def _parse_raw_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """
    Parse the user-provided stoxx600.csv (Date, Close columns).

    Accepts various date formats and thousands-separated numbers.
    Returns a stooq-format DataFrame or None on failure.
    """
    # Detect separator: try comma first, fall back to semicolon.
    # Some STOXX CSV exports use semicolons as the delimiter.
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
                # Accept if it produced at least 2 columns
                if len(candidate.columns) >= 2:
                    raw = candidate
                    logging.info(
                        "_parse_raw_csv: detected separator=%r  encoding=%s",
                        sep, encoding,
                    )
                    # If semicolon-delimited, the Close column may use a
                    # European decimal comma (e.g. "602,4500000").
                    # Detect this by checking whether the parsed numeric values
                    # are unreasonably large (> 1,000,000 for an equity index).
                    if sep == ";" and len(candidate.columns) >= 2:
                        close_col_guess = [
                            c for c in candidate.columns
                            if c.strip().lower() in (
                                "close", "price", "last", "adj close", "adjusted close"
                            )
                        ]
                        if close_col_guess:
                            col = close_col_guess[0]
                            try:
                                first_val = pd.to_numeric(
                                    candidate[col].iloc[0], errors="coerce"
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
                                    raw = candidate
                                    logging.info(
                                        "_parse_raw_csv: European decimal comma detected "
                                        "(value > 1M with decimal='.') — "
                                        "re-parsed with decimal=','"
                                    )
                            except Exception:
                                pass  # keep the first parse
                    break
            except Exception:
                continue
        if raw is not None:
            break
    if raw is None:
        logging.error("_parse_raw_csv: could not decode CSV.")
        return None

    logging.info(
        "_parse_raw_csv: raw shape=%s  columns=%s",
        raw.shape, list(raw.columns),
    )

    # Normalise column names — accept common variants
    col_map = {}
    for col in raw.columns:
        c = col.strip().lower()
        if c == "date":
            col_map[col] = "Date"
        elif c in ("close", "price", "last", "adj close", "adjusted close"):
            col_map[col] = "Close"
    raw = raw.rename(columns=col_map)

    if "Date" not in raw.columns:
        logging.error(
            "_parse_raw_csv: no date column found. Columns: %s",
            list(raw.columns),
        )
        return None
    if "Close" not in raw.columns:
        logging.error(
            "_parse_raw_csv: no close column found. Columns: %s",
            list(raw.columns),
        )
        return None

    raw["Date"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip(),
        format="mixed",
        dayfirst=False,
        errors="coerce",
    )
    raw = raw.dropna(subset=["Date"])

    if raw.empty:
        logging.error("_parse_raw_csv: no valid rows after date parsing.")
        return None

    close = pd.to_numeric(raw["Close"], errors="coerce")
    close.index = pd.to_datetime(raw["Date"].values).tz_localize(None)
    close = close.dropna()
    close.index.name = "Data"
    close = close.sort_index()
    close = close.loc[close.index >= pd.Timestamp(DATA_START)]
    close = close[~close.index.duplicated(keep="last")]

    out = _to_stooq_df(close)

    logging.info(
        "_parse_raw_csv: %d rows  %s to %s  close range [%.2f, %.2f]",
        len(out),
        out.index.min().date(), out.index.max().date(),
        out[CLOSE_COL].min(), out[CLOSE_COL].max(),
    )
    return out


# ============================================================
# YFINANCE EXTENSION
# ============================================================

def _download_stoxx_yfinance() -> pd.DataFrame | None:
    """
    Download ^STOXX from Yahoo Finance.
    Returns a stooq-format DataFrame or None on failure.
    """
    if not _YFINANCE_AVAILABLE:
        logging.error("yfinance not installed.  pip install yfinance")
        return None

    logging.info("Downloading %s from yfinance ...", STOXX_TICKER)
    try:
        raw = yf.download(
            STOXX_TICKER,
            start=DATA_START,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
    except Exception as exc:
        logging.error("yfinance download failed for %s: %s", STOXX_TICKER, exc)
        return None

    if raw is None or raw.empty:
        logging.warning("yfinance returned empty data for %s.", STOXX_TICKER)
        return None

    # Flatten MultiIndex (yfinance >= 0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(level=1, axis=1)

    close_cands = [c for c in raw.columns if c.lower() == "close"]
    if not close_cands:
        logging.error(
            "_download_stoxx_yfinance: no 'Close' column in yfinance output. "
            "Columns: %s", list(raw.columns),
        )
        return None

    close = raw[close_cands[0]].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Data"
    close = close.sort_index()
    close = close.loc[close.index >= pd.Timestamp(DATA_START)]

    out = _to_stooq_df(close)
    logging.info(
        "%s from yfinance: %d rows  %s to %s",
        STOXX_TICKER, len(out),
        out.index.min().date(), out.index.max().date(),
    )
    return out


# ============================================================
# EXTENSION  (pure return chain-link)
# ============================================================

def extend_with_yfinance(
    base_df:  pd.DataFrame,
    stoxx_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Extend the base STOXX 600 series forward using yfinance daily returns.

    The base series is kept exactly as-is.  yfinance is used only to add
    trading days that fall strictly after the last date in the base series.

    Extension method
    ----------------
    Let T   = last date in base_df (the anchor date).
    Let P_T = base_df[CLOSE_COL].iloc[-1]  (anchor price level).

    New prices for dates t > T:

        r_t     = stoxx_close[t] / stoxx_close[t-1] - 1   (yfinance return)
        P_{t}   = P_{t-1} * (1 + r_t)

    starting from P_T.  The yfinance price *level* at T is irrelevant —
    only daily return sequences are used.

    Parameters
    ----------
    base_df  : pd.DataFrame  — stooq-format base series (historical CSV)
    stoxx_df : pd.DataFrame  — stooq-format yfinance download

    Returns
    -------
    pd.DataFrame | None  — combined stooq-format series, or None on failure
    """
    if base_df is None or base_df.empty:
        logging.error("extend_with_yfinance: base_df is empty.")
        return None
    if stoxx_df is None or stoxx_df.empty:
        logging.warning(
            "extend_with_yfinance: yfinance data is empty — "
            "returning base series unchanged."
        )
        return base_df.copy()

    base_close  = base_df[CLOSE_COL].sort_index().dropna()
    stoxx_close = stoxx_df[CLOSE_COL].sort_index().dropna()

    anchor_date  = base_close.index.max()
    anchor_price = float(base_close.iloc[-1])

    # yfinance data at or after the anchor (need one day for pct_change seed)
    stoxx_from_anchor = stoxx_close.loc[stoxx_close.index >= anchor_date]

    if stoxx_from_anchor.empty:
        logging.warning(
            "extend_with_yfinance: no yfinance data at or after %s — "
            "returning base series unchanged.",
            anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    # Compute returns on the slice starting at anchor; keep only dates > anchor
    stoxx_rets_new = (
        stoxx_from_anchor
        .pct_change()
        .loc[lambda s: s.index > anchor_date]
        .dropna()
    )

    if stoxx_rets_new.empty:
        logging.info(
            "extend_with_yfinance: no new yfinance dates after %s — "
            "base series already up to date.",
            anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    # Chain-link extension
    extension = anchor_price * (1 + stoxx_rets_new).cumprod()

    combined = pd.concat([base_close, extension]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    logging.info(
        "extend_with_yfinance: base %d rows (%s to %s)  +  %d new rows "
        "(%s to %s)  =  %d total",
        len(base_close),
        base_close.index.min().date(), anchor_date.date(),
        len(stoxx_rets_new),
        stoxx_rets_new.index.min().date(),
        stoxx_rets_new.index.max().date(),
        len(combined),
    )

    return _to_stooq_df(combined)


# ============================================================
# DRIVE I/O — COMBINED CSV
# ============================================================

def load_combined_from_drive(
    service   = None,
    folder_id : str | None = None,
    filename  : str = COMBINED_DRIVE_FILENAME,
) -> pd.DataFrame | None:
    """
    Download the pre-built combined STOXX 600 CSV from Google Drive.

    Called by the runfile and diagnostic on every daily run after the
    combined series has been built at least once.

    Parameters
    ----------
    service   : Drive service object (None = build from credentials)
    folder_id : str | None  (None = from USER SETTINGS / env var)
    filename  : str

    Returns
    -------
    pd.DataFrame | None  — stooq-format combined series
    """
    if not _GDRIVE_AVAILABLE:
        logging.error("load_combined_from_drive: google-api packages not available.")
        return None

    if service is None:
        try:
            service = _get_drive_service()
        except Exception as exc:
            logging.error("load_combined_from_drive: Drive auth failed — %s", exc)
            return None

    if folder_id is None:
        folder_id = _get_folder_id()

    file_id = _find_file_id(service, folder_id, filename)
    if file_id is None:
        logging.warning(
            "load_combined_from_drive: '%s' not found in Drive folder %s.",
            filename, folder_id,
        )
        return None

    raw_bytes = _download_bytes(service, file_id)
    logging.info(
        "load_combined_from_drive: downloaded '%s'  (%d bytes).",
        filename, len(raw_bytes),
    )

    try:
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            parse_dates=["Data"],
            index_col="Data",
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Data"
        df = df.sort_index().dropna(subset=[CLOSE_COL])
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info(
            "load_combined_from_drive: %d rows  %s to %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df
    except Exception as exc:
        logging.error("load_combined_from_drive: parse error — %s", exc)
        return None


# ============================================================
# STOOQ-FORMAT HELPER
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


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_and_upload(
    folder_id        : str | None = None,
    raw_filename     : str = RAW_DRIVE_FILENAME,
    combined_filename: str = COMBINED_DRIVE_FILENAME,
) -> pd.DataFrame | None:
    """
    Full pipeline — builds or updates the combined STOXX 600 series.

    Auto-selects mode:

    Update (stoxx600_combined.csv already on Drive):
      1. Download combined CSV from Drive as the base
      2. Download ^STOXX from yfinance
      3. Extend combined with yfinance returns after last combined date
      4. Save updated combined CSV back to Drive

    First build (stoxx600_combined.csv not found on Drive):
      1. Download stoxx600.csv (raw historical) from Drive
      2. Parse Date, Close columns
      3. Download ^STOXX from yfinance
      4. Extend raw series with yfinance returns after last raw date
      5. Save combined CSV to Drive as stoxx600_combined.csv

    In both modes the base levels are preserved exactly.  Only yfinance
    daily returns (not price levels) are used for the extension.

    Parameters
    ----------
    folder_id         : str | None  (None = from USER SETTINGS / env var)
    raw_filename      : str  -- Drive filename for the user-uploaded raw CSV
    combined_filename : str  -- Drive filename for the output combined CSV

    Returns
    -------
    pd.DataFrame | None  — the combined stooq-format series
    """
    if folder_id is None:
        folder_id = _get_folder_id()

    if not folder_id:
        logging.error(
            "build_and_upload: GDRIVE_FOLDER_ID not set.\n"
            "  Set GDRIVE_FOLDER_ID_DEFAULT in USER SETTINGS or the env var."
        )
        return None

    logging.info("=" * 70)
    logging.info("STOXX 600  --  BUILD & UPLOAD")
    logging.info("  Drive folder : %s", folder_id)
    logging.info("  Raw source   : %s", raw_filename)
    logging.info("  Output file  : %s", combined_filename)
    logging.info("  Credentials  : %s", CREDENTIALS_PATH)
    logging.info("=" * 70)

    # Authenticate
    try:
        service = _get_drive_service()
    except Exception as exc:
        logging.error("Drive authentication failed: %s", exc)
        return None

    # ── Determine base series ─────────────────────────────────────────────
    base_df  = None
    mode_str = "unknown"

    combined_existing = load_combined_from_drive(service, folder_id, combined_filename)
    if combined_existing is not None:
        base_df  = combined_existing
        mode_str = "UPDATE  (base = existing combined CSV)"
        logging.info(
            "Mode: %s  last date = %s",
            mode_str, base_df.index.max().date(),
        )
    else:
        # No combined yet — build from raw CSV
        logging.info("Mode: FIRST BUILD  (combined CSV not found on Drive)")
        raw_id = _find_file_id(service, folder_id, raw_filename)
        if raw_id is None:
            logging.error(
                "build_and_upload: '%s' not found in Drive folder %s.\n"
                "  Upload your historical STOXX 600 CSV to Drive with "
                "exactly that filename.",
                raw_filename, folder_id,
            )
            return None
        raw_bytes = _download_bytes(service, raw_id)
        logging.info(
            "Downloaded raw CSV: '%s'  (%d bytes)", raw_filename, len(raw_bytes)
        )
        base_df = _parse_raw_csv(raw_bytes)
        if base_df is None:
            logging.error("build_and_upload: failed to parse raw CSV.")
            return None
        mode_str = "FIRST BUILD  (base = %s)" % raw_filename
        logging.info(
            "Raw CSV parsed: %d rows  %s to %s",
            len(base_df),
            base_df.index.min().date(), base_df.index.max().date(),
        )

    # ── Download yfinance extension ───────────────────────────────────────
    stoxx_yf = _download_stoxx_yfinance()
    if stoxx_yf is None:
        logging.error(
            "build_and_upload: yfinance download failed — "
            "returning base series without extension."
        )
        # Return base unchanged (better than None — at least callers get something)
        return base_df

    # ── Extend ────────────────────────────────────────────────────────────
    combined = extend_with_yfinance(base_df, stoxx_yf)
    if combined is None:
        logging.error("build_and_upload: extension failed.")
        return None

    # ── Save locally and upload ───────────────────────────────────────────
    tmp_path = os.path.join(tempfile.gettempdir(), combined_filename)
    combined.to_csv(tmp_path, date_format="%Y-%m-%d")
    logging.info("Local copy: %s  (%d rows)", tmp_path, len(combined))

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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _setup_logging()

    logging.info("stoxx600.py  started  %s", dt.datetime.now())
    logging.info("Credentials path : %s", CREDENTIALS_PATH)
    logging.info("Credentials exist: %s", os.path.exists(CREDENTIALS_PATH))

    result = build_and_upload()

    if result is not None:
        print(
            f"\nSTOXX 600 combined series ready.\n"
            f"  {len(result)} trading days\n"
            f"  {result.index.min().date()} to {result.index.max().date()}\n"
            f"  Uploaded to Drive as: {COMBINED_DRIVE_FILENAME}\n"
            f"  Local copy:           "
            f"{os.path.join(tempfile.gettempdir(), COMBINED_DRIVE_FILENAME)}\n"
        )
    else:
        print(
            "\nBuild failed.  Check stoxx600.log for details.\n\n"
            "Likely causes:\n"
            "  1. GDRIVE_FOLDER_ID not set — edit USER SETTINGS above\n"
            f"  2. credentials.json missing at {CREDENTIALS_PATH}\n"
            f"  3. stoxx600.csv not uploaded to Drive yet\n"
            "  4. yfinance not installed — pip install yfinance\n"
        )
        sys.exit(1)