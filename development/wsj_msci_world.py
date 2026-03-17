"""
wsj_msci_world.py
=================
Builds and maintains the combined MSCI World price series used by the
global equity framework (Mode B -- msci_world portfolio).

DATA CHAIN
----------
  Base series -- WSJ MSCI World (USD):
    Downloaded manually from:
      https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices
    Saved to Google Drive.  This script reads it from there.
    Provides history back to approximately 2010.

  Extension -- URTH ETF (iShares MSCI World, USD):
    Downloaded from Yahoo Finance via yfinance.
    Used ONLY to extend the base series forward from its last date.
    No level rescaling, no overlap ratio, no multiplier.

EXTENSION METHOD
----------------
  The WSJ base series is the authoritative source for all dates it covers.
  URTH is used exclusively to add new trading days after the base series
  ends, using a pure return chain:

    last_base_price = base_series.iloc[-1]
    urth_returns    = URTH[CLOSE_COL].pct_change()  (days after base end only)
    extension       = last_base_price * (1 + urth_returns).cumprod()
    combined        = concat(base_series, extension)

  This means:
  - The base series levels are preserved exactly, with no rescaling.
  - The extension inherits the same level as the last base observation.
  - Any level difference between WSJ and URTH (e.g. from dividends or
    index vs ETF tracking) is irrelevant -- only URTH daily returns are
    used, not its absolute level.

TWO PIPELINE MODES
------------------
  First build:
    WSJ CSV (Drive) --> parse --> extend with URTH --> save combined (Drive)

  Subsequent updates (daily runner):
    Combined CSV (Drive) --> extend with URTH --> save combined (Drive)
    The WSJ raw CSV is not re-read; the combined CSV is the new base.

HOW TO RUN FROM IDE
-------------------
  1. Set GDRIVE_FOLDER_ID in USER SETTINGS below (or set the env var).
  2. Ensure credentials.json is in your system temp directory.
  3. Upload your WSJ CSV to Drive with filename WSJ_DRIVE_FILENAME.
  4. Run:  python wsj_msci_world.py

OUTPUTS (Google Drive)
----------------------
  msci_world_combined.csv  --  stooq-format, columns:
                               Data (index, YYYY-MM-DD), Zamkniecie,
                               Najwyzszy, Najnizszy
                               Normalised to 100.0 on the first date.

IMPORTABLE API
--------------
  build_and_upload()           Full pipeline (first build or update).
  extend_with_urth(base, urth) Extend base series using URTH daily returns.
  load_combined_from_drive()   Load pre-built combined CSV from Drive.
  load_wsj_manual_csv(source)  Parse WSJ CSV (path, bytes, or file-like).

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

# Filename of the WSJ CSV you uploaded to Google Drive
WSJ_DRIVE_FILENAME = "msci_world_wsj_raw.csv"

# Output filename in Google Drive (read and written by this script)
COMBINED_DRIVE_FILENAME = "msci_world_combined.csv"

# yfinance ticker for MSCI World ETF extension
# URTH  = iShares MSCI World ETF (USD, from 2012)
# IWDA.L= iShares MSCI World UCITS ETF (USD, LSE-listed, from 2009)
URTH_TICKER          = "URTH"
URTH_FALLBACK_TICKER = "IWDA.L"

# Data floor -- all series clipped to this date
DATA_START = "1990-01-01"

# stooq close column name used throughout the framework
CLOSE_COL = "Zamkniecie"

# Credentials file path (same convention as all other strategy scripts)
CREDENTIALS_PATH = os.path.join(tempfile.gettempdir(), "credentials.json")

# Log file (written when the script runs as __main__)
LOG_FILE = "wsj_msci_world.log"


# ============================================================
# LOGGING
# ============================================================

def _setup_logging() -> None:
    """Configure logging to console + file (only when run as __main__)."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured by a caller runfile
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def _get_folder_id() -> str:
    """Return GDRIVE_FOLDER_ID from env var or USER SETTINGS fallback."""
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        folder_id = GDRIVE_FOLDER_ID_DEFAULT.strip()
    if not folder_id:
        raise ValueError(
            "GDRIVE_FOLDER_ID is not set.\n"
            "  Option A: set GDRIVE_FOLDER_ID_DEFAULT in USER SETTINGS "
            "in wsj_msci_world.py\n"
            "  Option B: set the environment variable GDRIVE_FOLDER_ID "
            "before running"
        )
    return folder_id


def _get_drive_service():
    """Build an authenticated Google Drive service from credentials.json."""
    if not _GDRIVE_AVAILABLE:
        raise ImportError(
            "google-api-python-client / google-auth not installed.\n"
            "  pip install google-api-python-client google-auth"
        )
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(
            f"credentials.json not found at: {CREDENTIALS_PATH}\n"
            "  Place the service account JSON at that path and retry."
        )
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return _gdrive_build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id: str, filename: str):
    """Return the Drive file ID for `filename` in `folder_id`, or None."""
    q = (
        f"name='{filename}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    results = service.files().list(q=q, fields="files(id,name)").execute()
    files   = results.get("files", [])
    return files[0]["id"] if files else None


def _download_bytes_from_drive(service, file_id: str) -> bytes:
    """Download a Drive file by ID and return its raw bytes."""
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
    """Upload a local file to Drive, updating in-place if it exists."""
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
# WSJ CSV PARSING
# ============================================================

def load_wsj_manual_csv(source) -> pd.DataFrame | None:
    """
    Parse a WSJ historical-prices CSV into a stooq-format DataFrame.

    Accepts a file path (str/Path), raw bytes, or any file-like object,
    so it can be called directly with Drive-downloaded bytes.

    WSJ CSV format (as of 2025):
      Date,Open,High,Low,Close,Volume
      01/15/25,3741.00,3748.10,3730.22,3741.88,0
    Dates: MM/DD/YY or MM/DD/YYYY.
    Prices may use commas as thousands separators.

    Returns
    -------
    pd.DataFrame | None  -- stooq-format with CLOSE_COL, Najwyzszy, Najnizszy
    """
    # Read raw bytes from path, bytes, or file-like
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            logging.error("load_wsj_manual_csv: file not found: %s", path)
            return None
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    else:
        raw_bytes = source.read()

    # Decode with fallback encodings (WSJ sometimes writes UTF-8-BOM)
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
        logging.error("load_wsj_manual_csv: could not decode CSV content.")
        return None

    logging.info(
        "load_wsj_manual_csv: raw shape=%s  columns=%s",
        raw.shape, list(raw.columns),
    )

    # Normalise column names
    col_map = {}
    for col in raw.columns:
        c = col.strip().lower()
        if c == "date":
            col_map[col] = "Date"
        elif c == "open":
            col_map[col] = "Open"
        elif c in ("high", "hi"):
            col_map[col] = "High"
        elif c in ("low", "lo"):
            col_map[col] = "Low"
        elif c in ("close", "price", "last", "adj close"):
            col_map[col] = "Close"
    raw = raw.rename(columns=col_map)

    if "Date" not in raw.columns:
        logging.error(
            "load_wsj_manual_csv: no date column found. Columns: %s",
            list(raw.columns),
        )
        return None
    if "Close" not in raw.columns:
        logging.error(
            "load_wsj_manual_csv: no close column found. Columns: %s",
            list(raw.columns),
        )
        return None

    # Parse dates -- handles MM/DD/YY and MM/DD/YYYY
    raw["Date"] = pd.to_datetime(
        raw["Date"].astype(str).str.strip(),
        format="mixed",
        dayfirst=False,
        errors="coerce",
    )
    raw = raw.dropna(subset=["Date"])

    if raw.empty:
        logging.error("load_wsj_manual_csv: no valid dates after parsing.")
        return None

    # Build stooq-format DataFrame
    out = pd.DataFrame(index=raw["Date"].values)
    out.index      = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Data"

    out[CLOSE_COL]   = pd.to_numeric(raw["Close"].values,  errors="coerce")
    out["Najwyzszy"] = (
        pd.to_numeric(raw["High"].values, errors="coerce")
        if "High" in raw.columns else out[CLOSE_COL]
    )
    out["Najnizszy"] = (
        pd.to_numeric(raw["Low"].values, errors="coerce")
        if "Low" in raw.columns else out[CLOSE_COL]
    )

    # Replace zero/negative high/low with close
    out["Najwyzszy"] = out["Najwyzszy"].where(out["Najwyzszy"] > 0, out[CLOSE_COL])
    out["Najnizszy"] = out["Najnizszy"].where(out["Najnizszy"] > 0, out[CLOSE_COL])

    out = (
        out.dropna(subset=[CLOSE_COL])
           .sort_index()
           .loc[lambda df: df.index >= pd.Timestamp(DATA_START)]
    )
    out = out[~out.index.duplicated(keep="last")]

    logging.info(
        "load_wsj_manual_csv: %d rows  %s to %s  close range [%.2f, %.2f]",
        len(out),
        out.index.min().date(), out.index.max().date(),
        out[CLOSE_COL].min(), out[CLOSE_COL].max(),
    )
    return out


# ============================================================
# URTH DOWNLOAD
# ============================================================

def _download_urth() -> pd.DataFrame | None:
    """
    Download URTH (or IWDA.L as fallback) from Yahoo Finance.
    Returns stooq-format DataFrame or None.
    """
    if not _YFINANCE_AVAILABLE:
        logging.error("yfinance not installed.  pip install yfinance")
        return None

    for ticker in (URTH_TICKER, URTH_FALLBACK_TICKER):
        logging.info("Downloading %s from yfinance ...", ticker)
        try:
            raw = yf.download(
                ticker,
                start=DATA_START,
                auto_adjust=True,
                progress=False,
                actions=False,
            )
        except Exception as exc:
            logging.warning("yfinance error for %s: %s", ticker, exc)
            continue

        if raw is None or raw.empty:
            logging.warning("yfinance returned empty data for %s.", ticker)
            continue

        # Flatten MultiIndex (yfinance >= 0.2)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(level=1, axis=1)

        close_cands = [c for c in raw.columns if c.lower() == "close"]
        if not close_cands:
            logging.warning("No 'Close' column in yfinance data for %s.", ticker)
            continue

        high_cands = [c for c in raw.columns if c.lower() == "high"]
        low_cands  = [c for c in raw.columns if c.lower() == "low"]

        out = pd.DataFrame(index=raw.index)
        out.index      = pd.to_datetime(out.index).tz_localize(None)
        out.index.name = "Data"
        out[CLOSE_COL]   = raw[close_cands[0]]
        out["Najwyzszy"] = raw[high_cands[0]] if high_cands else out[CLOSE_COL]
        out["Najnizszy"] = raw[low_cands[0]]  if low_cands  else out[CLOSE_COL]

        out = (
            out.dropna(subset=[CLOSE_COL])
               .sort_index()
               .loc[lambda df: df.index >= pd.Timestamp(DATA_START)]
        )

        logging.info(
            "%s: %d rows  %s to %s",
            ticker, len(out),
            out.index.min().date(), out.index.max().date(),
        )
        return out

    logging.error(
        "Could not download MSCI World proxy from yfinance "
        "(tried %s and %s).", URTH_TICKER, URTH_FALLBACK_TICKER,
    )
    return None


# ============================================================
# EXTENSION  (replaces the old stitch/multiplier logic)
# ============================================================

def extend_with_urth(
    base_df:  pd.DataFrame,
    urth_df:  pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Extend a base price series forward using URTH daily returns.

    The base series (WSJ raw data, or a previously combined series) is
    kept exactly as-is.  URTH is used only to add trading days that fall
    strictly after the last date in the base series.

    Extension method
    ----------------
    Let T  = last date in base_df (the 'anchor' date).
    Let P_T = base_df[CLOSE_COL].iloc[-1]  (anchor price level).

    For each URTH trading date t > T:
        urth_ret_t  = (URTH_close_t / URTH_close_{t-1}) - 1
        P_t         = P_{t-1} * (1 + urth_ret_t)

    where P_{T} = P_T (the base anchor level).

    No overlap analysis, no level ratio, no multiplier.  The extension
    inherits the base level exactly.  Any difference between the WSJ
    price-return index and URTH's total-return NAV is irrelevant because
    only URTH's *daily percentage changes* are used, not its level.

    If URTH has no dates after the base series end, the base series is
    returned unchanged (it is already up to date).

    Parameters
    ----------
    base_df  : pd.DataFrame  -- stooq-format base series (authoritative)
    urth_df  : pd.DataFrame  -- stooq-format URTH data (extension source)

    Returns
    -------
    pd.DataFrame | None  -- combined stooq-format series, or None on error
    """
    if base_df is None or base_df.empty:
        logging.error("extend_with_urth: base_df is None or empty.")
        return None
    if urth_df is None or urth_df.empty:
        logging.error("extend_with_urth: urth_df is None or empty.")
        return None

    base_close = base_df[CLOSE_COL].dropna().sort_index()
    urth_close = urth_df[CLOSE_COL].dropna().sort_index()

    anchor_date  = base_close.index.max()
    anchor_price = float(base_close.iloc[-1])

    # URTH dates strictly after the anchor date
    urth_new = urth_close.loc[urth_close.index > anchor_date]

    if urth_new.empty:
        logging.info(
            "extend_with_urth: base series already up to date "
            "(last date %s, URTH last date %s).  No extension needed.",
            anchor_date.date(), urth_close.index.max().date(),
        )
        return _to_stooq_df(base_close)

    # We need the URTH close on the anchor date (or the last available
    # date before it) to compute the first extension return correctly.
    # pct_change() on the URTH slice will produce NaN on the first new
    # row unless we include one preceding URTH observation as the base.
    #
    # Strategy: take URTH from (anchor_date - buffer) onwards, compute
    # pct_change(), then keep only dates > anchor_date.
    urth_with_anchor = urth_close.loc[
        urth_close.index >= anchor_date
    ]

    if urth_with_anchor.empty:
        # No URTH data at or after anchor -- nothing to extend with
        logging.warning(
            "extend_with_urth: URTH has no data at or after anchor date %s.",
            anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    # Compute URTH daily returns over the slice that starts at anchor_date.
    # The first return (anchor_date itself) becomes NaN from pct_change --
    # that is intentional: we don't overwrite the anchor, we extend from it.
    urth_rets = urth_with_anchor.pct_change()

    # Keep only the returns for dates strictly after anchor
    urth_rets_new = urth_rets.loc[urth_rets.index > anchor_date].dropna()

    if urth_rets_new.empty:
        logging.info(
            "extend_with_urth: no new URTH return dates after %s.",
            anchor_date.date(),
        )
        return _to_stooq_df(base_close)

    # Chain-link: each new price = previous price * (1 + daily_return)
    # Starting anchor is the last base price.
    extension_prices = anchor_price * (1 + urth_rets_new).cumprod()

    # Combine base with extension
    combined = pd.concat([base_close, extension_prices]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    n_new = len(urth_rets_new)
    logging.info(
        "extend_with_urth: base %d rows (%s to %s)  +  %d new rows "
        "(%s to %s)  =  %d total",
        len(base_close),
        base_close.index.min().date(), anchor_date.date(),
        n_new,
        urth_rets_new.index.min().date(),
        urth_rets_new.index.max().date(),
        len(combined),
    )

    return _to_stooq_df(combined)


def _to_stooq_df(close_series: pd.Series) -> pd.DataFrame:
    """
    Wrap a close price Series in a stooq-format DataFrame.
    High and low are set equal to close (no intraday data for this series).
    """
    out = pd.DataFrame({
        CLOSE_COL:   close_series,
        "Najwyzszy": close_series,
        "Najnizszy": close_series,
    })
    out.index.name = "Data"
    return out


# ============================================================
# DRIVE I/O -- COMBINED CSV
# ============================================================

def load_combined_from_drive(
    service=None,
    folder_id: str | None = None,
    filename:  str = COMBINED_DRIVE_FILENAME,
) -> pd.DataFrame | None:
    """
    Download the pre-built combined MSCI World CSV from Google Drive.

    Called by the runfile and diagnostic on every run after the combined
    series has been built at least once.

    Parameters
    ----------
    service   : Drive service object (None = build from credentials)
    folder_id : str | None  (None = from USER SETTINGS / env var)
    filename  : str

    Returns
    -------
    pd.DataFrame | None  -- stooq-format combined series
    """
    if service is None:
        service = _get_drive_service()
    if folder_id is None:
        folder_id = _get_folder_id()

    file_id = _find_file_id(service, folder_id, filename)
    if file_id is None:
        logging.warning(
            "load_combined_from_drive: '%s' not found in Drive folder %s.",
            filename, folder_id,
        )
        return None

    raw_bytes = _download_bytes_from_drive(service, file_id)
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
        df = df.sort_index().dropna(subset=[CLOSE_COL])
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info(
            "load_combined_from_drive: %d rows  %s to %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df
    except Exception as exc:
        logging.error("load_combined_from_drive: parse error -- %s", exc)
        return None


def _download_wsj_csv_from_drive(
    service,
    folder_id: str,
    filename:  str = WSJ_DRIVE_FILENAME,
) -> bytes | None:
    """Download the raw WSJ CSV bytes from Drive.  Returns None if absent."""
    file_id = _find_file_id(service, folder_id, filename)
    if file_id is None:
        logging.error(
            "WSJ CSV '%s' not found in Drive folder %s.\n"
            "  Download from: "
            "https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices\n"
            "  Upload to Drive with filename: %s",
            filename, folder_id, filename,
        )
        return None
    raw = _download_bytes_from_drive(service, file_id)
    logging.info(
        "_download_wsj_csv_from_drive: '%s'  (%d bytes)", filename, len(raw)
    )
    return raw


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_and_upload(
    folder_id:         str | None = None,
    wsj_filename:      str = WSJ_DRIVE_FILENAME,
    combined_filename: str = COMBINED_DRIVE_FILENAME,
) -> pd.DataFrame | None:
    """
    Full pipeline -- builds or updates the combined MSCI World series.

    Two modes, selected automatically:

    First build (combined CSV does not yet exist on Drive):
      1. Download WSJ raw CSV from Drive
      2. Parse it into base series
      3. Download URTH from yfinance
      4. Extend base series with URTH daily returns (dates after WSJ end)
      5. Save combined CSV to Drive

    Update (combined CSV already exists on Drive):
      1. Download combined CSV from Drive (this is now the base)
      2. Download URTH from yfinance
      3. Extend combined series with URTH daily returns (new dates only)
      4. Save updated combined CSV back to Drive

    In both modes, WSJ data is never modified or rescaled.  Only URTH
    daily percentage returns are applied to extend the series forward.

    Parameters
    ----------
    folder_id         : str | None  (None = from USER SETTINGS / env var)
    wsj_filename      : str  -- Drive filename for the WSJ raw CSV
    combined_filename : str  -- Drive filename for the output combined CSV

    Returns
    -------
    pd.DataFrame | None  -- the combined stooq-format series
    """
    if folder_id is None:
        folder_id = _get_folder_id()

    logging.info("=" * 70)
    logging.info("WSJ MSCI WORLD  --  BUILD & UPLOAD")
    logging.info("  Drive folder : %s", folder_id)
    logging.info("  WSJ source   : %s", wsj_filename)
    logging.info("  Output file  : %s", combined_filename)
    logging.info("  Credentials  : %s", CREDENTIALS_PATH)
    logging.info("=" * 70)

    # Authenticate
    try:
        service = _get_drive_service()
    except Exception as exc:
        logging.error("Drive authentication failed: %s", exc)
        return None

    # Determine base series: prefer combined (update mode) over WSJ raw (first build)
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
        # No combined yet -- build from WSJ raw
        logging.info("Mode: FIRST BUILD  (combined CSV not found on Drive)")
        wsj_bytes = _download_wsj_csv_from_drive(service, folder_id, wsj_filename)
        if wsj_bytes is None:
            return None
        base_df = load_wsj_manual_csv(wsj_bytes)
        if base_df is None:
            logging.error("Failed to parse WSJ CSV.")
            return None
        mode_str = "FIRST BUILD  (base = WSJ raw CSV)"
        logging.info(
            "WSJ base: %d rows  %s to %s",
            len(base_df),
            base_df.index.min().date(), base_df.index.max().date(),
        )

    # Download URTH for extension
    urth_df = _download_urth()
    if urth_df is None:
        logging.error("Failed to download URTH from yfinance.")
        return None

    # Extend base with URTH
    combined = extend_with_urth(base_df, urth_df)
    if combined is None:
        logging.error("Extension failed.")
        return None

    # Save locally then upload to Drive
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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _setup_logging()

    logging.info("wsj_msci_world.py  started  %s", dt.datetime.now())
    logging.info("Credentials path : %s", CREDENTIALS_PATH)
    logging.info("Credentials exist: %s", os.path.exists(CREDENTIALS_PATH))

    result = build_and_upload()

    if result is not None:
        print(
            f"\nCombined MSCI World series ready.\n"
            f"  {len(result)} trading days\n"
            f"  {result.index.min().date()} to {result.index.max().date()}\n"
            f"  Uploaded to Drive as: {COMBINED_DRIVE_FILENAME}\n"
            f"  Local copy:           "
            f"{os.path.join(tempfile.gettempdir(), COMBINED_DRIVE_FILENAME)}\n"
        )
    else:
        print(
            "\nBuild failed.  Check wsj_msci_world.log for details.\n\n"
            "Likely causes:\n"
            "  1. GDRIVE_FOLDER_ID not set -- edit USER SETTINGS above\n"
            f"  2. credentials.json missing at {CREDENTIALS_PATH}\n"
            f"  3. WSJ CSV not uploaded -- upload '{WSJ_DRIVE_FILENAME}' to Drive\n"
            "  4. yfinance not installed -- pip install yfinance\n"
        )
        sys.exit(1)