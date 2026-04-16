"""
daily_output_base.py
====================
Shared infrastructure for the three daily output modules:
  daily_output.py              (WIG20TR / generic single-asset)
  multiasset_daily_output.py   (WIG + TBSP + MMF)
  global_equity_daily_output.py (N-asset global equity)

Provides:
  atomic_write()            — safe text file write (write-temp-rename)
  atomic_write_bytes()      — safe binary file write
  load_existing_log()       — load a signal log CSV, return DataFrame or None
  append_log_row()          — append one dict as a CSV row (creates with header)
  fetch_file_from_drive()   — download a single named file from a Drive folder

These five utilities were previously duplicated verbatim across the three
output modules.  Any change to Drive authentication, file-write safety,
or CSV handling now only needs to be made here.

All functions are intentionally free of strategy-specific logic and have
no dependencies on the strategy libraries.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Google Drive imports — optional; only required when gdrive_folder_id supplied.
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload
    import io as _io
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------

def atomic_write(path: Path, content: str) -> None:
    """
    Write text content to path atomically.

    Writes to a .tmp sibling file then renames, so a mid-write crash never
    leaves a corrupt or zero-length file at the target path.

    Parameters
    ----------
    path    : Path — destination file path
    content : str  — text content to write (UTF-8)
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Write binary data to path atomically (write-temp-rename).

    Parameters
    ----------
    path : Path  — destination file path
    data : bytes — binary content to write
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Signal log helpers
# ---------------------------------------------------------------------------

def load_existing_log(log_path: Path) -> pd.DataFrame | None:
    """
    Load an existing signal log CSV.

    Returns the DataFrame on success, or None if the file does not exist,
    is empty, or cannot be parsed.  Callers use None to detect first-run
    or post-reset state when determining action (ENTER / EXIT / HOLD).

    Parameters
    ----------
    log_path : Path — path to the signal log CSV

    Returns
    -------
    pd.DataFrame | None
    """
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
        return df if not df.empty else None
    except Exception as exc:
        logging.warning("load_existing_log: could not read %s — %s", log_path, exc)
        return None


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """
    Append one row to a signal log CSV.

    Creates the file with a header row if it does not yet exist or is empty.
    The column order is determined by the key order of `row` (Python 3.7+
    dict insertion order).

    Parameters
    ----------
    log_path : Path           — path to the signal log CSV
    row      : dict[str, Any] — flat dict of column → value for this run date
    """
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Google Drive file download
# ---------------------------------------------------------------------------

def fetch_file_from_drive(
    local_path: Path,
    folder_id: str,
    filename: str,
    credentials_path: str,
) -> bool:
    """
    Download a single named file from a Google Drive folder into local_path.

    Called at the start of build_daily_outputs() so action determination
    (ENTER / EXIT / HOLD) always uses the authoritative Drive copy of the
    signal log rather than the empty ephemeral GitHub Actions workspace.

    Behaviour on file-not-found:
      Logs an INFO message and returns False — this is the expected state on
      the very first run before any log has been uploaded to Drive.

    Parameters
    ----------
    local_path       : Path — where to write the downloaded bytes
    folder_id        : str  — Google Drive folder ID
    filename         : str  — exact filename to search for in that folder
    credentials_path : str  — path to a service account credentials JSON

    Returns
    -------
    bool — True if the file was downloaded successfully, False otherwise.
           Callers should treat False as "no prior log available" rather
           than a fatal error.
    """
    if not _GDRIVE_AVAILABLE:
        logging.warning(
            "fetch_file_from_drive: google-api packages not installed — "
            "skipping Drive download.  "
            "pip install google-auth google-api-python-client"
        )
        return False

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = _gdrive_build("drive", "v3", credentials=creds)

        q = (
            f"name='{filename}' "
            f"and '{folder_id}' in parents "
            f"and trashed=false"
        )
        results = service.files().list(
            q=q, spaces="drive", fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if not files:
            logging.info(
                "fetch_file_from_drive: '%s' not found in Drive folder %s — "
                "expected on first ever run.",
                filename, folder_id,
            )
            return False

        file_id = files[0]["id"]
        request = service.files().get_media(fileId=file_id)
        buf = _io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(buf.getvalue())
        logging.info(
            "fetch_file_from_drive: downloaded '%s' (%d bytes).",
            filename, len(buf.getvalue()),
        )
        return True

    except Exception as exc:
        logging.error(
            "fetch_file_from_drive: failed to download '%s' — %s. "
            "Action determination will fall back to local file.",
            filename, exc,
        )
        return False
