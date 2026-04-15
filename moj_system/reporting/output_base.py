

# -*- coding: utf-8 -*-
"""
moj_system/reporting/output_base.py
====================================
Base module for daily output generation.
Replaces legacy `current_development/daily_output_base.py`.
"""

import csv
import logging
from pathlib import Path
from typing import Any
import pandas as pd

# Import GDriveClient
try:
    from moj_system.data.gdrive import GDriveClient
    _GDRIVE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GDriveClient module not available: {e}")
    _GDRIVE_AVAILABLE = False


def atomic_write(path: Path, content: str) -> None:
    """Writes text to file atomically (overwrites on Windows)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    # Using replace() instead of rename() to avoid FileExistsError on Windows
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Writes bytes to file atomically (overwrites on Windows)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    # Using replace() instead of rename() to avoid FileExistsError on Windows
    tmp.replace(path)


def load_existing_log(log_path: Path) -> pd.DataFrame | None:
    """Loads existing CSV log with header sanitization."""
    if not log_path.exists():
        return None
    try:
        # Load with automatic delimiter detection and UTF-8-SIG for BOM support
        df = pd.read_csv(log_path, sep=None, engine='python', encoding='utf-8-sig', skipinitialspace=True)
        
        if df.empty:
            return None
            
        # Clean column names (strip spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:
        logging.warning(f"load_existing_log: could not read {log_path} - {exc}")
        return None


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """Appends a single row to CSV log (creates with headers if needed)."""
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fetch_file_from_drive(
    local_path: Path,
    folder_id: str,
    filename: str,
    credentials_path: str = None,
) -> bool:
    """Downloads file from GDrive and ensures local CSV consistency."""
    if not _GDRIVE_AVAILABLE:
        return False

    try:
        client = GDriveClient(credentials_path=credentials_path)
        if not client.service:
            return False

        # Force comma separator for consistency
        df = client.download_csv(folder_id, filename, sep=",")

        if df is None:
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Save locally using comma as delimiter
        df.to_csv(local_path, index=False, sep=",") 
        
        logging.info(f"fetch_file_from_drive: downloaded and unified log: {local_path}")
        return True

    except Exception as exc:
        logging.error(f"fetch_file_from_drive error: {exc}")
        return False