# -*- coding: utf-8 -*-
"""
moj_system/reporting/output_base.py
====================================
Base module for daily output generation.
Replaces legacy `current_development/daily_output_base.py`.
"""

import datetime as dt
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

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
        df = pd.read_csv(
            log_path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            skipinitialspace=True,
        )

        if df.empty:
            return None

        # Clean column names (strip spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:
        logging.warning(f"load_existing_log: could not read {log_path} - {exc}")
        return None


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """Append row to signal_log.csv with robust overwrite for Windows."""
    new_row_df = pd.DataFrame([row])

    if log_path.exists():
        try:
            existing = pd.read_csv(log_path)
            existing = existing[existing["date"] != row["date"]]
            combined = pd.concat([existing, new_row_df], ignore_index=True)
        except Exception:
            # Jeśli plik jest uszkodzony, zacznij od nowa
            combined = new_row_df
    else:
        combined = new_row_df

    tmp = log_path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)

    # PANCERNE NADPISYWANIE:
    # Path.replace delegates to the same atomic replacement operation.
    if log_path.exists():
        tmp.replace(log_path)
    else:
        tmp.rename(log_path)


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


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations of the four daily artefacts of one strategy."""

    log_path: Path
    status_path: Path
    chart_path: Path
    snapshot_path: Path

    @property
    def log_filename(self) -> str:
        return self.log_path.name

    def upload_candidates(self) -> list[Path]:
        return [self.log_path, self.status_path, self.chart_path, self.snapshot_path]


def resolve_run_date(
    run_date: dt.date | str | None,
) -> dt.date:
    """Normalise an optional run date (date, ISO string or None) to a date."""
    if isinstance(run_date, str):
        try:
            return dt.date.fromisoformat(run_date)
        except ValueError:
            return dt.date.today()
    if run_date is None:
        return dt.date.today()
    return run_date


def resolve_artifact_paths(
    output_dir: str,
    asset_name: str,
) -> ArtifactPaths:
    """Create ``output_dir`` and return the artefact paths derived from ``asset_name``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = asset_name.lower()
    return ArtifactPaths(
        log_path=out_dir / f"{prefix}_signal_log.csv",
        status_path=out_dir / f"{prefix}_signal_status.txt",
        chart_path=out_dir / f"{prefix}_equity_chart.png",
        snapshot_path=out_dir / f"{prefix}_signal_snapshot.json",
    )


def upload_artifacts_to_drive(
    artifact_paths: Sequence[Path],
    gdrive_folder_id: str | None,
    gdrive_credentials: str | None,
) -> None:
    """Upload existing artefacts to Google Drive; failures are logged, never raised."""
    if not (gdrive_folder_id and gdrive_credentials):
        logging.info("No GDrive credentials provided. Artefacts saved locally only.")
        return

    logging.info("Uploading artefacts to Google Drive...")
    try:
        client = GDriveClient(credentials_path=gdrive_credentials)
        if not client.service:
            logging.warning("Drive service unavailable. Artefacts saved locally only.")
            return
        for file_path in artifact_paths:
            if file_path.exists():
                # upload_csv is mime-agnostic and also handles .txt/.png artefacts.
                client.upload_csv(gdrive_folder_id, str(file_path), file_path.name)
        logging.info("Successfully uploaded all daily artefacts to Google Drive.")
    except Exception as exc:
        logging.error(f"Failed to upload artefacts to Drive: {exc}")


def save_figure_atomically(
    figure: Figure,
    chart_path: Path,
    dpi: int,
) -> None:
    """Render ``figure`` to a temp PNG, close it and atomically move it to ``chart_path``."""
    import matplotlib.pyplot as plt

    buffer = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buffer.name, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    buffer.close()
    temp_path = Path(buffer.name)
    atomic_write_bytes(chart_path, temp_path.read_bytes())
    temp_path.unlink()
