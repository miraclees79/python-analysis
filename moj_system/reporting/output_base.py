# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:22:12 2026

@author: adamg
"""

"""
moj_system/reporting/output_base.py
====================================
Nowa baza dla wszystkich modułów generujących output dzienny.
Zastępuje stary plik `current_development/daily_output_base.py`.

Korzysta ze zunifikowanego klienta `moj_system.data.gdrive.GDriveClient`.
"""

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Import naszego nowego klienta GDrive
try:
    from moj_system.data.gdrive import GDriveClient
    _GDRIVE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Brak modułu GDriveClient: {e}. Upewnij się, że paczka jest poprawnie zainstalowana.")
    _GDRIVE_AVAILABLE = False


def atomic_write(path: Path, content: str) -> None:
    """Zapisuje tekst do pliku atomowo (z użyciem pliku tymczasowego .tmp)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Zapisuje bajty do pliku atomowo (z użyciem pliku tymczasowego .tmp)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def load_existing_log(log_path: Path) -> pd.DataFrame | None:
    """Wczytuje istniejący plik CSV z logami sygnałów."""
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
        return df if not df.empty else None
    except Exception as exc:
        logging.warning(f"load_existing_log: nie można odczytać {log_path} — {exc}")
        return None


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """Dopisuje pojedynczy wiersz do logu CSV (tworzy plik z nagłówkami, jeśli nie istnieje)."""
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
    """
    Pobiera plik z Google Drive, korzystając z moj_system.data.gdrive.GDriveClient.
    """
    if not _GDRIVE_AVAILABLE:
        logging.warning("fetch_file_from_drive: Moduł GDriveClient niedostępny — pomijam pobieranie z Drive.")
        return False

    try:
        client = GDriveClient(credentials_path=credentials_path)
        if not client.service:
            logging.info("Brak autoryzacji GDrive (tryb offline).")
            return False

        # Używamy metody pobierania DataFrame z naszego klienta
        df = client.download_csv(folder_id, filename)

        if df is None:
            logging.info(f"fetch_file_from_drive: Pliku '{filename}' nie ma na Drive (oczekiwane przy 1. uruchomieniu).")
            return False

        # Zapis do lokalnej ścieżki
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # GDriveClient.download_csv zwraca df, więc zapisujemy go lokalnie jako CSV bez indexu
        df.to_csv(local_path, index=False) 
        
        logging.info(f"fetch_file_from_drive: pobrano '{filename}' i nadpisano lokalny log: {local_path}")
        return True

    except Exception as exc:
        logging.error(f"fetch_file_from_drive: błąd pobierania '{filename}' — {exc}. Akcja użyje pliku lokalnego.")
        return False