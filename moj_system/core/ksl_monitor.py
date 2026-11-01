# -*- coding: utf-8 -*-
"""
moj_system/core/ksl_monitor.py
==============================
Logika monitoringu pozycji KSL niezależna od Streamlit: katalog instrumentów,
odświeżanie danych z Google Drive do katalogu ulotnego, obliczenia i odczyt/zapis
konfiguracji pozycji. Korzystają z niej zarówno dashboard, jak i raport dzienny.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from moj_system.config import DATA_DIR as REPO_DATA_DIR
from moj_system.data.gdrive import GDriveClient
from moj_system.data.ppe_manager import combine_ppe_history

POSITIONS_FILENAME = "ksl_positions.json"
CONFIG_SCHEMA_VERSION = 1
STALE_AFTER_DAYS = 7
REFRESH_STATUS_FILENAME = "refresh_status.json"
PPE_COMBINED_FILENAME = "ppe_combined.csv"

# Podfolder z historiami funduszy/ETF wgrywanymi przez DataUpdater (updater.GDRIVE_DATA_FOLDER_NAME).
DRIVE_DATA_FOLDER_NAME = "Dane"

PPE_INSTRUMENTS: dict[str, tuple[str, str]] = {
    "ppe:bezpieczna_jesien:equity": ("PPE — Bezpieczna Jesień — Akcyjny", "equity"),
    "ppe:bezpieczna_jesien:balanced": ("PPE — Bezpieczna Jesień — Zrównoważony", "balanced"),
    "ppe:bezpieczna_jesien:stable": ("PPE — Bezpieczna Jesień — Stabilnego Wzrostu", "stable"),
    "ppe:bezpieczna_jesien:bond": ("PPE — Bezpieczna Jesień — Obligacji", "bond"),
    "ppe:bezpieczna_jesien:mmf": ("PPE — Bezpieczna Jesień — Konserwatywny", "mmf"),
}

ETF_LABELS = {
    "etfbw20tr": "Beta ETF WIG20TR",
    "etfbm40tr": "Beta ETF mWIG40TR",
    "etfbs80tr": "Beta ETF sWIG80TR",
    "etfbtbsp": "Beta ETF TBSP",
}

_FUND_ID_RE = re.compile(r"^stooq:fund:(\d+)\.n$")
_ETF_ID_RE = re.compile(r"^stooq:etf:([a-z0-9]+)\.pl$")


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    label: str
    source: str
    source_ticker: str
    data_file: Path
    price_column: str


@dataclass(frozen=True)
class RefreshResult:
    instrument_id: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PositionResult:
    position: dict[str, Any]
    instrument: Instrument | None
    metrics: dict[str, Any] | None
    error: str
    status: str


def default_data_dir() -> Path:
    """Ulotny katalog roboczy na dane pobrane z Drive (KSL_DATA_DIR albo katalog tymczasowy)."""
    configured = os.environ.get("KSL_DATA_DIR", "").strip()
    path = Path(configured) if configured else Path(tempfile.gettempdir()) / "ksl_dashboard_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_stooq_fund_names() -> dict[str, str]:
    names: dict[str, str] = {}
    names_dir = REPO_DATA_DIR.parent / "fund_names_stooq"
    for path in names_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            match = re.match(r"^\s*(\d+)\.N\s+(.+?)\s*$", line, flags=re.IGNORECASE)
            if match:
                names[match.group(1)] = match.group(2).strip()
    return names


def instrument_from_id(
    instrument_id: str,
    data_dir: Path,
    stooq_names: dict[str, str] | None = None,
) -> Instrument | None:
    """Odtwarza opis instrumentu ze stabilnego id; plik danych nie musi jeszcze istnieć."""
    if instrument_id in PPE_INSTRUMENTS:
        label, column = PPE_INSTRUMENTS[instrument_id]
        return Instrument(instrument_id, label, "ppe", column, data_dir / PPE_COMBINED_FILENAME, column)

    fund_match = _FUND_ID_RE.match(instrument_id)
    if fund_match:
        fund_id = fund_match.group(1)
        names = stooq_names if stooq_names is not None else _read_stooq_fund_names()
        return Instrument(
            instrument_id,
            names.get(fund_id, f"Fundusz Stooq {fund_id}.n"),
            "stooq",
            f"{fund_id}.n",
            data_dir / f"fund_{fund_id}.csv",
            "Zamkniecie",
        )

    etf_match = _ETF_ID_RE.match(instrument_id)
    if etf_match:
        symbol = etf_match.group(1)
        return Instrument(
            instrument_id,
            ETF_LABELS.get(symbol, f"ETF Stooq {symbol.upper()}.PL"),
            "stooq",
            f"{symbol}.pl",
            data_dir / f"{symbol}.csv",
            "Zamkniecie",
        )
    return None


def drive_filename(instrument: Instrument) -> str | None:
    """Nazwa pliku z historią w podfolderze danych na Drive (zgodna z DataUpdater.update_ticker)."""
    if instrument.source != "stooq":
        return None
    if instrument.instrument_id.startswith("stooq:fund:"):
        return f"historia{instrument.source_ticker[:4]}.csv"
    return f"historia{instrument.source_ticker}.csv"


def list_drive_instrument_ids(client: GDriveClient, folder_id: str) -> set[str]:
    """Instrumenty Stooq, których historie leżą w podfolderze danych na Drive."""
    service: Any = client.service
    data_folder_id = client.find_file_id(parent_id=folder_id, filename=DRIVE_DATA_FOLDER_NAME)
    if not service or not data_folder_id:
        return set()
    ids: set[str] = set()
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{data_folder_id}' in parents and trashed=false and name contains 'historia'",
                fields="nextPageToken,files(name)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute(num_retries=5)
        )
        for item in response.get("files", []):
            name = str(item["name"]).lower()
            fund = re.fullmatch(r"historia(\d+)\.csv", name)
            etf = re.fullmatch(r"historia(etf[a-z0-9]+)\.pl\.csv", name)
            if fund:
                ids.add(f"stooq:fund:{fund.group(1)}.n")
            elif etf:
                ids.add(f"stooq:etf:{etf.group(1)}.pl")
        page_token = response.get("nextPageToken")
        if not page_token:
            return ids


def get_instrument_catalog(
    data_dir: Path | None = None,
    extra_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Instrument]:
    """Katalog do wyboru: PPE, pliki obecne w data_dir oraz instrumenty znane z Drive (extra_ids)."""
    data_dir = data_dir or default_data_dir()
    names = _read_stooq_fund_names()
    ids: set[str] = set(PPE_INSTRUMENTS) | set(extra_ids)
    for path in data_dir.glob("fund_*.csv"):
        match = re.fullmatch(r"fund_(\d+)", path.stem)
        if match:
            ids.add(f"stooq:fund:{match.group(1)}.n")
    for path in data_dir.glob("etf*.csv"):
        ids.add(f"stooq:etf:{path.stem.lower()}.pl")

    catalog: dict[str, Instrument] = {}
    for instrument_id in ids:
        instrument = instrument_from_id(instrument_id, data_dir, names)
        if instrument is not None:
            catalog[instrument_id] = instrument
    return catalog


def load_price_series(instrument: Instrument) -> pd.Series:
    """Wczytuje oczyszczoną dzienną serię cen/NAV jednego instrumentu."""
    sep = ";" if instrument.source == "ppe" else ","
    frame = pd.read_csv(instrument.data_file, sep=sep, encoding="utf-8-sig")
    date_column = "Data"
    if date_column not in frame.columns or instrument.price_column not in frame.columns:
        raise ValueError(
            f"Brak wymaganych kolumn {date_column!r} i {instrument.price_column!r} "
            f"w pliku {instrument.data_file.name}."
        )

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[instrument.price_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name="price")
    series = series[series.notna() & (series > 0)]
    series = series.loc[series.index.dropna()]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError(f"Brak poprawnych notowań w pliku {instrument.data_file.name}.")
    return series


def calculate_position_metrics(
    prices: pd.Series,
    selected_entry_date: date,
    ksl_pct: float,
    warning_pct: float,
) -> dict[str, Any]:
    """Zwrot od wejścia i obsunięcie od maksimum; wejście = pierwsze notowanie w dniu lub po nim."""
    if not 0 < warning_pct < ksl_pct < 1:
        raise ValueError("Progi muszą spełniać: 0 < warning < KSL < 1.")
    if prices.empty:
        raise ValueError("Brak dostępnych notowań.")

    series = prices.copy()
    series.index = pd.to_datetime(series.index)
    series = series.dropna().sort_index()
    selected_timestamp = pd.Timestamp(selected_entry_date)
    entry_prices = series.loc[series.index >= selected_timestamp]
    if entry_prices.empty:
        raise ValueError("Brak notowań w wybranym dniu i po nim.")

    entry_date = entry_prices.index[0]
    entry_price = float(entry_prices.iloc[0])
    latest_date = series.index[-1]
    latest_price = float(series.iloc[-1])
    period = series.loc[(series.index >= entry_date) & (series.index <= latest_date)]
    peak_price = float(period.max())
    drawdown_pct = latest_price / peak_price - 1

    ksl_status = "ALERT" if drawdown_pct <= -ksl_pct else "OK"
    warning_status = "ALERT" if drawdown_pct <= -warning_pct else "OK"
    status = "ALERT" if ksl_status == "ALERT" else ("WARNING" if warning_status == "ALERT" else "OK")

    prior_prices = series.loc[series.index < latest_date]
    daily_return_pct = latest_price / float(prior_prices.iloc[-1]) - 1 if not prior_prices.empty else None
    return {
        "selected_entry_date": selected_timestamp.date().isoformat(),
        "effective_entry_date": entry_date.date().isoformat(),
        "entry_price": entry_price,
        "latest_date": latest_date.date().isoformat(),
        "latest_price": latest_price,
        "peak_since_entry": peak_price,
        "return_since_entry_pct": latest_price / entry_price - 1,
        "drawdown_from_peak_pct": drawdown_pct,
        "daily_return_pct": daily_return_pct,
        "ksl_level": ksl_pct,
        "warning_level": warning_pct,
        "ksl_status": ksl_status,
        "warning_status": warning_status,
        "status": status,
        "is_stale": (date.today() - latest_date.date()).days > STALE_AFTER_DAYS,
        "price_history": period,
    }


def validate_positions_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(config, dict):
        raise ValueError("Konfiguracja pozycji musi być obiektem JSON.")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Nieobsługiwana wersja formatu pliku pozycji.")
    positions = config.get("positions")
    if not isinstance(positions, list):
        raise ValueError("Pole 'positions' musi być listą.")

    position_ids: set[str] = set()
    active_keys: set[tuple[str, str]] = set()
    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("Każda pozycja musi być obiektem JSON.")
        required = (
            "position_id",
            "account_id",
            "account_label",
            "instrument_id",
            "entry_date",
            "ksl_pct",
            "warning_pct",
        )
        if any(key not in position for key in required):
            raise ValueError(f"Pozycja nie zawiera wszystkich pól: {required}.")
        position_id = str(position["position_id"]).strip()
        account_id = str(position["account_id"]).strip()
        instrument_id = str(position["instrument_id"]).strip()
        if not position_id or not account_id or not instrument_id or not position["account_label"]:
            raise ValueError("Identyfikatory oraz nazwa rachunku nie mogą być puste.")
        if position_id in position_ids:
            raise ValueError(f"Powtórzony position_id: {position_id}.")
        position_ids.add(position_id)
        if position.get("enabled", True):
            key = (account_id, instrument_id)
            if key in active_keys:
                raise ValueError("Dozwolona jest jedna aktywna pozycja na parę rachunek + instrument.")
            active_keys.add(key)
        try:
            date.fromisoformat(str(position["entry_date"]))
            ksl_pct = float(position["ksl_pct"])
            warning_pct = float(position["warning_pct"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Niepoprawna data lub próg w pozycji {position_id}.") from exc
        if not 0 < warning_pct < ksl_pct < 1:
            raise ValueError(f"Pozycja {position_id}: progi muszą spełniać 0 < warning < KSL < 1.")
    return positions


def active_positions(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in validate_positions_config(config) if p.get("enabled", True)]


# --- Google Drive ---------------------------------------------------------------------


def get_drive_client() -> tuple[GDriveClient, str]:
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        raise RuntimeError("Brak zmiennej środowiskowej GDRIVE_FOLDER_ID.")
    credentials_path = str(Path(tempfile.gettempdir()) / "credentials.json")
    client = GDriveClient(credentials_path=credentials_path)
    if not client.service:
        raise RuntimeError(
            "Nie można uwierzytelnić się w Google Drive. Skonfiguruj GOOGLE_CREDENTIALS "
            "lub credentials.json oraz dostęp konta usługi do folderu."
        )
    return client, folder_id


def read_positions_from_drive(
    client: GDriveClient | None = None,
    folder_id: str | None = None,
) -> dict[str, Any]:
    if client is None or folder_id is None:
        client, folder_id = get_drive_client()
    with tempfile.TemporaryDirectory(prefix="ksl_dashboard_") as temp_dir:
        local_path = Path(temp_dir) / POSITIONS_FILENAME
        if not client.download_file(folder_id, POSITIONS_FILENAME, local_path):
            raise FileNotFoundError(f"Nie znaleziono {POSITIONS_FILENAME} w folderze GDRIVE_FOLDER_ID.")
        config = json.loads(local_path.read_text(encoding="utf-8"))
    validate_positions_config(config)
    return config


def write_positions_to_drive(config: dict[str, Any]) -> None:
    validate_positions_config(config)
    client, folder_id = get_drive_client()
    with tempfile.TemporaryDirectory(prefix="ksl_dashboard_") as temp_dir:
        local_path = Path(temp_dir) / POSITIONS_FILENAME
        local_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        file_id = client.upload_file(folder_id, str(local_path), POSITIONS_FILENAME)
    if not file_id:
        raise RuntimeError(f"Zapis {POSITIONS_FILENAME} na Google Drive nie powiódł się.")


# --- Odświeżanie danych ---------------------------------------------------------------


def _atomic_write_csv(frame: pd.DataFrame, target: Path, sep: str) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(tmp, sep=sep)
    tmp.replace(target)


def _refresh_ppe(
    instrument_ids: list[str],
    data_dir: Path,
    client: GDriveClient,
    folder_id: str,
) -> list[RefreshResult]:
    try:
        hist_df = client.download_csv(folder_id=folder_id, filename="full_ppe_history.csv", sep=",")
        ocr_df = client.download_csv(folder_id=folder_id, filename="filtered_table.csv", sep=";")
        if hist_df is None or hist_df.empty:
            raise ValueError("Brak full_ppe_history.csv na Drive.")
        combined = combine_ppe_history(hist_df=hist_df, ocr_df=ocr_df)
        if combined.empty:
            raise ValueError("Połączona historia PPE jest pusta.")
        _atomic_write_csv(combined, data_dir / PPE_COMBINED_FILENAME, sep=";")
        detail = f"PPE do {combined.index.max().date()}" + ("" if ocr_df is not None else " (bez OCR)")
        return [RefreshResult(i, True, detail) for i in instrument_ids]
    except Exception as exc:  # noqa: BLE001 - błąd jednego źródła nie może zatrzymać pozostałych
        logging.warning("Odświeżenie PPE nie powiodło się: %s", exc)
        return [RefreshResult(i, False, f"PPE: {exc}") for i in instrument_ids]


def refresh_instruments(
    instrument_ids: list[str] | set[str],
    data_dir: Path,
    client: GDriveClient,
    folder_id: str,
) -> list[RefreshResult]:
    """Pobiera z Drive dane tylko wskazanych instrumentów do (ulotnego) katalogu data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(instrument_ids))
    results: list[RefreshResult] = []

    ppe_ids = [i for i in ids if i in PPE_INSTRUMENTS]
    if ppe_ids:
        results.extend(_refresh_ppe(ppe_ids, data_dir, client, folder_id))

    stooq_names = _read_stooq_fund_names()
    data_folder_id: str | None = None
    for instrument_id in ids:
        if instrument_id in PPE_INSTRUMENTS:
            continue
        instrument = instrument_from_id(instrument_id, data_dir, stooq_names)
        file_name = drive_filename(instrument) if instrument else None
        if instrument is None or file_name is None:
            results.append(RefreshResult(instrument_id, False, "Nieznany instrument_id."))
            continue
        if data_folder_id is None:
            data_folder_id = client.find_file_id(parent_id=folder_id, filename=DRIVE_DATA_FOLDER_NAME)
        if not data_folder_id:
            results.append(
                RefreshResult(instrument_id, False, f"Brak folderu {DRIVE_DATA_FOLDER_NAME} na Drive.")
            )
            continue

        tmp_path = instrument.data_file.with_suffix(".csv.tmp")
        try:
            if not client.download_file(data_folder_id, file_name, tmp_path):
                raise FileNotFoundError(f"Brak pliku {file_name} na Drive.")
            series = load_price_series(replace(instrument, data_file=tmp_path))
            tmp_path.replace(instrument.data_file)
            results.append(RefreshResult(instrument_id, True, f"do {series.index.max().date()}"))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            logging.warning("Odświeżenie %s nie powiodło się: %s", instrument_id, exc)
            tmp_path.unlink(missing_ok=True)
            results.append(RefreshResult(instrument_id, False, str(exc)))

    write_refresh_status(data_dir, results)
    return results


def write_refresh_status(data_dir: Path, results: list[RefreshResult]) -> None:
    payload = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "results": [{"instrument_id": r.instrument_id, "ok": r.ok, "detail": r.detail} for r in results],
    }
    (data_dir / REFRESH_STATUS_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_refresh_status(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / REFRESH_STATUS_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def refresh_is_due(data_dir: Path, instrument_ids: set[str], max_age_hours: float = 6.0) -> bool:
    """Czy trzeba pobrać dane: brak statusu, status za stary albo brakuje któregoś pliku."""
    status = read_refresh_status(data_dir)
    if status is None:
        return True
    try:
        refreshed_at = datetime.fromisoformat(status["refreshed_at"])
    except (KeyError, ValueError):
        return True
    if (datetime.now(timezone.utc) - refreshed_at).total_seconds() > max_age_hours * 3600:
        return True
    names = _read_stooq_fund_names()
    for instrument_id in instrument_ids:
        instrument = instrument_from_id(instrument_id, data_dir, names)
        if instrument is not None and not instrument.data_file.is_file():
            return True
    return False


# --- Ocena pozycji --------------------------------------------------------------------


def evaluate_positions(positions: list[dict[str, Any]], data_dir: Path) -> list[PositionResult]:
    names = _read_stooq_fund_names()
    results: list[PositionResult] = []
    for position in positions:
        instrument = instrument_from_id(str(position["instrument_id"]), data_dir, names)
        if instrument is None:
            results.append(
                PositionResult(position, None, None, "Nieznany instrument_id.", "BRAK INSTRUMENTU")
            )
            continue
        if not instrument.data_file.is_file():
            results.append(
                PositionResult(position, instrument, None, "Dane nie zostały pobrane.", "BŁĄD DANYCH")
            )
            continue
        try:
            metrics = calculate_position_metrics(
                load_price_series(instrument),
                date.fromisoformat(position["entry_date"]),
                float(position["ksl_pct"]),
                float(position["warning_pct"]),
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            results.append(PositionResult(position, instrument, None, str(exc), "BŁĄD DANYCH"))
            continue
        results.append(PositionResult(position, instrument, metrics, "", metrics["status"]))
    return results


def result_to_row(result: PositionResult) -> dict[str, Any]:
    """Wiersz tabeli (dashboard i raport) w układzie zbliżonym do KSL_Overview."""
    position = result.position
    label = result.instrument.label if result.instrument else str(position["instrument_id"])
    base = {
        "Pozycja": position["position_id"],
        "Rachunek": position["account_label"],
        "Instrument": label,
        "Status": result.status,
    }
    metrics = result.metrics
    if metrics is None:
        return {**base, "Błąd": result.error}
    daily = metrics["daily_return_pct"]
    return {
        **base,
        "Data wejścia": metrics["effective_entry_date"],
        "Cena wejścia": metrics["entry_price"],
        "Ostatnie dane": metrics["latest_date"],
        "Cena bieżąca": metrics["latest_price"],
        "Wynik od wejścia (%)": metrics["return_since_entry_pct"] * 100,
        "Wynik od maksimum (%)": metrics["drawdown_from_peak_pct"] * 100,
        "KSL Alert": metrics["ksl_status"],
        "KSL Warning": metrics["warning_status"],
        "1D return (%)": daily * 100 if daily is not None else None,
        "Świeżość": "NIEAKTUALNE" if metrics["is_stale"] else "OK",
        "Błąd": "",
    }


def overall_status(results: list[PositionResult]) -> str:
    """ALERT > WARNING > PROBLEM DANYCH > OK; nieaktualne dane liczą się jako problem danych."""
    statuses = {r.status for r in results}
    if "ALERT" in statuses:
        return "ALERT"
    if "WARNING" in statuses:
        return "WARNING"
    stale = any(r.metrics and r.metrics["is_stale"] for r in results)
    if stale or statuses & {"BŁĄD DANYCH", "BRAK INSTRUMENTU"}:
        return "PROBLEM DANYCH"
    return "OK"
