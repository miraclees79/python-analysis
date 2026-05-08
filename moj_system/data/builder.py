# -*- coding: utf-8 -*-
import io
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from moj_system.config import DATA_DIR
from moj_system.data.gdrive import GDriveClient
from moj_system.data.data_manager import load_local_csv

RAW_DIR = DATA_DIR
DATA_START = "1990-01-01"
CLOSE_COL = "Zamkniecie"


def _parse_wsj_csv(
    raw_bytes: bytes,
) -> pd.DataFrame | None:

    """Parses WSJ export format using multiple encoding attempts."""
    raw = None
    encodings_to_try = ("utf-8-sig", "utf-8", "latin-1")
    
    for current_enc in encodings_to_try:
        try:
            # Tworzymy bufor bajtów
            initial_data = io.BytesIO(initial_bytes=raw_bytes)
            
            # Próba odczytu z jawnymi argumentami
            raw = pd.read_csv(
                filepath_or_buffer=initial_data,
                encoding=current_enc,
                thousands=",",
                skipinitialspace=True,
            )
            # Jeśli się udało, wychodzimy z pętli
            break
            
        except Exception as exc:
            # Naprawa Bandit B112: Logujemy błąd zamiast cichego 'continue'
            logging.debug(
                msg=f"Attempt with encoding {current_enc} failed: {exc}"
            )
            continue
            
    if raw is None:
        logging.error(
            msg="Failed to parse WSJ CSV: None of the attempted encodings worked."
        )
        return None

    # Normalizacja nazw kolumn
    col_map = {c: c.strip().capitalize() for c in raw.columns}
    for col_raw_name in raw.columns:
        low_name = col_raw_name.strip().lower()
        if low_name in ("close", "price", "last", "adj close"):
            col_map[col_raw_name] = "Close"
        elif low_name in ("date", "data"):
            col_map[col_raw_name] = "Date"
            
    raw = raw.rename(columns=col_map)
    
    # Konwersja daty
    raw["Date"] = pd.to_datetime(
        arg=raw["Date"], 
        format="mixed", 
        errors="coerce"
    )
    raw = raw.dropna(subset=["Date"])

    # Budowa wynikowego DataFrame
    out = pd.DataFrame(index=raw["Date"].dt.tz_localize(tz=None))
    out.index.name = "Data"
    
    out[CLOSE_COL] = pd.to_numeric(
        arg=raw["Close"], 
        errors="coerce"
    )
    out["Najwyzszy"] = pd.to_numeric(
        arg=raw.get("High", raw["Close"]), 
        errors="coerce"
    )
    out["Najnizszy"] = pd.to_numeric(
        arg=raw.get("Low", raw["Close"]), 
        errors="coerce"
    )
    
    return out.sort_index().dropna()


def _extend_series(
    base_df: pd.DataFrame | None, 
    ext_df:  pd.DataFrame | None,
) -> pd.DataFrame:

    """Extends base_df with new returns from ext_df (Chain-linking)."""
    if base_df is None or base_df.empty:
        return ext_df
    if ext_df is None or ext_df.empty:
        return base_df

    # POPRAWKA: Na indeksach (DatetimeIndex) wywołujemy metody bezpośrednio (bez .dt)
    base_df.index = pd.to_datetime(arg=base_df.index).tz_localize(tz=None).normalize()
    ext_df.index = pd.to_datetime(arg=ext_df.index).tz_localize(tz=None).normalize()

    anchor_date = base_df.index.max()
    anchor_price = float(base_df[CLOSE_COL].iloc[-1])

    new_data = ext_df.loc[ext_df.index > anchor_date].copy()
    if new_data.empty:
        return base_df

    try:
        ext_anchor_rows = ext_df.loc[ext_df.index <= anchor_date]
        if ext_anchor_rows.empty:
            returns = new_data[CLOSE_COL].pct_change().fillna(value=0)
        else:
            ext_anchor_price = float(ext_anchor_rows[CLOSE_COL].iloc[-1])
            first_return = (new_data[CLOSE_COL].iloc[0] / ext_anchor_price) - 1
            returns = new_data[CLOSE_COL].pct_change()
            returns.iloc[0] = first_return
    except Exception:
        returns = new_data[CLOSE_COL].pct_change().fillna(value=0)

    extension = anchor_price * (1 + returns).cumprod()

    combined_series = pd.concat(objs=[base_df[CLOSE_COL], extension], axis=0).sort_index()
    combined_series = combined_series.loc[~combined_series.index.duplicated(keep="last")]

    out = pd.DataFrame(index=combined_series.index)
    out[CLOSE_COL] = combined_series
    out["Najwyzszy"] = combined_series
    out["Najnizszy"] = combined_series
    out.index.name = "Data"
    return out


def _build_full_msci_world(
    client:          GDriveClient, 
    folder_id:       str, 
    wsj_combined_df: pd.DataFrame,
) -> pd.DataFrame:

    """Łączy syntetyczną bazę MSCI (1990-2010) z serią rzeczywistą."""
    if wsj_combined_df is not None and not wsj_combined_df.empty:
        if wsj_combined_df.index.min() <= pd.Timestamp("1990-01-05"):
            return wsj_combined_df

    logging.info(msg="Merging with MSCI World Synthetic Base (1990)...")
    synth_df = client.download_csv(folder_id=folder_id, filename="msci_world_synthetic.csv")
    if synth_df is None:
        return wsj_combined_df

    # 'Data' to kolumna (Series), więc .dt jest tutaj poprawne
    synth_df["Data"] = pd.to_datetime(arg=synth_df["Data"]).dt.tz_localize(tz=None).dt.normalize()
    synth_df = synth_df.set_index(keys="Data").sort_index()

    return _extend_series(base_df=synth_df, ext_df=wsj_combined_df)


def build_and_upload(
    folder_id:        str,
    raw_filename:     str,
    combined_filename:str,
    extension_ticker: str,
    extension_source: str  = "yfinance",
    credentials_path: str | None = None,
    is_msci_world:    bool = False,
) -> pd.DataFrame | None:

    """Main builder: fetches from Drive, extends from local/YF, and uploads back."""
    client = GDriveClient(credentials_path=credentials_path)
    base_df = None

    # 1. Pobranie bazy z Drive
    existing = client.download_csv(folder_id=folder_id, filename=combined_filename)
    if existing is not None:
        base_df = existing.set_index(keys="Data")
        # POPRAWKA: Usunięto .dt (operacja bezpośrednio na indeksie)
        base_df.index = pd.to_datetime(arg=base_df.index).tz_localize(tz=None).normalize()
        logging.info(msg=f"Loaded {combined_filename} from Drive. Last date: {base_df.index.max().date()}")

    # 2. Pobranie danych rozszerzających
    ext_df = None
    if extension_source == "yfinance":
        start_dt = base_df.index.max() if base_df is not None else "2010-01-01"
        try:
            ext_data = yf.download(tickers=extension_ticker, start=start_dt, progress=False, auto_adjust=True)
            if not ext_data.empty:
                if isinstance(ext_data.columns, pd.MultiIndex):
                    ext_data = ext_data.droplevel(level=1, axis=1)
                ext_df = ext_data.rename(columns={"Close": CLOSE_COL, "High": "Najwyzszy", "Low": "Najnizszy"})
                # POPRAWKA: Usunięto .dt
                ext_df.index = pd.to_datetime(arg=ext_df.index).tz_localize(tz=None).normalize()
        except Exception as e:
            logging.warning(msg=f"yFinance Error for {extension_ticker}: {e}")
    elif extension_source == "stooq":
        # UWAGA: Tutaj usuwamy daszek TYLKO po to, by wczytać plik 'tbsp.csv' (stworzony przez label w updaterze).
        # Oryginalny ticker (np. ^tbsp) w ZIP pozostaje nienaruszony w procesie updatu.
        file_name_to_load = extension_ticker.replace("^", "").lower()
        ext_df = load_local_csv(ticker=file_name_to_load, label=extension_ticker, mandatory=False)

    # 3. Połączenie serii
    combined = _extend_series(base_df=base_df, ext_df=ext_df)

    if is_msci_world:
        combined = _build_full_msci_world(client=client, folder_id=folder_id, wsj_combined_df=combined)

    if combined is None or combined.empty:
        return None

    # 4. Zapis lokalny i wysyłka na Drive
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / combined_filename
    combined.to_csv(path_or_buf=out_path)
    
    if base_df is None or combined.index.max() > base_df.index.max():
        logging.info(msg=f"Uploading updated {combined_filename} to Drive (New end date: {combined.index.max().date()})")
        client.upload_file(folder_id=folder_id, local_path=str(out_path), filename=combined_filename)
    else:
        logging.info(msg=f"No new data to upload for {combined_filename}.")

    return combined