# -*- coding: utf-8 -*-
"""
moj_system/data/builder.py
==========================
Buduje skomplikowane, połączone serie cenowe (np. MSCI World, STOXX 600)
poprzez łączenie surowych plików CSV z Google Drive z przedłużeniami z yFinance/Stooq.
Zastępuje stary `current_development/price_series_builder.py`.
"""

import os
import io
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path

from moj_system.data.gdrive import GDriveClient
from moj_system.data.data_manager import load_local_csv

DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw_csv"
DATA_START = "1990-01-01"
CLOSE_COL = "Zamkniecie"


def _parse_wsj_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """Parsuje specyficzny format eksportu z WSJ (Wall Street Journal)."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, thousands=",", skipinitialspace=True)
            break
        except Exception:
            continue
    else:
        return None

    col_map = {c: c.strip().capitalize() for c in raw.columns}
    col_map.update({c: "Close" for c in raw.columns if c.strip().lower() in ("close", "price", "last", "adj close")})
    col_map.update({c: "Date" for c in raw.columns if c.strip().lower() in ("date", "data")})
    raw = raw.rename(columns=col_map)

    raw["Date"] = pd.to_datetime(raw["Date"], format="mixed", errors="coerce")
    raw = raw.dropna(subset=["Date"])
    
    out = pd.DataFrame(index=raw["Date"].dt.tz_localize(None))
    out.index.name = "Data"
    out[CLOSE_COL] = pd.to_numeric(raw["Close"].values, errors="coerce")
    out["Najwyzszy"] = pd.to_numeric(raw.get("High", raw["Close"]).values, errors="coerce")
    out["Najnizszy"] = pd.to_numeric(raw.get("Low", raw["Close"]).values, errors="coerce")
    return out.sort_index().dropna()


def _extend_series(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Przedłuża historyczną bazę o nowe stopy zwrotu z ext_df."""
    if base_df is None or base_df.empty: return ext_df
    if ext_df is None or ext_df.empty: return base_df

    base_close = base_df[CLOSE_COL].dropna()
    anchor_date = base_close.index.max()
    anchor_price = float(base_close.iloc[-1])

    ext_from_anchor = ext_df[CLOSE_COL].loc[ext_df.index >= anchor_date]
    if ext_from_anchor.empty: return base_df

    ext_rets = ext_from_anchor.pct_change().dropna()
    extension = anchor_price * (1 + ext_rets).cumprod()

    combined = pd.concat([base_close, extension]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    
    out = pd.DataFrame({CLOSE_COL: combined, "Najwyzszy": combined, "Najnizszy": combined})
    out.index.name = "Data"
    return out.loc[out.index >= pd.Timestamp(DATA_START)]


def build_and_upload(
    folder_id: str,
    raw_filename: str,
    combined_filename: str,
    extension_ticker: str,
    extension_source: str = "yfinance",
    credentials_path: str = None
) -> pd.DataFrame | None:
    """Główna funkcja budująca i wysyłająca połączoną serię na GDrive."""
    logging.info(f"Budowanie połączonej serii: {combined_filename} (Ext: {extension_ticker})")
    client = GDriveClient(credentials_path)

    # 1. Spróbuj pobrać już gotowy plik Combined
    existing = client.download_csv(folder_id, combined_filename)
    if existing is not None:
        base_df = existing.set_index("Data")
        base_df.index = pd.to_datetime(base_df.index)
        logging.info(f"Pobrano bazę z {combined_filename} (ostatnia data: {base_df.index.max().date()})")
    else:
        # 2. Jeśli brak Combined, pobierz Raw i sparsuj
        logging.info(f"Brak połączonego pliku. Pobieram surowy {raw_filename}...")
        file_id = client.find_file_id(folder_id, raw_filename)
        if not file_id:
            logging.error(f"Nie znaleziono pliku surowego {raw_filename}!")
            return None
            
        request = client.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        
        raw_bytes = fh.getvalue()
        base_df = _parse_wsj_csv(raw_bytes) if "wsj" in raw_filename.lower() else pd.read_csv(io.BytesIO(raw_bytes), parse_dates=["Data"], index_col="Data")

    # 3. Pobierz rozszerzenie (yFinance lub Stooq)
    ext_df = None
    if extension_source == "yfinance":
        logging.info(f"Pobieranie rozszerzenia z yFinance: {extension_ticker}")
        try:
            raw_yf = yf.download(extension_ticker, start=base_df.index.max(), progress=False, auto_adjust=True)
            if not raw_yf.empty:
                if isinstance(raw_yf.columns, pd.MultiIndex): raw_yf = raw_yf.droplevel(1, axis=1)
                ext_df = raw_yf.rename(columns={"Close": CLOSE_COL, "High": "Najwyzszy", "Low": "Najnizszy"})
                ext_df.index = pd.to_datetime(ext_df.index).tz_localize(None)
        except Exception as e:
            logging.warning(f"Błąd yFinance: {e}")
    elif extension_source == "stooq":
        # Wczytujemy z lokalnie pobranych plików przez nasz data_manager
        logging.info(f"Pobieranie rozszerzenia lokalnego (Stooq): {extension_ticker}")
        # Usuwamy przedrostek '^' do wyszukiwania plików (np. '^tbsp' -> 'tbsp')
        clean_ticker = extension_ticker.replace('^', '').lower()
        ext_df = load_local_csv(clean_ticker, clean_ticker, mandatory=False)

    # 4. Łączenie i Zapis
    combined = _extend_series(base_df, ext_df)
    
    out_path = RAW_DIR / combined_filename
    combined.to_csv(out_path)
    
    # Upload z powrotem na Drive
    try:
        client.upload_csv(folder_id, str(out_path), combined_filename)
    except Exception as e:
        logging.error(f"Nie udało się wysłać {combined_filename} na Drive: {e}")
        
    return combined