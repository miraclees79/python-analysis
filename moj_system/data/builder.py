# -*- coding: utf-8 -*-
"""
moj_system/data/builder.py
==========================
Builds complex, linked price series (e.g. MSCI World, STOXX 600)
by merging raw CSV files from GDrive with yFinance/Stooq extensions.
Replaces legacy price_series_builder.py.
"""


import os
import io
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path

from moj_system.data.gdrive import GDriveClient
from moj_system.data.data_manager import load_local_csv

# ... (stałe DATA_ROOT, RAW_DIR, DATA_START, CLOSE_COL bez zmian) ...
DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw_csv"
DATA_START = "1990-01-01"
CLOSE_COL = "Zamkniecie"


def _parse_wsj_csv(raw_bytes: bytes) -> pd.DataFrame | None:
    """Parses WSJ export format using multiple encoding attempts."""
    raw = None
    encodings_to_try = ("utf-8-sig", "utf-8", "latin-1")
    
    for current_encoding in encodings_to_try:
        try:
            # Jawne przekazanie wszystkich argumentów
            raw = pd.read_csv(
                filepath_or_buffer=io.BytesIO(initial_bytes=raw_bytes), 
                encoding=current_encoding, 
                thousands=",", 
                skipinitialspace=True
            )
            logging.info(f"WSJ CSV parsed successfully using encoding: {current_encoding}")
            break
        except Exception as exc:
            logging.debug(f"Encoding attempt '{current_encoding}' failed: {exc}")
            continue
            
    if raw is None:
        logging.error("Failed to parse WSJ CSV: None of the attempted encodings worked.")
        return None

    # Normalizacja nazw kolumn
    col_map = {c: c.strip().capitalize() for c in raw.columns}
    for col_name in raw.columns:
        clean_name = col_name.strip().lower()
        if clean_name in ("close", "price", "last", "adj close"):
            col_map[col_name] = "Close"
        elif clean_name in ("date", "data"):
            col_map[col_name] = "Date"
            
    raw = raw.rename(columns=col_map)

    # Konwersja daty z jawnymi argumentami
    raw["Date"] = pd.to_datetime(
        arg=raw["Date"], 
        format="mixed", 
        errors="coerce"
    )
    raw = raw.dropna(subset=["Date"])
    
    # Budowa wynikowej ramki danych
    out = pd.DataFrame(index=raw["Date"].dt.tz_localize(tz=None))
    out.index.name = "Data"
    
    # Konwersje numeryczne z jawnymi argumentami
    out[CLOSE_COL] = pd.to_numeric(
        arg=raw["Close"].values, 
        errors="coerce"
    )
    out["Najwyzszy"] = pd.to_numeric(
        arg=raw.get("High", raw["Close"]).values, 
        errors="coerce"
    )
    out["Najnizszy"] = pd.to_numeric(
        arg=raw.get("Low", raw["Close"]).values, 
        errors="coerce"
    )
    
    return out.sort_index().dropna()


def _extend_series(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extends historical base with new returns from ext_df."""
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

# --- NOWA FUNKCJA: Łączenie Syntetyka z WSJ/URTH ---
def _build_full_msci_world(client: GDriveClient, folder_id: str, wsj_combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Łączy syntetyczną bazę MSCI z serii WSJ+URTH.
    Odtwarza logikę z oryginalnego `msci_world_synthetic.py`.
    """
    logging.info("Rozpoczynam budowę pełnej historii MSCI World (Syntetyk + WSJ/URTH)...")
    
    # 1. Pobierz syntetyczną bazę (1990-2010)
    synthetic_df = client.download_csv(folder_id, "msci_world_synthetic.csv")
    if synthetic_df is None:
        logging.warning("Brak syntetycznej bazy MSCI World. Zwracam tylko serię WSJ/URTH.")
        return wsj_combined_df
    
    synthetic_df['Data'] = pd.to_datetime(synthetic_df['Data'])
    synthetic_df = synthetic_df.set_index('Data')
    
    wsj_start = wsj_combined_df.index.min()
    synth_pre = synthetic_df.loc[synthetic_df.index < wsj_start].copy()
    
    if synth_pre.empty:
        logging.info("Baza syntetyczna nie poprzedza danych WSJ. Zwracam serię WSJ/URTH.")
        return wsj_combined_df
        
    # 2. Skalowanie i łączenie (chain-link)
    anchor_synth_price = float(synth_pre[CLOSE_COL].iloc[-1])
    anchor_wsj_price = float(wsj_combined_df[CLOSE_COL].iloc[0])
    scale = anchor_wsj_price / anchor_synth_price
    
    for col in [CLOSE_COL, "Najwyzszy", "Najnizszy"]:
        if col in synth_pre.columns:
            synth_pre[col] = synth_pre[col] * scale

    combined = pd.concat([synth_pre, wsj_combined_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    
    logging.info(f"Połączono {len(synth_pre)} wierszy syntetycznych z {len(wsj_combined_df)} wierszami WSJ/URTH.")
    return combined

def build_and_upload(
    folder_id: str,
    raw_filename: str,
    combined_filename: str,
    extension_ticker: str,
    extension_source: str = "yfinance",
    credentials_path: str = None,
    # --- NOWY PARAMETR ---
    is_msci_world: bool = False
) -> pd.DataFrame | None:
    """Główna funkcja budująca i wysyłająca połączoną serię na GDrive."""
    logging.info(f"Budowanie serii: {combined_filename} (Ext: {extension_ticker})")
    client = GDriveClient(credentials_path)

    # 1. Spróbuj pobrać już gotowy plik Combined
    existing = client.download_csv(folder_id, combined_filename)
    if existing is not None:
        base_df = existing.set_index("Data")
        base_df.index = pd.to_datetime(base_df.index)
    else:
        # 2. Jeśli brak, pobierz Raw i sparsuj
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

    # 3. Download extension (yFinance or Stooq)
    ext_df = None
    if extension_source == "yfinance":
        logging.info(f"Downloading yFinance extension: {extension_ticker}")
        try:
            raw_yf = yf.download(extension_ticker, start=base_df.index.max(), progress=False, auto_adjust=True)
            if not raw_yf.empty:
                if isinstance(raw_yf.columns, pd.MultiIndex): raw_yf = raw_yf.droplevel(1, axis=1)
                ext_df = raw_yf.rename(columns={"Close": CLOSE_COL, "High": "Najwyzszy", "Low": "Najnizszy"})
                ext_df.index = pd.to_datetime(ext_df.index).tz_localize(None)
        except Exception as e:
            logging.warning(f"yFinance Error: {e}")
    elif extension_source == "stooq":
        # Load from locally downloaded files via data_manager
        logging.info(f"Loading local extension (Stooq): {extension_ticker}")
        clean_ticker = extension_ticker.replace('^', '').lower()
        ext_df = load_local_csv(clean_ticker, clean_ticker, mandatory=False)

    # 4. Merge and Save
    combined = _extend_series(base_df, ext_df)
    
    # --- BLOK SPECJALNY DLA MSCI WORLD ---
    if is_msci_world:
        combined = _build_full_msci_world(client, folder_id, combined)

    if combined is None:
        return None
    
    out_path = RAW_DIR / combined_filename
    combined.to_csv(out_path)
    
    try:
        client.upload_csv(folder_id, str(out_path), combined_filename)
    except Exception as e:
        logging.error(f"Nie udało się wysłać {combined_filename} na Drive: {e}")
        
    return combined