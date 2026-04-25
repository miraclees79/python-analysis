# -*- coding: utf-8 -*-
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
    """Parses WSJ export format."""
    raw = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(filepath_or_buffer=io.BytesIO(initial_bytes=raw_bytes), 
                              encoding=encoding, thousands=",", skipinitialspace=True)
            break
        except Exception: continue
    if raw is None: return None

    col_map = {c: c.strip().capitalize() for c in raw.columns}
    for c in raw.columns:
        low = c.strip().lower()
        if low in ("close", "price", "last", "adj close"): col_map[c] = "Close"
        elif low in ("date", "data"): col_map[c] = "Date"
    raw = raw.rename(columns=col_map)
    raw["Date"] = pd.to_datetime(arg=raw["Date"], format="mixed", errors="coerce")
    raw = raw.dropna(subset=["Date"])
    
    out = pd.DataFrame(index=raw["Date"].dt.tz_localize(tz=None))
    out.index.name = "Data"
    out[CLOSE_COL] = pd.to_numeric(arg=raw["Close"], errors="coerce")
    out["Najwyzszy"] = pd.to_numeric(arg=raw.get("High", raw["Close"]), errors="coerce")
    out["Najnizszy"] = pd.to_numeric(arg=raw.get("Low", raw["Close"]), errors="coerce")
    return out.sort_index().dropna()

def _extend_series(base_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    """Extends base_df with new returns from ext_df (Chain-linking)."""
    if base_df is None or base_df.empty: return ext_df
    if ext_df is None or ext_df.empty: return base_df

    # Upewnienie się, że oba indeksy są tz-naive
    base_df.index = base_df.index.tz_localize(None) if base_df.index.tz is not None else base_df.index
    ext_df.index = ext_df.index.tz_localize(None) if ext_df.index.tz is not None else ext_df.index

    anchor_date = base_df.index.max()
    anchor_price = float(base_df[CLOSE_COL].iloc[-1])

    # Pobierz dane rozszerzenia TYLKO po dacie kotwicy
    new_data = ext_df.loc[ext_df.index > anchor_date].copy()
    if new_data.empty: return base_df

    # Oblicz skumulowane zwroty rozszerzenia względem ceny kotwicy
    returns = new_data[CLOSE_COL].pct_change().fillna(0) # Pierwszy zwrot w nowym oknie
    # Jeśli pierwsza dany w new_data jest dużo późniejsza niż anchor, 
    # pct_change() między anchor a nową daną musiałby być ręcznie liczony
    first_ret = (new_data[CLOSE_COL].iloc[0] / ext_df[CLOSE_COL].reindex([anchor_date], method='ffill').iloc[0]) - 1
    returns.iloc[0] = first_ret
    
    extension = anchor_price * (1 + returns).cumprod()

    combined_close = pd.concat(objs=[base_df[CLOSE_COL], extension], axis=0).sort_index()
    combined_close = combined_close.loc[~combined_close.index.duplicated(keep="last")]
    
    out = pd.DataFrame(index=combined_close.index)
    out[CLOSE_COL] = combined_close
    out["Najwyzszy"] = combined_close
    out["Najnizszy"] = combined_close
    out.index.name = "Data"
    return out

def _build_full_msci_world(client: GDriveClient, folder_id: str, wsj_combined_df: pd.DataFrame) -> pd.DataFrame:
    """Łączy syntetyczną bazę MSCI (1990-2010) z serią WSJ/URTH."""
    # Jeśli dane już zaczynają się w 1990, nic nie rób
    if wsj_combined_df is not None and wsj_combined_df.index.min() <= pd.Timestamp("1990-01-05"):
        return wsj_combined_df

    logging.info("Building MSCI World from Synthetic Base (1990)...")
    synth_df = client.download_csv(folder_id=folder_id, filename="msci_world_synthetic.csv")
    if synth_df is None:
        logging.warning("Missing msci_world_synthetic.csv on Drive!")
        return wsj_combined_df
    
    synth_df["Data"] = pd.to_datetime(arg=synth_df["Data"]).dt.tz_localize(tz=None)
    synth_df = synth_df.set_index(keys="Data").sort_index()
    
    if wsj_combined_df is None or wsj_combined_df.empty:
        return synth_df

    # Łączymy syntetyk z tym co mamy (WSJ/URTH)
    return _extend_series(base_df=synth_df, ext_df=wsj_combined_df)

def build_and_upload(
    folder_id: str,
    raw_filename: str,
    combined_filename: str,
    extension_ticker: str,
    extension_source: str = "yfinance",
    credentials_path: str = None,
    is_msci_world: bool = False
) -> pd.DataFrame | None:
    """Main builder with integrated 1990 synthetic support."""
    client = GDriveClient(credentials_path=credentials_path)
    base_df = None

    # 1. Próba pobrania istniejącego pliku skonsolidowanego
    existing = client.download_csv(folder_id=folder_id, filename=combined_filename)
    if existing is not None:
        base_df = existing.set_index(keys="Data")
        base_df.index = pd.to_datetime(arg=base_df.index).tz_localize(tz=None)
        logging.info(f"Loaded existing {combined_filename} from Drive (Starts: {base_df.index.min().date()})")
    
    # 2. Jeśli brak skonsolidowanego, spróbuj zbudować z Raw (WSJ) LUB z Syntetyka
    if base_df is None:
        if is_msci_world:
            # Dla MSCI World fundamentem jest Syntetyk
            base_df = _build_full_msci_world(client=client, folder_id=folder_id, wsj_combined_df=None)
        
        # Spróbuj dodać dane z pliku RAW (np. WSJ) jeśli plik istnieje
        file_id = client.find_file_id(parent_id=folder_id, filename=raw_filename)
        if file_id:
            request = client.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            from googleapiclient.http import MediaIoBaseDownload
            downloader = MediaIoBaseDownload(fd=fh, request=request)
            done = False
            while not done: _, done = downloader.next_chunk()
            raw_df = _parse_wsj_csv(raw_bytes=fh.getvalue())
            base_df = _extend_series(base_df=base_df, ext_df=raw_df)

    # 3. Pobierz najnowsze rozszerzenie (yFinance)
    if extension_source == "yfinance":
        start_dt = base_df.index.max() if base_df is not None else "2010-01-01"
        try:
            ext_data = yf.download(tickers=extension_ticker, start=start_dt, progress=False, auto_adjust=True)
            if not ext_data.empty:
                if isinstance(ext_data.columns, pd.MultiIndex): ext_data = ext_data.droplevel(level=1, axis=1)
                ext_df = ext_data.rename(columns={"Close": CLOSE_COL, "High": "Najwyzszy", "Low": "Najnizszy"})
                ext_df.index = pd.to_datetime(arg=ext_df.index).tz_localize(tz=None)
                base_df = _extend_series(base_df=base_df, ext_df=ext_df)
        except Exception as e: logging.warning(f"Extension Error ({extension_ticker}): {e}")

    if base_df is None: return None
    
    # Finalne sprawdzenie dla MSCI World (czy na pewno mamy 1990)
    if is_msci_world:
        base_df = _build_full_msci_world(client=client, folder_id=folder_id, wsj_combined_df=base_df)

    # Zapis i Upload
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / combined_filename
    base_df.to_csv(path_or_buf=out_path)
    client.upload_csv(folder_id=folder_id, local_path=str(out_path), filename=combined_filename)
    
    return base_df