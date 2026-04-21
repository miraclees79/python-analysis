# -*- coding: utf-8 -*-
"""
moj_system/data/updater.py
==========================
Hybrid data updater module. Handles default indices, ETFs, and KNF funds.
"""

import os
import io
import zipfile
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from pathlib import Path
from moj_system.data.gdrive import GDriveClient

# --- PATH CONFIGURATION ---
DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw_csv"
ZIP_DIR = DATA_ROOT / "zips"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DIR.mkdir(parents=True, exist_ok=True)

ZIP_MAPPING = {
    "index_pl": "d_pl_txt.zip",
    "index_world": "d_world_txt.zip",
    "currencies": "d_world_txt.zip",
    "fund_pl": "d_pl_txt.zip",
    "etf_pl": "d_pl_txt.zip",
    "bonds": "d_world_txt.zip"
}

CONFIRMED_FUNDS_FILE = "knf_stooq_confirmed.csv"
GDRIVE_DATA_FOLDER_NAME = "Dane"

# =========================================================================
# TICKER LISTS (Ported from legacy stooq_hybrid_updater.py)
# =========================================================================

DEFAULT_TICKERS = [
    {"label": "wig", "stooq": "wig", "yf": "WIG.WA", "type": "index_pl"},
    {"label": "wig20tr", "stooq": "wig20tr", "yf": "WIG20TR.WA", "type": "index_pl"},
    {"label": "mwig40tr", "stooq": "mwig40tr", "yf": "MWIG40TR.WA", "type": "index_pl"},
    {"label": "swig80tr", "stooq": "swig80tr", "yf": "SWIG80TR.WA", "type": "index_pl"},
    {"label": "tbsp", "stooq": "^tbsp", "yf": "TBSP-INDEX.WA", "type": "index_pl"},
    {"label": "sp500", "stooq": "^spx", "yf": "^GSPC", "type": "index_world"},
    {"label": "nk225", "stooq": "^nkx", "yf": "^N225", "type": "index_world"},
    {"label": "ndq100", "stooq": "^ndx", "yf": "^NDX", "type": "index_world"},
    {"label": "wibor1m", "stooq": "plopln1m", "yf": None, "type": "currencies"},
    {"label": "usdpln", "stooq": "usdpln", "yf": "USDPLN=X", "type": "currencies"},
    {"label": "eurpln", "stooq": "eurpln", "yf": "EURPLN=X", "type": "currencies"},
    {"label": "jpypln", "stooq": "jpypln", "yf": "JPYPLN=X", "type": "currencies"},
    {"label": "de10y", "stooq": "10ydey.b", "yf": None, "type": "bonds"},
    {"label": "pl10y", "stooq": "10yply.b", "yf": None, "type": "bonds"},
    {"label": "fund_2720", "stooq": "2720.n", "yf": None, "type": "fund_pl", "knf": "195983"},
    {"label": "wbbw", "stooq": "^gpwbbwz", "yf": None, "type": "index_pl"},
]

ETF_TICKERS = [
    {"label": "etfbdivpl", "stooq": "etfbdivpl.pl", "yf": "ETFBDIVPL.WA", "type": "etf_pl"},
    {"label": "etfbm40tr", "stooq": "etfbm40tr.pl", "yf": "ETFBM40TR.WA", "type": "etf_pl"},
    {"label": "etfbs80tr", "stooq": "etfbs80tr.pl", "yf": "ETFBS80TR.WA", "type": "etf_pl"},
    {"label": "etfbw20lv", "stooq": "etfbw20lv.pl", "yf": "ETFBW20LV.WA", "type": "etf_pl"},
    {"label": "etfbw20st", "stooq": "etfbw20st.pl", "yf": "ETFBW20ST.WA", "type": "etf_pl"},
    {"label": "etfbw20tr", "stooq": "etfbw20tr.pl", "yf": "ETFBW20TR.WA", "type": "etf_pl"},
    {"label": "etfpzuw20m40", "stooq": "etfpzuw20m40.pl", "yf": "ETFPZUW20M40.WA", "type": "etf_pl"},
    {"label": "etfbcash", "stooq": "etfbcash.pl", "yf": "ETFBCASH.WA", "type": "etf_pl"},
    {"label": "etfbtbsp", "stooq": "etfbtbsp.pl", "yf": "ETFBTBSP.WA", "type": "etf_pl"},
    {"label": "etfbndxpl", "stooq": "etfbndxpl.pl", "yf": "ETFBNDXPL.WA", "type": "etf_pl"},
    {"label": "etfbnq2st", "stooq": "etfbnq2st.pl", "yf": "ETFBNQ2ST.WA", "type": "etf_pl"},
    {"label": "etfbspxpl", "stooq": "etfbspxpl.pl", "yf": "ETFBSPXPL.WA", "type": "etf_pl"},
    {"label": "etfsp500", "stooq": "etfsp500.pl", "yf": "ETFSP500.WA", "type": "etf_pl"},
    {"label": "etfdax", "stooq": "etfdax.pl", "yf": "ETFDAX.WA", "type": "etf_pl"},
]


class DataUpdater:
    def __init__(self, gdrive_folder_id=None, credentials_path=None):
        self.gdrive = GDriveClient(credentials_path)
        self.root_folder_id = gdrive_folder_id or os.environ.get("GDRIVE_FOLDER_ID")
        self.data_folder_id = None
        if self.gdrive.service and self.root_folder_id:
            self.data_folder_id = self._get_or_create_subfolder(GDRIVE_DATA_FOLDER_NAME)

    def _get_or_create_subfolder(self, folder_name: str) -> str:
        """Gets or creates a subfolder on GDrive."""
        if not self.gdrive.service: return None
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.gdrive.service.files().list(q=query, fields='files(id)').execute()
        items = results.get('files', [])
        if items: return items[0]['id']
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [self.root_folder_id]}
        folder = self.gdrive.service.files().create(body=metadata, fields='id').execute()
        return folder.get('id')

    def _get_zip_content(self, zip_type: str) -> bytes:
        zip_name = ZIP_MAPPING.get(zip_type)
        if not zip_name: return None
        local_path = ZIP_DIR / zip_name
        if local_path.exists(): return local_path.read_bytes()
        if self.root_folder_id and self.gdrive.service:
            logging.info(f"Downloading {zip_name} from GDrive...")
            file_id = self.gdrive.find_file_id(self.root_folder_id, zip_name)
            if file_id:
                try:
                    request = self.gdrive.service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    from googleapiclient.http import MediaIoBaseDownload
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done: _, done = downloader.next_chunk()
                    content = fh.getvalue()
                    local_path.write_bytes(content)
                    return content
                except Exception as e: logging.error(f"GDrive ZIP error: {e}")
        return None

    def _extract_from_zip(self, zip_data: bytes, stooq_ticker: str) -> pd.DataFrame | None:
        """
        Wyodrebnia dane pojedynczego tickera z duzego archiwum ZIP pobranego ze Stooq.

        Mechanism:
        ----------
        Przeszukuje plik ZIP w pamięci, parsuje plik tekstowy tickera, mapuje nazwy 
        kolumn Stooq (<DATE>, <CLOSE>) na systemowe ('Data', 'Zamkniecie') i konwertuje daty.

        Returns:
            --------
            pd.DataFrame | None - Dane historyczne w formacie ramki danych.
            """
        if not zip_data: return None
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                search_name = f"{stooq_ticker.lower()}.txt"
                target_file = next((f for f in z.namelist() if f.lower().endswith(search_name)), None)
                if target_file:
                    with z.open(target_file) as f:
                        df = pd.read_csv(f)
                        col_map = {'<DATE>': 'Data', '<OPEN>': 'Otwarcie', '<HIGH>': 'Najwyzszy', '<LOW>': 'Najnizszy', '<CLOSE>': 'Zamkniecie'}
                        cols_to_keep = [c for c in col_map.keys() if c in df.columns]
                        df = df[cols_to_keep].rename(columns={k: col_map[k] for k in cols_to_keep})
                        df['Data'] = pd.to_datetime(df['Data'], format='%Y%m%d')
                        return df
        except Exception as e: logging.error(f"ZIP error for {stooq_ticker}: {e}")
        return None

    def _fetch_yfinance_data(self, ticker_yf: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
        try:
            df = yf.download(ticker_yf, start=start_date, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df = df.droplevel(1, axis=1)
            df = df.reset_index().rename(columns={'Date': 'Data', 'Open': 'Otwarcie', 'High': 'Najwyzszy', 'Low': 'Najnizszy', 'Close': 'Zamkniecie'})
            df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
            cols = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']
            return df[[c for c in cols if c in df.columns]]
        except Exception as e: logging.warning(f"yfinance error ({ticker_yf}): {e}"); return None

    def _fetch_knf_data(self, subfund_id: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
        if not subfund_id or pd.isna(subfund_id): return None
        url = "https://wybieramfundusze-api.knf.gov.pl/v1/valuations"
        params = {"subfundId": int(float(subfund_id)), "dateFrom": pd.to_datetime(start_date).normalize().strftime('%Y-%m-%d')}
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            items = data.get('content', []) if isinstance(data, dict) else data
            records = []
            for item in items:
                v_date = pd.to_datetime(item.get('date')).normalize()
                if v_date > start_date:
                    records.append({'Data': v_date, 'Otwarcie': item.get('valuation'), 'Najwyzszy': item.get('valuation'), 'Najnizszy': item.get('valuation'), 'Zamkniecie': item.get('valuation')})
            return pd.DataFrame(records).sort_values('Data') if records else None
        except Exception as e: logging.warning(f"KNF API Error: {e}"); return None

    def _validate_and_clean(self, df: pd.DataFrame, label: str) -> pd.DataFrame | None:
        if df is None or df.empty: return None
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Data').drop_duplicates(subset='Data', keep='last').set_index('Data')
        date_diffs = df.index.to_series().diff().dt.days
        breaks = date_diffs[date_diffs > 30].index
        if not breaks.empty: df = df.loc[df.index > breaks[-1]]
        return df.reset_index()

    def update_ticker(self, label: str, stooq_ticker: str, yf_ticker: str = None, 
                      knf_id: str = None, zip_type: str = "index_pl", 
                      upload_to_drive: bool = False):
        """
        Aktualizuje dane instrumentu korzystając z hybrydowych zrodeł (Stooq ZIP + API).

        Mechanism:
            ----------
            1. Pobiera bazę historyczną z lokalnego lub zdalnego (GDrive) pliku ZIP.
            2. Pobiera najnowsze dane z Yahoo Finance lub KNF API (dla funduszy).
            3. Łączy serie, usuwa duplikaty i waliduje ciągłość danych (brak dziur > 30 dni).
            4. Zapisuje wynik do raw_csv i opcjonalnie wysyła na GDrive.

        Returns:
            --------
            bool - True jesli aktualizacja zakończyła się sukcesem.
            """

        logging.info(f"--- Updating: {label} ({stooq_ticker}) ---")
        zip_data = self._get_zip_content(zip_type)
        df_hist = self._extract_from_zip(zip_data, stooq_ticker) if zip_data else None
        last_date = df_hist['Data'].max() if df_hist is not None else pd.Timestamp("1990-01-01")
        
        df_new = self._fetch_yfinance_data(yf_ticker, last_date) if yf_ticker else self._fetch_knf_data(knf_id, last_date)
        df_final = pd.concat([df_hist, df_new], ignore_index=True) if (df_hist is not None and df_new is not None) else (df_hist if df_hist is not None else df_new)

        df_validated = self._validate_and_clean(df_final, label)
        if df_validated is not None:
            safe_name = label.replace(" ", "_").lower()
            out_path = RAW_DIR / f"{safe_name}.csv"
            df_validated.to_csv(out_path, index=False)
            if upload_to_drive and self.gdrive.service and self.data_folder_id:
                fname = f"historia{stooq_ticker[:4] if zip_type=='fund_pl' else stooq_ticker}.csv"
                self.gdrive.upload_csv(self.data_folder_id, str(out_path), fname)
            return True
        return False

    def run_full_update(self, get_funds: bool = True):
        """Main entry point for hybrid update."""
        logging.info(f"Full Update Started. Funds/ETFs: {get_funds}")
        
        # 1. ALWAYS UPDATE DEFAULT TICKERS
        for item in DEFAULT_TICKERS:
            self.update_ticker(label=item["label"], stooq_ticker=item["stooq"], 
                               yf_ticker=item.get("yf"), knf_id=item.get("knf"), 
                               zip_type=item["type"])

        # 2. OPTIONAL: UPDATE ETFs AND DYNAMIC FUNDS
        if get_funds:
            for item in ETF_TICKERS:
                self.update_ticker(label=item["label"], stooq_ticker=item["stooq"], 
                                   yf_ticker=item.get("yf"), zip_type=item["type"], 
                                   upload_to_drive=True)
            
            if self.gdrive.service and self.root_folder_id:
                df_c = self.gdrive.download_csv(self.root_folder_id, CONFIRMED_FUNDS_FILE)
                if df_c is not None and not df_c.empty:
                    for _, row in df_c.dropna(subset=['stooq_id']).iterrows():
                        sid = str(row['stooq_id']).lower()
                        self.update_ticker(label=f"fund_{sid}", stooq_ticker=f"{sid}.n", 
                                           knf_id=str(row['subfundId']), zip_type="fund_pl", 
                                           upload_to_drive=True)