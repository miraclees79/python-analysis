# -*- coding: utf-8 -*-
"""
moj_system/data/updater.py
==========================
Hybrid data updater module. Combines historical ZIP packages (Stooq)
with recent API data (yfinance/KNF).
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

# ZIP file mapping
ZIP_MAPPING = {
    "index_pl": "d_pl_txt.zip",
    "index_world": "d_world_txt.zip",
    "currencies": "d_world_txt.zip",
    "fund_pl": "d_pl_txt.zip",
    "etf_pl": "d_pl_txt.zip",
    "bonds": "d_world_txt.zip" # Added mapping for bonds like PL10Y
}

CONFIRMED_FUNDS_FILE = "knf_stooq_confirmed.csv"
GDRIVE_DATA_FOLDER_NAME = "Dane"


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
        
        if items:
            return items[0]['id']
        else:
            logging.info(f"Creating new GDrive folder '{folder_name}'...")
            metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.root_folder_id]
            }
            folder = self.gdrive.service.files().create(body=metadata, fields='id').execute()
            return folder.get('id')

    def _get_zip_content(self, zip_type: str) -> bytes:
        """Fetches ZIP content locally or from GDrive."""
        zip_name = ZIP_MAPPING.get(zip_type)
        if not zip_name: return None
        
        local_path = ZIP_DIR / zip_name
        
        if local_path.exists():
            return local_path.read_bytes()
        
        if self.root_folder_id and self.gdrive.service:
            logging.info(f"Downloading {zip_name} from Google Drive...")
            file_id = self.gdrive.find_file_id(self.root_folder_id, zip_name)
            if file_id:
                try:
                    request = self.gdrive.service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    from googleapiclient.http import MediaIoBaseDownload
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    content = fh.getvalue()
                    local_path.write_bytes(content)
                    return content
                except Exception as e:
                    logging.error(f"GDrive ZIP download error: {e}")
        
        return None

    def _extract_from_zip(self, zip_data: bytes, stooq_ticker: str) -> pd.DataFrame | None:
        """Extracts a specific txt file from the Stooq ZIP."""
        if not zip_data: return None
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # Search case-insensitive
                search_name = f"{stooq_ticker.lower()}.txt"
                target_file = next((f for f in z.namelist() if f.lower().endswith(search_name)), None)
                
                if target_file:
                    with z.open(target_file) as f:
                        df = pd.read_csv(f)
                        col_map = {
                            '<DATE>': 'Data', '<OPEN>': 'Otwarcie', '<HIGH>': 'Najwyzszy',
                            '<LOW>': 'Najnizszy', '<CLOSE>': 'Zamkniecie', '<VOL>': 'Wolumen'
                        }
                        # Tylko kolumny, które istnieją w mapowaniu
                        cols_to_keep = [c for c in col_map.keys() if c in df.columns]
                        df = df[cols_to_keep].rename(columns={k: col_map[k] for k in cols_to_keep})
                        df['Data'] = pd.to_datetime(df['Data'], format='%Y%m%d')
                        return df
        except Exception as e:
            logging.error(f"ZIP extraction error for {stooq_ticker}: {e}")
        return None

    def _fetch_yfinance_data(self, ticker_yf: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
        """Fetches update data from yfinance."""
        try:
            df = yf.download(ticker_yf, start=start_date, progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): 
                df = df.droplevel(1, axis=1)
                
            df = df.reset_index().rename(columns={
                'Date': 'Data', 'Open': 'Otwarcie', 'High': 'Najwyzszy', 
                'Low': 'Najnizszy', 'Close': 'Zamkniecie'
            })
            df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
            cols = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']
            return df[[c for c in cols if c in df.columns]]
        except Exception as e:
            logging.warning(f"yfinance error ({ticker_yf}): {e}")
            return None

    def _fetch_knf_data(self, subfund_id: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
        """Fetches update data from KNF API."""
        if not subfund_id or pd.isna(subfund_id): return None
        
        url = "https://wybieramfundusze-api.knf.gov.pl/v1/valuations"
        start_date_str = pd.to_datetime(start_date).normalize().strftime('%Y-%m-%d')
        params = {"subfundId": int(float(subfund_id)), "dateFrom": start_date_str}
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            items = data.get('content', []) if isinstance(data, dict) else data

            records = []
            for item in items:
                d_val = item.get('date')
                v_val = item.get('valuation')
                if d_val and v_val is not None:
                    v_date = pd.to_datetime(d_val).normalize()
                    if v_date > start_date:
                        records.append({
                            'Data': v_date, 'Otwarcie': v_val, 'Najwyzszy': v_val, 
                            'Najnizszy': v_val, 'Zamkniecie': v_val
                        })
            
            if records:
                return pd.DataFrame(records).sort_values('Data')
            return None
        except Exception as e:
            logging.warning(f"KNF API Error (ID {subfund_id}): {e}")
            return None

    def _validate_and_clean(self, df: pd.DataFrame, label: str) -> pd.DataFrame | None:
        """Cleans dataframe and checks for staleness/gaps."""
        if df is None or df.empty: return None
        
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Data').drop_duplicates(subset='Data', keep='last').set_index('Data')
        
        newest_date = df.index.max()
        if (dt.now() - newest_date).days > 15:
            logging.debug(f"[{label}] Data might be stale. Last date: {newest_date.date()}")
            
        date_diffs = df.index.to_series().diff().dt.days
        breaks = date_diffs[date_diffs > 30].index
        if not breaks.empty:
            df = df.loc[df.index > breaks[-1]]
            logging.info(f"[{label}] Trimmed data due to >30 days gap.")
            
        return df.reset_index()

    def update_ticker(self, label: str, stooq_ticker: str, yf_ticker: str = None, 
                      knf_id: str = None, zip_type: str = "index_pl", 
                      upload_to_drive: bool = False):
        """Main update method for a single asset."""
        logging.info(f"--- Updating: {label} ---")
        
        zip_data = self._get_zip_content(zip_type)
        df_hist = self._extract_from_zip(zip_data, stooq_ticker) if zip_data else None
        
        last_date = df_hist['Data'].max() if df_hist is not None else pd.Timestamp("1990-01-01")
        df_new = None
        
        if yf_ticker:
            df_new = self._fetch_yfinance_data(yf_ticker, last_date)
        elif knf_id:
            df_new = self._fetch_knf_data(knf_id, last_date)

        if df_hist is not None and df_new is not None:
            df_final = pd.concat([df_hist, df_new], ignore_index=True)
        elif df_hist is not None:
            df_final = df_hist
        else:
            df_final = df_new

        df_validated = self._validate_and_clean(df_final, label)
        if df_validated is not None:
            # Wymuszamy nazwy plików małymi literami
            safe_name = label.replace(" ", "_").lower()
            out_path = RAW_DIR / f"{safe_name}.csv"
            df_validated.to_csv(out_path, index=False)
            logging.info(f"SAVED: {out_path.name} ({len(df_validated)} rows)")
            
            if upload_to_drive and self.gdrive.service and self.data_folder_id:
                prefix = "historia"
                file_name = f"{prefix}{stooq_ticker[:4]}.csv" if zip_type == "fund_pl" else f"{prefix}{stooq_ticker}.csv"
                try:
                    self.gdrive.upload_csv(self.data_folder_id, str(out_path), file_name)
                except Exception as e:
                    logging.error(f"GDrive upload failed for {file_name}: {e}")
            return True
        else:
            logging.error(f"NO DATA for {label}")
            return False

    def run_full_update(self, asset_registry: dict, get_funds: bool = True):
        """Runs update across entire registry."""
        logging.info(f"Starting full data update (Fetch KNF funds: {get_funds})...")
        
        dynamic_funds = []
        if get_funds and self.gdrive.service and self.root_folder_id:
            df_confirmed = self.gdrive.download_csv(self.root_folder_id, CONFIRMED_FUNDS_FILE)
            if df_confirmed is not None and not df_confirmed.empty:
                for _, row in df_confirmed.dropna(subset=['stooq_id']).iterrows():
                    sid = str(row['stooq_id']).lower()
                    dynamic_funds.append({
                        "label": f"fund_{sid}", "stooq_ticker": f"{sid}.n",
                        "knf_id": str(row['subfundId']), "zip_type": "fund_pl"
                    })

        for name, cfg in asset_registry.items():
            if cfg.get("type") == "portfolio": continue
            
            yf_ticker = cfg.get("yf_ticker")
            stooq_ticker = cfg.get("ticker", name)
            
            z_type = "index_world" if cfg.get("source") == "drive" or name in ["SP500", "Nikkei225"] else "index_pl"
            if "USDPLN" in name or "EURPLN" in name or "JPYPLN" in name: z_type = "currencies"

            self.update_ticker(
                label=name, stooq_ticker=stooq_ticker, yf_ticker=yf_ticker, zip_type=z_type
            )
            
        # Niezbędne helpery - nazywamy je konsekwentnie małymi literami dla load_local_csv
        self.update_ticker("fund_2720", "2720.n", knf_id="195983", zip_type="fund_pl")
        self.update_ticker("wibor1m", "plopln1m", zip_type="currencies")
        self.update_ticker("pl10y", "10YPLY.B", zip_type="bonds") 
        self.update_ticker("de10y", "10YDEY.B", zip_type="bonds")
        self.update_ticker("wig", "wig", zip_type="index_pl")  # Szeroki WIG dla portfela Multi-Asset

        if get_funds:
            for fund in dynamic_funds:
                self.update_ticker(
                    label=fund["label"], stooq_ticker=fund["stooq_ticker"],
                    knf_id=fund["knf_id"], zip_type=fund["zip_type"], upload_to_drive=True
                )