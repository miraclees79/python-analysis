# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:19:47 2026

@author: adamg
"""

import os
import io
import json
import logging
import tempfile
import zipfile
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
import datetime

# --- KONFIGURACJA LOGOWANIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- STAŁE I KONFIGURACJA ---
DATA_START = "1990-01-01"
CLOSE_COL = "Zamkniecie"
COL_MAP_STOOQ = {
    '<DATE>': 'Data',
    '<OPEN>': 'Otwarcie',
    '<HIGH>': 'Najwyzszy',
    '<LOW>': 'Najnizszy',
    '<CLOSE>': 'Zamkniecie'
}
FINAL_COLS = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']

# Ścieżki do plików (obsługa lokalna i GitHub)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Pobieranie Folder ID i Credentials z env (GitHub Secrets)
FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")
CREDENTIALS_PATH = os.path.join(tempfile.gettempdir(), "credentials.json")


# Cache dla ZIPów z Google Drive
_ZIP_CACHE = {}

# --- LISTA TICKERÓW (Twoja tabela) ---
TICKERS_TABLE = [
    {"label": "WIG", "stooq": "wig", "yf": "WIG.WA", "type": "index_pl", "knf": None},
    {"label": "WIG20TR", "stooq": "wig20tr", "yf": "WIG20TR.WA", "type": "index_pl", "knf": None},
    {"label": "MWIG40TR", "stooq": "mwig40tr", "yf": "MWIG40TR.WA", "type": "index_pl", "knf": None},
    {"label": "SWIG80TR", "stooq": "swig80tr", "yf": "SWIG80TR.WA", "type": "index_pl", "knf": None},
    {"label": "TBSP", "stooq": "^tbsp", "yf": "TBSP-INDEX.WA", "type": "index_pl", "knf": None},
    {"label": "MMF", "stooq": "2720.n", "yf": None, "type": "fund_pl", "knf": "195983"},
    {"label": "SP500", "stooq": "^spx", "yf": "^GSPC", "type": "index_world", "knf": None},
    {"label": "WIBOR1M", "stooq": "plopln1m", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "PL10Y", "stooq": "10yply.b", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "DE10Y", "stooq": "10ydey.b", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "USDPLN", "stooq": "usdpln", "yf": "USDPLN=X", "type": "fx_rate", "knf": None},
    {"label": "EURPLN", "stooq": "eurpln", "yf": "EURPLN=X", "type": "fx_rate", "knf": None},
    {"label": "JPYPLN", "stooq": "jpypln", "yf": "JPYPLN=X", "type": "fx_rate", "knf": None},
]

# --- FUNKCJE POMOCNICZE (Google Drive & Load) ---

def _get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    

    credentials_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        
    return build("drive", "v3", credentials=creds)

def _get_zip_from_gdrive(service, zip_name):
    if zip_name in _ZIP_CACHE:
        return _ZIP_CACHE[zip_name]
    
    from googleapiclient.http import MediaIoBaseDownload
    
    # Znajdź plik
    q = f"name='{zip_name}' and '{FOLDER_ID}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    if not files:
        return None
    
    file_id = files[0]['id']
    buf = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    
    _ZIP_CACHE[zip_name] = buf.getvalue()
    return _ZIP_CACHE[zip_name]

# --- FUNKCJE POBIERANIA (YF / KNF) ---

def _fetch_knf_data(subfund_id, start_date):
    url = "https://wybieramfundusze-api.knf.gov.pl/v1/valuations"
    start_date = pd.to_datetime(start_date).normalize()
    
    params = {
        "subfundId": subfund_id,
        "dateFrom": start_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        # Z Twojego logu wynika, że dane są w słowniku pod kluczem 'content'
        if isinstance(data, dict) and 'content' in data:
            items = data['content']
        elif isinstance(data, list):
            items = data
        else:
            logging.warning(f"KNF: Nieoczekiwany format danych dla {subfund_id}")
            return None

        records = []
        for item in items:
            # Używamy nazw pól prosto z Twojego logu: 'date' i 'valuation'
            d_val = item.get('date')
            v_val = item.get('valuation')
            
            if d_val and v_val is not None:
                v_date = pd.to_datetime(d_val).normalize()
                if v_date > start_date:
                    records.append({
                        'Data': v_date,
                        'Otwarcie': v_val,
                        'Najwyzszy': v_val,
                        'Najnizszy': v_val,
                        'Zamkniecie': v_val
                    })
        
        if not records:
            logging.info(f"KNF: Brak rekordów nowszych niż {start_date.date()}")
            return None
            
        df_new = pd.DataFrame(records)
        return df_new.sort_values('Data')

    except Exception as e:
        logging.error(f"KNF API Error (ID {subfund_id}): {e}")
        return None

def _fetch_yfinance_data(ticker_yf, start_date):
    try:
        df = yf.download(ticker_yf, start=start_date, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df = df.droplevel(1, axis=1)
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Data', 'Open': 'Otwarcie', 'High': 'Najwyzszy', 'Low': 'Najnizszy', 'Close': 'Zamkniecie'})
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df[FINAL_COLS]
    except Exception as e:
        logging.warning(f"YFinance Error ({ticker_yf}): {e}")
        return None

# --- LOGIKA WALIDACJI (Twoja funkcja load_csv) ---

def load_csv_and_validate(df, label):
    """Przetwarza i waliduje DataFrame zgodnie z Twoimi regułami."""
    if df is None or df.empty: return None
    
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').set_index('Data')
    
    # 1. Staleness check (10 dni)
    newest_date = df.index.max()
    if (dt.now() - newest_date).days > 10:
        logging.warning(f"[{label}] Dane przestarzałe (>10 dni). Odrzucam.")
        return None
        
    # 2. Continuity check (30 dni)
    date_diffs = df.index.to_series().diff().dt.days
    breaks = date_diffs[date_diffs > 30].index
    if not breaks.empty:
        df = df.loc[df.index > breaks[-1]]
        logging.info(f"[{label}] Wykryto lukę >30 dni. Przycięto dane do {breaks[-1]}.")
        
    return df

# --- GŁÓWNY PROCES ---

def run_update():
    service = _get_drive_service()
    
    for row in TICKERS_TABLE:
        label = row['label']
        ticker_stooq = row['stooq']
        data_type = row['type']
        
        logging.info(f"--- Przetwarzanie: {label} ---")
        
        # 1. Wybierz paczkę ZIP
        zip_name = "d_pl_txt.zip" if data_type in ["index_pl", "fund_pl"] else "d_world_txt.zip"
        zip_content = _get_zip_from_gdrive(service, zip_name)
        
        # 2. Ekstrakcja z ZIP
        df_hist = None
        if zip_content:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                search = f"{ticker_stooq.lower()}.txt"
                target = next((f for f in z.namelist() if f.lower().endswith(search)), None)
                if target:
                    with z.open(target) as f:
                        df_hist = pd.read_csv(f)
                        df_hist = df_hist[COL_MAP_STOOQ.keys()].rename(columns=COL_MAP_STOOQ)
                        df_hist['Data'] = pd.to_datetime(df_hist['Data'], format='%Y%m%d')

        # 3. Update (YF / KNF)
        df_new = None
        last_date = df_hist['Data'].max() if df_hist is not None else pd.Timestamp(DATA_START)
        
        if row['yf']:
            df_new = _fetch_yfinance_data(row['yf'], last_date)
        elif row['knf']:
            df_new = _fetch_knf_data(row['knf'], last_date)
            
        # 4. Łączenie
        df_final = pd.concat([df_hist, df_new], ignore_index=True) if df_hist is not None else df_new
        
        if df_final is not None and not df_final.empty:
            # Usuń duplikaty i zachowaj tylko potrzebne kolumny
            df_final = df_final.drop_duplicates(subset='Data', keep='last')
            
            # Walidacja
            df_validated = load_csv_and_validate(df_final, label)
            
            if df_validated is not None:
                out_path = os.path.join(DATA_DIR, f"{label.lower()}.csv")
                df_validated.to_csv(out_path)
                logging.info(f"SUKCES: Zapisano {label} do {out_path}")
        else:
            logging.error(f"PORAŻKA: Brak danych dla {label}")

if __name__ == "__main__":
    run_update()