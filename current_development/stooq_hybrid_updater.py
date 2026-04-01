# -*- coding: utf-8 -*-
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
COL_MAP_STOOQ = {
    '<DATE>': 'Data',
    '<OPEN>': 'Otwarcie',
    '<HIGH>': 'Najwyzszy',
    '<LOW>': 'Najnizszy',
    '<CLOSE>': 'Zamkniecie'
}
FINAL_COLS = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")
CONFIRMED_FILE_NAME = "knf_stooq_confirmed.csv"

# Cache dla ZIPów z Google Drive
_ZIP_CACHE = {}

# --- STAŁA LISTA BAZOWA (Indeksy i Waluty) ---
DEFAULT_TICKERS = [
    {"label": "WIG", "stooq": "wig", "yf": "WIG.WA", "type": "index_pl", "knf": None},
    {"label": "WIG20TR", "stooq": "wig20tr", "yf": "WIG20TR.WA", "type": "index_pl", "knf": None},
    {"label": "MWIG40TR", "stooq": "mwig40tr", "yf": "MWIG40TR.WA", "type": "index_pl", "knf": None},
    {"label": "SWIG80TR", "stooq": "swig80tr", "yf": "SWIG80TR.WA", "type": "index_pl", "knf": None},
    {"label": "TBSP", "stooq": "^tbsp", "yf": "TBSP-INDEX.WA", "type": "index_pl", "knf": None},
    {"label": "SP500", "stooq": "^spx", "yf": "^GSPC", "type": "index_world", "knf": None},
    {"label": "NK225", "stooq": "^nkx", "yf": "^N225", "type": "index_world", "knf": None},
    {"label": "WIBOR1M", "stooq": "plopln1m", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "USDPLN", "stooq": "usdpln", "yf": "USDPLN=X", "type": "fx_rate", "knf": None},
    {"label": "EURPLN", "stooq": "eurpln", "yf": "EURPLN=X", "type": "fx_rate", "knf": None},
    {"label": "JPYPLN", "stooq": "jpypln", "yf": "JPYPLN=X", "type": "fx_rate", "knf": None},
    {"label": "DE10Y", "stooq": "10YDEY.B", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "PL10Y", "stooq": "10YPLY.B", "yf": None, "type": "interest_rate", "knf": None},
]

# --- FUNKCJE GOOGLE DRIVE ---

def _get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    
    # Próba odczytu credentials z temp 
    credentials_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    

    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _get_dynamic_tickers(service):
    """Pobiera listę funduszy z knf_stooq_confirmed.csv na Google Drive."""
    logging.info(f"Pobieranie listy tickerów z {CONFIRMED_FILE_NAME}...")
    try:
        from googleapiclient.http import MediaIoBaseDownload
        
        q = f"name='{CONFIRMED_FILE_NAME}' and '{FOLDER_ID}' in parents and trashed=false"
        res = service.files().list(q=q, fields="files(id)").execute()
        files = res.get("files", [])
        
        if not files:
            logging.warning("Nie znaleziono pliku potwierdzonego na GDrive.")
            return []

        file_id = files[0]['id']
        buf = io.BytesIO()
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        
        buf.seek(0)
        df = pd.read_csv(buf)
        df = df.dropna(subset=['stooq_id'])
        
        dynamic_list = []
        for _, row in df.iterrows():
            # ZMIANA: label przyjmuje postać fund_stooqid
            sid = str(row['stooq_id']).lower()
            dynamic_list.append({
                "label": f"fund_{sid}",        # Nowy format label
                "stooq": f"{sid}.n",                  # Oryginalny id dla Stooq
                "yf": None,
                "type": "fund_pl",
                "knf": str(row['subfundId']),
                "original_name": str(row['knf_name']) # Opcjonalnie zachowujemy nazwę dla logów
            })
        
        logging.info(f"Dodano {len(dynamic_list)} funduszy z Google Drive.")
        return dynamic_list
    except Exception as e:
        logging.error(f"Błąd podczas pobierania dynamicznej listy: {e}")
        return []

def _get_zip_from_gdrive(service, zip_name):
    if zip_name in _ZIP_CACHE:
        return _ZIP_CACHE[zip_name]
    
    from googleapiclient.http import MediaIoBaseDownload
    q = f"name='{zip_name}' and '{FOLDER_ID}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    if not files:
        logging.error(f"Nie znaleziono ZIPa {zip_name} na GDrive!")
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

# --- POBIERANIE API ---

def _fetch_knf_data(subfund_id, start_date):
    url = "https://wybieramfundusze-api.knf.gov.pl/v1/valuations"
    start_date = pd.to_datetime(start_date).normalize()
    params = {"subfundId": subfund_id, "dateFrom": start_date.strftime('%Y-%m-%d')}
    
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
                    records.append({'Data': v_date, 'Otwarcie': v_val, 'Najwyzszy': v_val, 'Najnizszy': v_val, 'Zamkniecie': v_val})
        
        return pd.DataFrame(records).sort_values('Data') if records else None
    except Exception as e:
        logging.error(f"KNF API Error (ID {subfund_id}): {e}")
        return None

def _fetch_yfinance_data(ticker_yf, start_date):
    try:
        df = yf.download(ticker_yf, start=start_date, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df = df.droplevel(1, axis=1)
        df = df.reset_index().rename(columns={'Date': 'Data', 'Open': 'Otwarcie', 'High': 'Najwyzszy', 'Low': 'Najnizszy', 'Close': 'Zamkniecie'})
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)
        return df[FINAL_COLS]
    except Exception as e:
        logging.warning(f"YFinance Error ({ticker_yf}): {e}")
        return None

# --- WALIDACJA ---

def load_csv_and_validate(df, label):
    if df is None or df.empty: return None
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data').drop_duplicates(subset='Data', keep='last').set_index('Data')
    
    # 1. Staleness check
    newest_date = df.index.max()
    if (dt.now() - newest_date).days > 15: # Zwiększono do 15 dla funduszy KNF (rzadsze wyceny)
        logging.warning(f"[{label}] Dane przestarzałe. Ostatnia data: {newest_date.date()}")
        
    # 2. Continuity check
    date_diffs = df.index.to_series().diff().dt.days
    breaks = date_diffs[date_diffs > 30].index
    if not breaks.empty:
        df = df.loc[df.index > breaks[-1]]
        logging.info(f"[{label}] Przycięto dane z powodu luki >30 dni.")
        
    return df

# --- GŁÓWNY PROCES ---

def run_update():
    service = _get_drive_service()
    dynamic_tickers = _get_dynamic_tickers(service)
    full_tickers_table = DEFAULT_TICKERS + dynamic_tickers
    
    for row in full_tickers_table:
        label = row['label']
        ticker_stooq = row['stooq']
        data_type = row['type']
        
        logging.info(f"--- Przetwarzanie: {label} ---")
        
        # 1. Pobierz dane historyczne z ZIPa na GDrive
        zip_name = "d_pl_txt.zip" if data_type in ["index_pl", "fund_pl", "fx_rate", "interest_rate"] else "d_world_txt.zip"
        zip_content = _get_zip_from_gdrive(service, zip_name)
        
        df_hist = None
        if zip_content:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                # Szukamy f_{id}.txt dla funduszy lub {id}.txt dla reszty
                search_pattern = f"{ticker_stooq.lower()}.txt"
                target = next((f for f in z.namelist() if f.lower().endswith(search_pattern)), None)
                
                if target:
                    with z.open(target) as f:
                        df_raw = pd.read_csv(f)
                        df_hist = df_raw[list(COL_MAP_STOOQ.keys())].rename(columns=COL_MAP_STOOQ)
                        df_hist['Data'] = pd.to_datetime(df_hist['Data'], format='%Y%m%d')

        # 2. Pobierz aktualizację (YF lub KNF)
        last_date = df_hist['Data'].max() if df_hist is not None else pd.Timestamp(DATA_START)
        df_new = None
        
        if row.get('yf'):
            df_new = _fetch_yfinance_data(row['yf'], last_date)
        elif row.get('knf'):
            df_new = _fetch_knf_data(row['knf'], last_date)
            
        # 3. Połącz i zapisz lokalnie (na Runnerze)
        df_final = pd.concat([df_hist, df_new], ignore_index=True) if df_hist is not None else df_new
        
        if df_final is not None and not df_final.empty:
            df_validated = load_csv_and_validate(df_final, label)
            if df_validated is not None:
                # Nazwa pliku: zamieniamy spacje na podkreślenia dla bezpieczeństwa
                safe_name = label.replace(" ", "_").lower()
                out_path = os.path.join(DATA_DIR, f"{safe_name}.csv")
                df_validated.to_csv(out_path)
                logging.info(f"ZAPISANO: {out_path} ({len(df_validated)} wierszy)")
        else:
            logging.error(f"BRAK DANYCH dla {label}")

if __name__ == "__main__":
    run_update()