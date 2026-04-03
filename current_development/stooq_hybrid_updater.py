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
from googleapiclient.http import MediaIoBaseUpload

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
    {"label": "NDQ100", "stooq": "^ndx", "yf": "^N225", "type": "index_world", "knf": None},
    {"label": "WIBOR1M", "stooq": "plopln1m", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "USDPLN", "stooq": "usdpln", "yf": "USDPLN=X", "type": "fx_rate", "knf": None},
    {"label": "EURPLN", "stooq": "eurpln", "yf": "EURPLN=X", "type": "fx_rate", "knf": None},
    {"label": "JPYPLN", "stooq": "jpypln", "yf": "JPYPLN=X", "type": "fx_rate", "knf": None},
    {"label": "DE10Y", "stooq": "10YDEY.B", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "PL10Y", "stooq": "10YPLY.B", "yf": None, "type": "interest_rate", "knf": None},
    {"label": "fund_2720", "stooq": "2720.n", "yf": None, "type": "index_pl", "knf": None},
    {"label": "WBBW", "stooq": "^GPWBBWZ", "yf": None, "type": "index_pl", "knf": None},
]
# fund_2720 used as MMF - classed as index_pl to ensure it always loads

# - ETFs z GPW
ETF_TICKERS = [
    {"label": "ETFBDIVPL", "stooq": "ETFBDIVPL.PL", "yf": "ETFBDIVPL.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBM40TR", "stooq": "ETFBM40TR.PL", "yf": "ETFBM40TR.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBS80TR", "stooq": "ETFBS80TR.PL", "yf": "ETFBS80TR.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBW20LV", "stooq": "ETFBW20LV.PL", "yf": "ETFBW20LV.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBW20ST", "stooq": "ETFBW20ST.PL", "yf": "ETFBW20ST.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBW20TR", "stooq": "ETFBW20TR.PL", "yf": "ETFBW20TR.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFPZUW20M40", "stooq": "ETFPZUW20M40.PL", "yf": "ETFPZUW20M40.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBCASH", "stooq": "ETFBCASH.PL", "yf": "ETFBCASH.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBTBSP", "stooq": "ETFBTBSP.PL", "yf": "ETFBTBSP.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBNDXPL", "stooq": "ETFBNDXPL.PL", "yf": "ETFBNDXPL.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBNQ2ST", "stooq": "ETFBNQ2ST.PL", "yf": "ETFBNQ2ST.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBSPXPL", "stooq": "ETFBSPXPL.PL", "yf": "ETFBSPXPL.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFSP500", "stooq": "ETFSP500.PL", "yf": "ETFSP500.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFDAX", "stooq": "ETFDAX.PL", "yf": "ETFDAX.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFAIFS", "stooq": "ETFAIFS.PL", "yf": "ETFAIFS.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFEUNM", "stooq": "ETFEUNM.PL", "yf": "ETFEUNM.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFISIJPA", "stooq": "ETFISIJPA.PL", "yf": "ETFISIJPA.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFIWDA", "stooq": "ETFIWDA.PL", "yf": "ETFIWDA.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFNATO", "stooq": "ETFNATO.PL", "yf": "ETFNATO.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFV60A", "stooq": "ETFV60A.PL", "yf": "ETFV60A.WA", "type": "etf_pl", "knf": None},
    {"label": "ETCGLDRMAU", "stooq": "ETCGLDRMAU.PL", "yf": "ETCGLDRMAU.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFSLVR", "stooq": "ETFSLVR.PL", "yf": "ETFSLVR.WA", "type": "etf_pl", "knf": None},
    {"label": "ETFBTCPL", "stooq": "ETFBTCPL.PL", "yf": "ETFBTCPL.WA", "type": "etf_pl", "knf": None},
    {"label": "ETNVIRBTCP", "stooq": "ETNVIRBTCP.PL", "yf": "ETNVIRBTCP.WA", "type": "etf_pl", "knf": None},
    {"label": "ETNVIRETH", "stooq": "ETNVIRETH.PL", "yf": "ETNVIRETH.WA", "type": "etf_pl", "knf": None},
    {"label": "ETNVIRSOL", "stooq": "ETNVIRSOL.PL", "yf": "ETNVIRSOL.WA", "type": "etf_pl", "knf": None},
    {"label": "ETNVIRXRP", "stooq": "ETNVIRXRP.PL", "yf": "ETNVIRXRP.WA", "type": "etf_pl", "knf": None},
    
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

def run_update(get_funds = True):
    service = _get_drive_service()
    if get_funds: 
        dynamic_tickers = _get_dynamic_tickers(service)
        full_tickers_table = DEFAULT_TICKERS + dynamic_tickers + ETF_TICKERS
        
        def get_folder_id(name):
            query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            res = service.files().list(q=query, fields='files(id)').execute()
            items = res.get('files', [])
            return items[0]['id'] if items else None
        FOLDER_ID = get_folder_id("Dane")
    else:
        full_tickers_table = DEFAULT_TICKERS 
    
    
    
    for row in full_tickers_table:
        label = row['label']
        ticker_stooq = row['stooq']
        data_type = row['type']
        
        if data_type in ["fund_pl", "etf_pl"] and not get_funds:
            continue
        if data_type == "fund_pl" :
            name = row['original_name']
            logging.info(f"--- Przetwarzanie: {label} --- {name} ---")
        else:
            logging.info(f"--- Przetwarzanie: {label} ------")
        
        # 1. Pobierz dane historyczne z ZIPa na GDrive
        zip_name = "d_pl_txt.zip" if data_type in ["index_pl", "fund_pl", "etf_pl"] else "d_world_txt.zip"
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
            
        # 3. Połącz i zapisz lokalnie (na Runnerze) + upload do Drive
        df_final = pd.concat([df_hist, df_new], ignore_index=True) if df_hist is not None else df_new
        
        if df_final is not None and not df_final.empty:
            df_validated = load_csv_and_validate(df_final, label)
            if df_validated is not None:
                # Nazwa pliku: zamieniamy spacje na podkreślenia dla bezpieczeństwa
                safe_name = label.replace(" ", "_").lower()
                out_path = os.path.join(DATA_DIR, f"{safe_name}.csv")
                df_validated.to_csv(out_path)
                logging.info(f"ZAPISANO: {out_path} ({len(df_validated)} wierszy)")
                
                if get_funds and data_type in ["fund_pl", "etf_pl"]:
                    # Build Drive file name
                    if data_type == "fund_pl":
                        file_name = f"historia{ticker_stooq[:4]}.csv"
                    else:
                        file_name = f"historia{ticker_stooq}.csv"

                   
                    folder_id = FOLDER_ID

                    # Check if file exists in folder
                    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
                    res = service.files().list(q=query, fields='files(id)').execute()
                    items = res.get('files', [])

                    with open(out_path, "rb") as f:
                        media = MediaIoBaseUpload(f, mimetype='text/csv', resumable=True)

                        if items:
                            file_id = items[0]['id']
                            service.files().update(fileId=file_id, media_body=media).execute()
                            logging.info(f"Updated file: {file_name}")
                        else:
                            metadata = {'name': file_name, 'parents': [folder_id]}
                            service.files().create(body=metadata, media_body=media).execute()
                            logging.info(f"Created file: {file_name}")

                    
                
                
        else:
            logging.error(f"BRAK DANYCH dla {label}")

if __name__ == "__main__":
    run_update()