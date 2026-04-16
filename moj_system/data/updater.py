# -*- coding: utf-8 -*-
"""
moj_system/data/updater.py
==========================
Moduł odpowiedzialny za hybrydowe aktualizowanie danych notowań.
Łączy historię z paczek ZIP (Stooq) z bieżącymi notowaniami z Yahoo Finance (indeksy)
oraz KNF API (fundusze inwestycyjne).

Zastępuje stary skrypt: `current_development/stooq_hybrid_updater.py`
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

# --- KONFIGURACJA ŚCIEŻEK ---
DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw_csv"
ZIP_DIR = DATA_ROOT / "zips"

# Zapewnienie istnienia struktury katalogów
RAW_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DIR.mkdir(parents=True, exist_ok=True)

# Mapowanie plików ZIP (Stooq)
ZIP_MAPPING = {
    "index_pl": "d_pl_txt.zip",
    "index_world": "d_world_txt.zip",
    "currencies": "d_world_txt.zip",
    "fund_pl": "d_pl_txt.zip",
    "etf_pl": "d_pl_txt.zip"
}

# Nazwy plików konfiguracyjnych na GDrive
CONFIRMED_FUNDS_FILE = "knf_stooq_confirmed.csv"
GDRIVE_DATA_FOLDER_NAME = "Dane"


class DataUpdater:
    def __init__(self, gdrive_folder_id=None, credentials_path=None):
        self.gdrive = GDriveClient(credentials_path)
        # Główny folder projektu na Drive (tam, gdzie leży plik confirmed)
        self.root_folder_id = gdrive_folder_id or os.environ.get("GDRIVE_FOLDER_ID")
        
        # ID podfolderu "Dane", do którego uploadujemy pliki CSV funduszy
        self.data_folder_id = None
        if self.gdrive.service and self.root_folder_id:
            self.data_folder_id = self._get_or_create_subfolder(GDRIVE_DATA_FOLDER_NAME)

    def _get_or_create_subfolder(self, folder_name: str) -> str:
        """Pobiera ID podfolderu (np. 'Dane') na GDrive, lub go tworzy."""
        if not self.gdrive.service: return None
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.gdrive.service.files().list(q=query, fields='files(id)').execute()
        items = results.get('files', [])
        
        if items:
            return items[0]['id']
        else:
            logging.info(f"Tworzę nowy folder '{folder_name}' na Google Drive...")
            metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.root_folder_id]
            }
            folder = self.gdrive.service.files().create(body=metadata, fields='id').execute()
            return folder.get('id')

    # =========================================================================
    # LOKALNA OBSŁUGA PLIKÓW ZIP STOOQ
    # =========================================================================
    
    def _get_zip_content(self, zip_type: str) -> bytes:
        """Pobiera zawartość ZIPa z lokalnego dysku lub z Google Drive."""
        zip_name = ZIP_MAPPING.get(zip_type)
        if not zip_name: return None
        
        local_path = ZIP_DIR / zip_name
        
        # 1. Próba odczytu lokalnego
        if local_path.exists():
            logging.debug(f"Używam lokalnego pliku ZIP: {local_path}")
            return local_path.read_bytes()
        
        # 2. Próba pobrania z Drive
        if self.root_folder_id and self.gdrive.service:
            logging.info(f"Nie znaleziono lokalnie. Pobieram {zip_name} z Google Drive...")
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
                    
                    # Zapisz lokalnie jako cache na przyszłość
                    local_path.write_bytes(content)
                    return content
                except Exception as e:
                    logging.error(f"Błąd podczas pobierania ZIP z GDrive: {e}")
        
        logging.error(f"Nie znaleziono pliku {zip_name} lokalnie ani na Drive.")
        return None

    def _extract_from_zip(self, zip_data: bytes, stooq_ticker: str) -> pd.DataFrame | None:
        """Wyciąga konkretny plik .txt z danych ZIP Stooq."""
        if not zip_data: return None
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                # Szukamy pliku kończącego się na <ticker>.txt
                search_name = f"{stooq_ticker.lower()}.txt"
                target_file = next((f for f in z.namelist() if f.lower().endswith(search_name)), None)
                
                if target_file:
                    with z.open(target_file) as f:
                        df = pd.read_csv(f)
                        col_map = {
                            '<DATE>': 'Data', '<OPEN>': 'Otwarcie', '<HIGH>': 'Najwyzszy',
                            '<LOW>': 'Najnizszy', '<CLOSE>': 'Zamkniecie', '<VOL>': 'Wolumen'
                        }
                        df = df[list(col_map.keys())].rename(columns=col_map)
                        df['Data'] = pd.to_datetime(df['Data'], format='%Y%m%d')
                        return df
        except Exception as e:
            logging.error(f"Błąd ekstrakcji {stooq_ticker} z ZIP: {e}")
        return None


    # =========================================================================
    # POBIERANIE DANYCH Z ZEWNĘTRZNYCH API
    # =========================================================================

    def _fetch_yfinance_data(self, ticker_yf: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
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
            # YF nie zawsze zwraca wszystkie kolumny dla każdego assetu, upewniamy się że mamy to czego szukamy
            cols = ['Data', 'Otwarcie', 'Najwyzszy', 'Najnizszy', 'Zamkniecie']
            return df[[c for c in cols if c in df.columns]]
        except Exception as e:
            logging.warning(f"YFinance Error ({ticker_yf}): {e}")
            return None

    def _fetch_knf_data(self, subfund_id: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
        """Pobiera wyceny funduszu z oficjalnego API KNF."""
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


    # =========================================================================
    # GŁÓWNA LOGIKA AKTUALIZACJI
    # =========================================================================

    def _validate_and_clean(self, df: pd.DataFrame, label: str) -> pd.DataFrame | None:
        """Czyszczenie danych i sprawdzanie luk (odpowiednik load_csv_and_validate ze starego kodu)."""
        if df is None or df.empty: return None
        
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.sort_values('Data').drop_duplicates(subset='Data', keep='last').set_index('Data')
        
        # 1. Staleness check
        newest_date = df.index.max()
        if (dt.now() - newest_date).days > 15: # Zwiększono do 15 dla funduszy KNF (rzadsze wyceny)
            logging.debug(f"[{label}] Dane mogą być przestarzałe. Ostatnia data: {newest_date.date()}")
            
        # 2. Continuity check
        date_diffs = df.index.to_series().diff().dt.days
        breaks = date_diffs[date_diffs > 30].index
        if not breaks.empty:
            df = df.loc[df.index > breaks[-1]]
            logging.info(f"[{label}] Przycięto dane z powodu luki >30 dni.")
            
        return df.reset_index()


    def update_ticker(self, label: str, stooq_ticker: str, yf_ticker: str = None, 
                      knf_id: str = None, zip_type: str = "index_pl", 
                      upload_to_drive: bool = False):
        """Aktualizuje jeden konkretny walor i opcjonalnie wysyła go na GDrive."""
        logging.info(f"--- Przetwarzanie: {label} ---")
        
        # 1. Historia ze Stooq (ZIP)
        zip_data = self._get_zip_content(zip_type)
        df_hist = self._extract_from_zip(zip_data, stooq_ticker) if zip_data else None
        
        # 2. Nowe dane z API (YFinance lub KNF)
        last_date = df_hist['Data'].max() if df_hist is not None else pd.Timestamp("1990-01-01")
        df_new = None
        
        if yf_ticker:
            df_new = self._fetch_yfinance_data(yf_ticker, last_date)
        elif knf_id:
            df_new = self._fetch_knf_data(knf_id, last_date)

        # 3. Łączenie
        if df_hist is not None and df_new is not None:
            df_final = pd.concat([df_hist, df_new], ignore_index=True)
        elif df_hist is not None:
            df_final = df_hist
        else:
            df_final = df_new

        # 4. Czyszczenie i Zapis
        df_validated = self._validate_and_clean(df_final, label)
        if df_validated is not None:
            safe_name = label.replace(" ", "_").lower()
            out_path = RAW_DIR / f"{safe_name}.csv"
            df_validated.to_csv(out_path, index=False)
            logging.info(f"ZAPISANO: {out_path.name} ({len(df_validated)} wierszy)")
            
            # 5. Opcjonalny Upload na GDrive (dla Funduszy / ETF)
            if upload_to_drive and self.gdrive.service and self.data_folder_id:
                # Odtworzenie nazewnictwa ze starego skryptu
                prefix = "historia"
                if zip_type == "fund_pl":
                    file_name = f"{prefix}{stooq_ticker[:4]}.csv" # fundusze miały obcięte nazwy
                else:
                    file_name = f"{prefix}{stooq_ticker}.csv"
                    
                try:
                    self.gdrive.upload_csv(self.data_folder_id, str(out_path), file_name)
                except Exception as e:
                    logging.error(f"Nie udało się wysłać {file_name} na GDrive: {e}")
                    
            return True
        else:
            logging.error(f"BRAK DANYCH dla {label}")
            return False


    def run_full_update(self, asset_registry: dict, get_funds: bool = True):
        """Przechodzi przez cały rejestr i aktualizuje co trzeba."""
        logging.info(f"Rozpoczynam pełną aktualizację danych (Pobieranie funduszy KNF: {get_funds})...")
        
        # 1. POBIERANIE LISTY FUNDUSZY Z GDRIVE
        dynamic_funds = []
        if get_funds and self.gdrive.service and self.root_folder_id:
            df_confirmed = self.gdrive.download_csv(self.root_folder_id, CONFIRMED_FUNDS_FILE)
            if df_confirmed is not None and not df_confirmed.empty:
                for _, row in df_confirmed.dropna(subset=['stooq_id']).iterrows():
                    sid = str(row['stooq_id']).lower()
                    dynamic_funds.append({
                        "label": f"fund_{sid}",
                        "stooq_ticker": f"{sid}.n",
                        "knf_id": str(row['subfundId']),
                        "zip_type": "fund_pl"
                    })
                logging.info(f"Znaleziono {len(dynamic_funds)} potwierdzonych funduszy.")

        # 2. AKTUALIZACJA GŁÓWNEGO REJESTRU (Indeksy i Waluty)
        for name, cfg in asset_registry.items():
            if cfg.get("type") == "portfolio": continue
            
            # POPRAWKA: Przekazujemy yf_ticker bezpośrednio z konfiguracji!
            yf_ticker = cfg.get("yf_ticker")
            stooq_ticker = cfg.get("ticker", name)
            
            z_type = "index_world" if cfg.get("source") == "drive" or name in ["SP500", "Nikkei225"] else "index_pl"
            if "USDPLN" in name or "EURPLN" in name or "JPYPLN" in name: z_type = "currencies"

            self.update_ticker(
                label=name, 
                stooq_ticker=stooq_ticker, 
                yf_ticker=yf_ticker,      # <-- Kluczowa poprawka! Teraz yFinance zadziała.
                zip_type=z_type,
                upload_to_drive=False
            )
            
        # Specjalne walory, niezbędne do działania MMF i bramek
        self.update_ticker("MMF", "2720.n", knf_id="195983", zip_type="fund_pl")
        self.update_ticker("WIBOR1M", "plopln1m", zip_type="currencies")
        self.update_ticker("PL10Y", "10YPLY.B", zip_type="index_pl")
        self.update_ticker("DE10Y", "10YDEY.B", zip_type="index_pl")

        # 3. AKTUALIZACJA FUNDUSZY KNF Z GDRIVE
        if get_funds:
            logging.info("Aktualizacja funduszy KNF i eksport na GDrive...")
            for fund in dynamic_funds:
                self.update_ticker(
                    label=fund["label"],
                    stooq_ticker=fund["stooq_ticker"],
                    knf_id=fund["knf_id"],
                    zip_type=fund["zip_type"],
                    upload_to_drive=True
                )