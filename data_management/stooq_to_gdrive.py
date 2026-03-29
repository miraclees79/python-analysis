import os
import logging
import asyncio
import sys
import json
from playwright.async_api import async_playwright
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Konfiguracja
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# Zmiana nazwy sekretu zgodnie z Twoim życzeniem
SERVICE_ACCOUNT_INFO = json.loads(os.environ.get('GOOGLE_CREDENTIALS', '{}'))
FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

# Lista plików do pobrania (URL oraz nazwa docelowa)
STOOQ_TARGETS = [
    {"url": "https://stooq.pl/db/d/?b=d_pl_txt", "name": "d_pl_txt.zip"},
    {"url": "https://stooq.pl/db/d/?b=d_world_txt", "name": "d_world_txt.zip"}
]

def get_gdrive_service():
    creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def upload_to_gdrive(file_path, folder_id):
    service = get_gdrive_service()
    file_name = os.path.basename(file_path)
    
    query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id)").execute()
    items = results.get('files', [])

    media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True)
    
    if items:
        file_id = items[0]['id']
        logging.info(f"Aktualizacja: {file_name} (ID: {file_id})")
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        logging.info(f"Wgrywanie nowego: {file_name}")
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()

import random

async def download_file(url, target_name):
    async with async_playwright() as p:
        # Dodajemy argumenty maskujące bota
        browser = await p.chromium.launch(headless=True, args=[
            '--disable-blink-features=AutomationControlled',
        ])
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            accept_language="pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7"
        )
        page = await context.new_page()
        
        try:
            # KROK 1: Wchodzimy najpierw na stronę główną stooq.pl, żeby dostać ciastka
            logging.info("Inicjalizacja sesji na stooq.pl...")
            await page.goto("https://stooq.pl", wait_until="networkidle")
            await asyncio.sleep(random.uniform(2, 5)) # Udajemy, że czytamy stronę
            
            logging.info(f"Pobieranie: {url}")
            
            # KROK 2: Rejestrujemy oczekiwanie na pobieranie
            async with page.expect_download(timeout=900000) as download_info:
                try:
                    # Idziemy pod link pobierania
                    await page.goto(url, wait_until="commit")
                except Exception as e:
                    if "Download is starting" in str(e):
                        logging.info("Start pobierania wykryty.")
                    else:
                        raise e
            
            download = await download_info.value
            await download.save_as(target_name)
            
            # KROK 3: Walidacja rozmiaru
            file_size = os.path.getsize(target_name)
            if file_size < 1000: # Jeśli plik ma mniej niż 1KB, to na pewno nie jest to baza danych
                logging.error(f"BŁĄD: Pobrano uszkodzony plik ({file_size} bajtów). Prawdopodobnie blokada IP.")
                # Opcjonalnie: zapiszmy co jest w środku, żeby wiedzieć co mówi Stooq
                with open(target_name, 'r') as f:
                    logging.info(f"Treść odpowiedzi serwera: {f.read()[:100]}")
                return None

            logging.info(f"SUKCES: Pobrano {target_name} ({file_size // 1024} KB)")
            return target_name
            
        except Exception as e:
            logging.error(f"Błąd przy {target_name}: {e}")
            return None
        finally:
            await browser.close()

def get_gdrive_service():
    # Naprawienie błędu file_cache
    creds = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

async def main():
    if not SERVICE_ACCOUNT_INFO or not FOLDER_ID:
        logging.error("Brak poświadczeń Google lub FOLDER_ID!")
        sys.exit(1)

    for target in STOOQ_TARGETS:
        local_file = await download_file(target['url'], target['name'])
        if local_file and os.path.exists(local_file):
            logging.info(f"Wysyłanie {local_file} na GDrive...")
            upload_to_gdrive(local_file, FOLDER_ID)
            # Opcjonalnie: usuwamy lokalny plik po wgraniu, żeby nie zajmować miejsca w runnerze
            os.remove(local_file)
        else:
            logging.warning(f"Pominięto wysyłkę dla {target['name']}")

if __name__ == "__main__":
    asyncio.run(main())