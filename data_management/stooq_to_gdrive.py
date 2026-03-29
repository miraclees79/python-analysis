import os
import logging
import asyncio
import sys
import json
import random
import multiprocessing
from playwright.async_api import async_playwright
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# --- KONFIGURACJA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SCOPES = ['https://www.googleapis.com/auth/drive.file']
# Upewnij się, że ten plik istnieje w Twoim folderze temp lub podaj pełną ścieżkę
CREDENTIALS_PATH = os.path.join(os.environ.get('TEMP', os.getcwd()), "credentials.json")
FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

STOOQ_TARGETS = [
    {"url": "https://stooq.pl/db/d/?b=d_pl_txt", "name": "d_pl_txt.zip"},
    {"url": "https://stooq.pl/db/d/?b=d_world_txt", "name": "d_world_txt.zip"}
]

# --- FUNKCJE GOOGLE DRIVE ---
def get_gdrive_service():
    # Używamy metody 'from_service_account_file' bo podajesz ścieżkę do pliku
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

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

# --- FUNKCJE POBIERANIA ---
async def download_file(url, target_name):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="pl-PL"
        )
        page = await context.new_page()
        # Zwiększamy domyślny timeout dla wolnych odpowiedzi Stooq
        page.set_default_timeout(60000) 
        
        try:
            # Krok 1: Główna + Cookies
            logging.info(f"[{target_name}] Krok 1: Strona główna...")
            await page.goto("https://stooq.pl", wait_until="domcontentloaded")
            try:
                # Szukamy przycisku RODO/Cookies
                consent = page.get_by_role("button", name="Zgadzam się")
                if await consent.is_visible(timeout=5000):
                    await consent.click()
                    logging.info("Kliknięto akceptację cookies.")
            except: pass
            
            await asyncio.sleep(random.uniform(2, 4))

            # Krok 2: /db/h/ (historia jest stabilniejsza)
            logging.info(f"[{target_name}] Krok 2: Wejście do /db/h/...")
            await page.goto("https://stooq.pl/db/h/", wait_until="domcontentloaded")
            await asyncio.sleep(random.uniform(2, 4))

            # Krok 3: Pobieranie
            logging.info(f"[{target_name}] Krok 3: Inicjacja pobierania...")
            
            # Przygotowujemy "pułapkę" na plik
            download_task = page.expect_download(timeout=900000)
            
            try:
                # Używamy nawigacji, ale spodziewamy się, że rzuci błąd "Download is starting"
                await page.goto(url, wait_until="commit")
            except Exception as e:
                if "Download is starting" in str(e):
                    logging.info("Pobieranie rozpoczęte (nawigacja przerwana poprawnie).")
                else:
                    raise e
            
            # Czekamy na fizyczne zapisanie pliku
            async with download_task as download_info:
                download = await download_info.value
                await download.save_as(target_name)
            
            file_size = os.path.getsize(target_name)
            if file_size < 10000:
                logging.error(f"Plik {target_name} jest za mały ({file_size} B).")
                return None

            logging.info(f"SUKCES: {target_name} ({file_size // 1024} KB)")
            return target_name

        except Exception as e:
            logging.error(f"Błąd krytyczny dla {target_name}: {e}")
            return None
        finally:
            await browser.close()
            # BARDZO WAŻNE: Odpoczynek między d_pl a d_world, żeby uniknąć Timeout 30s
            await asyncio.sleep(random.uniform(5, 10))

async def run_sync():
    if not os.path.exists(CREDENTIALS_PATH) or not FOLDER_ID:
        logging.error(f"Brak pliku poświadczeń w {CREDENTIALS_PATH} lub brak FOLDER_ID!")
        return

    for target in STOOQ_TARGETS:
        local_file = await download_file(target['url'], target['name'])
        if local_file and os.path.exists(local_file):
            logging.info(f"Wysyłanie {local_file} na GDrive...")
            upload_to_gdrive(local_file, FOLDER_ID)
            os.remove(local_file)

def process_wrapper():
    """Punkt wejścia dla nowego procesu - tutaj tworzymy świeżą pętlę asyncio."""
    asyncio.run(run_sync())

# --- MAIN ---
if __name__ == "__main__":
    # Multiprocessing jest kluczowy w Windows przy IDE (Spyder/Jupyter)
    # Tworzy kompletnie odizolowany proces z własną pętlą event loop
    p = multiprocessing.Process(target=process_wrapper)
    p.start()
    p.join()
    logging.info("Główny proces zakończył pracę.")