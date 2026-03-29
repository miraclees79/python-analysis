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

async def download_file(url, target_name):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36")
        page = await context.new_page()
        
        try:
            logging.info(f"Pobieranie: {url}")
            async with page.expect_download(timeout=600000) as download_info:
                await page.goto(url, wait_until="commit")
            
            download = await download_info.value
            await download.save_as(target_name)
            logging.info(f"Zapisano lokalnie: {target_name}")
            return target_name
        except Exception as e:
            logging.error(f"Błąd pobierania {target_name}: {e}")
            return None
        finally:
            await browser.close()

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