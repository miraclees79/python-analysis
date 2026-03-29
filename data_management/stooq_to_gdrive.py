import os
import logging
import asyncio
import sys
import random
import multiprocessing
from playwright.async_api import async_playwright
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# --- KONFIGURACJA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_PATH = os.path.join(os.environ.get('TEMP', os.getcwd()), "credentials.json")
FOLDER_ID = os.environ.get('GDRIVE_FOLDER_ID')

STOOQ_TARGETS = [
    {"url": "https://stooq.pl/db/d/?b=d_pl_txt", "name": "d_pl_txt.zip"},
    {"url": "https://stooq.pl/db/d/?b=d_world_txt", "name": "d_world_txt.zip"}
]

# --- GOOGLE DRIVE ---
def get_gdrive_service():
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def upload_to_gdrive(file_path, folder_id):
    service = get_gdrive_service()
    file_name = os.path.basename(file_path)
    media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True)
    query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
    items = service.files().list(q=query).execute().get('files', [])
    if items:
        service.files().update(fileId=items[0]['id'], media_body=media).execute()
    else:
        service.files().create(body={'name': file_name, 'parents': [folder_id]}, media_body=media).execute()

# --- POBIERANIE ---
async def download_file(url, target_name):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="pl-PL",
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        try:
            # 1. Start i RODO
            
            # Krok 1: Agresywne usuwanie nakładek Google FC
            logging.info(f"[{target_name}] Krok 1: Przebijanie się przez RODO...")
            await page.goto("https://stooq.pl", wait_until="domcontentloaded")
            await asyncio.sleep(3)

            # 1. Próba kliknięcia (standardowa)
            try:
                # Google FC często używa klasy .fc-cta-consent
                consent_btn = page.locator(".fc-cta-consent, .fc-button-label").get_by_text("Zgadzam się")
                if await consent_btn.is_visible(timeout=5000):
                    await consent_btn.click()
                    logging.info("Kliknięto przycisk zgody Google FC.")
            except:
                pass

            # 2. BRUTALNE USUNIĘCIE BLOKADY (Kluczowe!)
            # Usuwamy cały root nakładki Google, który przechwytuje kliknięcia (intercepts pointer events)
            await page.evaluate("""() => {
                const elementsToRemove = [
                    '.fc-consent-root', 
                    '.fc-dialog-overlay', 
                    '.fc-dialog-container',
                    '#qc-cmp2-container',
                    '.sp_veil'
                ];
                elementsToRemove.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => el.remove());
                });
                // Przywracamy przewijanie strony, jeśli zostało zablokowane
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            }""")
            logging.info("Warstwy blokujące usunięte z kodu strony.")
            await asyncio.sleep(1)
            # 2. Bezpośrednie wejście na stronę z bazami (pominiecie zawodnego menu)
            logging.info(f"[{target_name}] Krok 2: Skok bezpośrednio do /db/d/...")
            try:
                # Wchodzimy bezpośrednio, bo RODO zostało już "zabite" w Kroku 1
                await page.goto("https://stooq.pl/db/d/", wait_until="domcontentloaded")
                await asyncio.sleep(2)
            except Exception as e:
                if "Download is starting" not in str(e): raise e

            # 3. Pobieranie właściwego pliku
            logging.info(f"[{target_name}] Krok 3: Pobieranie pliku...")
            file_id = target_name.replace(".zip", "")
            
            async with page.expect_download(timeout=900000) as download_info:
                # Szukamy linku, który w adresie (href) ma ID bazy (np. b=d_pl_txt)
                # To najpewniejszy sposób, bo etykiety tekstowe (rozmiary MB) się zmieniają
                link = page.locator(f"a[href*='b={file_id}']").first
                
                if await link.count() == 0:
                    logging.warning(f"Nie znaleziono linku dla {file_id}. Robię screenshot diagnostyczny.")
                    await page.screenshot(path=f"debug_list_{file_id}.png")
                    # Wypisujemy tekst strony, żeby zobaczyć co tam jest zamiast linków
                    content = await page.content()
                    if "Zbyt wiele zapytań" in content:
                        logging.error("Stooq zablokował IP: 'Zbyt wiele zapytań'")
                    raise Exception(f"Brak linku {file_id} na stronie /db/d/")

                logging.info(f"Link znaleziony. Klikam...")
                await link.click(force=True)

                # --- SPRAWDZANIE CAPTCHA (jeśli wyskoczy) ---
                await asyncio.sleep(2)
                captcha_input = page.locator("input[name='l']")
                if await captcha_input.is_visible(timeout=3000):
                    logging.warning("!!! WYKRYTO CAPTCHA !!!")
                    await page.locator("img[src*='captcha']").screenshot(path="captcha.png")
                    print("\n" + "!"*40)
                    print("KOD CAPTCHA WYMAGANY! Sprawdź captcha.png")
                    code = input("WPISZ KOD I ENTER: ")
                    print("!"*40 + "\n")
                    await captcha_input.fill(code)
                    await page.keyboard.press("Enter")
            
            # Rejestrujemy download
            async with page.expect_download(timeout=900000) as download_info:
                # Szukamy linku, który w ADRESIE (href) ma nasz identyfikator (np. b=d_pl_txt)
                # Skoro etykieta to "XXXMB", szukanie po href jest najpewniejsze
                link = page.locator(f"a[href*='b={file_id}']").first
                
                if await link.count() == 0:
                    # Jeśli nie znaleziono, wypisz co widać na stronie (diagnostyka)
                    all_text = await page.locator("a").all_inner_texts()
                    logging.warning(f"Nie znaleziono linku. Widoczne etykiety: {all_text[:15]}")
                    await page.screenshot(path=f"missing_{file_id}.png")
                    raise Exception(f"Nie znaleziono linku dla {file_id}")

                label = await link.inner_text()
                logging.info(f"Znaleziono link! Etykieta: {label}. Klikam...")
                await link.click(force=True)

                # --- OBSŁUGA CAPTCHA (jeśli wyskoczy po kliknięciu) ---
                await asyncio.sleep(2)
                captcha_input = page.locator("input[name='l']")
                if await captcha_input.is_visible(timeout=3000):
                    logging.warning("!!! CAPTCHA !!!")
                    await page.locator("img[src*='captcha']").screenshot(path="captcha.png")
                    print("\nPOJAWIŁA SIĘ CAPTCHA! Sprawdź captcha.png")
                    code = input("WPISZ KOD: ")
                    await captcha_input.fill(code)
                    await page.keyboard.press("Enter")

            download = await download_info.value
            await download.save_as(target_name)
            
            if os.path.getsize(target_name) > 10000:
                logging.info(f"SUKCES: {target_name}")
                return target_name
            return None

        except Exception as e:
            logging.error(f"Błąd: {e}")
            await page.screenshot(path=f"error_{target_name}.png")
            return None
        finally:
            await browser.close()
def start_sync_process():
    asyncio.run(run_sync())

async def run_sync():
    for target in STOOQ_TARGETS:
        local_file = await download_file(target['url'], target['name'])
        if local_file:
            upload_to_gdrive(local_file, FOLDER_ID)
            os.remove(local_file)

if __name__ == "__main__":
    p = multiprocessing.Process(target=start_sync_process)
    p.start()
    p.join()