import os
import asyncio
import logging
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

async def test_stooq_download_async():
    url = "https://stooq.pl/db/d/?b=d_pl_txt"
    download_dir = os.path.join(os.getcwd(), "test_downloads")
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    async with async_playwright() as p:
        logging.info("Uruchamianie przeglądarki (Chromium) w trybie Async...")
        browser = await p.chromium.launch(headless=True) 
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        
        try:
            logging.info(f"Wchodzenie na stronę: {url}")
            # Zwiększony timeout i oczekiwanie na brak ruchu w sieci
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            await asyncio.sleep(5) # Symulacja "czytania" strony
            
            content = await page.content()
            if "captcha" in content.lower() or "robot" in content.lower():
                logging.warning("WYKRYTO BLOKADĘ: Strona prosi o weryfikację (CAPTCHA).")
                await page.screenshot(path="screenshot_blocked.png")
                return False

            logging.info("Próba znalezienia linku do pobrania...")
            
            # Rejestrujemy oczekiwanie na pobieranie przed kliknięciem
            async with page.expect_download(timeout=90000) as download_info:
                # Szukamy linku, który zawiera tekst '.zip'
                await page.click("a[href*='.zip']")
            
            download = await download_info.value
            save_path = os.path.join(download_dir, download.suggested_filename)
            await download.save_as(save_path)
            
            logging.info(f"SUKCES! Plik zapisany jako: {save_path}")
            return True

        except Exception as e:
            logging.error(f"BŁĄD podczas testu async: {e}")
            await page.screenshot(path="screenshot_error.png")
            return False
        finally:
            await browser.close()

if __name__ == "__main__":
    import asyncio
    
    try:
        # Próba standardowego uruchomienia
        success = asyncio.run(test_stooq_download_async())
    except RuntimeError:
        # Jeśli pętla już działa (np. w IDE lub przez inne biblioteki)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Tworzymy zadanie w działającej pętli
            task = loop.create_task(test_stooq_download_async())
            # Uwaga: w skrypcie blokującym to może wymagać dodatkowej obsługi, 
            # ale w większości środowisk to wystarczy do wykonania.
            print("\n[!] Zadanie dodane do działającej pętli event loop.")
        else:
            success = loop.run_until_complete(test_stooq_download_async())

    if 'success' in locals() and success:
        print("\n[!] Automatyzacja ZIP wydaje się MOŻLIWA.")
    else:
        print("\n[!] Sprawdź logi powyżej.")