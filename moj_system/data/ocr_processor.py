# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:50:13 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/data/ocr_processor.py
================================
Moduł do pobierania zaszyfrowanych plików ZIP, ekstrakcji i przetwarzania OCR plików PDF.
Zastępuje stary skrypt `Download-ppe-data-v3.py`.
"""

import os
import io
import re
import sys
import tempfile
import logging
import requests
import pyzipper
import cv2
import pytesseract
import numpy as np
import pandas as pd
import gc  # Garbage Collector
from pdf2image import convert_from_path, pdfinfo_from_path

# Import naszego nowego klienta GDrive
try:
    from moj_system.data.gdrive import GDriveClient
    _GDRIVE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Brak modułu GDriveClient: {e}.")
    _GDRIVE_AVAILABLE = False


# --- Konfiguracja (przeniesiona z globalnego zasięgu) ---
ROW_PATTERN = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)")


def setup_logging():
    log_file = "ocr_download.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )


def fix_ocr_number(value: str) -> tuple[str, bool]:
    """Ulepszona funkcja czyszcząca liczby w polskim formacie dziesiętnym."""
    if not value or not isinstance(value, str):
        return value, False
    
    original = value.strip()
    clean = re.sub(r"[^\d,.]", "", original)
    fixed = clean.replace('.', ',')
    
    if ',' not in fixed and len(fixed) in [3, 4]:
        fixed = f"{fixed[:-2]},{fixed[-2:]}" if len(fixed) == 4 else f"{fixed[0]},{fixed[1:]}"
            
    return fixed, (fixed != original)


def run_ocr_pipeline():
    """Główna funkcja uruchamiająca cały proces OCR."""
    setup_logging()
    
    # 1. Wczytanie zmiennych środowiskowych
    zip_url = os.getenv('ZIP_URL')
    zip_password = os.getenv('ZIP_PASSWORD')
    pdf_target_name = os.getenv('INT_FILE_NAME')
    folder_name = os.getenv('FOLDER_NAME')

    # --- POPRAWKA: Czyszczenie adresu URL ---
    if zip_url:
        zip_url = zip_url.strip().strip("'\"") # Usuwa białe znaki, a potem apostrofy i cudzysłowy
    # ----------------------------------------

    if not all([zip_url, zip_password, pdf_target_name]):
        raise ValueError("Krytyczne zmienne środowiskowe nie są ustawione.")
        
    
    # GOOGLE_CREDENTIALS już nie jest potrzebne, GDriveClient sam znajdzie plik

    if not all([zip_url, zip_password, pdf_target_name]):
        raise ValueError("Krytyczne zmienne środowiskowe (ZIP_URL, ZIP_PASSWORD, INT_FILE_NAME) nie są ustawione.")

    work_dir = "/tmp" if sys.platform != "win32" else os.getenv("TEMP")
    pdf_path = os.path.join(work_dir, pdf_target_name)

    # 2. Pobieranie i rozpakowywanie pliku
    try:
        logging.info("Pobieranie pliku ZIP...")
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        
        with pyzipper.AESZipFile(io.BytesIO(resp.content)) as zf:
            zf.setpassword(zip_password.encode('utf-8'))
            zf.extract(pdf_target_name, path=work_dir)
        logging.info(f"Pomyślnie rozpakowano {pdf_target_name}")
    except Exception as e:
        raise RuntimeError(f"Błąd podczas pobierania lub rozpakowywania PDF: {e}")

    # 3. Przetwarzanie OCR z uwzględnieniem zarządzania pamięcią
    all_rows = []
    try:
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
        logging.info(f"Przetwarzanie {total_pages} stron...")

        for i in range(1, total_pages + 1):
            images = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=300, thread_count=2)
            if not images: continue
            
            img = np.array(images[0])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            custom_config = r'--oem 3 --psm 6 -l pol+eng'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            page_count = 0
            for line in text.split('\n'):
                match = ROW_PATTERN.search(line)
                if match:
                    all_rows.append(list(match.groups()))
                    page_count += 1
            
            logging.info(f"Strona {i}: Wyodrębniono {page_count} wierszy.")
            
            # --- ZARZĄDZANIE PAMIĘCIĄ ---
            del images, img, gray, binary, text
            gc.collect() # Wymuszenie zwolnienia pamięci
            # ---------------------------

    except Exception as e:
        raise RuntimeError(f"Błąd podczas przetwarzania OCR: {e}")

    # 4. Czyszczenie danych i transformacja
    df = pd.DataFrame(all_rows, columns=['Date', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6'])
    
    correction_count = 0
    for col in df.columns[1:]:
        processed = df[col].apply(fix_ocr_number)
        df[col] = [x[0] for x in processed]
        correction_count += sum(x[1] for x in processed)

    logging.info(f"Łączna liczba zastosowanych korekt: {correction_count}")
    
    # --- NOWA LOGIKA UPLOADU Z UŻYCIEM GDRIVECLIENT ---
    if not _GDRIVE_AVAILABLE or not folder_name:
        logging.warning("Brak klienta GDrive lub nazwy folderu. Pomijam upload.")
    else:
        try:
            client = GDriveClient()
            if not client.service:
                raise ConnectionError("Nie można połączyć się z Google Drive.")
            
            # Znajdź folder (na razie po nazwie, w przyszłości można po ID)
            folder_id = client.find_file_id(client.root_folder_id, folder_name) # Zakładam, że folder jest w głównym katalogu
            if not folder_id:
                raise FileNotFoundError(f"Folder '{folder_name}' nie został znaleziony na Google Drive.")

            file_name = "filtered_table.csv"
            
            # Zapisz DataFrame do tymczasowego pliku lokalnego
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv", encoding='utf-8') as tmp_file:
                df.to_csv(tmp_file, index=False, header=False, sep=";")
                tmp_file_path = tmp_file.name

            # Użyj metody upload_csv z naszego klienta
            client.upload_csv(folder_id, tmp_file_path, file_name)
            
            # Posprzątaj po sobie
            os.remove(tmp_file_path)

        except Exception as e:
            logging.error(f"Błąd interakcji z Google Drive: {e}")

    logging.info("Potok OCR zakończony pomyślnie.")

# Umożliwia uruchomienie pliku jako skryptu z terminala
if __name__ == "__main__":
    try:
        run_ocr_pipeline()
    except (ValueError, RuntimeError, FileNotFoundError, ConnectionError) as e:
        logging.error(f"Krytyczny błąd: {e}")
        sys.exit(1)