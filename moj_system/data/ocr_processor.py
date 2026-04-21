

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

# W pliku moj_system/data/ocr_processor.py

# W pliku moj_system/data/ocr_processor.py

import calendar

def sanitize_date_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zaawansowany filtr dat OCR wykorzystujący metodę 'kotwicy czasowej' (anchor date).
    Wykrywa całe łańcuchy wierszy z błędnym miesiącem (np. 23.03, 24.04, 25.04, 27.03)
    i koryguje je, wymuszając chronologiczny porządek względem ostatniej 'pewnej' daty.
    """
    if df.empty or len(df) < 2:
        return df

    # Kolumna robocza z obiektami Datetime
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
    corrections_made = 0
    
    # Inicjalizacja kotwicy: ufamy pierwszej poprawnej dacie w dokumencie
    anchor_idx = 0
    while anchor_idx < len(df) and pd.isna(df.loc[anchor_idx, 'Date_dt']):
        anchor_idx += 1
        
    if anchor_idx >= len(df):
        return df.drop(columns=['Date_dt']) # Brak poprawnych dat
        
    anchor_date = df.loc[anchor_idx, 'Date_dt']

    for i in range(anchor_idx + 1, len(df)):
        curr_date = df.loc[i, 'Date_dt']

        if pd.isna(curr_date):
            continue

        # Zawsze porównujemy do ostatniej 'pewnej' daty (kotwicy), a nie do poprzedniego wiersza,
        # który mógł być błędnie skorygowany lub mógł być błędny w OCR.
        diff_from_anchor = (curr_date - anchor_date).days

        # Scenariusz A: Normalna kontynuacja (0 do 20 dni do przodu)
        # 0 oznacza ten sam dzień (czasem zdarza się w OCR, że coś jest dwa razy).
        if 0 <= diff_from_anchor <= 20:
            # Uznajemy tę datę za poprawną i przesuwamy do niej kotwicę
            anchor_date = curr_date

        # Scenariusz B: Podejrzany, wielki skok w przód (powyżej 20 dni)
        elif diff_from_anchor > 20:
            try:
                # Próbujemy wymusić poprzedni miesiąc
                new_month = curr_date.month - 1
                new_year = curr_date.year
                
                if new_month == 0:
                    new_month = 12
                    new_year -= 1
                
                # Zabezpieczenie np. dla '31 kwietnia -> 31 marca'
                _, last_day_of_new_month = calendar.monthrange(new_year, new_month)
                new_day = min(curr_date.day, last_day_of_new_month)

                candidate_date = curr_date.replace(year=new_year, month=new_month, day=new_day)
                
                # Kluczowe sprawdzenie:
                # Jeśli po COFNIĘCIU miesiąca data jest logicznym (0-20 dni) następcą naszej
                # ostatniej pewnej daty, uznajemy to za dowód na literówkę ludzką w źródle.
                diff_candidate_to_anchor = (candidate_date - anchor_date).days
                
                if 0 < diff_candidate_to_anchor <= 20:
                    logging.info(
                        f"KOREKTA ŁAŃCUCHA OCR (wiersz {i}): "
                        f"{curr_date.strftime('%d.%m.%Y')} -> {candidate_date.strftime('%d.%m.%Y')} "
                        f"(Kotwica: {anchor_date.strftime('%d.%m.%Y')})"
                    )
                    # Aplikujemy poprawkę
                    df.loc[i, 'Date_dt'] = candidate_date
                    df.loc[i, 'Date'] = candidate_date.strftime('%d.%m.%Y')
                    corrections_made += 1
                    
                    # W tym specjalnym scenariuszu PRZESUWAMY kotwicę na skorygowaną datę,
                    # ponieważ cały blok (24, 25, 26) idzie jednym ciągiem na złym miesiącu.
                    anchor_date = candidate_date

            except Exception as e:
                logging.warning(f"Błąd korekty daty w wierszu {i}: {e}")

        # Scenariusz C: Data jest mniejsza niż nasza kotwica (krok wstecz w czasie)
        elif diff_from_anchor < 0:
            # OCR pomylił się lub strona z PDF była wyjęta. Zostawiamy wiersz, ale NIE ruszamy kotwicy, 
            # czekając aż sekwencja wróci na właściwe tory.
            pass

    if corrections_made > 0:
        logging.info(f"Łącznie skorygowano {corrections_made} błędów sekwencji dat.")

    return df.drop(columns=['Date_dt'])

def run_ocr_pipeline():
    """Główna funkcja uruchamiająca cały proces OCR, z obsługą ZIP lub bezpośredniego PDF."""
    setup_logging()
    logging.info(f"Odczytana ścieżka POPPLER_PATH: {os.getenv('POPPLER_PATH')}")
    # 1. Wczytanie zmiennych środowiskowych
    zip_url = os.getenv('ZIP_URL')
    zip_password = os.getenv('ZIP_PASSWORD')
    pdf_target_name = os.getenv('INT_FILE_NAME') # Nazwa pliku PDF wewnątrz ZIP-a
    folder_name = os.getenv('FOLDER_NAME')
    
    # --- AUTO-DETEKCJA POPPLERA ---
    poppler_path = None 
    if sys.platform == "win32":
        poppler_path = os.getenv("POPPLER_PATH")
        if not poppler_path:
            logging.warning("System Windows wykryty, ale brak zmiennej środowiskowej POPPLER_PATH!")
        else:
                logging.info(f"Używam ścieżki Popplera: {poppler_path}")
            # 2. Ścieżka do Tesseracta (DODAJ TO TERAZ)
        tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_exe):
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
            logging.info("Tesseract skonfigurowany poprawnie.")
        else:
                logging.warning(f"Nie znaleziono Tesseracta w: {tesseract_exe}")
    if zip_url:
        zip_url = zip_url.strip().strip("'\"")

    if not all([zip_url, zip_password]):
        raise ValueError("Krytyczne zmienne środowiskowe (ZIP_URL, ZIP_PASSWORD) nie są ustawione.")

    work_dir = "/tmp" if sys.platform != "win32" else os.getenv("TEMP")
    # Definiujemy ścieżkę do pliku PDF, który będzie na dysku tymczasowym
    pdf_path = os.path.join(work_dir, "processed_document.pdf")

    try:
        logging.info("Pobieranie pliku z URL...")
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        file_content = resp.content
        
        # --- NOWA LOGIKA: ZIP CZY PDF? ---
        is_zip = False
        try:
            # Próba 1: Traktujemy plik jak zaszyfrowany ZIP
            with pyzipper.AESZipFile(io.BytesIO(file_content)) as zf:
                zf.setpassword(zip_password.encode('utf-8'))
                
                # Upewniamy się, że plik PDF, którego szukamy, jest w środku
                if pdf_target_name not in zf.namelist():
                    raise FileNotFoundError(f"Plik '{pdf_target_name}' nie został znaleziony wewnątrz archiwum ZIP.")
                
                # Rozpakowujemy plik PDF na dysk
                zf.extract(pdf_target_name, path=work_dir)
                # Zmieniamy nazwę na naszą standardową, jeśli trzeba
                os.rename(os.path.join(work_dir, pdf_target_name), pdf_path)
                
            logging.info(f"Pomyślnie rozpakowano {pdf_target_name} z archiwum ZIP.")
            is_zip = True
        except (pyzipper.zipfile.BadZipFile, RuntimeError):
            # RuntimeError jest rzucany przez pyzipper przy złym haśle
            logging.info("Plik nie jest poprawnym archiwum ZIP lub hasło jest błędne. Próbuję jako bezpośredni PDF...")
            is_zip = False
        
        if not is_zip:
            # Próba 2: Zapisujemy pobraną treść bezpośrednio jako plik PDF
            with open(pdf_path, 'wb') as f:
                f.write(file_content)
            logging.info(f"Zapisano pobrany plik bezpośrednio jako {pdf_path}.")

    except Exception as e:
        raise RuntimeError(f"Błąd podczas pobierania lub przygotowywania pliku PDF: {e}")

    # --- Dalsza część bez zmian, ale z przekazaniem hasła do pdf2image ---
    all_rows = []
    try:
        # Przekazujemy hasło do obu funkcji z pdf2image
        info = pdfinfo_from_path(pdf_path, userpw=zip_password, poppler_path=poppler_path)
        total_pages = info["Pages"]
        logging.info(f"Przetwarzanie {total_pages} stron...")

        for i in range(1, total_pages + 1):
            images = convert_from_path(
                pdf_path, 
                first_page=i, 
                last_page=i, 
                dpi=300, 
                thread_count=2,
                userpw=zip_password, 
                poppler_path=poppler_path # <-- KLUCZOWY PARAMETR
            )
            # ... reszta pętli OCR bez zmian ...
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
            
            del images, img, gray, binary, text
            gc.collect()

    except Exception as e:
        # Próba posprzątania po sobie
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise RuntimeError(f"Błąd podczas przetwarzania OCR: {e}")
    finally:
        # Zawsze usuwamy plik tymczasowy po zakończeniu pracy
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    # 4. Czyszczenie danych i transformacja
    df = pd.DataFrame(all_rows, columns=['Date', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6'])
    
        
    # --- NOWA LOGIKA: Korekta sekwencji dat ---
    df = sanitize_date_sequence(df)
    # ------------------------------------------

        
    correction_count = 0
    for col in df.columns[1:]:
        processed = df[col].apply(fix_ocr_number)
        df[col] = [x[0] for x in processed]
        correction_count += sum(x[1] for x in processed)

    logging.info(f"Łączna liczba zastosowanych korekt: {correction_count}")
    
# --- LOGIKA UPLOADU (POPRAWIONA) ---
    if not _GDRIVE_AVAILABLE or not folder_name:
        logging.warning(f"Brak klienta GDrive ({_GDRIVE_AVAILABLE}) lub nazwy folderu ({folder_name}). Pomijam upload.")
    else:
        try:
            # GDriveClient domyślnie szuka credentials.json w Temp - to już mamy przetestowane!
            client = GDriveClient() 
            if not client.service:
                raise ConnectionError("Nie można utworzyć serwisu Google Drive. Sprawdź credentials.json.")
            
            # 1. Znajdź ID folderu po nazwie
            logging.info(f"Szukam folderu '{folder_name}' na Drive...")
            folder_id = client.find_file_id(client.root_folder_id, folder_name)
            
            if not folder_id:
                # Jeśli nie ma, spróbujmy poszukać w głównym katalogu (root)
                # (Zmieniam na bardziej elastyczne szukanie)
                query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
                res = client.service.files().list(q=query, fields='files(id)').execute()
                files = res.get('files', [])
                if files:
                    folder_id = files[0]['id']

            if not folder_id:
                raise FileNotFoundError(f"Folder '{folder_name}' nie został znaleziony na Twoim Drive.")

            file_name = "filtered_table.csv"
            
            # 2. Zapisz DataFrame do pliku tymczasowego
            temp_path = os.path.join(work_dir, "upload_temp.csv")
            df.to_csv(temp_path, index=False, header=False, sep=";", encoding='utf-8')

            # 3. Upload za pomocą zaufanej metody upload_csv
            logging.info(f"Wysyłam plik {file_name} do folderu {folder_name} ...")
            client.upload_csv(folder_id, temp_path, file_name)
            
            # 4. Sprzątanie
            os.remove(temp_path)
            logging.info("Plik pomyślnie wysłany na Google Drive.")

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