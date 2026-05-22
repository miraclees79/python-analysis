# -*- coding: utf-8 -*-
"""
moj_system/data/ocr_processor.py
================================
Moduł do pobierania zaszyfrowanych plików ZIP, ekstrakcji i przetwarzania OCR plików PDF.
Zastępuje stary skrypt `Download-ppe-data-v3.py`.
"""

import calendar
import gc  # Garbage Collector
import io
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import pyzipper
import requests
from pdf2image import convert_from_path, pdfinfo_from_path

# Import naszego nowego klienta GDrive
try:
    from moj_system.data.gdrive import GDriveClient

    _GDRIVE_AVAILABLE = True
except ImportError as e:
    logging.warning(msg=f"Brak modułu GDriveClient: {e}.")
    _GDRIVE_AVAILABLE = False


# --- Konfiguracja ---
ROW_PATTERN = re.compile(
    r"(\d{2}\.\d{2}\.\d{4})\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)",
)


def setup_logging() -> None:
    log_file = "ocr_download.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(filename=log_file, mode="w"), logging.StreamHandler(stream=sys.stdout)],
    )


def fix_ocr_number(
    value: str,
) -> tuple[str, bool]:
    """Ulepszona funkcja czyszcząca liczby w polskim formacie dziesiętnym."""
    if not value or not isinstance(value, str):
        return value, False

    original = value.strip()
    clean = re.sub(pattern=r"[^\d,.]", repl="", string=original)
    fixed = clean.replace(".", ",")

    if "," not in fixed and len(fixed) in [3, 4]:
        fixed = f"{fixed[:-2]},{fixed[-2:]}" if len(fixed) == 4 else f"{fixed[0]},{fixed[1:]}"

    return fixed, (fixed != original)


def sanitize_date_sequence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zaawansowany filtr dat OCR wykorzystujący metodę 'kotwicy czasowej' (anchor date).
    Wykrywa całe łańcuchy wierszy z błędnym miesiącem (np. 23.03, 24.04, 25.04, 27.03)
    i koryguje je, wymuszając chronologiczny porządek względem ostatniej 'pewnej' daty.
    """
    if df.empty or len(df) < 2:
        return df

    # Kolumna robocza z obiektami Datetime
    df["Date_dt"] = pd.to_datetime(arg=df["Date"], format="%d.%m.%Y", errors="coerce")

    corrections_made = 0

    # Inicjalizacja kotwicy: ufamy pierwszej poprawnej dacie w dokumencie
    anchor_idx = 0
    while anchor_idx < len(df) and pd.isna(df.loc[anchor_idx, "Date_dt"]):
        anchor_idx += 1

    if anchor_idx >= len(df):
        return df.drop(columns=["Date_dt"])  # Brak poprawnych dat

    anchor_date = df.loc[anchor_idx, "Date_dt"]

    for i in range(anchor_idx + 1, len(df)):
        curr_date = df.loc[i, "Date_dt"]

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
                _, last_day_of_new_month = calendar.monthrange(year=new_year, month=new_month)
                new_day = min(curr_date.day, last_day_of_new_month)

                candidate_date = curr_date.replace(year=new_year, month=new_month, day=new_day)

                # Kluczowe sprawdzenie:
                # Jeśli po COFNIĘCIU miesiąca data jest logicznym (0-20 dni) następcą naszej
                # ostatniej pewnej daty, uznajemy to za dowód na literówkę ludzką w źródle.
                diff_candidate_to_anchor = (candidate_date - anchor_date).days

                if 0 < diff_candidate_to_anchor <= 20:
                    logging.info(
                        msg=f"KOREKTA ŁAŃCUCHA OCR (wiersz {i}): "
                            f"{curr_date.strftime('%d.%m.%Y')} -> {candidate_date.strftime('%d.%m.%Y')} "
                            f"(Kotwica: {anchor_date.strftime('%d.%m.%Y')})"
                    )
                    # Aplikujemy poprawkę
                    df.loc[i, "Date_dt"] = candidate_date
                    df.loc[i, "Date"] = candidate_date.strftime("%d.%m.%Y")
                    corrections_made += 1

                    # W tym specjalnym scenariuszu PRZESUWAMY kotwicę na skorygowaną datę,
                    # ponieważ cały blok (24, 25, 26) idzie jednym ciągiem na złym miesiącu.
                    anchor_date = candidate_date

            except Exception as e:
                logging.warning(msg=f"Błąd korekty daty w wierszu {i}: {e}")

        # Scenariusz C: Data jest mniejsza niż nasza kotwica (krok wstecz w czasie)
        elif diff_from_anchor < 0:
            # OCR pomylił się lub strona z PDF była wyjęta. Zostawiamy wiersz, ale NIE ruszamy kotwicy,
            # czekając aż sekwencja wróci na właściwe tory.
            pass

    if corrections_made > 0:
        logging.info(msg=f"Łącznie skorygowano {corrections_made} błędów sekwencji dat.")

    return df.drop(columns=["Date_dt"])


def run_ocr_pipeline() -> None:
    """Główna funkcja uruchamiająca cały proces OCR, z obsługą ZIP lub bezpośredniego PDF."""
    setup_logging()
    logging.info(msg=f"Odczytana ścieżka POPPLER_PATH: {os.getenv('POPPLER_PATH')}")
    
    # 1. Wczytanie zmiennych środowiskowych
    zip_url = os.getenv("ZIP_URL")
    zip_password = os.getenv("ZIP_PASSWORD")
    pdf_target_name = os.getenv("INT_FILE_NAME")  # Nazwa pliku PDF wewnątrz ZIP-a
    gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID") # UŻYWAMY BEZPOŚREDNIO ID GŁÓWNEGO FOLDERU

    # --- AUTO-DETEKCJA POPPLERA ---
    poppler_path = None
    if sys.platform == "win32":
        poppler_path = os.getenv("POPPLER_PATH")
        if not poppler_path:
            logging.warning(msg="System Windows wykryty, ale brak zmiennej środowiskowej POPPLER_PATH!")
        else:
            logging.info(msg=f"Używam ścieżki Popplera: {poppler_path}")
        tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_exe):
            pytesseract.pytesseract.tesseract_cmd = tesseract_exe
            logging.info(msg="Tesseract skonfigurowany poprawnie.")
        else:
            logging.warning(msg=f"Nie znaleziono Tesseracta w: {tesseract_exe}")

    if zip_url:
        zip_url = zip_url.strip().strip("'\"")

    if not all([zip_url, zip_password]):
        raise ValueError("Krytyczne zmienne środowiskowe (ZIP_URL, ZIP_PASSWORD) nie są ustawione.")

    all_rows = []

    with tempfile.TemporaryDirectory() as work_dir:
        work_dir_path = Path(work_dir)
        pdf_path = work_dir_path / "processed_document.pdf"

        try:
            logging.info(msg="Pobieranie pliku z URL...")
            resp = requests.get(url=zip_url, timeout=60)
            resp.raise_for_status()
            file_content = resp.content

            is_zip = False
            try:
                with pyzipper.AESZipFile(io.BytesIO(initial_bytes=file_content)) as zf:
                    zf.setpassword(pwd=zip_password.encode("utf-8"))
                    if pdf_target_name not in zf.namelist():
                        raise FileNotFoundError(
                            f"Plik '{pdf_target_name}' nie został znaleziony wewnątrz archiwum ZIP.",
                        )

                    zf.extract(member=pdf_target_name, path=work_dir_path)
                    (work_dir_path / pdf_target_name).rename(pdf_path)

                logging.info(msg=f"Pomyślnie rozpakowano {pdf_target_name} z archiwum ZIP.")
                is_zip = True
            except (pyzipper.zipfile.BadZipFile, RuntimeError):
                logging.info(
                    msg="Plik nie jest poprawnym archiwum ZIP lub hasło jest błędne. Próbuję jako bezpośredni PDF..."
                )
                is_zip = False

            if not is_zip:
                pdf_path.write_bytes(data=file_content)
                logging.info(msg=f"Zapisano pobrany plik bezpośrednio jako {pdf_path}.")

        except Exception as e:
            raise RuntimeError(f"Błąd podczas pobierania lub przygotowywania pliku PDF: {e}")

        try:
            info = pdfinfo_from_path(
                pdf_path=str(pdf_path), 
                userpw=zip_password, 
                poppler_path=poppler_path
            )
            total_pages = info["Pages"]
            logging.info(msg=f"Przetwarzanie {total_pages} stron...")

            for i in range(1, total_pages + 1):
                images = convert_from_path(
                    pdf_path=str(pdf_path),
                    first_page=i,
                    last_page=i,
                    dpi=300,
                    thread_count=2,
                    userpw=zip_password,
                    poppler_path=poppler_path,
                )
                if not images:
                    continue

                img = np.array(object=images[0])
                gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(
                    src=gray, 
                    thresh=0, 
                    maxval=255, 
                    type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                custom_config = r"--oem 3 --psm 6 -l pol+eng"
                text = pytesseract.image_to_string(
                    image=binary, 
                    config=custom_config
                )

                page_count = 0
                for line in text.split("\n"):
                    match = ROW_PATTERN.search(string=line)
                    if match:
                        all_rows.append(list(match.groups()))
                        page_count += 1

                logging.info(msg=f"Strona {i}: Wyodrębniono {page_count} wierszy.")

                del images, img, gray, binary, text
                gc.collect()

        except Exception as e:
            raise RuntimeError(f"Błąd podczas przetwarzania OCR: {e}")

        # --- LOGIKA UPLOADU BEZPOŚREDNIO DO GDRIVE_FOLDER_ID ---

        df = pd.DataFrame(
            data=all_rows, 
            columns=["Date", "Col2", "Col3", "Col4", "Col5", "Col6"]
        )
        df = sanitize_date_sequence(df=df)

        correction_count = 0
        for col in df.columns[1:]:
            processed = df[col].apply(func=fix_ocr_number)
            df[col] = [x[0] for x in processed]
            correction_count += sum(x[1] for x in processed)
            
        logging.info(msg=f"Łączna liczba zastosowanych korekt: {correction_count}")

        if not _GDRIVE_AVAILABLE or not gdrive_folder_id:
            logging.warning(
                msg=f"Brak klienta GDrive ({_GDRIVE_AVAILABLE}) lub GDRIVE_FOLDER_ID ({gdrive_folder_id}). Pomijam upload.",
            )
        else:
            try:
                client = GDriveClient()
                if not client.service:
                    raise ConnectionError(
                        "Nie można utworzyć serwisu Google Drive. Sprawdź credentials.json.",
                    )

                file_name = "filtered_table.csv"
                temp_path = work_dir_path / "upload_temp.csv"
                df.to_csv(
                    path_or_buf=temp_path, 
                    index=False, 
                    header=False, 
                    sep=";", 
                    encoding="utf-8"
                )

                logging.info(msg=f"Wysyłam plik {file_name} bezpośrednio do folderu docelowego o ID: {gdrive_folder_id} ...")
                
                client.upload_csv(
                    folder_id=gdrive_folder_id, 
                    local_path=str(temp_path), 
                    filename=file_name
                )

                logging.info(msg="Plik pomyślnie wysłany na Google Drive.")

            except Exception as e:
                logging.error(msg=f"Błąd interakcji z Google Drive: {e}")

    if not all_rows:
        logging.warning(msg="Nie znaleziono żadnych wierszy danych w pliku PDF.")
        df = pd.DataFrame(columns=["Date", "Col2", "Col3", "Col4", "Col5", "Col6"])
    else:
        # --- NOWA LOGIKA: Wyszukanie najnowszej daty ---
        latest_date_series = pd.to_datetime(
            arg=df["Date"], 
            format="%d.%m.%Y", 
            errors="coerce"
        )
        latest_date = latest_date_series.max()
        
        if pd.notna(obj=latest_date):
            logging.info(
                msg=f"Najnowsza data wyodrębniona z dokumentu: {latest_date.strftime(format='%Y-%m-%d')}"
            )
        # -----------------------------------------------

    logging.info(msg="Potok OCR zakończony pomyślnie.")


if __name__ == "__main__":
    try:
        run_ocr_pipeline()
    except (ValueError, RuntimeError, FileNotFoundError, ConnectionError) as e:
        logging.error(msg=f"Krytyczny błąd: {e}")
        sys.exit(1)
