import os
import io
import re
import sys
import json
import logging
import requests
import pyzipper
import cv2
import pytesseract
import numpy as np
import pandas as pd
from pdf2image import convert_from_path, pdfinfo_from_path
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account

# --- Configuration & Logging ---
LOG_FILE = "download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHeader(LOG_FILE, mode='w'), logging.StreamHandler(sys.stdout)]
)

# Regex for Date (DD.MM.YYYY) and 5 decimal/comma columns
# This is more robust than .split() for 6-column tables
ROW_PATTERN = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)")

def fix_ocr_number(value):
    """Refined numeric cleaner for Polish decimal comma format."""
    if not value or not isinstance(value, str):
        return value, False
    
    original = value.strip()
    # Remove everything except digits and comma/dot
    clean = re.sub(r"[^\d,.]", "", original)
    
    # Standardize to comma as per your requirement
    fixed = clean.replace('.', ',')
    
    # Heuristic: If 4 digits with no comma, insert it (e.g., 1234 -> 12,34)
    if ',' not in fixed:
        if len(fixed) == 4:
            fixed = f"{fixed[:2]},{fixed[2:]}"
        elif len(fixed) == 3:
            fixed = f"{fixed[0]},{fixed[1:]}"
            
    return fixed, (fixed != original)

def run_pipeline():
    # 1. Load Environment Variables
    zip_url = os.getenv('ZIP_URL')
    zip_password = os.getenv('ZIP_PASSWORD')
    pdf_target_name = os.getenv('INT_FILE_NAME')
    folder_name = os.getenv('FOLDER_NAME')
    creds_json = os.getenv('GOOGLE_CREDENTIALS')

    if not all([zip_url, zip_password, pdf_target_name, creds_json]):
        logging.error("Critical environment variables missing.")
        sys.exit(1)

    work_dir = "/tmp"
    pdf_path = os.path.join(work_dir, pdf_target_name)

    # 2. Download and Extract
    try:
        logging.info("Downloading ZIP file...")
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        
        with pyzipper.AESZipFile(io.BytesIO(resp.content)) as zf:
            zf.setpassword(zip_password.encode('utf-8'))
            zf.extract(pdf_target_name, path=work_dir)
        logging.info(f"Successfully extracted {pdf_target_name}")
    except Exception as e:
        logging.error(f"Failed to retrieve PDF: {e}")
        sys.exit(1)

    # 3. Memory-Safe OCR Processing (Page-by-Page)
    all_rows = []
    try:
        info = pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]
        logging.info(f"Processing {total_pages} pages...")

        for i in range(1, total_pages + 1):
            # Process one page at a time to stay under GHA RAM limits
            images = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=300, thread_count=2)
            if not images: continue
            
            img = np.array(images[0])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's Binarization for high contrast text
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # PSM 6 is best for uniform blocks/tables; -l pol for Polish chars
            custom_config = r'--oem 3 --psm 6 -l pol+eng'
            text = pytesseract.image_to_string(binary, config=custom_config)
            
            page_count = 0
            for line in text.split('\n'):
                match = ROW_PATTERN.search(line)
                if match:
                    all_rows.append(list(match.groups()))
                    page_count += 1
            
            logging.info(f"Page {i}: Extracted {page_count} rows.")
            del images, gray, binary # Clean up memory

    except Exception as e:
        logging.error(f"OCR Processing failed: {e}")
        sys.exit(1)

    # 4. Data Cleaning & Transformation
    df = pd.DataFrame(all_rows, columns=['Date', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6'])
    
    correction_count = 0
    for col in df.columns[1:]:
        processed = df[col].apply(fix_ocr_number)
        df[col] = processed.apply(lambda x: x[0])
        correction_count += processed.apply(lambda x: x[1]).sum()

    logging.info(f"Total corrections applied: {correction_count}")
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, header=False, sep=";", encoding='utf-8')
    csv_data = csv_buffer.getvalue()

    # 5. Google Drive Upload
    try:
        creds_dict = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        drive_service = build('drive', 'v3', credentials=creds)

        # Find Folder ID
        folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_results = drive_service.files().list(q=folder_query, fields='files(id)').execute()
        if not folder_results.get('files'):
            logging.error(f"Folder '{folder_name}' not found.")
            sys.exit(1)
        folder_id = folder_results['files'][0]['id']

        # Find or Create File
        file_name = "filtered_table.csv"
        file_query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        file_results = drive_service.files().list(q=file_query, fields='files(id)').execute()
        
        media = MediaIoBaseUpload(io.BytesIO(csv_data), mimetype='text/csv', resumable=True)
        
        if file_results.get('files'):
            file_id = file_results['files'][0]['id']
            drive_service.files().update(fileId=file_id, media_body=media).execute()
            logging.info(f"Updated existing file: {file_id}")
        else:
            meta = {'name': file_name, 'parents': [folder_id]}
            new_file = drive_service.files().create(body=meta, media_body=media, fields='id').execute()
            logging.info(f"Created new file: {new_file.get('id')}")

    except Exception as e:
        logging.error(f"Google Drive interaction failed: {e}")
        sys.exit(1)

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()