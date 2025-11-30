import requests
import pyzipper  # To handle password-protected ZIP files

# Google Drive API dependencies
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload
import os
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import re
import pandas as pd
import io
import logging

# Setup logging
LOG_FILE = "download.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",  # Overwrites the log file each run
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)


#
# Function to correct misread values
#

def fix_ocr_number(value):
    """
    Fix OCR-misread numeric strings.
    Returns: (corrected_value, changed_flag)
    """
    import re

    original_value = value

    if not isinstance(value, str):
        return value, False

    v = value.strip()

    # If already correct: NN,NN
    if re.match(r"^\d{1,3},\d{1,2}$", v):
        return v, False

    digits = re.sub(r"\D", "", v)
    if digits == "":
        return value, False

    corrected = value

    # CASE 1: "1234" → "12,34"
    if len(digits) == 4:
        corrected = f"{digits[:2]},{digits[2:]}"

    # CASE 2: "745" → "7,45"
    elif len(digits) == 3:
        corrected = f"{digits[0]},{digits[1:]}"

    # CASE 3: "56" → "56,0"
    elif len(digits) == 2:
        corrected = f"{digits},0"

    else:
        # Try numeric interpretation
        try:
            num = float(v.replace(",", "."))
            if num > 100:
                corrected = f"{digits[:-2]},{digits[-2:]}"
        except:
            return value, False

    # Flag if changed
    changed = (corrected != original_value)
    return corrected, changed


# -----------------------------
# 1️⃣ DOWNLOAD ZIP FILE FROM WEB URL
# -----------------------------

zip_path = "/tmp/protected.zip"  # 🔹 Temporary storage path

zip_url = os.getenv('ZIP_URL')
zip_password = os.getenv('ZIP_PASSWORD')

response = requests.get(zip_url)
if response.status_code == 200:
    with open(zip_path, "wb") as f:
        f.write(response.content)
    logging.info(f"✅ ZIP file downloaded successfully: {zip_path}")
else:
    logging.error("❌ Failed to download ZIP file")
    exit()

# -----------------------------
# 2️⃣ EXTRACT PDF FROM PASSWORD-PROTECTED ZIP
# -----------------------------

zip_password_bytes = zip_password.encode('utf-8')
pdf_filename = os.getenv('INT_FILE_NAME')  # 🔹 Replace with the actual PDF file name inside ZIP
pdf_path = f"/tmp/{pdf_filename}"  # 🔹 Extracted PDF storage


with pyzipper.AESZipFile(zip_path, 'r') as zf:
    try:
        zf.setpassword(zip_password_bytes)
        zf.extract(pdf_filename, "/tmp/")  # Extract to home directory

        logging.info(f"✅ PDF extracted: {pdf_path}")
    except Exception as e:
        logging.error(f"❌ Error extracting PDF: {e}")
        exit()


## -----------------------------
# 3️⃣ EXTRACT TABLE FROM PDF
# -----------------------------



# Convert all pages of the PDF to images
images = convert_from_path(pdf_path)

# Store extracted data from all pages
all_table_data = []

# Regex pattern for DD.MM.YYYY date format
date_pattern = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")

# Loop through each page
for page_num, img in enumerate(images):
    logging.info(f"Processing page {page_num + 1}/{len(images)}...")

    # Convert image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Improve contrast using CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 10)

    # Remove noise using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Extract text using OCR (use PSM 4 for single-column text)
    custom_config = r'--oem 3 --psm 4'
    extracted_text = pytesseract.image_to_string(binary, config=custom_config)

    # Split text into lines
    extracted_lines = extracted_text.strip().split("\n")

    # Debug: Print extracted text from each page
    logging.info(f"Extracted text from page {page_num + 1}:")
    logging.info("\n".join(extracted_lines[:1]))  # Show first 1 lines for debugging

    # Store extracted rows for this page
    page_table_data = []

    # Process each line to filter only valid rows
    for line in extracted_lines:
        row_data = line.strip().split()  # Split row into columns based on spaces

        if row_data and date_pattern.match(row_data[0]):  # Check if first column is a date
            page_table_data.append(row_data)

    logging.info(f"Extracted {len(page_table_data)} valid rows from page {page_num + 1}")

    # Append current page's rows to the main table
    all_table_data.extend(page_table_data)




# -----------------------------
# 4️⃣ UPLOAD CSV TO GOOGLE DRIVE
# -----------------------------

# Convert to DataFrame and save with semicolon separator

# Assuming all_table_data is your data
# Your CSV data processing
df = pd.DataFrame(all_table_data)  # Assuming all_table_data is defined


# Fix numeric OCR errors and count corrections
total_corrections = 0
column_corrections = {}

for col in df.columns[1:]:  # skip date column
    col_count = 0

    new_values = []
    for val in df[col].astype(str):
        fixed, changed = fix_ocr_number(val)
        new_values.append(fixed)
        if changed:
            col_count += 1
            total_corrections += 1

    df[col] = new_values
    column_corrections[col] = col_count

# Log detailed correction statistics
for col, count in column_corrections.items():
    logging.info(f"Column '{col}': corrected {count} values")

logging.info(f"TOTAL numeric OCR corrections applied: {total_corrections}")

csv_data = df.to_csv(index=False, header=False, sep=";").encode('utf-8')


creds = service_account.Credentials.from_service_account_file('/tmp/credentials.json', scopes=['https://www.googleapis.com/auth/drive'])

drive_service = build('drive', 'v3', credentials=creds)

# File details
file_name = "filtered_table.csv"

# Correctly get the folder ID (replace  with the actual folder name)

folder_name = os.getenv('FOLDER_NAME')

query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
items = results.get('files', [])

if items:
    folder_id = items[0]['id']
else:
    logging.error(f"❌ Folder '{folder_name}' not found in Google Drive.")
    exit()

# Now use the correct folder ID in the file search query
query = f"name='{file_name}' and '{folder_id}' in parents"
results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
items = results.get('files', [])

if items:
    file_id = items[0]['id']
    # Update the existing file as a new version
    # The MediaIoBaseUpload class needs to be called directly, not as an attribute of drive_service.
    media = MediaIoBaseUpload(io.BytesIO(csv_data), mimetype='text/csv', resumable=True)
    updated_file = drive_service.files().update(fileId=file_id, media_body=media).execute()
    logging.info(f"File '{file_name}' updated as a new version. File ID: {file_id}")
else:
    # If the file doesn't exist, create it
    file_metadata = {
        'name': file_name,
        'parents': [folder_id] 
    }
    # The MediaIoBaseUpload class needs to be called directly, not as an attribute of drive_service.
    media = MediaIoBaseUpload(io.BytesIO(csv_data), mimetype='text/csv', resumable=True)
    created_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = created_file.get('id')
    logging.info(f"File '{file_name}' created. File ID: {file_id}")

logging.info(f"Filtering complete. Extracted data from {len(images)} pages saved in 'filtered_table.csv'.")

# credentials cleanup
if os.path.exists("/tmp/credentials.json"):
    os.remove("/tmp/credentials.json")
