import pandas as pd
import numpy as np
import requests
import time
import random
import os
import io
import datetime as dt

# Google Drive API dependencies
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

import tempfile
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

# Get the temporary directory
tmp_dir = tempfile.gettempdir()
logging.info(f"Temporary directory: {tmp_dir}")

# Create a temporary file inside the temp directory # Filepath for CSV
csv_filename = os.path.join(tmp_dir, "data.csv")


# Base URLs

csv_base_url = "https://stooq.pl/q/d/l/?s={}&i=d"

# List of User-Agent headers for different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
]


# List to store all results
all_results = []

def download_csv(url, filename, numer):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        logging.info(f"Downloading {url} with User-Agent: {headers['User-Agent']}")
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if not response.content.strip():
            logging.warning(f"The file from {url} is empty.")
            return False

        # Save file locally
        with open(filename, "wb") as f:
            f.write(response.content)

        logging.info(f"Downloaded and saved: {filename}")

        # Build Drive file name
        file_name = f"historia{numer}.csv"

        # Locate folder
        folder_name = "Dane"
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        res = drive_service.files().list(q=query, fields='files(id)').execute()
        items = res.get('files', [])

        if not items:
            logging.error(f"Folder '{folder_name}' not found in Drive.")
            return False

        folder_id = items[0]['id']

        # Check if file exists in folder
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        res = drive_service.files().list(q=query, fields='files(id)').execute()
        items = res.get('files', [])

        with open(filename, "rb") as f:
            media = MediaIoBaseUpload(f, mimetype='text/csv', resumable=True)

            if items:
                file_id = items[0]['id']
                drive_service.files().update(fileId=file_id, media_body=media).execute()
                logging.info(f"Updated file: {file_name}")
            else:
                metadata = {'name': file_name, 'parents': [folder_id]}
                drive_service.files().create(body=metadata, media_body=media).execute()
                logging.info(f"Created file: {file_name}")

        return True

    except requests.exceptions.Timeout:
        logging.error(f"Timeout while downloading {url}")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False
    






credentials_path=os.path.join(tmp_dir, "credentials.json")
creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=['https://www.googleapis.com/auth/drive'])
drive_service = build('drive', 'v3', credentials=creds)

# download indexes for comparison


download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename, 'w20t')

# separate download


id_list = [
    "ETFBDIVPL.PL",
    "ETFBM40TR.PL",
    "ETFBS80TR.PL",
    "ETFBW20LV.PL",
    "ETFBW20ST.PL",
    "ETFBW20TR.PL",
    "ETFBCASH.PL",
    "ETFBTBSP.PL",
    "ETFBNDXPL.PL",
    "ETFBNQ2ST.PL",
    "ETFBNQ3LV.PL",
    "ETFBSPXPL.PL",
    "ETFSP500.PL",
    "ETFDAX.PL",
    "ETFNATO.PL",
    "ETCGLDRMAU.PL",
    "ETFBTCPL.PL"
]

for xxxx in id_list:
    xxxx_lcase = xxxx.lower()
    csv_url = csv_base_url.format(xxxx_lcase)

    download_csv(csv_url, csv_filename, xxxx)
  
  
    # Delete CSV file after processing
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    
    delay = random.uniform(0, 5)

    logging.info(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

    logging.info("Done!")  # Optional delay
    


# credentials cleanup
if os.path.exists("/tmp/credentials.json"):
    os.remove("/tmp/credentials.json")
