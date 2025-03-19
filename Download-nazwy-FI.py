import pandas as pd
import numpy as np
import requests
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
import os
import io

# Google Drive API dependencies
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

import tempfile
import logging

# Get the temporary directory
tmp_dir = tempfile.gettempdir()
print(f"Temporary directory: {tmp_dir}")

# Create a temporary file inside the temp directory # Filepath for CSV
csv_filename = os.path.join(tmp_dir, "data.csv")

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


# Base URLs
csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"
title_base_url = "https://stooq.pl/q/g/?s={}.n"




# List to store all results
all_results = []

# Function to download the CSV file
def download_csv(url, filename):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to download {url} (Status Code: {response.status_code})")
            return False
        with open(filename, "wb") as file:
            file.write(response.content)
        return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return False

def get_webpage_title(url):
    # Set up Chrome options for headless mode (no GUI)
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--user-data-dir=/tmp/chrome_test_data') 
    options.add_argument('--disable-dev-shm-usage')
    
    logging.info("URL: ", url)
    
    # Automatically download and manage ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    #    print("URL: ", url)
    # Open the webpage URL
    driver.get(url)
    
    # Wait for the cookie acceptance pop-up to appear
    try:
        # Wait until the button is present and clickable
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "fc-cta-consent"))
        )
        accept_button.click()  # Click the accept button
        logging.info("Cookie pop-up accepted.")
    
    except Exception as e:
        logging.error("No cookie acceptance pop-up found or timeout occurred: {e} ")
    
    # Wait for the page to load fully
    time.sleep(3)  
    
    # Get the rendered HTML of the page
    page_source = driver.page_source
    
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Look for the <title> tag
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        title = "No <title> tag found"
    
    # Close the Selenium driver
    driver.quit()
    
    return title


# Main function to process the data
def process_data(url_title, filename, numer):
    

    # Get the title of the webpage
    nazwa = get_webpage_title(url_title)
    if nazwa == "Wyszukiwanie symbolu - Stooq" or nazwa == "No <title> tag found":
        logging.warning("⚠️ Brak symbolu w bazie Stooq")
        return None
    
    # Save the required data into an array
    result = [
        nazwa,  # Name of the file
        numer  # numer kolejny
    ]    
    logging.info(f"✅ Processed Data: {result}")
    return result
  
  # Loop through each XXXX value
min_index = int(os.getenv('MIN_INDEX'))
max_index = int(os.getenv('MAX_INDEX'))

for xxxx in range(min_index, max_index):
    csv_url = csv_base_url.format(xxxx)
    title_url = title_base_url.format(xxxx)

    logging.info(f"Processing: {title_url}")
    result = process_data(title_url, csv_filename, xxxx)

    all_results.append(result)
    # print("✅ Processed and appended Data:", all_results)
    # Delete CSV file after processing
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    
    delay = random.uniform(0, 5)

    logging.info(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

    logging.info(f"Done!")  # Optional delay
    
  # Convert results to DataFrame and save
cleaned_results = [row for row in all_results if row is not None]
final_df = pd.DataFrame(cleaned_results, columns=["Title",  "Numer kolejny"])
csv_data = final_df.to_csv(index=False, header=True, sep=";").encode('utf-8')
logging.info(final_df.head())

creds = service_account.Credentials.from_service_account_file('/tmp/credentials.json', scopes=['https://www.googleapis.com/auth/drive'])

drive_service = build('drive', 'v3', credentials=creds)

# File details
file_name = f"nazwy{min_index}.csv"

# Correctly get the folder ID (replace 'Dane' with the actual folder name)
folder_name = 'Dane'
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

logging.info(f"Name download complete. Extracted titles from {len(all_results)} pages saved in {file_name}.")

