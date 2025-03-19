import pandas as pd
import numpy as np
import requests
import time
import random
import os
import io

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
print(f"Temporary directory: {tmp_dir}")

# Create a temporary file inside the temp directory # Filepath for CSV
csv_filename = os.path.join(tmp_dir, "data.csv")



# Base URLs
csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"


# List of User-Agent headers for different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
]


# List to store all results
all_results = []

# Function to download the CSV file
def download_csv(url, filename):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        logging.info(f"Downloading {url} with User-Agent: {headers['User-Agent']}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/csv" not in content_type and "text/plain" not in content_type:
            logging.warning(f"Unexpected content type: {content_type}. Proceeding anyway...")

        if not response.content.strip():
            logging.warning(f"The file from {url} is empty.")
            return False

        with open(filename, "wb") as file:
            file.write(response.content)

        if os.path.getsize(filename) == 0:
            logging.warning(f"The downloaded file {filename} is empty.")
            return False

        logging.info(f"✅ Using User-Agent: {headers['User-Agent']} successfully downloaded: {filename}"  )
        return True

    except requests.exceptions.Timeout:
        logging.error(f"❌ Timeout error: The request to {url} took too long.")
    except requests.exceptions.HTTPError as e:
        logging.error(f"❌ HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Request error: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")

    return False

# Function to calculate daily returns
def calculate_daily_returns(prices):
    return prices.pct_change().dropna()

# Function to calculate 22-day returns
def calculate_22_day_returns(prices):
    return prices.pct_change(periods=22).dropna()

# Function to calculate statistics on 22-day returns
def calculate_22_day_statistics(returns_22):
    last_66_returns = returns_22.iloc[-66:]  # Use iloc for integer-based indexing
    min_return = last_66_returns.min()
    max_return = last_66_returns.max()
    first_quartile = np.percentile(last_66_returns, 25)
    return min_return, max_return, first_quartile




# Main function to process the data
def process_data(url, numer, filename):
    
    # Download the CSV file
    download_csv(url, filename)

    # Read the CSV into a DataFrame with proper settings
    try:
        df = pd.read_csv(filename, on_bad_lines='skip', delimiter=',', decimal='.', encoding='utf-8')
    except Exception as e:
         logging.error(f"❌ Error reading CSV file: {e}")
         return None

    if df.empty:
         logging.warning("⚠️ The DataFrame is empty.")
         return None

    # Strip any leading or trailing whitespace from the column names
    df.columns = df.columns.str.strip()

    # Ensure the date column is in datetime format
    date_column = 'Data'  # The actual name of the date column
    if date_column not in df.columns:
        logging.warning(f"⚠️ Warning: Column '{date_column}' not found. Available columns: {df.columns}")
        return None

    # Convert date column
    df['date'] = pd.to_datetime(df[date_column])

    # Sort the data by date in ascending order
    df = df.sort_values(by='date')

    # Calculate daily returns from the second column (prices)
    prices_column = df.columns[1]  # Assuming the second column contains the prices
    daily_returns = calculate_daily_returns(df[prices_column])

    # Calculate 22-day returns
    returns_22 = calculate_22_day_returns(df[prices_column])

    # Check if we have enough data points
    if len(daily_returns) == 0 or len(returns_22) == 0:
        logging.warning("⚠️ Not enough data to calculate returns.")
        return None

    # Calculate statistics for the last 66 22-day returns
    min_return, max_return, first_quartile = calculate_22_day_statistics(returns_22)

    # Get the newest 1-day and 22-day returns
    newest_1_day_return = daily_returns.iloc[-1] if len(daily_returns) > 0 else None
    newest_22_day_return = returns_22.iloc[-1] if len(returns_22) > 0 else None

    
    
    # Save the required data into an array
    result = [
        numer,  # numer kolejny
        min_return,  # Minimum of the last 66 22-day returns
        max_return,  # Maximum of the last 66 22-day returns
        first_quartile,  # First quartile of the last 66 22-day returns
        newest_1_day_return,  # Newest 1-day return
        newest_22_day_return  # Newest 22-day return
    ]    
    logging.info("✅ Processed Data: %s", result)
    return result
  
  
  # Loop through each XXXX value
min_index = int(os.getenv('MIN_INDEX'))
max_index = int(os.getenv('MAX_INDEX'))
for xxxx in range(min_index, max_index):
    csv_url = csv_base_url.format(xxxx)
    

    print(f"Processing: {csv_url}")
    result = process_data(csv_url, xxxx, csv_filename)

    all_results.append(result)
  
    # Delete CSV file after processing
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    
    delay = random.uniform(0, 5)

    logging.info(f"Sleeping for {delay:.2f} seconds...")
    time.sleep(delay)

    logging.info("Done!")  # Optional delay
    
  # Convert results to DataFrame and save
cleaned_results = [row for row in all_results if row is not None]
final_df = pd.DataFrame(cleaned_results, columns=["Numer", "Min 22-day Return", "Max 22-day Return", "1st Quartile", "Latest 1-day Return", "Latest 22-day Return"])
csv_data = final_df.to_csv(index=False, header=True, sep=";").encode('utf-8')
print(final_df.head())

creds = service_account.Credentials.from_service_account_file('/tmp/credentials.json', scopes=['https://www.googleapis.com/auth/drive'])

drive_service = build('drive', 'v3', credentials=creds)

# File details
file_name = f"oceny{min_index}.csv"

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
    print(f"File '{file_name}' updated as a new version. File ID: {file_id}")
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
    print(f"File '{file_name}' created. File ID: {file_id}")

logging.info(f"Analysis complete. Extracted data from {len(cleaned_results)} pages saved in 'oceny.csv'.")

