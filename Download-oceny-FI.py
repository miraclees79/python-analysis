import pandas as pd
import numpy as np
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from bs4 import BeautifulSoup
import os
import io

# Google Drive API dependencies
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

import tempfile


# Get the temporary directory
tmp_dir = tempfile.gettempdir()
print(f"Temporary directory: {tmp_dir}")

# Create a temporary file inside the temp directory # Filepath for CSV
csv_filename = os.path.join(tmp_dir, "data.csv")



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
            print(f"Failed to download {url} (Status Code: {response.status_code})")
            return False
        with open(filename, "wb") as file:
            file.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_webpage_title(url):
    # Set up Chrome options for headless mode (no GUI)
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--user-data-dir=/tmp/chrome_test_data') 
    options.add_argument('--disable-dev-shm-usage')
    
    print("URL: ", url)
    
    # Automatically download and manage ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    print("URL: ", url)
    # Open the webpage URL
    driver.get(url)
    
    # Wait for the cookie acceptance pop-up to appear
    try:
        # Wait until the button is present and clickable
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "fc-cta-consent"))
        )
        accept_button.click()  # Click the accept button
        print("Cookie pop-up accepted.")
    
    except Exception as e:
        print("No cookie acceptance pop-up found or timeout occurred:", e)
    
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
def process_data(url, url_title, filename):
    
    # Download the CSV file
    download_csv(url, filename)

    # Read the CSV into a DataFrame with proper settings
    try:
        df = pd.read_csv(filename, on_bad_lines='skip', delimiter=',', decimal='.', encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None

    if df.empty:
        print("⚠️ The DataFrame is empty.")
        return None

    # Strip any leading or trailing whitespace from the column names
    df.columns = df.columns.str.strip()

    # Ensure the date column is in datetime format
    date_column = 'Data'  # The actual name of the date column
    if date_column not in df.columns:
        print(f"⚠️ Warning: Column '{date_column}' not found. Available columns: {df.columns}")
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
        print("⚠️ Not enough data to calculate returns.")
        return None

    # Calculate statistics for the last 66 22-day returns
    min_return, max_return, first_quartile = calculate_22_day_statistics(returns_22)

    # Get the newest 1-day and 22-day returns
    newest_1_day_return = daily_returns.iloc[-1] if len(daily_returns) > 0 else None
    newest_22_day_return = returns_22.iloc[-1] if len(returns_22) > 0 else None

    # Get the title of the webpage
    nazwa = get_webpage_title(url_title)
    if nazwa == "Wyszukiwanie symbolu - Stooq" or nazwa == "No <title> tag found":
        print("⚠️ Brak symbolu w bazie Stooq")
        return None
    
    # Save the required data into an array
    result = [
        nazwa,  # Name of the file
        min_return,  # Minimum of the last 66 22-day returns
        max_return,  # Maximum of the last 66 22-day returns
        first_quartile,  # First quartile of the last 66 22-day returns
        newest_1_day_return,  # Newest 1-day return
        newest_22_day_return  # Newest 22-day return
    ]    
    print("✅ Processed Data:", result)
    return result
  
  
  # Loop through each XXXX value
for xxxx in range(1000, 1010):
    csv_url = csv_base_url.format(xxxx)
    title_url = title_base_url.format(xxxx)

    print(f"Processing: {csv_url}")
    result = process_data(csv_url, title_url, csv_filename)

    all_results.append(result)
   # print("✅ Processed and appended Data:", all_results)
    # Delete CSV file after processing
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
    
    time.sleep(2)  # Optional delay
    
  # Convert results to DataFrame and save
cleaned_results = [row for row in all_results if row is not None]
final_df = pd.DataFrame(cleaned_results, columns=["Title", "Min 22-day Return", "Max 22-day Return", "1st Quartile", "Latest 1-day Return", "Latest 22-day Return"])
csv_data = final_df.to_csv(index=False, header=True, sep=";").encode('utf-8')
print(final_df.head())

creds = service_account.Credentials.from_service_account_file('/tmp/credentials.json', scopes=['https://www.googleapis.com/auth/drive'])

drive_service = build('drive', 'v3', credentials=creds)

# File details
file_name = "oceny.csv"

# Correctly get the folder ID (replace 'Dane' with the actual folder name)
folder_name = 'Dane'
query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
results = drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
items = results.get('files', [])

if items:
    folder_id = items[0]['id']
else:
    print(f"❌ Folder '{folder_name}' not found in Google Drive.")
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

print(f"Analysis complete. Extracted data from {len(all_results)} pages saved in 'oceny.csv'.")

