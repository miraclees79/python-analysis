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




# Main function to process the data
def process_data(url, url_title, filename):

    # Get the title of the webpage
    nazwa = get_webpage_title(url_title)
    if nazwa == "Wyszukiwanie symbolu - Stooq" or nazwa == "No <title> tag found":
        print("⚠️ Brak symbolu w bazie Stooq")
        return None
    
    # Save the required data into an array
    result = [
        nazwa,  # Name of the file
        ]
    
    print("✅ Processed Data:", result)
    return result
  
  
  # Loop through each XXXX value
for xxxx in range(1000, 1010):
    csv_url = csv_base_url.format(xxxx)
    title_url = title_base_url.format(xxxx)

    print(f"Processing: {csv_url}")
    result = process_data(csv_url, "https://stooq.pl/q/g/?s=1006.n", csv_filename)

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
