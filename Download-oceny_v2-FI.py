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
csv_filename_w20tr = os.path.join(tmp_dir, "w20tr.csv")
csv_filename_m40tr = os.path.join(tmp_dir, "m40tr.csv")
csv_filename_s80tr = os.path.join(tmp_dir, "s80tr.csv")
csv_filename_wbbwz = os.path.join(tmp_dir, "wbbwz.csv")

# csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"

# Base URLs
csv_base_url = os.getenv('CSV_BASE_URL')
# csv_base_url = os.getenv('CSV_BASE_URL')

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

# Function to calculate returns for given periods
def calculate_returns(series, periods):
    return series.pct_change(periods=periods).iloc[-760:]  # Keep only the newest 760 observations


# Function to calculate statistics on 22-day returns
def calculate_22_day_statistics(returns_22):
    last_66_returns = returns_22.iloc[-66:]  # Use iloc for integer-based indexing
    min_return = last_66_returns.min()
    max_return = last_66_returns.max()
    first_quartile = np.percentile(last_66_returns, 25)
    return min_return, max_return, first_quartile

def load_csv(filename):
    try:
        df = pd.read_csv(filename, on_bad_lines='skip', delimiter=',', decimal='.', encoding='utf-8')
    except Exception as e:
        logging.error(f"❌ Error reading CSV file: {e}")
        return None

    if df.empty or df.columns.size == 0:
        logging.error("❌ CSV file is empty or corrupted.")
        return None

    # Strip whitespace and inspect column names
    df.columns = df.columns.str.strip()
    print("Available columns after stripping:", df.columns)

    date_column = 'Data'  # Expected date column

    # Double-check the column names for hidden characters
    if date_column not in df.columns:
        exact_matches = [col for col in df.columns if col.strip() == date_column]
        if exact_matches:
            date_column = exact_matches[0]  # If a match is found, update the column name
            logging.info(f"ℹ️ Using corrected column name: '{date_column}'")
        else:
            # If still not found, display and exit
            logging.error(f"❌ Column '{date_column}' not found after processing. Available columns: {df.columns}")
            return None

    # Check if the date column contains valid data
    if df[date_column].isnull().all():
        logging.error(f"❌ Column '{date_column}' contains only NaN values.")
        return None

    # Convert to datetime, handling errors and dropping invalid dates
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df.dropna(subset=[date_column], inplace=True)

    # Check if there are still valid dates left
    if df.empty:
        logging.error("❌ No valid dates after conversion. Data is discarded.")
        return None

    # Sort by date and set as index
    df = df.sort_values(by=date_column).set_index(date_column)

    # Check 1: Discard the data if the newest observation is older than 10 days
    newest_date = df.index.max()
    if (dt.datetime.now() - newest_date).days > 10:
        logging.warning(f"⚠️ The newest observation ({newest_date}) is older than 10 days. Data is discarded.")
        return None

    # Check 2: Discard data before the most recent break longer than 30 days
    date_diffs = df.index.to_series().diff().dt.days  # Calculate gaps in the date series
    breaks = date_diffs[date_diffs > 30].index  # Identify breaks longer than 30 days

    if not breaks.empty:
        # Keep only the data from the newest observation to the most recent break
        last_valid_date = breaks[-1]
        df = df.loc[last_valid_date:]  # Slice the DataFrame from the break to the end
        logging.info(f"ℹ️ Data contains a break longer than 30 days. Keeping data from {last_valid_date} onward.")
    
    logging.info("✅ CSV file loaded successfully and processed.")
    return df
    

# Comparison with index
def compare_to_index(filename, index2_filename):
    # Comparison with Index-2
    data_series_df = load_csv(filename)
    data_index2_df = load_csv(index2_filename)
    

    if data_index2_df is None or data_series_df is None:
        raise ValueError("Error loading one or both CSV files. Please check the logs.")
        return None

    
    # Align both datasets to have a common date range
    common_dates = data_index2_df.index.intersection(data_series_df.index)
    data_index2_df = data_index2_df.loc[common_dates]
    data_series_df = data_series_df.loc[common_dates]
    
    #output_dir = os.getenv('GITHUB_WORKSPACE', '.')
    #data_index2_df.to_csv(os.path.join(output_dir, "out_index.csv"), index=True)
    #data_series_df.to_csv(os.path.join(output_dir, "out_series.csv"), index=True)

    #data_index2_df.to_csv("out_index.csv", index=False)
    #data_series_df.to_csv("out_series.csv", index=False)
    #print("DataFrame has been saved to 'output.csv'.")
    
    


    # Get the price column name (e.g., 'Price', or any other column with prices)
    prices_column_name = data_series_df.columns[3]  # Assuming the second column contains prices  - USING Column indexed as 3 - close price
    
    #print(f"For the prices of FI, I am using column named:  {prices_column_name} ")
    #print(f"For the prices of FI, the columns ar named:  {data_series_df.columns[0] } {data_series_df.columns[1] } {data_series_df.columns[2] } {data_series_df.columns[3] }")


    prices_column_series = data_series_df[prices_column_name]

    # Get index-2 returns for comparison
    
    index_2_column_name = data_index2_df.columns[3]
    #print(f"For the index, the columns ar named:  {data_index2_df.columns[0] } {data_index2_df.columns[1] } {data_index2_df.columns[2] } {data_index2_df.columns[3] }")
    #print(f"For the prices of index, I am using column named:  {index_2_column_name} ")
    index_2 = data_index2_df[index_2_column_name]

    # index_2 = data_index2_df.iloc[:, 1]  # Assuming the second column of the index DataFrame
    index_2_returns = {
        '22-day': calculate_returns(index_2, 22),
        '66-day': calculate_returns(index_2, 66),
        '252-day': calculate_returns(index_2, 252),
    }

    # Combine the three return Series into a DataFrame
    # combined_returns_df = pd.DataFrame(index_2_returns)

    # Save the DataFrame to a CSV file, keeping the index (e.g., dates)
    # combined_returns_df.to_csv(os.path.join(output_dir,"index_2_returns.csv"), index=True)

    # Calculate returns for the comparison series (your main data series)
    comparison_returns = {
        '22-day': calculate_returns(prices_column_series, 22),
        '66-day': calculate_returns(prices_column_series, 66),
        '252-day': calculate_returns(prices_column_series, 252),
    }
    latest_252_day_return_series = comparison_returns['252-day'].iloc[-1]
    latest_252_day_return_index = index_2_returns['252-day'].iloc[-1]
    
    # Calculate the percentage where the comparison series has higher returns
    percentages = {
        period: (comparison_returns[period] > index_2_returns[period]).mean() * 100
        for period in ['22-day', '66-day', '252-day']
    }
    return percentages, latest_252_day_return_series, latest_252_day_return_index 


# Comparison with index
def compare_to_index_portfolio(filename, index1_filename, index2_filename, index3_filename):
    # Comparison with average of Indexes 1 to 3
    data_series_df = load_csv(filename)
    data_index1_df = load_csv(index1_filename)
    data_index2_df = load_csv(index2_filename)
    data_index3_df = load_csv(index3_filename)

    if data_index2_df is None or data_series_df is None or data_index1_df is None or data_index3_df is None:
        raise ValueError("Error loading one or more CSV files. Please check the logs.")
        return None
    
    # Align both datasets to have a common date range
    common_dates = data_index1_df.index.intersection(data_series_df.index).intersection(data_index2_df.index).intersection(data_index3_df.index)
    data_index1_df = data_index1_df.loc[common_dates]
    data_index2_df = data_index2_df.loc[common_dates]
    data_index3_df = data_index3_df.loc[common_dates]
    data_series_df = data_series_df.loc[common_dates]
    
    # Get the price column name (e.g., 'Price', or any other column with prices)
    prices_column_name = data_series_df.columns[1]  # Assuming the second column contains prices
    prices_column_series = data_series_df[prices_column_name]

    # Get index-1 returns for comparison
    index_1 = data_index2_df.iloc[:, 1]  # Assuming the second column of the index DataFrame
    index_1_returns = {
        '22-day': calculate_returns(index_1, 22),
        '66-day': calculate_returns(index_1, 66),
        '252-day': calculate_returns(index_1, 252),
    }

    # Get index-2 returns for comparison
    index_2 = data_index2_df.iloc[:, 1]  # Assuming the second column of the index DataFrame
    index_2_returns = {
        '22-day': calculate_returns(index_2, 22),
        '66-day': calculate_returns(index_2, 66),
        '252-day': calculate_returns(index_2, 252),
    }

    # Get index-3 returns for comparison
    index_3 = data_index3_df.iloc[:, 1]  # Assuming the second column of the index DataFrame
    index_3_returns = {
        '22-day': calculate_returns(index_3, 22),
        '66-day': calculate_returns(index_3, 66),
        '252-day': calculate_returns(index_3, 252),
    }


    # Now calculate the average return for each period across all three series
    average_returns = {}
    for period in ['22-day', '66-day', '252-day']:
        # Calculate the average for each period by taking the mean of the returns for the three series
        average_returns[period] = (
        (index_1_returns[period] + index_2_returns[period] + index_3_returns[period]) / 3
        )


    # Calculate returns for the comparison series (your main data series)
    comparison_returns = {
        '22-day': calculate_returns(prices_column_series, 22),
        '66-day': calculate_returns(prices_column_series, 66),
        '252-day': calculate_returns(prices_column_series, 252),
    }
    
    latest_252_day_return_index = average_returns['252-day'].iloc[-1]
    
    # Calculate the percentage where the comparison series has higher returns
    percentages = {
        period: (comparison_returns[period] > average_returns[period]).mean() * 100
        for period in ['22-day', '66-day', '252-day']
    }
    return percentages,  latest_252_day_return_index 


# Main function to process the data
def process_data(url, numer, filename, index1_filename, index2_filename, index3_filename, index4_filename):
    
    # download the fund data for analysi
    download_csv(url, filename)
    data_series_df = load_csv(filename)
 
    

    if  data_series_df is None:
        logging.error("Error loading fund data CSV file. Please check the logs.")
        return None

    
    # Get the price column name (e.g., 'Price', or any other column with prices)
    prices_column_name = data_series_df.columns[1]  # Assuming the second column contains prices
    prices_column_series = data_series_df[prices_column_name]

    # Calculate daily returns
    daily_returns_series = calculate_returns(prices_column_series, 1)

    # Calculate 22-day returns
    returns_22_series = calculate_returns(prices_column_series, 22)

    # Check if we have enough data points
    if len(daily_returns_series) == 0 or len(returns_22_series) == 0:
        logging.warning("⚠️ Not enough data to calculate returns.")
        return None

    # Calculate statistics for the last 66 22-day returns
    min_return, max_return, first_quartile = calculate_22_day_statistics(returns_22_series)

    # Get the newest 1-day and 22-day returns
    newest_1_day_return = daily_returns_series.iloc[-1] if len(daily_returns_series) > 0 else None
    newest_22_day_return = returns_22_series.iloc[-1] if len(returns_22_series) > 0 else None

    # compare with index1
    percentages, latest_252_day_return_series, latest_252_day_return_index  = compare_to_index(filename, index1_filename)

    # Display the results
    print("Comparison results:")
    for period, percentage in percentages.items():
        print(f"  {period} return: {percentage:.2f}% of events where the comparison series had a higher return than Index 1")
    
    # Save the required data into an array
    result = [
        numer,  # numer kolejny
        min_return,  # Minimum of the last 66 22-day returns
        max_return,  # Maximum of the last 66 22-day returns
        first_quartile,  # First quartile of the last 66 22-day returns
        newest_1_day_return,  # Newest 1-day return
        newest_22_day_return,  # Newest 22-day return
        latest_252_day_return_series # Newest 252-day return
    ]

    # Add percentages to the result array
    result.extend(list(percentages.values()))  # Add the values from percentages (22-day, 66-day, 252-day)  
    result.extend([latest_252_day_return_index]) # Add the last 252 return of index
    
    # compare with index2
    percentages, latest_252_day_return_series, latest_252_day_return_index  = compare_to_index(filename, index2_filename)

    # Display the results
    print("Comparison results:")
    for period, percentage in percentages.items():
        print(f"  {period} return: {percentage:.2f}% of events where the comparison series had a higher return than Index 2")

    # Add percentages to the result array
    result.extend(list(percentages.values()))  # Add the values from percentages (22-day, 66-day, 252-day)  
    result.extend([latest_252_day_return_index]) # Add the last 252 return of index
    
    
    # compare with index3
    percentages, latest_252_day_return_series, latest_252_day_return_index   = compare_to_index(filename, index3_filename)

    # Display the results
    print("Comparison results:")
    for period, percentage in percentages.items():
        print(f"  {period} return: {percentage:.2f}% of events where the comparison series had a higher return than Index 3")

    # Add percentages to the result array
    result.extend(list(percentages.values()))  # Add the values from percentages (22-day, 66-day, 252-day)  
    result.extend([latest_252_day_return_index]) # Add the last 252 return of index
    
    # compare with index4
    percentages, latest_252_day_return_series, latest_252_day_return_index   = compare_to_index(filename, index4_filename)

    # Display the results
    print("Comparison results:")
    for period, percentage in percentages.items():
        print(f"  {period} return: {percentage:.2f}% of events where the comparison series had a higher return than Index 4")

    # Add percentages to the result array
    result.extend(list(percentages.values()))  # Add the values from percentages (22-day, 66-day, 252-day)  
    result.extend([latest_252_day_return_index]) # Add the last 252 return of index

    # compare with average returns of indexes 1-3 (return on equal weight portfolio in each index)

    percentages, latest_252_day_return_index = compare_to_index_portfolio(filename, index1_filename, index2_filename, index3_filename)

    # Display the results
    print("Comparison results:")
    for period, percentage in percentages.items():
        print(f"  {period} return: {percentage:.2f}% of events where the comparison series had a higher return than average return on indexes 1 through 3")

    # Add percentages to the result array
    result.extend(list(percentages.values()))  # Add the values from percentages (22-day, 66-day, 252-day)  
    result.extend([latest_252_day_return_index]) # Add the last 252 return of index

    logging.info("✅ Processed Data: %s", result)
    return result



  
  # Loop through each XXXX value
min_index = int(os.getenv('MIN_INDEX'))
# min_index = 1007
max_index = int(os.getenv('MAX_INDEX'))
# max_index = 1008


# download indexes for comparispn

download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr)
download_csv('https://stooq.pl/q/d/l/?s=mwig40tr&i=d', csv_filename_m40tr)
download_csv('https://stooq.pl/q/d/l/?s=swig80tr&i=d', csv_filename_s80tr)
download_csv('https://stooq.pl/q/d/l/?s=^gpwbbwz&i=d', csv_filename_wbbwz)





for xxxx in range(min_index, max_index):
    csv_url = csv_base_url.format(xxxx)
    

    logging.info(f"Processing: {csv_url}")
    result = process_data(csv_url, xxxx, csv_filename, csv_filename_w20tr, csv_filename_m40tr,csv_filename_s80tr,csv_filename_wbbwz)

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
final_df = pd.DataFrame(cleaned_results, 
                        columns=["Numer", 
                                 "Min 22-day Return", 
                                 "Max 22-day Return", 
                                 "1st Quartile", 
                                 "Latest 1-day Return", 
                                 "Latest 22-day Return",
                                 "Latest 252-day Return",
                                 "Percentage of beating WIG20TR over 22D",
                                 "Percentage of beating WIG20TR over 66D", 
                                 "Percentage of beating WIG20TR over 252D",
                                 "Latest 252-day Return - WIG20TR",
                                 "Percentage of beating mWIG40TR over 22D",
                                 "Percentage of beating mWIG40TR over 66D", 
                                 "Percentage of beating mWIG40TR over 252D",
                                 "Latest 252-day Return - mWIG40TR",
                                 "Percentage of beating sWIG80TR over 22D",
                                 "Percentage of beating sWIG80TR over 66D", 
                                 "Percentage of beating sWIG80TR over 252D",
                                 "Latest 252-day Return - sWIG80TR",
                                 "Percentage of beating GPWB-BWZ over 22D",
                                 "Percentage of beating GPWB-BWZ over 66D", 
                                 "Percentage of beating GPWB-BWZ over 252D",
                                 "Latest 252-day Return - GPWB-BWZ",
                                 "Percentage of beating AVG204080 over 22D",
                                 "Percentage of beating AVG204080 over 66D", 
                                 "Percentage of beating AVG204080 over 252D",
                                 "Latest 252-day Return - AVG204080"
                                 ])
csv_data = final_df.to_csv(index=False, header=True, sep=";").encode('utf-8')
print(final_df.head())




#credentials_path='/tmp/credentials.json'
credentials_path=os.path.join(tmp_dir, "credentials.json")

creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=['https://www.googleapis.com/auth/drive'])

drive_service = build('drive', 'v3', credentials=creds)

# File details
file_name = f"ocenyv2{min_index}.csv"

# Correctly get the folder ID (replace 'Dane' with the actual folder name)
# folder_name = os.getenv('FOLDER_NAME')
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

logging.info(f"Analysis complete. Extracted data from {len(cleaned_results)} pages saved in {file_name}.")

