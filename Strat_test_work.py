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
# %%


# Create a temporary file inside the temp directory # Filepath for CSV
csv_filename = os.path.join(tmp_dir, "data.csv")
csv_filename_w20tr = os.path.join(tmp_dir, "w20tr.csv")
csv_filename_m40tr = os.path.join(tmp_dir, "m40tr.csv")
csv_filename_s80tr = os.path.join(tmp_dir, "s80tr.csv")
csv_filename_wbbwz = os.path.join(tmp_dir, "wbbwz.csv")

# csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"

# Base URLs
csv_base_url = os.getenv('CSV_BASE_URL')


# List of User-Agent headers for different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
]


def get_folder_id(name):
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = drive_service.files().list(q=query, fields='files(id)').execute()
    items = res.get('files', [])
    return items[0]['id'] if items else None



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


    except requests.exceptions.Timeout:
        logging.error(f"Timeout while downloading {url}")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False
    


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
    


# ============================
# Indicators
# ============================

def compute_ma(series, window):
    return series.rolling(window).mean()


def compute_momentum(series, lookback=252, skip=21):
    return series.shift(skip) / series.shift(lookback) - 1


# ============================
# Performance Metrics
# ============================

def compute_metrics(equity, freq=252):

    ret = equity.pct_change().dropna()

    cagr = (equity.iloc[-1] / equity.iloc[0])**(freq/len(ret)) - 1
    vol = ret.std() * np.sqrt(freq)
    sharpe = cagr / vol if vol > 0 else 0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd
    }


# ============================
# Strategy Engine
# ============================

def run_strategy(
    df,
    price_col,
    X=0.2,
    Y=0.1,
    ma_window=200,
    use_momentum=True,
    safe_rate=0.0
):
    
    df = df.copy()
    df["price"] = df[price_col]

    # Indicators
    df["MA"] = compute_ma(df["price"], ma_window)

    if use_momentum:
        df["MOM"] = compute_momentum(df["price"])
    else:
        df["MOM"] = 1

    # Trend filter
    df["FILTER"] = (
        (df["price"] > df["MA"]) &
        (df["MOM"] > 0)
    )

    state = "OUT"
    equity = 1.0
    equity_curve = []

    M = None  # rolling max
    m = None  # rolling min

    prev_price = df["price"].iloc[0]

    for i, row in df.iterrows():

        price = row["price"]
        filter_on = row["FILTER"]

        # -------------------
        # IN Market
        # -------------------
        if state == "IN":

            M = max(M, price)

            ret = price / prev_price

            equity *= ret

            # Exit
            if (price < (1-X)*M) or not filter_on:

                state = "OUT"
                m = price

        # -------------------
        # OUT of Market
        # -------------------
        else:

            equity *= (1 + safe_rate/252)

            m = price if m is None else min(m, price)

            # Entry
            if (price > (1+Y)*m) and filter_on:

                state = "IN"
                M = price

        equity_curve.append(equity)
        prev_price = price

    df["equity"] = equity_curve

    metrics = compute_metrics(df["equity"])

    return df, metrics



def walk_forward(
    df,
    train_years=8,
    test_years=2
):

    results = []

    dates = df.index

    start = dates.min()

    while True:

        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        train = df.loc[train_start:train_end]
        test = df.loc[train_end:test_end]

        if len(test) < 200:
            break

        best = None
        best_score = -np.inf

        # Parameter grid (small!)
        for X in [0.15, 0.2, 0.25]:
            for Y in [0.05, 0.1, 0.15]:
                for ma in [150, 200, 250]:

                    _, m = run_strategy(
                        train,
                        "Zamkniecie",
                        X=X,
                        Y=Y,
                        ma_window=ma
                    )

                    score = (
                        m["CAGR"]
                        - 0.5*m["Vol"]
                        + 0.2*m["Sharpe"]
                    )

                    if score > best_score:
                        best_score = score
                        best = (X, Y, ma)

        # Run on test
        X, Y, ma = best

        test_df, test_metrics = run_strategy(
            test,
            "Zamkniecie",
            X=X,
            Y=Y,
            ma_window=ma
        )

        results.append({
            "start": train_start,
            "X": X,
            "Y": Y,
            "MA": ma,
            **test_metrics
        })

        start += pd.DateOffset(years=test_years)

    return pd.DataFrame(results)




# Main function to process the data


# Load data
download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr, 'w20t')
INDEX_W20 = load_csv(csv_filename_w20tr)

df = INDEX_W20

# Basic backtest
bt, metrics = run_strategy(df)

print("Single Run:")
print(metrics)


# Walk-forward
wf = walk_forward(df)

print("\nWalk Forward Results:")
print(wf)
print("\nAverage:")
print(wf.mean(numeric_only=True))




  











