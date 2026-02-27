import pandas as pd
import numpy as np
import requests
#import time
import random
import os
#import io
import datetime as dt

import matplotlib.pyplot as plt



# Google Drive API dependencies
#from googleapiclient.discovery import build
#from google.oauth2 import service_account
#from googleapiclient.http import MediaIoBaseUpload

import tempfile
import logging
from pprint import pformat



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
        return True

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
    

def prepare_cash_returns(cash_df, price_col="Zamkniecie"):
    cash = cash_df.copy()

    cash["cash_price"] = cash[price_col]

    cash["cash_ret"] = cash["cash_price"].pct_change()

    cash = cash[["cash_ret"]].dropna()

    return cash


# ============================
# Indicators
# ============================



def compute_momentum(series, lookback=252, skip=21):
    return series.shift(skip) / series.shift(lookback) - 1


    
# ============================
# Performance Metrics
# ============================

def compute_metrics(equity, freq=252):

    ret = equity.pct_change().dropna()

    years = len(ret) / freq
    cagr = (equity.iloc[-1]/equity.iloc[0])**(1/years)-1
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

def run_strategy_with_trades(
    df,
    price_col="price",

    X=0.1,
    Y=0.1,

    fast=50,
    slow=200,

    vol_window=20,
    target_vol=0.10,
    max_leverage=1.0,

    position_mode="vol_entry",  
    # options: "vol_entry", "vol_dynamic", "full"

    use_momentum=False,
    cash_df=None,          # NEW
    safe_rate=0.0 # may be adjusted later for realism with a money market fund series, keep zero for now
):

    df = df.copy()
    df["price"] = df[price_col]

    # Merge cash returns if provided
    if cash_df is not None:

        cash = prepare_cash_returns(cash_df)

        df = df.merge(
            cash,
            left_index=True,
            right_index=True,
            how="left"
            )

        # Forward-fill missing cash returns
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()

    else:
        df["cash_ret"] = safe_rate / 252
    
    
    if df["cash_ret"].isna().all():
        logging.warning("Cash series missing — falling back to flat safe_rate")
        df["cash_ret"] = safe_rate / 252

    # -----------------------
    # Indicators
    # -----------------------

    min_len = max(vol_window, slow, fast, 252) + 5

    if len(df) < min_len:
        return None, None, pd.DataFrame()

    # Returns
    df["ret"] = df["price"].pct_change()

    # Volatility
    vol = df["ret"].rolling(vol_window).std() * np.sqrt(252)
    df["vol"] = vol.shift(1)

    # Dual MA
    df["ma_fast"] = df["price"].rolling(fast).mean().shift(1)
    df["ma_slow"] = df["price"].rolling(slow).mean().shift(1)

    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    # Momentum
    if use_momentum:
        df["MOM"] = compute_momentum(df["price"]).shift(1)
    else:
        df["MOM"] = 1

    df.dropna(inplace=True)

    # -----------------------
    # State
    # -----------------------

    equity = 1.0
    equity_curve = []

    position = 0.0
    entry_pos = None
    entry_price = None
    entry_date = None
    entry_reason = None 
    M = None
    m = None

    trades = []

    def calc_position(vol):

        # Mode 3: Always fully invested
        if position_mode == "full":
            return 1.0

        # Vol targeting
        if pd.notna(vol) and vol > 0:
            pos = target_vol / vol
        else:
            pos = 1.0

        return min(pos, max_leverage)

    # -----------------------
    # Main Loop
    # -----------------------

    for i, row in df.iterrows():

        price = row["price"]
        ret = row["ret"]
        cash_ret = row["cash_ret"]

        trend = row["trend"]
        mom = row["MOM"]
        vol = row["vol"]

        filter_on = (trend == 1) and (mom > 0)

        # Update equity
        if position > 0:
            equity *= (1 + position * ret + (1 - position) * cash_ret)
        else:
            equity *= (1 + cash_ret)

        exit_reason = None
        
        
        # Stop loss 
        if position > 0:

            dd = (price - entry_price) / entry_price

            if dd < -X:
                exit_reason = "STOP"

        # Dynamic volatility targeting
        if position > 0 and position_mode == "vol_dynamic":

            new_pos = calc_position(vol)

            # Optional: smooth / reduce turnover
            position = new_pos

        # Trend / breakout exit
        if position > 0:

            # Update running max
            M = max(M, price) if M is not None else price

            # Trailing stop from peak
            if price < (1 - X) * M:
                exit_reason = exit_reason or "TRAIL_STOP"

            # Filter turns off
            elif not filter_on:
                exit_reason = exit_reason or "FILTER_EXIT"

        # EXIT LOGGING OF TRADE
        if position > 0 and exit_reason:
            
            COST = 0.0005  # 5 bps
            trade_ret = price / entry_price - 1 - COST
            days = (i - entry_date).days

            trades.append({
                "EntryDate": entry_date,
                "ExitDate": i,
                "EntryPrice": entry_price,
                "Position": entry_pos,
                "ExitPrice": price,
                "Return": trade_ret,
                "Days": days,
                "Entry Reason": entry_reason,
                "Exit Reason": exit_reason
            })

            position = 0
            entry_price = None
            entry_date = None
            entry_reason = None
            M = None
            m = None
            entry_pos = None

        # ENTRY
        if position == 0:

            m = price if m is None else min(m, price)

            if (price > (1 + Y) * m) and filter_on:
                entry_reason = "BREAKOUT & FILTER"
                position = calc_position(vol)

                entry_price = price
                entry_date = i
                entry_pos = position   # STORE SIZE

                M = price
    



        equity_curve.append(equity)

    # -----------------------
    # FORCE EXIT AT SAMPLE END
    # -----------------------

    if position > 0 and entry_price is not None:

        last_date = df.index[-1]
        last_price = df["price"].iloc[-1]

        trade_ret = last_price / entry_price - 1
        days = (last_date - entry_date).days

        trades.append({
            "EntryDate": entry_date,
            "ExitDate": last_date,
            "EntryPrice": entry_price,
            "Position": entry_pos,
            "ExitPrice": last_price,
            "Return": trade_ret,
            "Days": days,
            "Entry Reason": entry_reason,
            "Exit Reason": "SAMPLE_END"
             })

        position = 0   

        
    df["equity"] = equity_curve

    metrics = compute_metrics(df["equity"])
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(trades)

    return df, metrics, trades_df

def walk_forward(
    df,
    cash_df,
    train_years=8,
    test_years=2,
    selected_mode="full" 
):

    results = []

    start = df.index.min()

    while True:

        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        train = df.loc[train_start:train_end]
        test = df.loc[train_end:test_end]
        cash_train = cash_df.loc[train.index.min():train.index.max()]
        cash_test = cash_df.loc[test.index.min():test.index.max()]

        if len(test) < 300:
            break

        best_params = None
        best_score = -np.inf

        # ------------------------
        # GRID SEARCH
        # ------------------------

        for X in [0.08, 0.1, 0.15, 0.2]:
            for Y in [0.02, 0.035, 0.05, 0.1]:

                for fast, slow in [(50,200),(100,300)]:

                    for tv in [0.08, 0.10, 0.12]:

                        bt, metrics, trades = run_strategy_with_trades(
                            train,
                            cash_df=cash_train,
                            price_col="Zamkniecie",
                            
                            X=X,
                            Y=Y,

                            fast=fast,
                            slow=slow,
                            
                            target_vol=tv,
                            position_mode=selected_mode
                            
                        )

                        # Skip failed runs
                        if metrics is None:
                            continue

                        # Objective
                        score = (
                            metrics["CAGR"]
                            - 0.5 * metrics["Vol"]
                            + 0.2 * metrics["Sharpe"]
                        )

                        if score > best_score:

                            best_score = score

                            best_params = {
                                "X": X,
                                "Y": Y,
                                "fast": fast,
                                "slow": slow,
                                "target_vol": tv
                            }

        if best_params is None:
            break

        # ------------------------
        # TEST ON OOS
        # ------------------------

        bt, test_metrics, trades = run_strategy_with_trades(
            test,
            price_col="Zamkniecie",
            cash_df=cash_test,
            position_mode=selected_mode,
            **best_params
        )
        
        if test_metrics is None:
            start += pd.DateOffset(years=test_years)
            continue
        
        results.append({
            "start": train_start,
            **best_params,
            **test_metrics
        })

        start += pd.DateOffset(years=test_years)

    return pd.DataFrame(results)


def analyze_trades(trades):

    # --- analyze_trades ---
    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())

    pf = np.inf if loss == 0 else (
    trades.loc[trades["Return"] > 0, "Return"].sum() / loss
    )   
    return {
        "Trades": len(trades),
        "WinRate": (trades["Return"] > 0).mean(),
        "AvgWin": trades.loc[trades["Return"] > 0, "Return"].mean(),
        "AvgLoss": trades.loc[trades["Return"] < 0, "Return"].mean(),
        "ProfitFactor": pf,
        "AvgDays": trades["Days"].mean()
    }


# ------------------------
# Report Printing Function with Best Parameters
# ------------------------
def print_backtest_report(metrics, trades, trade_stats, best_params=None, position_mode=None):
    logging.info("="*80)
    logging.info(f"FULL SAMPLE BACKTEST REPORT   mode = {position_mode}")
    logging.info("="*80)

    # Best Parameters
    if best_params:
        logging.info("BEST PARAMETERS USED:")
        logging.info(
            "X: %.3f | Y: %.3f | Fast MA: %d | Slow MA: %d | Target Vol: %.2f",
            best_params['X'],
            best_params['Y'],
            best_params['fast'],
            best_params['slow'],
            best_params['target_vol']
        )
        logging.info("-"*80)

    # Metrics
    logging.info("METRICS:")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%%",
        metrics["CAGR"]*100,
        metrics["Vol"]*100,
        metrics["Sharpe"],
        metrics["MaxDD"]*100
    )
    logging.info("-"*80)

    # Trade Statistics
    if trade_stats:
        logging.info("TRADE STATISTICS:")
        logging.info(
            "Total Trades: %d | Win Rate: %.1f%% | Avg Win: %.2f%% | Avg Loss: %.2f%% | Profit Factor: %.2f | Avg Days: %.1f",
            trade_stats['Trades'],
            trade_stats['WinRate']*100,
            trade_stats['AvgWin']*100,
            trade_stats['AvgLoss']*100,
            trade_stats['ProfitFactor'],
            trade_stats['AvgDays']
        )
        logging.info("-"*80)
    else:
        logging.info("No trades executed in the backtest.")
        logging.info("-"*80)

    # Trade Log
    if not trades.empty:
        trades_fmt = trades.copy()
        trades_fmt['Return'] = (trades_fmt['Return']*100).round(2).astype(str) + '%'
        trades_fmt['EntryPrice'] = trades_fmt['EntryPrice'].round(2)
        trades_fmt['ExitPrice'] = trades_fmt['ExitPrice'].round(2)
        logging.info("TRADE LOG:")
        logging.info("\n%s", trades_fmt.to_string(index=False))
    logging.info("="*80)



### Initials


# Setup logging
LOG_FILE = "StratRun.log"
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
#csv_filename = os.path.join(tmp_dir, "data.csv")
csv_filename_w20tr = os.path.join(tmp_dir, "w20tr.csv")
#csv_filename_m40tr = os.path.join(tmp_dir, "m40tr.csv")
#csv_filename_s80tr = os.path.join(tmp_dir, "s80tr.csv")
#csv_filename_wbbwz = os.path.join(tmp_dir, "wbbwz.csv")


csv_filename_cash = os.path.join(tmp_dir, "GS_konserw2720.csv")

# csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"

# Base URLs
#csv_base_url = os.getenv('CSV_BASE_URL')


# List of User-Agent headers for different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
]

# Folder ID for Google Drive upload 

#def get_folder_id(name):
#    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
#    res = drive_service.files().list(q=query, fields='files(id)').execute()
#    items = res.get('files', [])
#    return items[0]['id'] if items else None



# List to store all results - we'll see if necessary out for now
# all_results = []



# Main function to process the data


logging.info("RUN START: %s", dt.datetime.now())


# Load data
download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr, 'w20t')
INDEX_W20 = load_csv(csv_filename_w20tr)

df = INDEX_W20

# Download money market / bond fund
download_csv(
    "https://stooq.pl/q/d/l/?s=2720.n&i=d",   
    csv_filename_cash,
    "cash"
)

CASH = load_csv(csv_filename_cash)

#cash_ret = prepare_cash_returns(CASH)

# Set POSITION MODE

chosen_mode="full"  
# options: "vol_entry", "vol_dynamic", "full"


# ============================
# REPORTING
# ============================


# -------- Single Run --------

logging.info("=" * 80)
logging.info("SINGLE RUN RESULTS")
logging.info("=" * 80)

bt, metrics, trades = run_strategy_with_trades(
    df,
    price_col="Zamkniecie",
    cash_df=CASH,
    position_mode=chosen_mode
)



logging.info("Metrics:\n%s", pformat(metrics))

if not trades.empty:
    logging.info("Trades:\n%s", trades.to_string(index=False))
else:
    logging.info("No trades in single run.")


# -------- Walk Forward --------

logging.info("=" * 80)
logging.info("WALK-FORWARD OPTIMIZATION")
logging.info("=" * 80)

wf = walk_forward(df, cash_df=CASH, selected_mode=chosen_mode)

if wf.empty:
    logging.warning("Walk-forward produced no results.")
else:

    logging.info("Walk-forward Results:\n%s", wf.to_string(index=False))

    avg = wf.mean(numeric_only=True)

    logging.info("Walk-forward Averages:\n%s", avg.to_string())


# -------- Best Parameters --------

if not wf.empty:

    best_row = wf.sort_values("Sharpe", ascending=False).iloc[0]

    best_params = {
        "X": best_row["X"],
        "Y": best_row["Y"],
        "fast": int(best_row["fast"]),
        "slow": int(best_row["slow"]),
        "target_vol": best_row["target_vol"]
    }

    logging.info("=" * 80)
    logging.info("BEST PARAMETER SET (BY SHARPE)")
    logging.info("=" * 80)

    logging.info("Best Parameters:\n%s", pformat(best_params))


# -------- Full-Sample Backtest --------

bt_full, metrics_full, trades_full = run_strategy_with_trades(
    df,
    price_col="Zamkniecie",
    position_mode=chosen_mode,
    cash_df=CASH,
    **best_params
)

# Analyze trades
trade_stats = analyze_trades(trades_full) if not trades_full.empty else None

# ------------------------
# Call Report Function
# ------------------------
print_backtest_report(metrics_full, trades_full, trade_stats, best_params, position_mode=chosen_mode)


# PLOTTING BLOCK
# Use full-sample data
bt = bt_full
trades = trades_full

# Ensure datetime
bt.index = pd.to_datetime(bt.index)
trades['EntryDate'] = pd.to_datetime(trades['EntryDate'])
trades['ExitDate'] = pd.to_datetime(trades['ExitDate'])

plt.figure(figsize=(16,8))

# Plot equity curve
plt.plot(bt.index, bt['equity'], label='Equity Curve', color='blue', linewidth=2)

# Highlight trades and annotate returns
for _, trade in trades.iterrows():
    color = 'green' if trade['Return'] > 0 else 'red'
    
    # Shade trade period
    plt.axvspan(trade['EntryDate'], trade['ExitDate'], color=color, alpha=0.2)
    
    # Annotate return just above/below trade equity range
    trade_equity = bt.loc[trade['EntryDate']:trade['ExitDate'], 'equity']
    y_pos = trade_equity.max()*1.02 if trade['Return']>0 else trade_equity.min()*0.98
    plt.text(trade['ExitDate'], y_pos, f"{trade['Return']*100:.1f}%", 
             color=color, fontsize=9, ha='left', va='bottom', rotation=45)

plt.title("Equity Curve with Trades Highlighted and Returns Annotated", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Equity", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


logging.info("RUN END")