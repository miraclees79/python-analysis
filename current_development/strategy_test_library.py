import pandas as pd
import numpy as np
import requests
import time
import random
import os
import datetime as dt
from joblib import Parallel, delayed
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse


import logging



# ============================================================
# DATA ACQUISITION
# ============================================================



import subprocess


# Only import headless browser when needed
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except ImportError:
    webdriver = None

def download_csv_v2(url: str, filename: str) -> bool:
    """
    Download a CSV from Stooq using 3-step fallback:
      1. requests with browser headers
      2. curl subprocess
      3. headless Chrome (Selenium)
    Returns True if download succeeds, False otherwise.
    """
    # ===== Step 1: Hardened Python requests =====
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15"
    ]

    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://stooq.pl/",
        "Connection": "keep-alive"
    }

    for attempt in range(1, 4):
        try:
            logging.info(f"Step 1: Attempt {attempt} - requests GET {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            if not resp.content.strip() or b"<html" in resp.content.lower():
                logging.warning(f"Step 1: Attempt {attempt} returned empty or HTML")
                time.sleep(0.4)
                continue

            with open(filename, "wb") as f:
                f.write(resp.content)
            logging.info(f"Step 1: Downloaded via requests: {filename}")
            return True

        except Exception as e:
            logging.warning(f"Step 1: Attempt {attempt} failed: {e}")
            time.sleep(0.4)

    # ===== Step 2: Curl fallback =====
    logging.info(f"Step 2: Falling back to curl")
    cmd = [
        "curl", "-L", "--fail",
        "--retry", "3", "--retry-delay", "1",
        "-A", "Mozilla/5.0",
        "-H", "Referer: https://stooq.pl/",
        "--http1.1",  # force HTTP/1.1
        "-o", filename,
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0 and os.path.exists(filename) and os.path.getsize(filename) > 0:
            logging.info(f"Step 2: Downloaded via curl: {filename}")
            return True
        logging.warning(f"Step 2: Curl failed or empty file. stderr: {result.stderr.strip()}")
    except Exception as e:
        logging.warning(f"Step 2: Curl execution failed: {e}")

    # ===== Step 3: Headless Chrome / Selenium fallback =====
    if webdriver is None:
        logging.error("Step 3: Selenium not installed; cannot perform headless browser fallback")
        return False

    logging.info(f"Step 3: Falling back to headless Chrome")
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        content = driver.page_source.encode("utf-8")
        driver.quit()

        # Detect if HTML page instead of CSV
        if b"<html" in content.lower():
            logging.error("Step 3: Headless Chrome returned HTML instead of CSV")
            return False

        with open(filename, "wb") as f:
            f.write(content)
        logging.info(f"Step 3: Downloaded via headless Chrome: {filename}")
        return True

    except Exception as e:
        logging.error(f"Step 3: Headless Chrome failed: {e}")

    logging.error(f"All 3 download steps failed for {url}")
    return False



def _add_noise(url: str) -> str:
    """Add a random query param to bypass caching and anti-bot filters."""
    noise = f"t={int(time.time()*1000)}{random.randint(100,999)}"
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{noise}"


def _switch_to_i10(url: str) -> str:
    """If i=d is used, switch to i=10. Preserves other parameters."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if "i" in qs and qs["i"] == ["d"]:
        qs["i"] = ["10"]

    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def download_csv_hardened(url, filename):
    """
    Extremely robust CSV downloader for Stooq endpoints.
    Uses:
      - rotating browser headers
      - random query noise
      - retries
      - HTML detection
      - fallback from i=d → i=10
      - requests.Session() for persistent TLS
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15"
    ]

    session = requests.Session()

    # First try original URL, then fallback to i=10 version
    candidate_urls = [url, _switch_to_i10(url)]
    tried_fallback = False

    for base_url in candidate_urls:
        for attempt in range(1, 4):

            # Fully random headers *each attempt*
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://stooq.pl/",
                "Connection": "keep-alive"
            }

            noisy_url = _add_noise(base_url)

            logging.info(
                f"Attempt {attempt}{' (fallback i=10)' if tried_fallback else ''}: "
                f"Downloading {noisy_url} with User-Agent: {headers['User-Agent']}"
            )

            try:
                response = session.get(
                    noisy_url,
                    headers=headers,
                    timeout=10,
                    allow_redirects=True,
                    stream=True
                )
                response.raise_for_status()

                content = response.content

                # Empty -> retry
                if not content.strip():
                    logging.warning(f"Attempt {attempt}: Empty response from {noisy_url}")
                    time.sleep(0.4)
                    continue

                # Stooq sometimes responds with HTML
                if b"<html" in content.lower():
                    snippet = content[:200].decode("utf-8", errors="ignore")
                    logging.warning(f"Attempt {attempt}: HTML received:\n{snippet}")
                    time.sleep(0.4)
                    continue

                # Save file
                with open(filename, "wb") as f:
                    f.write(content)

                logging.info(f"Downloaded and saved: {filename}")
                return True

            except requests.exceptions.Timeout:
                logging.error(f"Attempt {attempt}: Timeout for {noisy_url}")
                time.sleep(0.4)

            except requests.exceptions.HTTPError as e:
                logging.error(f"Attempt {attempt}: HTTP error: {e}")
                return False

            except Exception as e:
                logging.error(f"Attempt {attempt}: Unexpected error: {e}")
                time.sleep(0.4)

        # If primary URL failed, try i=10 version
        tried_fallback = True

    logging.error(f"Failed to download {url} after all retries (including i=10 fallback).")
    return False


def download_csv_old(url, filename):


    """
    Download a CSV file from a URL and save it to disk.

    Rotates User-Agent headers randomly to reduce the likelihood of
    being blocked by the server. Returns True on success, False on
    any failure (timeout, HTTP error, empty response).

    Parameters
    ----------
    url      : str  — full URL to fetch
    filename : str  — local path where the file will be saved
    """
    
    # List of User-Agent headers for different browsers
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
    ]

    
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
    
    """
    Load and validate a price series CSV downloaded from stooq.pl.

    Performs the following steps in order:
      1. Read CSV with UTF-8 encoding, skipping malformed lines.
      2. Strip whitespace from column names and locate the 'Data'
         date column, including fuzzy matching for hidden characters.
      3. Parse dates, drop invalid rows, sort ascending, set date index.
      4. Staleness check: discard the entire series if the most recent
         observation is older than 10 calendar days — this prevents
         running a strategy on stale data without warning.
      5. Continuity check: if a gap longer than 30 calendar days is
         found in the date sequence, discard all data before the most
         recent such gap. This handles fund/index series that have been
         relaunched or restructured mid-history.

    Returns a DataFrame indexed by date, or None if any validation
    step fails.

    Parameters
    ----------
    filename : str — path to the CSV file on disk
    """
    
    
    try:
        df = pd.read_csv(filename, on_bad_lines='skip', delimiter=',', decimal='.', encoding='utf-8')
    except Exception as e:
        logging.error(f" Error reading CSV file: {e}")
        return None

    if df.empty or df.columns.size == 0:
        logging.error(" CSV file is empty or corrupted.")
        return None

    # Strip whitespace and inspect column names
    df.columns = df.columns.str.strip()
    logging.debug("Available columns after stripping:", df.columns)

    date_column = 'Data'  # Expected date column

    # Double-check the column names for hidden characters
    if date_column not in df.columns:
        exact_matches = [col for col in df.columns if col.strip() == date_column]
        if exact_matches:
            date_column = exact_matches[0]  # If a match is found, update the column name
            logging.info(f" Using corrected column name: '{date_column}'")
        else:
            # If still not found, display and exit
            logging.error(f" Column '{date_column}' not found after processing. Available columns: {df.columns}")
            return None

    # Check if the date column contains valid data
    if df[date_column].isnull().all():
        logging.error(f" Column '{date_column}' contains only NaN values.")
        return None

    # Convert to datetime, handling errors and dropping invalid dates
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df.dropna(subset=[date_column], inplace=True)

    # Check if there are still valid dates left
    if df.empty:
        logging.error(" No valid dates after conversion. Data is discarded.")
        return None

    # Sort by date and set as index
    df = df.sort_values(by=date_column).set_index(date_column)

    # Check 1: Discard the data if the newest observation is older than 10 days
    newest_date = df.index.max()
    if (dt.datetime.now() - newest_date).days > 10:
        logging.warning(f" The newest observation ({newest_date}) is older than 10 days. Data is discarded.")
        return None

    # Check 2: Discard data before the most recent break longer than 30 days
    date_diffs = df.index.to_series().diff().dt.days  # Calculate gaps in the date series
    breaks = date_diffs[date_diffs > 30].index  # Identify breaks longer than 30 days

    if not breaks.empty:
        # Keep only the data from the newest observation to the most recent break
        last_valid_date = breaks[-1]
        df = df.loc[df.index > last_valid_date]  # Slice the DataFrame from the break to the end
        logging.info(f" Data contains a break longer than 30 days. Keeping data from {last_valid_date} onward.")
    
    logging.info("SUCCESS! CSV file loaded successfully and processed.")


    
    return df

def download_fund_navs(fund_codes, tmp_dir):
    """
    Download fund NAV series from stooq.pl and build the FUND_FILES dict.
    """

    BASE_URL = "https://stooq.pl/q/d/l/?s={code}.n&i=d"

    fund_files = {}

    for code, name in fund_codes.items():
        url      = BASE_URL.format(code=code)
        filepath = os.path.join(tmp_dir, f"fund_{code}_{name}.csv")

        success = download_csv(url, filepath)

        if success:
            fund_files[name] = filepath
            logging.info("Fund %s (%s) — downloaded to %s.", name, code, filepath)
        else:
            logging.warning(
                "Fund %s (%s) — download failed, will be excluded from panel.",
                name, code
            )
    delay = random.uniform(0.3, 1)
    time.sleep(delay)
    logging.info(
        "download_fund_navs: %d of %d funds downloaded successfully.",
        len(fund_files), len(fund_codes)
    )

    return fund_files

    
def build_funds_df(fund_files, price_col="Zamkniecie", min_history_years=10):
    """
    Build a combined fund NAV panel from a list of CSV files.
    """

    MAX_GAP_DAYS = 30

    series_list = []
    excluded    = []

    for fund_id, filepath in fund_files.items():

        df = load_csv(filepath)

        if df is None:
            logging.warning("Fund %s — load_csv returned None, skipping.", fund_id)
            excluded.append((fund_id, "load failed"))
            continue

        if price_col not in df.columns:
            logging.warning(
                "Fund %s — column '%s' not found. Available: %s. Skipping.",
                fund_id, price_col, list(df.columns)
            )
            excluded.append((fund_id, f"missing column {price_col}"))
            continue

        series = df[price_col].copy()
        series.name = fund_id

        years = (series.index.max() - series.index.min()).days / 365.25
        if years < min_history_years:
            logging.warning(
                "Fund %s — only %.1f years of history (min %.0f). Skipping.",
                fund_id, years, min_history_years
            )
            excluded.append((fund_id, f"insufficient history ({years:.1f}y)"))
            continue

        series_list.append(series)
        logging.info(
            "Fund %s — loaded %d rows from %s to %s.",
            fund_id,
            len(series),
            series.index.min().date(),
            series.index.max().date()
        )

    if len(series_list) < 2:
        logging.error(
            "build_funds_df: fewer than 2 funds loaded successfully. "
            "Fund breadth filter requires at least 2 funds."
        )
        return pd.DataFrame()

    funds_df = pd.concat(series_list, axis=1, join="outer", sort=True)
    funds_df.sort_index(inplace=True)
    funds_df = funds_df.ffill()

    for fund_id in funds_df.columns:
        col = funds_df[fund_id]
        first_valid = col.first_valid_index()
        if first_valid is None:
            logging.warning(
                "Fund %s — entirely NaN after merge, dropping.", fund_id
            )
            funds_df.drop(columns=[fund_id], inplace=True)
            excluded.append((fund_id, "entirely NaN"))
            continue

        interior = col.loc[first_valid:]
        max_gap = interior.isna().astype(int).groupby(
            interior.notna().astype(int).cumsum()
        ).sum().max()

        if pd.notna(max_gap) and max_gap > MAX_GAP_DAYS:
            logging.warning(
                "Fund %s — gap of %d consecutive missing days after "
                "forward-fill. Dropping.",
                fund_id, int(max_gap)
            )
            funds_df.drop(columns=[fund_id], inplace=True)
            excluded.append((fund_id, f"gap of {int(max_gap)} days"))

    min_funds_required = max(2, len(funds_df.columns) // 2)
    funds_df = funds_df.dropna(thresh=min_funds_required)

    if funds_df.empty:
        logging.error("build_funds_df: panel is empty after cleaning.")
        return pd.DataFrame()

    logging.info(
        "build_funds_df: panel ready — %d funds, %d rows, %s to %s.",
        len(funds_df.columns),
        len(funds_df),
        funds_df.index.min().date(),
        funds_df.index.max().date()
    )

    if excluded:
        logging.info(
            "build_funds_df: excluded funds — %s",
            ", ".join(f"{fid} ({reason})" for fid, reason in excluded)
        )

    return funds_df


def prepare_cash_returns(cash_df, price_col="Zamkniecie"):
    cash = cash_df.copy()
    cash["cash_price"] = cash[price_col]
    cash["cash_ret"] = cash["cash_price"].pct_change()
    cash = cash[["cash_ret"]].dropna()
    return cash


# ============================
# Indicators
# ============================

def compute_momentum(
    series,
    lookback=252,
    skip=21,
    blend=False,
    blend_lookbacks=(21, 63, 126, 252),
    blend_skip=5,
):
    """
    Compute a momentum signal from a price series.

    Single-horizon mode (blend=False, default — preserves existing behaviour):
        momentum = series.shift(skip) / series.shift(lookback) - 1
        Returns the return from lookback days ago to skip days ago.
        lookback and skip parameters are used exactly as before.

    Blended multi-horizon mode (blend=True):
        Computes momentum over each horizon in blend_lookbacks, applies a
        uniform blend_skip to all horizons as a microstructure buffer, then
        returns the equally-weighted average signal.

        A short skip (default 5 days) is used for all horizons rather than
        the standard 21-day skip. Using skip=21 on a 21-day lookback would
        skip the entire signal window; 5 days avoids microstructure noise
        while preserving meaningful short-horizon signal.

        The blend_lookbacks tuple should cover the range of horizons you
        want to combine. Default (21, 63, 126, 252) covers 1m, 3m, 6m, 12m.
        The composite is the unweighted mean — each horizon contributes
        equally regardless of its own vol, keeping interpretation simple
        and avoiding an extra estimation step.

        When blend=True, the lookback and skip parameters are ignored;
        blend_lookbacks and blend_skip govern the calculation instead.

    Parameters
    ----------
    series          : pd.Series  — price series (DatetimeIndex)
    lookback        : int        — single-horizon lookback (days); ignored when blend=True
    skip            : int        — single-horizon skip (days); ignored when blend=True
    blend           : bool       — False = single horizon (default), True = multi-horizon blend
    blend_lookbacks : tuple[int] — lookback horizons for blend mode
    blend_skip      : int        — microstructure skip applied to all blend horizons

    Returns
    -------
    pd.Series — momentum signal, same index as series
    """
    if not blend:
        # Original single-horizon behaviour — unchanged
        return series.shift(skip) / series.shift(lookback) - 1

    # Multi-horizon blend: compute one signal per horizon, average them
    signals = []
    for lb in blend_lookbacks:
        sig = series.shift(blend_skip) / series.shift(lb) - 1
        signals.append(sig)

    # Equal-weight average across horizons
    import pandas as pd
    blended = pd.concat(signals, axis=1).mean(axis=1)
    blended.name = series.name
    return blended


# ============================
# FUND BREADTH SIGNAL
# ============================

def compute_fund_breadth_signal(
    funds_df,
    lookback_days=30,
    n_top=2,
    entry_roll_thresh=0.03,
    entry_since_thresh=0.05,
    exit_roll_thresh=-0.03,
    exit_since_thresh=-0.05,
):
    """
    Compute a binary IN/OUT signal from a panel of fund NAV series.
    """

    fund_rets = funds_df.pct_change()

    roll_ret = (1 + fund_rets).rolling(lookback_days).apply(
        np.prod, raw=True
    ) - 1

    signal   = pd.Series(0, index=funds_df.index)
    state    = 0
    last_change_idx = funds_df.index[0]

    for i, date in enumerate(funds_df.index):

        if i < lookback_days:
            signal.iloc[i] = state
            continue

        todays_roll = roll_ret.loc[date].dropna()

        if todays_roll.empty:
            signal.iloc[i] = state
            continue

        ref_prices = funds_df.loc[:last_change_idx].iloc[-1]
        curr_prices = funds_df.loc[date]

        since_rets = (curr_prices / ref_prices - 1).dropna()

        common_funds = todays_roll.index.intersection(since_rets.index)

        if len(common_funds) < n_top:
            signal.iloc[i] = state
            continue

        todays_roll  = todays_roll.loc[common_funds]
        since_rets   = since_rets.loc[common_funds]

        top_funds    = todays_roll.nlargest(n_top)
        bottom_funds = todays_roll.nsmallest(n_top)

        top_since    = since_rets.loc[top_funds.index]
        bottom_since = since_rets.loc[bottom_funds.index]

        if state == 0:
            roll_condition  = top_funds.mean()  >= entry_roll_thresh
            since_condition = top_since.mean()  >= entry_since_thresh

            if roll_condition or since_condition:
                state = 1
                last_change_idx = date

        elif state == 1:
            roll_condition  = bottom_funds.mean()  <= exit_roll_thresh
            since_condition = bottom_since.mean()  <= exit_since_thresh

            if roll_condition or since_condition:
                state = 0
                last_change_idx = date

        signal.iloc[i] = state

    return signal.shift(1).fillna(0)
    
# ============================
# Performance Metrics
# ============================

def compute_metrics(equity, 
                    risk_free_rate=0, 
                    freq=252):
    ret = equity.pct_change().dropna()

    years = len(ret) / freq
    cagr = (equity.iloc[-1]/equity.iloc[0])**(1/years)-1
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0 else 0.0
    
    daily_rf   = (1 + risk_free_rate) ** (1 / 252) - 1
    daily_rets = equity.pct_change().dropna()
    downside   = daily_rets[daily_rets < daily_rf] - daily_rf
    if len(downside) > 0:
        downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    else:
        downside_vol = 0.0
    sortino = excess_return / downside_vol if downside_vol > 0 else 0.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino":     sortino,
        "MaxDD": max_dd,
        "CalMAR": calmar
    }

# ============================================================
# PARAMETER STABILITY
# ============================================================


def neighbour_mean(key, scores, stop_grid, Y_grid):
    """
    Compute the mean score of neighbouring parameter sets for a given key.

    Neighbours vary by one step in the stop dimension (X or N_atr depending
    on stop mode) or the Y dimension, holding all other parameters fixed.

    Parameters
    ----------
    key       : tuple        — (filter_mode, fund_idx, stop_param, Y, fast,
                               slow, tv, stop_loss, mom_lookback)
    scores    : dict         — maps parameter tuples to objective scores
    stop_grid : list[float]  — ordered list of values for the stop parameter
                               in position 2 of the key (X_grid in fixed mode,
                               N_atr_grid in ATR mode)
    Y_grid    : list[float]  — ordered list of Y values used in the search

    Returns
    -------
    float — mean score of all neighbours including the key itself, or the
            key's own score if no neighbours exist in scores.
    """

    filter_mode, fund_idx, stop_param, Y, fast, slow, tv, sl, mom_lookback = key

    # Find index of current stop_param and Y in their grids
    si = min(range(len(stop_grid)), key=lambda i: abs(stop_grid[i] - stop_param))
    yi = min(range(len(Y_grid)),    key=lambda i: abs(Y_grid[i] - Y))

    neighbours = []
    for ds in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nsi, nyi = si + ds, yi + dy
            if 0 <= nsi < len(stop_grid) and 0 <= nyi < len(Y_grid):
                nkey = (filter_mode, fund_idx,
                        stop_grid[nsi], Y_grid[nyi],
                        fast, slow, tv, sl, mom_lookback)
                if nkey in scores:
                    neighbours.append(scores[nkey])

    return np.mean(neighbours) if neighbours else scores[key]

def calc_position(vol, position_mode, target_vol, max_leverage):
    if position_mode == "full":
        return 1.0
    if pd.notna(vol) and vol > 0:
        pos = target_vol / vol
    else:
        pos = 1.0
    return min(pos, max_leverage)


# ============================
# Buy & Hold for comparison
# ============================

def compute_buy_and_hold(df, price_col="Zamkniecie", start=None, end=None):
    bh = df[price_col].copy()
    
    if start is not None:
        bh = bh.loc[bh.index >= start]
    if end is not None:
        bh = bh.loc[bh.index <= end]
    
    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(bh_equity)
    
    return bh_equity, {k: float(v) for k, v in bh_metrics.items()}


# ============================
# Strategy Engine
# ============================
def run_strategy_with_trades(
    df,
    price_col="price",
    X=0.1,
    Y=0.1,
    stop_loss=0.1,
    fast=50,
    slow=200,
    vol_window=20,
    target_vol=0.10,
    max_leverage=1.0,
    position_mode="vol_entry",
    filter_mode="ma",
    mom_lookback=252,
    cash_df=None,
    safe_rate=0.0,
    initial_state=None,
    warmup_df=None,
    fund_signal=None,
    entry_gate=None,
    # -------------------------------------------------------
    # ATR trailing stop parameters
    # -------------------------------------------------------
    use_atr_stop=False,   # False = fixed % (X), True = ATR-scaled (N_atr)
    N_atr=3.0,            # ATR multiplier for trailing stop (used when use_atr_stop=True)
    atr_window=20,        # Rolling window for close-only ATR estimate (days)
    # -------------------------------------------------------
    fast_mode=True,
):
    """
    Run the trend-following strategy on a single price series and return
    the equity curve, performance metrics, and trade log.

    TRAILING STOP MODES
    -------------------
    use_atr_stop=False (default, backward-compatible):
        stop_level = (1 - X) * M
        X is the fixed fraction below the running peak. Identical to the
        original implementation. X is searched over X_grid in walk_forward.

    use_atr_stop=True (normalised ATR Chandelier exit):
        stop_level = M * (1 - N_atr * ATR_pct)
        ATR_pct is the rolling mean of |close_t - close_{t-1}| / close_{t-1}
        over atr_window bars — a dimensionless daily-return ATR. This makes
        N_atr directly comparable to X: N_atr=0.10 means trail by 10% of M
        when the average daily move equals 10% of price, with the stop
        widening in high-vol regimes and tightening in low-vol regimes.
        N_atr is searched over N_atr_grid in walk_forward; values are the
        same order of magnitude as X_grid (e.g. 0.08 to 0.20 for equity).

    The absolute stop (stop_loss, ABSOLUTE_STOP exit) is always a fixed
    fraction below entry price regardless of ATR mode — it is a backstop
    for gap-down events and is not made volatility-adaptive.

    All other parameters and the carry-state mechanism are unchanged.
    """

    df = df.copy()
    df["price"] = df[price_col]
    
    # Detect if high/low columns exist (and not entirely NaN)
    has_hl = ("Najwyzszy" in df.columns 
          and "Najnizszy" in df.columns
          and not df["Najwyzszy"].isna().all()
          and not df["Najnizszy"].isna().all())
    if has_hl:
        df["high"] = df["Najwyzszy"]
        df["low"]  = df["Najnizszy"]    
        
    if warmup_df is not None:
        warmup = warmup_df.copy()
        warmup["price"] = warmup[price_col]
        warmup_has_hl = ("Najwyzszy" in warmup.columns 
              and "Najnizszy" in warmup.columns
              and not warmup["Najwyzszy"].isna().all()
              and not warmup["Najnizszy"].isna().all())
        if warmup_has_hl:
            warmup["high"] = warmup["Najwyzszy"]
            warmup["low"]  = warmup["Najnizszy"]
            
        warmup["_warmup"] = True
        df["_warmup"] = False
        df = pd.concat([warmup, df])
    else:
        df["_warmup"] = False
    
    if entry_gate is not None:
        gate_aligned = entry_gate.reindex(df.index, method="ffill").fillna(1).astype(int)
    else:
        gate_aligned = None
    
    test_start = df[~df["_warmup"]].index[0]

    if cash_df is not None:
        cash = prepare_cash_returns(cash_df)
        df = df.merge(cash, left_index=True, right_index=True, how="left")
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()
    else:
        df["cash_ret"] = safe_rate / 252

    if df["cash_ret"].isna().all():
        logging.info("Cash series missing — falling back to flat safe_rate")
        df["cash_ret"] = safe_rate / 252

    oos_cash = df.loc[~df["_warmup"], "cash_ret"]
    if len(oos_cash) > 0 and oos_cash.notna().any():
        cumulative = (1 + oos_cash).prod()
        n_years    = max(len(oos_cash) / 252, 0.01)
        rf_rate    = cumulative ** (1 / n_years) - 1
    else:
        rf_rate = safe_rate

    if fund_signal is not None:
        df = df.merge(
            fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left"
        )
        df["fund_filter"] = df["fund_filter"].ffill().fillna(0)
    else:
        df["fund_filter"] = 1
        
    df["ret"] = df["price"].pct_change()
    vol = df["ret"].rolling(vol_window).std() * np.sqrt(252)
    df["vol"] = vol.shift(1)
    df["ma_fast"] = df["price"].rolling(fast).mean().shift(1)
    df["ma_slow"] = df["price"].rolling(slow).mean().shift(1)
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    if filter_mode == "mom":
        df["MOM"] = compute_momentum(df["price"], 
                        lookback=mom_lookback,
                        blend = False).shift(1)
    elif filter_mode == "mom_blend":
        df["MOM"] = compute_momentum(df["price"], 
                        blend = True).shift(1)
    else:
        df["MOM"] = 1

    # -------------------------------------------------------
    # ATR series — rolling mean of |daily return| (close-only,
    # normalised to price, i.e. |ΔP / P_prev| * 100 as fallback, if
    # high and low are available - use high-low ATR: 
    #    [max(high, close(-1)) - min(low, close(-1))]/close(-1) * 100
    # 
    # Expressed as a dimensionless fraction in pct pts so that
    #   stop_level = M * (1 - N_atr * atr_val)
    # is directly comparable to the fixed-% stop
    #   stop_level = M * (1 - X)
    # and N_atr has the same units as X (a fraction of price).
    #
    # This means N_ATR_GRID values are directly comparable to
    # X_GRID values:  N_atr=0.10 trails by 10% of M when
    # the average daily move equals 1% — and more in high-vol
    # regimes, less in low-vol regimes (the adaptive benefit).
    #
    # Shifted by 1 so today's stop uses yesterday's ATR estimate
    # (no look-ahead). Computed regardless of use_atr_stop so it
    # is always available in the pre-extracted numpy arrays.
    # -------------------------------------------------------
    
    if has_hl:
        prev_close = df["price"].shift(1)

        tr = (
            np.maximum(df["high"], prev_close)
            - np.minimum(df["low"], prev_close)
            )

        df["relative_tr"] = tr / prev_close

        df["atr"] = (
            df["relative_tr"]
            .rolling(atr_window)
            .mean()
            .shift(1)            # avoid lookahead
            * 100
            )
        

    else:
        # fallback: absolute daily % move * 100 to use 0.08 etc grid
        df["atr"] = (
            (df["price"].diff().abs() / df["price"].shift(1))
            .rolling(atr_window)
            .mean()
            .shift(1)
            * 100
            )
        
    df.dropna(inplace=True)

    # -----------------------
    # Initialise state
    # -----------------------

    equity = 1.0
    equity_curve = []
    trades = []

    if initial_state is not None:
        position    = initial_state["position"]
        entry_price = initial_state["entry_price"]
        entry_date  = initial_state["entry_date"]
        entry_reason= initial_state["entry_reason"]
        entry_pos   = initial_state["entry_pos"]
        M           = initial_state["M"]
        m           = initial_state["m"]
        entry_carried = True
        rebal_count      = initial_state.get("rebal_count", 0)
        rebal_cost_total = initial_state.get("rebal_cost_total", 0.0)
    else:
        position    = 0.0
        entry_price = None
        entry_date  = None
        entry_reason= None
        entry_pos   = None
        M           = None
        m           = None
        entry_carried = False 
        rebal_count      = 0
        rebal_cost_total = 0.0    
    
    if fund_signal is not None:
        filter_mode_active = "fund"
    elif filter_mode == "mom":
        filter_mode_active = "mom"
    elif filter_mode == "mom_blend":
        filter_mode_active = "mom_blend"
    else:
        filter_mode_active = "ma"

    if fast_mode:
        _prices    = df["price"].to_numpy()
        _rets      = df["ret"].to_numpy()
        _cash_rets = df["cash_ret"].to_numpy()
        _trends    = df["trend"].to_numpy()
        _moms      = df["MOM"].to_numpy()
        _vols      = df["vol"].to_numpy()
        _atrs      = df["atr"].to_numpy()   # ATR array — always extracted
        _warmups   = df["_warmup"].to_numpy(dtype=bool)
        _gate_vals = (
            gate_aligned.reindex(df.index).fillna(1).to_numpy().astype(int)
            if gate_aligned is not None else None
        )
        _fund_vals = (
            df["fund_filter"].to_numpy()
            if "fund_filter" in df.columns else None
        )
        _index = df.index

    if fast_mode:
        # ------------------------------------------------------------------
        # Fast path: integer loop over pre-extracted numpy arrays.
        # ------------------------------------------------------------------
        for _n in range(len(_prices)):

            i        = _index[_n]
            price    = float(_prices[_n])
            ret      = float(_rets[_n])
            cash_ret = float(_cash_rets[_n])
            trend    = int(_trends[_n])
            mom      = float(_moms[_n])
            vol      = float(_vols[_n])
            atr_val  = float(_atrs[_n])    # ATR as dimensionless fraction (daily-return ATR)

            if filter_mode_active == "fund":
                filter_on = bool(_fund_vals[_n]) if _fund_vals is not None else True
            elif filter_mode_active == "mom" or filter_mode_active == "mom_blend":
                filter_on = mom > 0
            else:
                filter_on = (trend == 1)

            is_warmup_row = bool(_warmups[_n])
            if is_warmup_row:
                equity_curve.append(equity)
                continue

            if position > 0:
                equity *= (1 + position * ret + (1 - position) * cash_ret)
            else:
                equity *= (1 + cash_ret)

            exit_reasons = []

            if position > 0:
                dd = (price - entry_price) / entry_price
                if dd < -stop_loss:
                    exit_reasons.append("ABSOLUTE_STOP")

            if position > 0 and position_mode == "vol_dynamic":
                new_pos = calc_position(vol, position_mode, target_vol, max_leverage)
                size_change = abs(new_pos - position)
                if size_change > 0.1:
                    REBAL_COST = 0.0005
                    equity *= (1 - size_change * REBAL_COST)
                    position = new_pos
                    rebal_count      += 1
                    rebal_cost_total += size_change * REBAL_COST

            if position > 0:
                M = max(M, price) if M is not None else price
                # -------------------------------------------------------
                # TRAILING STOP — branch on stop mode
                # -------------------------------------------------------
                if use_atr_stop:
                    # Normalised ATR Chandelier exit.
                    # atr_val = rolling mean of |ΔP/P| (dimensionless fraction).
                    # stop_level = M * (1 - N_atr * atr_val)
                    # N_atr is directly comparable to X: at typical daily vol
                    # of 1%, N_atr=0.10 gives a ~10% stop (10 days × 1%).
                    # In high-vol regimes the stop widens; in low-vol it tightens.
                    # Fall back to no stop breach if ATR is NaN (window start).
                    if np.isfinite(atr_val) and atr_val > 0:
                        stop_level = M * (1 - N_atr * atr_val)
                        trail_breached = price < stop_level
                    else:
                        trail_breached = False
                else:
                    # Fixed percentage trailing stop (original behaviour)
                    trail_breached = price < (1 - X) * M

                if trail_breached:
                    if "ABSOLUTE_STOP" not in exit_reasons:
                        exit_reasons.append("TRAIL_STOP")
                elif not filter_on:
                    exit_reasons.append("FILTER_EXIT")

            exit_reason = " + ".join(exit_reasons) if exit_reasons else None

            if position > 0 and exit_reason:
                COST = 0.0020
                trade_ret = price / entry_price - 1 - COST
                days = (i - entry_date).days
                trades.append({
                    "EntryDate":    entry_date,
                    "ExitDate":     i,
                    "EntryPrice":   entry_price,
                    "Position":     entry_pos,
                    "ExitPrice":    price,
                    "Return":       trade_ret,
                    "Days":         days,
                    "Entry Reason": entry_reason,
                    "Exit Reason":  exit_reason,
                    "CrossWindow":  entry_carried
                })
                position    = 0
                entry_price = None
                entry_date  = None
                entry_reason= None
                M           = None
                m           = None
                entry_pos   = None
                entry_carried = False

            if position == 0:
                m = price if m is None else min(m, price)
                gate_allows = (_gate_vals is None or int(_gate_vals[_n]) == 1)
                if (price > (1 + Y) * m) and filter_on and gate_allows:
                    entry_reason  = "BREAKOUT & FILTER"
                    position      = calc_position(vol, position_mode, target_vol, max_leverage)
                    entry_price   = price
                    entry_date    = i
                    entry_pos     = position
                    M             = price
                    entry_carried = False

            equity_curve.append(equity)

    else:
        # ------------------------------------------------------------------
        # Slow path: original iterrows loop.
        # ------------------------------------------------------------------
        for i, row in df.iterrows():

            price    = row["price"]
            ret      = row["ret"]
            cash_ret = row["cash_ret"]
            trend    = row["trend"]
            mom      = row["MOM"]
            vol      = row["vol"]
            atr_val  = row["atr"]    # ATR as dimensionless fraction (daily-return ATR)

            if filter_mode_active == "fund":
                filter_on = bool(row["fund_filter"])
            elif filter_mode_active == "mom" or filter_mode_active == "mom_blend":
                filter_on = mom > 0
            else:
                filter_on = (trend == 1)

            is_warmup_row = row["_warmup"]
            if is_warmup_row:
                equity_curve.append(equity)
                continue

            if position > 0:
                equity *= (1 + position * ret + (1 - position) * cash_ret)
            else:
                equity *= (1 + cash_ret)

            exit_reasons = []

            if position > 0:
                dd = (price - entry_price) / entry_price
                if dd < -stop_loss:
                    exit_reasons.append("ABSOLUTE_STOP")

            if position > 0 and position_mode == "vol_dynamic":
                new_pos = calc_position(vol, position_mode, target_vol, max_leverage)
                size_change = abs(new_pos - position)

                if size_change > 0.1:
                    REBAL_COST = 0.0005
                    equity *= (1 - size_change * REBAL_COST)
                    position = new_pos
                    rebal_count  += 1
                    rebal_cost_total += size_change * REBAL_COST

            if position > 0:
                M = max(M, price) if M is not None else price
                # -------------------------------------------------------
                # TRAILING STOP — branch on stop mode
                # -------------------------------------------------------
                if use_atr_stop:
                    if np.isfinite(atr_val) and atr_val > 0:
                        stop_level = M * (1 - N_atr * atr_val)
                        trail_breached = price < stop_level
                    else:
                        trail_breached = False
                else:
                    trail_breached = price < (1 - X) * M

                if trail_breached:
                    if "ABSOLUTE_STOP" not in exit_reasons:
                        exit_reasons.append("TRAIL_STOP")
                elif not filter_on:
                    exit_reasons.append("FILTER_EXIT")

            exit_reason = " + ".join(exit_reasons) if exit_reasons else None

            if position > 0 and exit_reason:
                COST = 0.0020
                trade_ret = price / entry_price - 1 - COST
                days = (i - entry_date).days
                
                trades.append({
                    "EntryDate":    entry_date,
                    "ExitDate":     i,
                    "EntryPrice":   entry_price,
                    "Position":     entry_pos,
                    "ExitPrice":    price,
                    "Return":       trade_ret,
                    "Days":         days,
                    "Entry Reason": entry_reason,
                    "Exit Reason":  exit_reason,
                    "CrossWindow":  entry_carried
                })

                position    = 0
                entry_price = None
                entry_date  = None
                entry_reason= None
                M           = None
                m           = None
                entry_pos   = None
                entry_carried = False

            if position == 0:
                m = price if m is None else min(m, price)
                gate_allows = (gate_aligned is None or gate_aligned[i] == 1)
                if (price > (1 + Y) * m) and filter_on and gate_allows:
                    entry_reason = "BREAKOUT & FILTER"
                    position     = calc_position(vol, position_mode, target_vol, max_leverage)
                    entry_price  = price
                    entry_date   = i
                    entry_pos    = position
                    M            = price
                    entry_carried   = False

            equity_curve.append(equity)
    
    
    if position_mode == "vol_dynamic" and rebal_count > 0:
        logging.debug(
            "vol_dynamic rebalancing: %d adjustments, "
            "total cost drag %.4f%% (%.1f bps)",
            rebal_count,
            rebal_cost_total * 100,
            rebal_cost_total * 10000
            )

    end_state = None

    if position > 0 and entry_price is not None:

        last_date  = df.index[-1]
        last_price = df["price"].iloc[-1]
        trade_ret  = last_price / entry_price - 1
        days       = (last_date - entry_date).days
        
        if entry_date < test_start:  
            logging.debug(
                "CARRY trade entry date %s predates test window %s — "
                "trade return and equity curve are on different bases",
                entry_date, test_start
                )
        
        trades.append({
            "EntryDate":    entry_date,
            "ExitDate":     last_date,
            "EntryPrice":   entry_price,
            "Position":     entry_pos,
            "ExitPrice":    last_price,
            "Return":       trade_ret,
            "Days":         days,
            "Entry Reason": entry_reason,
            "Exit Reason":  "CARRY",
            "CrossWindow":  entry_carried
        })

        end_state = {
            "position":     position,
            "entry_price":  entry_price,
            "entry_date":   entry_date,
            "entry_reason": entry_reason,
            "entry_pos":    entry_pos,
            "M":            M,
            "m":            m,
            "rebal_count":       rebal_count,
            "rebal_cost_total":  rebal_cost_total
            }

    df["equity"] = equity_curve
    df = df[~df["_warmup"]].copy()
    df.drop(columns=["_warmup"], inplace=True)
    
    if "fund_filter" in df.columns:
        df.drop(columns=["fund_filter"], inplace=True)
        
    if df.isnull().any().any():
        logging.warning("NaN values remain in test rows after dropna — check cash merge")
    
    first_val = df["equity"].iloc[0]
    if initial_state is not None and abs(first_val - 1.0) > 0.001:
        logging.debug(
            "Warmup P&L on carried position: %.2f%% — excluded from OOS equity",
            (first_val - 1.0) * 100
            )
    if first_val != 0:
            df["equity"] = df["equity"] / first_val
    
    metrics = compute_metrics(df["equity"], risk_free_rate=rf_rate)
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(trades)

    return df, metrics, trades_df, end_state

# -------------------------------------------------------
# walk_forward — threads state across windows
# -------------------------------------------------------



def evaluate_params(
    filter_mode, fund_idx, fund_params, X, Y, fast, slow, tv, stop_loss,
    train, cash_train, vol_window, selected_mode, funds_df, train_start, train_end,
    objective="calmar",
    mom_lookback=252,
    entry_gate=None,
    fast_mode=True,
    # ATR stop parameters
    use_atr_stop=False,
    N_atr=3.0,
    atr_window=20,
):
    """
    Evaluate a single parameter combination on the training window.

    When use_atr_stop=True, X is not used by run_strategy_with_trades
    (N_atr is used instead). X is still passed as part of the key tuple
    in position 2, but in ATR mode walk_forward substitutes N_atr values
    from N_atr_grid into that position so the key structure is consistent.
    """

    #use_mom = (filter_mode == "mom")

    train_fund_signal = None
    if filter_mode == "fund" and fund_params is not None:
        funds_train = funds_df.loc[
            (funds_df.index >= train_start) &
            (funds_df.index < train_end)
        ]
        train_fund_signal = compute_fund_breadth_signal(
            funds_train, **fund_params
        )

    bt, metrics, trades, _ = run_strategy_with_trades(
        train,
        cash_df=cash_train,
        price_col="Zamkniecie",
        X=X, Y=Y,
        stop_loss=stop_loss,
        fast=fast, slow=slow,
        target_vol=tv,
        vol_window=vol_window,
        position_mode=selected_mode,
        filter_mode=filter_mode,
        mom_lookback=mom_lookback,    
        fund_signal=train_fund_signal,
        fast_mode=fast_mode,
        use_atr_stop=use_atr_stop,
        N_atr=N_atr,
        atr_window=atr_window,
    )

    if metrics is None:
        return None

    max_dd  = metrics.get("MaxDD", 0)
    sharpe  = metrics.get("Sharpe", 0)
    calmar  = metrics["CAGR"] / abs(max_dd) if max_dd != 0 else None
    sortino = metrics.get("Sortino", 0)

    if objective == "calmar":
        if calmar is None:
            return None
        obj_value = calmar
    elif objective == "sharpe":
        obj_value = sharpe
    elif objective == "sortino":
        obj_value = sortino
    elif objective == "calmar_sharpe":
        if calmar is None:
            return None
        obj_value = 0.5 * calmar + 0.5 * sharpe
    elif objective == "calmar_sortino":
        if calmar is None:
            return None
        obj_value = 0.5 * calmar + 0.5 * sortino
    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    # Key tuple: position 2 holds the stop parameter.
    # In fixed mode: X (a fraction).
    # In ATR mode:   N_atr (a multiplier).
    # walk_forward uses the same convention when building param_combinations,
    # so the key is consistent across both modes.
    stop_key_val = N_atr if use_atr_stop else X
    key = (filter_mode, fund_idx, stop_key_val, Y, fast, slow, tv, stop_loss, mom_lookback)
    return key, obj_value




def walk_forward(
    df,
    cash_df,
    train_years=8,
    test_years=2,
    vol_window  = 20,
    funds_df=None,
    fund_params_grid=None,
    selected_mode="full",
    filter_modes_override=None,
    X_grid = [0.08, 0.10, 0.12, 0.15, 0.20],
    Y_grid = [0.02, 0.03, 0.05, 0.07, 0.10],
    fast_grid   = [50, 75, 100],
    slow_grid = [150, 200, 250 ],
    tv_grid = [0.08, 0.10, 0.12, 0.15, 0.20],
    sl_grid     = [0.05, 0.08, 0.10, 0.15],
    mom_lookback_grid = [126, 252],    
    objective = "calmar",
    n_jobs=1,
    entry_gate_series=None,
    fast_mode=True,
    # -------------------------------------------------------
    # ATR trailing stop parameters
    # -------------------------------------------------------
    use_atr_stop=False,       # False = fixed % X, True = ATR-scaled N_atr
    N_atr_grid=None,          # ATR multiplier grid; replaces X_grid when
                              # use_atr_stop=True. Default: [2.0,3.0,4.0,5.0,6.0]
    atr_window=20,            # Rolling window for ATR estimate (days)
):
    """
    Run a rolling walk-forward optimisation and return a stitched
    out-of-sample equity curve.

    ATR STOP MODE
    -------------
    When use_atr_stop=True, the trailing stop parameter searched over is
    N_atr (ATR multiplier) rather than X (fixed fraction). N_atr_grid
    replaces X_grid in the parameter combinations loop. The key tuple
    structure is unchanged — position 2 simply holds N_atr instead of X.
    neighbour_mean receives N_atr_grid as the stop_grid argument.

    All other walk-forward mechanics (carry state, warmup, stability
    penalty, OOS stitching) are identical to the fixed-stop mode.

    Backward compatibility: use_atr_stop defaults to False. Existing
    callers that do not pass ATR parameters are completely unaffected.
    """

    # Resolve ATR grid default
    if N_atr_grid is None:
        N_atr_grid = [2.0, 3.0, 4.0, 5.0, 6.0]

    # The stop grid for neighbour_mean: X_grid in fixed mode, N_atr_grid in ATR mode
    stop_grid = N_atr_grid if use_atr_stop else X_grid

    data_end = df.index.max()
    logging.info("walk_forward received data from %s to %s (%d rows)",
             df.index.min(), data_end, len(df))
    logging.info("Objective function: %s", objective)
    logging.info(
        "Trailing stop mode: %s  (ATR window=%d)",
        "ATR-scaled (Chandelier)" if use_atr_stop else "fixed percentage",
        atr_window
    )

    oos_equity_slices = []
    results           = []
    all_oos_trades    = []

    start       = df.index.min()
    carry_state = None
    
    if filter_modes_override is not None:
        logging.info("filter_modes overridden to: %s", filter_modes_override)

    while True:
        gate_train = None
        gate_oos   = None
        train_start = start
        train_end   = train_start + pd.DateOffset(years=train_years)
        test_end    = train_end   + pd.DateOffset(years=test_years)

        train = df.loc[(df.index >= train_start) & (df.index < train_end)]

        next_window_has_enough = (
            len(df.loc[df.index >= test_end]) >= (test_years * 252)
            )
        if not next_window_has_enough:
            test_end = data_end + pd.DateOffset(days=1)

        test = df.loc[(df.index >= train_end) & (df.index < test_end)]

        logging.info(
            "Iteration: train=%s to %s (%d rows) | test=%s to %s (%d rows) | "
            "next_window_has_enough=%s | data_end=%s",
            train_start.date(), train_end.date(), len(train),
            train_end.date(), test_end.date(), len(test),
            next_window_has_enough,
            data_end.date()
            )

        if train.empty or len(test) < 30:
            logging.info("Breaking — train empty or test too short")
            break

        cash_train = cash_df.loc[
            (cash_df.index >= train_start) & (cash_df.index < train_end)
            ]
        
        gate_train = None
        if entry_gate_series is not None:
            gate_train = entry_gate_series.reindex(
                train.index, method="ffill"
                ).fillna(1).astype(int)

        # -------------------------------------------------------
        # Build parameter combinations
        # In ATR mode: iterate over N_atr_grid instead of X_grid.
        # The loop variable is named stop_val in both cases and placed
        # at key position 2 — consistent with evaluate_params key tuple.
        # -------------------------------------------------------
        param_scores = {}

        filter_modes = ["ma", "mom", "mom_blend"]
        if funds_df is not None:
            filter_modes.append("fund")
        
        if filter_modes_override is not None:
            filter_modes = filter_modes_override

        param_combinations = []

        for filter_mode in filter_modes:
            fast_iter     = fast_grid if filter_mode == "ma" else [50]
            slow_iter     = slow_grid if filter_mode == "ma" else [200]
            mom_lb_iter   = mom_lookback_grid if filter_mode == "mom" else [252]
            fund_iter     = (list(enumerate(fund_params_grid))
                             if filter_mode == "fund" else [(None, None)])
            # Stop grid: N_atr_grid in ATR mode, X_grid in fixed mode
            stop_iter     = N_atr_grid if use_atr_stop else X_grid

            for fund_idx, fund_params in fund_iter:
                for stop_val in stop_iter:           # stop_val = N_atr or X
                    for Y in Y_grid:
                        for fast in fast_iter:
                            for slow in slow_iter:
                                if filter_mode == "ma" and slow - fast < 75:
                                    continue
                                for tv in tv_grid if selected_mode != "full" else [0.10]:
                                    for stop_loss in sl_grid:
                                        # In fixed mode: stop_loss must be < X
                                        # In ATR mode: no equivalent constraint
                                        # (absolute stop fraction is independent
                                        # of the ATR multiplier)
                                        if not use_atr_stop and stop_loss >= stop_val:
                                            continue
                                        for mom_lookback in mom_lb_iter:
                                            param_combinations.append((
                                                filter_mode, fund_idx, fund_params,
                                                stop_val, Y, fast, slow, tv,
                                                stop_loss, mom_lookback
                                            ))

        for backend, n_jobs_inner, label in [
                ("loky",      n_jobs, "multiprocessing"),
                ("threading", n_jobs, "threading"),
                (None,        1,      "sequential"),
                ]:
            try:
                if backend is None:
                    results_list = [
                        evaluate_params(
                            filter_mode, fund_idx, fund_params,
                            stop_val, Y, fast, slow, tv, stop_loss,
                            train, cash_train, vol_window, selected_mode,
                            funds_df, train_start, train_end,
                            objective=objective, mom_lookback=mom_lookback,
                            entry_gate=gate_train,
                            fast_mode=fast_mode,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val,
                            atr_window=atr_window,
                            )
                        for (filter_mode, fund_idx, fund_params,
                             stop_val, Y, fast, slow, tv, stop_loss, mom_lookback)
                        in param_combinations
                        ]
                else:
                    results_list = Parallel(n_jobs=n_jobs_inner, backend=backend)(
                        delayed(evaluate_params)(
                            filter_mode, fund_idx, fund_params,
                            stop_val, Y, fast, slow, tv, stop_loss,
                            train, cash_train, vol_window, selected_mode,
                            funds_df, train_start, train_end,
                            objective=objective, mom_lookback=mom_lookback,
                            entry_gate=gate_train, fast_mode=fast_mode,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val,
                            atr_window=atr_window,
                            )
                        for (filter_mode, fund_idx, fund_params,
                             stop_val, Y, fast, slow, tv, stop_loss, mom_lookback)
                        in param_combinations
                        )

                logging.info(
                    "Grid search completed using %s backend (%d jobs).",
                    label, n_jobs_inner
                    )
                break

            except Exception as e:
                logging.warning(
                    "Grid search backend '%s' failed: %s — trying next option.",
                    label, e
                    )
                results_list = None

        if results_list is None:
            logging.error("All grid search backends failed. Skipping window.")
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue
            
        param_scores = {
            key: score
            for result in results_list
            if result is not None
            for key, score in [result]
        }

        if not param_scores:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        # -------------------------------------------------------
        # Stability-penalised selection
        # pass stop_grid (X_grid or N_atr_grid) to neighbour_mean
        # -------------------------------------------------------
        best_score      = -np.inf
        best_params     = None
        best_raw_score  = -np.inf
        
        for key, raw_score in param_scores.items():
            same_mode_scores = {
                k: v for k, v in param_scores.items()
                if k[0] == key[0]
                }
            stability = neighbour_mean(key, same_mode_scores, stop_grid, Y_grid)
            combined  = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score     = combined
                best_raw_score = raw_score
                fund_idx = key[1]

                best_params = {
                    "filter_mode":  key[0],
                    "fund_idx":     fund_idx,
                    "fund_params":  (fund_params_grid[fund_idx]
                                     if fund_idx is not None and fund_params_grid is not None
                                     else None),
                    # key[2] is stop parameter: X in fixed mode, N_atr in ATR mode
                    "X":            key[2] if not use_atr_stop else X_grid[0],
                    "N_atr":        key[2] if use_atr_stop else N_atr_grid[0],
                    "Y":            key[3],
                    "fast":         key[4],
                    "slow":         key[5],
                    "stop_loss":    key[7],
                    "mom_lookback": key[8],
                    "use_atr_stop": use_atr_stop,
                    "atr_window":   atr_window,
                }
                if selected_mode != "full":
                    best_params["target_vol"] = key[6]
            
        if best_params is None:
            break
        else:
            stop_label = (f"N_atr={best_params['N_atr']:.1f}"
                          if use_atr_stop else f"X={best_params['X']:.2f}")
            logging.info(
                "Window %s: best raw_%s=%.4f | penalised_%s=%.4f | filter=%s | "
                "%s Y=%.2f fast=%d slow=%d sl=%.2f tv=%s mom_lookback=%s",
                train_start.date(),
                objective, best_raw_score,
                objective, best_score,
                best_params["filter_mode"],
                stop_label,
                best_params["Y"],
                best_params["fast"], best_params["slow"],
                best_params["stop_loss"],
                best_params.get("target_vol", "N/A"),
                best_params["mom_lookback"]                
                )   
        
        
        # -------------------------------------------------------
        # OOS run — pass carry_state and ATR parameters
        # -------------------------------------------------------
        
        WARMUP_BARS = best_params["slow"] + vol_window + 10
        warmup      = train.iloc[-WARMUP_BARS:]
        
        warmup_start = warmup.index.min()
        cash_warmup_and_test = cash_df.loc[
            (cash_df.index >= warmup_start) &
            (cash_df.index < test_end)
            ]
        
        oos_fund_signal = None
        if best_params["filter_mode"] == "fund" and best_params["fund_params"] is not None:
            funds_warmup_and_test = funds_df.loc[
                (funds_df.index >= warmup.index.min()) &
                (funds_df.index < test_end)
            ]
            full_fund_signal = compute_fund_breadth_signal(
                funds_warmup_and_test,
                **best_params["fund_params"]
            )
            oos_fund_signal = full_fund_signal.loc[
                full_fund_signal.index >= train_end
            ]
            gate_oos = None
            if entry_gate_series is not None:
                gate_oos = entry_gate_series.reindex(
                    test.index, method="ffill"
                    ).fillna(1).astype(int)

        # Build kwargs for run_strategy_with_trades — exclude meta keys
        _strategy_keys_to_exclude = {
            "filter_mode", "fund_params", "fund_idx",
             "target_vol", "use_atr_stop", "mom_lookback",
            "atr_window", "N_atr",
        }

        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            test,
            price_col="Zamkniecie",
            cash_df=cash_warmup_and_test,
            position_mode=selected_mode,
            vol_window=vol_window,
            initial_state=carry_state,
            warmup_df=warmup,
            entry_gate=gate_oos if 'gate_oos' in dir() else None,
            fund_signal=oos_fund_signal,
            fast_mode=fast_mode,
            filter_mode=best_params["filter_mode"],
            mom_lookback=best_params["mom_lookback"],
            use_atr_stop=best_params["use_atr_stop"],
            N_atr=best_params["N_atr"],
            atr_window=best_params["atr_window"],
            **{k: v for k, v in best_params.items()
               if k not in _strategy_keys_to_exclude}
            )

        if test_metrics is None or bt_oos is None:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        carry_state = end_state

        equity_slice = bt_oos["equity"].copy()
        if oos_equity_slices:
            prev_end     = oos_equity_slices[-1].iloc[-1]
            equity_slice = equity_slice * prev_end

        oos_equity_slices.append(equity_slice)

        if not oos_trades.empty:
            oos_trades = oos_trades.copy()
            oos_trades["WF_Window"] = train_start
            all_oos_trades.append(oos_trades)

        results.append({
            "TrainStart":   train_start,
            "TrainEnd":     train_end,
            "TestStart":    train_end,
            "TestEnd":      test_end,
            "filter_mode":  best_params["filter_mode"],
            "fund_idx":     best_params["fund_idx"],
            "fund_params":  str(best_params["fund_params"]),
            **{k: v for k, v in best_params.items()
               if k not in ("filter_mode", "fund_params", "fund_idx",
                     "target_vol", "use_atr_stop",
                    "atr_window")},
            "target_vol":   best_params.get("target_vol", "N/A"),
            "mom_lookback": best_params.get("mom_lookback", 252),
            "use_atr_stop": best_params["use_atr_stop"],
            "atr_window":   best_params["atr_window"],
            **test_metrics
            })
        
        start += pd.DateOffset(years=test_years)
        
        if not next_window_has_enough:
            logging.info(
                "Window: train=%s to %s | test=%s to %s | rows in test=%d | "
                "data_end=%s | next_window_data_rows=%d | next_window_has_enough=%s",
                train_start, train_end,
                train_end, test_end,
                len(test),
                data_end,
                len(df.loc[df.index >= test_end]),
                next_window_has_enough
            )
            logging.info("Final OOS window extended to %s (end of data).", data_end)
            break

    if carry_state is not None and all_oos_trades:
        logging.info(
            "Position still open at end of final window. "
            "Last CARRY trade represents the open P&L."
        )

    if not oos_equity_slices:
        logging.warning("Walk-forward produced no OOS results.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    oos_equity    = pd.concat(oos_equity_slices).sort_index()
    results_df    = pd.DataFrame(results)
    oos_trades_df = pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()

    return oos_equity, results_df, oos_trades_df
        



# ============================================================
# TRADE ANALYSIS
# ============================================================



def analyze_trades(trades, 
                   boundary_exits= {"CARRY", "SAMPLE_END"}):
    
    if trades.empty:
        return None

    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()

    if trades.empty:
        return None

    n_cross = trades["CrossWindow"].sum() if "CrossWindow" in trades.columns else 0
    if n_cross > 0:
        logging.info("%d trades carried across window boundaries", n_cross)

    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())

    pf = np.inf if loss == 0 else (
        trades.loc[trades["Return"] > 0, "Return"].sum() / loss
    )

    return {
        "Trades":       len(trades),
        "WinRate":      (trades["Return"] > 0).mean(),
        "AvgWin":       trades.loc[trades["Return"] > 0, "Return"].mean(),
        "AvgLoss":      trades.loc[trades["Return"] < 0, "Return"].mean(),
        "ProfitFactor": pf,
        "AvgDays":      trades["Days"].mean(),
        "CrossWindow":  int(n_cross)
    }


# ------------------------
# Report Printing Function with Best Parameters
# ------------------------
def print_backtest_report(metrics, 
                          trades, 
                          trade_stats, 
                          best_params=None, 
                          wf_results=None, 
                          position_mode=None,
                          filter_modes_override=None  
                          ):

    logging.info("="*80)
    logging.info(f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}")
    if filter_modes_override is not None:
        logging.info(f"Filter mode was forced to:    {filter_modes_override}")
    else:
        logging.info("Filter mode selection set to automatic")
    logging.info("="*80)

    if wf_results is not None and not wf_results.empty:
        # Determine which stop column to show based on what is in wf_results
        use_atr = (
            "use_atr_stop" in wf_results.columns
            and wf_results["use_atr_stop"].any()
        )
        stop_col = "N_atr" if use_atr else "X"

        cols = ["TrainStart", "TestStart", "filter_mode",
                stop_col, "Y", "fast", "slow", "target_vol",
                "stop_loss", "mom_lookback"]
        # Only include columns that actually exist (graceful fallback)
        cols = [c for c in cols if c in wf_results.columns]

        if "fund_params" in wf_results.columns and \
           wf_results["filter_mode"].eq("fund").any():
            cols.insert(3, "fund_params")

        if use_atr and "atr_window" in wf_results.columns:
            # Show atr_window once as a header note, not per-row
            aw = wf_results["atr_window"].iloc[0]
            logging.info("ATR trailing stop mode active (atr_window=%d)", aw)

        logging.info("\n%s", wf_results[cols].to_string(index=False))

    logging.info("-"*80)

    # Metrics
    logging.info("METRICS:")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f | Sortino: %.2f",
        metrics["CAGR"]*100,
        metrics["Vol"]*100,
        metrics["Sharpe"],
        metrics["MaxDD"]*100,
        metrics["CalMAR"],
        metrics["Sortino"]
    )
    logging.info("-"*80)

    if trade_stats:
        logging.info("TRADE STATISTICS:")
        logging.info(
            "Total Trades: %d | Win Rate: %.1f%% | Avg Win: %.2f%% | "
            "Avg Loss: %.2f%% | Profit Factor: %.2f | Avg Days: %.1f",
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

    carry_trades = pd.DataFrame()

    if not trades.empty and "Exit Reason" in trades.columns:
        carry_trades = trades[trades["Exit Reason"] == "CARRY"]

        n_carry = len(carry_trades)
        if n_carry > 0:
            logging.info(
                "Note: trade log includes %d CARRY boundary records "
                "excluded from statistics above.", n_carry
            )

        trades_fmt = trades.copy()
        trades_fmt['Return'] = (trades_fmt['Return']*100).round(2).astype(str) + '%'
        trades_fmt['EntryPrice'] = trades_fmt['EntryPrice'].round(2)
        trades_fmt['ExitPrice'] = trades_fmt['ExitPrice'].round(2)
        logging.info("TRADE LOG:")
        logging.info("\n%s", trades_fmt.to_string(index=False))

    if not trades.empty and trades.iloc[-1]["Exit Reason"] == "CARRY":
        last_carry = trades.iloc[-1]
        logging.info(
            "Open position at report date: entry %s at %.2f, "
            "current value %.2f, unrealised return %.1f%%",
            last_carry["EntryDate"],
            last_carry["EntryPrice"],
            last_carry["ExitPrice"],
            last_carry["Return"] * 100
        )
    logging.info("="*80)



#----------------------------------
# Regime analysis
#---------------------------------


def prepare_regime_inputs(df, wf_results, wf_equity, bh_equity):
    oos_start = pd.Timestamp(wf_results["TestStart"].min())
    oos_end   = pd.Timestamp(wf_results["TestEnd"].max())

    mask = (df.index >= oos_start) & (df.index <= oos_end)
    close = df.loc[mask, "Zamkniecie"].copy()
    high  = df.loc[mask, "Najwyzszy"].copy()
    low   = df.loc[mask, "Najnizszy"].copy()

    equity_strat = wf_equity.reindex(close.index).ffill()
    equity_bh    = bh_equity.reindex(close.index).ffill()

    daily_returns_strat = equity_strat.pct_change().dropna()
    daily_returns_bh    = equity_bh.pct_change().dropna()

    close = close.reindex(daily_returns_strat.index)
    high  = high.reindex(daily_returns_strat.index)
    low   = low.reindex(daily_returns_strat.index)

    return dict(
        close               = close,
        high                = high,
        low                 = low,
        daily_returns_strat = daily_returns_strat,
        daily_returns_bh    = daily_returns_bh,
        equity_strat        = equity_strat,
        equity_bh           = equity_bh,
        oos_start           = oos_start,
        oos_end             = oos_end,
    )


def compute_adx(high, low, close, period=20):
    delta_high = high.diff()
    delta_low  = -low.diff()
    plus_dm  = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0.0)
    minus_dm = np.where((delta_low > delta_high) & (delta_low  > 0), delta_low,  0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr     = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di

def label_regime_adx(close, high, low, period=20, trend_thresh=25, chop_thresh=20):
    adx, plus_di, minus_di = compute_adx(high, low, close, period)
    regime = pd.Series("sideways", index=close.index)
    regime[adx > trend_thresh] = np.where(
        plus_di[adx > trend_thresh] > minus_di[adx > trend_thresh], "uptrend", "downtrend"
    )
    regime[adx < chop_thresh] = "sideways"
    return regime.rename("regime_adx")

def label_regime_momentum(close, window=63, thresh=0.03):
    ret = close.pct_change(window)
    regime = pd.Series("sideways", index=close.index)
    regime[ret >  thresh] = "uptrend"
    regime[ret < -thresh] = "downtrend"
    return regime.rename("regime_mom")

def label_regime_vol(close, window=21, hi_pct=0.67, lo_pct=0.33):
    rv = close.pct_change().rolling(window).std() * np.sqrt(252)
    rolling_med = rv.rolling(252, min_periods=60).median()
    regime = pd.Series("normal_vol", index=close.index)
    regime[rv > rolling_med * 1.3] = "high_vol"
    regime[rv < rolling_med * 0.7] = "low_vol"
    return regime.rename("regime_vol")


def regime_stats(daily_returns_strat, daily_returns_bh, regime_series, label="regime"):
    df = pd.DataFrame({
        "strat": daily_returns_strat,
        "bh"   : daily_returns_bh,
        "regime": regime_series
    }).dropna()

    rows = []
    for reg, grp in df.groupby("regime"):
        n_days  = len(grp)
        pct_time = n_days / len(df) * 100
        ann = 252

        def ann_return(r):
            return (1 + r).prod() ** (ann / len(r)) - 1

        def ann_vol(r):
            return r.std() * np.sqrt(ann)

        def max_dd(r):
            cum = (1 + r).cumprod()
            return (cum / cum.cummax() - 1).min()

        rows.append({
            label          : reg,
            "days"         : n_days,
            "pct_time"     : round(pct_time, 1),
            "strat_cagr"   : round(ann_return(grp.strat) * 100, 2),
            "bh_cagr"      : round(ann_return(grp.bh)    * 100, 2),
            "strat_vol"    : round(ann_vol(grp.strat) * 100, 2),
            "bh_vol"       : round(ann_vol(grp.bh)    * 100, 2),
            "strat_maxdd"  : round(max_dd(grp.strat)   * 100, 2),
            "bh_maxdd"     : round(max_dd(grp.bh)      * 100, 2),
            "strat_sharpe" : round(ann_return(grp.strat) / ann_vol(grp.strat), 3)
                             if ann_vol(grp.strat) > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index(label)


def regime_transition_matrix(regime_series):
    s = regime_series.dropna()
    regimes = sorted(s.unique())
    mat = pd.DataFrame(0.0, index=regimes, columns=regimes)
    for a, b in zip(s.iloc[:-1], s.iloc[1:]):
        mat.loc[a, b] += 1
    mat = mat.div(mat.sum(axis=1), axis=0)
    return mat.round(3)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLOURS = {
    "uptrend"   : "#c6efce",
    "downtrend" : "#ffc7ce",
    "sideways"  : "#ffeb9c",
    "high_vol"  : "#ffc7ce",
    "normal_vol": "#c6efce",
    "low_vol"   : "#deebf7",
}

def plot_regime_overlay(close, regime_series, equity_strat, equity_bh,
                        title="Regime decomposition"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    prev_regime = None
    prev_date   = regime_series.index[0]
    for date, reg in regime_series.items():
        if reg != prev_regime and prev_regime is not None:
            ax1.axvspan(prev_date, date,
                        color=REGIME_COLOURS.get(prev_regime, "#eeeeee"), alpha=0.6)
            ax2.axvspan(prev_date, date,
                        color=REGIME_COLOURS.get(prev_regime, "#eeeeee"), alpha=0.3)
        prev_regime = reg
        prev_date   = date
    ax1.axvspan(prev_date, regime_series.index[-1],
                color=REGIME_COLOURS.get(prev_regime, "#eeeeee"), alpha=0.6)
    ax2.axvspan(prev_date, regime_series.index[-1],
                color=REGIME_COLOURS.get(prev_regime, "#eeeeee"), alpha=0.3)

    ax1.plot(close / close.iloc[0], color="steelblue", linewidth=1, label="Equity (normalised)")
    ax1.set_ylabel("Index level")
    ax1.legend(fontsize=9)
    ax1.set_title(title)

    ax2.plot(equity_strat / equity_strat.iloc[0], color="navy",   linewidth=1.5, label="Strategy")
    ax2.plot(equity_bh    / equity_bh.iloc[0],    color="grey",   linewidth=1,   linestyle="--", label="Buy & Hold")
    ax2.set_ylabel("Normalised equity")
    ax2.legend(fontsize=9)

    patches = [mpatches.Patch(color=c, alpha=0.6, label=l)
               for l, c in REGIME_COLOURS.items()]
    ax2.legend(handles=patches + ax2.get_lines(), fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


def plot_regime_bar_comparison(stats_df, title="Performance by regime"):
    colours = ["#2E75B6", "#9DC3E6", "#C00000", "#FF9999"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(stats_df))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, stats_df["strat_cagr"], w, label="Strategy",   color=colours[0])
    ax.bar(x + w/2, stats_df["bh_cagr"],    w, label="Buy & Hold", color=colours[1])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df.index, rotation=15)
    ax.set_ylabel("Annualised return (%)")
    ax.set_title("CAGR by regime")
    ax.legend()

    ax = axes[1]
    ax.bar(x - w/2, stats_df["strat_maxdd"], w, label="Strategy",   color=colours[2])
    ax.bar(x + w/2, stats_df["bh_maxdd"],    w, label="Buy & Hold", color=colours[3])
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df.index, rotation=15)
    ax.set_ylabel("Max drawdown (%)")
    ax.set_title("Max drawdown by regime")
    ax.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def run_regime_decomposition(close, high, low,
                              daily_returns_strat, daily_returns_bh,
                              equity_strat, equity_bh):
    results = {}

    for name, regime_fn in [
        ("ADX trend",  lambda: label_regime_adx(close, high, low)),
        ("Momentum",   lambda: label_regime_momentum(close)),
        ("Volatility", lambda: label_regime_vol(close)),
    ]:
        regime = regime_fn().reindex(daily_returns_strat.index)
        stats  = regime_stats(daily_returns_strat, daily_returns_bh, regime, label="regime")
        trans  = regime_transition_matrix(regime)

        fig_overlay = plot_regime_overlay(
            close.reindex(daily_returns_strat.index), regime,
            equity_strat, equity_bh, title=f"Regime overlay — {name}"
        )
        fig_bars = plot_regime_bar_comparison(stats, title=f"Performance by regime — {name}")

        results[name] = {
            "stats"      : stats,
            "transitions": trans,
            "fig_overlay": fig_overlay,
            "fig_bars"   : fig_bars,
        }
        logging.info(f"\n{'─'*60}")
        logging.info(f"  {name} regime decomposition")
        logging.info(f"{'─'*60}")
        logging.info(stats.to_string())
        logging.info(f"\nTransition matrix:\n{trans.to_string()}")

    return results