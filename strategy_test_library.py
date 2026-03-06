import pandas as pd
import numpy as np
import requests
import time
import random
import os
import sys
import datetime as dt
from joblib import Parallel, delayed



import logging



# ============================================================
# DATA ACQUISITION
# ============================================================

def download_csv(url, filename):


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

    Each fund is identified by a four-digit stooq code. The URL pattern
    is https://stooq.pl/q/d/l/?s={code}.n&i=d — the same format used
    for the money market fund (2720.n).

    Successfully downloaded funds are added to FUND_FILES. Failed
    downloads are logged and skipped — the caller should check that
    the returned dict has enough entries before passing to build_funds_df.

    Parameters
    ----------
    fund_codes : dict[str, str] — mapping of four-digit stooq code to
                 human-readable fund name, e.g. {"1234": "PKO_Akcji"}
    tmp_dir    : str            — directory for temporary CSV files

    Returns
    -------
    dict[str, str] — mapping of fund name to local CSV filepath,
                     suitable for passing directly to build_funds_df.
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

    Each file is loaded via load_csv and the closing price column is
    extracted and renamed to the fund identifier. Files that fail to
    load, lack the price column, or have insufficient history are
    skipped with a warning.

    The panel is constructed as an outer join so no dates are dropped
    if funds have slightly different trading calendars. Missing values
    are forward-filled (fund NAV not published on a given day) and then
    any remaining leading NaNs are dropped.

    Funds with excessive missing data after forward-fill (more than
    max_gap_days consecutive NaNs at any point) are excluded — this
    catches funds that were suspended or had long reporting gaps.

    Parameters
    ----------
    fund_files       : dict[str, str] — mapping of fund identifier to
                       CSV filepath, e.g. {"fund_A": "C:/tmp/fund_a.csv"}
    price_col        : str            — column name for NAV/closing price
    min_history_years: int            — exclude funds with fewer than this
                       many years of data (default 10)

    Returns
    -------
    pd.DataFrame — date-indexed panel, one column per fund, forward-filled.
                   Returns empty DataFrame if fewer than 2 funds loaded.
    """

    MAX_GAP_DAYS = 30   # fund suspended or data missing — exclude

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

        # Check minimum history
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

    # Outer join — preserves all dates across funds
    funds_df = pd.concat(series_list, axis=1, join="outer", sort=True)
    funds_df.sort_index(inplace=True)

    # Forward-fill NAV gaps (weekends, holidays, reporting lags)
    funds_df = funds_df.ffill()

    # Check for excessive gaps after forward-fill
    # (remaining NaNs are leading NaNs before fund inception)
    for fund_id in funds_df.columns:
        col = funds_df[fund_id]
        # Find longest consecutive NaN run in the interior of the series
        # (exclude leading NaNs which are handled by dropna below)
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

    # Drop leading rows where fewer than half the funds have data
    # This avoids a sparse panel at the start where few funds existed
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
    
    """
    Compute daily returns from a money market or bond fund price series.

    The resulting 'cash_ret' column is used in run_strategy_with_trades
    to credit the out-of-market portion of capital with a realistic
    return rather than assuming zero. This avoids overstating the
    opportunity cost of being flat.

    Parameters
    ----------
    cash_df   : DataFrame — price series with a date index
    price_col : str       — column containing the NAV/price
    
    Returns
    -------
    DataFrame with a single column 'cash_ret', indexed by date.
    """
    
    
    cash = cash_df.copy()

    cash["cash_price"] = cash[price_col]

    cash["cash_ret"] = cash["cash_price"].pct_change()

    cash = cash[["cash_ret"]].dropna()

    return cash


# ============================
# Indicators
# ============================



def compute_momentum(series, lookback=252, skip=21):

    """
    Compute a 12-month momentum signal with a 1-month skip.

    Returns the ratio of price 'skip' days ago to price 'lookback'
    days ago, minus 1. The skip avoids the well-documented short-term
    reversal effect that would otherwise contaminate a pure momentum
    signal.

    All values are shifted (lagged) so that the signal on any given
    day uses only information available before that day's open.

    Parameters
    ----------
    series   : pd.Series — price series
    lookback : int       — total lookback in trading days (default 252 = 1 year)
    skip     : int       — recent period to exclude (default 21 = 1 month)
    """

    return series.shift(skip) / series.shift(lookback) - 1




# ============================
# FUND BREADTH SIGNAL
# ============================

def compute_fund_breadth_signal(
    funds_df,           # DataFrame: date index, one column per fund NAV
    lookback_days=30,   # B / E — rolling window for return calculation
    n_top=2,            # number of best/worst funds to consider
    entry_roll_thresh=0.03,    # A — min return over lookback_days to trigger entry
    entry_since_thresh=0.05,   # C — min return since last signal change for entry
    exit_roll_thresh=-0.03,    # D — max return over lookback_days to trigger exit
    exit_since_thresh=-0.05,   # F — max return since last signal change for exit
):
    """
    Compute a binary IN/OUT signal from a panel of fund NAV series.

    Entry (OUT -> IN) triggered if EITHER:
      - The mean return of the top n_top funds over the last lookback_days
        exceeds entry_roll_thresh
      - The mean return of the top n_top funds since the last signal change
        exceeds entry_since_thresh

    Exit (IN -> OUT) triggered if EITHER:
      - The mean return of the bottom n_top funds over the last lookback_days
        is below exit_roll_thresh
      - The mean return of the bottom n_top funds since the last signal change
        is below exit_since_thresh

    Returns
    -------
    pd.Series of 1 (IN) / 0 (OUT), indexed by date
    """

    # Daily returns for all funds
    fund_rets = funds_df.pct_change()

    # Rolling return over lookback_days for each fund
    roll_ret = (1 + fund_rets).rolling(lookback_days).apply(
        np.prod, raw=True
    ) - 1

    signal   = pd.Series(0, index=funds_df.index)
    state    = 0       # 0 = OUT, 1 = IN
    last_change_idx = funds_df.index[0]

    for i, date in enumerate(funds_df.index):

        if i < lookback_days:
            signal.iloc[i] = state
            continue

        todays_roll = roll_ret.loc[date].dropna()

        if todays_roll.empty:
            signal.iloc[i] = state
            continue

        # "Since last signal change" returns
        # Use last available value on or before last_change_idx
        # to handle cases where that date is not in the fund index
        ref_prices = funds_df.loc[:last_change_idx].iloc[-1]
        curr_prices = funds_df.loc[date]

        since_rets = (curr_prices / ref_prices - 1).dropna()

        # Only rank funds that have valid data in BOTH rolling
        # and since-last-change calculations — take intersection
        common_funds = todays_roll.index.intersection(since_rets.index)

        if len(common_funds) < n_top:
            # Not enough funds with complete data — hold current state
            signal.iloc[i] = state
            continue

        todays_roll  = todays_roll.loc[common_funds]
        since_rets   = since_rets.loc[common_funds]

        top_funds    = todays_roll.nlargest(n_top)
        bottom_funds = todays_roll.nsmallest(n_top)

        top_since    = since_rets.loc[top_funds.index]
        bottom_since = since_rets.loc[bottom_funds.index]

        if state == 0:   # currently OUT — look for entry
            roll_condition  = top_funds.mean()  >= entry_roll_thresh
            since_condition = top_since.mean()  >= entry_since_thresh

            if roll_condition or since_condition:
                state = 1
                last_change_idx = date

        elif state == 1:   # currently IN — look for exit
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

    """
    Compute standard performance metrics from an equity curve.

    Metrics returned:
      CAGR   — compound annual growth rate
      Vol    — annualised volatility of daily returns
      Sharpe — CAGR divided by Vol (simplified, no risk-free rate)
      MaxDD  — maximum peak-to-trough drawdown (negative value)
      CalMAR — CAGR divided by absolute MaxDD

    Note: Sharpe here uses CAGR as the return numerator rather than
    mean daily return * freq. This is consistent across the codebase
    but differs slightly from the textbook definition.

    Parameters
    ----------
    equity : pd.Series — equity curve (should start at 1.0)
    freq   : int       — trading days per year for annualisation
    """

    ret = equity.pct_change().dropna()

    years = len(ret) / freq
    cagr = (equity.iloc[-1]/equity.iloc[0])**(1/years)-1
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0 else 0.0
    
    
    # Sortino — penalises downside volatility only
    # Downside returns = daily returns below the daily risk-free rate
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


def neighbour_mean(key, scores, X_grid, Y_grid):

    """
    Compute the mean score of neighbouring parameter sets for a given key.

    Used during walk-forward grid search to penalise parameter sets that
    sit in a narrow performance spike rather than a broad plateau.
    A strategy that is only optimal for very specific X/Y values is more
    likely to be overfit than one that performs well across a range.

    Neighbours are defined as all parameter sets that differ by at most
    one step in the X or Y dimension, holding fast/slow/tv/stop_loss
    fixed. The grids must be passed explicitly to guarantee they match
    the grids used in the search loop — this prevents silent bugs if the
    search grid is changed without updating this function.

    Parameters
    ----------
    key    : tuple        — (filter_mode, fund_paras, X, Y, fast, slow, tv, stop_loss)
    scores : dict         — maps parameter tuples to CalMAR scores
    X_grid : list[float]  — ordered list of X values used in the search
    Y_grid : list[float]  — ordered list of Y values used in the search

    Returns
    -------
    float — mean CalMAR of all neighbours including the key itself,
            or the key's own score if no neighbours exist in scores.
    """


    filter_mode, fund_idx, X, Y, fast, slow, tv, sl, mom_lookback = key

    xi = min(range(len(X_grid)), key=lambda i: abs(X_grid[i] - X))
    yi = min(range(len(Y_grid)), key=lambda i: abs(Y_grid[i] - Y))

    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nxi, nyi = xi + dx, yi + dy
            if 0 <= nxi < len(X_grid) and 0 <= nyi < len(Y_grid):
                nkey = (filter_mode, fund_idx,
                        X_grid[nxi], Y_grid[nyi],
                        fast, slow, tv, sl, mom_lookback)
                if nkey in scores:
                    neighbours.append(scores[nkey])

    return np.mean(neighbours) if neighbours else scores[key]

def calc_position(vol, position_mode, target_vol, max_leverage):

    """
    Compute the position size (fraction of capital to allocate).

    Three modes are supported:

      'full'        — always fully invested (position = 1.0).
                      Volatility targeting is ignored.

      'vol_entry'   — position sized at entry using volatility targeting:
                      position = target_vol / realised_vol, capped at
                      max_leverage. Size is fixed for the life of the trade.

      'vol_dynamic' — same formula as vol_entry but recomputed daily.
                      Rebalancing only occurs if the new size differs from
                      the current size by more than 10 percentage points,
                      to reduce unnecessary turnover.

    Parameters
    ----------
    vol           : float — annualised realised volatility (rolling window)
    position_mode : str   — one of 'full', 'vol_entry', 'vol_dynamic'
    target_vol    : float — desired annualised portfolio volatility
    max_leverage  : float — upper bound on position size (1.0 = no leverage)
    """
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
    """
    Compute buy-and-hold equity curve and metrics over the OOS period.
    Normalised to start at 1.0 to match the stitched OOS equity curve.
    """
    bh = df[price_col].copy()
    
    if start is not None:
        bh = bh.loc[bh.index >= start]
    if end is not None:
        bh = bh.loc[bh.index <= end]
    
    bh_equity = bh / bh.iloc[0]   # normalise to 1.0
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
    use_momentum=False,
    mom_lookback=252,       # ADD: 252 for 12M momentum, 126 for 6M momentum
    cash_df=None,
    safe_rate=0.0,
    initial_state=None,       # NEW: accept carry-over state from prior window,
    warmup_df=None,    # NEW: pre-window data for indicator warm-up
    fund_signal=None   # NEW: precomputed pd.Series from compute_fund_breadth_signal
):


    """
    Run the trend-following strategy on a single price series and return
    the equity curve, performance metrics, and trade log.

    ---------------------------------------------------------------
    STRATEGY LOGIC
    ---------------------------------------------------------------
    The strategy is a breakout/trend filter system. It enters long
    when two conditions are simultaneously met:

      Breakout  : price rises more than Y% above the running minimum
                  since the last exit (or start of data).
      Filter    : the fast moving average is above the slow moving
                  average (golden cross), AND momentum > 0 if enabled.

    It exits on the first of three conditions:

      TRAIL_STOP    : price falls more than X% below the running maximum
                      since entry.
      ABSOLUTE_STOP : price falls more than stop_loss% below entry price.
                      This limits the nominal loss on any single trade
                      independently of the trailing stop.
      FILTER_EXIT   : the trend/momentum filter turns off.

    When flat, capital earns the cash return (money market fund or
    safe_rate). When invested, the position fraction earns the index
    return and (1 - position) earns cash.

    ---------------------------------------------------------------
    WARMUP MECHANISM
    ---------------------------------------------------------------
    If warmup_df is provided, it is prepended to df before computing
    indicators. Warmup rows are tagged with _warmup=True and are used
    solely to initialise the rolling windows (MA, vol, momentum).
    During the main loop, warmup rows update the equity curve but
    suppress all entry/exit logic. After the loop, warmup rows are
    stripped from the output so the returned equity curve covers only
    the actual test period.

    This eliminates the data gaps that would otherwise appear at the
    start of each walk-forward test window due to rolling window
    initialisation consuming the first slow+vol_window bars.

    ---------------------------------------------------------------
    CARRY STATE MECHANISM
    ---------------------------------------------------------------
    If initial_state is provided, the strategy resumes an open position
    carried over from the previous walk-forward window. The position
    size, entry price, entry date, running maximum (M), running minimum
    (m), and cross-window flag are all restored from initial_state.

    At the end of the window, if a position is still open, end_state
    is returned containing the current position details. The caller
    (walk_forward) passes this into the next window's initial_state,
    ensuring seamless position continuity across window boundaries.

    Positions that span window boundaries are marked CrossWindow=True
    in the trade log. CARRY records (window-boundary snapshots of open
    positions) are written to the trade log for completeness but should
    be excluded from trade statistics — they are not real exits.

    ---------------------------------------------------------------
    EQUITY RENORMALISATION
    ---------------------------------------------------------------
    The equity curve is renormalised to start at 1.0 at the first test
    row (after warmup stripping). This is necessary for correct chain-
    linking in walk_forward, which assumes each window's equity slice
    begins at 1.0 and scales it to continue from the previous window's
    endpoint. Warmup-period P&L on carried positions is excluded from
    the OOS equity curve because warmup rows cover training-period dates,
    not OOS dates.

    ---------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------
    df            : DataFrame  — price series for the test period
    price_col     : str        — column name containing prices
    X             : float      — trailing stop threshold from peak (e.g. 0.10 = 10%)
    Y             : float      — breakout threshold from trough (e.g. 0.10 = 10%)
    stop_loss     : float      — absolute stop from entry price
    fast          : int        — fast MA window (bars)
    slow          : int        — slow MA window (bars)
    vol_window    : int        — rolling window for volatility estimate (bars)
    target_vol    : float      — target annualised volatility for position sizing
    max_leverage  : float      — maximum allowed position size (1.0 = no leverage)
    position_mode : str        — 'full', 'vol_entry', or 'vol_dynamic'
    use_momentum  : bool       — whether use momentum as the filter or MA cross
    cash_df       : DataFrame  — money market fund price series (optional)
    safe_rate     : float      — fallback annual cash rate if cash_df is None
    initial_state : dict|None  — carry-over position state from prior window
    warmup_df     : DataFrame|None — price rows to prepend for indicator warm-up
    fund_signal   : DataFrame  - series of signals computed from fund panel

    ---------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------
    df         : DataFrame  — test-period data with equity curve appended
    metrics    : dict       — CAGR, Vol, Sharpe, MaxDD, CalMAR
    trades_df  : DataFrame  — trade log including CARRY boundary records
    end_state  : dict|None  — open position state to carry into next window,
                              or None if flat at end of window
    """


    df = df.copy()
    df["price"] = df[price_col]

    # Check length BEFORE adding warmup
    min_required = max(vol_window, slow, fast, 252) + 5
    if fund_signal is not None:
        min_required = max(vol_window, 252) + 5   # MA params irrelevant
    if len(df) < min_required:
        return None, None, pd.DataFrame(), None

    # If warmup data is provided, prepend it for indicator calculation
    # but tag it so we can strip it from the equity curve after
    if warmup_df is not None:
        warmup = warmup_df.copy()
        warmup["price"] = warmup[price_col]
        warmup["_warmup"] = True
        df["_warmup"] = False
        df = pd.concat([warmup, df])
    else:
        df["_warmup"] = False

    test_start = df[~df["_warmup"]].index[0]  # capture before stripping

    if cash_df is not None:
        cash = prepare_cash_returns(cash_df)
        df = df.merge(cash, left_index=True, right_index=True, how="left")
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()
            
            
    else:
        df["cash_ret"] = safe_rate / 252

    if df["cash_ret"].isna().all():
        logging.warning("Cash series missing — falling back to flat safe_rate")
        df["cash_ret"] = safe_rate / 252

    # Compute risk-free rate from OOS cash returns only (exclude warmup)
    # Used for Sharpe calculation — MMF CAGR over the relevant period
    oos_cash = df.loc[~df["_warmup"], "cash_ret"]
    if len(oos_cash) > 0 and oos_cash.notna().any():
        cumulative = (1 + oos_cash).prod()
        n_years    = max(len(oos_cash) / 252, 0.01)
        rf_rate    = cumulative ** (1 / n_years) - 1
    else:
        rf_rate = safe_rate

    # Merge fund breadth signal if provided
    if fund_signal is not None:
        df = df.merge(
            fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left"
        )
        df["fund_filter"] = df["fund_filter"].ffill().fillna(0)
    else:
        df["fund_filter"] = 1   # always on if not provided
        
        
    df["ret"] = df["price"].pct_change()
    vol = df["ret"].rolling(vol_window).std() * np.sqrt(252)
    df["vol"] = vol.shift(1)
    df["ma_fast"] = df["price"].rolling(fast).mean().shift(1)
    df["ma_slow"] = df["price"].rolling(slow).mean().shift(1)
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    if use_momentum:
        df["MOM"] = compute_momentum(df["price"], lookback=mom_lookback).shift(1)
    else:
        df["MOM"] = 1

    df.dropna(inplace=True) # dropna now hits warmup rows, not test rows

    # -----------------------
    # Initialise state
    # -----------------------

    equity = 1.0
    equity_curve = []
    trades = []

    if initial_state is not None:
        # Restore carry-over position from previous window
        position    = initial_state["position"]
        entry_price = initial_state["entry_price"]
        entry_date  = initial_state["entry_date"]
        entry_reason= initial_state["entry_reason"]
        entry_pos   = initial_state["entry_pos"]
        M           = initial_state["M"]
        m           = initial_state["m"]
        entry_carried = True    # this position was carried in from previous window
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
    
    
    # Determine filter mode once before the loop
    if fund_signal is not None:
        filter_mode_active = "fund"
    elif use_momentum:
        filter_mode_active = "mom"
    else:
        filter_mode_active = "ma"
    
    # -----------------------

    # -----------------------
    # Main loop — track warmup rows separately

    

    for i, row in df.iterrows():

        price    = row["price"]
        ret      = row["ret"]
        cash_ret = row["cash_ret"]
        trend    = row["trend"]
        mom      = row["MOM"]
        vol      = row["vol"]
        
        
        if filter_mode_active == "fund":
            filter_on = bool(row["fund_filter"])
        elif filter_mode_active == "mom":
            filter_on = mom > 0
        else:
            filter_on = (trend == 1)
        
                
        is_warmup_row = row["_warmup"]    
        if is_warmup_row:
            equity_curve.append(equity)
            continue
        
        # Update equity — standard daily P&L
        if position > 0:
            equity *= (1 + position * ret + (1 - position) * cash_ret)
        else:
            equity *= (1 + cash_ret)

        exit_reasons = []

        if position > 0:
            dd = (price - entry_price) / entry_price
            if dd < -stop_loss:
                exit_reasons.append("ABSOLUTE_STOP")

        # Dynamic volatility rebalancing with transaction cost
        if position > 0 and position_mode == "vol_dynamic":
            new_pos = calc_position(vol, position_mode, target_vol, max_leverage)
            size_change = abs(new_pos - position)

            if size_change > 0.1:
                # Cost applies only to the fraction of capital being traded
                # e.g. if position moves from 0.6 to 0.8, we trade 0.2 * capital
                # Cost = size_change * COST_PER_UNIT
                REBAL_COST = 0.0005  # 5 bps, same as exit cost
                equity *= (1 - size_change * REBAL_COST)

                position = new_pos

                # Track rebalancing for diagnostics
                rebal_count  += 1
                rebal_cost_total += size_change * REBAL_COST

        if position > 0:
            M = max(M, price) if M is not None else price
            if price < (1 - X) * M:
                if "ABSOLUTE_STOP" not in exit_reasons:
                    exit_reasons.append("TRAIL_STOP")
            elif not filter_on:
                exit_reasons.append("FILTER_EXIT")

        exit_reason = " + ".join(exit_reasons) if exit_reasons else None

        if position > 0 and exit_reason:
            COST = 0.0020  # 20 bps
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

            if (price > (1 + Y) * m) and filter_on:
                entry_reason = "BREAKOUT & FILTER"
                position     = calc_position(vol, position_mode, target_vol, max_leverage)
                entry_price  = price
                entry_date   = i
                entry_pos    = position
                M            = price
                entry_carried   = False   # new entry in this window, not carried

        equity_curve.append(equity)
    
    
    if position_mode == "vol_dynamic" and rebal_count > 0:
        logging.debug(
            "vol_dynamic rebalancing: %d adjustments, "
            "total cost drag %.4f%% (%.1f bps)",
            rebal_count,
            rebal_cost_total * 100,
            rebal_cost_total * 10000
            )
    # -----------------------
    # End of window — build end_state instead of force-exiting
    # -----------------------

    # If position is still open, do NOT force exit.
    # Instead, package the current state so walk_forward can
    # pass it into the next window's initial_state.
    # We still record a provisional SAMPLE_END trade for the
    # equity curve to be complete, but flag it so callers can
    # exclude it from trade statistics.

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
        
        # Mark as CARRY — not a real exit, just a window boundary
        trades.append({
            "EntryDate":    entry_date,
            "ExitDate":     last_date,
            "EntryPrice":   entry_price,
            "Position":     entry_pos,
            "ExitPrice":    last_price,
            "Return":       trade_ret,
            "Days":         days,
            "Entry Reason": entry_reason,
            "Exit Reason":  "CARRY",          # distinguish from SAMPLE_END
            "CrossWindow":  entry_carried   # preserve flag for boundary record
        })

        end_state = {
            "position":     position,
            "entry_price":  entry_price,
            "entry_date":   entry_date,
            "entry_reason": entry_reason,
            "entry_pos":    entry_pos,
            "M":            M,
            "m":            m,
            "rebal_count":       rebal_count,       # cumulative across windows
            "rebal_cost_total":  rebal_cost_total   # cumulative across windows            
            }

    # Assign equity to full df (warmup + test rows) while lengths match
    df["equity"] = equity_curve   # safe here — same length

    # NOW strip warmup rows
    df = df[~df["_warmup"]].copy()
    df.drop(columns=["_warmup"], inplace=True)
    
    if "fund_filter" in df.columns:
        df.drop(columns=["fund_filter"], inplace=True)
        
    # After strip, all remaining rows are test rows — just check directly
    if df.isnull().any().any():
        logging.warning("NaN values remain in test rows after dropna — check cash merge")
    
    # Renormalise equity so each window starts at 1.0
    # (warmup period may have drifted equity away from 1.0
    # if carry_state brought in an open position)
    
    first_val = df["equity"].iloc[0]
    if initial_state is not None and abs(first_val - 1.0) > 0.001:
        logging.debug(
            "Warmup P&L on carried position: %.2f%% — excluded from OOS equity",
            (first_val - 1.0) * 100
            )
    if first_val != 0:
            df["equity"] = df["equity"] / first_val
    
            
        
    metrics = compute_metrics(df["equity"],risk_free_rate=rf_rate)
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(trades)

    return df, metrics, trades_df, end_state

# -------------------------------------------------------
# walk_forward — threads state across windows
# -------------------------------------------------------



def evaluate_params(
    filter_mode, fund_idx, fund_params, X, Y, fast, slow, tv, stop_loss,
    train, cash_train, vol_window, selected_mode, funds_df, train_start, train_end,
    objective="calmar",   # <-- new parameter, default preserves v0.1 behaviour,
    mom_lookback=252       
):
    """
    Evaluate a single parameter combination on the training window.

    This function is the unit of work dispatched to each parallel worker
    by joblib.Parallel during the walk-forward grid search. It is designed
    to be fully self-contained — it carries all required inputs as arguments
    and has no dependency on shared mutable state, making it safe to run
    concurrently across multiple processes or threads.

    ---------------------------------------------------------------
    FILTER MODES
    ---------------------------------------------------------------
    The function handles all three filter modes transparently:

      'ma'   — MA cross filter. fast/slow MA parameters are used.
               fund_signal is None.

      'mom'  — Momentum filter. fast/slow are dummy values (ignored
               inside run_strategy_with_trades when use_momentum=True).
               fund_signal is None.

      'fund' — Fund breadth filter. The fund signal is computed from
               scratch over the training window using fund_params.
               fast/slow are dummy values, ignored by the strategy
               engine since fund_signal takes precedence over both
               MA cross and momentum when provided.

    ---------------------------------------------------------------
    FUND SIGNAL COMPUTATION
    ---------------------------------------------------------------
    When filter_mode='fund', compute_fund_breadth_signal is called
    with the training-period slice of funds_df. The signal is
    initialised to OUT (state=0) at the start of the training window.
    No warmup is applied here — training runs are always clean starts
    without carry state, so the cold-start bias at the beginning of
    the training window is acceptable. Warmup is applied only for
    OOS runs in walk_forward, where signal continuity matters.

    ---------------------------------------------------------------
    RETURN VALUE
    ---------------------------------------------------------------
    Returns a (key, calmar) tuple if the evaluation succeeds, or None
    if run_strategy_with_trades returns no metrics or if MaxDD is zero
    (which would cause division by zero in the CalMAR calculation).

    None results are filtered out by the caller before building
    param_scores — they represent parameter combinations that produced
    degenerate results (e.g. no trades, flat equity) on the training
    window and should be excluded from the stability-penalised selection.

    The key tuple structure is:
      (filter_mode, fund_idx, X, Y, fast, slow, tv, stop_loss)

    fund_idx is an integer index into fund_params_grid (or None for
    non-fund modes) rather than the params dict itself, ensuring the
    key is hashable for use in param_scores.

    ---------------------------------------------------------------
    PARALLELISATION NOTES
    ---------------------------------------------------------------
    Called via joblib.Parallel with backend='loky' (multiprocessing)
    or 'threading'. All arguments must be serialisable for loky.
    DataFrames (train, cash_train, funds_df) are serialised via pickle
    for each worker — for very large DataFrames this overhead can
    reduce the parallelisation benefit. If serialisation becomes a
    bottleneck, switch to backend='threading' which shares memory
    directly but is subject to Python's GIL (largely released during
    numpy/pandas operations).

    ---------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------
    filter_mode  : str          — 'ma', 'mom', or 'fund'
    fund_idx     : int|None     — index into fund_params_grid, or None
    fund_params  : dict|None    — parameters for compute_fund_breadth_signal,
                                  or None for non-fund modes
    X            : float        — trailing stop threshold from peak
    Y            : float        — breakout threshold from trough
    fast         : int          — fast MA window (bars)
    slow         : int          — slow MA window (bars)
    tv           : float        — target volatility for position sizing
    stop_loss    : float        — absolute stop from entry price
    train        : DataFrame    — training window price series
    cash_train   : DataFrame    — training window cash/money market series
    vol_window   : int          — rolling volatility window (bars)
    selected_mode: str          — position mode ('full', 'vol_entry',
                                  'vol_dynamic')
    funds_df     : DataFrame|None — full fund NAV panel; sliced to
                                  training window inside this function
    train_start  : Timestamp    — start date of training window
    train_end    : Timestamp    — end date of training window
    objective    :  str         - sets objective function
 
    ---------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------
    tuple (key, obj_value) if evaluation succeeds, None otherwise.
    key     : tuple — hashable parameter identifier for param_scores
    calmar  : float — objective fctn val on the training window
    """

    use_mom = (filter_mode == "mom")

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
        use_momentum=use_mom,
        mom_lookback=mom_lookback,    
        fund_signal=train_fund_signal
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
        # Equal-weight blend — penalises both drawdown and vol-adjusted return
        if calmar is None:
            return None
        obj_value = 0.5 * calmar + 0.5 * sharpe

    elif objective == "calmar_sortino":
        # Blend preferring downside-only vol penalty over symmetric Sharpe
        if calmar is None:
            return None
        obj_value = 0.5 * calmar + 0.5 * sortino

    else:
        raise ValueError(f"Unknown objective: {objective!r}")

    key = (filter_mode, fund_idx, X, Y, fast, slow, tv, stop_loss, mom_lookback)
    return key, obj_value




def walk_forward(
    df,
    cash_df,
    train_years=8,
    test_years=2,
    vol_window  = 20,
    funds_df=None,          # NEW: panel of fund NAV series
    fund_params_grid=None,    # NEW, # NEW: dict of params for compute_fund_breadth_signal
    selected_mode="full",
    filter_modes_override=None,    # NEW
    #DEFAULT VALUES FOR GRIDS
    X_grid = [0.08, 0.10, 0.12, 0.15, 0.20],
    Y_grid = [0.02, 0.03, 0.05, 0.07, 0.10],
    fast_grid   = [50, 75, 100],
    slow_grid = [150, 200, 250 ],
    tv_grid = [0.08, 0.10, 0.12, 0.15, 0.20],
    sl_grid     = [0.05, 0.08, 0.10, 0.15],
    mom_lookback_grid = [126, 252],    
    objective = "calmar"
    
):

    """
    Run a rolling walk-forward optimisation and return a stitched
    out-of-sample equity curve.

    ---------------------------------------------------------------
    METHODOLOGY
    ---------------------------------------------------------------
    Walk-forward optimisation simulates live trading as faithfully as
    possible by ensuring that no future information influences parameter
    selection. Each iteration:

      1. TRAIN  : Run a grid search over all parameter combinations on
                  the training window. Score each combination by objective function,
                  then apply a stability penalty that blends the raw
                  score with the mean score of neighbouring X/Y values.
                  This prefers parameter sets on broad performance
                  plateaus over narrow spikes, reducing overfitting.

      2. TEST   : Apply the best parameters to the OOS test window.
                  Parameters are fixed before the test window opens —
                  no peeking at test-period results.

      3. STITCH : Chain-link the OOS equity slice to the previous
                  slice so the combined curve compounds continuously.
                  Each slice is renormalised to start at 1.0 and then
                  scaled by the endpoint of the previous slice.

      4. ADVANCE: Move the training start forward by test_years and
                  repeat.

    The final reported result is the stitched OOS equity curve. This
    is the only honest performance measure — per-window metrics in
    wf_results are useful for diagnosing parameter stability but
    understate drawdowns that span window boundaries.

    ---------------------------------------------------------------
    PARAMETER GRIDS
    ---------------------------------------------------------------
    All search grids are defined once at the top of the function and
    passed explicitly to neighbour_mean. This guarantees the stability
    penalty always uses the same grid as the search loop — changing
    one automatically changes the other.

    When selected_mode='full', target_vol has no effect on position
    sizing and is excluded from the search grid and best_params to
    avoid misleading output.

    ---------------------------------------------------------------
    CARRY STATE
    ---------------------------------------------------------------
    Open positions are carried across window boundaries rather than
    being force-closed. At the end of each OOS window, if a position
    is open, its state (entry price, entry date, running max/min,
    position size) is packaged into carry_state and passed into the
    next window's initial_state. This avoids artificial stop-outs at
    arbitrary calendar dates and produces a more realistic simulation
    of how the strategy would actually be managed.

    CARRY records in the trade log mark window-boundary snapshots of
    open positions. They are excluded from trade statistics but retained
    for transparency.

    ---------------------------------------------------------------
    LAST WINDOW EXTENSION
    ---------------------------------------------------------------
    If there is insufficient data for a further full test window after
    the current one (fewer than test_years * 252 rows remaining), the
    current test window is extended to the end of available data. This
    prevents the final months of data from being silently ignored.

    ---------------------------------------------------------------
    WARMUP
    ---------------------------------------------------------------
    Each OOS run receives the last WARMUP_BARS rows of its training
    window as warmup_df. WARMUP_BARS = slow + vol_window + 10 (padding),
    using the best_params slow value for that window. This ensures
    indicators are fully initialised at the start of each test window,
    eliminating gaps in the stitched equity curve.
    
    ---------------------------------------------------------------
    PARALLELISATION
    ---------------------------------------------------------------
    Grid search combinations are evaluated in parallel using
    joblib.Parallel. The number of jobs is computed at call time
    from os.cpu_count(): all cores are used on 2-core machines
    (e.g. GitHub Actions), and cpu_count-1 cores are used on
    machines with more than 2 cores, leaving one core free for
    system processes.

    The unit of parallel work is evaluate_params() — one call per
    parameter combination. Each call is fully self-contained with
    no shared mutable state, making it safe for concurrent execution.
    The loky backend (multiprocessing) is used by default. If loky
    fails (e.g. in Spyder/IPython environments on Windows), execution
    falls back to sequential evaluation automatically.

    Walk-forward windows are NOT parallelised — each window depends
    on the carry_state from the previous window and must run
    sequentially.
    
    Parallelisation uses a three-tier fallback:
        1. loky (multiprocessing) — default, best isolation
        2. threading — fallback if loky fails (e.g. Spyder/IPython)
        3. sequential — final fallback, always succeeds
        The successful backend is logged for each window.

    ---------------------------------------------------------------
    FUND BREADTH FILTER
    ---------------------------------------------------------------
    If funds_df and fund_params_grid are provided, a third filter
    mode 'fund' is added to the grid search alongside 'ma' and 'mom'.
    For fund filter combinations, compute_fund_breadth_signal is
    called inside evaluate_params with the training-period slice of
    funds_df. For the OOS run, the signal is computed over the
    combined warmup+test period and warmup rows are stripped before
    passing to run_strategy_with_trades, ensuring the signal state
    is correctly initialised at the test window start.

    fund_params_grid entries are referenced by integer index in the
    parameter key tuple rather than by dict value, ensuring keys
    remain hashable for use in param_scores.
    ---------------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------------
    df            : DataFrame — full price series covering all windows
    cash_df       : DataFrame — money market fund price series
    train_years   : int       — length of each training window in years
    test_years    : int       — length of each test window in years
    vol_window    : int       — rolling volatility window passed to strategy
    selected_mode : str       — position mode passed to strategy engine
    
    funds_df         : DataFrame|None — panel of fund NAV series, one column
                                    per fund, date-indexed. None disables
                                    the fund breadth filter mode.
    fund_params_grid : list[dict]|None — list of parameter dicts passed to
                                    compute_fund_breadth_signal during
                                    grid search. None disables fund mode.
    ---------------------------------------------------------------
    RETURNS
    ---------------------------------------------------------------
    oos_equity    : pd.Series   — stitched chain-linked OOS equity curve
    results_df    : pd.DataFrame — per-window params and OOS metrics
    oos_trades_df : pd.DataFrame — all OOS trades concatenated with
                                   WF_Window and CrossWindow columns
    """


    
    data_end          = df.index.max()   # capture once
    logging.info("walk_forward received data from %s to %s (%d rows)",
             df.index.min(), data_end, len(df))
    
    logging.info("Objective function: %s", objective)

    
    _cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    logging.info(
        "Parallel grid search: %d logical cores detected, using %d jobs.",
        _cpu_count, N_JOBS
    )
    
    oos_equity_slices = []
    results           = []
    all_oos_trades    = []

    start       = df.index.min()
    carry_state = None          # state carried from previous OOS window
    
    
    # Override for diagnostic runs
    if filter_modes_override is not None:
        logging.info("filter_modes overridden to: %s", filter_modes_override)

    while True:

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

        # --- DIAGNOSTIC: log every iteration ---
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
        

        # -------------------------------------------------------
        # Grid search — always runs WITHOUT carry state
        # -------------------------------------------------------
        param_scores = {}

        # Determine which filter modes to search
        # "ma"   — MA cross filter (fast/slow grid)
        # "mom"  — momentum filter (fixed MA params, ignored)
        # "fund" — fund breadth filter (fund_params_grid)
        filter_modes = ["ma", "mom"]
        if funds_df is not None:
            filter_modes.append("fund")
        
        
        # Override for diagnostic runs
        if filter_modes_override is not None:
            filter_modes = filter_modes_override
            
            
        # Build list of all parameter combinations to evaluate
        param_combinations = []

        for filter_mode in filter_modes:
            fast_iter = fast_grid if filter_mode == "ma" else [50]
            slow_iter = slow_grid if filter_mode == "ma" else [200]
            mom_lb_iter   = mom_lookback_grid if filter_mode == "mom" else [252]  # ADD
            fund_iter = (list(enumerate(fund_params_grid))
                         if filter_mode == "fund" else [(None, None)])

            for fund_idx, fund_params in fund_iter:
                for X in X_grid:
                    for Y in Y_grid:
                        for fast in fast_iter:
                            for slow in slow_iter:
                                if filter_mode == "ma" and slow - fast < 75:
                                    continue
                                for tv in tv_grid if selected_mode != "full" else [0.10]:
                                    for stop_loss in sl_grid:
                                        if stop_loss >= X:
                                            continue
                                        for mom_lookback in mom_lb_iter:    # ADD
                                            param_combinations.append((
                                                filter_mode, fund_idx, fund_params,
                                                X, Y, fast, slow, tv, stop_loss,mom_lookback 
                                                ))

        # Evaluate all combinations in parallel
        
        for backend, n_jobs, label in [
                ("loky",      N_JOBS, "multiprocessing"),
                ("threading", N_JOBS, "threading"),
                (None,        1,      "sequential"),
                ]:
            try:
                if backend is None:
                    # Sequential fallback — plain list comprehension
                    results_list = [
                        evaluate_params(
                            filter_mode, fund_idx, fund_params,
                            X, Y, fast, slow, tv, stop_loss,
                            train, cash_train, vol_window, selected_mode,
                            funds_df, train_start, train_end, objective=objective, mom_lookback=mom_lookback
                            )
                        for (filter_mode, fund_idx, fund_params,
                             X, Y, fast, slow, tv, stop_loss, mom_lookback)
                        in param_combinations
                        ]
                else:
                    results_list = Parallel(n_jobs=n_jobs, backend=backend)(
                        delayed(evaluate_params)(
                            filter_mode, fund_idx, fund_params,
                            X, Y, fast, slow, tv, stop_loss,
                            train, cash_train, vol_window, selected_mode,
                            funds_df, train_start, train_end, objective=objective, mom_lookback=mom_lookback
                            )
                        for (filter_mode, fund_idx, fund_params,
                             X, Y, fast, slow, tv, stop_loss, mom_lookback)
                        in param_combinations
                        )

                logging.info(
                    "Grid search completed using %s backend (%d jobs).",
                    label, n_jobs
                    )
                break   # success — exit the fallback loop

            except Exception as e:
                logging.warning(
                    "Grid search backend '%s' failed: %s — trying next option.",
                    label, e
                    )
                results_list = None   # ensure next iteration starts clean

        if results_list is None:
            logging.error(
                "All grid search backends failed. Skipping window."
                )
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue
            
        # Collect results — filter out None (failed evaluations)
        param_scores = {
            key: score
            for result in results_list
            if result is not None
            for key, score in [result]
        }

        if not param_scores:   # <- correctly outside all for loops, inside while
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        # -------------------------------------------------------
        # Stability-penalised selection
        # -------------------------------------------------------
        # neighbour_mean only varies X and Y — filter_mode, fund_params,
        # fast, slow, tv, stop_loss are all held fixed when finding neighbours
        
        best_score  = -np.inf
        best_params = None
        best_raw_score  = -np.inf   # raw objective — logged for transparency
        
        for key, raw_score in param_scores.items():
            same_mode_scores = {
                k: v for k, v in param_scores.items()
                if k[0] == key[0]   # same filter_mode
                }
            stability = neighbour_mean(key, same_mode_scores, X_grid, Y_grid)
            combined  = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score  = combined
                best_raw_score = raw_score
                fund_idx = key[1]   # read from key, not loop variable

                best_params = {
                    "filter_mode":  key[0],
                    "fund_idx":     fund_idx,
                    "fund_params":  (fund_params_grid[fund_idx]
                                     if fund_idx is not None and fund_params_grid is not None
                                     else None),
                    "X":            key[2],
                    "Y":            key[3],
                    "fast":         key[4],
                    "slow":         key[5],
                    "stop_loss":    key[7],
                    "mom_lookback": key[8]
                }
                if selected_mode != "full":
                    best_params["target_vol"] = key[6]
            
        if best_params is None:
            break
        else:
            logging.info(
                "Window %s: best raw_%s=%.4f | penalised_%s=%.4f | filter=%s | "
                "X=%.2f Y=%.2f fast=%d slow=%d sl=%.2f tv=%s mom_lookback=%s",
                train_start.date(),
                objective, best_raw_score,
                objective, best_score,
                best_params["filter_mode"],
                best_params["X"], best_params["Y"],
                best_params["fast"], best_params["slow"],
                best_params["stop_loss"],
                best_params.get("target_vol", "N/A"),
                best_params["mom_lookback"]                
                )   
        
        
        # -------------------------------------------------------
        # OOS run — pass carry_state from previous window
        # -------------------------------------------------------
        
        WARMUP_BARS = best_params["slow"] + vol_window + 10 # 10 is padding
        warmup      = train.iloc[-WARMUP_BARS:]
        
        
        
        # cash for OOS run must cover both warmup and test periods
        
        # Find the actual start date of the warmup slice directly
        warmup_start = warmup.index.min()
        cash_warmup_and_test = cash_df.loc[
            (cash_df.index >= warmup_start) &
            (cash_df.index < test_end)
            ]
        
        
        # In the OOS run section, after best_params selection:

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
            # Strip warmup rows — state at test start inherits warmup computation
            oos_fund_signal = full_fund_signal.loc[
                full_fund_signal.index >= train_end
            ]

        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            test,
            price_col="Zamkniecie",
            cash_df=cash_warmup_and_test,
            position_mode=selected_mode,
            vol_window=vol_window,
            initial_state=carry_state,
            warmup_df=warmup,
            fund_signal=oos_fund_signal,
            use_momentum=(best_params["filter_mode"] == "mom"),
            mom_lookback=best_params["mom_lookback"],    # ADD
            **{k: v for k, v in best_params.items()
               if k not in ("filter_mode", "fund_params", "fund_idx",  "mom_lookback")}  # add fund_idx
            )

        if test_metrics is None or bt_oos is None:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        # Thread state forward to next window
        carry_state = end_state   # None if flat at window end

        # Chain-link equity
        equity_slice = bt_oos["equity"].copy()
        if oos_equity_slices:
            prev_end     = oos_equity_slices[-1].iloc[-1]
            equity_slice = equity_slice * prev_end

        oos_equity_slices.append(equity_slice)

        # Collect trades — exclude CARRY boundary records from stats
        # but keep them so the trade log is complete
        if not oos_trades.empty:
            oos_trades = oos_trades.copy()
            oos_trades["WF_Window"] = train_start
            all_oos_trades.append(oos_trades)

        results.append({
            "TrainStart":  train_start,
            "TrainEnd":    train_end,
            "TestStart":   train_end,
            "TestEnd":     test_end,
            "filter_mode": best_params["filter_mode"],
            "fund_idx":    best_params["fund_idx"],
            "fund_params": str(best_params["fund_params"]),
            **{k: v for k, v in best_params.items()
               if k not in ("filter_mode", "fund_params", "fund_idx",
                    "use_momentum", "target_vol")},
                    "target_vol":  best_params.get("target_vol", "N/A"),
            "mom_lookback": best_params.get("mom_lookback", 252),    # ADD
            **test_metrics
            })
        
        # Advance start before checking if this was the last window
        start += pd.DateOffset(years=test_years)
        
        # --- stop after extending the last window ---
        if not next_window_has_enough:
            
            # --- DIAGNOSTIC ---
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
            # --- END DIAGNOSTIC ---
            
            logging.info("Final OOS window extended to %s (end of data).", data_end)
            break

    # Final forced exit for any position still open at end of last window
    # This only fires once, at the very end of the entire OOS period.
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
    
    """
    Compute summary statistics from a completed trade log.

    Boundary records (CARRY, SAMPLE_END) are excluded before computing
    statistics — these are window-management artefacts, not real strategy
    decisions. For cross-window trades, the closing record in the later
    window already contains the correct full return from the original
    entry price, so no deduplication is needed once CARRY records are
    excluded.

    Parameters
    ----------
    trades         : DataFrame         — trade log from walk_forward
    boundary_exits : set               — exit reason codes to exclude

    Returns
    -------
    dict with keys: Trades, WinRate, AvgWin, AvgLoss, ProfitFactor,
                    AvgDays, CrossWindow
    Returns None if no qualifying trades remain after filtering.
    """

    if trades.empty:
        return None

    # Drop window-boundary snapshots entirely
    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()

    if trades.empty:
        return None

    # For cross-window trades, the closing record in window 2 already
    # contains the correct full return from original entry price.
    # No deduplication needed as long as CARRY records are excluded.
    # But log how many cross-window trades there were for transparency.
    
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

    """
    Write a formatted backtest report to the log.

    Sections:
      1. Per-window parameter table (from wf_results) or single
         best_params if walk-forward was not used.
      2. Aggregate performance metrics (CAGR, Vol, Sharpe, MaxDD, CalMAR, Sortino).
      3. Trade statistics (win rate, avg win/loss, profit factor, avg days).
      4. Full trade log with returns formatted as percentages.
      5. Open position summary if the last trade is a CARRY record,
         showing entry date, entry price, current mark, and unrealised
         return.

    Parameters
    ----------
    metrics       : dict        — output of compute_metrics
    trades        : DataFrame   — full trade log including CARRY records
    trade_stats   : dict|None   — output of analyze_trades
    best_params   : dict|None   — single parameter set (non-WF use)
    wf_results    : DataFrame|None — per-window results from walk_forward
    position_mode : str         — mode label for report header
    filter_modes_override    : str - which trend filter was forced if any
    """

    logging.info("="*80)
    logging.info(f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}")
    if filter_modes_override is not None:
        logging.info(f"Filter mode was forced to:    {filter_modes_override}")
    else:
        logging.info("Filter mode selection set to automatic")
    logging.info("="*80)

    # --- show per-window params if available, else single best_params ---
    if wf_results is not None and not wf_results.empty:
        cols = ["TrainStart", "TestStart", "filter_mode",
                "X", "Y", "fast", "slow", "target_vol", "stop_loss"]
        # Only show fund_params if fund filter was ever selected
        if "fund_params" in wf_results.columns and \
           wf_results["filter_mode"].eq("fund").any():
            cols.insert(3, "fund_params")
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

    # Trade Statistics
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

    # Trade Log
    # Initialise carry_trades here so it's always defined for the
    # open position block below, regardless of whether trade_stats exists
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

    # Open position summary — only if there is a CARRY record
    if not carry_trades.empty:
        last_carry = carry_trades.iloc[-1]
        logging.info(
            "Open position at report date: entry %s at %.2f, "
            "current value %.2f, unrealised return %.1f%%",
            last_carry["EntryDate"],
            last_carry["EntryPrice"],
            last_carry["ExitPrice"],
            last_carry["Return"] * 100
        )
    logging.info("="*80)

