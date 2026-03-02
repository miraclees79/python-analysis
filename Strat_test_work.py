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
# Performance Metrics
# ============================

def compute_metrics(equity, freq=252):

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
    sharpe = cagr / vol if vol > 0 else 0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
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
    key    : tuple        — (X, Y, fast, slow, tv, stop_loss)
    scores : dict         — maps parameter tuples to CalMAR scores
    X_grid : list[float]  — ordered list of X values used in the search
    Y_grid : list[float]  — ordered list of Y values used in the search

    Returns
    -------
    float — mean CalMAR of all neighbours including the key itself,
            or the key's own score if no neighbours exist in scores.
    """

    X, Y, fast, slow, tv, sl = key

    
    # Use nearest match to avoid floating point equality failures
    xi = min(range(len(X_grid)), key=lambda i: abs(X_grid[i] - X))
    yi = min(range(len(Y_grid)), key=lambda i: abs(Y_grid[i] - Y))

    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nxi, nyi = xi + dx, yi + dy
            if 0 <= nxi < len(X_grid) and 0 <= nyi < len(Y_grid):
                nkey = (X_grid[nxi], Y_grid[nyi], fast, slow, tv, sl)
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
    cash_df=None,
    safe_rate=0.0,
    initial_state=None,       # NEW: accept carry-over state from prior window,
    warmup_df=None    # NEW: pre-window data for indicator warm-up
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
    use_momentum  : bool       — whether to include momentum in the filter
    cash_df       : DataFrame  — money market fund price series (optional)
    safe_rate     : float      — fallback annual cash rate if cash_df is None
    initial_state : dict|None  — carry-over position state from prior window
    warmup_df     : DataFrame|None — price rows to prepend for indicator warm-up

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
    if len(df) < max(vol_window, slow, fast, 252) + 5:
        return None, None, pd.DataFrame(), None # extra None = end_state

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

  

    df["ret"] = df["price"].pct_change()
    vol = df["ret"].rolling(vol_window).std() * np.sqrt(252)
    df["vol"] = vol.shift(1)
    df["ma_fast"] = df["price"].rolling(fast).mean().shift(1)
    df["ma_slow"] = df["price"].rolling(slow).mean().shift(1)
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    if use_momentum:
        df["MOM"] = compute_momentum(df["price"]).shift(1)
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
    else:
        position    = 0.0
        entry_price = None
        entry_date  = None
        entry_reason= None
        entry_pos   = None
        M           = None
        m           = None
        entry_carried = False 
    

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
        # In run_strategy_with_trades main loop:
    
        filter_on = (trend == 1) and (mom > 0)
        
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
            if abs(new_pos - position) > 0.1:
                position = new_pos

        if position > 0:
            M = max(M, price) if M is not None else price
            if price < (1 - X) * M:
                if "ABSOLUTE_STOP" not in exit_reasons:
                    exit_reasons.append("TRAIL_STOP")
            elif not filter_on:
                exit_reasons.append("FILTER_EXIT")

        exit_reason = " + ".join(exit_reasons) if exit_reasons else None

        if position > 0 and exit_reason:
            COST = 0.0005
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
        
        if entry_date < test_start:  # recheck if still needed?
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
            "Exit Reason":  "CARRY"          # distinguish from SAMPLE_END
        })

        end_state = {
            "position":     position,
            "entry_price":  entry_price,
            "entry_date":   entry_date,
            "entry_reason": entry_reason,
            "entry_pos":    entry_pos,
            "M":            M,
            "m":            m            
            }

    # Assign equity to full df (warmup + test rows) while lengths match
    df["equity"] = equity_curve   # safe here — same length

    # NOW strip warmup rows
    df = df[~df["_warmup"]].copy()
    df.drop(columns=["_warmup"], inplace=True)

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
            
    metrics = compute_metrics(df["equity"])
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(trades)

    return df, metrics, trades_df, end_state

# -------------------------------------------------------
# walk_forward — threads state across windows
# -------------------------------------------------------

def walk_forward(
    df,
    cash_df,
    train_years=8,
    test_years=2,
    vol_window  = 20,   
    selected_mode="full"
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
                  the training window. Score each combination by CalMAR,
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
    PARAMETERS
    ---------------------------------------------------------------
    df            : DataFrame — full price series covering all windows
    cash_df       : DataFrame — money market fund price series
    train_years   : int       — length of each training window in years
    test_years    : int       — length of each test window in years
    vol_window    : int       — rolling volatility window passed to strategy
    selected_mode : str       — position mode passed to strategy engine

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
    
    # Define search grids once — passed to both the loop and neighbour_mean
    # so they are guaranteed to stay in sync
    X_grid      = [0.08, 0.10, 0.15, 0.20]
    Y_grid      = [0.02, 0.05, 0.10]
    fast_slow   = [(50, 200), (100, 300)]
    tv_grid     = [0.08, 0.10, 0.12]
    sl_grid     = [0.05, 0.08, 0.10, 0.15]
    
    oos_equity_slices = []
    results           = []
    all_oos_trades    = []

    start       = df.index.min()
    carry_state = None          # state carried from previous OOS window
    


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
        # Training is always a clean run; carry only applies OOS
        # -------------------------------------------------------

        param_scores = {}
        
        
        
        
        for X in X_grid:
            for Y in Y_grid:
                for fast, slow in fast_slow:
                    for tv in tv_grid if selected_mode!="full" else [0.10]:
                        for stop_loss in sl_grid:

                            bt, metrics, trades, _ = run_strategy_with_trades(
                                train,
                                cash_df=cash_train,
                                price_col="Zamkniecie",
                                X=X, Y=Y,
                                stop_loss=stop_loss,
                                fast=fast, slow=slow,
                                target_vol=tv,
                                vol_window=vol_window,
                                position_mode=selected_mode
                                # no initial_state for training runs
                            )

                            if metrics is None:
                                continue

                            max_dd = metrics.get("MaxDD", 0)
                            if max_dd == 0:
                                continue

                            calmar = metrics["CAGR"] / abs(max_dd)
                            key = (X, Y, fast, slow, tv, stop_loss)
                            param_scores[key] = calmar

        if not param_scores:
            start += pd.DateOffset(years=test_years)
            carry_state = None   # reset carry if window is skipped
            continue

        

        # Stability-penalised selection
        best_score  = -np.inf
        best_params = None

        for key, raw_score in param_scores.items():
            stability = neighbour_mean(key, param_scores, X_grid, Y_grid)
            combined  = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score  = combined
                best_params = {
                    "X":          key[0],
                    "Y":          key[1],
                    "fast":       key[2],
                    "slow":       key[3],
                    "stop_loss":  key[5]
                }
                # Only include target_vol if it's actually used
                if selected_mode != "full":
                    best_params["target_vol"] = key[4]
                    
        if best_params is None:
            break

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
        
        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            test,
            price_col="Zamkniecie",
            cash_df=cash_warmup_and_test,
            position_mode=selected_mode,
            vol_window=vol_window,
            initial_state=carry_state, # carry open position in
            warmup_df=warmup, 
            **best_params
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
            "TrainStart": train_start,
            "TrainEnd":   train_end,
            "TestStart":  train_end,
            "TestEnd":    test_end,
            **best_params,
            "target_vol": key[4] if selected_mode != "full" else "N/A",
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
def print_backtest_report(metrics, trades, trade_stats, best_params=None, wf_results=None, position_mode=None):

        """
    Write a formatted backtest report to the log.

    Sections:
      1. Per-window parameter table (from wf_results) or single
         best_params if walk-forward was not used.
      2. Aggregate performance metrics (CAGR, Vol, Sharpe, MaxDD, CalMAR).
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
    """

    logging.info("="*80)
    logging.info(f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}")
    logging.info("="*80)

    # --- show per-window params if available, else single best_params ---
    if wf_results is not None and not wf_results.empty:
        logging.info("PER-WINDOW PARAMETERS:")
        logging.info("\n%s", wf_results[["TrainStart","TestStart","X","Y","fast","slow","target_vol","stop_loss"]].to_string(index=False))
    elif best_params:
        logging.info("PARAMETERS: %s", pformat(best_params))

    logging.info("-"*80)

    # Metrics
    logging.info("METRICS:")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f ",
        metrics["CAGR"]*100,
        metrics["Vol"]*100,
        metrics["Sharpe"],
        metrics["MaxDD"]*100,
        metrics["CalMAR"]
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



### Initials


# Setup logging
LOG_FILE = "StratRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Remove any existing handlers — critical in notebook environments
# where the kernel may have pre-attached handlers, and where re-running
# cells accumulates duplicate handlers
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logging.info("Logging initialised — file: %s", LOG_FILE)

# Get the temporary directory
tmp_dir = tempfile.gettempdir()
logging.info(f"Temporary directory: {tmp_dir}")



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



# Main function to process the data


logging.info("RUN START: %s", dt.datetime.now())


# Load data
download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr)
INDEX_W20 = load_csv(csv_filename_w20tr)

df = INDEX_W20

# Download money market / bond fund
download_csv(
    "https://stooq.pl/q/d/l/?s=2720.n&i=d",   
    csv_filename_cash    
)

CASH = load_csv(csv_filename_cash)

# Set POSITION MODE
chosen_mode="full"  
# options: "vol_entry", "vol_dynamic", "full"

VOL_WINDOW  = 20   # keep in one place, pass explicitly if needed

# ============================
# REPORTING
# ============================



# -------- Walk Forward --------

logging.info("=" * 80)
logging.info("WALK-FORWARD OPTIMIZATION")
logging.info("=" * 80)

# -------------------------------------------------------
# Walk-Forward
# -------------------------------------------------------

# Fix — initialise before the if/else block
wf_trades = pd.DataFrame()
wf_metrics = None
trade_stats = None


wf_equity, wf_results, wf_trades = walk_forward(df, cash_df=CASH, selected_mode=chosen_mode, vol_window=VOL_WINDOW)


BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}

# --- FIX 1: was checking wf.empty (old variable name, no longer exists) ---
if wf_equity.empty or wf_results.empty:
    logging.warning("Walk-forward produced no results.")
    wf_metrics = None
    trade_stats = None

else:
    logging.info("Walk-forward Results per Window:\n%s", wf_results.to_string(index=False))

    avg = wf_results.mean(numeric_only=True)
    logging.info("Walk-forward Per-Window Averages:\n%s", avg.to_string())

    # --- FIX 2: compute metrics on the stitched OOS equity curve ---
    # Previously metrics came from a cherry-picked full-sample re-run.
    # Now we report on the actual OOS track record.
    wf_metrics = compute_metrics(wf_equity)
    wf_metrics = {k: float(v) for k, v in wf_metrics.items()}
    logging.info("Stitched OOS Metrics:\n")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f ",
        wf_metrics["CAGR"]*100,
        wf_metrics["Vol"]*100,
        wf_metrics["Sharpe"],
        wf_metrics["MaxDD"]*100,
        wf_metrics["CalMAR"]
    )
    # --- FIX 3: trade stats from OOS trades, not full-sample trades ---
    # Exclude boundary records from trade statistics
    
    # Safe filter regardless of whether trades exist or have the column
    if not wf_trades.empty and "Exit Reason" in wf_trades.columns:
        wf_trades_closed = wf_trades[
            ~wf_trades["Exit Reason"].isin(BOUNDARY_EXITS)
            ].copy()
    else:
            wf_trades_closed = pd.DataFrame()

    trade_stats = analyze_trades(wf_trades_closed, boundary_exits=BOUNDARY_EXITS) if not wf_trades_closed.empty else None

# -------------------------------------------------------
# Report
# -------------------------------------------------------

# Fix — guard the call
if wf_metrics is not None:
    print_backtest_report(
        metrics=wf_metrics,
        trades=wf_trades,
        trade_stats=trade_stats,
        wf_results=wf_results,
        position_mode=chosen_mode
    )
else:
    logging.warning("Skipping report — no WF metrics available.")

# -------------------------------------------------------
# Plotting — switch to OOS equity and OOS trades
# -------------------------------------------------------

# --- FIX 5: was plotting bt_full/trades_full (full-sample, in-sample) ---
# Plot the stitched OOS equity curve — this is the honest track record.

if wf_equity is not None and not wf_equity.empty:

    wf_equity.index = pd.to_datetime(wf_equity.index)

    if not wf_trades.empty:
        wf_trades['EntryDate'] = pd.to_datetime(wf_trades['EntryDate'])
        wf_trades['ExitDate']  = pd.to_datetime(wf_trades['ExitDate'])

    plt.figure(figsize=(16, 8))
    plt.plot(wf_equity.index, wf_equity.values, label='OOS Equity Curve', color='blue', linewidth=2)

    # Shade OOS windows for orientation
    for _, row in wf_results.iterrows():
        plt.axvspan(row["TestStart"], row["TestEnd"], color='grey', alpha=0.07)

    # Annotate trades
    if not wf_trades.empty:
        # --- FIX 6: guard against trade dates outside equity index ---
        for _, trade in wf_trades.iterrows():
            color = 'green' if trade['Return'] > 0 else 'red'
            plt.axvspan(trade['EntryDate'], trade['ExitDate'], color=color, alpha=0.15)

            trade_eq = wf_equity.loc[
                (wf_equity.index >= trade['EntryDate']) &
                (wf_equity.index <= trade['ExitDate'])
            ]

            if trade_eq.empty:
                continue   # skip if dates don't align after stitching

            y_pos = trade_eq.max() * 1.02 if trade['Return'] > 0 else trade_eq.min() * 0.98

            plt.text(
                trade['ExitDate'], y_pos,
                f"{trade['Return']*100:.1f}%",
                color=color, fontsize=8, ha='left', va='bottom', rotation=45
            )

    plt.title("Walk-Forward OOS Equity Curve", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Equity", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

logging.info("RUN END")