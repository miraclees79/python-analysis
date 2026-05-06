import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from moj_system.core.fund_filter import compute_fund_breadth_signal

# ============================================================
# CANONICAL N_JOBS CALCULATION
# ============================================================


def get_n_jobs() -> int:
    """
    Return the recommended number of parallel jobs for this machine.

    Logic:
      - On Windows with > 3 cores: use (cpu_count - 1) to keep the UI
        responsive (loky/spawn overhead is higher on Windows).
      - All other platforms: use all logical cores.
      - Always returns at least 1.

    Replaces the following block that was duplicated verbatim in ~8 files::

        _cpu_count = os.cpu_count() or 1
        N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    Returns
    -------
    int — recommended n_jobs value for joblib.Parallel / walk_forward

    Example
    -------
    >>> from strategy_test_library import get_n_jobs
    >>> N_JOBS = get_n_jobs()
    """
    cpu_count = os.cpu_count() or 1
    if cpu_count > 3 and sys.platform == "win32":
        return max(1, cpu_count - 1)
    return cpu_count


# ============================================================
# ANNUAL PERFORMANCE UTILITIES  (moved from objective_review.py)
# ============================================================


def annual_cagr_by_year(portfolio_equity: pd.Series) -> dict[int, float]:
    """
    Compute the annualised return for each full calendar year in the equity curve.

    Years with fewer than 50 trading days of data (partial years at the start
    or end of the series) are excluded.

    Parameters
    ----------
    portfolio_equity : pd.Series — portfolio equity curve (DatetimeIndex,
                                   values normalised to an arbitrary base)

    Returns
    -------
    dict[int, float]  — {year: annual_cagr} for all complete years

    Example
    -------
    >>> annual = annual_cagr_by_year(portfolio_equity)
    >>> print(annual)
    {2020: 0.142, 2021: -0.031, 2022: 0.087, 2023: 0.211, 2024: 0.053}
    """
    annual = {}
    df = portfolio_equity.copy()
    df.index = pd.to_datetime(df.index)

    for year in df.index.year.unique():
        yr = df[df.index.year == year]
        if len(yr) < 50:
            continue  # skip partial years
        start_val = yr.iloc[0]
        end_val = yr.iloc[-1]
        days = (yr.index[-1] - yr.index[0]).days
        if days < 1 or start_val <= 0:
            continue
        cagr = (end_val / start_val) ** (365.25 / days) - 1
        annual[year] = cagr

    return annual


def count_year_wins(
    cand_annual: dict[int, float],
    incumb_annual: dict[int, float],
    years: list[int],
) -> int:
    """
    Count how many of the specified calendar years the challenger
    outperforms the incumbent.

    Used in the annual objective function review (objective_review.py) as
    one of the five switching gates.

    Parameters
    ----------
    cand_annual   : dict[int, float] — challenger annual CAGRs by year
    incumb_annual : dict[int, float] — incumbent annual CAGRs by year
    years         : list[int]        — calendar years to compare (e.g. last 5)

    Returns
    -------
    int — number of years in `years` where cand_annual[y] > incumb_annual[y].
          Years missing from either dict are skipped (not counted as a win or
          a loss).

    Example
    -------
    >>> wins = count_year_wins(cand_annual, incumb_annual, [2020, 2021, 2022, 2023, 2024])
    >>> print(f"{wins}/5 year wins")
    """
    wins = 0
    for y in years:
        c = cand_annual.get(y)
        i = incumb_annual.get(y)
        if c is not None and i is not None and c > i:
            wins += 1
    return wins


#


def load_csv(
    filename: str
) -> pd.DataFrame | None:
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
        df = pd.read_csv(
            filename, on_bad_lines="skip", delimiter=",", decimal=".", encoding="utf-8-sig",
        )
    except Exception as e:
        logging.error(f" Error reading CSV file: {e}")
        return None

    if df.empty or df.columns.size == 0:
        logging.error(" CSV file is empty or corrupted.")
        return None

    # Strip whitespace and inspect column names
    df.columns = df.columns.str.strip()
    logging.debug("Available columns after stripping:", df.columns)

    date_column = "Data"  # Expected date column

    # Double-check the column names for hidden characters
    if date_column not in df.columns:
        exact_matches = [col for col in df.columns if col.strip() == date_column]
        if exact_matches:
            date_column = exact_matches[0]  # If a match is found, update the column name
            logging.info(f" Using corrected column name: '{date_column}'")
        else:
            # If still not found, display and exit
            logging.error(
                f" Column '{date_column}' not found after processing. Available columns: {df.columns}",
            )
            return None

    # Check if the date column contains valid data
    if df[date_column].isnull().all():
        logging.error(f" Column '{date_column}' contains only NaN values.")
        return None

    # Convert to datetime, handling errors and dropping invalid dates
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
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
        logging.warning(
            f" The newest observation ({newest_date}) is older than 10 days. Data is discarded.",
        )
        return None

    # Check 2: Discard data before the most recent break longer than 30 days
    date_diffs = df.index.to_series().diff().dt.days  # Calculate gaps in the date series
    breaks = date_diffs[date_diffs > 30].index  # Identify breaks longer than 30 days

    if not breaks.empty:
        # Keep only the data from the newest observation to the most recent break
        last_valid_date = breaks[-1]
        df = df.loc[df.index > last_valid_date]  # Slice the DataFrame from the break to the end
        logging.info(
            f" Data contains a break longer than 30 days. Keeping data from {last_valid_date} onward.",
        )

    logging.info("SUCCESS! CSV file loaded successfully and processed.")

    return df


def prepare_cash_returns(
    cash_df: pd.DataFrame, 
    price_col: str = "Zamkniecie"
) -> pd.DataFrame:
    cash = cash_df.copy()
    cash["cash_price"] = cash[price_col]
    cash["cash_ret"] = cash["cash_price"].pct_change()
    cash = cash[["cash_ret"]].dropna()
    return cash


# ============================
# Indicators
# ============================


def compute_momentum(
    series: pd.Series,
    lookback: int = 252,
    skip: int = 21,
    blend: bool = False,
    blend_lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    blend_skip: int = 5,
) -> pd.Series:
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
# Performance Metrics
# ============================


def compute_metrics(
    equity: pd.Series, 
    risk_free_rate: float = 0.0, 
    freq: int = 252
) -> dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CalMAR": 0.0}
    ret = equity.pct_change().dropna()

    years = len(ret) / freq
    if years == 0: return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CalMAR": 0.0}
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0 else 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    daily_rets = equity.pct_change().dropna()
    downside = daily_rets[daily_rets < daily_rf] - daily_rf
    if len(downside) > 0:
        downside_vol = np.sqrt((downside**2).mean()) * np.sqrt(252)
    else:
        downside_vol = 0.0
    sortino = excess_return / downside_vol if downside_vol > 0 else 0.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(max_dd),
        "CalMAR": float(calmar),
    }


# ============================================================
# PARAMETER STABILITY
# ============================================================


def neighbour_mean(
    key: tuple, 
    scores: dict, 
    stop_grid: list[float], 
    Y_grid: list[float]
) -> float:
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
    yi = min(range(len(Y_grid)), key=lambda i: abs(Y_grid[i] - Y))

    neighbours = []
    for ds in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nsi, nyi = si + ds, yi + dy
            if 0 <= nsi < len(stop_grid) and 0 <= nyi < len(Y_grid):
                nkey = (
                    filter_mode,
                    fund_idx,
                    stop_grid[nsi],
                    Y_grid[nyi],
                    fast,
                    slow,
                    tv,
                    sl,
                    mom_lookback,
                )
                if nkey in scores:
                    neighbours.append(scores[nkey])

    return float(np.mean(neighbours)) if neighbours else float(scores[key])


def calc_position(
    vol: float, 
    position_mode: str, 
    target_vol: float, 
    max_leverage: float
) -> float:
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


def compute_buy_and_hold(
    df: pd.DataFrame, 
    price_col: str = "Zamkniecie", 
    start: pd.Timestamp | None = None, 
    end: pd.Timestamp | None = None
) -> tuple[pd.Series, dict[str, float]]:
    
    bh = df[price_col].copy()

    if start is not None:
        bh = bh.loc[bh.index >= start]
    if end is not None:
        bh = bh.loc[bh.index <= end]

    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(bh_equity)

    return bh_equity, {k: float(v) for k, v in bh_metrics.items()}


# ============================
# NUMBA CORE ENGINES
# ============================

@njit(cache=True, nogil=True)
def calc_position_numba(
    vol: float,
    position_mode_int: int,
    target_vol: float,
    max_leverage: float
) -> float:
    if position_mode_int == 0:
        return 1.0
    if not np.isnan(vol) and vol > 0.0:
        pos = target_vol / vol
        return min(pos, max_leverage)
    return 1.0

@njit(cache=True, nogil=True)
def run_numba_light(
    prices: np.ndarray, rets: np.ndarray, cash_rets: np.ndarray, trends: np.ndarray,
    moms: np.ndarray, vols: np.ndarray, atrs: np.ndarray, warmups: np.ndarray,
    gate_vals: np.ndarray, fund_vals: np.ndarray, dates: np.ndarray,
    filter_mode_int: int, position_mode_int: int, stop_loss: float,
    target_vol: float, max_leverage: float, use_atr_stop: bool,
    N_atr: float, X: float, Y: float,
    init_pos: float, init_entry_px: float, init_entry_dt: int, init_M: float, init_m: float
) -> tuple:
    n = len(prices)
    equity_curve = np.zeros(n)
    equity = 1.0
    position = init_pos
    entry_price = init_entry_px
    entry_date = init_entry_dt
    M = init_M
    m = init_m
    entry_pos = init_pos

    for i in range(n):
        if filter_mode_int == 3: filter_on = fund_vals[i] == 1
        elif filter_mode_int in (1, 2): filter_on = moms[i] > 0.0
        else: filter_on = trends[i] == 1

        if warmups[i]:
            equity_curve[i] = equity
            continue

        if position > 0.0: equity *= 1.0 + position * rets[i] + (1.0 - position) * cash_rets[i]
        else: equity *= 1.0 + cash_rets[i]

        exit_triggered = False
        if position > 0.0:
            if (prices[i] - entry_price) / entry_price < -stop_loss: exit_triggered = True
            M = max(M, prices[i])
            stop_lvl = M * (1.0 - N_atr * atrs[i]) if use_atr_stop else M * (1.0 - X)
            if prices[i] < stop_lvl or not filter_on: exit_triggered = True
            
        if position > 0.0 and exit_triggered:
            position = 0.0; entry_price = np.nan; M = np.nan; m = prices[i]

        if position == 0.0:
            m = min(m, prices[i]) if not np.isnan(m) else prices[i]
            if prices[i] > (1.0 + Y) * m and filter_on and gate_vals[i] == 1:
                position = calc_position_numba(vols[i], position_mode_int, target_vol, max_leverage)
                entry_price = prices[i]; entry_date = dates[i]; M = prices[i]; entry_pos = position
        
        equity_curve[i] = equity
        
    return equity_curve, position, entry_price, entry_date, entry_pos, M, m

@njit(cache=True, nogil=True)
def run_numba_full(
    prices: np.ndarray, rets: np.ndarray, cash_rets: np.ndarray, trends: np.ndarray,
    moms: np.ndarray, vols: np.ndarray, atrs: np.ndarray, warmups: np.ndarray,
    gate_vals: np.ndarray, fund_vals: np.ndarray, dates: np.ndarray,
    filter_mode_int: int, position_mode_int: int, stop_loss: float,
    target_vol: float, max_leverage: float, use_atr_stop: bool,
    N_atr: float, X: float, Y: float,
    init_pos: float, init_entry_px: float, init_entry_dt: int, init_M: float, init_m: float, init_carried: bool
) -> tuple:
    n = len(prices)
    equity_curve = np.zeros(n)
    equity = 1.0
    position = init_pos
    entry_price = init_entry_px
    entry_date = init_entry_dt
    entry_pos = init_pos
    M = init_M
    m = init_m
    is_carried = init_carried
    entry_reason_int = 1 if init_pos > 0.0 else 0

    out_en_dt, out_ex_dt = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    out_en_px, out_ex_px, out_rets, out_pos = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    out_ex_rs, out_days = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    out_cross = np.zeros(n, dtype=np.bool_)
    t_idx = 0

    for i in range(n):
        if filter_mode_int == 3: filter_on = fund_vals[i] == 1
        elif filter_mode_int in (1, 2): filter_on = moms[i] > 0.0
        else: filter_on = trends[i] == 1

        if warmups[i]:
            equity_curve[i] = equity
            continue

        if position > 0.0: equity *= 1.0 + position * rets[i] + (1.0 - position) * cash_rets[i]
        else: equity *= 1.0 + cash_rets[i]

        exit_code = 0
        if position > 0.0:
            if (prices[i] - entry_price) / entry_price < -stop_loss: exit_code |= 1
            M = max(M, prices[i])
            stop_lvl = M * (1.0 - N_atr * atrs[i]) if use_atr_stop else M * (1.0 - X)
            if prices[i] < stop_lvl: exit_code |= 2
            if not filter_on: exit_code |= 4

        if position > 0.0 and exit_code > 0:
            out_en_dt[t_idx] = entry_date; out_ex_dt[t_idx] = dates[i]
            out_en_px[t_idx] = entry_price; out_ex_px[t_idx] = prices[i]
            out_pos[t_idx] = entry_pos; out_rets[t_idx] = (prices[i] / entry_price - 1.0 - 0.0020)
            out_ex_rs[t_idx] = exit_code; out_cross[t_idx] = is_carried
            out_days[t_idx] = (dates[i] - entry_date) // 86400000000000
            t_idx += 1
            position = 0.0; entry_price = np.nan; M = np.nan; m = prices[i]; is_carried = False; entry_reason_int = 0

        if position == 0.0:
            m = min(m, prices[i]) if not np.isnan(m) else prices[i]
            if prices[i] > (1.0 + Y) * m and filter_on and gate_vals[i] == 1:
                position = calc_position_numba(vols[i], position_mode_int, target_vol, max_leverage)
                entry_price = prices[i]; entry_date = dates[i]; M = prices[i]; entry_pos = position; is_carried = False; entry_reason_int = 1

        equity_curve[i] = equity

    return (equity_curve, position, entry_price, entry_date, entry_pos, M, m, is_carried, entry_reason_int,
            out_en_dt[:t_idx], out_ex_dt[:t_idx], out_en_px[:t_idx], out_pos[:t_idx], 
            out_ex_px[:t_idx], out_rets[:t_idx], out_days[:t_idx], out_ex_rs[:t_idx], out_cross[:t_idx])


def decode_exit_reasons(
    reasons_int: int
) -> str | None:
    if reasons_int == 0:
        return None
    reasons =[]
    if reasons_int & 1:
        reasons.append("ABSOLUTE_STOP")
    if reasons_int & 2:
        reasons.append("TRAIL_STOP")
    if reasons_int & 4:
        reasons.append("FILTER_EXIT")
    return " + ".join(reasons)

def decode_entry_reason(
    reason_int: int
) -> str | None:
    if reason_int == 1:
        return "BREAKOUT & FILTER"
    return None





# ============================
# Strategy Engine
# ============================
def run_strategy_with_trades(
    df:                pd.DataFrame,
    price_col:         str   = "price",
    X:                 float = 0.1,
    Y:                 float = 0.1,
    stop_loss:         float = 0.1,
    fast:              int   = 50,
    slow:              int   = 200,
    vol_window:        int   = 20,
    target_vol:        float = 0.10,
    max_leverage:      float = 1.0,
    position_mode:     str   = "vol_entry",
    filter_mode:       str   = "ma",
    mom_lookback:      int   = 252,
    cash_df:           pd.DataFrame | None = None,
    safe_rate:         float = 0.0,
    initial_state:     dict | None = None,
    warmup_df:         pd.DataFrame | None = None,
    fund_signal:       pd.Series | None = None,
    entry_gate:        pd.Series | None = None,
    use_atr_stop:      bool  = False,
    N_atr:             float = 0.1,
    atr_window:        int   = 20,
    engine_mode:       str   = "numba_full"
) -> tuple[pd.DataFrame, dict[str, float] | None, pd.DataFrame, dict | None]:

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
    has_hl = (
        "Najwyzszy" in df.columns
        and "Najnizszy" in df.columns
        and not df["Najwyzszy"].isna().all()
        and not df["Najnizszy"].isna().all()
    )
    if has_hl:
        df["high"] = df["Najwyzszy"]
        df["low"] = df["Najnizszy"]

    if warmup_df is not None:
        warmup = warmup_df.copy()
        warmup["price"] = warmup[price_col]
        warmup_has_hl = (
            "Najwyzszy" in warmup.columns
            and "Najnizszy" in warmup.columns
            and not warmup["Najwyzszy"].isna().all()
            and not warmup["Najnizszy"].isna().all()
        )
        if warmup_has_hl:
            warmup["high"] = warmup["Najwyzszy"]
            warmup["low"] = warmup["Najnizszy"]

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
        n_years = max(len(oos_cash) / 252, 0.01)
        rf_rate = cumulative ** (1 / n_years) - 1
    else:
        rf_rate = safe_rate

    if fund_signal is not None:
        df = df.merge(
            fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left",
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
        df["MOM"] = compute_momentum(df["price"], lookback=mom_lookback, blend=False).shift(1)
    elif filter_mode == "mom_blend":
        df["MOM"] = compute_momentum(df["price"], blend=True).shift(1)
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

        tr = np.maximum(df["high"], prev_close) - np.minimum(df["low"], prev_close)

        df["relative_tr"] = tr / prev_close

        df["atr"] = (
            df["relative_tr"].rolling(atr_window).mean().shift(1)  # avoid lookahead
            
        )

    else:
        # fallback: absolute daily % move * 100 to use 0.08 etc grid
        df["atr"] = (df["price"].diff().abs() / df["price"].shift(1)).rolling(
            atr_window,
        ).mean().shift(1) * 100

    df.dropna(inplace=True)

    # -----------------------
    # Initialise state
    # -----------------------

    filter_mode_map = {"ma": 0, "mom": 1, "mom_blend": 2, "fund": 3}
    filter_mode_int = filter_mode_map.get(filter_mode, 0)
    
    position_mode_map = {"full": 0, "vol_dynamic": 1, "vol_entry": 2}
    position_mode_int = position_mode_map.get(position_mode, 2)

    prices_arr = df["price"].to_numpy(dtype=np.float64)
    rets_arr = df["ret"].fillna(value=0.0).to_numpy(dtype=np.float64)
    cash_rets_arr = df["cash_ret"].fillna(value=0.0).to_numpy(dtype=np.float64)
    trends_arr = df["trend"].fillna(value=0).to_numpy(dtype=np.int64)
    moms_arr = df["MOM"].fillna(value=0.0).to_numpy(dtype=np.float64)
    vols_arr = df["vol"].fillna(value=0.0).to_numpy(dtype=np.float64)
    atrs_arr = df["atr"].to_numpy(dtype=np.float64)
    warmups_arr = df["_warmup"].to_numpy(dtype=np.bool_)
    if gate_aligned is not None:
        gate_vals_arr = gate_aligned.reindex(index=df.index).fillna(value=1).to_numpy(dtype=np.int64)
    else:
        gate_vals_arr = np.ones(shape=len(df), dtype=np.int64)
    if "fund_filter" in df.columns:
        fund_vals_arr = df["fund_filter"].fillna(value=1).to_numpy(dtype=np.int64)
    else:
        fund_vals_arr = np.ones(shape=len(df), dtype=np.int64)
    dates_arr = df.index.values.astype(dtype=np.int64)

    init_position = float(initial_state.get("position", 0.0)) if initial_state else 0.0
    init_entry_price = float(initial_state.get("entry_price", np.nan)) if initial_state and initial_state.get("entry_price") is not None else np.nan
    init_entry_date = int(pd.Timestamp(initial_state["entry_date"]).value) if initial_state and initial_state.get("entry_date") is not None else 0
    init_entry_pos = float(initial_state.get("entry_pos", np.nan)) if initial_state and initial_state.get("entry_pos") is not None else np.nan
    init_M = float(initial_state.get("M", np.nan)) if initial_state and initial_state.get("M") is not None else np.nan
    init_m = float(initial_state.get("m", np.nan)) if initial_state and initial_state.get("m") is not None else np.nan
    init_entry_carried = True if initial_state else False
    init_rebal_count = int(initial_state.get("rebal_count", 0)) if initial_state else 0
    init_rebal_cost_total = float(initial_state.get("rebal_cost_total", 0.0)) if initial_state else 0.0
    init_entry_reason_int = 1 if initial_state and initial_state.get("entry_reason") == "BREAKOUT & FILTER" else (1 if init_position > 0.0 else 0)

    trades_list =[]
    end_state = None

    if engine_mode == "numba_light":
        equity_curve, position, entry_price, entry_date, entry_pos, M, m, rebal_count, rebal_cost_total = run_numba_light(
            prices=prices_arr,
            rets=rets_arr,
            cash_rets=cash_rets_arr,
            trends=trends_arr,
            moms=moms_arr,
            vols=vols_arr,
            atrs=atrs_arr,
            warmups=warmups_arr,
            gate_vals=gate_vals_arr,
            fund_vals=fund_vals_arr,
            dates=dates_arr,
            filter_mode_int=filter_mode_int,
            position_mode_int=position_mode_int,
            stop_loss=stop_loss,
            target_vol=target_vol,
            max_leverage=max_leverage,
            use_atr_stop=use_atr_stop,
            N_atr=N_atr,
            X=X,
            Y=Y,
            init_position=init_position,
            init_entry_price=init_entry_price,
            init_entry_date=init_entry_date,
            init_entry_pos=init_entry_pos,
            init_M=init_M,
            init_m=init_m,
            init_rebal_count=init_rebal_count,
            init_rebal_cost_total=init_rebal_cost_total
        )
        
        if position > 0.0 and not np.isnan(entry_price):
            end_state = {
                "position": position,
                "entry_price": entry_price,
                "entry_date": pd.Timestamp(entry_date) if entry_date > 0 else None,
                "entry_reason": "BREAKOUT & FILTER",
                "entry_pos": entry_pos,
                "M": M,
                "m": m,
                "rebal_count": rebal_count,
                "rebal_cost_total": rebal_cost_total
            }

    elif engine_mode == "numba_full":
        equity_curve, position, entry_price, entry_date, entry_pos, M, m, entry_carried, rebal_count, rebal_cost_total, entry_reason_int, out_entry_dates, out_exit_dates, out_entry_prices, out_positions, out_exit_prices, out_returns, out_days, out_entry_reasons, out_exit_reasons, out_cross_window = run_numba_full(
            prices=prices_arr,
            rets=rets_arr,
            cash_rets=cash_rets_arr,
            trends=trends_arr,
            moms=moms_arr,
            vols=vols_arr,
            atrs=atrs_arr,
            warmups=warmups_arr,
            gate_vals=gate_vals_arr,
            fund_vals=fund_vals_arr,
            dates=dates_arr,
            filter_mode_int=filter_mode_int,
            position_mode_int=position_mode_int,
            stop_loss=stop_loss,
            target_vol=target_vol,
            max_leverage=max_leverage,
            use_atr_stop=use_atr_stop,
            N_atr=N_atr,
            X=X,
            Y=Y,
            init_position=init_position,
            init_entry_price=init_entry_price,
            init_entry_date=init_entry_date,
            init_entry_pos=init_entry_pos,
            init_M=init_M,
            init_m=init_m,
            init_entry_carried=init_entry_carried,
            init_rebal_count=init_rebal_count,
            init_rebal_cost_total=init_rebal_cost_total,
            init_entry_reason_int=init_entry_reason_int
        )

        for trade_idx in range(len(out_entry_dates)):
            trades_list.append({
                "EntryDate": pd.Timestamp(out_entry_dates[trade_idx]),
                "ExitDate": pd.Timestamp(out_exit_dates[trade_idx]),
                "EntryPrice": out_entry_prices[trade_idx],
                "Position": out_positions[trade_idx],
                "ExitPrice": out_exit_prices[trade_idx],
                "Return": out_returns[trade_idx],
                "Days": out_days[trade_idx],
                "Entry Reason": decode_entry_reason(reason_int=out_entry_reasons[trade_idx]),
                "Exit Reason": decode_exit_reasons(reasons_int=out_exit_reasons[trade_idx]),
                "CrossWindow": bool(out_cross_window[trade_idx])
            })

        if position > 0.0 and not np.isnan(entry_price):
            last_date_ts = df.index[-1]
            last_price_val = df["price"].iloc[-1]
            trade_ret_val = last_price_val / entry_price - 1.0
            days_in = (last_date_ts - pd.Timestamp(entry_date)).days
            
            trades_list.append({
                "EntryDate": pd.Timestamp(entry_date),
                "ExitDate": last_date_ts,
                "EntryPrice": entry_price,
                "Position": entry_pos,
                "ExitPrice": last_price_val,
                "Return": trade_ret_val,
                "Days": days_in,
                "Entry Reason": decode_entry_reason(reason_int=entry_reason_int),
                "Exit Reason": "CARRY",
                "CrossWindow": entry_carried
            })

            end_state = {
                "position": position,
                "entry_price": entry_price,
                "entry_date": pd.Timestamp(entry_date) if entry_date > 0 else None,
                "entry_reason": decode_entry_reason(reason_int=entry_reason_int),
                "entry_pos": entry_pos,
                "M": M,
                "m": m,
                "rebal_count": rebal_count,
                "rebal_cost_total": rebal_cost_total
            }

    elif engine_mode == "legacy":
        equity = 1.0
        equity_curve_list =[]
        
        position = init_position
        entry_price = init_entry_price if not np.isnan(init_entry_price) else None
        entry_date = pd.Timestamp(init_entry_date) if init_entry_date > 0 else None
        entry_pos = init_entry_pos if not np.isnan(init_entry_pos) else None
        M = init_M if not np.isnan(init_M) else None
        m = init_m if not np.isnan(init_m) else None
        entry_carried = init_entry_carried
        rebal_count = init_rebal_count
        rebal_cost_total = init_rebal_cost_total
        entry_reason = decode_entry_reason(reason_int=init_entry_reason_int)

        for arr_idx in range(len(prices_arr)):
            current_date_ts = pd.Timestamp(dates_arr[arr_idx])
            price = float(prices_arr[arr_idx])
            ret = float(rets_arr[arr_idx])
            cash_ret = float(cash_rets_arr[arr_idx])
            trend = int(trends_arr[arr_idx])
            mom = float(moms_arr[arr_idx])
            vol = float(vols_arr[arr_idx])
            atr_val = float(atrs_arr[arr_idx])

            if filter_mode_int == 3:
                filter_on = bool(fund_vals_arr[arr_idx] == 1)
            elif filter_mode_int == 1 or filter_mode_int == 2:
                filter_on = mom > 0.0
            else:
                filter_on = trend == 1

            is_warmup_row = bool(warmups_arr[arr_idx])
            if is_warmup_row:
                equity_curve_list.append(equity)
                continue

            if position > 0.0:
                equity *= 1.0 + position * ret + (1.0 - position) * cash_ret
            else:
                equity *= 1.0 + cash_ret

            exit_reasons =[]

            if position > 0.0:
                dd = (price - entry_price) / entry_price
                if dd < -stop_loss:
                    exit_reasons.append("ABSOLUTE_STOP")

            if position > 0.0 and position_mode == "vol_dynamic":
                new_pos = calc_position(
                    vol=vol, 
                    position_mode=position_mode, 
                    target_vol=target_vol, 
                    max_leverage=max_leverage
                )
                size_change = abs(new_pos - position)
                if size_change > 0.1:
                    rebal_cost = 0.0005
                    equity *= 1.0 - size_change * rebal_cost
                    position = new_pos
                    rebal_count += 1
                    rebal_cost_total += size_change * rebal_cost

            if position > 0.0:
                M = max(M, price) if M is not None else price
                if use_atr_stop:
                    if np.isfinite(atr_val) and atr_val > 0.0:
                        stop_level = M * (1.0 - N_atr * atr_val)
                        trail_breached = price < stop_level
                    else:
                        trail_breached = False
                else:
                    trail_breached = price < (1.0 - X) * M

                if trail_breached:
                    if "ABSOLUTE_STOP" not in exit_reasons:
                        exit_reasons.append("TRAIL_STOP")
                elif not filter_on:
                    exit_reasons.append("FILTER_EXIT")

            exit_reason = " + ".join(exit_reasons) if exit_reasons else None

            if position > 0.0 and exit_reason:
                cost = 0.0020
                trade_ret = price / entry_price - 1.0 - cost
                days_in = (current_date_ts - entry_date).days
                trades_list.append(
                    {
                        "EntryDate": entry_date,
                        "ExitDate": current_date_ts,
                        "EntryPrice": entry_price,
                        "Position": entry_pos,
                        "ExitPrice": price,
                        "Return": trade_ret,
                        "Days": days_in,
                        "Entry Reason": entry_reason,
                        "Exit Reason": exit_reason,
                        "CrossWindow": entry_carried,
                    }
                )
                position = 0.0
                entry_price = None
                entry_date = None
                entry_reason = None
                M = None
                m = None
                entry_pos = None
                entry_carried = False

            if position == 0.0:
                m = price if m is None else min(m, price)
                gate_allows = gate_vals_arr[arr_idx] == 1
                if (price > (1.0 + Y) * m) and filter_on and gate_allows:
                    entry_reason = "BREAKOUT & FILTER"
                    position = calc_position(
                        vol=vol, 
                        position_mode=position_mode, 
                        target_vol=target_vol, 
                        max_leverage=max_leverage
                    )
                    entry_price = price
                    entry_date = current_date_ts
                    entry_pos = position
                    M = price
                    entry_carried = False

            equity_curve_list.append(equity)
            
        equity_curve = np.array(object=equity_curve_list, dtype=np.float64)

        if position > 0.0 and entry_price is not None:
            last_date_ts = df.index[-1]
            last_price_val = df["price"].iloc[-1]
            trade_ret_val = last_price_val / entry_price - 1.0
            days_in = (last_date_ts - entry_date).days
            
            trades_list.append({
                "EntryDate": entry_date,
                "ExitDate": last_date_ts,
                "EntryPrice": entry_price,
                "Position": entry_pos,
                "ExitPrice": last_price_val,
                "Return": trade_ret_val,
                "Days": days_in,
                "Entry Reason": entry_reason,
                "Exit Reason": "CARRY",
                "CrossWindow": entry_carried
            })

            end_state = {
                "position": position,
                "entry_price": entry_price,
                "entry_date": entry_date,
                "entry_reason": entry_reason,
                "entry_pos": entry_pos,
                "M": M,
                "m": m,
                "rebal_count": rebal_count,
                "rebal_cost_total": rebal_cost_total
            }
            
    else:
        raise ValueError(f"Unknown engine_mode: {engine_mode}")

    df["equity"] = equity_curve
    df = df[~df["_warmup"]].copy()
    df.drop(columns=["_warmup"], inplace=True)

    if "fund_filter" in df.columns:
        df.drop(columns=["fund_filter"], inplace=True)

    if df.isnull().any().any():
        logging.warning(msg="NaN values remain in test rows after dropna — check cash merge")

    first_val = df["equity"].iloc[0]
    if initial_state is not None and abs(first_val - 1.0) > 0.001:
        logging.debug(
            msg=f"Warmup P&L on carried position: {(first_val - 1.0) * 100:.2f}% — excluded from OOS equity",
        )
    if first_val != 0:
        df["equity"] = df["equity"] / first_val

    metrics_dict = compute_metrics(equity=df["equity"], risk_free_rate=rf_rate)
    metrics_dict = {k: float(v) for k, v in metrics_dict.items()}
    trades_df = pd.DataFrame(data=trades_list)

    return df, metrics_dict, trades_df, end_state


# -------------------------------------------------------
# walk_forward — threads state across windows
# -------------------------------------------------------


def evaluate_params(
    filter_mode: str,
    fund_idx: int | None,
    fund_params: dict | None,
    X: float,
    Y: float,
    fast: int,
    slow: int,
    tv: float,
    stop_loss: float,
    train: pd.DataFrame,
    cash_train: pd.DataFrame,
    vol_window: int,
    selected_mode: str,
    funds_df: pd.DataFrame | None,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    objective: str = "calmar",
    mom_lookback: int = 252,
    entry_gate: pd.Series | None = None,
    use_atr_stop: bool = False,
    N_atr: float = 3.0,
    atr_window: int = 20,
    engine_mode: str = "numba_light"
) -> tuple[tuple, float] | None:
    """
    Evaluate a single parameter combination on the training window.

    When use_atr_stop=True, X is not used by run_strategy_with_trades
    (N_atr is used instead). X is still passed as part of the key tuple
    in position 2, but in ATR mode walk_forward substitutes N_atr values
    from N_atr_grid into that position so the key structure is consistent.
    """

    # use_mom = (filter_mode == "mom")

    train_fund_signal = None
    if filter_mode == "fund" and fund_params is not None:
        funds_train = funds_df.loc[(funds_df.index >= train_start) & (funds_df.index < train_end)]
        train_fund_signal = compute_fund_breadth_signal(
            funds_train,
            **fund_params,
        )

    bt_df, metrics_dict, trades_ignored, state_ignored = run_strategy_with_trades(
        df=train,
        cash_df=cash_train,
        price_col="Zamkniecie",
        X=X,
        Y=Y,
        stop_loss=stop_loss,
        fast=fast,
        slow=slow,
        target_vol=tv,
        vol_window=vol_window,
        position_mode=selected_mode,
        filter_mode=filter_mode,
        mom_lookback=mom_lookback,
        fund_signal=train_fund_signal,
        entry_gate=entry_gate,
        use_atr_stop=use_atr_stop,
        N_atr=N_atr,
        atr_window=atr_window,
        engine_mode=engine_mode,
    )

    if metrics_dict is None:
        return None

    max_dd = metrics_dict.get("MaxDD", 0.0)
    sharpe = metrics_dict.get("Sharpe", 0.0)
    calmar = metrics_dict["CAGR"] / abs(max_dd) if max_dd != 0.0 else None
    sortino = metrics_dict.get("Sortino", 0.0)

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
    df: pd.DataFrame,
    cash_df: pd.DataFrame,
    train_years: int = 8,
    test_years: int = 2,
    vol_window: int = 20,
    funds_df: pd.DataFrame | None = None,
    fund_params_grid: list | None = None,
    selected_mode: str = "full",
    filter_modes_override: list[str] | None = None,
    X_grid: list[float] | None = None,
    Y_grid: list[float] | None = None,
    fast_grid: list[int] | None = None,
    slow_grid: list[int] | None = None,
    tv_grid: list[float] | None = None,
    sl_grid: list[float] | None = None,
    mom_lookback_grid: list[int] | None = None,
    objective: str = "calmar",
    n_jobs: int = 1,
    entry_gate_series: pd.Series | None = None,
    use_atr_stop: bool = False,
    N_atr_grid: list[float] | None = None,
    atr_window: int = 20,
    engine_mode: str = "numba_full"
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
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

    # Resolve  grid default
    if X_grid is None:
        X_grid = [0.08, 0.10, 0.12, 0.15, 0.20]
    if Y_grid is None:
        Y_grid =[0.02, 0.03, 0.05, 0.07, 0.10]
    if fast_grid is None:
        fast_grid = [50, 75, 100]
    if slow_grid is None:
        slow_grid = [150, 200, 250]
    if tv_grid is None:
        tv_grid =[0.08, 0.10, 0.12, 0.15, 0.20]
    if sl_grid is None:
        sl_grid =[0.05, 0.08, 0.10, 0.15]
    if mom_lookback_grid is None:
        mom_lookback_grid = [126, 252]
    if N_atr_grid is None:
        N_atr_grid =[0.08, 0.10, 0.12, 0.15, 0.20]

    # The stop grid for neighbour_mean: X_grid in fixed mode, N_atr_grid in ATR mode
    stop_grid = N_atr_grid if use_atr_stop else X_grid

    data_end = df.index.max()
    logging.info(
        msg=f"walk_forward received data from {df.index.min()} to {data_end} ({len(df)} rows)", 
    )
    logging.info("Objective function: %s", objective)
    logging.info(
        "Trailing stop mode: %s  (ATR window=%d)",
        "ATR-scaled (Chandelier)" if use_atr_stop else "fixed percentage",
        atr_window,
    )

    oos_equity_slices = []
    results = []
    all_oos_trades = []

    start = df.index.min()
    carry_state = None

    if filter_modes_override is not None:
        logging.info(msg=f"filter_modes overridden to: {filter_modes_override}")

    while True:
        gate_train = None
        gate_oos   = None
        train_start = start
        train_end   = train_start + pd.DateOffset(years=train_years)
        test_end    = train_end   + pd.DateOffset(years=test_years)

        train = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test = df.loc[(df.index >= train_end) & (df.index < test_end)]

        logging.info(
            msg=f"Iteration: train={train_start.date()} to {train_end.date()} ({len(train)} rows) | "
                f"test={train_end.date()} to {test_end.date()} ({len(test)} rows) | "
                f"data_end={data_end.date()}"
        )

        # Break only if there is zero data left to trade
        if train.empty or test.empty:
            logging.info(msg="Breaking — train or test empty")
            break

        cash_train = cash_df.loc[(cash_df.index >= train_start) & (cash_df.index < train_end)]

        gate_train = None
        if entry_gate_series is not None:
            gate_train = (
                entry_gate_series.reindex(
                    train.index,
                    method="ffill",
                )
                .fillna(1)
                .astype(int)
            )

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
            fast_iter = fast_grid if filter_mode == "ma" else [50]
            slow_iter = slow_grid if filter_mode == "ma" else [200]
            mom_lb_iter = mom_lookback_grid if filter_mode == "mom" else [252]
            fund_iter = (
                list(enumerate(fund_params_grid)) if filter_mode == "fund" else [(None, None)]
            )
            # Stop grid: N_atr_grid in ATR mode, X_grid in fixed mode
            stop_iter = N_atr_grid if use_atr_stop else X_grid

            for fund_idx, fund_params in fund_iter:
                for stop_val in stop_iter:  # stop_val = N_atr or X
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
                                            param_combinations.append(
                                                (
                                                    filter_mode,
                                                    fund_idx,
                                                    fund_params,
                                                    stop_val,
                                                    Y,
                                                    fast,
                                                    slow,
                                                    tv,
                                                    stop_loss,
                                                    mom_lookback,
                                                ),
                                            )

        for backend, n_jobs_inner, label in [
            ("loky", n_jobs, "multiprocessing"),
            ("threading", n_jobs, "threading"),
            (None, 1, "sequential"),
        ]:
            try:
                if backend is None:
                    results_list = [
                        evaluate_params(
                            filter_mode=filter_mode_val,
                            fund_idx=fund_idx_val,
                            fund_params=fund_params_val,
                            X=stop_val_val,
                            Y=Y_val,
                            fast=fast_val,
                            slow=slow_val,
                            tv=tv_val,
                            stop_loss=stop_loss_val,
                            train=train,
                            cash_train=cash_train,
                            vol_window=vol_window,
                            selected_mode=selected_mode,
                            funds_df=funds_df,
                            train_start=train_start,
                            train_end=train_end,
                            objective=objective,
                            mom_lookback=mom_lookback_val,
                            entry_gate=gate_train,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val_val,
                            atr_window=atr_window,
                            engine_mode="numba_light"
                        )
                        for (
                            filter_mode_val,
                            fund_idx_val,
                            fund_params_val,
                            stop_val_val,
                            Y_val,
                            fast_val,
                            slow_val,
                            tv_val,
                            stop_loss_val,
                            mom_lookback_val,
                        ) in param_combinations
                    ]
                else:
                    results_list = Parallel(n_jobs=n_jobs_inner, backend=backend)(
                        delayed(evaluate_params)(
                            filter_mode=filter_mode_val,
                            fund_idx=fund_idx_val,
                            fund_params=fund_params_val,
                            X=stop_val_val,
                            Y=Y_val,
                            fast=fast_val,
                            slow=slow_val,
                            tv=tv_val,
                            stop_loss=stop_loss_val,
                            train=train,
                            cash_train=cash_train,
                            vol_window=vol_window,
                            selected_mode=selected_mode,
                            funds_df=funds_df,
                            train_start=train_start,
                            train_end=train_end,
                            objective=objective,
                            mom_lookback=mom_lookback_val,
                            entry_gate=gate_train,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val_val,
                            atr_window=atr_window,
                            engine_mode="numba_light"
                        )
                        for (
                            filter_mode_val,
                            fund_idx_val,
                            fund_params_val,
                            stop_val_val,
                            Y_val,
                            fast_val,
                            slow_val,
                            tv_val,
                            stop_loss_val,
                            mom_lookback_val,
                        ) in param_combinations
                    )

                logging.info(
                    "Grid search completed using %s backend (%d jobs).",
                    label,
                    n_jobs_inner,
                )
                break

            except Exception as e:
                logging.warning(
                    "Grid search backend '%s' failed: %s — trying next option.",
                    label,
                    e,
                )
                results_list = None

        if results_list is None:
            logging.error("All grid search backends failed. Skipping window.")
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        param_scores = {
            key: score for result in results_list if result is not None for key, score in [result]
        }

        if not param_scores:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        # -------------------------------------------------------
        # Stability-penalised selection
        # pass stop_grid (X_grid or N_atr_grid) to neighbour_mean
        # -------------------------------------------------------
        best_score = -np.inf
        best_params = None
        best_raw_score = -np.inf

        for key, raw_score in param_scores.items():
            same_mode_scores = {k: v for k, v in param_scores.items() if k[0] == key[0]}
            stability = neighbour_mean(key, same_mode_scores, stop_grid, Y_grid)
            combined = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score = combined
                best_raw_score = raw_score
                fund_idx = key[1]

                best_params = {
                    "filter_mode": key[0],
                    "fund_idx": fund_idx,
                    "fund_params": (
                        fund_params_grid[fund_idx]
                        if fund_idx is not None and fund_params_grid is not None
                        else None
                    ),
                    # key[2] is stop parameter: X in fixed mode, N_atr in ATR mode
                    "X": key[2] if not use_atr_stop else X_grid[0],
                    "N_atr": key[2] if use_atr_stop else N_atr_grid[0],
                    "Y": key[3],
                    "fast": key[4],
                    "slow": key[5],
                    "stop_loss": key[7],
                    "mom_lookback": key[8],
                    "use_atr_stop": use_atr_stop,
                    "atr_window": atr_window,
                }
                if selected_mode != "full":
                    best_params["target_vol"] = key[6]

        if best_params is None:
            break
        else:
            stop_label = (
                f"N_atr={best_params['N_atr']:.2f}" if use_atr_stop else f"X={best_params['X']:.2f}"
            )
            logging.info(
                "Window %s: best raw_%s=%.4f | penalised_%s=%.4f | filter=%s | "
                "%s Y=%.2f fast=%d slow=%d sl=%.2f tv=%s mom_lookback=%s",
                train_start.date(),
                objective,
                best_raw_score,
                objective,
                best_score,
                best_params["filter_mode"],
                stop_label,
                best_params["Y"],
                best_params["fast"],
                best_params["slow"],
                best_params["stop_loss"],
                best_params.get("target_vol", "N/A"),
                best_params["mom_lookback"],
            )

        # -------------------------------------------------------
        # OOS run — pass carry_state and ATR parameters
        # -------------------------------------------------------

        WARMUP_BARS = best_params["slow"] + vol_window + 10
        warmup = train.iloc[-WARMUP_BARS:]

        warmup_start = warmup.index.min()
        cash_warmup_and_test = cash_df.loc[
            (cash_df.index >= warmup_start) & (cash_df.index < test_end)
        ]

        oos_fund_signal = None
        if best_params["filter_mode"] == "fund" and best_params["fund_params"] is not None:
            funds_warmup_and_test = funds_df.loc[
                (funds_df.index >= warmup.index.min()) & (funds_df.index < test_end)
            ]
            full_fund_signal = compute_fund_breadth_signal(
                funds_warmup_and_test,
                **best_params["fund_params"],
            )
            oos_fund_signal = full_fund_signal.loc[full_fund_signal.index >= train_end]
            gate_oos = None
            if entry_gate_series is not None:
                gate_oos = (
                    entry_gate_series.reindex(
                        test.index,
                        method="ffill",
                    )
                    .fillna(1)
                    .astype(int)
                )

        # Build kwargs for run_strategy_with_trades — exclude meta keys
        strategy_keys_to_exclude = {
            "filter_mode",
            "fund_params",
            "fund_idx",
            "target_vol",
            "use_atr_stop",
            "mom_lookback",
            "atr_window",
            "N_atr",
        }

        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            df=test,
            price_col="Zamkniecie",
            cash_df=cash_warmup_and_test,
            position_mode=selected_mode,
            vol_window=vol_window,
            initial_state=carry_state,
            warmup_df=warmup,
            entry_gate=gate_oos,
            fund_signal=oos_fund_signal,
            filter_mode=best_params["filter_mode"],
            mom_lookback=best_params["mom_lookback"],
            use_atr_stop=best_params["use_atr_stop"],
            N_atr=best_params["N_atr"],
            atr_window=best_params["atr_window"],
            engine_mode=engine_mode,
            **{k: v for k, v in best_params.items() if k not in strategy_keys_to_exclude},
        )

        if test_metrics is None or bt_oos is None:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        carry_state = end_state

        # --- STUB RULE: Mute stats if OOS is less than 60 days ---
        if len(test) < 60:
            logging.info(msg=f"Stub window detected ({len(test)} days). Muting OOS statistics.")
            for k in test_metrics.keys():
                test_metrics[k] = float('nan')

        equity_slice = bt_oos["equity"].copy()
        if oos_equity_slices:
            prev_end = oos_equity_slices[-1].iloc[-1]
            equity_slice = equity_slice * prev_end

        oos_equity_slices.append(equity_slice)

        if not oos_trades.empty:
            oos_trades = oos_trades.copy()
            oos_trades["WF_Window"] = train_start
            all_oos_trades.append(oos_trades)

        results.append(
            {
                "TrainStart": train_start,
                "TrainEnd": train_end,
                "TestStart": train_end,
                "TestEnd": test_end,
                "filter_mode": best_params["filter_mode"],
                "fund_idx": best_params["fund_idx"],
                "fund_params": str(best_params["fund_params"]),
                **{
                    k: v
                    for k, v in best_params.items()
                    if k
                    not in (
                        "filter_mode",
                        "fund_params",
                        "fund_idx",
                        "target_vol",
                        "use_atr_stop",
                        "atr_window",
                    )
                },
                "target_vol": best_params.get("target_vol", "N/A"),
                "mom_lookback": best_params.get("mom_lookback", 252),
                "use_atr_stop": best_params["use_atr_stop"],
                "atr_window": best_params["atr_window"],
                **test_metrics,
            },
        )

        start += pd.DateOffset(years=test_years)



    if carry_state is not None and all_oos_trades:
        logging.info(
            "Position still open at end of final window. Last CARRY trade represents the open P&L.",
        )

    if not oos_equity_slices:
        logging.warning("Walk-forward produced no OOS results.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    oos_equity = pd.concat(oos_equity_slices).sort_index()
    results_df = pd.DataFrame(results)
    oos_trades_df = pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()

    return oos_equity, results_df, oos_trades_df


# ============================================================
# TRADE ANALYSIS
# ============================================================


def analyze_trades(
    trades: pd.DataFrame, 
    boundary_exits: set[str] | None = None
) -> dict[str, float] | None:

    if boundary_exits is None:
        boundary_exits = {"CARRY", "SAMPLE_END"}
        
    if trades.empty:
        return None

    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()

    if trades.empty:
        return None

    n_cross = trades["CrossWindow"].sum() if "CrossWindow" in trades.columns else 0
    if n_cross > 0:
        logging.info("%d trades carried across window boundaries", n_cross)

    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())

    pf = np.inf if loss == 0 else (trades.loc[trades["Return"] > 0, "Return"].sum() / loss)

    return {
        "Trades": float(len(trades)),
        "WinRate": float((trades["Return"] > 0).mean()),
        "AvgWin": float(trades.loc[trades["Return"] > 0, "Return"].mean()),
        "AvgLoss": float(trades.loc[trades["Return"] < 0, "Return"].mean()),
        "ProfitFactor": float(pf),
        "AvgDays": float(trades["Days"].mean()),
        "CrossWindow": float(n_cross),
    }


# ------------------------
# Report Printing Function with Best Parameters
# ------------------------
def print_backtest_report(
    metrics: dict[str, float],
    trades: pd.DataFrame,
    trade_stats: dict[str, float] | None,
    best_params: dict | None = None,
    wf_results: pd.DataFrame | None = None,
    position_mode: str | None = None,
    filter_modes_override: list[str] | None = None,
) -> None:

    logging.info("=" * 80)
    logging.info(f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}")
    if filter_modes_override is not None:
        logging.info(f"Filter mode was forced to:    {filter_modes_override}")
    else:
        logging.info("Filter mode selection set to automatic")
    logging.info("=" * 80)

    if wf_results is not None and not wf_results.empty:
        # Determine which stop column to show based on what is in wf_results
        use_atr = "use_atr_stop" in wf_results.columns and wf_results["use_atr_stop"].any()
        stop_col = "N_atr" if use_atr else "X"

        cols = [
            "TrainStart",
            "TestStart",
            "filter_mode",
            stop_col,
            "Y",
            "fast",
            "slow",
            "target_vol",
            "stop_loss",
            "mom_lookback",
        ]
        # Only include columns that actually exist (graceful fallback)
        cols = [c for c in cols if c in wf_results.columns]

        if "fund_params" in wf_results.columns and wf_results["filter_mode"].eq("fund").any():
            cols.insert(3, "fund_params")

        if use_atr and "atr_window" in wf_results.columns:
            # Show atr_window once as a header note, not per-row
            aw = wf_results["atr_window"].iloc[0]
            logging.info("ATR trailing stop mode active (atr_window=%d)", aw)

        logging.info("\n%s", wf_results[cols].to_string(index=False))

    logging.info("-" * 80)

    # Metrics
    logging.info("METRICS:")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f | Sortino: %.2f",
        metrics["CAGR"] * 100,
        metrics["Vol"] * 100,
        metrics["Sharpe"],
        metrics["MaxDD"] * 100,
        metrics["CalMAR"],
        metrics["Sortino"],
    )
    logging.info("-" * 80)

    if trade_stats:
        logging.info("TRADE STATISTICS:")
        logging.info(
            "Total Trades: %d | Win Rate: %.1f%% | Avg Win: %.2f%% | "
            "Avg Loss: %.2f%% | Profit Factor: %.2f | Avg Days: %.1f",
            trade_stats["Trades"],
            trade_stats["WinRate"] * 100,
            trade_stats["AvgWin"] * 100,
            trade_stats["AvgLoss"] * 100,
            trade_stats["ProfitFactor"],
            trade_stats["AvgDays"],
        )
        logging.info("-" * 80)
    else:
        logging.info("No trades executed in the backtest.")
        logging.info("-" * 80)

    carry_trades = pd.DataFrame()

    if not trades.empty and "Exit Reason" in trades.columns:
        carry_trades = trades[trades["Exit Reason"] == "CARRY"]

        n_carry = len(carry_trades)
        if n_carry > 0:
            logging.info(
                "Note: trade log includes %d CARRY boundary records "
                "excluded from statistics above.",
                n_carry,
            )

        trades_fmt = trades.copy()
        trades_fmt["Return"] = (trades_fmt["Return"] * 100).round(2).astype(str) + "%"
        trades_fmt["EntryPrice"] = trades_fmt["EntryPrice"].round(2)
        trades_fmt["ExitPrice"] = trades_fmt["ExitPrice"].round(2)
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
            last_carry["Return"] * 100,
        )
    logging.info("=" * 80)

