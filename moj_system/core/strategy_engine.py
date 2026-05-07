# -*- coding: utf-8 -*-
"""
moj_system/core/strategy_engine.py
==================================
Tri-Engine Strategy Runner & Evaluator.
Contains standard loop, fast python loop, and ultra-fast Numba JIT engine.
"""

import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from moj_system.config import USE_NUMBA_ENGINE

# Dodany brakujący import potrzebny do ewaluacji parametrów funduszy
from moj_system.core.fund_filter import compute_fund_breadth_signal


# ============================================================
# CANONICAL N_JOBS CALCULATION
# ============================================================

def get_n_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count > 3 and sys.platform == "win32":
        return max(1, cpu_count - 1)
    return cpu_count


# ============================================================
# ANNUAL PERFORMANCE UTILITIES
# ============================================================

def annual_cagr_by_year(
    portfolio_equity: pd.Series
) -> dict[int, float]:
    annual = {}
    df = portfolio_equity.copy()
    df.index = pd.to_datetime(
        arg=df.index
    )

    for year in df.index.year.unique():
        yr = df[df.index.year == year]
        if len(yr) < 50:
            continue
        start_val = yr.iloc[0]
        end_val = yr.iloc[-1]
        days = (yr.index[-1] - yr.index[0]).days
        if days < 1 or start_val <= 0:
            continue
        cagr = (end_val / start_val) ** (365.25 / days) - 1.0
        annual[year] = cagr

    return annual


def count_year_wins(
    cand_annual: dict[int, float],
    incumb_annual: dict[int, float],
    years: list[int],
) -> int:
    wins = 0
    for y in years:
        c = cand_annual.get(y)
        i = incumb_annual.get(y)
        if c is not None and i is not None and c > i:
            wins += 1
    return wins


# ============================================================
# IO & PREPARATION
# ============================================================

def load_csv(
    filename: str
) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            filepath_or_buffer=filename,
            on_bad_lines="skip",
            delimiter=",",
            decimal=".",
            encoding="utf-8-sig",
        )
    except Exception as e:
        logging.error(
            msg=f" Error reading CSV file: {e}"
        )
        return None

    if df.empty or df.columns.size == 0:
        logging.error(
            msg=" CSV file is empty or corrupted."
        )
        return None

    df.columns = df.columns.str.strip()
    logging.debug(
        msg="Available columns after stripping:",
        *df.columns
    )

    date_column = "Data"
    if date_column not in df.columns:
        exact_matches =[col for col in df.columns if col.strip() == date_column]
        if exact_matches:
            date_column = exact_matches[0]
            logging.info(
                msg=f" Using corrected column name: '{date_column}'"
            )
        else:
            logging.error(
                msg=f" Column '{date_column}' not found after processing. Available columns: {df.columns}"
            )
            return None

    if df[date_column].isnull().all():
        logging.error(
            msg=f" Column '{date_column}' contains only NaN values."
        )
        return None

    df[date_column] = pd.to_datetime(
        arg=df[date_column],
        errors="coerce"
    )
    df.dropna(
        subset=[date_column],
        inplace=True
    )

    if df.empty:
        logging.error(
            msg=" No valid dates after conversion. Data is discarded."
        )
        return None

    df = df.sort_values(
        by=date_column
    ).set_index(
        keys=date_column
    )

    newest_date = df.index.max()
    if (dt.datetime.now() - newest_date).days > 10:
        logging.warning(
            msg=f" The newest observation ({newest_date}) is older than 10 days. Data is discarded."
        )
        return None

    date_diffs = df.index.to_series().diff().dt.days
    breaks = date_diffs[date_diffs > 30].index

    if not breaks.empty:
        last_valid_date = breaks[-1]
        df = df.loc[df.index > last_valid_date]
        logging.info(
            msg=f" Data contains a break longer than 30 days. Keeping data from {last_valid_date} onward."
        )

    logging.info(
        msg="SUCCESS! CSV file loaded successfully and processed."
    )

    return df


def prepare_cash_returns(
    cash_df: pd.DataFrame,
    price_col: str = "Zamkniecie",
) -> pd.DataFrame:
    cash = cash_df.copy()
    cash["cash_price"] = cash[price_col]
    cash["cash_ret"] = cash["cash_price"].pct_change()
    cash = cash[["cash_ret"]].dropna()
    return cash


# ============================================================
# INDICATORS & METRICS
# ============================================================

def compute_momentum(
    series: pd.Series,
    lookback: int = 252,
    skip: int = 21,
    blend: bool = False,
    blend_lookbacks: tuple[int, ...] = (21, 63, 126, 252),
    blend_skip: int = 5,
) -> pd.Series:
    if not blend:
        return series.shift(periods=skip) / series.shift(periods=lookback) - 1.0

    signals =[]
    for lb in blend_lookbacks:
        sig = series.shift(periods=blend_skip) / series.shift(periods=lb) - 1.0
        signals.append(sig)

    blended = pd.concat(
        objs=signals,
        axis=1
    ).mean(
        axis=1
    )
    blended.name = series.name
    return blended


def compute_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
    freq: int = 252,
) -> dict[str, float]:
    ret = equity.pct_change().dropna()

    years = len(ret) / freq
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0.0 else 0.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    daily_rets = equity.pct_change().dropna()
    downside = daily_rets[daily_rets < daily_rf] - daily_rf
    if len(downside) > 0:
        downside_vol = np.sqrt((downside**2).mean()) * np.sqrt(252.0)
    else:
        downside_vol = 0.0
    sortino = excess_return / downside_vol if downside_vol > 0.0 else 0.0

    cummax = equity.cummax()
    drawdown = equity / cummax - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0.0 else 0.0

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "CalMAR": calmar,
    }


def neighbour_mean(
    key: tuple,
    scores: dict,
    stop_grid: list[float],
    Y_grid: list[float],
) -> float:
    filter_mode, fund_idx, stop_param, Y, fast, slow, tv, sl, mom_lookback = key

    si = min(range(len(stop_grid)), key=lambda i: abs(stop_grid[i] - stop_param))
    yi = min(range(len(Y_grid)), key=lambda i: abs(Y_grid[i] - Y))

    neighbours =[]
    for ds in [-1, 0, 1]:
        for dy in[-1, 0, 1]:
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

    return np.mean(neighbours) if neighbours else scores[key]


def calc_position(
    vol: float,
    position_mode: str,
    target_vol: float,
    max_leverage: float,
) -> float:
    if position_mode == "full":
        return 1.0
    if pd.notna(vol) and vol > 0.0:
        pos = target_vol / vol
    else:
        pos = 1.0
    return min(pos, max_leverage)


def compute_buy_and_hold(
    df: pd.DataFrame,
    price_col: str = "Zamkniecie",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    bh = df[price_col].copy()

    if start is not None:
        bh = bh.loc[bh.index >= start]
    if end is not None:
        bh = bh.loc[bh.index <= end]

    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(
        equity=bh_equity
    )

    return bh_equity, {k: float(v) for k, v in bh_metrics.items()}


# ============================================================
# NUMBA CORE ENGINE
# ============================================================

@njit(cache=True, nogil=True)
def _numba_simulation_loop(
    prices: np.ndarray,
    rets: np.ndarray,
    cash_rets: np.ndarray,
    trends: np.ndarray,
    moms: np.ndarray,
    vols: np.ndarray,
    atrs: np.ndarray,
    warmups: np.ndarray,
    gate_vals: np.ndarray,
    fund_vals: np.ndarray,
    filter_mode_active: int,
    position_mode_dynamic: bool,
    target_vol: float,
    max_leverage: float,
    X: float,
    Y: float,
    stop_loss: float,
    use_atr_stop: bool,
    N_atr: float,
) -> np.ndarray:
    """Core numerical calculation of the equity curve to avoid Python loop overhead."""
    n_days = len(prices)
    equity_curve = np.zeros(
        shape=n_days,
        dtype=np.float64
    )
    equity = 1.0

    position = 0.0
    entry_price = 0.0
    M = 0.0
    m = 0.0
    m_is_set = False

    for day_idx in range(n_days):
        price = prices[day_idx]
        ret = rets[day_idx]
        cash_ret = cash_rets[day_idx]
        trend = trends[day_idx]
        mom = moms[day_idx]
        vol = vols[day_idx]
        atr_val = atrs[day_idx]

        if filter_mode_active == 3:  # fund
            filter_on = fund_vals[day_idx] > 0.5
        elif filter_mode_active == 1 or filter_mode_active == 2:  # mom or mom_blend
            filter_on = mom > 0.0
        else:  # ma
            filter_on = trend == 1

        if warmups[day_idx]:
            equity_curve[day_idx] = equity
            continue

        if position > 0.0:
            equity = equity * (1.0 + position * ret + (1.0 - position) * cash_ret)
        else:
            equity = equity * (1.0 + cash_ret)

        exit_reason_triggered = False

        if position > 0.0:
            dd = (price - entry_price) / entry_price
            if dd < -stop_loss:
                exit_reason_triggered = True

            if position_mode_dynamic:
                new_pos = 1.0
                if vol > 0.0:
                    new_pos = target_vol / vol
                if new_pos > max_leverage:
                    new_pos = max_leverage

                size_change = abs(new_pos - position)
                if size_change > 0.1:
                    rebal_cost = 0.0005
                    equity = equity * (1.0 - size_change * rebal_cost)
                    position = new_pos

            if price > M:
                M = price

            trail_breached = False
            if use_atr_stop:
                if np.isfinite(atr_val) and atr_val > 0.0:
                    stop_level = M * (1.0 - N_atr * atr_val)
                    trail_breached = price < stop_level
                else:
                    trail_breached = False
            else:
                trail_breached = price < (1.0 - X) * M

            if trail_breached:
                exit_reason_triggered = True
            elif not filter_on:
                exit_reason_triggered = True

        if position > 0.0 and exit_reason_triggered:
            # Slippage COST jest używane w logu trades, ale nie odejmuje się od samej krzywej equity w kodzie źródłowym
            position = 0.0
            entry_price = 0.0
            M = 0.0
            m = 0.0
            m_is_set = False

        if position == 0.0:
            if not m_is_set or price < m:
                m = price
                m_is_set = True

            gate_allows = gate_vals[day_idx] == 1

            if (price > (1.0 + Y) * m) and filter_on and gate_allows:
                pos = 1.0
                if position_mode_dynamic:
                    if vol > 0.0:
                        pos = target_vol / vol
                    if pos > max_leverage:
                        pos = max_leverage
                position = pos
                entry_price = price
                M = price
                m_is_set = False

        equity_curve[day_idx] = equity

    return equity_curve


def run_strategy_numba(
    df: pd.DataFrame,
    price_col: str = "price",
    X: float = 0.1,
    Y: float = 0.1,
    stop_loss: float = 0.1,
    fast: int = 50,
    slow: int = 200,
    vol_window: int = 20,
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    position_mode: str = "vol_entry",
    filter_mode: str = "ma",
    mom_lookback: int = 252,
    cash_df: pd.DataFrame | None = None,
    safe_rate: float = 0.0,
    warmup_df: pd.DataFrame | None = None,
    fund_signal: pd.Series | None = None,
    entry_gate: pd.Series | None = None,
    use_atr_stop: bool = False,
    N_atr: float = 3.0,
    atr_window: int = 20,
) -> dict[str, float] | None:
    """
    Blazing fast strategy simulator resolving to Numba core. 
    Ideal for repeated calls in parameter evaluation sweeps.
    """
    df_copy = df.copy()
    df_copy["price"] = df_copy[price_col]

    has_hl = (
        "Najwyzszy" in df_copy.columns
        and "Najnizszy" in df_copy.columns
        and not df_copy["Najwyzszy"].isna().all()
        and not df_copy["Najnizszy"].isna().all()
    )
    if has_hl:
        df_copy["high"] = df_copy["Najwyzszy"]
        df_copy["low"] = df_copy["Najnizszy"]

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
        df_copy["_warmup"] = False
        df_copy = pd.concat(
            objs=[warmup, df_copy]
        )
    else:
        df_copy["_warmup"] = False

    if entry_gate is not None:
        gate_aligned = entry_gate.reindex(
            index=df_copy.index,
            method="ffill"
        ).fillna(
            value=1.0
        ).astype(int)
    else:
        gate_aligned = None

    if cash_df is not None:
        cash = prepare_cash_returns(
            cash_df=cash_df
        )
        df_copy = df_copy.merge(
            right=cash,
            left_index=True,
            right_index=True,
            how="left"
        )
        if df_copy["cash_ret"].isna().any():
            df_copy["cash_ret"] = df_copy["cash_ret"].ffill()
    else:
        df_copy["cash_ret"] = safe_rate / 252.0

    if df_copy["cash_ret"].isna().all():
        df_copy["cash_ret"] = safe_rate / 252.0

    oos_cash = df_copy.loc[~df_copy["_warmup"], "cash_ret"]
    if len(oos_cash) > 0 and oos_cash.notna().any():
        cumulative = (1.0 + oos_cash).prod()
        n_years = max(len(oos_cash) / 252.0, 0.01)
        rf_rate = cumulative ** (1.0 / n_years) - 1.0
    else:
        rf_rate = safe_rate

    if fund_signal is not None:
        df_copy = df_copy.merge(
            right=fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left",
        )
        df_copy["fund_filter"] = df_copy["fund_filter"].ffill().fillna(
            value=0.0
        )
    else:
        df_copy["fund_filter"] = 1.0

    df_copy["ret"] = df_copy["price"].pct_change()
    vol = df_copy["ret"].rolling(
        window=vol_window
    ).std() * np.sqrt(252.0)
    df_copy["vol"] = vol.shift(
        periods=1
    )
    df_copy["ma_fast"] = df_copy["price"].rolling(
        window=fast
    ).mean().shift(
        periods=1
    )
    df_copy["ma_slow"] = df_copy["price"].rolling(
        window=slow
    ).mean().shift(
        periods=1
    )
    df_copy["trend"] = (df_copy["ma_fast"] > df_copy["ma_slow"]).astype(int)

    if filter_mode == "mom":
        df_copy["MOM"] = compute_momentum(
            series=df_copy["price"],
            lookback=mom_lookback,
            blend=False
        ).shift(
            periods=1
        )
    elif filter_mode == "mom_blend":
        df_copy["MOM"] = compute_momentum(
            series=df_copy["price"],
            blend=True
        ).shift(
            periods=1
        )
    else:
        df_copy["MOM"] = 1.0

    if has_hl:
        prev_close = df_copy["price"].shift(
            periods=1
        )
        tr = np.maximum(df_copy["high"], prev_close) - np.minimum(df_copy["low"], prev_close)
        df_copy["relative_tr"] = tr / prev_close
        df_copy["atr"] = df_copy["relative_tr"].rolling(
            window=atr_window
        ).mean().shift(
            periods=1
        ) * 100.0
    else:
        df_copy["atr"] = (df_copy["price"].diff().abs() / df_copy["price"].shift(
            periods=1
        )).rolling(
            window=atr_window
        ).mean().shift(
            periods=1
        ) * 100.0

    df_copy.dropna(
        inplace=True
    )

    if fund_signal is not None:
        filter_mode_active = 3
    elif filter_mode == "mom":
        filter_mode_active = 1
    elif filter_mode == "mom_blend":
        filter_mode_active = 2
    else:
        filter_mode_active = 0

    position_mode_dynamic = (position_mode == "vol_dynamic")

    prices_arr = df_copy["price"].to_numpy(dtype=np.float64)
    rets_arr = df_copy["ret"].to_numpy(dtype=np.float64)
    cash_rets_arr = df_copy["cash_ret"].to_numpy(dtype=np.float64)
    trends_arr = df_copy["trend"].to_numpy(dtype=np.int64)
    moms_arr = df_copy["MOM"].to_numpy(dtype=np.float64)
    vols_arr = df_copy["vol"].to_numpy(dtype=np.float64)
    atrs_arr = df_copy["atr"].to_numpy(dtype=np.float64)
    warmups_arr = df_copy["_warmup"].to_numpy(dtype=np.bool_)

    if gate_aligned is not None:
        gate_vals_arr = gate_aligned.reindex(
            index=df_copy.index
        ).fillna(
            value=1.0
        ).to_numpy(
            dtype=np.int64
        )
    else:
        gate_vals_arr = np.ones(
            shape=len(df_copy),
            dtype=np.int64
        )

    if "fund_filter" in df_copy.columns:
        fund_vals_arr = df_copy["fund_filter"].to_numpy(dtype=np.float64)
    else:
        fund_vals_arr = np.ones(
            shape=len(df_copy),
            dtype=np.float64
        )

    equity_curve_arr = _numba_simulation_loop(
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
        filter_mode_active=filter_mode_active,
        position_mode_dynamic=position_mode_dynamic,
        target_vol=target_vol,
        max_leverage=max_leverage,
        X=X,
        Y=Y,
        stop_loss=stop_loss,
        use_atr_stop=use_atr_stop,
        N_atr=N_atr,
    )

    df_copy["equity"] = equity_curve_arr
    df_oos = df_copy[~df_copy["_warmup"]].copy()

    if df_oos.empty:
        return None

    first_val = df_oos["equity"].iloc[0]
    if first_val != 0.0:
        df_oos["equity"] = df_oos["equity"] / first_val

    metrics = compute_metrics(
        equity=df_oos["equity"],
        risk_free_rate=rf_rate,
        freq=252
    )
    metrics_float = {k: float(v) for k, v in metrics.items()}

    return metrics_float


# ============================================================
# PYTHON STRATEGY ENGINE
# ============================================================

def run_strategy_with_trades(
    df: pd.DataFrame,
    price_col: str = "price",
    X: float = 0.1,
    Y: float = 0.1,
    stop_loss: float = 0.1,
    fast: int = 50,
    slow: int = 200,
    vol_window: int = 20,
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    position_mode: str = "vol_entry",
    filter_mode: str = "ma",
    mom_lookback: int = 252,
    cash_df: pd.DataFrame | None = None,
    safe_rate: float = 0.0,
    initial_state: dict | None = None,
    warmup_df: pd.DataFrame | None = None,
    fund_signal: pd.Series | None = None,
    entry_gate: pd.Series | None = None,
    use_atr_stop: bool = False,
    N_atr: float = 3.0,
    atr_window: int = 20,
    fast_mode: bool = True,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict | None]:

    df = df.copy()
    df["price"] = df[price_col]

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
        df = pd.concat(
            objs=[warmup, df]
        )
    else:
        df["_warmup"] = False

    if entry_gate is not None:
        gate_aligned = entry_gate.reindex(
            index=df.index,
            method="ffill"
        ).fillna(
            value=1
        ).astype(int)
    else:
        gate_aligned = None

    test_start = df[~df["_warmup"]].index[0]

    if cash_df is not None:
        cash = prepare_cash_returns(
            cash_df=cash_df
        )
        df = df.merge(
            right=cash,
            left_index=True,
            right_index=True,
            how="left"
        )
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()
    else:
        df["cash_ret"] = safe_rate / 252.0

    if df["cash_ret"].isna().all():
        logging.info(
            msg="Cash series missing — falling back to flat safe_rate"
        )
        df["cash_ret"] = safe_rate / 252.0

    oos_cash = df.loc[~df["_warmup"], "cash_ret"]
    if len(oos_cash) > 0 and oos_cash.notna().any():
        cumulative = (1.0 + oos_cash).prod()
        n_years = max(len(oos_cash) / 252.0, 0.01)
        rf_rate = cumulative ** (1.0 / n_years) - 1.0
    else:
        rf_rate = safe_rate

    if fund_signal is not None:
        df = df.merge(
            right=fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left",
        )
        df["fund_filter"] = df["fund_filter"].ffill().fillna(
            value=0
        )
    else:
        df["fund_filter"] = 1

    df["ret"] = df["price"].pct_change()
    vol = df["ret"].rolling(
        window=vol_window
    ).std() * np.sqrt(252.0)
    df["vol"] = vol.shift(
        periods=1
    )
    df["ma_fast"] = df["price"].rolling(
        window=fast
    ).mean().shift(
        periods=1
    )
    df["ma_slow"] = df["price"].rolling(
        window=slow
    ).mean().shift(
        periods=1
    )
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    if filter_mode == "mom":
        df["MOM"] = compute_momentum(
            series=df["price"],
            lookback=mom_lookback,
            blend=False
        ).shift(
            periods=1
        )
    elif filter_mode == "mom_blend":
        df["MOM"] = compute_momentum(
            series=df["price"],
            blend=True
        ).shift(
            periods=1
        )
    else:
        df["MOM"] = 1

    if has_hl:
        prev_close = df["price"].shift(
            periods=1
        )
        tr = np.maximum(df["high"], prev_close) - np.minimum(df["low"], prev_close)
        df["relative_tr"] = tr / prev_close
        df["atr"] = (
            df["relative_tr"].rolling(
                window=atr_window
            ).mean().shift(
                periods=1
            ) * 100.0
        )
    else:
        df["atr"] = (df["price"].diff().abs() / df["price"].shift(
            periods=1
        )).rolling(
            window=atr_window
        ).mean().shift(
            periods=1
        ) * 100.0

    df.dropna(
        inplace=True
    )

    equity = 1.0
    equity_curve = []
    trades = []

    if initial_state is not None:
        position = initial_state["position"]
        entry_price = initial_state["entry_price"]
        entry_date = initial_state["entry_date"]
        entry_reason = initial_state["entry_reason"]
        entry_pos = initial_state["entry_pos"]
        M = initial_state["M"]
        m = initial_state["m"]
        entry_carried = True
        rebal_count = initial_state.get("rebal_count", 0)
        rebal_cost_total = initial_state.get("rebal_cost_total", 0.0)
    else:
        position = 0.0
        entry_price = None
        entry_date = None
        entry_reason = None
        entry_pos = None
        M = None
        m = None
        entry_carried = False
        rebal_count = 0
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
        prices_arr = df["price"].to_numpy()
        rets_arr = df["ret"].to_numpy()
        cash_rets_arr = df["cash_ret"].to_numpy()
        trends_arr = df["trend"].to_numpy()
        moms_arr = df["MOM"].to_numpy()
        vols_arr = df["vol"].to_numpy()
        atrs_arr = df["atr"].to_numpy()
        warmups_arr = df["_warmup"].to_numpy(dtype=bool)
        gate_vals_arr = (
            gate_aligned.reindex(
                index=df.index
            ).fillna(
                value=1
            ).to_numpy().astype(int)
            if gate_aligned is not None
            else None
        )
        fund_vals_arr = df["fund_filter"].to_numpy() if "fund_filter" in df.columns else None
        index_arr = df.index

        for day_idx in range(len(prices_arr)):
            current_date = index_arr[day_idx]
            price = float(prices_arr[day_idx])
            ret = float(rets_arr[day_idx])
            cash_ret = float(cash_rets_arr[day_idx])
            trend = int(trends_arr[day_idx])
            mom = float(moms_arr[day_idx])
            vol = float(vols_arr[day_idx])
            atr_val = float(atrs_arr[day_idx])

            if filter_mode_active == "fund":
                filter_on = bool(fund_vals_arr[day_idx]) if fund_vals_arr is not None else True
            elif filter_mode_active == "mom" or filter_mode_active == "mom_blend":
                filter_on = mom > 0.0
            else:
                filter_on = trend == 1

            is_warmup_row = bool(warmups_arr[day_idx])
            if is_warmup_row:
                equity_curve.append(equity)
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
                days = (current_date - entry_date).days
                trades.append(
                    {
                        "EntryDate": entry_date,
                        "ExitDate": current_date,
                        "EntryPrice": entry_price,
                        "Position": entry_pos,
                        "ExitPrice": price,
                        "Return": trade_ret,
                        "Days": days,
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
                gate_allows = gate_vals_arr is None or int(gate_vals_arr[day_idx]) == 1
                if (price > (1.0 + Y) * m) and filter_on and gate_allows:
                    entry_reason = "BREAKOUT & FILTER"
                    position = calc_position(
                        vol=vol,
                        position_mode=position_mode,
                        target_vol=target_vol,
                        max_leverage=max_leverage
                    )
                    entry_price = price
                    entry_date = current_date
                    entry_pos = position
                    M = price
                    entry_carried = False

            equity_curve.append(equity)

    else:
        for iter_date, row in df.iterrows():
            price = row["price"]
            ret = row["ret"]
            cash_ret = row["cash_ret"]
            trend = row["trend"]
            mom = row["MOM"]
            vol = row["vol"]
            atr_val = row["atr"]

            if filter_mode_active == "fund":
                filter_on = bool(row["fund_filter"])
            elif filter_mode_active == "mom" or filter_mode_active == "mom_blend":
                filter_on = mom > 0.0
            else:
                filter_on = trend == 1

            is_warmup_row = row["_warmup"]
            if is_warmup_row:
                equity_curve.append(equity)
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
                days = (iter_date - entry_date).days

                trades.append(
                    {
                        "EntryDate": entry_date,
                        "ExitDate": iter_date,
                        "EntryPrice": entry_price,
                        "Position": entry_pos,
                        "ExitPrice": price,
                        "Return": trade_ret,
                        "Days": days,
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
                gate_allows = gate_aligned is None or gate_aligned[iter_date] == 1
                if (price > (1.0 + Y) * m) and filter_on and gate_allows:
                    entry_reason = "BREAKOUT & FILTER"
                    position = calc_position(
                        vol=vol,
                        position_mode=position_mode,
                        target_vol=target_vol,
                        max_leverage=max_leverage
                    )
                    entry_price = price
                    entry_date = iter_date
                    entry_pos = position
                    M = price
                    entry_carried = False

            equity_curve.append(equity)

    if position_mode == "vol_dynamic" and rebal_count > 0:
        logging.debug(
            msg="vol_dynamic rebalancing: %d adjustments, total cost drag %.4f%% (%.1f bps)",
            * (rebal_count, rebal_cost_total * 100.0, rebal_cost_total * 10000.0)
        )

    end_state = None

    if position > 0.0 and entry_price is not None:
        last_date = df.index[-1]
        last_price = df["price"].iloc[-1]
        trade_ret = last_price / entry_price - 1.0
        days = (last_date - entry_date).days

        if entry_date < test_start:
            logging.debug(
                msg="CARRY trade entry date %s predates test window %s — trade return and equity curve are on different bases",
                * (entry_date, test_start)
            )

        trades.append(
            {
                "EntryDate": entry_date,
                "ExitDate": last_date,
                "EntryPrice": entry_price,
                "Position": entry_pos,
                "ExitPrice": last_price,
                "Return": trade_ret,
                "Days": days,
                "Entry Reason": entry_reason,
                "Exit Reason": "CARRY",
                "CrossWindow": entry_carried,
            }
        )

        end_state = {
            "position": position,
            "entry_price": entry_price,
            "entry_date": entry_date,
            "entry_reason": entry_reason,
            "entry_pos": entry_pos,
            "M": M,
            "m": m,
            "rebal_count": rebal_count,
            "rebal_cost_total": rebal_cost_total,
        }

    df["equity"] = equity_curve
    df = df[~df["_warmup"]].copy()
    df.drop(
        columns=["_warmup"],
        inplace=True
    )

    if "fund_filter" in df.columns:
        df.drop(
            columns=["fund_filter"],
            inplace=True
        )

    if df.isnull().any().any():
        logging.warning(
            msg="NaN values remain in test rows after dropna — check cash merge"
        )

    first_val = df["equity"].iloc[0]
    if initial_state is not None and abs(first_val - 1.0) > 0.001:
        logging.debug(
            msg="Warmup P&L on carried position: %.2f%% — excluded from OOS equity",
            * ((first_val - 1.0) * 100.0,)
        )
    if first_val != 0.0:
        df["equity"] = df["equity"] / first_val

    metrics = compute_metrics(
        equity=df["equity"],
        risk_free_rate=rf_rate
    )
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(data=trades)

    return df, metrics, trades_df, end_state


# ============================================================
# EVALUATE PARAMS & WALK FORWARD
# ============================================================

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
    fast_mode: bool = True,
    use_atr_stop: bool = False,
    N_atr: float = 3.0,
    atr_window: int = 20,
) -> tuple[tuple, float] | None:

    train_fund_signal = None
    if filter_mode == "fund" and fund_params is not None and funds_df is not None:
        funds_train = funds_df.loc[(funds_df.index >= train_start) & (funds_df.index < train_end)]
        train_fund_signal = compute_fund_breadth_signal(
            funds_df=funds_train,
            **fund_params,
        )

    if USE_NUMBA_ENGINE:
        metrics = run_strategy_numba(
            df=train,
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
            cash_df=cash_train,
            safe_rate=0.0,
            warmup_df=None,
            fund_signal=train_fund_signal,
            entry_gate=entry_gate,
            use_atr_stop=use_atr_stop,
            N_atr=N_atr,
            atr_window=atr_window,
        )
    else:
        bt, metrics, trades, unused_state = run_strategy_with_trades(
            df=train,
            price_col="Zamkniecie",
            X=X,
            Y=Y,
            stop_loss=stop_loss,
            fast=fast,
            slow=slow,
            target_vol=tv,
            vol_window=vol_window,
            max_leverage=1.0,
            position_mode=selected_mode,
            filter_mode=filter_mode,
            mom_lookback=mom_lookback,
            cash_df=cash_train,
            safe_rate=0.0,
            initial_state=None,
            warmup_df=None,
            fund_signal=train_fund_signal,
            entry_gate=entry_gate,
            use_atr_stop=use_atr_stop,
            N_atr=N_atr,
            atr_window=atr_window,
            fast_mode=fast_mode,
        )

    if metrics is None:
        return None

    max_dd = metrics.get("MaxDD", 0.0)
    sharpe = metrics.get("Sharpe", 0.0)
    calmar = metrics["CAGR"] / abs(max_dd) if max_dd != 0.0 else None
    sortino = metrics.get("Sortino", 0.0)

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
    fund_params_grid: list[dict] | None = None,
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
    fast_mode: bool = True,
    use_atr_stop: bool = False,
    N_atr_grid: list[float] | None = None,
    atr_window: int = 20,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:

    if X_grid is None:
        X_grid =[0.08, 0.10, 0.12, 0.15, 0.20]
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

    stop_grid = N_atr_grid if use_atr_stop else X_grid

    data_end = df.index.max()
    logging.info(
        msg="walk_forward received data from %s to %s (%d rows)",
        *(df.index.min(), data_end, len(df))
    )
    logging.info(
        msg="Objective function: %s",
        *(objective,)
    )
    logging.info(
        msg="Trailing stop mode: %s  (ATR window=%d)",
        *("ATR-scaled (Chandelier)" if use_atr_stop else "fixed percentage", atr_window)
    )

    oos_equity_slices = []
    results =[]
    all_oos_trades =[]

    start = df.index.min()
    carry_state = None

    if filter_modes_override is not None:
        logging.info(
            msg="filter_modes overridden to: %s",
            *(filter_modes_override,)
        )

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

        if train.empty or test.empty:
            logging.info(
                msg="Breaking — train or test empty"
            )
            break

        cash_train = cash_df.loc[(cash_df.index >= train_start) & (cash_df.index < train_end)]

        gate_train = None
        if entry_gate_series is not None:
            gate_train = (
                entry_gate_series.reindex(
                    index=train.index,
                    method="ffill",
                )
                .fillna(
                    value=1
                )
                .astype(int)
            )

        param_scores = {}

        filter_modes = ["ma", "mom", "mom_blend"]
        if funds_df is not None:
            filter_modes.append("fund")

        if filter_modes_override is not None:
            filter_modes = filter_modes_override

        param_combinations =[]

        for filter_mode in filter_modes:
            fast_iter = fast_grid if filter_mode == "ma" else [50]
            slow_iter = slow_grid if filter_mode == "ma" else [200]
            mom_lb_iter = mom_lookback_grid if filter_mode == "mom" else[252]
            fund_iter = (
                list(enumerate(fund_params_grid)) if filter_mode == "fund" and fund_params_grid is not None else [(None, None)]
            )
            stop_iter = N_atr_grid if use_atr_stop else X_grid

            for fund_idx, fund_params in fund_iter:
                for stop_val in stop_iter:
                    for Y in Y_grid:
                        for fast in fast_iter:
                            for slow in slow_iter:
                                if filter_mode == "ma" and slow - fast < 75:
                                    continue
                                for tv in tv_grid if selected_mode != "full" else [0.10]:
                                    for stop_loss in sl_grid:
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
                                                )
                                            )

        for backend, n_jobs_inner, label in[
            ("loky", n_jobs, "multiprocessing"),
            ("threading", n_jobs, "threading"),
            (None, 1, "sequential"),
        ]:
            try:
                if backend is None:
                    results_list =[
                        evaluate_params(
                            filter_mode=filter_mode,
                            fund_idx=fund_idx,
                            fund_params=fund_params,
                            X=stop_val,
                            Y=Y,
                            fast=fast,
                            slow=slow,
                            tv=tv,
                            stop_loss=stop_loss,
                            train=train,
                            cash_train=cash_train,
                            vol_window=vol_window,
                            selected_mode=selected_mode,
                            funds_df=funds_df,
                            train_start=train_start,
                            train_end=train_end,
                            objective=objective,
                            mom_lookback=mom_lookback,
                            entry_gate=gate_train,
                            fast_mode=fast_mode,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val,
                            atr_window=atr_window,
                        )
                        for (
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
                        ) in param_combinations
                    ]
                else:
                    results_list = Parallel(
                        n_jobs=n_jobs_inner,
                        backend=backend
                    )(
                        delayed(
                            function=evaluate_params
                        )(
                            filter_mode=filter_mode,
                            fund_idx=fund_idx,
                            fund_params=fund_params,
                            X=stop_val,
                            Y=Y,
                            fast=fast,
                            slow=slow,
                            tv=tv,
                            stop_loss=stop_loss,
                            train=train,
                            cash_train=cash_train,
                            vol_window=vol_window,
                            selected_mode=selected_mode,
                            funds_df=funds_df,
                            train_start=train_start,
                            train_end=train_end,
                            objective=objective,
                            mom_lookback=mom_lookback,
                            entry_gate=gate_train,
                            fast_mode=fast_mode,
                            use_atr_stop=use_atr_stop,
                            N_atr=stop_val,
                            atr_window=atr_window,
                        )
                        for (
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
                        ) in param_combinations
                    )

                logging.info(
                    msg="Grid search completed using %s backend (%d jobs).",
                    *(label, n_jobs_inner)
                )
                break

            except Exception as e:
                logging.warning(
                    msg="Grid search backend '%s' failed: %s — trying next option.",
                    *(label, e)
                )
                results_list = None

        if results_list is None:
            logging.error(
                msg="All grid search backends failed. Skipping window."
            )
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        param_scores = {
            key: score for result in results_list if result is not None for key, score in[result]
        }

        if not param_scores:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        best_score = -np.inf
        best_params = None
        best_raw_score = -np.inf

        for key, raw_score in param_scores.items():
            same_mode_scores = {k: v for k, v in param_scores.items() if k[0] == key[0]}
            stability = neighbour_mean(
                key=key,
                scores=same_mode_scores,
                stop_grid=stop_grid,
                Y_grid=Y_grid
            )
            combined = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score = combined
                best_raw_score = raw_score
                fund_idx_res = key[1]

                best_params = {
                    "filter_mode": key[0],
                    "fund_idx": fund_idx_res,
                    "fund_params": (
                        fund_params_grid[fund_idx_res]
                        if fund_idx_res is not None and fund_params_grid is not None
                        else None
                    ),
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
                msg="Window %s: best raw_%s=%.4f | penalised_%s=%.4f | filter=%s | %s Y=%.2f fast=%d slow=%d sl=%.2f tv=%s mom_lookback=%s",
                *(
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
                    best_params["mom_lookback"]
                )
            )

        WARMUP_BARS = best_params["slow"] + vol_window + 10
        warmup = train.iloc[-WARMUP_BARS:]

        warmup_start = warmup.index.min()
        cash_warmup_and_test = cash_df.loc[
            (cash_df.index >= warmup_start) & (cash_df.index < test_end)
        ]

        oos_fund_signal = None
        if best_params["filter_mode"] == "fund" and best_params["fund_params"] is not None and funds_df is not None:
            funds_warmup_and_test = funds_df.loc[
                (funds_df.index >= warmup.index.min()) & (funds_df.index < test_end)
            ]
            full_fund_signal = compute_fund_breadth_signal(
                funds_df=funds_warmup_and_test,
                **best_params["fund_params"],
            )
            oos_fund_signal = full_fund_signal.loc[full_fund_signal.index >= train_end]
            gate_oos = None
            if entry_gate_series is not None:
                gate_oos = (
                    entry_gate_series.reindex(
                        index=test.index,
                        method="ffill",
                    )
                    .fillna(
                        value=1
                    )
                    .astype(int)
                )

        _strategy_keys_to_exclude = {
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
            entry_gate=gate_oos if "gate_oos" in dir() else None,
            fund_signal=oos_fund_signal,
            fast_mode=fast_mode,
            filter_mode=best_params["filter_mode"],
            mom_lookback=best_params["mom_lookback"],
            use_atr_stop=best_params["use_atr_stop"],
            N_atr=best_params["N_atr"],
            atr_window=best_params["atr_window"],
            **{k: v for k, v in best_params.items() if k not in _strategy_keys_to_exclude},
        )

        if test_metrics is None or bt_oos is None:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        carry_state = end_state

        if len(test) < 60:
            logging.info(
                msg=f"Stub window detected ({len(test)} days). Muting OOS statistics."
            )
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
            msg="Position still open at end of final window. Last CARRY trade represents the open P&L."
        )

    if not oos_equity_slices:
        logging.warning(
            msg="Walk-forward produced no OOS results."
        )
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    oos_equity = pd.concat(
        objs=oos_equity_slices
    ).sort_index()
    results_df = pd.DataFrame(
        data=results
    )
    oos_trades_df = pd.concat(
        objs=all_oos_trades
    ) if all_oos_trades else pd.DataFrame()

    return oos_equity, results_df, oos_trades_df


# ============================================================
# TRADE ANALYSIS & REPORTING
# ============================================================

def analyze_trades(
    trades: pd.DataFrame,
    boundary_exits: set[str] = {"CARRY", "SAMPLE_END"}
) -> dict[str, float] | None:

    if trades.empty:
        return None

    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()

    if trades.empty:
        return None

    n_cross = trades["CrossWindow"].sum() if "CrossWindow" in trades.columns else 0
    if n_cross > 0:
        logging.info(
            msg="%d trades carried across window boundaries",
            *(n_cross,)
        )

    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())

    pf = np.inf if loss == 0.0 else (trades.loc[trades["Return"] > 0, "Return"].sum() / loss)

    return {
        "Trades": float(len(trades)),
        "WinRate": float((trades["Return"] > 0).mean()),
        "AvgWin": float(trades.loc[trades["Return"] > 0, "Return"].mean()),
        "AvgLoss": float(trades.loc[trades["Return"] < 0, "Return"].mean()),
        "ProfitFactor": float(pf),
        "AvgDays": float(trades["Days"].mean()),
        "CrossWindow": float(n_cross),
    }


def print_backtest_report(
    metrics: dict[str, float],
    trades: pd.DataFrame,
    trade_stats: dict[str, float] | None,
    best_params: dict | None = None,
    wf_results: pd.DataFrame | None = None,
    position_mode: str | None = None,
    filter_modes_override: list[str] | None = None,
) -> None:

    logging.info(
        msg="=" * 80
    )
    logging.info(
        msg=f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}"
    )
    if filter_modes_override is not None:
        logging.info(
            msg=f"Filter mode was forced to:    {filter_modes_override}"
        )
    else:
        logging.info(
            msg="Filter mode selection set to automatic"
        )
    logging.info(
        msg="=" * 80
    )

    if wf_results is not None and not wf_results.empty:
        use_atr = "use_atr_stop" in wf_results.columns and wf_results["use_atr_stop"].any()
        stop_col = "N_atr" if use_atr else "X"

        cols =[
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
        cols =[c for c in cols if c in wf_results.columns]

        if "fund_params" in wf_results.columns and wf_results["filter_mode"].eq("fund").any():
            cols.insert(3, "fund_params")

        if use_atr and "atr_window" in wf_results.columns:
            aw = wf_results["atr_window"].iloc[0]
            logging.info(
                msg="ATR trailing stop mode active (atr_window=%d)",
                *(aw,)
            )

        logging.info(
            msg="\n%s",
            *(wf_results[cols].to_string(index=False),)
        )

    logging.info(
        msg="-" * 80
    )

    logging.info(
        msg="METRICS:"
    )
    logging.info(
        msg="CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f | Sortino: %.2f",
        *(
            metrics["CAGR"] * 100.0,
            metrics["Vol"] * 100.0,
            metrics["Sharpe"],
            metrics["MaxDD"] * 100.0,
            metrics["CalMAR"],
            metrics["Sortino"]
        )
    )
    logging.info(
        msg="-" * 80
    )

    if trade_stats:
        logging.info(
            msg="TRADE STATISTICS:"
        )
        logging.info(
            msg="Total Trades: %d | Win Rate: %.1f%% | Avg Win: %.2f%% | Avg Loss: %.2f%% | Profit Factor: %.2f | Avg Days: %.1f",
            *(
                int(trade_stats["Trades"]),
                trade_stats["WinRate"] * 100.0,
                trade_stats["AvgWin"] * 100.0,
                trade_stats["AvgLoss"] * 100.0,
                trade_stats["ProfitFactor"],
                trade_stats["AvgDays"]
            )
        )
        logging.info(
            msg="-" * 80
        )
    else:
        logging.info(
            msg="No trades executed in the backtest."
        )
        logging.info(
            msg="-" * 80
        )

    if not trades.empty and "Exit Reason" in trades.columns:
        carry_trades = trades[trades["Exit Reason"] == "CARRY"]

        n_carry = len(carry_trades)
        if n_carry > 0:
            logging.info(
                msg="Note: trade log includes %d CARRY boundary records excluded from statistics above.",
                *(n_carry,)
            )

        trades_fmt = trades.copy()
        trades_fmt["Return"] = (trades_fmt["Return"] * 100.0).round(decimals=2).astype(str) + "%"
        trades_fmt["EntryPrice"] = trades_fmt["EntryPrice"].round(decimals=2)
        trades_fmt["ExitPrice"] = trades_fmt["ExitPrice"].round(decimals=2)
        logging.info(
            msg="TRADE LOG:"
        )
        logging.info(
            msg="\n%s",
            *(trades_fmt.to_string(index=False),)
        )

    if not trades.empty and trades.iloc[-1]["Exit Reason"] == "CARRY":
        last_carry = trades.iloc[-1]
        logging.info(
            msg="Open position at report date: entry %s at %.2f, current value %.2f, unrealised return %.1f%%",
            *(
                last_carry["EntryDate"],
                last_carry["EntryPrice"],
                last_carry["ExitPrice"],
                last_carry["Return"] * 100.0
            )
        )
    logging.info(
        msg="=" * 80
    )
