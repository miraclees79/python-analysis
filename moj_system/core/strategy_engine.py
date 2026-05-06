# -*- coding: utf-8 -*-
"""
moj_system/core/strategy_engine.py
==================================
Core trend-following engine. 
Optimized with Numba JIT compilation for extreme performance during Monte Carlo and Bootstrap.
"""

import datetime as dt
import logging
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit


# ============================================================
# CANONICAL N_JOBS CALCULATION
# ============================================================


def get_n_jobs() -> int:
    """Return the recommended number of parallel jobs for this machine."""
    cpu_count = os.cpu_count() or 1
    if cpu_count > 3 and sys.platform == "win32":
        return max(1, cpu_count - 1)
    return cpu_count


# ============================================================
# ANNUAL PERFORMANCE UTILITIES
# ============================================================


def annual_cagr_by_year(portfolio_equity: pd.Series) -> dict[int, float]:
    annual = {}
    df = portfolio_equity.copy()
    df.index = pd.to_datetime(arg=df.index)

    for year in df.index.year.unique():
        yr = df[df.index.year == year]
        if len(yr) < 50:
            continue
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
    wins = 0
    for y in years:
        c = cand_annual.get(y)
        i = incumb_annual.get(y)
        if c is not None and i is not None and c > i:
            wins += 1
    return wins


def load_csv(filename: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            filepath_or_buffer=filename, 
            on_bad_lines="skip", 
            delimiter=",", 
            decimal=".", 
            encoding="utf-8-sig"
        )
    except Exception as exc:
        logging.error(msg=f" Error reading CSV file: {exc}")
        return None

    if df.empty or df.columns.size == 0:
        logging.error(msg=" CSV file is empty or corrupted.")
        return None

    df.columns = df.columns.str.strip()
    date_column = "Data"

    if date_column not in df.columns:
        exact_matches =[col for col in df.columns if col.strip() == date_column]
        if exact_matches:
            date_column = exact_matches[0]
        else:
            logging.error(msg=f" Column '{date_column}' not found. Columns: {df.columns}")
            return None

    if df[date_column].isnull().all():
        logging.error(msg=f" Column '{date_column}' contains only NaN values.")
        return None

    df[date_column] = pd.to_datetime(arg=df[date_column], errors="coerce")
    df.dropna(subset=[date_column], inplace=True)

    if df.empty:
        logging.error(msg=" No valid dates after conversion. Data is discarded.")
        return None

    df = df.sort_values(by=date_column).set_index(keys=date_column)

    newest_date = df.index.max()
    if (dt.datetime.now() - newest_date).days > 10:
        logging.warning(msg=f" The newest observation ({newest_date}) is older than 10 days.")
        return None

    date_diffs = df.index.to_series().diff().dt.days
    breaks = date_diffs[date_diffs > 30].index

    if not breaks.empty:
        last_valid_date = breaks[-1]
        df = df.loc[df.index > last_valid_date]
        logging.info(msg=f" Data contains a break > 30 days. Keeping data from {last_valid_date}.")

    logging.info(msg="SUCCESS! CSV file loaded successfully and processed.")
    return df


def prepare_cash_returns(cash_df: pd.DataFrame, price_col: str = "Zamkniecie") -> pd.DataFrame:
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
    if not blend:
        return series.shift(periods=skip) / series.shift(periods=lookback) - 1

    signals =[]
    for lb in blend_lookbacks:
        sig = series.shift(periods=blend_skip) / series.shift(periods=lb) - 1
        signals.append(sig)

    blended = pd.concat(objs=signals, axis=1).mean(axis=1)
    blended.name = series.name
    return blended


# ============================
# Performance Metrics
# ============================


def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.0, freq: int = 252) -> dict:
    ret = equity.pct_change().dropna()
    years = len(ret) / freq

    if years <= 0 or equity.iloc[0] <= 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "MaxDD": np.nan, "CalMAR": np.nan}

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0 else 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    downside = ret[ret < daily_rf] - daily_rf
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
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "CalMAR": calmar,
    }


def neighbour_mean(key: tuple, scores: dict, stop_grid: list, Y_grid: list) -> float:
    filter_mode, fund_idx, stop_param, Y, fast, slow, tv, sl, mom_lookback = key

    si = min(range(len(stop_grid)), key=lambda idx: abs(stop_grid[idx] - stop_param))
    yi = min(range(len(Y_grid)), key=lambda idx: abs(Y_grid[idx] - Y))

    neighbours =[]
    for ds in [-1, 0, 1]:
        for dy in[-1, 0, 1]:
            nsi, nyi = si + ds, yi + dy
            if 0 <= nsi < len(stop_grid) and 0 <= nyi < len(Y_grid):
                nkey = (filter_mode, fund_idx, stop_grid[nsi], Y_grid[nyi], fast, slow, tv, sl, mom_lookback)
                if nkey in scores:
                    neighbours.append(scores[nkey])

    return np.mean(neighbours) if neighbours else scores[key]


def calc_position(vol: float, position_mode: str, target_vol: float, max_leverage: float) -> float:
    if position_mode == "full":
        return 1.0
    if pd.notna(vol) and vol > 0:
        pos = target_vol / vol
    else:
        pos = 1.0
    return min(pos, max_leverage)


def compute_buy_and_hold(df: pd.DataFrame, price_col: str = "Zamkniecie", start=None, end=None) -> tuple:
    bh = df[price_col].copy()
    if start is not None:
        bh = bh.loc[bh.index >= start]
    if end is not None:
        bh = bh.loc[bh.index <= end]

    if bh.empty:
        return pd.Series(dtype=float), {}

    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(equity=bh_equity)
    return bh_equity, {k: float(v) for k, v in bh_metrics.items()}


# ============================================================
# NUMBA CORE ENGINE (High-Performance JIT)
# ============================================================


@njit(cache=True, nogil=True)
def _numba_core_engine(
    dates_int: np.ndarray, prices: np.ndarray, rets: np.ndarray, cash_rets: np.ndarray, 
    filter_mask: np.ndarray, gate_vals: np.ndarray, vols: np.ndarray, atrs: np.ndarray, warmups: np.ndarray,
    Y: float, stop_loss: float, use_atr_stop: bool, stop_val: float, target_vol: float, 
    max_leverage: float, is_vol_dynamic: bool,
    init_position: float, init_entry_price: float, init_M: float, init_m: float, 
    init_entry_date_int: np.int64, init_entry_pos: float, init_entry_carried: bool, 
    init_rebal_count: int, init_rebal_cost_total: float
):
    """
    Skompilowany rdzeń strategii. Nie używa obiektów Pythona (nogil=True).
    Zabezpiecza nanosekundy używając rozdzielonych macierzy INT i FLOAT.
    """
    n = len(prices)
    equity = np.ones(n, dtype=np.float64)
    
    position = init_position
    entry_price = init_entry_price
    M = init_M
    m = init_m
    entry_idx_val = init_entry_date_int
    entry_pos = init_entry_pos
    entry_carried = init_entry_carried
    rebal_count = init_rebal_count
    rebal_cost_total = init_rebal_cost_total

    # Rozdzielamy daty i metryki, aby nie zgubić precyzji w int64 -> float64 castingu!
    # trade_dates: [entry_date_int, exit_date_int]
    trade_dates = np.zeros((n, 2), dtype=np.int64)
    # trade_metrics:[entry_px, entry_pos, exit_px, exit_ret, exit_reason_bitmask, cross_window]
    trade_metrics = np.zeros((n, 6), dtype=np.float64)
    trade_count = 0
    
    for row_idx in range(n):
        p = prices[row_idx]
        ret = rets[row_idx]
        cash_ret = cash_rets[row_idx]
        filt = filter_mask[row_idx]
        vol = vols[row_idx]
        atr_v = atrs[row_idx]
        gate = gate_vals[row_idx]
        is_warmup = warmups[row_idx]

        if is_warmup:
            equity[row_idx] = equity[row_idx-1] if row_idx > 0 else 1.0
            continue

        # 1. Update Equity
        if position > 0:
            eq_prev = equity[row_idx-1] if row_idx > 0 else 1.0
            equity[row_idx] = eq_prev * (1.0 + position * ret + (1.0 - position) * cash_ret)
        else:
            eq_prev = equity[row_idx-1] if row_idx > 0 else 1.0
            equity[row_idx] = eq_prev * (1.0 + cash_ret)

        exit_reasons = 0 # Bitmask: 1=ABS, 2=TRAIL, 4=FILTER
        
        # 2. Exit Logic
        if position > 0:
            dd = (p - entry_price) / entry_price if entry_price > 0 else 0.0
            if dd < -stop_loss:
                exit_reasons |= 1 
                
        if position > 0 and is_vol_dynamic:
            new_pos = target_vol / vol if vol > 0 else 1.0
            if new_pos > max_leverage: new_pos = max_leverage
            
            size_change = abs(new_pos - position)
            if size_change > 0.1:
                rebal_cost = size_change * 0.0005
                equity[row_idx] *= (1.0 - rebal_cost)
                position = new_pos
                rebal_count += 1
                rebal_cost_total += rebal_cost
                
        if position > 0:
            if M == -1.0: 
                M = p
            elif p > M:
                M = p
                
            trail_breached = False
            if use_atr_stop:
                if not np.isnan(atr_v) and atr_v > 0:
                    stop_level = M * (1.0 - stop_val * atr_v)
                    trail_breached = p < stop_level
            else:
                trail_breached = p < (1.0 - stop_val) * M
                
            if trail_breached:
                if (exit_reasons & 1) == 0: 
                    exit_reasons |= 2 
            elif not filt:
                exit_reasons |= 4 
                
        if position > 0 and exit_reasons > 0:
            cost = 0.0020
            trade_ret = (p / entry_price) - 1.0 - cost if entry_price > 0 else 0.0
            
            # Save trade safely into separated arrays
            trade_dates[trade_count, 0] = entry_idx_val
            trade_dates[trade_count, 1] = dates_int[row_idx]
            
            trade_metrics[trade_count, 0] = entry_price
            trade_metrics[trade_count, 1] = entry_pos
            trade_metrics[trade_count, 2] = p
            trade_metrics[trade_count, 3] = trade_ret
            trade_metrics[trade_count, 4] = float(exit_reasons)
            trade_metrics[trade_count, 5] = 1.0 if entry_carried else 0.0
            trade_count += 1
            
            # Reset state
            position = 0.0
            entry_price = 0.0
            entry_idx_val = -1
            M = -1.0
            m = -1.0
            entry_pos = 0.0
            entry_carried = False
            
        # 3. Entry Logic
        if position == 0.0:
            if m == -1.0:
                m = p
            elif p < m:
                m = p
                
            gate_allows = (gate == 1)
            
            if (p > (1.0 + Y) * m) and filt and gate_allows:
                if is_vol_dynamic:
                    new_pos = target_vol / vol if vol > 0 else 1.0
                    if new_pos > max_leverage: new_pos = max_leverage
                    position = new_pos
                else:
                    position = 1.0
                    
                entry_price = p
                entry_idx_val = dates_int[row_idx]
                entry_pos = position
                M = p
                entry_carried = False
                
    return equity, trade_dates[:trade_count], trade_metrics[:trade_count], position, entry_price, M, m, entry_idx_val, entry_pos, entry_carried, rebal_count, rebal_cost_total


# ============================
# Strategy Engine Wrapper
# ============================


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
    cash_df: pd.DataFrame = None,
    safe_rate: float = 0.0,
    initial_state: dict = None,
    warmup_df: pd.DataFrame = None,
    fund_signal: pd.Series = None,
    entry_gate: pd.Series = None,
    use_atr_stop: bool = False,
    N_atr: float = 0.1,
    atr_window: int = 20,
    fast_mode: bool = True,
):
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
        df = pd.concat(objs=[warmup, df], axis=0)
    else:
        df["_warmup"] = False

    if entry_gate is not None:
        gate_aligned = entry_gate.reindex(index=df.index, method="ffill").fillna(value=1).astype(int)
    else:
        gate_aligned = None

    test_start = df[~df["_warmup"]].index[0]

    if cash_df is not None:
        cash = prepare_cash_returns(cash_df=cash_df)
        df = df.merge(right=cash, left_index=True, right_index=True, how="left")
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()
    else:
        df["cash_ret"] = safe_rate / 252

    if df["cash_ret"].isna().all():
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
            right=fund_signal.rename("fund_filter"),
            left_index=True,
            right_index=True,
            how="left",
        )
        df["fund_filter"] = df["fund_filter"].ffill().fillna(value=0)
    else:
        df["fund_filter"] = 1

    df["ret"] = df["price"].pct_change()
    vol = df["ret"].rolling(window=vol_window).std() * np.sqrt(252)
    df["vol"] = vol.shift(periods=1)
    df["ma_fast"] = df["price"].rolling(window=fast).mean().shift(periods=1)
    df["ma_slow"] = df["price"].rolling(window=slow).mean().shift(periods=1)
    df["trend"] = (df["ma_fast"] > df["ma_slow"]).astype(int)

    if filter_mode == "mom":
        df["MOM"] = compute_momentum(series=df["price"], lookback=mom_lookback, blend=False).shift(periods=1)
    elif filter_mode == "mom_blend":
        df["MOM"] = compute_momentum(series=df["price"], blend=True).shift(periods=1)
    else:
        df["MOM"] = 1

    if has_hl:
        prev_close = df["price"].shift(periods=1)
        tr = np.maximum(df["high"], prev_close) - np.minimum(df["low"], prev_close)
        df["relative_tr"] = tr / prev_close
        df["atr"] = df["relative_tr"].rolling(window=atr_window).mean().shift(periods=1) * 100
    else:
        df["atr"] = (df["price"].diff().abs() / df["price"].shift(periods=1)).rolling(window=atr_window).mean().shift(periods=1) * 100

    df.dropna(inplace=True)

    trades =[]
    
    if filter_mode == "fund":
        filter_mode_active = "fund"
    elif filter_mode in ["mom", "mom_blend"]:
        filter_mode_active = "mom"
    else:
        filter_mode_active = "ma"

    # --- NUMBA FAST PATH ---
    if fast_mode:
        prices_arr = df["price"].to_numpy(dtype=np.float64)
        rets_arr = df["ret"].to_numpy(dtype=np.float64)
        cash_rets_arr = df["cash_ret"].to_numpy(dtype=np.float64)
        vols_arr = df["vol"].to_numpy(dtype=np.float64)
        atrs_arr = df["atr"].to_numpy(dtype=np.float64)
        warmups_arr = df["_warmup"].to_numpy(dtype=bool)
        dates_int_arr = df.index.to_numpy(dtype=np.int64) 
        
        if filter_mode_active == "fund":
            filter_mask_arr = df["fund_filter"].to_numpy(dtype=bool) if "fund_filter" in df.columns else np.ones(shape=len(df), dtype=bool)
        elif filter_mode_active == "mom":
            filter_mask_arr = (df["MOM"] > 0).to_numpy(dtype=bool)
        else:
            filter_mask_arr = (df["trend"] == 1).to_numpy(dtype=bool)
            
        gate_vals_arr = gate_aligned.reindex(index=df.index).fillna(value=1).to_numpy(dtype=np.int32) if gate_aligned is not None else np.ones(shape=len(df), dtype=np.int32)
        
        is_vol_dyn = (position_mode == "vol_dynamic")
        _tv = float(target_vol) if target_vol is not None else 0.0
        _ml = float(max_leverage) if max_leverage is not None else 1.0
        _stop_val = float(N_atr) if use_atr_stop else float(X)
        _sl = float(stop_loss)
        _Y = float(Y)
        
        init_pos = float(initial_state["position"]) if initial_state else 0.0
        init_ep = float(initial_state["entry_price"]) if initial_state and initial_state["entry_price"] is not None else 0.0
        init_M = float(initial_state["M"]) if initial_state and initial_state["M"] is not None else -1.0
        init_m = float(initial_state["m"]) if initial_state and initial_state["m"] is not None else -1.0
        init_ed = np.int64(initial_state["entry_date"].value) if initial_state and initial_state["entry_date"] is not None else np.int64(-1)
        init_epos = float(initial_state["entry_pos"]) if initial_state and initial_state["entry_pos"] is not None else 0.0
        init_ec = bool(initial_state["entry_carried"]) if initial_state and "entry_carried" in initial_state else False
        init_rc = int(initial_state.get("rebal_count", 0)) if initial_state else 0
        init_rct = float(initial_state.get("rebal_cost_total", 0.0)) if initial_state else 0.0

        # Execute Numba C-compiled core (Now with separated int and float matrices)
        out_eq, out_t_dates, out_t_metrics, out_pos, out_ep, out_M, out_m, out_ed, out_epos, out_ec, out_rc, out_rct = _numba_core_engine(
            dates_int_arr, prices_arr, rets_arr, cash_rets_arr, filter_mask_arr, gate_vals_arr, vols_arr, atrs_arr, warmups_arr,
            _Y, _sl, use_atr_stop, _stop_val, _tv, _ml, is_vol_dyn,
            init_pos, init_ep, init_M, init_m, init_ed, init_epos, init_ec, init_rc, init_rct
        )

        equity_curve = out_eq.tolist()
        
        # Decode Numba trade matrix back to Python dictionaries without precision loss
        for idx in range(len(out_t_dates)):
            t_ed = pd.Timestamp(out_t_dates[idx, 0])
            t_xd = pd.Timestamp(out_t_dates[idx, 1])
            exit_bitmask = int(out_t_metrics[idx, 4])
            
            exit_reasons_list =[]
            if exit_bitmask & 1: exit_reasons_list.append("ABSOLUTE_STOP")
            if exit_bitmask & 2: exit_reasons_list.append("TRAIL_STOP")
            if exit_bitmask & 4: exit_reasons_list.append("FILTER_EXIT")
            
            trades.append({
                "EntryDate": t_ed,
                "ExitDate": t_xd,
                "EntryPrice": out_t_metrics[idx, 0],
                "Position": out_t_metrics[idx, 1],
                "ExitPrice": out_t_metrics[idx, 2],
                "Return": out_t_metrics[idx, 3],
                "Days": (t_xd - t_ed).days,
                "Entry Reason": "BREAKOUT & FILTER",
                "Exit Reason": " + ".join(exit_reasons_list),
                "CrossWindow": bool(out_t_metrics[idx, 5])
            })
            
        position = out_pos
        entry_price = out_ep if out_ep > 0 else None
        entry_date = pd.Timestamp(int(out_ed)) if out_ed != -1 else None
        M = out_M if out_M != -1.0 else None
        m = out_m if out_m != -1.0 else None
        entry_pos = out_epos if out_epos > 0 else None
        entry_reason = "BREAKOUT & FILTER" if position > 0 else None
        entry_carried = out_ec
        rebal_count = out_rc
        rebal_cost_total = out_rct

    else:
        # Original Python numpy path (preserved for debug/reference but rarely used)
        logging.error(msg="fast_mode=False is no longer supported. Please use fast_mode=True.")
        
        _prices = df["price"].to_numpy()
        _rets = df["ret"].to_numpy()
        _cash_rets = df["cash_ret"].to_numpy()
        _trends = df["trend"].to_numpy()
        _moms = df["MOM"].to_numpy()
        _vols = df["vol"].to_numpy()
        _atrs = df["atr"].to_numpy()  # ATR array
        _warmups = df["_warmup"].to_numpy(dtype=bool)
        _gate_vals = (
            gate_aligned.reindex(index=df.index).fillna(value=1).to_numpy(dtype=np.int32)
            if gate_aligned is not None
            else None
        )
        _fund_vals = df["fund_filter"].to_numpy() if "fund_filter" in df.columns else None
        _index = df.index

        for _n in range(len(_prices)):
            i = _index[_n]
            price = float(_prices[_n])
            ret = float(_rets[_n])
            cash_ret = float(_cash_rets[_n])
            trend = int(_trends[_n])
            mom = float(_moms[_n])
            vol = float(_vols[_n])
            atr_val = float(_atrs[_n])  

            if filter_mode_active == "fund":
                filter_on = bool(_fund_vals[_n]) if _fund_vals is not None else True
            elif filter_mode_active == "mom" or filter_mode_active == "mom_blend":
                filter_on = mom > 0
            else:
                filter_on = trend == 1

            is_warmup_row = bool(_warmups[_n])
            if is_warmup_row:
                equity_curve.append(equity)
                continue

            if position > 0:
                equity *= 1 + position * ret + (1 - position) * cash_ret
            else:
                equity *= 1 + cash_ret

            exit_reasons =[]

            if position > 0:
                dd = (price - entry_price) / entry_price
                if dd < -stop_loss:
                    exit_reasons.append("ABSOLUTE_STOP")

            if position > 0 and position_mode == "vol_dynamic":
                new_pos = calc_position(vol=vol, position_mode=position_mode, target_vol=target_vol, max_leverage=max_leverage)
                size_change = abs(new_pos - position)

                if size_change > 0.1:
                    rebal_cost = 0.0005
                    equity *= 1 - size_change * rebal_cost
                    position = new_pos
                    rebal_count += 1
                    rebal_cost_total += size_change * rebal_cost

            if position > 0:
                M = max(M, price) if M is not None else price
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
                cost = 0.0020
                trade_ret = price / entry_price - 1 - cost
                days = (i - entry_date).days

                trades.append(
                    {
                        "EntryDate": entry_date,
                        "ExitDate": i,
                        "EntryPrice": entry_price,
                        "Position": entry_pos,
                        "ExitPrice": price,
                        "Return": trade_ret,
                        "Days": days,
                        "Entry Reason": entry_reason,
                        "Exit Reason": exit_reason,
                        "CrossWindow": entry_carried,
                    },
                )

                position = 0
                entry_price = None
                entry_date = None
                entry_reason = None
                M = None
                m = None
                entry_pos = None
                entry_carried = False

            if position == 0:
                m = price if m is None else min(m, price)
                gate_allows = _gate_vals is None or int(_gate_vals[_n]) == 1
                if (price > (1 + Y) * m) and filter_on and gate_allows:
                    entry_reason = "BREAKOUT & FILTER"
                    position = calc_position(vol=vol, position_mode=position_mode, target_vol=target_vol, max_leverage=max_leverage)
                    entry_price = price
                    entry_date = i
                    entry_pos = position
                    M = price
                    entry_carried = False

            equity_curve.append(equity)

    if position_mode == "vol_dynamic" and rebal_count > 0:
        logging.debug(
            msg=f"vol_dynamic rebalancing: {rebal_count} adjustments, total cost drag {rebal_cost_total * 100:.4f}%",
        )

    # -----------------------
    # Finalize State
    # -----------------------
    end_state = None

    if position > 0 and entry_price is not None:
        last_date = df.index[-1]
        last_price = df["price"].iloc[-1]
        trade_ret = last_price / entry_price - 1
        days = (last_date - entry_date).days

        if entry_date < test_start:
            logging.debug(
                msg=f"CARRY trade entry date {entry_date} predates test window {test_start} — "
                "trade return and equity curve are on different bases",
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
            },
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
    df.drop(columns=["_warmup"], inplace=True)

    first_val = df["equity"].iloc[0]
    if first_val != 0:
        df["equity"] = df["equity"] / first_val

    metrics = compute_metrics(equity=df["equity"], risk_free_rate=rf_rate)
    metrics = {k: float(v) for k, v in metrics.items()}
    trades_df = pd.DataFrame(data=trades)

    return df, metrics, trades_df, end_state


# -------------------------------------------------------
# walk_forward — threads state across windows
# -------------------------------------------------------


def evaluate_params(
    filter_mode,
    fund_idx,
    fund_params,
    X,
    Y,
    fast,
    slow,
    tv,
    stop_loss,
    train,
    cash_train,
    vol_window,
    selected_mode,
    funds_df,
    train_start,
    train_end,
    objective="calmar",
    mom_lookback=252,
    entry_gate=None,
    fast_mode=True,
    use_atr_stop=False,
    N_atr=3.0,
    atr_window=20,
):
    """Evaluate a single parameter combination on the training window."""
    train_fund_signal = None
    
    bt, metrics, trades, _ = run_strategy_with_trades(
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
        fast_mode=fast_mode,
        use_atr_stop=use_atr_stop,
        N_atr=N_atr,
        atr_window=atr_window,
    )

    if metrics is None:
        return None

    max_dd = metrics.get("MaxDD", 0)
    calmar = metrics["CAGR"] / abs(max_dd) if max_dd != 0 else None

    if objective == "calmar":
        if calmar is None:
            return None
        obj_value = calmar
    else:
        obj_value = calmar

    stop_key_val = N_atr if use_atr_stop else X
    key = (filter_mode, fund_idx, stop_key_val, Y, fast, slow, tv, stop_loss, mom_lookback)
    return key, obj_value


def walk_forward(
    df,
    cash_df,
    train_years=8,
    test_years=2,
    vol_window=20,
    funds_df=None,
    fund_params_grid=None,
    selected_mode="full",
    filter_modes_override=None,
    X_grid=[0.08, 0.10, 0.12, 0.15, 0.20],
    Y_grid=[0.02, 0.03, 0.05, 0.07, 0.10],
    fast_grid=[50, 75, 100],
    slow_grid=[150, 200, 250],
    tv_grid=[0.08, 0.10, 0.12, 0.15, 0.20],
    sl_grid=[0.05, 0.08, 0.10, 0.15],
    mom_lookback_grid=[126, 252],
    objective="calmar",
    n_jobs=1,
    entry_gate_series=None,
    fast_mode=True,
    use_atr_stop=False,
    N_atr_grid=None,
    atr_window=20,
):
    """Run a rolling walk-forward optimisation and return a stitched OOS equity curve."""

    if N_atr_grid is None:
        N_atr_grid =[0.08, 0.10, 0.12, 0.15, 0.20]

    stop_grid = N_atr_grid if use_atr_stop else X_grid
    data_end = df.index.max()

    oos_equity_slices =[]
    results = []
    all_oos_trades =[]

    start = df.index.min()
    carry_state = None

    while True:
        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        train = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test = df.loc[(df.index >= train_end) & (df.index < test_end)]

        logging.info(
            msg=f"Iteration: train={train_start.date()} to {train_end.date()} ({len(train)} rows) | "
                f"test={train_end.date()} to {test_end.date()} ({len(test)} rows) | "
                f"data_end={data_end.date()}"
        )

        if train.empty or test.empty:
            logging.info(msg="Breaking — train or test empty")
            break

        cash_train = cash_df.loc[(cash_df.index >= train_start) & (cash_df.index < train_end)]

        gate_train = None
        if entry_gate_series is not None:
            gate_train = entry_gate_series.reindex(index=train.index, method="ffill").fillna(value=1).astype(int)

        param_scores = {}
        filter_modes = filter_modes_override if filter_modes_override is not None else ["ma", "mom", "mom_blend"]
        param_combinations =[]

        for filter_mode in filter_modes:
            fast_iter = fast_grid if filter_mode == "ma" else [50]
            slow_iter = slow_grid if filter_mode == "ma" else[200]
            mom_lb_iter = mom_lookback_grid if filter_mode == "mom" else [252]
            fund_iter = [(None, None)]
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
                                                ),
                                            )

        backend = "threading"

        try:
            results_list = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(evaluate_params)(
                    filter_mode,
                    fund_idx,
                    fund_params,
                    stop_val,
                    Y,
                    fast,
                    slow,
                    tv,
                    stop_loss,
                    train,
                    cash_train,
                    vol_window,
                    selected_mode,
                    funds_df,
                    train_start,
                    train_end,
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

        except Exception as e:
            logging.error(msg=f"Parallel execution failed: {e}")
            break

        param_scores = {key: score for result in results_list if result is not None for key, score in [result]}

        if not param_scores:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        best_score = -np.inf
        best_params = None

        for key, raw_score in param_scores.items():
            same_mode_scores = {k: v for k, v in param_scores.items() if k[0] == key[0]}
            stability = neighbour_mean(key=key, scores=same_mode_scores, stop_grid=stop_grid, Y_grid=Y_grid)
            combined = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score = combined
                fund_idx = key[1]

                best_params = {
                    "filter_mode": key[0],
                    "fund_idx": fund_idx,
                    "fund_params": None,
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

        WARMUP_BARS = best_params["slow"] + vol_window + 10
        warmup = train.iloc[-WARMUP_BARS:]
        cash_warmup_and_test = cash_df.loc[(cash_df.index >= warmup.index.min()) & (cash_df.index < test_end)]

        gate_oos = None
        if entry_gate_series is not None:
            gate_oos = entry_gate_series.reindex(index=test.index, method="ffill").fillna(value=1).astype(int)

        _keys_to_exclude = {"filter_mode", "fund_params", "fund_idx", "target_vol", "use_atr_stop", "mom_lookback", "atr_window", "N_atr"}

        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            df=test,
            price_col="Zamkniecie",
            cash_df=cash_warmup_and_test,
            position_mode=selected_mode,
            vol_window=vol_window,
            initial_state=carry_state,
            warmup_df=warmup,
            entry_gate=gate_oos,
            fast_mode=fast_mode,
            filter_mode=best_params["filter_mode"],
            mom_lookback=best_params["mom_lookback"],
            use_atr_stop=best_params["use_atr_stop"],
            N_atr=best_params["N_atr"],
            atr_window=best_params["atr_window"],
            **{k: v for k, v in best_params.items() if k not in _keys_to_exclude},
        )

        if test_metrics is None or bt_oos is None:
            start += pd.DateOffset(years=test_years)
            carry_state = None
            continue

        carry_state = end_state

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
                **{k: v for k, v in best_params.items() if k not in _keys_to_exclude},
                "target_vol": best_params.get("target_vol", "N/A"),
                "mom_lookback": best_params.get("mom_lookback", 252),
                "use_atr_stop": best_params["use_atr_stop"],
                "atr_window": best_params["atr_window"],
                **test_metrics,
            },
        )

        start += pd.DateOffset(years=test_years)

    if not oos_equity_slices:
        logging.warning(msg="Walk-forward produced no OOS results.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    oos_equity = pd.concat(objs=oos_equity_slices, axis=0).sort_index()
    results_df = pd.DataFrame(data=results)
    oos_trades_df = pd.concat(objs=all_oos_trades, axis=0) if all_oos_trades else pd.DataFrame()

    return oos_equity, results_df, oos_trades_df


def analyze_trades(trades: pd.DataFrame, boundary_exits: set = None) -> dict | None:
    if boundary_exits is None:
        boundary_exits = {"CARRY", "SAMPLE_END"}
    if trades.empty:
        return None

    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()
    if trades.empty:
        return None

    n_cross = trades["CrossWindow"].sum() if "CrossWindow" in trades.columns else 0
    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())
    pf = np.inf if loss == 0 else (trades.loc[trades["Return"] > 0, "Return"].sum() / loss)

    return {
        "Trades": len(trades),
        "WinRate": (trades["Return"] > 0).mean(),
        "AvgWin": trades.loc[trades["Return"] > 0, "Return"].mean(),
        "AvgLoss": trades.loc[trades["Return"] < 0, "Return"].mean(),
        "ProfitFactor": pf,
        "AvgDays": trades["Days"].mean(),
        "CrossWindow": int(n_cross),
    }


def print_backtest_report(
    metrics: dict,
    trades: pd.DataFrame,
    trade_stats: dict,
    best_params: dict = None,
    wf_results: pd.DataFrame = None,
    position_mode: str = None,
    filter_modes_override: list = None,
) -> None:

    logging.info(msg="=" * 80)
    logging.info(msg=f"WALK-FORWARD OOS BACKTEST REPORT   mode = {position_mode}")
    if filter_modes_override is not None:
        logging.info(msg=f"Filter mode was forced to:    {filter_modes_override}")
    else:
        logging.info(msg="Filter mode selection set to automatic")
    logging.info(msg="=" * 80)

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
            logging.info(msg=f"ATR trailing stop mode active (atr_window={aw})")

        logging.info(msg="\n" + wf_results[cols].to_string(index=False))

    logging.info(msg="-" * 80)
    logging.info(msg="METRICS:")
    logging.info(
        msg=f"CAGR:  {metrics['CAGR'] * 100:.2f}% | Vol: {metrics['Vol'] * 100:.2f}% | "
            f"Sharpe: {metrics['Sharpe']:.2f} | MaxDD: {metrics['MaxDD'] * 100:.2f}% | "
            f"CalMAR: {metrics['CalMAR']:.2f} | Sortino: {metrics['Sortino']:.2f}"
    )
    logging.info(msg="-" * 80)

    if trade_stats:
        logging.info(msg="TRADE STATISTICS:")
        logging.info(
            msg=f"Total Trades: {trade_stats['Trades']} | Win Rate: {trade_stats['WinRate'] * 100:.1f}% | "
                f"Avg Win: {trade_stats['AvgWin'] * 100:.2f}% | Avg Loss: {trade_stats['AvgLoss'] * 100:.2f}% | "
                f"Profit Factor: {trade_stats['ProfitFactor']:.2f} | Avg Days: {trade_stats['AvgDays']:.1f}"
        )
    else:
        logging.info(msg="No trades executed in the backtest.")

    if not trades.empty and trades.iloc[-1]["Exit Reason"] == "CARRY":
        last_carry = trades.iloc[-1]
        logging.info(msg="-" * 80)
        logging.info(
            msg=f"Open position at report date: entry {last_carry['EntryDate']} at {last_carry['EntryPrice']:.2f}, "
                f"current value {last_carry['ExitPrice']:.2f}, unrealised return {last_carry['Return'] * 100:.1f}%"
        )
    logging.info(msg="=" * 80)