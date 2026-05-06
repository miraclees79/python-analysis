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
    df.index = pd.to_datetime(arg=df.index)

    for year_val in df.index.year.unique():
        yr = df[df.index.year == year_val]
        if len(yr) < 50:
            continue
        start_val = yr.iloc[0]
        end_val = yr.iloc[-1]
        days = (yr.index[-1] - yr.index[0]).days
        if days < 1 or start_val <= 0:
            continue
        cagr = (end_val / start_val) ** (365.25 / days) - 1
        annual[year_val] = cagr

    return annual


def count_year_wins(
    cand_annual: dict[int, float],
    incumb_annual: dict[int, float],
    years: list[int],
) -> int:
    wins = 0
    for year_val in years:
        cand_score = cand_annual.get(year_val)
        incumb_score = incumb_annual.get(year_val)
        if cand_score is not None and incumb_score is not None and cand_score > incumb_score:
            wins += 1
    return wins

# ============================================================
# DATA UTILITIES
# ============================================================

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


def compute_metrics(
    equity: pd.Series, 
    risk_free_rate: float = 0.0, 
    freq: int = 252
) -> dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CalMAR": 0.0}
    
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CalMAR": 0.0}

    years = len(ret) / freq
    if years == 0:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "CalMAR": 0.0}

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(freq)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    downside = ret[ret < daily_rf] - daily_rf
    downside_vol = np.sqrt((downside**2).mean()) * np.sqrt(252) if not downside.empty else 0.0
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe),
        "Sortino": float(sortino), "MaxDD": float(max_dd), "CalMAR": float(calmar),
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
    filter_mode, fund_idx, stop_param, Y, fast, slow, tv, sl, mom_lookback = key

    si = min(range(len(stop_grid)), key=lambda idx: abs(stop_grid[idx] - stop_param))
    yi = min(range(len(Y_grid)), key=lambda idx: abs(Y_grid[idx] - Y))

    neighbours =[]
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

    if bh.empty:
        return pd.Series(dtype=float), {}

    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(equity=bh_equity)

    return bh_equity, bh_metrics

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
    prices:                np.ndarray,
    rets:                  np.ndarray,
    cash_rets:             np.ndarray,
    trends:                np.ndarray,
    moms:                  np.ndarray,
    vols:                  np.ndarray,
    atrs:                  np.ndarray,
    warmups:               np.ndarray,
    gate_vals:             np.ndarray,
    fund_vals:             np.ndarray,
    dates:                 np.ndarray,
    filter_mode_int:       int,
    position_mode_int:     int,
    stop_loss:             float,
    target_vol:            float,
    max_leverage:          float,
    use_atr_stop:          bool,
    N_atr:                 float,
    X:                     float,
    Y:                     float,
    init_pos:              float,
    init_entry_px:         float,
    init_entry_dt:         int,
    init_entry_pos:        float,
    init_M:                float,
    init_m:                float,
    init_rebal_count:      int,
    init_rebal_cost_total: float,
    init_carried:          bool,
    init_entry_reason_int: int
) -> tuple:
    n_bars = len(prices)
    equity_curve = np.zeros(shape=n_bars, dtype=np.float64)
    equity_val = 1.0

    current_pos = init_pos
    entry_px = init_entry_px
    entry_dt = init_entry_dt
    entry_pos = init_entry_pos
    running_M = init_M
    running_m = init_m
    rebal_c = init_rebal_count
    rebal_cost = init_rebal_cost_total

    for i in range(n_bars):
        if filter_mode_int == 3:
            filter_on = fund_vals[i] == 1
        elif filter_mode_int == 1 or filter_mode_int == 2:
            filter_on = moms[i] > 0.0
        else:
            filter_on = trends[i] == 1

        if warmups[i]:
            equity_curve[i] = equity_val
            continue

        if current_pos > 0.0:
            equity_val *= 1.0 + current_pos * rets[i] + (1.0 - current_pos) * cash_rets[i]
        else:
            equity_val *= 1.0 + cash_rets[i]

        exit_triggered = False
        if current_pos > 0.0:
            if (prices[i] - entry_px) / entry_px < -stop_loss:
                exit_triggered = True
            
            running_M = max(running_M, prices[i])
            current_stop = running_M * (1.0 - N_atr * atrs[i]) if use_atr_stop else running_M * (1.0 - X)
            
            if prices[i] < current_stop or not filter_on:
                exit_triggered = True

        if current_pos > 0.0 and exit_triggered:
            current_pos = 0.0
            entry_px = np.nan
            entry_pos = np.nan
            running_M = np.nan
            running_m = prices[i]

        if current_pos == 0.0:
            if np.isnan(running_m):
                running_m = prices[i]
            else:
                running_m = min(running_m, prices[i])
            
            if (prices[i] > (1.0 + Y) * running_m) and filter_on and gate_vals[i] == 1:
                current_pos = calc_position_numba(
                    vol=vols[i], 
                    position_mode_int=position_mode_int, 
                    target_vol=target_vol, 
                    max_leverage=max_leverage
                )
                entry_px = prices[i]
                entry_dt = dates[i]
                entry_pos = current_pos
                running_M = prices[i]

        equity_curve[i] = equity_val

    return equity_curve, current_pos, entry_px, entry_dt, entry_pos, running_M, running_m, rebal_c, rebal_cost


@njit(cache=True, nogil=True)
def run_numba_full(
    prices:                np.ndarray,
    rets:                  np.ndarray,
    cash_rets:             np.ndarray,
    trends:                np.ndarray,
    moms:                  np.ndarray,
    vols:                  np.ndarray,
    atrs:                  np.ndarray,
    warmups:               np.ndarray,
    gate_vals:             np.ndarray,
    fund_vals:             np.ndarray,
    dates:                 np.ndarray,
    filter_mode_int:       int,
    position_mode_int:     int,
    stop_loss:             float,
    target_vol:            float,
    max_leverage:          float,
    use_atr_stop:          bool,
    N_atr:                 float,
    X:                     float,
    Y:                     float,
    init_pos:              float,
    init_entry_px:         float,
    init_entry_dt:         int,
    init_entry_pos:        float,
    init_M:                float,
    init_m:                float,
    init_rebal_count:      int,
    init_rebal_cost_total: float,
    init_carried:          bool,
    init_entry_reason_int: int
) -> tuple:
    n_bars = len(prices)
    equity_curve = np.zeros(shape=n_bars, dtype=np.float64)
    equity_val = 1.0

    current_pos = init_pos
    entry_px = init_entry_px
    entry_dt = init_entry_dt
    entry_pos = init_entry_pos
    running_M = init_M
    running_m = init_m
    is_carried_flag = init_carried
    rebal_c = init_rebal_count
    rebal_cost = init_rebal_cost_total
    entry_reason = init_entry_reason_int

    out_en_dt = np.zeros(shape=n_bars, dtype=np.int64)
    out_ex_dt = np.zeros(shape=n_bars, dtype=np.int64)
    out_en_px = np.zeros(shape=n_bars, dtype=np.float64)
    out_ex_px = np.zeros(shape=n_bars, dtype=np.float64)
    out_pos   = np.zeros(shape=n_bars, dtype=np.float64)
    out_rets  = np.zeros(shape=n_bars, dtype=np.float64)
    out_days  = np.zeros(shape=n_bars, dtype=np.int64)
    out_en_rs = np.zeros(shape=n_bars, dtype=np.int64)
    out_ex_rs = np.zeros(shape=n_bars, dtype=np.int64)
    out_cross = np.zeros(shape=n_bars, dtype=np.bool_)
    trade_idx = 0

    for i in range(n_bars):
        if filter_mode_int == 3:
            filter_on = fund_vals[i] == 1
        elif filter_mode_int == 1 or filter_mode_int == 2:
            filter_on = moms[i] > 0.0
        else:
            filter_on = trends[i] == 1

        if warmups[i]:
            equity_curve[i] = equity_val
            continue

        if current_pos > 0.0:
            equity_val *= 1.0 + current_pos * rets[i] + (1.0 - current_pos) * cash_rets[i]
        else:
            equity_val *= 1.0 + cash_rets[i]

        exit_code = 0
        if current_pos > 0.0:
            if (prices[i] - entry_px) / entry_px < -stop_loss:
                exit_code |= 1
            
            running_M = max(running_M, prices[i])
            current_stop = running_M * (1.0 - N_atr * atrs[i]) if use_atr_stop else running_M * (1.0 - X)
            
            if prices[i] < current_stop:
                exit_code |= 2
            if not filter_on:
                exit_code |= 4

        if current_pos > 0.0 and exit_code > 0:
            cost_fixed = 0.0020
            trade_return = prices[i] / entry_px - 1.0 - cost_fixed
            days_in_trade = (dates[i] - entry_dt) // 86400000000000
            
            out_en_dt[trade_idx] = entry_dt
            out_ex_dt[trade_idx] = dates[i]
            out_en_px[trade_idx] = entry_px
            out_pos[trade_idx]   = entry_pos
            out_ex_px[trade_idx] = prices[i]
            out_rets[trade_idx]  = trade_return
            out_days[trade_idx]  = days_in_trade
            out_en_rs[trade_idx] = entry_reason
            out_ex_rs[trade_idx] = exit_code
            out_cross[trade_idx] = is_carried_flag
            trade_idx += 1

            current_pos = 0.0
            entry_px = np.nan
            entry_dt = 0
            entry_pos = np.nan
            running_M = np.nan
            running_m = prices[i]
            is_carried_flag = False
            entry_reason = 0

        if current_pos == 0.0:
            if np.isnan(running_m):
                running_m = prices[i]
            else:
                running_m = min(running_m, prices[i])
            
            if (prices[i] > (1.0 + Y) * running_m) and filter_on and gate_vals[i] == 1:
                current_pos = calc_position_numba(
                    vol=vols[i], 
                    position_mode_int=position_mode_int, 
                    target_vol=target_vol, 
                    max_leverage=max_leverage
                )
                entry_px = prices[i]
                entry_dt = dates[i]
                entry_pos = current_pos
                running_M = prices[i]
                is_carried_flag = False
                entry_reason = 1

        equity_curve[i] = equity_val

    return (
        equity_curve, current_pos, entry_px, entry_dt, entry_pos, running_M, running_m, is_carried_flag, rebal_c, rebal_cost, entry_reason,
        out_en_dt[:trade_idx], out_ex_dt[:trade_idx], out_en_px[:trade_idx], out_pos[:trade_idx],
        out_ex_px[:trade_idx], out_rets[:trade_idx], out_days[:trade_idx], out_en_rs[:trade_idx], out_ex_rs[:trade_idx], out_cross[:trade_idx]
    )


def decode_exit_reasons(reasons_int: int) -> str | None:
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

def decode_entry_reason(reason_int: int) -> str | None:
    if reason_int == 1:
        return "BREAKOUT & FILTER"
    return None


# ============================
# Strategy Engine Wrapper
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
) -> tuple[pd.DataFrame | None, dict[str, float] | None, pd.DataFrame, dict | None]:

    # 1. Przygotowanie danych roboczych
    work_df = df.copy()
    work_df["price"] = work_df[price_col]
    
    has_hl = (
        "Najwyzszy" in work_df.columns
        and "Najnizszy" in work_df.columns
        and not work_df["Najwyzszy"].isna().all()
        and not work_df["Najnizszy"].isna().all()
    )
    if has_hl:
        work_df["high"] = work_df["Najwyzszy"]
        work_df["low"] = work_df["Najnizszy"]
    
    if warmup_df is not None:
        w_df = warmup_df.copy()
        w_df["price"] = w_df[price_col]
        if has_hl:
            w_df["high"] = w_df.get("Najwyzszy", w_df["price"])
            w_df["low"] = w_df.get("Najnizszy", w_df["price"])
        w_df["_warmup"] = True
        work_df["_warmup"] = False
        work_df = pd.concat(objs=[w_df, work_df])
    else:
        work_df["_warmup"] = False

    # 2. Obliczanie wskaźników
    work_df["ret"] = work_df["price"].pct_change()
    work_df["vol"] = work_df["ret"].rolling(window=vol_window).std() * np.sqrt(252)
    work_df["ma_fast"] = work_df["price"].rolling(window=fast).mean().shift(periods=1)
    work_df["ma_slow"] = work_df["price"].rolling(window=slow).mean().shift(periods=1)
    work_df["trend"] = (work_df["ma_fast"] > work_df["ma_slow"]).astype(dtype=int)
    
    if filter_mode.startswith("mom"):
        work_df["MOM"] = compute_momentum(
            series=work_df["price"], 
            lookback=mom_lookback, 
            blend=(filter_mode == "mom_blend")
        ).shift(periods=1)
    else:
        work_df["MOM"] = 1.0
    
    # ATR Fix: Mnożnik * 100.0 z powrotem dodany - to zapewnia, że N_atr dziesiętny zachowuje odpowiednią moc stopa
    if has_hl:
        prev_close_prices = work_df["price"].shift(periods=1)
        true_range = np.maximum(work_df["high"], prev_close_prices) - np.minimum(work_df["low"], prev_close_prices)
        work_df["relative_tr"] = true_range / prev_close_prices
        work_df["atr"] = work_df["relative_tr"].rolling(window=atr_window).mean().shift(periods=1) * 100.0
    else:
        work_df["atr"] = (work_df["price"].diff().abs() / work_df["price"].shift(periods=1)).rolling(window=atr_window).mean().shift(periods=1) * 100.0

    if cash_df is not None:
        c_ret = prepare_cash_returns(cash_df=cash_df, price_col=price_col)["cash_ret"]
        work_df = work_df.merge(right=c_ret, left_index=True, right_index=True, how="left").ffill()
    else:
        work_df["cash_ret"] = safe_rate / 252.0

    if work_df["cash_ret"].isna().all():
        work_df["cash_ret"] = safe_rate / 252.0

    oos_cash = work_df.loc[~work_df["_warmup"], "cash_ret"]
    if len(oos_cash) > 0 and oos_cash.notna().any():
        cumulative_cash = (1 + oos_cash).prod()
        n_years_cash = max(len(oos_cash) / 252, 0.01)
        rf_rate = cumulative_cash ** (1 / n_years_cash) - 1
    else:
        rf_rate = safe_rate

    work_df.dropna(subset=["price", "ret", "ma_slow", "atr"], inplace=True)
    if work_df.empty:
        return None, None, pd.DataFrame(), None

    # 3. Przygotowanie argumentów dla JIT
    gate_arr = entry_gate.reindex(index=work_df.index).ffill().fillna(value=1).to_numpy(dtype=int) if entry_gate is not None else np.ones(shape=len(work_df), dtype=int)
    fund_arr = fund_signal.reindex(index=work_df.index).ffill().fillna(value=1).to_numpy(dtype=int) if fund_signal is not None else np.ones(shape=len(work_df), dtype=int)
    
    f_map = {"ma": 0, "mom": 1, "mom_blend": 2, "fund": 3}
    p_map = {"full": 0, "vol_dynamic": 1, "vol_entry": 2}
    
    # 64-bit Timestamp array konwertowany explicite z datetime64[ns]
    dates_arr = work_df.index.astype(dtype="datetime64[ns]").astype(dtype=np.int64)
    
    # Stan początkowy
    i_pos = float(initial_state.get("position", 0.0)) if initial_state else 0.0
    i_px  = float(initial_state.get("entry_price", np.nan)) if initial_state and initial_state.get("entry_price") is not None else np.nan
    i_dt  = int(pd.Timestamp(initial_state["entry_date"]).value) if (initial_state and initial_state.get("entry_date") is not None) else 0
    i_epos = float(initial_state.get("entry_pos", np.nan)) if initial_state and initial_state.get("entry_pos") is not None else np.nan
    i_M   = float(initial_state.get("M", np.nan)) if initial_state else np.nan
    i_m   = float(initial_state.get("m", np.nan)) if initial_state else np.nan
    i_rc  = int(initial_state.get("rebal_count", 0)) if initial_state else 0
    i_rct = float(initial_state.get("rebal_cost_total", 0.0)) if initial_state else 0.0
    i_car = bool(initial_state)
    i_rsn = 1 if initial_state and initial_state.get("entry_reason") == "BREAKOUT & FILTER" else (1 if i_pos > 0.0 else 0)

    # 30 Argumentów (Idealnie dopasowane z JIT)
    numba_args = (
        work_df["price"].to_numpy(dtype=np.float64),
        work_df["ret"].to_numpy(dtype=np.float64),
        work_df["cash_ret"].to_numpy(dtype=np.float64),
        work_df["trend"].to_numpy(dtype=np.int64),
        work_df["MOM"].to_numpy(dtype=np.float64),
        work_df["vol"].to_numpy(dtype=np.float64),
        work_df["atr"].to_numpy(dtype=np.float64),
        work_df["_warmup"].to_numpy(dtype=np.bool_),
        gate_arr,
        fund_arr,
        dates_arr,
        f_map.get(filter_mode, 0),
        p_map.get(position_mode, 2),
        float(stop_loss),
        float(target_vol) if target_vol is not None else 0.10,
        float(max_leverage),
        bool(use_atr_stop),
        float(N_atr),
        float(X),
        float(Y),
        i_pos,
        i_px,
        i_dt,
        i_M,
        i_m,
        i_rc,
        i_rct,
        i_car
    )

    trades_df = pd.DataFrame()
    resulting_end_state = None

    if engine_mode == "numba_light":
        eq_curve, p_f, px_f, dt_f, epos_f, M_f, m_f, rc_f, rct_f = run_numba_light(*numba_args, i_rsn)
        if p_f > 0.0:
            resulting_end_state = {
                "position": p_f, "entry_price": px_f, "entry_date": pd.Timestamp(dt_f, unit='ns'), 
                "entry_pos": epos_f, "M": M_f, "m": m_f, "rebal_count": rc_f, "rebal_cost_total": rct_f,
                "entry_reason": "BREAKOUT & FILTER"
            }
            
    elif engine_mode == "numba_full":
        (
            eq_curve, p_f, px_f, dt_f, epos_f, M_f, m_f, car_f, rc_f, rct_f, rsn_f,
            en_dts, ex_dts, en_pxs, poss, ex_pxs, rets, days, en_rss, ex_rss, cross
        ) = run_numba_full(*numba_args, i_rsn)
        
        trades_list =[]
        for j in range(len(en_dts)):
            trades_list.append({
                "EntryDate": pd.Timestamp(en_dts[j], unit='ns'),
                "ExitDate": pd.Timestamp(ex_dts[j], unit='ns'),
                "EntryPrice": en_pxs[j],
                "Position": poss[j],
                "ExitPrice": ex_pxs[j],
                "Return": rets[j],
                "Days": days[j],
                "Entry Reason": decode_entry_reason(reason_int=en_rss[j]),
                "Exit Reason": decode_exit_reasons(reasons_int=ex_rss[j]),
                "CrossWindow": bool(cross[j])
            })
        trades_df = pd.DataFrame(data=trades_list)
        
        if p_f > 0.0 and not np.isnan(px_f):
            resulting_end_state = {
                "position": p_f, "entry_price": px_f, "entry_date": pd.Timestamp(dt_f, unit='ns'), 
                "entry_pos": epos_f, "M": M_f, "m": m_f, "rebal_count": rc_f, "rebal_cost_total": rct_f,
                "entry_reason": decode_entry_reason(reason_int=rsn_f)
            }
            
            last_date_ns = dates_arr[-1]
            last_carry = {
                "EntryDate": pd.Timestamp(dt_f, unit='ns'),
                "ExitDate": pd.Timestamp(last_date_ns, unit='ns'),
                "EntryPrice": px_f,
                "Position": epos_f,
                "ExitPrice": work_df["price"].iloc[-1],
                "Return": (work_df["price"].iloc[-1] / px_f - 1.0),
                "Days": (last_date_ns - dt_f) // 86400000000000,
                "Entry Reason": decode_entry_reason(reason_int=rsn_f),
                "Exit Reason": "CARRY",
                "CrossWindow": car_f
            }
            trades_df = pd.concat(objs=[trades_df, pd.DataFrame(data=[last_carry])], ignore_index=True)

    # Budowa wyniku OOS
    work_df["equity"] = eq_curve
    result_df = work_df[~work_df["_warmup"]].copy()
    result_df.drop(columns=["_warmup"], inplace=True)
    
    if not result_df.empty:
        result_df["equity"] /= result_df["equity"].iloc[0]

    return result_df, compute_metrics(equity=result_df["equity"], risk_free_rate=rf_rate), trades_df, resulting_end_state

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
    use_atr_stop: bool = False,
    N_atr: float = 3.0,
    atr_window: int = 20,
    engine_mode: str = "numba_light"
) -> tuple[tuple, float] | None:

    train_fund_signal = None
    if filter_mode == "fund" and fund_params is not None and funds_df is not None:
        funds_train = funds_df.loc[(funds_df.index >= train_start) & (funds_df.index < train_end)]
        train_fund_signal = compute_fund_breadth_signal(
            funds_df=funds_train,
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

    if X_grid is None:
        X_grid =[0.08, 0.10, 0.12, 0.15, 0.20]
    if Y_grid is None:
        Y_grid =[0.02, 0.03, 0.05, 0.07, 0.10]
    if fast_grid is None:
        fast_grid = [50, 75, 100]
    if slow_grid is None:
        slow_grid =[150, 200, 250]
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
        msg=f"walk_forward received data from {df.index.min()} to {data_end} ({len(df)} rows)", 
    )
    logging.info(msg=f"Objective function: {objective}")
    logging.info(
        msg=f"Trailing stop mode: {'ATR-scaled (Chandelier)' if use_atr_stop else 'fixed percentage'}  (ATR window={atr_window})",
    )

    oos_equity_slices = []
    results =[]
    all_oos_trades =[]

    start = df.index.min()
    carry_state = None

    if filter_modes_override is not None:
        logging.info(msg=f"filter_modes overridden to: {filter_modes_override}")

    while True:
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
            logging.info(msg="Breaking — train or test empty")
            break

        cash_train = cash_df.loc[(cash_df.index >= train_start) & (cash_df.index < train_end)]

        gate_train = None
        if entry_gate_series is not None:
            gate_train = (
                entry_gate_series.reindex(
                    index=train.index,
                    method="ffill",
                )
                .fillna(value=1)
                .astype(dtype=int)
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
            mom_lb_iter = mom_lookback_grid if filter_mode == "mom" else [252]
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
                                for tv in tv_grid if selected_mode != "full" else[0.10]:
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
                    msg=f"Grid search completed using {label} backend ({n_jobs_inner} jobs).",
                )
                break

            except Exception as e:
                logging.warning(
                    msg=f"Grid search backend '{label}' failed: {e} — trying next option.",
                )
                results_list = None

        if results_list is None:
            logging.error(msg="All grid search backends failed. Skipping window.")
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
            stability = neighbour_mean(key=key, scores=same_mode_scores, stop_grid=stop_grid, Y_grid=Y_grid)
            combined = 0.5 * raw_score + 0.5 * stability
            if combined > best_score:
                best_score = combined
                best_raw_score = raw_score
                fund_idx_selected = key[1]

                best_params = {
                    "filter_mode": key[0],
                    "fund_idx": fund_idx_selected,
                    "fund_params": (
                        fund_params_grid[fund_idx_selected]
                        if fund_idx_selected is not None and fund_params_grid is not None
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
                msg=f"Window {train_start.date()}: best raw_{objective}={best_raw_score:.4f} | penalised_{objective}={best_score:.4f} | filter={best_params['filter_mode']} | "
                    f"{stop_label} Y={best_params['Y']:.2f} fast={best_params['fast']} slow={best_params['slow']} sl={best_params['stop_loss']:.2f} tv={best_params.get('target_vol', 'N/A')} mom_lookback={best_params['mom_lookback']}"
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
            oos_fund_signal = compute_fund_breadth_signal(
                funds_df=funds_warmup_and_test,
                **best_params["fund_params"],
            ).loc[lambda x: x.index >= train_end]

        gate_oos = None
        if entry_gate_series is not None:
            gate_oos = (
                entry_gate_series.reindex(
                    index=test.index,
                    method="ffill",
                )
                .fillna(value=1)
                .astype(dtype=int)
            )

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
            msg="Position still open at end of final window. Last CARRY trade represents the open P&L.",
        )

    if not oos_equity_slices:
        logging.warning(msg="Walk-forward produced no OOS results.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    oos_equity = pd.concat(objs=oos_equity_slices).sort_index()
    results_df = pd.DataFrame(data=results)
    oos_trades_df = pd.concat(objs=all_oos_trades) if all_oos_trades else pd.DataFrame()

    return oos_equity, results_df, oos_trades_df