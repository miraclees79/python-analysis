# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:41:19 2026

@author: adamg
"""

import numpy as np
import pandas as pd
import logging
from moj_system.core.indicators import compute_momentum, compute_metrics

def calc_position(vol: float, position_mode: str, target_vol: float, max_leverage: float) -> float:
    if position_mode == "full":
        return 1.0
    if pd.notna(vol) and vol > 0:
        pos = target_vol / vol
    else:
        pos = 1.0
    return min(pos, max_leverage)

def run_strategy_with_trades(
    df: pd.DataFrame,
    price_col: str = "price",
    X: float = 0.1, Y: float = 0.1, stop_loss: float = 0.1,
    fast: int = 50, slow: int = 200, vol_window: int = 20, target_vol: float = 0.10,
    max_leverage: float = 1.0, position_mode: str = "vol_entry", filter_mode: str = "ma",
    mom_lookback: int = 252, cash_df: pd.DataFrame = None, safe_rate: float = 0.0,
    initial_state: dict = None, warmup_df: pd.DataFrame = None, fund_signal: pd.Series = None,
    entry_gate: pd.Series = None, use_atr_stop: bool = False, N_atr: float = 3.0,
    atr_window: int = 20, fast_mode: bool = True
):
    """
    Uruchamia strategię śledzenia trendu (Trend-Following).
    Wspiera stałe trailing stopy (X) oraz wskaźniki adaptacyjne ATR-Chandelier (N_atr).
    Zoptymalizowane poprzez szybkie wektory numpy (fast_mode).
    """
    df = df.copy()
    df["price"] = df[price_col]
    
    # Detekcja HL
    has_hl = ("Najwyzszy" in df.columns and "Najnizszy" in df.columns 
              and not df["Najwyzszy"].isna().all() and not df["Najnizszy"].isna().all())
    if has_hl:
        df["high"] = df["Najwyzszy"]
        df["low"]  = df["Najnizszy"]    
        
    if warmup_df is not None:
        warmup = warmup_df.copy()
        warmup["price"] = warmup[price_col]
        if has_hl and "Najwyzszy" in warmup.columns and "Najnizszy" in warmup.columns:
            warmup["high"] = warmup["Najwyzszy"]
            warmup["low"]  = warmup["Najnizszy"]
        warmup["_warmup"] = True
        df["_warmup"] = False
        df = pd.concat([warmup, df])
    else:
        df["_warmup"] = False
    
    gate_aligned = entry_gate.reindex(df.index, method="ffill").fillna(1).astype(int) if entry_gate is not None else None
    test_start = df[~df["_warmup"]].index[0]

    # Przygotowanie zwrotu cashowego
    if cash_df is not None:
        cash = cash_df.copy()
        cash["cash_price"] = cash[price_col] if price_col in cash.columns else cash.iloc[:, 0]
        cash["cash_ret"] = cash["cash_price"].pct_change()
        cash = cash[["cash_ret"]].dropna()
        df = df.merge(cash, left_index=True, right_index=True, how="left")
        if df["cash_ret"].isna().any():
            df["cash_ret"] = df["cash_ret"].ffill()
    else:
        df["cash_ret"] = safe_rate / 252

    if df["cash_ret"].isna().all(): df["cash_ret"] = safe_rate / 252

    # Obliczenia bazowe
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

    # Obliczenia ATR 
    if has_hl:
        prev_close = df["price"].shift(1)
        tr = np.maximum(df["high"], prev_close) - np.minimum(df["low"], prev_close)
        df["relative_tr"] = tr / prev_close
        df["atr"] = df["relative_tr"].rolling(atr_window).mean().shift(1) * 100
    else:
        df["atr"] = (df["price"].diff().abs() / df["price"].shift(1)).rolling(atr_window).mean().shift(1) * 100
        
    df.dropna(inplace=True)

    # Inicjalizacja Stanu
    equity, equity_curve, trades = 1.0, [],[]
    position = entry_price = entry_date = entry_reason = entry_pos = M = m = None
    entry_carried = False 
    rebal_count, rebal_cost_total = 0, 0.0    
    
    if initial_state is not None:
        position    = initial_state["position"]
        entry_price = initial_state["entry_price"]
        entry_date  = initial_state["entry_date"]
        M           = initial_state["M"]
        m           = initial_state["m"]
        entry_carried = True

    if fast_mode:
        _prices    = df["price"].to_numpy()
        _rets      = df["ret"].to_numpy()
        _cash_rets = df["cash_ret"].to_numpy()
        _trends    = df["trend"].to_numpy()
        _moms      = df["MOM"].to_numpy()
        _vols      = df["vol"].to_numpy()
        _atrs      = df["atr"].to_numpy()
        _warmups   = df["_warmup"].to_numpy(dtype=bool)
        _gate_vals = gate_aligned.reindex(df.index).fillna(1).to_numpy().astype(int) if gate_aligned is not None else None
        _index = df.index

        for _n in range(len(_prices)):
            i, price, ret, cash_ret = _index[_n], float(_prices[_n]), float(_rets[_n]), float(_cash_rets[_n])
            trend, mom, vol, atr_val = int(_trends[_n]), float(_moms[_n]), float(_vols[_n]), float(_atrs[_n])

            filter_on = (mom > 0) if filter_mode in["mom", "mom_blend"] else (trend == 1)

            if _warmups[_n]:
                equity_curve.append(equity)
                continue

            if position and position > 0:
                equity *= (1 + position * ret + (1 - position) * cash_ret)
            else:
                equity *= (1 + cash_ret)

            exit_reasons =[]

            # Stop Loss & Trailing
            if position and position > 0:
                if (price - entry_price) / entry_price < -stop_loss:
                    exit_reasons.append("ABSOLUTE_STOP")

                M = max(M, price) if M is not None else price
                
                if use_atr_stop:
                    trail_breached = (price < M * (1 - N_atr * atr_val)) if np.isfinite(atr_val) and atr_val > 0 else False
                else:
                    trail_breached = price < (1 - X) * M

                if trail_breached:
                    if "ABSOLUTE_STOP" not in exit_reasons: exit_reasons.append("TRAIL_STOP")
                elif not filter_on:
                    exit_reasons.append("FILTER_EXIT")

            exit_reason = " + ".join(exit_reasons) if exit_reasons else None

            # Wyjście
            if position and position > 0 and exit_reason:
                trade_ret = price / entry_price - 1 - 0.0020
                trades.append({
                    "EntryDate": entry_date, "ExitDate": i, "EntryPrice": entry_price,
                    "ExitPrice": price, "Return": trade_ret, "Days": (i - entry_date).days,
                    "Entry Reason": entry_reason, "Exit Reason": exit_reason, "CrossWindow": entry_carried
                })
                position = entry_price = entry_date = entry_reason = M = m = entry_pos = None
                entry_carried = False

            # Wejście
            if not position:
                m = price if m is None else min(m, price)
                gate_allows = (_gate_vals is None or int(_gate_vals[_n]) == 1)
                if (price > (1 + Y) * m) and filter_on and gate_allows:
                    entry_reason  = "BREAKOUT & FILTER"
                    position      = calc_position(vol, position_mode, target_vol, max_leverage)
                    entry_price   = price
                    entry_date    = i
                    M             = price
                    entry_carried = False

            equity_curve.append(equity)

    df["equity"] = equity_curve
    df = df[~df["_warmup"]].copy()
    first_val = df["equity"].iloc[0]
    if first_val != 0: df["equity"] = df["equity"] / first_val

    # rf_rate calculation handled safely outside the loop
    rf_rate = safe_rate
    return df, compute_metrics(df["equity"], risk_free_rate=rf_rate), pd.DataFrame(trades), {"position": position, "entry_price": entry_price, "entry_date": entry_date, "M": M, "m": m}