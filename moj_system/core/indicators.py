# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:40:48 2026

@author: adamg
"""

import numpy as np
import pandas as pd

# ============================================================
# Performance Metrics (CAGR, Sharpe, MaxDD itp.)
# ============================================================

def compute_metrics(equity: pd.Series, risk_free_rate: float = 0, freq: int = 252) -> dict:
    """Oblicza kluczowe metryki z krzywej kapitału (equity curve)."""
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return {}

    years = len(ret) / freq
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(freq)

    excess_return = cagr - risk_free_rate
    sharpe = excess_return / vol if vol > 0 else 0.0
    
    daily_rf   = (1 + risk_free_rate) ** (1 / freq) - 1
    daily_rets = equity.pct_change().dropna()
    downside   = daily_rets[daily_rets < daily_rf] - daily_rf
    downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(freq) if len(downside) > 0 else 0.0
    
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
        "CalMAR": calmar
    }

def compute_buy_and_hold(df: pd.DataFrame, price_col: str = "Zamkniecie", start=None, end=None) -> tuple:
    """Zwraca krzywą equity B&H oraz metryki dla waloru."""
    bh = df[price_col].copy()
    if start is not None: bh = bh.loc[bh.index >= start]
    if end is not None:   bh = bh.loc[bh.index <= end]
    
    bh_equity = bh / bh.iloc[0]
    bh_metrics = compute_metrics(bh_equity)
    return bh_equity, {k: float(v) for k, v in bh_metrics.items()}

# ============================================================
# Technical Indicators (MOM, ATR, ADX)
# ============================================================

def compute_momentum(series: pd.Series, lookback: int = 252, skip: int = 21, 
                     blend: bool = False, blend_lookbacks=(21, 63, 126, 252), blend_skip=5) -> pd.Series:
    """Oblicza wskaźnik Momentum z opcją blendowania (uśredniania z wielu horyzontów)."""
    if not blend:
        return series.shift(skip) / series.shift(lookback) - 1

    signals =[]
    for lb in blend_lookbacks:
        sig = series.shift(blend_skip) / series.shift(lb) - 1
        signals.append(sig)

    blended = pd.concat(signals, axis=1).mean(axis=1)
    blended.name = series.name
    return blended

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> tuple:
    """Zwraca ADX, +DI, -DI na potrzeby analizy reżimu rynkowego."""
    delta_high = high.diff()
    delta_low  = -low.diff()
    plus_dm  = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0.0)
    minus_dm = np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0.0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=close.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr
    
    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di



