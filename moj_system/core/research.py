# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:28:31 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/core/research.py
===========================
Core engine for batch research, window generation, and strategy comparison.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple

def get_common_oos_start(assets_data: dict, window_configs: List[Tuple[int, int]]) -> pd.Timestamp:
    """
    Calculates the latest possible start date for Out-Of-Sample period
    to ensure all tested configurations cover the exact same time range.
    """
    latest_starts = []
    
    for df in assets_data.values():
        data_start = df.index.min()
        for train_y, _ in window_configs:
            # Each config needs at least 'train_y' of history
            potential_start = data_start + pd.DateOffset(years=train_y)
            latest_starts.append(potential_start)
            
    # The common start is the latest of all required starts
    common_start = max(latest_starts)
    logging.info(f"Calculated Common OOS Start Date: {common_start.date()}")
    return common_start

def rank_research_results(results_df: pd.DataFrame, objective: str = "CalMAR") -> pd.DataFrame:
    """Ranks sweep results based on primary objective and robustness."""
    if results_df.empty: return results_df
    
    # Simple ranking logic - can be extended with 'Robustness' scores
    sorted_df = results_df.sort_values(by=objective, ascending=False).reset_index(drop=True)
    return sorted_df



# --- IMPORTS FOR REGIME ANALYSIS ---
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- REGIME DECOMPOSITION FUNCTIONS ---

def prepare_regime_inputs(df: pd.DataFrame, wf_results: pd.DataFrame, wf_equity: pd.Series, bh_equity: pd.Series) -> dict:
    """Przygotowuje zsynchronizowane serie danych dla analizy reżimów w okresie OOS."""
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

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=20) -> tuple:
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

def label_regime_adx(close: pd.Series, high: pd.Series, low: pd.Series, period=20, trend_thresh=25, chop_thresh=20) -> pd.Series:
    adx, plus_di, minus_di = compute_adx(high, low, close, period)
    regime = pd.Series("sideways", index=close.index)
    regime[adx > trend_thresh] = np.where(
        plus_di[adx > trend_thresh] > minus_di[adx > trend_thresh], "uptrend", "downtrend"
    )
    regime[adx < chop_thresh] = "sideways"
    return regime.rename("regime_adx")

def label_regime_momentum(close: pd.Series, window=63, thresh=0.03) -> pd.Series:
    ret = close.pct_change(window)
    regime = pd.Series("sideways", index=close.index)
    regime[ret >  thresh] = "uptrend"
    regime[ret < -thresh] = "downtrend"
    return regime.rename("regime_mom")

def label_regime_vol(close: pd.Series, window=21, hi_pct=0.67, lo_pct=0.33) -> pd.Series:
    rv = close.pct_change().rolling(window).std() * np.sqrt(252)
    rolling_med = rv.rolling(252, min_periods=60).median()
    regime = pd.Series("normal_vol", index=close.index)
    regime[rv > rolling_med * 1.3] = "high_vol"
    regime[rv < rolling_med * 0.7] = "low_vol"
    return regime.rename("regime_vol")

def regime_stats(daily_returns_strat: pd.Series, daily_returns_bh: pd.Series, regime_series: pd.Series, label="regime") -> pd.DataFrame:
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

        def ann_return(r): return (1 + r).prod() ** (ann / max(1, len(r))) - 1
        def ann_vol(r): return r.std() * np.sqrt(ann)
        def max_dd(r):
            cum = (1 + r).cumprod()
            return (cum / cum.cummax() - 1).min()

        strat_vol = ann_vol(grp.strat)
        
        rows.append({
            label          : reg,
            "days"         : n_days,
            "pct_time"     : round(pct_time, 1),
            "strat_cagr"   : round(ann_return(grp.strat) * 100, 2),
            "bh_cagr"      : round(ann_return(grp.bh)    * 100, 2),
            "strat_vol"    : round(strat_vol * 100, 2),
            "bh_vol"       : round(ann_vol(grp.bh)    * 100, 2),
            "strat_maxdd"  : round(max_dd(grp.strat)   * 100, 2),
            "bh_maxdd"     : round(max_dd(grp.bh)      * 100, 2),
            "strat_sharpe" : round(ann_return(grp.strat) / strat_vol, 3) if strat_vol > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index(label)

def regime_transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
    s = regime_series.dropna()
    regimes = sorted(s.unique())
    mat = pd.DataFrame(0.0, index=regimes, columns=regimes)
    for a, b in zip(s.iloc[:-1], s.iloc[1:]):
        mat.loc[a, b] += 1
    mat = mat.div(mat.sum(axis=1), axis=0)
    return mat.round(3)

def run_regime_decomposition(inputs: dict, generate_plots: bool = False) -> dict:
    """
    Główna funkcja wykonująca dekompozycję reżimów (ADX, Mom, Vol).
    Jeśli generate_plots=False (np. w Sweep), zwraca tylko statystyki (bardzo szybkie).
    """
    results = {}
    close, high, low = inputs["close"], inputs["high"], inputs["low"]
    ret_strat, ret_bh = inputs["daily_returns_strat"], inputs["daily_returns_bh"]
    
    for name, regime_fn in [
        ("ADX trend",  lambda: label_regime_adx(close, high, low)),
        ("Momentum",   lambda: label_regime_momentum(close)),
        ("Volatility", lambda: label_regime_vol(close)),
    ]:
        regime = regime_fn().reindex(ret_strat.index)
        stats  = regime_stats(ret_strat, ret_bh, regime, label="regime")
        trans  = regime_transition_matrix(regime)
        
        res_dict = {
            "stats": stats,
            "transitions": trans
        }
        
        # Opcjonalne generowanie wykresów
        if generate_plots:
            # (Tutaj wklej definicje funkcji plot_regime_overlay i plot_regime_bar_comparison z oryginalnego pliku,
            #  jeśli chcesz, by skrypt je zapisywał na dysk). 
            pass 
            
        results[name] = res_dict
        
    return results

def extract_flat_regime_stats(regime_results: dict) -> dict:
    """Spłaszcza wielowymiarowy słownik wyników reżimów do jednowymiarowego słownika na potrzeby tabeli (Sweep)."""
    out = {}
    if not regime_results: return out

    # ADX trend
    adx_stats = regime_results.get("ADX trend", {}).get("stats")
    if adx_stats is not None and not adx_stats.empty:
        for reg in ["uptrend", "downtrend", "sideways"]:
            if reg in adx_stats.index:
                r = adx_stats.loc[reg]
                out[f"adx_{reg}_strat_cagr"]  = float(r.get("strat_cagr", np.nan))
                out[f"adx_{reg}_bh_cagr"]      = float(r.get("bh_cagr", np.nan))
            else:
                out[f"adx_{reg}_strat_cagr"] = np.nan
                out[f"adx_{reg}_bh_cagr"] = np.nan

    # Volatility
    vol_stats = regime_results.get("Volatility", {}).get("stats")
    if vol_stats is not None and not vol_stats.empty:
        for reg in ["high_vol", "normal_vol", "low_vol"]:
            if reg in vol_stats.index:
                r = vol_stats.loc[reg]
                out[f"vol_{reg}_strat_cagr"] = float(r.get("strat_cagr", np.nan))
            else:
                out[f"vol_{reg}_strat_cagr"] = np.nan

    return out
