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
    """Przygotowuje dane zsynchronizowane z faktycznym zakresem dostarczonej krzywej kapitału."""
    
    # Używamy zakresu dat z krzywej kapitału (wf_equity), a nie z okien WF
    # To rozwiązuje problem przesunięcia common_start
    common_idx = wf_equity.index.intersection(bh_equity.index).intersection(df.index)
    
    if len(common_idx) < 30:
        logging.warning("Za mało wspólnych dat dla analizy reżimów.")
        return {}

    close = df.loc[common_idx, "Zamkniecie"].copy()
    # Obsługa braku kolumn High/Low (fallback na Close)
    high  = df.loc[common_idx, "Najwyzszy"].copy() if "Najwyzszy" in df.columns else close
    low   = df.loc[common_idx, "Najnizszy"].copy() if "Najnizszy" in df.columns else close

    equity_strat = wf_equity.reindex(common_idx).ffill()
    equity_bh    = bh_equity.reindex(common_idx).ffill()

    daily_returns_strat = equity_strat.pct_change().dropna()
    daily_returns_bh    = equity_bh.pct_change().dropna()

    # Finalna synchronizacja do zwrotów
    final_idx = daily_returns_strat.index
    return dict(
        close               = close.reindex(final_idx),
        high                = high.reindex(final_idx),
        low                 = low.reindex(final_idx),
        daily_returns_strat = daily_returns_strat,
        daily_returns_bh    = daily_returns_bh,
        equity_strat        = equity_strat.reindex(final_idx),
        equity_bh           = equity_bh.reindex(final_idx)
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
    """Zawsze zwraca pełny zestaw kluczy, nawet jeśli dane są puste (NaN)."""
    out = {}
    metrics = ["strat_cagr", "bh_cagr"]
    regimes = [
        "adx_uptrend", "adx_downtrend", "adx_sideways",
        "vol_high_vol", "vol_normal_vol", "vol_low_vol"
    ]
    
    # Inicjalizacja Nanami, aby kolumny zawsze istniały w CSV
    for r in regimes:
        for m in metrics:
            out[f"{r}_{m}"] = np.nan

    if not regime_results:
        return out

    # Mapowanie wyników ADX
    adx_stats = regime_results.get("ADX trend", {}).get("stats")
    if adx_stats is not None and not adx_stats.empty:
        for reg in ["uptrend", "downtrend", "sideways"]:
            if reg in adx_stats.index:
                out[f"adx_{reg}_strat_cagr"] = adx_stats.loc[reg, "strat_cagr"]
                out[f"adx_{reg}_bh_cagr"] = adx_stats.loc[reg, "bh_cagr"]

    # Mapowanie wyników Volatility
    vol_stats = regime_results.get("Volatility", {}).get("stats")
    if vol_stats is not None and not vol_stats.empty:
        for reg in ["high_vol", "normal_vol", "low_vol"]:
            if reg in vol_stats.index:
                out[f"vol_{reg}_strat_cagr"] = vol_stats.loc[reg, "strat_cagr"]
                out[f"vol_{reg}_bh_cagr"] = vol_stats.loc[reg, "bh_cagr"]
                
    return out 
    
def print_live_regime_report(regime_metrics: dict):
    if not regime_metrics: return
    logging.info("  --- REGIME ANALYSIS (CAGR %) ---")
    header = f"  {'Regime':<15} | {'Strategy':>10} | {'Buy&Hold':>10}"
    logging.info(header)
    logging.info("  " + "-" * len(header))
    regimes = [("ADX Uptrend", "adx_uptrend"), ("ADX Downtrend", "adx_downtrend"),
               ("ADX Sideways", "adx_sideways"), ("Vol High", "vol_high_vol"),
               ("Vol Normal", "vol_normal_vol"), ("Vol Low", "vol_low_vol")]
    for label, prefix in regimes:
        strat_val = regime_metrics.get(f"{prefix}_strat_cagr", pd.NA)
        bh_val = regime_metrics.get(f"{prefix}_bh_cagr", pd.NA)
        s_str = f"{strat_val:.2f}%" if pd.notna(strat_val) else "N/A"
        b_str = f"{bh_val:.2f}%" if pd.notna(bh_val) else "N/A"
        logging.info(f"  {label:<15} | {s_str:>10} | {b_str:>10}")
    logging.info("  --------------------------------")


# moj_system/core/research.py (dodaj na końcu pliku)

def get_current_adx_regime(df: pd.DataFrame) -> str:
    """
    Oblicza i zwraca bieżący reżim rynkowy (ADX) na podstawie najnowszych danych.

    Używa funkcji label_regime_adx do wygenerowania serii reżimów i zwraca
    ostatnią dostępną wartość.

    Args:
        df: DataFrame z danymi cenowymi, musi zawierać kolumny 
            'Zamkniecie', 'Najwyzszy', 'Najnizszy'.

    Returns:
        Nazwa bieżącego reżimu ('uptrend', 'downtrend', 'sideways') lub 'N/A', 
        jeśli danych jest za mało lub wystąpi błąd.
    """
    if df is None or len(df) < 50:  # ADX potrzebuje historii do obliczeń
        return "N/A"
    
    # Upewnijmy się, że mamy potrzebne kolumny
    required_cols = ["Zamkniecie", "Najwyzszy", "Najnizszy"]
    if not all(col in df.columns for col in required_cols):
        logging.warning("Brak kolumn High/Low w danych, reżim ADX może być mniej dokładny.")
        close = df["Zamkniecie"]
        high = df.get("Najwyzszy", close) # Fallback na 'Zamkniecie'
        low = df.get("Najnizszy", close)  # Fallback na 'Zamkniecie'
    else:
        close, high, low = df["Zamkniecie"], df["Najwyzszy"], df["Najnizszy"]

    try:
        regime_series = label_regime_adx(close=close, high=high, low=low)
        # Zwracamy ostatnią wartość z serii
        return regime_series.iloc[-1]
    except Exception as e:
        logging.error(f"Błąd podczas obliczania reżimu ADX: {e}")
        return "N/A"