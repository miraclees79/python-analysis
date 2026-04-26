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

def prepare_regime_inputs(df, wf_results, wf_equity, bh_equity):
    """
    Przygotowuje zsynchronizowane serie danych dla analizy reżimów.
    Odporna na zduplikowane etykiety dat.
    """
    # --- PANCERNE USUWANIE DUPLIKATÓW ---
    if not wf_equity.index.is_unique:
        duplicates_count = wf_equity.index.duplicated().sum()
        logging.warning(msg=f"Found {duplicates_count} duplicate dates in wf_equity. Removing...")
        wf_equity = wf_equity.loc[~wf_equity.index.duplicated(keep='last')]
        
    if not bh_equity.index.is_unique:
        bh_duplicates_count = bh_equity.index.duplicated().sum()
        logging.warning(msg=f"Found {bh_duplicates_count} duplicate dates in bh_equity. Removing...")
        bh_equity = bh_equity.loc[~bh_equity.index.duplicated(keep='last')]

    # Używamy zakresu dat z krzywej kapitału
    common_idx = wf_equity.index.intersection(other=bh_equity.index).intersection(other=df.index)
    
    if len(common_idx) < 30:
        logging.warning(msg="Za mało wspólnych dat dla analizy reżimów.")
        return {}

    close = df.loc[common_idx, "Zamkniecie"].copy()
    high  = df.loc[common_idx, "Najwyzszy"].copy() if "Najwyzszy" in df.columns else close
    low   = df.loc[common_idx, "Najnizszy"].copy() if "Najnizszy" in df.columns else close

    # Używamy keyword 'index' zamiast 'labels' dla Series/DataFrame
    equity_strat = wf_equity.reindex(index=common_idx).ffill()
    equity_bh    = bh_equity.reindex(index=common_idx).ffill()

    daily_returns_strat = equity_strat.pct_change().dropna()
    daily_returns_bh    = equity_bh.pct_change().dropna()

    final_idx = daily_returns_strat.index
    
    return dict(
        close               = close.reindex(index=final_idx),
        high                = high.reindex(index=final_idx),
        low                 = low.reindex(index=final_idx),
        daily_returns_strat = daily_returns_strat,
        daily_returns_bh    = daily_returns_bh,
        equity_strat        = equity_strat.reindex(index=final_idx),
        equity_bh           = equity_bh.reindex(index=final_idx)
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
        bh_vol = ann_vol(grp.bh)
        
        
        rows.append({
            label          : reg,
            "days"         : n_days,
            "pct_time"     : round(pct_time, 1),
            "strat_cagr"   : round(ann_return(grp.strat) * 100, 2),
            "bh_cagr"      : round(ann_return(grp.bh)    * 100, 2),
            "strat_vol"    : round(strat_vol * 100, 2),
            "bh_vol"       : round(bh_vol    * 100, 2),
            "strat_maxdd"  : round(max_dd(grp.strat)   * 100, 2),
            "bh_maxdd"     : round(max_dd(grp.bh)      * 100, 2),
            "strat_sharpe" : round(ann_return(grp.strat) / strat_vol, 3) if strat_vol > 0 else np.nan,
            "bh_sharpe"    : round(ann_return(grp.bh) / bh_vol, 3) if bh_vol > 0 else np.nan,
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
    # Definicja wszystkich metryk, które chcemy mieć w słowniku/pliku CSV
    metrics = ["strat_cagr", "bh_cagr", "strat_sharpe", "pct_time"]
    regimes = [
        "adx_uptrend", "adx_downtrend", "adx_sideways",
        "vol_high_vol", "vol_normal_vol", "vol_low_vol"
    ]
    
    # Inicjalizacja Nanami, aby kolumny zawsze istniały w strukturze danych
    for r in regimes:
        for m in metrics:
            out[f"{r}_{m}"] = np.nan

    if not regime_results:
        return out

    # Mapowanie wyników ADX trend
    adx_res = regime_results.get("ADX trend", {})
    adx_stats = adx_res.get("stats")
    if adx_stats is not None and not adx_stats.empty:
        for reg in ["uptrend", "downtrend", "sideways"]:
            if reg in adx_stats.index:
                r = adx_stats.loc[reg]
                out[f"adx_{reg}_strat_cagr"]   = round(float(r.get("strat_cagr",   np.nan)), 2)
                out[f"adx_{reg}_bh_cagr"]       = round(float(r.get("bh_cagr",      np.nan)), 2)
                out[f"adx_{reg}_strat_sharpe"]  = round(float(r.get("strat_sharpe", np.nan)), 3)
                out[f"adx_{reg}_pct_time"]      = round(float(r.get("pct_time",     np.nan)), 1)

    # Mapowanie wyników Volatility
    vol_res = regime_results.get("Volatility", {})
    vol_stats = vol_res.get("stats")
    if vol_stats is not None and not vol_stats.empty:
        for reg in ["high_vol", "normal_vol", "low_vol"]:
            if reg in vol_stats.index:
                r = vol_stats.loc[reg]
                out[f"vol_{reg}_strat_cagr"]   = round(float(r.get("strat_cagr",   np.nan)), 2)
                out[f"vol_{reg}_bh_cagr"]       = round(float(r.get("bh_cagr",      np.nan)), 2)
                out[f"vol_{reg}_strat_sharpe"]  = round(float(r.get("strat_sharpe", np.nan)), 3) # Dodano Sharpe
                out[f"vol_{reg}_pct_time"]      = round(float(r.get("pct_time",     np.nan)), 1)
                
    return out
    
def print_live_regime_report(regime_metrics: dict):
    if not regime_metrics: 
        logging.warning("No regime metrics available to print.")
        return
        
    logging.info("  --- DETAILED REGIME ANALYSIS ---")
    # Nowy nagłówek z większą liczbą kolumn
    header = f"  {'Regime':<15} | {'Strat CAGR':>10} | {'B&H CAGR':>10} | {'Sharpe':>8} | {'% Time':>8}"
    logging.info(header)
    logging.info("  " + "-" * len(header))
    
    regimes = [
        ("ADX Uptrend",   "adx_uptrend"), 
        ("ADX Downtrend", "adx_downtrend"),
        ("ADX Sideways",  "adx_sideways"), 
        ("Vol High",      "vol_high_vol"),
        ("Vol Normal",    "vol_normal_vol"), 
        ("Vol Low",       "vol_low_vol")
    ]
    
    for label, prefix in regimes:
        # Pobieranie wartości
        s_cagr   = regime_metrics.get(f"{prefix}_strat_cagr", pd.NA)
        b_cagr   = regime_metrics.get(f"{prefix}_bh_cagr", pd.NA)
        s_sharpe = regime_metrics.get(f"{prefix}_strat_sharpe", pd.NA)
        p_time   = regime_metrics.get(f"{prefix}_pct_time", pd.NA)
        
        # Formatowanie do wyświetlenia
        s_str  = f"{s_cagr:>.2f}%" if pd.notna(s_cagr) else "N/A"
        b_str  = f"{b_cagr:>.2f}%" if pd.notna(b_cagr) else "N/A"
        sh_str = f"{s_sharpe:>.2f}" if pd.notna(s_sharpe) else "N/A"
        t_str  = f"{p_time:>.1f}%" if pd.notna(p_time) else "N/A"
        
        logging.info(f"  {label:<15} | {s_str:>10} | {b_str:>10} | {sh_str:>8} | {t_str:>8}")
        
    logging.info("  " + "-" * len(header))
    
    
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
    
def analyze_production_candidates(results_df: pd.DataFrame, require_bootstrap: bool = False) -> pd.DataFrame:
    """
    Filters sweep results based on mandatory production gates and 
    calculates a weighted Ranking Score.
    """
    if results_df.empty:
        return results_df

    df = results_df.copy()

    # --- 1. MANDATORY GATES (Boolean Filters) ---
    
    # Używamy tyldy (~) jako operatora negacji bitowej dla Series w Pandas.
    # na=True w str.contains sprawia, że puste wartości (NaN) zostaną potraktowane 
    # jako zawierające 'FRA' (czyli zostaną odrzucone po negacji), co jest bezpieczne.
    mask_robust_MC = ~df['MC_Verdict'].str.contains(pat='FRA', case=False, na=True)
    mask_robust_BB = ~df['BB_Verdict'].str.contains(pat='FRA', case=False, na=True)
    
    if require_bootstrap:
        mask_robust = mask_robust_MC & mask_robust_BB
    else:
        # Jeśli nie wymagamy Bootstrapa, sprawdzamy tylko MC
        mask_robust = mask_robust_MC 
    
    # OOS CalMAR > B&H CalMAR
    mask_calmar = df['oos_calmar'] > df['bh_calmar']
    
    # Sideways Alpha > 0
    mask_sideways = (df['adx_sideways_strat_cagr'] - df['adx_sideways_bh_cagr']) > 0
    
    # MaxDD >= -20% (Zwróć uwagę, czy dane są w % (np -15.0) czy frakcjach (np -0.15))
    # Zakładając format z sweep_optimizer (-15.0):
    mask_maxdd = df['oos_maxdd'] >= -0.2

    # Combine Mandatory Gates
    df['is_candidate'] = mask_robust & mask_calmar & mask_sideways & mask_maxdd

    # --- 2. RANKING SCORE CALCULATION ---
    # Używamy .replace(0, np.nan) aby uniknąć ZeroDivisionError
    
    # A. Protection Score: 1 - (strat_down / bh_down)
    df['score_protection'] = 1 - (df['adx_downtrend_strat_cagr'] / df['adx_downtrend_bh_cagr'].replace(to_replace=0, value=np.nan))
    
    # B. Uptrend Capture: (strat_up / bh_up)
    df['score_uptrend'] = df['adx_uptrend_strat_cagr'] / df['adx_uptrend_bh_cagr'].replace(to_replace=0, value=np.nan)
    
    # C. CAGR Excess Ratio: (strat_cagr - bh_cagr) / bh_cagr
    df['score_excess'] = (df['oos_cagr'] - df['bh_cagr']) / df['bh_cagr'].replace(to_replace=0, value=np.nan)

    # Weighted Ranking Score (40/35/25)
    df['ranking_score'] = (
        0.40 * df['score_protection'] + 
        0.35 * df['score_uptrend'] + 
        0.25 * df['score_excess']
    )

    # Zwracamy tylko kandydatów, posortowanych od najlepszego wyniku
    return df[df['is_candidate'] == True].sort_values(by='ranking_score', ascending=False)