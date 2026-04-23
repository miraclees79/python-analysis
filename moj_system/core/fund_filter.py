# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:03:12 2026

@author: adamg
"""

# moj_system/core/fund_filter.py

"""
moj_system/core/fund_filter.py
==============================
Moduł badawczy dla strategii opartych na "szerokości rynku" funduszy (fund breadth).

Zawiera logikę do pobierania danych funduszy, budowania panelu NAV 
i generowania sygnału transakcyjnego na podstawie wyników najlepszych/najgorszych
funduszy w danym okresie.

Funkcjonalność ta została wydzielona z `strategy_engine.py` w celu odizolowania
jej od głównego, aktywnego silnika strategii. Jest przeznaczona do dalszych badań.
"""

import logging
import os
import random
import time
import pandas as pd
import numpy as np
import requests
import sys

# Import funkcji, od których te moduły zależą
from moj_system.core.strategy_engine import load_csv, download_csv_old # (Zakładając, że download_csv_old jest chwilowo potrzebny)

def download_fund_navs(fund_codes: dict, tmp_dir: str) -> dict:
    """
    Download fund NAV series from stooq.pl and build the FUND_FILES dict.
    """
    BASE_URL = "https://stooq.pl/q/d/l/?s={code}.n&i=d"
    fund_files = {}

    for code, name in fund_codes.items():
        url = BASE_URL.format(code=code)
        filepath = os.path.join(tmp_dir, f"fund_{code}_{name}.csv")
        
        # UWAGA: Ta funkcja zależy od download_csv_old. Jeśli usuniesz ją z strategy_engine,
        # musisz przenieść ją tutaj lub zastąpić nowym mechanizmem pobierania.
        # Na razie zostawiamy, zakładając tymczasową zależność.
        success = download_csv_old(url, filepath)

        if success:
            fund_files[name] = filepath
            logging.info(f"Fund {name} ({code}) — downloaded to {filepath}.")
        else:
            logging.warning(f"Fund {name} ({code}) — download failed, will be excluded.")
        
        time.sleep(random.uniform(0.3, 1))
        
    logging.info(f"download_fund_navs: {len(fund_files)} of {len(fund_codes)} funds downloaded.")
    return fund_files

def build_funds_df(fund_files: dict, price_col: str = "Zamkniecie", min_history_years: int = 10) -> pd.DataFrame:
    """
    Build a combined fund NAV panel from a list of CSV files.
    """
    MAX_GAP_DAYS = 30
    series_list = []
    excluded = []

    for fund_id, filepath in fund_files.items():
        df = load_csv(filepath)
        if df is None:
            excluded.append((fund_id, "load failed"))
            continue

        if price_col not in df.columns:
            excluded.append((fund_id, f"missing column {price_col}"))
            continue

        series = df[price_col].copy()
        series.name = fund_id
        years = (series.index.max() - series.index.min()).days / 365.25
        if years < min_history_years:
            excluded.append((fund_id, f"insufficient history ({years:.1f}y)"))
            continue
        series_list.append(series)

    if len(series_list) < 2:
        return pd.DataFrame()

    funds_df = pd.concat(series_list, axis=1, join="outer", sort=True).ffill()
    min_funds_required = max(2, len(funds_df.columns) // 2)
    funds_df = funds_df.dropna(thresh=min_funds_required)

    if funds_df.empty:
        logging.error("build_funds_df: panel is empty after cleaning.")
    return funds_df

def compute_fund_breadth_signal(
    funds_df: pd.DataFrame,
    lookback_days: int = 30,
    n_top: int = 2,
    entry_roll_thresh: float = 0.03,
    entry_since_thresh: float = 0.05,
    exit_roll_thresh: float = -0.03,
    exit_since_thresh: float = -0.05,
) -> pd.Series:
    """
    Compute a binary IN/OUT signal from a panel of fund NAV series.
    """
    fund_rets = funds_df.pct_change()
    roll_ret = (1 + fund_rets).rolling(lookback_days).apply(np.prod, raw=True) - 1
    signal = pd.Series(0, index=funds_df.index)
    state = 0
    last_change_idx = funds_df.index[0]

    for i, date in enumerate(funds_df.index):
        if i < lookback_days:
            signal.iloc[i] = state
            continue

        todays_roll = roll_ret.loc[date].dropna()
        if todays_roll.empty:
            signal.iloc[i] = state
            continue

        ref_prices = funds_df.loc[:last_change_idx].iloc[-1]
        curr_prices = funds_df.loc[date]
        since_rets = (curr_prices / ref_prices - 1).dropna()
        common_funds = todays_roll.index.intersection(since_rets.index)

        if len(common_funds) < n_top:
            signal.iloc[i] = state
            continue

        top_funds = todays_roll.loc[common_funds].nlargest(n_top)
        bottom_funds = todays_roll.loc[common_funds].nsmallest(n_top)
        top_since = since_rets.loc[top_funds.index]
        bottom_since = since_rets.loc[bottom_funds.index]

        if state == 0:
            if top_funds.mean() >= entry_roll_thresh or top_since.mean() >= entry_since_thresh:
                state = 1
                last_change_idx = date
        elif state == 1:
            if bottom_funds.mean() <= exit_roll_thresh or bottom_since.mean() <= exit_since_thresh:
                state = 0
                last_change_idx = date
        signal.iloc[i] = state
    return signal.shift(1).fillna(0)