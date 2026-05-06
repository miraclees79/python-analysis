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

import numpy as np
import pandas as pd

# Import funkcji, od których te moduły zależą
from moj_system.data.data_manager import (  # (Zakładając, że download_csv_old jest chwilowo potrzebny)
    load_local_csv,
)

"""

# -------------------------------------------------------
# Fund NAV downloads for breadth filter
# -------------------------------------------------------

FUND_CODES = {
    "2718": "GS_Akcji",
    "3872": "Skarbiec_Akcji",
    "2847": "Investor_FundamentalnyDywWzr",
    "1422": "Allianz_Selektywny",
    "1626": "Rockbridge_Akcji",
    "4650": "Uniqa_Selektywny",
    "2869": "Ipopema_MiS",
    "4544": "Uniqa_Akcji",
    "3165": "Rockbridge_NeoAkcji",
    "3199": "Millenium_Akcji",
    "3959": "GenKorona_Akcji",
    "1056": "Superfund_Akcji",
    "3396": "PKO_Akcji",
    "3187": "Rockbridge_NeoAkcjiPL",
    "3360": "Pekao_AkcjiAktywna",
    "1137": "PZU_AkcjiPL",
    "1140": "PZU_AkcjiKrak",
    "1656": "Santander_AkcjiPL",
    "1621": "INPZU_AkcjiPL",
    "2159": "CA_Akcji",
    "1692": "SantanderPR_AkcjiPL",
    "2719": "GS_POI",
    "3151": "Esaliens_Akcji",
    "3166": "Rockbridge_NeoMid",
    "3306": "Velo_AkcjiPL",
    "3441": "Quercus_Agr",
    "1043": "Alior_Akcji"
    }

# Check for duplicate codes before downloading
seen_codes = {}
for code, name in FUND_CODES.items():
    if code in seen_codes:
        logging.warning(
            "Duplicate fund code %s — '%s' overwrites '%s'. Check FUND_CODES.",
            code, name, seen_codes[code]
        )
    seen_codes[code] = name

# Check for duplicate names
seen_names = {}
for code, name in FUND_CODES.items():
    if name in seen_names:
        logging.warning(
            "Duplicate fund name '%s' — code %s overwrites code %s. Check FUND_CODES.",
            name, code, seen_names[name]
        )
    seen_names[name] = code



if FORCE_FILTER_MODE is None or "fund" in FORCE_FILTER_MODE:
    FUND_FILES = download_fund_navs(FUND_CODES, tmp_dir)

    FUNDS = build_funds_df(
        fund_files=FUND_FILES,
        price_col="Zamkniecie",
        min_history_years=10
    ) if FUND_FILES else None

    if FUNDS is None or FUNDS.empty:
        logging.warning(
            "Fund panel unavailable — fund breadth filter will not be used."
        )
        FUNDS      = None
        FUND_PARAMS_GRID = None
    else:

 
        FUND_PARAMS_GRID = [    
            {
            "lookback_days":      30,   # medium asymmetric
            "entry_roll_thresh":  0.05,
            "entry_since_thresh": 0.08,
            "exit_roll_thresh":  -0.06,
            "exit_since_thresh": -0.10
            },
            {
            "lookback_days":      30, #strong asymmetric tight entry loose exit
            "entry_roll_thresh":  0.03,
            "entry_since_thresh": 0.05,
            "exit_roll_thresh":  -0.10,
            "exit_since_thresh": -0.15
            },
            {
            "lookback_days":      30, #original idea
            "entry_roll_thresh":  0.10,
            "entry_since_thresh": 0.15,
            "exit_roll_thresh":  -0.10,
            "exit_since_thresh": -0.15
            }
        ]
        #============================
        # Fund correlation check

        funds_df=FUNDS

        corr_matrix = funds_df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if r > 0.98:
                    high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    round(r, 4)
            ))

        if high_corr_pairs:
            logging.info("High correlation fund pairs (r>0.98):")
            for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
                logging.info("  %s / %s  r=%.4f", f1, f2, r)
else:
    FUNDS      = None 
    FUND_PARAMS_GRID = None
    
"""

def build_funds_df(
    fund_files: dict, price_col: str = "Zamkniecie", min_history_years: int = 10,
) -> pd.DataFrame:
    """
    Build a combined fund NAV panel from a list of CSV files.
    """
    MAX_GAP_DAYS = 30
    series_list = []
    excluded = []

    for fund_id, filepath in fund_files.items():
        df = load_local_csv(filepath)
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
