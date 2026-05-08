# -*- coding: utf-8 -*-
"""
moj_system/core/fund_filter.py
==============================
Moduł badawczy dla strategii opartych na "szerokości rynku" funduszy (fund breadth).

Zawiera logikę do pobierania danych funduszy z archiwum ZIP (via DataUpdater),
budowania zsynchronizowanego panelu NAV i generowania sygnału transakcyjnego 
na podstawie stóp zwrotu z najlepszych i najgorszych funduszy w danym okresie.
"""

import logging

import numpy as np
import pandas as pd

from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater


def prepare_fund_data(
    fund_codes:       dict[str, str],
    credentials_path: str | None = None,
) -> None:
    """
    Wyodrębnia i zapisuje do lokalnego folderu raw_csv dane historyczne
    z pliku ZIP na GDrive dla funduszy podanych w fund_codes.
    """
    updater = DataUpdater(
        credentials_path=credentials_path
    )

    for code, fund_name in fund_codes.items():
        safe_label = f"fund_{code}"
        stooq_ticker = f"{code}.n"
        
        logging.info(
            msg=f"Preparing fund data for: {fund_name} ({stooq_ticker})"
        )
        
        updater.update_ticker(
            label=safe_label,
            stooq_ticker=stooq_ticker,
            zip_type="fund_pl",
            upload_to_drive=False,
        )


def build_funds_df(
    fund_codes:        dict[str, str], 
    price_col:         str = "Zamkniecie", 
    min_history_years: int = 10,
) -> pd.DataFrame:
    """
    Buduje zsynchronizowany panel cen funduszy, oczyszcza brakujące dane
    i raportuje silnie skorelowane pary funduszy, aby uniknąć duplikacji sygnału.
    """
    series_list = []
    excluded =[]

    for code, fund_name in fund_codes.items():
        label = f"fund_{code}"
        
        df = load_local_csv(
            ticker=label,
            label=fund_name,
            mandatory=False,
        )
        
        if df is None:
            excluded.append((fund_name, "load failed"))
            continue

        if price_col not in df.columns:
            excluded.append((fund_name, f"missing column {price_col}"))
            continue

        series = df[price_col].copy()
        series.name = fund_name
        
        years_available = (series.index.max() - series.index.min()).days / 365.25
        if years_available < min_history_years:
            excluded.append((fund_name, f"insufficient history ({years_available:.1f}y)"))
            continue
            
        series_list.append(series)

    if len(series_list) < 2:
        return pd.DataFrame()

    funds_df = pd.concat(
        objs=series_list, 
        axis=1, 
        join="outer",
    ).sort_index().ffill()
    
    min_funds_required = max(2, len(funds_df.columns) // 2)
    funds_df = funds_df.dropna(
        thresh=min_funds_required
    )

    if funds_df.empty:
        logging.error(
            msg="build_funds_df: panel is empty after cleaning."
        )
        return funds_df

    # Analiza korelacji par funduszy
    corr_matrix = funds_df.corr()
    high_corr_pairs =[]
    matrix_cols = corr_matrix.columns
    
    for i in range(len(matrix_cols)):
        for j in range(i + 1, len(matrix_cols)):
            r_val = corr_matrix.iloc[i, j]
            if r_val > 0.98:
                high_corr_pairs.append((
                    matrix_cols[i],
                    matrix_cols[j],
                    round(number=r_val, ndigits=4)
                ))

    if high_corr_pairs:
        logging.info(
            msg="High correlation fund pairs (r > 0.98):"
        )
        for f1, f2, r_val in sorted(high_corr_pairs, key=lambda x: -x[2]):
            logging.info(
                msg=f"  {f1} / {f2}  r={r_val:.4f}"
            )

    return funds_df


def compute_fund_breadth_signal(
    funds_df:           pd.DataFrame,
    lookback_days:      int   = 30,
    n_top:              int   = 2,
    entry_roll_thresh:  float = 0.03,
    entry_since_thresh: float = 0.05,
    exit_roll_thresh:   float = -0.03,
    exit_since_thresh:  float = -0.05,
) -> pd.Series:
    """
    Compute a binary IN/OUT signal from a panel of fund NAV series 
    using comparative performance logic.
    """
    fund_rets = funds_df.pct_change()
    
    roll_ret = (1.0 + fund_rets).rolling(
        window=lookback_days
    ).apply(
        func=np.prod, 
        raw=True
    ) - 1.0
    
    signal = pd.Series(
        data=0.0, 
        index=funds_df.index
    )
    
    state = 0.0
    last_change_idx = funds_df.index[0]

    for day_idx, iter_date in enumerate(funds_df.index):
        if day_idx < lookback_days:
            signal.iloc[day_idx] = state
            continue

        todays_roll = roll_ret.loc[iter_date].dropna()
        if todays_roll.empty:
            signal.iloc[day_idx] = state
            continue

        ref_prices = funds_df.loc[:last_change_idx].iloc[-1]
        curr_prices = funds_df.loc[iter_date]
        since_rets = (curr_prices / ref_prices - 1.0).dropna()
        
        common_funds = todays_roll.index.intersection(
            other=since_rets.index
        )

        if len(common_funds) < n_top:
            signal.iloc[day_idx] = state
            continue

        top_funds = todays_roll.loc[common_funds].nlargest(
            n=n_top
        )
        bottom_funds = todays_roll.loc[common_funds].nsmallest(
            n=n_top
        )
        top_since = since_rets.loc[top_funds.index]
        bottom_since = since_rets.loc[bottom_funds.index]

        if state == 0.0:
            if top_funds.mean() >= entry_roll_thresh or top_since.mean() >= entry_since_thresh:
                state = 1.0
                last_change_idx = iter_date
        elif state == 1.0:
            if bottom_funds.mean() <= exit_roll_thresh or bottom_since.mean() <= exit_since_thresh:
                state = 0.0
                last_change_idx = iter_date
                
        signal.iloc[day_idx] = state
        
    return signal.shift(
        periods=1
    ).fillna(
        value=0.0
    )


def generate_fund_filter_signal(
    fund_codes:       dict[str, str],
    fund_params:      dict,
    credentials_path: str | None = None,
) -> pd.Series | None:
    """
    Główna funkcja orkiestrująca. Pobiera dane funduszy, buduje zsynchronizowany panel
    i wylicza binarny sygnał na podstawie zdefiniowanych w fund_params progów.
    
    Zwraca gotową serię `pd.Series`, którą można podpiąć jako `fund_signal`
    w silniku strategii.
    """
    
    logging.info(
        msg="Initiating fund filter signal generation..."
    )

    # 1. Pobranie / weryfikacja dostępności danych z pliku ZIP
    prepare_fund_data(
        fund_codes=fund_codes,
        credentials_path=credentials_path,
    )

    # 2. Budowa zsynchronizowanego panelu NAV
    funds_df = build_funds_df(
        fund_codes=fund_codes,
        price_col="Zamkniecie",
        min_history_years=10,
    )

    if funds_df.empty:
        logging.error(
            msg="Cannot generate fund filter signal: funds_df is empty."
        )
        return None

    # 3. Rozpakowanie parametrów z bezpiecznym fallbackiem na wartości domyślne
    lookback_days      = int(fund_params.get("lookback_days", 30))
    n_top              = int(fund_params.get("n_top", 2))
    entry_roll_thresh  = float(fund_params.get("entry_roll_thresh", 0.03))
    entry_since_thresh = float(fund_params.get("entry_since_thresh", 0.05))
    exit_roll_thresh   = float(fund_params.get("exit_roll_thresh", -0.03))
    exit_since_thresh  = float(fund_params.get("exit_since_thresh", -0.05))

    # 4. Wyliczenie docelowego sygnału
    signal_series = compute_fund_breadth_signal(
        funds_df=funds_df,
        lookback_days=lookback_days,
        n_top=n_top,
        entry_roll_thresh=entry_roll_thresh,
        entry_since_thresh=entry_since_thresh,
        exit_roll_thresh=exit_roll_thresh,
        exit_since_thresh=exit_since_thresh,
    )

    logging.info(
        msg=f"Fund filter signal generated successfully. Signal 'ON' time: {(signal_series.mean() * 100.0):.1f}%"
    )

    return signal_series