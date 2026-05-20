# -*- coding: utf-8 -*-
"""
moj_system/core/execution_engine.py
===================================
Silnik symulujący faktyczną egzekucję na instrumentach PPE.
"""

import logging

import pandas as pd

from moj_system.core.strategy_engine import compute_metrics
from moj_system.data.ppe_manager import PPE_COLUMN_MAPPING

def simulate_ppe_execution(
    target_weights: pd.Series,
    ppe_df:         pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Symuluje faktyczne zachowanie portfela nakładając wagi teoretyczne
    na stopy zwrotu funduszy PPE. Zwraca krzywą equity i metryki.
    """
    # 1. Wyliczenie dziennych stóp zwrotu funduszy PPE
    ret_df = pd.DataFrame(
        index=ppe_df.index
    )
    
    col_eq  = PPE_COLUMN_MAPPING["equity"]
    col_bd  = PPE_COLUMN_MAPPING["bond"]
    col_mmf = PPE_COLUMN_MAPPING["mmf"]
    
    ret_df["ret_equity"] = ppe_df[col_eq].pct_change()
    ret_df["ret_bond"]   = ppe_df[col_bd].pct_change()
    ret_df["ret_mmf"]    = ppe_df[col_mmf].pct_change()
    
    ret_df.dropna(
        inplace=True
    )

    # 2. Synchronizacja kalendarzy (WIG/TBSP vs fundusze PPE)
    # Wagi wyliczone na zamknięciu dnia T. Pracują i generują zwrot w dniu T+1.
    weights_df = pd.DataFrame(
        data=list(target_weights.values), 
        index=target_weights.index
    )
    
    weights_shifted = weights_df.shift(
        periods=1
    ).dropna()
    
    eval_idx = weights_shifted.index.intersection(
        other=ret_df.index
    )
    
    if eval_idx.empty:
        logging.error(
            msg="PPE Execution: No overlapping dates between shifted weights and PPE returns."
        )
        return pd.Series(dtype=float), {}

    w_eval = weights_shifted.loc[eval_idx]
    r_eval = ret_df.loc[eval_idx]

    # 3. Obliczenie zwrotu portfela egzekucyjnego
    daily_port_ret = (
        w_eval["equity"] * r_eval["ret_equity"] +
        w_eval["bond"]   * r_eval["ret_bond"] +
        w_eval["mmf"]    * r_eval["ret_mmf"]
    )
    
    execution_equity = (1.0 + daily_port_ret).cumprod()
    execution_equity = execution_equity / execution_equity.iloc[0]
    
    # 4. Obliczenie metryk
    exec_metrics = compute_metrics(
        equity=execution_equity,
        risk_free_rate=0.0
    )
    exec_metrics_float = {k: float(v) for k, v in exec_metrics.items()}
    
    logging.info(
        msg=f"PPE Execution Simulation completed. Execution CAGR: {exec_metrics_float['CAGR']*100.0:.2f}%"
    )

    return execution_equity, exec_metrics_float