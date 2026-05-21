# -*- coding: utf-8 -*-
"""
moj_system/core/execution_engine.py
===================================
Silnik symulujący faktyczną egzekucję na instrumentach PPE.
Wzbogacony o dekompozycję Implementation Drag.
"""

import logging

import pandas as pd

from moj_system.core.strategy_engine import compute_metrics
from moj_system.data.ppe_manager import PPE_COLUMN_MAPPING


def _calc_cagr(
    r: pd.Series,
) -> float:
    """Pomocnicza funkcja licząca CAGR na czystej serii zwrotów."""
    if len(r) == 0:
        return 0.0
    return float(((1.0 + r).prod() ** (252.0 / len(r))) - 1.0)


def simulate_ppe_execution(
    target_weights:  pd.Series,
    ppe_df:          pd.DataFrame,
    theory_eq_ret:   pd.Series,
    theory_bd_ret:   pd.Series,
    theory_mmf_ret:  pd.Series,
    execution_delay: int = 5,
) -> tuple[pd.Series, dict[str, float], dict[str, float]]:
    """
    Symuluje faktyczne zachowanie portfela nakładając wagi teoretyczne
    na stopy zwrotu funduszy PPE. Zwraca krzywą equity, metryki oraz 
    szczegółową dekompozycję błędów odwzorowania.
    """
    ret_df = pd.DataFrame(index=ppe_df.index)

    col_eq  = PPE_COLUMN_MAPPING["equity"]
    col_bd  = PPE_COLUMN_MAPPING["bond"]
    col_mmf = PPE_COLUMN_MAPPING["mmf"]

    ret_df["ret_equity"] = ppe_df[col_eq].pct_change()
    ret_df["ret_bond"]   = ppe_df[col_bd].pct_change()
    ret_df["ret_mmf"]    = ppe_df[col_mmf].pct_change()
    ret_df.dropna(inplace=True)

    weights_df = pd.DataFrame(data=list(target_weights.values), index=target_weights.index)
    weights_shifted = weights_df.shift(periods=execution_delay).dropna()
    eval_idx = weights_shifted.index.intersection(other=ret_df.index)

    if eval_idx.empty:
        logging.error(msg="PPE Execution: No overlapping dates between shifted weights and PPE returns.")
        return pd.Series(dtype=float), {}, {}

    w_eval = weights_shifted.loc[eval_idx]
    r_eval = ret_df.loc[eval_idx]

    th_eq = theory_eq_ret.reindex(index=eval_idx).fillna(value=0.0)
    th_bd = theory_bd_ret.reindex(index=eval_idx).fillna(value=0.0)
    th_mmf = theory_mmf_ret.reindex(index=eval_idx).fillna(value=0.0)

    daily_port_ret = (
        w_eval["equity"] * r_eval["ret_equity"] +
        w_eval["bond"]   * r_eval["ret_bond"] +
        w_eval["mmf"]    * r_eval["ret_mmf"]
    )
    execution_equity = (1.0 + daily_port_ret).cumprod()
    execution_equity = execution_equity / execution_equity.iloc[0]

    exec_metrics = compute_metrics(equity=execution_equity, risk_free_rate=0.0)
    exec_metrics_float = {k: float(v) for k, v in exec_metrics.items()}

    # =========================================================================
    # DEKOMPOZYCJA DRAGU
    # =========================================================================
    decomp = {
        "eq_fund_cagr":  _calc_cagr(r=r_eval["ret_equity"]) * 100.0,
        "eq_idx_cagr":   _calc_cagr(r=th_eq) * 100.0,
        "bd_fund_cagr":  _calc_cagr(r=r_eval["ret_bond"]) * 100.0,
        "bd_idx_cagr":   _calc_cagr(r=th_bd) * 100.0,
        "mmf_fund_cagr": _calc_cagr(r=r_eval["ret_mmf"]) * 100.0,
        "mmf_idx_cagr":  _calc_cagr(r=th_mmf) * 100.0,
    }

    # B&H MaxDD dla poszczególnych komponentów (do celów analitycznych)
    decomp["eq_fund_maxdd"] = float(compute_metrics(equity=(1.0 + r_eval["ret_equity"]).cumprod(), risk_free_rate=0.0).get("MaxDD", 0.0)) * 100.0
    decomp["eq_idx_maxdd"] = float(compute_metrics(equity=(1.0 + th_eq).cumprod(), risk_free_rate=0.0).get("MaxDD", 0.0)) * 100.0
    decomp["bd_fund_maxdd"] = float(compute_metrics(equity=(1.0 + r_eval["ret_bond"]).cumprod(), risk_free_rate=0.0).get("MaxDD", 0.0)) * 100.0
    decomp["bd_idx_maxdd"] = float(compute_metrics(equity=(1.0 + th_bd).cumprod(), risk_free_rate=0.0).get("MaxDD", 0.0)) * 100.0

    port_th_rets = (w_eval["equity"] * th_eq) + (w_eval["bond"] * th_bd) + (w_eval["mmf"] * th_mmf)
    cagr_th_port = _calc_cagr(r=port_th_rets) * 100.0

    th_equity = (1.0 + port_th_rets).cumprod()
    th_equity = th_equity / th_equity.iloc[0]
    decomp["theory_port_cagr"] = cagr_th_port
    decomp["theory_port_maxdd"] = float(compute_metrics(equity=th_equity, risk_free_rate=0.0).get("MaxDD", 0.0)) * 100.0

    port_sub_eq = (w_eval["equity"] * r_eval["ret_equity"]) + (w_eval["bond"] * th_bd) + (w_eval["mmf"] * th_mmf)
    decomp["port_impact_eq"] = (_calc_cagr(r=port_sub_eq) * 100.0) - cagr_th_port

    port_sub_bd = (w_eval["equity"] * th_eq) + (w_eval["bond"] * r_eval["ret_bond"]) + (w_eval["mmf"] * th_mmf)
    decomp["port_impact_bd"] = (_calc_cagr(r=port_sub_bd) * 100.0) - cagr_th_port

    port_sub_mmf = (w_eval["equity"] * th_eq) + (w_eval["bond"] * th_bd) + (w_eval["mmf"] * r_eval["ret_mmf"])
    decomp["port_impact_mmf"] = (_calc_cagr(r=port_sub_mmf) * 100.0) - cagr_th_port

    logging.info(
        msg=f"PPE Execution Simulation completed. Execution CAGR: {exec_metrics_float['CAGR']*100.0:.2f}% (Delay: {execution_delay} days)",
    )

    return execution_equity, exec_metrics_float, decomp
