# -*- coding: utf-8 -*-
"""
moj_system/reporting/report_context.py
======================================
DTOs bundling everything a daily report builder needs, so that
``build_daily_outputs`` takes one object instead of 15-20 arguments.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(kw_only=True)
class DailyReportContext:
    """Fields shared by every strategy variant."""

    asset_name: str
    output_dir: str = "outputs"
    run_date: dt.date | str | None = None
    gdrive_folder_id: str | None = None
    gdrive_credentials: str | None = None


@dataclass(kw_only=True)
class SingleAssetReportContext(DailyReportContext):
    wf_equity: pd.Series
    wf_trades: pd.DataFrame
    wf_metrics: dict[str, float]
    wf_results: pd.DataFrame
    bh_equity: pd.Series
    bh_metrics: dict[str, float]
    df: pd.DataFrame
    price_col: str = "Zamkniecie"


@dataclass(kw_only=True)
class PensionReportContext(DailyReportContext):
    wf_trades_eq: pd.DataFrame
    wf_results_eq: pd.DataFrame
    wf_trades_bd: pd.DataFrame
    wf_results_bd: pd.DataFrame
    portfolio_equity: pd.Series
    portfolio_metrics: dict[str, float]
    weights_series: pd.Series
    reallocation_log: list[dict[str, Any]]
    bh_eq_equity: pd.Series
    bh_eq_metrics: dict[str, float]
    bh_bd_equity: pd.Series
    bh_bd_metrics: dict[str, float]
    wig_df: pd.DataFrame
    tbsp_df: pd.DataFrame
    sig_eq_oos: pd.Series
    sig_bd_oos: pd.Series
    exec_equity: pd.Series | None = None
    exec_metrics: dict[str, float] | None = None
    exec_decomp: dict[str, float] | None = None


@dataclass(kw_only=True)
class GlobalReportContext(DailyReportContext):
    wf_results_dict: dict[str, pd.DataFrame]
    wf_trades_dict: dict[str, pd.DataFrame]
    price_df_dict: dict[str, pd.DataFrame]
    portfolio_equity: pd.Series
    portfolio_metrics: dict[str, float]
    weights_series: pd.Series
    reallocation_log: list[dict[str, Any]]
    bh_metrics_dict: dict[str, dict[str, float]]
    returns_dict: dict[str, pd.Series]
    signals_oos_dict: dict[str, pd.Series]
    asset_keys: list[str]
    portfolio_mode: str
    fx_hedged: bool
    display_price_df_dict: dict[str, pd.DataFrame] = field(default_factory=dict)
