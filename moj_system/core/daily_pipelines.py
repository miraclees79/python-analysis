# -*- coding: utf-8 -*-
"""
moj_system/core/daily_pipelines.py
==================================
Production (daily) pipelines: ``SingleAssetPipeline``, ``PensionPipeline`` and
``GlobalPipeline``.

Each pipeline is the composition root for its variant: it obtains market data
from ``MarketDataFacade`` and injects the optimisation grids
(``OptimizationGridConfig``) and allocation settings
(``AllocationOptimizationConfig``) into the engines.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, OUTPUT_DIR
from moj_system.core.execution_engine import simulate_ppe_execution
from moj_system.core.global_engine import (
    allocation_walk_forward_n,
    print_global_equity_report,
)
from moj_system.core.grid_config import AllocationOptimizationConfig, OptimizationGridConfig
from moj_system.core.pension_engine import (
    allocation_walk_forward,
    build_signal_series,
    print_multiasset_report,
)
from moj_system.core.pipeline_base import BaseStrategyPipeline, drop_duplicate_dates
from moj_system.core.research import (
    extract_flat_regime_stats,
    prepare_regime_inputs,
    print_live_regime_report,
    run_regime_decomposition,
)
from moj_system.core.strategy_engine import (
    analyze_trades,
    compute_buy_and_hold,
    compute_metrics,
    get_n_jobs,
    print_backtest_report,
    walk_forward,
)
from moj_system.core.universe import (
    AssetSpec,
    build_display_price_df,
    convert_trade_prices_for_display,
    prepare_asset_series,
    resolve_train_years,
    wf_grid_config,
)
from moj_system.data.market_data_facade import (
    GlobalMarketContext,
    MarketDataError,
    MarketDataFacade,
    PensionMarketContext,
)
from moj_system.reporting.daily_output import build_daily_outputs
from moj_system.reporting.global_equity_daily_output import (
    build_daily_outputs as build_global_outputs,
)
from moj_system.reporting.multiasset_daily_output import (
    build_daily_outputs as build_multiasset_outputs,
)
from moj_system.reporting.pension_email import send_pension_report
from moj_system.reporting.report_context import (
    GlobalReportContext,
    PensionReportContext,
    SingleAssetReportContext,
)
from moj_system.reporting.strategy_email import send_global_report, send_single_asset_report

BOND_ENTRY_FILTER_MODES: list[str] = ["ma"]
PPE_EXECUTION_DELAY_DAYS = 5


def _log_regime_report(
    regime_inputs: dict[str, pd.Series],
) -> None:
    raw_regimes = run_regime_decomposition(
        inputs=regime_inputs,
        generate_plots=False,
    )
    regime_metrics = extract_flat_regime_stats(regime_results=raw_regimes)
    print_live_regime_report(regime_metrics=regime_metrics)


def _float_metrics(
    equity: pd.Series,
) -> dict[str, float]:
    return {name: float(value) for name, value in compute_metrics(equity=equity).items()}


class DailyStrategyPipeline(BaseStrategyPipeline[Any, Any]):
    """Shared state of the daily pipelines."""

    def __init__(
        self,
        facade: MarketDataFacade,
        stop_mode_arg: str,
        send_email: bool = False,
    ) -> None:
        self.facade = facade
        self.stop_mode_arg = stop_mode_arg
        self.send_email = send_email

    @property
    def creds_path(self) -> str:
        return self.facade.credentials_path or ""


# ---------------------------------------------------------------------------
# Single asset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SingleAssetData:
    df: pd.DataFrame
    cash_df: pd.DataFrame


@dataclass(frozen=True, eq=False)
class SingleAssetSimulation:
    wf_equity: pd.Series
    wf_results: pd.DataFrame
    wf_trades: pd.DataFrame
    wf_metrics: dict[str, float]
    bh_equity: pd.Series
    bh_metrics: dict[str, float]


class SingleAssetPipeline(DailyStrategyPipeline):
    def __init__(
        self,
        asset_name: str,
        facade: MarketDataFacade,
        stop_mode_arg: str,
        send_email: bool = False,
        grid_config: OptimizationGridConfig | None = None,
    ) -> None:
        super().__init__(
            facade=facade,
            stop_mode_arg=stop_mode_arg,
            send_email=send_email,
        )
        self.asset_name = asset_name
        self.asset_config = ASSET_REGISTRY[asset_name]
        self.train_years = cast("int", self.asset_config["train"])
        self.test_years = cast("int", self.asset_config["test"])
        self.output_prefix = asset_name.lower()
        self.stop_mode = (
            self.asset_config.get("default_stop", "fixed") if stop_mode_arg == "auto" else stop_mode_arg
        )
        self.use_atr_stop = self.stop_mode == "atr"
        base_grids = OptimizationGridConfig.from_mapping(grids=BASE_GRIDS)
        self.grid_config = (
            base_grids.with_overrides(grids=cast("Mapping[str, Any]", self.asset_config.get("grids", {})))
            if grid_config is None
            else grid_config
        )

    def load_data(self) -> SingleAssetData:
        logging.info(
            f"SINGLE ASSET ENGINE: {self.asset_name} | Stop Mode: {self.stop_mode} | "
            f"Train: {self.train_years} y | Test: {self.test_years} y"
        )
        try:
            cash_df = self.facade.get_cash_series()
            asset_df = self.facade.get_asset_series(asset_key=self.asset_name)
        except MarketDataError as error:
            sys.exit(str(error))
        return SingleAssetData(df=asset_df, cash_df=cash_df)

    def run_simulation(
        self,
        market_data: SingleAssetData,
    ) -> SingleAssetSimulation:
        wf_equity, wf_results, wf_trades = walk_forward(
            df=market_data.df,
            cash_df=market_data.cash_df,
            train_years=self.train_years,
            test_years=self.test_years,
            objective="calmar",
            n_jobs=get_n_jobs(),
            fast_mode=True,
            use_atr_stop=self.use_atr_stop,
            grid_config=self.grid_config,
        )
        if wf_equity.empty:
            sys.exit(f"Walk-forward returned no results for {self.asset_name}.")

        wf_equity = drop_duplicate_dates(series=wf_equity)
        wf_metrics = _float_metrics(equity=wf_equity)
        trade_stats = analyze_trades(trades=wf_trades)

        print_backtest_report(
            metrics=wf_metrics,
            trades=wf_trades,
            trade_stats=trade_stats,
            wf_results=wf_results,
            position_mode="full",
            filter_modes_override=None,
        )

        oos_start = wf_results["TestStart"].min()
        oos_end = wf_results["TestEnd"].max()
        bh_equity, bh_metrics = compute_buy_and_hold(
            df=market_data.df,
            price_col="Zamkniecie",
            start=oos_start,
            end=oos_end,
        )

        regime_inputs = prepare_regime_inputs(
            df=market_data.df,
            wf_results=wf_results,
            wf_equity=wf_equity,
            bh_equity=bh_equity,
        )
        if regime_inputs:
            _log_regime_report(regime_inputs=regime_inputs)
        else:
            logging.warning(
                f"Skipping regime analysis for {self.asset_name} due to insufficient data overlap.",
            )
            print_live_regime_report(regime_metrics=extract_flat_regime_stats(regime_results={}))

        return SingleAssetSimulation(
            wf_equity=wf_equity,
            wf_results=wf_results,
            wf_trades=wf_trades,
            wf_metrics=wf_metrics,
            bh_equity=bh_equity,
            bh_metrics=bh_metrics,
        )

    def generate_reports(
        self,
        market_data: SingleAssetData,
        simulation: SingleAssetSimulation,
    ) -> dict[str, Any]:
        outputs = build_daily_outputs(
            context=SingleAssetReportContext(
                asset_name=self.asset_name,
                output_dir=str(OUTPUT_DIR / self.output_prefix),
                gdrive_folder_id=self.facade.folder_id,
                gdrive_credentials=self.creds_path if Path(self.creds_path).exists() else None,
                wf_equity=simulation.wf_equity,
                wf_trades=simulation.wf_trades,
                wf_metrics=simulation.wf_metrics,
                wf_results=simulation.wf_results,
                bh_equity=simulation.bh_equity,
                bh_metrics=simulation.bh_metrics,
                df=market_data.df,
                price_col="Zamkniecie",
            ),
        )
        if self.send_email:
            send_single_asset_report(
                asset_name=self.asset_name,
                snapshot_path=outputs["snapshot_path"],
                chart_path=outputs["chart_path"],
            )
        return outputs


# ---------------------------------------------------------------------------
# Pension (WIG + TBSP + MMF)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PensionSimulation:
    wf_eq: pd.Series
    wf_res_eq: pd.DataFrame
    wf_tr_eq: pd.DataFrame
    wf_bd: pd.Series
    wf_res_bd: pd.DataFrame
    wf_tr_bd: pd.DataFrame
    portfolio_equity: pd.Series
    portfolio_metrics: dict[str, float]
    weights_series: pd.Series
    reallocation_log: list[dict[str, Any]]
    bh_eq_equity: pd.Series
    bh_eq_metrics: dict[str, float]
    bh_bd_equity: pd.Series
    bh_bd_metrics: dict[str, float]
    sig_eq_oos: pd.Series
    sig_bd_oos: pd.Series
    exec_equity: pd.Series | None
    exec_metrics: dict[str, float]
    exec_decomp: dict[str, float]


class PensionPipeline(DailyStrategyPipeline):
    def __init__(
        self,
        facade: MarketDataFacade,
        stop_mode_arg: str,
        send_email: bool = False,
        equity_grid_config: OptimizationGridConfig | None = None,
        bond_grid_config: OptimizationGridConfig | None = None,
        allocation_config: AllocationOptimizationConfig | None = None,
    ) -> None:
        super().__init__(
            facade=facade,
            stop_mode_arg=stop_mode_arg,
            send_email=send_email,
        )
        self.asset_config = ASSET_REGISTRY["PENSION"]
        self.train_years = cast("int", self.asset_config["train"])
        self.test_years = cast("int", self.asset_config["test"])
        self.stop_mode = (
            self.asset_config.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
        )
        self.use_atr_eq = self.stop_mode == "atr"
        self.equity_grid_config = equity_grid_config or OptimizationGridConfig()
        self.bond_grid_config = bond_grid_config or OptimizationGridConfig()
        self.allocation_config = allocation_config or AllocationOptimizationConfig()

    def load_data(self) -> PensionMarketContext:
        logging.info(
            f"PENSION PORTFOLIO ENGINE (WIG+TBSP+MMF) | WIG Stop: {self.stop_mode} | "
            f"Train: {self.train_years} y | Test: {self.test_years} y"
        )
        try:
            return self.facade.get_pension_context()
        except MarketDataError as error:
            sys.exit(str(error))

    def run_simulation(
        self,
        market_data: PensionMarketContext,
    ) -> PensionSimulation:
        derived = market_data.derived
        n_jobs = get_n_jobs()

        logging.info("========== RUNNING: WIG ==========")
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
            df=market_data.wig,
            cash_df=derived["mmf_ext"],
            train_years=self.train_years,
            test_years=self.test_years,
            use_atr_stop=self.use_atr_eq,
            n_jobs=n_jobs,
            grid_config=self.equity_grid_config,
        )
        if wf_eq.empty:
            sys.exit("Walk-forward returned no results for WIG.")

        wf_eq = drop_duplicate_dates(series=wf_eq)
        wf_metrics = _float_metrics(equity=wf_eq)
        trade_stats = analyze_trades(trades=wf_tr_eq)
        print_backtest_report(
            metrics=wf_metrics,
            trades=wf_tr_eq,
            trade_stats=trade_stats,
            wf_results=wf_res_eq,
            position_mode="full",
            filter_modes_override=None,
        )

        logging.info("========== RUNNING: TBSP ==========")
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=market_data.tbsp,
            cash_df=derived["mmf_ext"],
            train_years=self.train_years,
            test_years=self.test_years,
            filter_modes_override=BOND_ENTRY_FILTER_MODES,
            n_jobs=n_jobs,
            entry_gate_series=derived["bond_gate"],
            grid_config=self.bond_grid_config,
        )

        sig_eq = build_signal_series(wf_equity=wf_eq, wf_trades=wf_tr_eq)
        sig_bd = build_signal_series(wf_equity=wf_bd, wf_trades=wf_tr_bd)
        oos_start = max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min())
        oos_end = min(wf_res_eq["TestEnd"].max(), wf_res_bd["TestEnd"].max())
        sig_eq_oos = sig_eq.loc[oos_start:oos_end]
        sig_bd_oos = sig_bd.loc[oos_start:oos_end]

        portfolio_equity, weights_series, reallocation_log, alloc_results_df = allocation_walk_forward(
            equity_returns=derived["ret_eq"],
            bond_returns=derived["ret_bd"],
            mmf_returns=derived["ret_mmf"],
            sig_equity_full=sig_eq,
            sig_bond_full=sig_bd,
            sig_equity_oos=sig_eq_oos,
            sig_bond_oos=sig_bd_oos,
            wf_results_eq=wf_res_eq,
            wf_results_bd=wf_res_bd,
            allocation_config=self.allocation_config,
        )
        portfolio_equity = drop_duplicate_dates(series=portfolio_equity)
        portfolio_metrics = compute_metrics(equity=portfolio_equity)
        bh_eq_equity, bh_eq_metrics = compute_buy_and_hold(
            df=market_data.wig,
            price_col="Zamkniecie",
            start=oos_start,
            end=oos_end,
        )
        bh_bd_equity, bh_bd_metrics = compute_buy_and_hold(
            df=market_data.tbsp,
            price_col="Zamkniecie",
            start=oos_start,
            end=oos_end,
        )

        print_multiasset_report(
            portfolio_metrics=portfolio_metrics,
            bh_equity_metrics=bh_eq_metrics,
            bh_bond_metrics=bh_bd_metrics,
            alloc_results_df=alloc_results_df,
            reallocation_log=reallocation_log,
            sig_equity=sig_eq_oos,
            sig_bond=sig_bd_oos,
            oos_start=oos_start,
            oos_end=oos_end,
            sig_bond_raw=sig_bd,
        )

        exec_equity, exec_metrics, exec_decomp = self._simulate_execution(
            weights_series=weights_series,
            market_data=market_data,
        )

        _log_regime_report(
            regime_inputs=prepare_regime_inputs(
                df=market_data.wig,
                wf_results=wf_res_eq,
                wf_equity=portfolio_equity,
                bh_equity=bh_eq_equity,
            ),
        )

        return PensionSimulation(
            wf_eq=wf_eq,
            wf_res_eq=wf_res_eq,
            wf_tr_eq=wf_tr_eq,
            wf_bd=wf_bd,
            wf_res_bd=wf_res_bd,
            wf_tr_bd=wf_tr_bd,
            portfolio_equity=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            weights_series=weights_series,
            reallocation_log=reallocation_log,
            bh_eq_equity=bh_eq_equity,
            bh_eq_metrics=bh_eq_metrics,
            bh_bd_equity=bh_bd_equity,
            bh_bd_metrics=bh_bd_metrics,
            sig_eq_oos=sig_eq_oos,
            sig_bd_oos=sig_bd_oos,
            exec_equity=exec_equity,
            exec_metrics=exec_metrics,
            exec_decomp=exec_decomp,
        )

    def _simulate_execution(
        self,
        weights_series: pd.Series,
        market_data: PensionMarketContext,
    ) -> tuple[pd.Series | None, dict[str, float], dict[str, float]]:
        ppe_df = self.facade.get_ppe_data()
        if ppe_df is None or ppe_df.empty:
            logging.warning(msg="PPE Data unavailable. Skipping execution simulation.")
            return None, {}, {}

        derived = market_data.derived
        return simulate_ppe_execution(
            target_weights=weights_series,
            ppe_df=ppe_df,
            theory_eq_ret=derived["ret_eq"],
            theory_bd_ret=derived["ret_bd"],
            theory_mmf_ret=derived["ret_mmf"],
            execution_delay=PPE_EXECUTION_DELAY_DAYS,
        )

    def generate_reports(
        self,
        market_data: PensionMarketContext,
        simulation: PensionSimulation,
    ) -> dict[str, Any]:
        outputs = build_multiasset_outputs(
            context=PensionReportContext(
                asset_name="PENSION",
                output_dir=str(OUTPUT_DIR / "pension"),
                gdrive_folder_id=self.facade.folder_id,
                gdrive_credentials=self.creds_path,
                wf_trades_eq=simulation.wf_tr_eq,
                wf_results_eq=simulation.wf_res_eq,
                wf_trades_bd=simulation.wf_tr_bd,
                wf_results_bd=simulation.wf_res_bd,
                portfolio_equity=simulation.portfolio_equity,
                portfolio_metrics=simulation.portfolio_metrics,
                weights_series=simulation.weights_series,
                reallocation_log=simulation.reallocation_log,
                bh_eq_equity=simulation.bh_eq_equity,
                bh_eq_metrics=simulation.bh_eq_metrics,
                bh_bd_equity=simulation.bh_bd_equity,
                bh_bd_metrics=simulation.bh_bd_metrics,
                wig_df=market_data.wig,
                tbsp_df=market_data.tbsp,
                sig_eq_oos=simulation.sig_eq_oos,
                sig_bd_oos=simulation.sig_bd_oos,
                exec_equity=simulation.exec_equity,
                exec_metrics=simulation.exec_metrics,
                exec_decomp=simulation.exec_decomp,
            ),
        )
        if self.send_email:
            send_pension_report(
                snapshot_path=outputs["snapshot_path"],
                chart_path=outputs["chart_path"],
            )
        return outputs


# ---------------------------------------------------------------------------
# Global portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GlobalSimulation:
    portfolio_equity: pd.Series
    portfolio_metrics: dict[str, float]
    weights_series: pd.Series
    reallocation_log: list[dict[str, Any]]
    bh_metrics_dict: dict[str, dict[str, float]]
    returns_dict: dict[str, pd.Series]
    signals_dict: dict[str, pd.Series]
    wf_results_dict: dict[str, pd.DataFrame]
    wf_trades_dict: dict[str, pd.DataFrame]
    price_df_dict: dict[str, pd.DataFrame]
    display_price_df_dict: dict[str, pd.DataFrame] = field(default_factory=dict)


class GlobalPipeline(DailyStrategyPipeline):
    def __init__(
        self,
        asset_key: str,
        facade: MarketDataFacade,
        stop_mode_arg: str,
        send_email: bool = False,
        bond_grid_config: OptimizationGridConfig | None = None,
    ) -> None:
        super().__init__(
            facade=facade,
            stop_mode_arg=stop_mode_arg,
            send_email=send_email,
        )
        self.asset_key = asset_key
        self.asset_config = ASSET_REGISTRY[asset_key]
        self.mode = str(self.asset_config["mode"])
        self.train_years = cast("int", self.asset_config["train"])
        self.test_years = cast("int", self.asset_config["test"])
        self.fx_hedged = bool(self.asset_config["fx_hedged"])
        self.stop_mode = (
            self.asset_config.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
        )
        self.use_atr_eq = self.stop_mode == "atr"
        self.bond_grid_config = bond_grid_config or OptimizationGridConfig()

    def load_data(self) -> GlobalMarketContext:
        logging.info(
            msg=(
                f"GLOBAL PORTFOLIO ENGINE: {self.mode} | Equity Stop: {self.stop_mode} | "
                f"FX Hedged: {self.fx_hedged} | Train: {self.train_years} y | Test: {self.test_years} y"
            ),
        )
        try:
            return self.facade.get_global_context(variant=self.asset_key)
        except MarketDataError as error:
            sys.exit(str(error))

    def run_simulation(
        self,
        market_data: GlobalMarketContext,
    ) -> GlobalSimulation:
        pension_data = market_data.pension
        mmf_df = pension_data.mmf
        n_jobs = get_n_jobs()

        returns_dict: dict[str, pd.Series] = {}
        signals_dict: dict[str, pd.Series] = {}
        wf_results_dict: dict[str, pd.DataFrame] = {}
        wf_trades_dict: dict[str, pd.DataFrame] = {}
        price_df_dict: dict[str, pd.DataFrame] = {}
        display_price_df_dict: dict[str, pd.DataFrame] = {}

        wig_wf_results: pd.DataFrame | None = None
        for label, spec in market_data.assets.items():
            asset_returns, processed_prices = prepare_asset_series(label=label, spec=spec)
            returns_dict[label] = asset_returns

            logging.info(msg=f"========== RUNNING: {label} ==========")
            wf_equity, wf_results, wf_trades = self._run_equity_leg(
                spec=spec,
                processed_prices=processed_prices,
                cash_df=mmf_df,
                n_jobs=n_jobs,
                label=label,
            )
            signals_dict[label] = build_signal_series(
                wf_equity=wf_equity,
                wf_trades=wf_trades,
            )
            wf_results_dict[label] = wf_results
            wf_trades_dict[label] = wf_trades
            price_df_dict[label] = processed_prices
            display_prices = build_display_price_df(spec=spec)
            if display_prices is not None:
                display_price_df_dict[label] = display_prices

            if label == "WIG":
                wig_wf_results = wf_results

        logging.info(msg="========== RUNNING: TBSP ==========")
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=pension_data.tbsp,
            cash_df=mmf_df,
            train_years=self.train_years,
            test_years=self.test_years,
            filter_modes_override=BOND_ENTRY_FILTER_MODES,
            n_jobs=n_jobs,
            entry_gate_series=pension_data.derived["bond_gate"],
            grid_config=self.bond_grid_config,
        )
        returns_dict["TBSP"] = pension_data.tbsp["Zamkniecie"].pct_change().dropna()
        signals_dict["TBSP"] = build_signal_series(
            wf_equity=wf_bd,
            wf_trades=wf_tr_bd,
        )
        wf_results_dict["TBSP"] = wf_res_bd
        wf_trades_dict["TBSP"] = wf_tr_bd
        price_df_dict["TBSP"] = pension_data.tbsp

        settings = market_data.settings
        portfolio_equity, weights_series, reallocation_log, alloc_results_df = allocation_walk_forward_n(
            returns_dict=returns_dict,
            signals_full_dict=signals_dict,
            signals_oos_dict=signals_dict,
            mmf_returns=mmf_df["Zamkniecie"].pct_change().dropna(),
            wf_results_ref=wf_res_bd,
            asset_keys=list(returns_dict.keys()),
            train_years=self.train_years,
            allocation_config=AllocationOptimizationConfig(
                asset_caps=settings.asset_caps,
                optional_keys=settings.optional_keys,
                min_delta=settings.min_delta,
                delta_tol=settings.delta_tol,
            ),
        )
        portfolio_equity = drop_duplicate_dates(series=portfolio_equity)
        portfolio_metrics = _float_metrics(equity=portfolio_equity)

        bh_metrics_dict: dict[str, dict[str, float]] = {}
        for label, asset_returns in returns_dict.items():
            aligned_returns = asset_returns.reindex(index=portfolio_equity.index).fillna(value=0.0)
            bh_metrics_dict[label] = compute_metrics(equity=(1.0 + aligned_returns).cumprod())

        print_global_equity_report(
            portfolio_metrics=portfolio_metrics,
            bh_metrics_dict=bh_metrics_dict,
            alloc_results_df=alloc_results_df,
            reallocation_log=reallocation_log,
            signals_oos_dict=signals_dict,
            oos_start=portfolio_equity.index.min(),
            oos_end=portfolio_equity.index.max(),
            portfolio_mode=self.mode,
            fx_hedged=self.fx_hedged,
        )

        aligned_wig_returns = returns_dict["WIG"].reindex(index=portfolio_equity.index).fillna(value=0.0)
        bh_wig_equity = (1.0 + aligned_wig_returns).cumprod()
        _log_regime_report(
            regime_inputs=prepare_regime_inputs(
                df=pension_data.wig,
                wf_results=wig_wf_results,
                wf_equity=portfolio_equity,
                bh_equity=bh_wig_equity,
            ),
        )

        return GlobalSimulation(
            portfolio_equity=portfolio_equity,
            portfolio_metrics=portfolio_metrics,
            weights_series=weights_series,
            reallocation_log=reallocation_log,
            bh_metrics_dict=bh_metrics_dict,
            returns_dict=returns_dict,
            signals_dict=signals_dict,
            wf_results_dict=wf_results_dict,
            wf_trades_dict=wf_trades_dict,
            price_df_dict=price_df_dict,
            display_price_df_dict=display_price_df_dict,
        )

    def _run_equity_leg(
        self,
        spec: AssetSpec,
        processed_prices: pd.DataFrame,
        cash_df: pd.DataFrame,
        n_jobs: int,
        label: str,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        wf_equity, wf_results, wf_trades = walk_forward(
            df=processed_prices,
            cash_df=cash_df,
            train_years=resolve_train_years(
                cfg=self.asset_config,
                spec=spec,
                default_train=self.train_years,
            ),
            test_years=self.test_years,
            use_atr_stop=self.use_atr_eq,
            n_jobs=n_jobs,
            grid_config=wf_grid_config(spec=spec),
        )
        if wf_equity.empty:
            sys.exit(f"Walk-forward returned no results for {label}.")

        wf_equity = drop_duplicate_dates(series=wf_equity)
        print_backtest_report(
            metrics=_float_metrics(equity=wf_equity),
            trades=wf_trades,
            trade_stats=analyze_trades(trades=wf_trades),
            wf_results=wf_results,
            position_mode="full",
            filter_modes_override=None,
            display_trades=convert_trade_prices_for_display(
                trades=wf_trades,
                spec=spec,
            ),
        )
        return wf_equity, wf_results, wf_trades

    def generate_reports(
        self,
        market_data: GlobalMarketContext,
        simulation: GlobalSimulation,
    ) -> dict[str, Any]:
        outputs = build_global_outputs(
            context=GlobalReportContext(
                asset_name=self.asset_key,
                output_dir=str(OUTPUT_DIR / self.asset_key.lower()),
                gdrive_folder_id=self.facade.folder_id,
                gdrive_credentials=self.creds_path,
                wf_results_dict=simulation.wf_results_dict,
                wf_trades_dict=simulation.wf_trades_dict,
                price_df_dict=simulation.price_df_dict,
                portfolio_equity=simulation.portfolio_equity,
                portfolio_metrics=simulation.portfolio_metrics,
                weights_series=simulation.weights_series,
                reallocation_log=simulation.reallocation_log,
                bh_metrics_dict=simulation.bh_metrics_dict,
                returns_dict=simulation.returns_dict,
                signals_oos_dict=simulation.signals_dict,
                asset_keys=list(simulation.returns_dict.keys()),
                portfolio_mode=self.mode,
                fx_hedged=market_data.fx_hedged,
                display_price_df_dict=simulation.display_price_df_dict,
            ),
        )
        if self.send_email:
            send_global_report(
                asset_name=self.asset_key,
                snapshot_path=outputs["snapshot_path"],
                chart_path=outputs["chart_path"],
            )
        return outputs
