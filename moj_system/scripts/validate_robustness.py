# moj_system/scripts/validate_robustness.py

# -*- coding: utf-8 -*-
"""
moj_system/scripts/validate_robustness.py
=========================================
Deep Validation Engine. Performs Level 1 (MC), Level 2 (Bootstrap),
and optionally Level 3 (Allocation Weight Perturbation) testing for
chosen configurations of Single, Pension, or Global strategies.
Includes automated OOS chart generation and full component-level validation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# --- PATH SETUP ---
from moj_system.config import (
    ASSET_REGISTRY,
    BASE_GRIDS,
    BOND_GRIDS,
    BOND_THRESHOLDS_BOOTSTRAP,
    BOND_THRESHOLDS_MC,
    EQUITY_THRESHOLDS_BOOTSTRAP,
    EQUITY_THRESHOLDS_MC,
    OUTPUT_DIR,
    PORTFOLIO_THRESHOLDS_BOOTSTRAP,
    PORTFOLIO_THRESHOLDS_MC,
    THRESHOLDS_BOOTSTRAP,
    THRESHOLDS_MC,
)
from moj_system.core.global_engine import (
    allocation_walk_forward_n,
    allocation_weight_robustness_n,
    print_allocation_robustness_report_n,
)
from moj_system.core.grid_config import OptimizationGridConfig
from moj_system.core.pension_engine import (
    allocation_walk_forward,
    allocation_weight_robustness,
    build_signal_series,
    build_standard_two_asset_data,
    print_allocation_robustness_report,
)
from moj_system.core.pipeline_base import BaseStrategyPipeline
from moj_system.core.portfolio_robustness import (
    PortfolioLeg,
    add_ohlc_columns_for_context,
    analyze_portfolio_robustness,
    combine_portfolio_verdicts,
    run_parameterized_signals,
)
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.robustness_engine import (
    analyze_bootstrap,
    analyze_robustness,
    extract_best_params_from_wf_results,
)

# --- CORE ENGINE IMPORTS ---
from moj_system.core.strategy_engine import (
    compute_buy_and_hold,
    compute_metrics,
    get_n_jobs,
    walk_forward,
)
from moj_system.core.universe import (
    prepare_asset_series,
    resolve_train_years,
    wf_grid_config,
)
from moj_system.data.market_data_facade import GlobalMarketContext, MarketDataFacade, PensionMarketContext
from moj_system.data.updater import DataUpdater


class ValidationManager:
    def __init__(
        self,
        n_mc: int,
        n_boot: int,
        run_weights_perturb: bool,
        n_mc_portfolio: int | None = None,
        n_boot_portfolio: int | None = None,
    ) -> None:

        self.n_mc = n_mc
        self.n_boot = n_boot
        self.n_mc_portfolio = n_mc if n_mc_portfolio is None else n_mc_portfolio
        self.n_boot_portfolio = n_boot if n_boot_portfolio is None else n_boot_portfolio
        self.run_weights_perturb = run_weights_perturb
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = str(Path(tempfile.gettempdir()) / "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        self.facade = MarketDataFacade(
            folder_id=self.folder_id,
            credentials_path=self.creds_path,
        )
        self.equity_grid_config = OptimizationGridConfig.from_mapping(
            grids=BASE_GRIDS,
            keys=("X_GRID", "Y_GRID", "FAST_GRID", "SLOW_GRID", "N_ATR_GRID"),
        )
        self.bond_grid_config = OptimizationGridConfig.from_mapping(
            grids=BOND_GRIDS,
            keys=("X_GRID", "Y_GRID", "FAST_GRID", "SLOW_GRID"),
        )

    def _run_pension_portfolio_robustness(
        self,
        wig: pd.DataFrame,
        tbsp: pd.DataFrame,
        cash: pd.DataFrame,
        bond_gate: pd.Series,
        wf_eq: pd.Series,
        wf_res_eq: pd.DataFrame,
        wf_res_bd: pd.DataFrame,
        ret_eq: pd.Series,
        ret_bd: pd.Series,
        ret_mmf: pd.Series,
        baseline_portfolio: pd.Series,
        train_y: int,
        test_y: int,
        use_atr_eq: bool,
        pl10y: pd.DataFrame,
        de10y: pd.DataFrame,
        wibor1m: pd.DataFrame | None,
        sample_offset: int = 0,
        emit_report: bool = True,
    ) -> dict[str, pd.DataFrame]:
        legs = [
            PortfolioLeg(
                name="WIG",
                best_params=extract_best_params_from_wf_results(wf_res_eq),
                frame=wig,
            ),
            PortfolioLeg(
                name="TBSP",
                best_params=extract_best_params_from_wf_results(wf_res_bd),
                frame=tbsp,
            ),
        ]
        wf_results = {"WIG": wf_res_eq, "TBSP": wf_res_bd}
        entry_gates = {"TBSP": bond_gate}
        results: dict[str, pd.DataFrame] = {}
        summaries: list[dict[str, object]] = []

        def allocate(signals: dict[str, pd.Series]) -> pd.Series:
            equity, unused_weights, unused_realloc, unused_alloc_df = allocation_walk_forward(
                equity_returns=ret_eq,
                bond_returns=ret_bd,
                mmf_returns=ret_mmf,
                sig_equity_full=signals["WIG"],
                sig_bond_full=signals["TBSP"],
                sig_equity_oos=signals["WIG"].loc[wf_eq.index.min() :],
                sig_bond_oos=signals["TBSP"].loc[wf_eq.index.min() :],
                wf_results_eq=wf_res_eq,
                wf_results_bd=wf_res_bd,
            )
            return equity

        def mc_runner(universe: dict[str, dict[int, dict]]) -> pd.Series:
            signals = run_parameterized_signals(
                universe=universe,
                legs=legs,
                wf_results=wf_results,
                cash_df=cash,
                entry_gates=entry_gates,
            )
            return allocate(signals)

        if self.n_mc_portfolio > 0:
            mc_results = self.rob_engine.run_portfolio_mc_test(
                legs=legs,
                runner=mc_runner,
                n_samples=self.n_mc_portfolio,
                sample_offset=sample_offset,
            )
            results["mc"] = mc_results
            if emit_report:
                mc_summary = analyze_portfolio_robustness(
                    results_df=mc_results,
                    baseline_metrics=compute_metrics(equity=baseline_portfolio),
                    thresholds=PORTFOLIO_THRESHOLDS_MC,
                    label="PENSION PORTFOLIO MC",
                )
                summaries.append(mc_summary)

        def bootstrap_runner(
            frames: dict[str, pd.DataFrame],
            synthetic_cash: pd.DataFrame,
        ) -> pd.Series:
            wibor1m_frame = frames.get("WIBOR1M")
            synthetic_derived = build_standard_two_asset_data(
                wig=frames["WIG"],
                tbsp=frames["TBSP"],
                mmf=synthetic_cash,
                wibor1m=wibor1m_frame,
                pl10y=frames["PL10Y"],
                de10y=frames["DE10Y"],
                mmf_floor=str(frames["WIG"].index.min().date()),
            )
            wf_eq_s, wf_res_eq_s, trades_eq_s = walk_forward(
                df=frames["WIG"],
                cash_df=synthetic_derived["mmf_ext"],
                train_years=train_y,
                test_years=test_y,
                grid_config=self.equity_grid_config,
                use_atr_stop=use_atr_eq,
                n_jobs=1,
            )
            wf_bd_s, wf_res_bd_s, trades_bd_s = walk_forward(
                df=frames["TBSP"],
                cash_df=synthetic_derived["mmf_ext"],
                train_years=train_y,
                test_years=test_y,
                filter_modes_override=["ma"],
                grid_config=self.bond_grid_config,
                n_jobs=1,
                entry_gate_series=synthetic_derived["bond_gate"],
            )
            signals = {
                "WIG": build_signal_series(wf_eq_s, trades_eq_s),
                "TBSP": build_signal_series(wf_bd_s, trades_bd_s),
            }
            return allocation_walk_forward(
                equity_returns=frames["WIG"]["Zamkniecie"].pct_change().dropna(),
                bond_returns=frames["TBSP"]["Zamkniecie"].pct_change().dropna(),
                mmf_returns=synthetic_derived["ret_mmf"],
                sig_equity_full=signals["WIG"],
                sig_bond_full=signals["TBSP"],
                sig_equity_oos=signals["WIG"].loc[wf_eq_s.index.min() :],
                sig_bond_oos=signals["TBSP"].loc[wf_eq_s.index.min() :],
                wf_results_eq=wf_res_eq_s,
                wf_results_bd=wf_res_bd_s,
            )[0]

        if self.n_boot_portfolio > 0:
            context_frames = {
                "PL10Y": add_ohlc_columns_for_context(pl10y),
                "DE10Y": add_ohlc_columns_for_context(de10y),
            }
            if wibor1m is not None:
                context_frames["WIBOR1M"] = add_ohlc_columns_for_context(wibor1m)
            bootstrap_results = self.rob_engine.run_portfolio_bootstrap_test(
                frames={"WIG": wig, "TBSP": tbsp, **context_frames},
                cash_df=cash,
                runner=bootstrap_runner,
                n_samples=self.n_boot_portfolio,
                sample_offset=sample_offset,
            )
            results["bootstrap"] = bootstrap_results
            if emit_report:
                bootstrap_summary = analyze_portfolio_robustness(
                    results_df=bootstrap_results,
                    baseline_metrics=compute_metrics(equity=baseline_portfolio),
                    thresholds=PORTFOLIO_THRESHOLDS_BOOTSTRAP,
                    label="PENSION PORTFOLIO BOOTSTRAP",
                )
                summaries.append(bootstrap_summary)
        if emit_report and summaries:
            logging.info("PENSION PORTFOLIO FINAL VERDICT: %s", combine_portfolio_verdicts(summaries))
        return results

    def _run_global_portfolio_robustness(
        self,
        frames: dict[str, pd.DataFrame],
        cash: pd.DataFrame,
        wf_results: dict[str, pd.DataFrame],
        wf_configs: dict[str, dict[str, object]],
        returns: dict[str, pd.Series],
        bond_gate: pd.Series,
        asset_keys: list[str],
        settings: object,
        train_y: int,
        test_y: int,
        baseline_portfolio: pd.Series,
        sample_offset: int = 0,
        emit_report: bool = True,
    ) -> dict[str, pd.DataFrame]:
        legs = [
            PortfolioLeg(
                name=name,
                best_params=extract_best_params_from_wf_results(wf_results[name]),
                frame=frames[name],
            )
            for name in asset_keys
        ]
        entry_gates = {"TBSP": bond_gate}
        results: dict[str, pd.DataFrame] = {}
        summaries: list[dict[str, object]] = []

        def allocate(
            signal_map: dict[str, pd.Series],
            return_map: dict[str, pd.Series],
            cash_returns: pd.Series,
            schedule: pd.DataFrame,
        ) -> pd.Series:
            equity, unused_weights, unused_realloc, unused_alloc_df = allocation_walk_forward_n(
                returns_dict=return_map,
                signals_full_dict=signal_map,
                signals_oos_dict=signal_map,
                mmf_returns=cash_returns,
                wf_results_ref=schedule,
                asset_keys=asset_keys,
                train_years=train_y,
                asset_caps=settings.asset_caps,
                optional_keys=settings.optional_keys,
                min_delta=settings.min_delta,
                delta_tol=settings.delta_tol,
            )
            return equity

        def mc_runner(universe: dict[str, dict[int, dict]]) -> pd.Series:
            parameterized = run_parameterized_signals(
                universe=universe,
                legs=legs,
                wf_results=wf_results,
                cash_df=cash,
                entry_gates=entry_gates,
            )
            return allocate(
                signal_map=parameterized,
                return_map=returns,
                cash_returns=cash["Zamkniecie"].pct_change().dropna(),
                schedule=wf_results["TBSP"],
            )

        if self.n_mc_portfolio > 0:
            mc_results = self.rob_engine.run_portfolio_mc_test(
                legs=legs,
                runner=mc_runner,
                n_samples=self.n_mc_portfolio,
                sample_offset=sample_offset,
            )
            results["mc"] = mc_results
            if emit_report:
                mc_summary = analyze_portfolio_robustness(
                    results_df=mc_results,
                    baseline_metrics=compute_metrics(equity=baseline_portfolio),
                    thresholds=PORTFOLIO_THRESHOLDS_MC,
                    label="GLOBAL PORTFOLIO MC",
                )
                summaries.append(mc_summary)

        def bootstrap_runner(
            synthetic_frames: dict[str, pd.DataFrame],
            synthetic_cash: pd.DataFrame,
        ) -> pd.Series:
            synthetic_returns: dict[str, pd.Series] = {}
            synthetic_signals: dict[str, pd.Series] = {}
            synthetic_schedule = pd.DataFrame()
            for name in asset_keys:
                config = dict(wf_configs[name])
                leg_train = int(config.pop("train_years", train_y))
                wf_equity, wf_result, wf_trades = walk_forward(
                    df=synthetic_frames[name],
                    cash_df=synthetic_cash,
                    train_years=leg_train,
                    test_years=test_y,
                    n_jobs=1,
                    **config,
                )
                if name == "TBSP":
                    synthetic_schedule = wf_result
                synthetic_signals[name] = build_signal_series(wf_equity, wf_trades)
                synthetic_returns[name] = synthetic_frames[name]["Zamkniecie"].pct_change().dropna()
            return allocate(
                signal_map=synthetic_signals,
                return_map=synthetic_returns,
                cash_returns=synthetic_cash["Zamkniecie"].pct_change().dropna(),
                schedule=synthetic_schedule,
            )

        if self.n_boot_portfolio > 0:
            bootstrap_results = self.rob_engine.run_portfolio_bootstrap_test(
                frames=frames,
                cash_df=cash,
                runner=bootstrap_runner,
                n_samples=self.n_boot_portfolio,
                sample_offset=sample_offset,
            )
            results["bootstrap"] = bootstrap_results
            if emit_report:
                bootstrap_summary = analyze_portfolio_robustness(
                    results_df=bootstrap_results,
                    baseline_metrics=compute_metrics(equity=baseline_portfolio),
                    thresholds=PORTFOLIO_THRESHOLDS_BOOTSTRAP,
                    label="GLOBAL PORTFOLIO BOOTSTRAP",
                )
                summaries.append(bootstrap_summary)
        if emit_report and summaries:
            logging.info("GLOBAL PORTFOLIO FINAL VERDICT: %s", combine_portfolio_verdicts(summaries))
        return results

    def _save_validation_chart(
        self,
        strategy_equity: pd.Series,
        bh_equity: pd.Series | None,
        title: str,
        filename: str,
    ) -> None:
        """Generates a 2-panel OOS validation chart (Equity + Drawdown)."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = OUTPUT_DIR / filename

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.2)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Panel 1: Equity
        ax1.plot(
            strategy_equity.index,
            strategy_equity.values,
            label="Strategy (OOS)",
            color="steelblue",
            linewidth=2,
        )
        if bh_equity is not None:
            bh_aligned = bh_equity.reindex(index=strategy_equity.index).ffill()
            bh_norm = bh_aligned / bh_aligned.iloc[0]
            ax1.plot(
                bh_norm.index,
                bh_norm.values,
                label="Buy & Hold",
                color="grey",
                linestyle="--",
                alpha=0.7,
            )

        ax1.set_title(label=title, fontsize=14, fontweight="bold")
        ax1.set_ylabel(ylabel="Normalised Equity")
        ax1.legend(loc="upper left")
        ax1.grid(visible=True, alpha=0.3)

        # Panel 2: Drawdown
        dd_strat = strategy_equity / strategy_equity.cummax() - 1
        ax2.fill_between(
            dd_strat.index,
            dd_strat.values,
            0,
            color="steelblue",
            alpha=0.3,
            label="Strategy DD",
        )

        if bh_equity is not None:
            dd_bh = bh_norm / bh_norm.cummax() - 1
            ax2.plot(dd_bh.index, dd_bh.values, color="grey", linestyle="--", alpha=0.5)

        ax2.set_ylabel(ylabel="Drawdown")
        ax2.grid(visible=True, alpha=0.3)
        ax2.set_ylim(top=0)

        plt.savefig(fname=chart_path, dpi=72, bbox_inches="tight")
        plt.close(fig=fig)
        logging.info(msg=f"Validation chart saved to: {chart_path}")

    # ------------------------------------------------------------------
    # SINGLE
    # ------------------------------------------------------------------

    def simulate_single(
        self,
        train_y: int,
        test_y: int,
        stop_type: str,
        df: pd.DataFrame,
        cash_df: pd.DataFrame,
    ) -> SingleValidationSimulation:
        use_atr = stop_type == "atr"
        wf_eq, wf_res, wf_tr = walk_forward(
            df=df,
            cash_df=cash_df,
            train_years=train_y,
            test_years=test_y,
            grid_config=self.equity_grid_config,
            use_atr_stop=use_atr,
            n_jobs=get_n_jobs(),
        )
        return SingleValidationSimulation(
            wf_equity=wf_eq,
            wf_results=wf_res,
            wf_trades=wf_tr,
        )

    def report_single(
        self,
        asset_name: str,
        train_y: int,
        test_y: int,
        stop_type: str,
        df: pd.DataFrame,
        cash_df: pd.DataFrame,
        simulation: SingleValidationSimulation,
    ) -> None:
        use_atr = stop_type == "atr"
        wf_eq = simulation.wf_equity

        bh_eq, unused_bh_metrics = compute_buy_and_hold(
            df=df,
            price_col="Zamkniecie",
            start=wf_eq.index.min(),
            end=wf_eq.index.max(),
        )
        self._save_validation_chart(
            strategy_equity=wf_eq,
            bh_equity=bh_eq,
            title=f"OOS Validation: {asset_name} ({train_y}+{test_y} {stop_type})",
            filename=f"validate_{asset_name.lower()}_{train_y}_{test_y}.png",
        )

        if self.n_mc > 0:
            mc_results = self.rob_engine.run_mc_test(
                wf_results=simulation.wf_results,
                df=df,
                cash_df=cash_df,
                n_samples=self.n_mc,
            )
            analyze_robustness(
                results_df=mc_results,
                baseline_metrics=compute_metrics(wf_eq),
                thresholds=EQUITY_THRESHOLDS_MC,
            )

        if self.n_boot > 0:
            bb_results = self.rob_engine.run_bootstrap_test(
                df=df,
                cash_df=cash_df,
                n_samples=self.n_boot,
                train_years=train_y,
                test_years=test_y,
                grid_config=self.equity_grid_config,
                use_atr_stop=use_atr,
            )
            analyze_bootstrap(
                results_df=bb_results,
                baseline_metrics=compute_metrics(wf_eq),
                thresholds=EQUITY_THRESHOLDS_BOOTSTRAP,
            )

    # ------------------------------------------------------------------
    # Shared bond leg (PENSION and GLOBAL)
    # ------------------------------------------------------------------

    def simulate_bond_leg(
        self,
        tbsp_df: pd.DataFrame,
        cash_df: pd.DataFrame,
        bond_gate: pd.Series,
        train_y: int,
        test_y: int,
    ) -> SingleValidationSimulation:
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=tbsp_df,
            cash_df=cash_df,
            train_years=train_y,
            test_years=test_y,
            filter_modes_override=["ma"],
            grid_config=self.bond_grid_config,
            n_jobs=get_n_jobs(),
            entry_gate_series=bond_gate,
        )
        return SingleValidationSimulation(
            wf_equity=wf_bd,
            wf_results=wf_res_bd,
            wf_trades=wf_tr_bd,
        )

    def report_bond_leg(
        self,
        tbsp_df: pd.DataFrame,
        cash_df: pd.DataFrame,
        train_y: int,
        test_y: int,
        simulation: SingleValidationSimulation,
    ) -> None:
        if self.n_mc > 0:
            mc_bd = self.rob_engine.run_mc_test(
                wf_results=simulation.wf_results,
                df=tbsp_df,
                cash_df=cash_df,
                n_samples=self.n_mc,
            )
            analyze_robustness(
                results_df=mc_bd,
                baseline_metrics=compute_metrics(equity=simulation.wf_equity),
                thresholds=BOND_THRESHOLDS_MC,
            )

        if self.n_boot > 0:
            bb_bd = self.rob_engine.run_bootstrap_test(
                df=tbsp_df,
                cash_df=cash_df,
                n_samples=self.n_boot,
                train_years=train_y,
                test_years=test_y,
                filter_modes_override=["ma"],
                grid_config=self.bond_grid_config,
                use_atr_stop=False,
            )
            analyze_bootstrap(
                results_df=bb_bd,
                baseline_metrics=compute_metrics(equity=simulation.wf_equity),
                thresholds=BOND_THRESHOLDS_BOOTSTRAP,
            )

    # ------------------------------------------------------------------
    # PENSION
    # ------------------------------------------------------------------

    def simulate_pension(
        self,
        market: PensionMarketContext,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> PensionValidationSimulation:
        derived = market.derived
        equity_simulation = self.simulate_single(
            train_y=train_y,
            test_y=test_y,
            stop_type=stop_type_eq,
            df=market.wig,
            cash_df=derived["mmf_ext"],
        )
        bond_simulation = self.simulate_bond_leg(
            tbsp_df=market.tbsp,
            cash_df=derived["mmf_ext"],
            bond_gate=derived["bond_gate"],
            train_y=train_y,
            test_y=test_y,
        )
        sig_eq = build_signal_series(
            wf_equity=equity_simulation.wf_equity,
            wf_trades=equity_simulation.wf_trades,
        )
        sig_bd = build_signal_series(
            wf_equity=bond_simulation.wf_equity,
            wf_trades=bond_simulation.wf_trades,
        )
        oos_start = equity_simulation.wf_equity.index.min()
        port_eq, unused_weights, unused_realloc, alloc_df = allocation_walk_forward(
            equity_returns=derived["ret_eq"],
            bond_returns=derived["ret_bd"],
            mmf_returns=derived["ret_mmf"],
            sig_equity_full=sig_eq,
            sig_bond_full=sig_bd,
            sig_equity_oos=sig_eq.loc[oos_start:],
            sig_bond_oos=sig_bd.loc[oos_start:],
            wf_results_eq=equity_simulation.wf_results,
            wf_results_bd=bond_simulation.wf_results,
        )
        return PensionValidationSimulation(
            equity=equity_simulation,
            bond=bond_simulation,
            sig_equity=sig_eq,
            sig_bond=sig_bd,
            portfolio_equity=port_eq,
            alloc_df=alloc_df,
        )

    def report_pension(
        self,
        market: PensionMarketContext,
        simulation: PensionValidationSimulation,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> None:
        derived = market.derived
        use_atr_eq = stop_type_eq == "atr"
        port_eq = simulation.portfolio_equity
        oos_start = simulation.equity.wf_equity.index.min()

        logging.info("\n" + "=" * 60 + "\n--- Component 1: WIG Robustness ---\n" + "=" * 60)
        self.report_single(
            asset_name="WIG",
            train_y=train_y,
            test_y=test_y,
            stop_type=stop_type_eq,
            df=market.wig,
            cash_df=derived["mmf_ext"],
            simulation=simulation.equity,
        )

        logging.info("\n" + "=" * 60 + "\n--- Component 2: TBSP Robustness ---\n" + "=" * 60)
        self.report_bond_leg(
            tbsp_df=market.tbsp,
            cash_df=derived["mmf_ext"],
            train_y=train_y,
            test_y=test_y,
            simulation=simulation.bond,
        )

        self._run_pension_portfolio_robustness(
            wig=market.wig,
            tbsp=market.tbsp,
            cash=derived["mmf_ext"],
            bond_gate=derived["bond_gate"],
            wf_eq=simulation.equity.wf_equity,
            wf_res_eq=simulation.equity.wf_results,
            wf_res_bd=simulation.bond.wf_results,
            ret_eq=derived["ret_eq"],
            ret_bd=derived["ret_bd"],
            ret_mmf=derived["ret_mmf"],
            baseline_portfolio=port_eq,
            train_y=train_y,
            test_y=test_y,
            use_atr_eq=use_atr_eq,
            pl10y=market.pl10y,
            de10y=market.de10y,
            wibor1m=market.wibor1m,
        )

        bh_wig, unused_bh_metrics = compute_buy_and_hold(
            df=market.wig,
            price_col="Zamkniecie",
            start=port_eq.index.min(),
            end=port_eq.index.max(),
        )
        self._save_validation_chart(
            strategy_equity=port_eq,
            bh_equity=bh_wig,
            title=f"OOS Validation: PENSION Portfolio ({train_y}+{test_y})",
            filename=f"validate_pension_{train_y}_{test_y}.png",
        )

        if self.run_weights_perturb:
            logging.info(
                "\n"
                + "=" * 80
                + "\n--- Level 3: Allocation Weight Perturbation Test (PENSION) ---\n"
                + "=" * 80,
            )
            robust_df = allocation_weight_robustness(
                alloc_results_df=simulation.alloc_df,
                equity_returns=derived["ret_eq"],
                bond_returns=derived["ret_bd"],
                mmf_returns=derived["ret_mmf"],
                sig_equity_oos=simulation.sig_equity.loc[oos_start:],
                sig_bond_oos=simulation.sig_bond.loc[oos_start:],
                baseline_metrics=compute_metrics(equity=port_eq),
            )
            print_allocation_robustness_report(results_df=robust_df)

    # ------------------------------------------------------------------
    # GLOBAL
    # ------------------------------------------------------------------

    def simulate_global(
        self,
        market: GlobalMarketContext,
        variant: str,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> GlobalValidationSimulation:
        variant_config = ASSET_REGISTRY[variant]
        use_atr = stop_type_eq == "atr"
        derived = market.pension.derived
        mmf_ext = derived["mmf_ext"]
        tbsp_df = market.pension.tbsp

        rets_dict: dict[str, pd.Series] = {}
        sigs_full: dict[str, pd.Series] = {}
        frames_dict: dict[str, pd.DataFrame] = {}
        wf_results_dict: dict[str, pd.DataFrame] = {}
        wf_equity_dict: dict[str, pd.Series] = {}
        wf_configs: dict[str, dict[str, object]] = {}
        leg_train_years: dict[str, int] = {}

        for label, spec in market.assets.items():
            ret_s, proc_px = prepare_asset_series(
                label=label,
                spec=spec,
            )
            rets_dict[label] = ret_s
            frames_dict[label] = proc_px
            leg_train = resolve_train_years(
                cfg=variant_config,
                spec=spec,
                default_train=train_y,
            )
            leg_train_years[label] = leg_train
            leg_grid_config = wf_grid_config(spec=spec)

            wf_e, wf_r, wf_t = walk_forward(
                df=proc_px,
                cash_df=mmf_ext,
                train_years=leg_train,
                test_years=test_y,
                n_jobs=get_n_jobs(),
                use_atr_stop=use_atr,
                grid_config=leg_grid_config,
            )
            sigs_full[label] = build_signal_series(wf_equity=wf_e, wf_trades=wf_t)
            wf_results_dict[label] = wf_r
            wf_equity_dict[label] = wf_e
            wf_configs[label] = {
                "grid_config": leg_grid_config,
                "train_years": leg_train,
                "use_atr_stop": use_atr,
            }

        bond_simulation = self.simulate_bond_leg(
            tbsp_df=tbsp_df,
            cash_df=mmf_ext,
            bond_gate=derived["bond_gate"],
            train_y=train_y,
            test_y=test_y,
        )
        rets_dict["TBSP"] = tbsp_df["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(
            wf_equity=bond_simulation.wf_equity,
            wf_trades=bond_simulation.wf_trades,
        )
        frames_dict["TBSP"] = tbsp_df
        wf_results_dict["TBSP"] = bond_simulation.wf_results
        wf_configs["TBSP"] = {
            "filter_modes_override": ["ma"],
            "grid_config": self.bond_grid_config,
            "use_atr_stop": False,
            "entry_gate_series": derived["bond_gate"],
            "train_years": train_y,
        }

        logging.info("\n" + "-" * 60 + "\nRunning N-Asset allocation walk-forward...\n" + "-" * 60)
        settings = market.settings
        port_eq, unused_weights, unused_realloc, alloc_df = allocation_walk_forward_n(
            returns_dict=rets_dict,
            signals_full_dict=sigs_full,
            signals_oos_dict=sigs_full,
            mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(),
            wf_results_ref=bond_simulation.wf_results,
            asset_keys=list(rets_dict.keys()),
            train_years=train_y,
            asset_caps=settings.asset_caps,
            optional_keys=settings.optional_keys,
            min_delta=settings.min_delta,
            delta_tol=settings.delta_tol,
        )
        return GlobalValidationSimulation(
            bond=bond_simulation,
            returns=rets_dict,
            signals=sigs_full,
            frames=frames_dict,
            wf_results=wf_results_dict,
            wf_equity=wf_equity_dict,
            wf_configs=wf_configs,
            leg_train_years=leg_train_years,
            portfolio_equity=port_eq,
            alloc_df=alloc_df,
        )

    def report_global(
        self,
        market: GlobalMarketContext,
        simulation: GlobalValidationSimulation,
        variant: str,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> None:
        use_atr = stop_type_eq == "atr"
        derived = market.pension.derived
        mmf_ext = derived["mmf_ext"]
        settings = market.settings
        port_eq = simulation.portfolio_equity

        for label, spec in market.assets.items():
            logging.info("\n" + "=" * 60 + f"\n--- Component Robustness: {label} ---\n" + "=" * 60)
            wf_e = simulation.wf_equity[label]
            if self.n_mc > 0:
                mc_res = self.rob_engine.run_mc_test(
                    wf_results=simulation.wf_results[label],
                    df=simulation.frames[label],
                    cash_df=mmf_ext,
                    n_samples=self.n_mc,
                )
                analyze_robustness(
                    results_df=mc_res,
                    baseline_metrics=compute_metrics(equity=wf_e),
                    thresholds=THRESHOLDS_MC[spec.asset_class],
                )

            if self.n_boot > 0:
                bb_res = self.rob_engine.run_bootstrap_test(
                    df=simulation.frames[label],
                    cash_df=mmf_ext,
                    n_samples=self.n_boot,
                    train_years=simulation.leg_train_years[label],
                    test_years=test_y,
                    use_atr_stop=use_atr,
                    grid_config=wf_grid_config(spec=spec),
                )
                analyze_bootstrap(
                    results_df=bb_res,
                    baseline_metrics=compute_metrics(equity=wf_e),
                    thresholds=THRESHOLDS_BOOTSTRAP[spec.asset_class],
                )

        logging.info("\n" + "=" * 60 + "\n--- Component Robustness: TBSP ---\n" + "=" * 60)
        self.report_bond_leg(
            tbsp_df=market.pension.tbsp,
            cash_df=mmf_ext,
            train_y=train_y,
            test_y=test_y,
            simulation=simulation.bond,
        )

        self._run_global_portfolio_robustness(
            frames=simulation.frames,
            cash=mmf_ext,
            wf_results=simulation.wf_results,
            wf_configs=simulation.wf_configs,
            returns=simulation.returns,
            bond_gate=derived["bond_gate"],
            asset_keys=list(simulation.returns.keys()),
            settings=settings,
            train_y=train_y,
            test_y=test_y,
            baseline_portfolio=port_eq,
        )

        bh_wig, unused_bh_metrics = compute_buy_and_hold(
            df=market.pension.wig,
            price_col="Zamkniecie",
            start=port_eq.index.min(),
            end=port_eq.index.max(),
        )
        self._save_validation_chart(
            strategy_equity=port_eq,
            bh_equity=bh_wig,
            title=f"OOS Validation: GLOBAL {variant} ({train_y}+{test_y})",
            filename=f"validate_{variant.lower()}_{train_y}_{test_y}.png",
        )

        if self.run_weights_perturb:
            logging.info(
                "\n"
                + "=" * 80
                + "\n--- Level 3: Allocation Weight Perturbation Test (GLOBAL) ---\n"
                + "=" * 80,
            )
            robust_df = allocation_weight_robustness_n(
                alloc_results_df=simulation.alloc_df,
                returns_dict=simulation.returns,
                mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(),
                signals_oos_dict={
                    asset_label: asset_signal.loc[port_eq.index.min() :]
                    for asset_label, asset_signal in simulation.signals.items()
                },
                asset_keys=list(simulation.returns.keys()),
                baseline_metrics=compute_metrics(equity=port_eq),
                focus_asset="WIG",
                asset_caps=settings.asset_caps,
                min_delta=settings.min_delta,
                delta_tol=settings.delta_tol,
            )
            print_allocation_robustness_report_n(results_df=robust_df, focus_asset="WIG")


# ---------------------------------------------------------------------------
# Validation pipelines (Template Method: load -> simulate -> robustness reports)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SingleValidationSimulation:
    wf_equity: pd.Series
    wf_results: pd.DataFrame
    wf_trades: pd.DataFrame


@dataclass(frozen=True, eq=False)
class PensionValidationSimulation:
    equity: SingleValidationSimulation
    bond: SingleValidationSimulation
    sig_equity: pd.Series
    sig_bond: pd.Series
    portfolio_equity: pd.Series
    alloc_df: pd.DataFrame


@dataclass(frozen=True, eq=False)
class GlobalValidationSimulation:
    bond: SingleValidationSimulation
    returns: dict[str, pd.Series]
    signals: dict[str, pd.Series]
    frames: dict[str, pd.DataFrame]
    wf_results: dict[str, pd.DataFrame]
    wf_equity: dict[str, pd.Series]
    wf_configs: dict[str, dict[str, object]]
    leg_train_years: dict[str, int]
    portfolio_equity: pd.Series
    alloc_df: pd.DataFrame


@dataclass(frozen=True, eq=False)
class SingleValidationData:
    asset_df: pd.DataFrame
    cash_df: pd.DataFrame


class SingleAssetValidationPipeline(BaseStrategyPipeline[SingleValidationData, SingleValidationSimulation]):
    def __init__(
        self,
        manager: ValidationManager,
        asset_name: str,
        train_y: int,
        test_y: int,
        stop_type: str,
    ) -> None:
        self.manager = manager
        self.asset_name = asset_name
        self.train_y = train_y
        self.test_y = test_y
        self.stop_type = stop_type

    def load_data(self) -> SingleValidationData:
        return SingleValidationData(
            asset_df=self.manager.facade.get_asset_series(
                asset_key=self.asset_name,
                local_only=True,
            ),
            cash_df=self.manager.facade.get_mmf(),
        )

    def run_simulation(
        self,
        market_data: SingleValidationData,
    ) -> SingleValidationSimulation:
        logging.info(
            f"VALIDATING SINGLE ASSET: {self.asset_name} | {self.train_y}+{self.test_y} | {self.stop_type}",
        )
        return self.manager.simulate_single(
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type=self.stop_type,
            df=market_data.asset_df,
            cash_df=market_data.cash_df,
        )

    def generate_reports(
        self,
        market_data: SingleValidationData,
        simulation: SingleValidationSimulation,
    ) -> dict[str, Any]:
        self.manager.report_single(
            asset_name=self.asset_name,
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type=self.stop_type,
            df=market_data.asset_df,
            cash_df=market_data.cash_df,
            simulation=simulation,
        )
        return {}


class PensionValidationPipeline(BaseStrategyPipeline[PensionMarketContext, PensionValidationSimulation]):
    def __init__(
        self,
        manager: ValidationManager,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> None:
        self.manager = manager
        self.train_y = train_y
        self.test_y = test_y
        self.stop_type_eq = stop_type_eq

    def load_data(self) -> PensionMarketContext:
        return self.manager.facade.get_pension_context()

    def run_simulation(
        self,
        market_data: PensionMarketContext,
    ) -> PensionValidationSimulation:
        logging.info(
            f"VALIDATING PENSION PORTFOLIO | Train: {self.train_y} | Test: {self.test_y} "
            f"| EQ Stop: {self.stop_type_eq}",
        )
        return self.manager.simulate_pension(
            market=market_data,
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type_eq=self.stop_type_eq,
        )

    def generate_reports(
        self,
        market_data: PensionMarketContext,
        simulation: PensionValidationSimulation,
    ) -> dict[str, Any]:
        self.manager.report_pension(
            market=market_data,
            simulation=simulation,
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type_eq=self.stop_type_eq,
        )
        return {}


class GlobalValidationPipeline(BaseStrategyPipeline[GlobalMarketContext, GlobalValidationSimulation]):
    def __init__(
        self,
        manager: ValidationManager,
        variant: str,
        train_y: int,
        test_y: int,
        stop_type_eq: str,
    ) -> None:
        self.manager = manager
        self.variant = variant
        self.train_y = train_y
        self.test_y = test_y
        self.stop_type_eq = stop_type_eq

    def load_data(self) -> GlobalMarketContext:
        return self.manager.facade.get_global_context(variant=self.variant)

    def run_simulation(
        self,
        market_data: GlobalMarketContext,
    ) -> GlobalValidationSimulation:
        logging.info(
            f"VALIDATING GLOBAL - {self.variant} | Train: {self.train_y} | Test: {self.test_y} "
            f"| Stop: {self.stop_type_eq}",
        )
        return self.manager.simulate_global(
            market=market_data,
            variant=self.variant,
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type_eq=self.stop_type_eq,
        )

    def generate_reports(
        self,
        market_data: GlobalMarketContext,
        simulation: GlobalValidationSimulation,
    ) -> dict[str, Any]:
        self.manager.report_global(
            market=market_data,
            simulation=simulation,
            variant=self.variant,
            train_y=self.train_y,
            test_y=self.test_y,
            stop_type_eq=self.stop_type_eq,
        )
        return {}


def build_validation_pipeline(
    mode: str,
    manager: ValidationManager,
    asset: str | None,
    train_y: int,
    test_y: int,
    stop_type: str,
) -> BaseStrategyPipeline[Any, Any]:
    if mode == "SINGLE":
        return SingleAssetValidationPipeline(
            manager=manager,
            asset_name=str(asset),
            train_y=train_y,
            test_y=test_y,
            stop_type=stop_type,
        )
    if mode == "PENSION":
        return PensionValidationPipeline(
            manager=manager,
            train_y=train_y,
            test_y=test_y,
            stop_type_eq=stop_type,
        )
    return GlobalValidationPipeline(
        manager=manager,
        variant=str(asset),
        train_y=train_y,
        test_y=test_y,
        stop_type_eq=stop_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Robustness Validator")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL"], required=True)
    parser.add_argument("--asset", help="Dla mode SINGLE lub GLOBAL (np. WIG20TR, GLOBAL_A)")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--stop", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--n_mc", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--n-mc-portfolio", type=int, default=None, help="Portfolio MC samples (default: --n_mc).")
    parser.add_argument(
        "--n-boot-portfolio", type=int, default=None, help="Portfolio bootstrap samples (default: --n_boot)."
    )
    parser.add_argument(
        "--weights_perturb",
        action="store_true",
        help="Uruchom dodatkowy test perturbacji wag alokacji.",
    )
    args = parser.parse_args()

    log_file = OUTPUT_DIR / f"validate_{args.mode.lower()}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    DataUpdater().run_full_update(get_funds=False)
    validator = ValidationManager(
        n_mc=args.n_mc,
        n_boot=args.n_boot,
        run_weights_perturb=args.weights_perturb,
        n_mc_portfolio=args.n_mc_portfolio,
        n_boot_portfolio=args.n_boot_portfolio,
    )

    pipeline = build_validation_pipeline(
        mode=args.mode,
        manager=validator,
        asset=args.asset,
        train_y=args.train,
        test_y=args.test,
        stop_type=args.stop,
    )
    pipeline.execute_pipeline()


if __name__ == "__main__":
    main()
