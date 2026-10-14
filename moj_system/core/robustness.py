# -*- coding: utf-8 -*-
"""
moj_system/core/robustness.py
=============================
Engine for Monte Carlo Parameter Perturbation and Block Bootstrap.
Uses legacy simulation engine to ensure results consistency.
"""

import logging

import pandas as pd

from moj_system.core.portfolio_robustness import (
    BootstrapRunner,
    PortfolioFrames,
    PortfolioLeg,
    PortfolioRunner,
    run_portfolio_block_bootstrap,
    run_portfolio_monte_carlo,
)
from moj_system.core.robustness_engine import (
    extract_best_params_from_wf_results,
    extract_windows_from_wf_results,
    run_block_bootstrap_robustness,
    run_monte_carlo_robustness,
)

# Importujemy oryginalny silnik


class RobustnessEngine:
    def __init__(
        self,
        n_jobs: int = -1,
    ) -> None:
        self.n_jobs = n_jobs

    def run_mc_test(
        self,
        wf_results: pd.DataFrame,
        df: pd.DataFrame,
        cash_df: pd.DataFrame,
        n_samples: int = 100,
        perturb_pct: float = 0.20,
    ) -> pd.DataFrame:
        """Runs Monte Carlo parameter perturbation."""
        logging.info(f"Starting MC Perturbation Test (n={n_samples})...")

        windows = extract_windows_from_wf_results(wf_results)
        best_params = extract_best_params_from_wf_results(wf_results)

        mc_results_df = run_monte_carlo_robustness(
            best_params=best_params,
            windows=windows,
            df=df,
            cash_df=cash_df,
            vol_window=20,
            selected_mode="full",
            n_samples=n_samples,
            n_jobs=self.n_jobs,
            perturb_pct=perturb_pct,
            price_col="Zamkniecie",
        )
        return mc_results_df

    def run_bootstrap_test(
        self,
        df: pd.DataFrame,
        cash_df: pd.DataFrame,
        n_samples: int = 500,
        **wf_kwargs: object,
    ) -> pd.DataFrame:
        """Runs Block Bootstrap history reshuffling."""
        logging.info(f"Starting Block Bootstrap Test (n={n_samples})...")

        bb_results_df = run_block_bootstrap_robustness(
            df=df,
            cash_df=cash_df,
            n_samples=n_samples,
            block_size=250,
            **wf_kwargs,
        )
        return bb_results_df

    def run_portfolio_mc_test(
        self,
        legs: list[PortfolioLeg],
        runner: PortfolioRunner,
        n_samples: int = 1000,
        perturb_pct: float = 0.20,
        seed: int = 42,
        sample_offset: int = 0,
    ) -> pd.DataFrame:
        """Run end-to-end parameter perturbation for all portfolio legs."""
        logging.info(
            "Starting portfolio MC perturbation (n=%d, legs=%d)...",
            n_samples,
            len(legs),
        )
        return run_portfolio_monte_carlo(
            legs=legs,
            runner=runner,
            n_samples=n_samples,
            perturb_pct=perturb_pct,
            seed=seed,
            n_jobs=self.n_jobs,
            sample_offset=sample_offset,
        )

    def run_portfolio_bootstrap_test(
        self,
        frames: PortfolioFrames,
        cash_df: pd.DataFrame,
        runner: BootstrapRunner,
        n_samples: int = 500,
        block_size: int = 250,
        seed: int = 1000,
        sample_offset: int = 0,
    ) -> pd.DataFrame:
        """Run joint full-OHLC block bootstrap for the complete portfolio."""
        logging.info(
            "Starting portfolio block bootstrap (n=%d, legs=%d, block=%d)...",
            n_samples,
            len(frames),
            block_size,
        )
        return run_portfolio_block_bootstrap(
            frames=frames,
            cash_df=cash_df,
            runner=runner,
            n_samples=n_samples,
            block_size=block_size,
            seed=seed,
            n_jobs=self.n_jobs,
            sample_offset=sample_offset,
        )
