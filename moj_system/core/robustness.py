# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:35:24 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/core/robustness.py
=============================
Engine for Monte Carlo Parameter Perturbation and Block Bootstrap.
Uses legacy simulation engine to ensure results consistency.
"""

import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
# Importujemy oryginalny silnik
from current_development.strategy_test_library import compute_metrics, run_strategy_with_trades
from current_development.mc_robustness import (
    run_monte_carlo_robustness, analyze_robustness,
    run_block_bootstrap_robustness, analyze_bootstrap,
    extract_windows_from_wf_results, extract_best_params_from_wf_results
)

class RobustnessEngine:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def run_mc_test(self, wf_results, df, cash_df, n_samples=100, perturb_pct=0.20):
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
            price_col="Zamkniecie"
        )
        return mc_results_df

    def run_bootstrap_test(self, df, cash_df, n_samples=500, **wf_kwargs):
        """Runs Block Bootstrap history reshuffling."""
        logging.info(f"Starting Block Bootstrap Test (n={n_samples})...")
        
        bb_results_df = run_block_bootstrap_robustness(
            df=df,
            cash_df=cash_df,
            n_samples=n_samples,
            block_size=250,
            **wf_kwargs
        )
        return bb_results_df