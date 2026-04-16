# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:20:16 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/scripts/validate_robustness.py
=========================================
Deep Validation Engine. Performs Level 1 (MC) and Level 2 (Bootstrap) 
testing for chosen configurations of Single, Pension, or Global strategies.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import tempfile
import matplotlib
matplotlib.use('Agg')

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'current_development'))

from moj_system.config import (
    ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, 
    EQUITY_THRESHOLDS_MC, EQUITY_THRESHOLDS_BOOTSTRAP,
    BOND_THRESHOLDS_MC, BOND_THRESHOLDS_BOOTSTRAP
)
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.research import get_common_oos_start

# Core Strategy & Robustness Imports
from current_development.strategy_test_library import walk_forward, compute_metrics, get_n_jobs
from current_development.multiasset_library import build_signal_series, allocation_walk_forward, build_standard_two_asset_data
from current_development.global_equity_library import build_return_series, build_price_df_from_returns, allocation_walk_forward_n
from current_development.mc_robustness import analyze_robustness, analyze_bootstrap

class ValidationManager:
    def __init__(self, n_mc, n_boot):
        self.n_mc = n_mc
        self.n_boot = n_boot
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    def validate_single(self, asset_name, train_y, test_y, stop_type, common_start):
        logging.info(f"VALIDATING SINGLE ASSET: {asset_name} | {train_y}+{test_y} | {stop_type}")
        
        df = load_local_csv(asset_name.lower(), asset_name)
        cash_df = load_local_csv("fund_2720", "MMF")
        use_atr = (stop_type == "atr")

        # 1. Base Walk-Forward
        wf_eq, wf_res, _ = walk_forward(
            df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
            use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs()
        )
        
        # 2. MC Test
        mc_results = self.rob_engine.run_mc_test(wf_res, df, cash_df, n_samples=self.n_mc)
        analyze_robustness(mc_results, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)

        # 3. Bootstrap Test
        if self.n_boot > 0:
            bb_results = self.rob_engine.run_bootstrap_test(
                df, cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y, use_atr_stop=use_atr
            )
            analyze_bootstrap(bb_results, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

    def validate_pension(self, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING PENSION PORTFOLIO | EQ Stop: {stop_type_eq}")
        
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
        derived = build_standard_two_asset_data(WIG, TBSP, MMF, None, PL10Y, DE10Y, "1995-01-02")

        # Robustness of components (WIG and TBSP separately)
        logging.info("--- Component 1: WIG Robustness ---")
        self.validate_single("WIG", train_y, test_y, stop_type_eq, WIG.index.min())
        
        logging.info("--- Component 2: TBSP Robustness ---")
        # TBSP validation (always fixed stop per config)
        wf_bd_eq, wf_bd_res, _ = walk_forward(TBSP, derived["mmf_ext"], train_y, test_y, filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"])
        mc_bd = self.rob_engine.run_mc_test(wf_bd_res, TBSP, derived["mmf_ext"], n_samples=self.n_mc)
        analyze_robustness(mc_bd, compute_metrics(wf_bd_eq), thresholds=BOND_THRESHOLDS_MC)

    def validate_global(self, variant, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING GLOBAL {variant} | Train: {train_y}")
        # Global validation typically focuses on the main equity component (WIG)
        # and the overall Portfolio metrics. 
        self.validate_single("WIG", train_y, test_y, stop_type_eq, pd.Timestamp("1995-01-02"))

def main():
    parser = argparse.ArgumentParser(description="Full Robustness Validator")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL"], required=True)
    parser.add_argument("--asset", help="Dla mode SINGLE (np. WIG20TR)")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--stop", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--n_mc", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=500)
    args = parser.parse_args()

    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    validator = ValidationManager(args.n_mc, args.n_boot)

    if args.mode == "SINGLE":
        # Calculate start just for this asset
        df = load_local_csv(args.asset.lower(), args.asset)
        common_start = get_common_oos_start({args.asset: df}, [(args.train, args.test)])
        validator.validate_single(args.asset, args.train, args.test, args.stop, common_start)

    elif args.mode == "PENSION":
        validator.validate_pension(args.train, args.test, args.stop)

    elif args.mode == "GLOBAL":
        validator.validate_global(args.asset, args.train, args.test, args.stop)

if __name__ == "__main__":
    main()