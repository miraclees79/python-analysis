# moj_system/scripts/validate_robustness.py

# -*- coding: utf-8 -*-
"""
moj_system/scripts/validate_robustness.py
=========================================
Deep Validation Engine. Performs Level 1 (MC), Level 2 (Bootstrap),
and optionally Level 3 (Allocation Weight Perturbation) testing for
chosen configurations of Single, Pension, or Global strategies.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import tempfile
import matplotlib
matplotlib.use('Agg')

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CORE ENGINE IMPORTS ---
from moj_system.core.strategy_engine import get_n_jobs, walk_forward, compute_metrics
from moj_system.core.pension_engine import (
    build_signal_series, allocation_walk_forward, build_standard_two_asset_data,
    allocation_weight_robustness, print_allocation_robustness_report
)
from moj_system.core.global_engine import (
    build_return_series, build_price_df_from_returns, allocation_walk_forward_n,
    allocation_weight_robustness_n, print_allocation_robustness_report_n
)
from moj_system.core.robustness_engine import analyze_robustness, analyze_bootstrap
from moj_system.core.utils import build_mmf_extended
from moj_system.data.updater import DataUpdater
from moj_system.config import (
    ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, 
    EQUITY_THRESHOLDS_MC, EQUITY_THRESHOLDS_BOOTSTRAP, 
    BOND_THRESHOLDS_MC, BOND_THRESHOLDS_BOOTSTRAP
)
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.research import get_common_oos_start


class ValidationManager:
    def __init__(self, n_mc, n_boot, run_weights_perturb):
        self.n_mc = n_mc
        self.n_boot = n_boot
        self.run_weights_perturb = run_weights_perturb
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    def validate_single(self, asset_name, train_y, test_y, stop_type, df, cash_df):
        logging.info(f"VALIDATING SINGLE ASSET: {asset_name} | {train_y}+{test_y} | {stop_type}")
        use_atr = (stop_type == "atr")

        # 1. Base Walk-Forward
        wf_eq, wf_res, _ = walk_forward(
            df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
            X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs(), fast_mode=True
        )
        
        # 2. MC Test
        if self.n_mc > 0:
            mc_results = self.rob_engine.run_mc_test(wf_res, df, cash_df, n_samples=self.n_mc)
            analyze_robustness(mc_results, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)

        # 3. Bootstrap Test
        if self.n_boot > 0:
            bb_results = self.rob_engine.run_bootstrap_test(
                df, cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None
            )
            analyze_bootstrap(bb_results, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

    def validate_pension(self, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING PENSION PORTFOLIO | EQ Stop: {stop_type_eq}")
        
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
        derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR1M, PL10Y, DE10Y, "1995-01-02")
        use_atr_eq = (stop_type_eq == 'atr')
        
        logging.info("\n" + "="*60 + "\n--- Component 1: WIG Robustness ---\n" + "="*60)
        # Przekazujemy WIG już z nałożonym floor'em 1995-01-02 z portfela PENSION
        self.validate_single("WIG", train_y, test_y, stop_type_eq, WIG, derived["mmf_ext"])
        
        logging.info("\n" + "="*60 + "\n--- Component 2: TBSP Robustness ---\n" + "="*60)
        wf_bd_eq, wf_bd_res, _ = walk_forward(
            TBSP, derived["mmf_ext"], train_y, test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(), fast_mode=True
        )
        if self.n_mc > 0:
            mc_bd = self.rob_engine.run_mc_test(wf_bd_res, TBSP, derived["mmf_ext"], n_samples=self.n_mc)
            analyze_robustness(mc_bd, compute_metrics(wf_bd_eq), thresholds=BOND_THRESHOLDS_MC)
        if self.n_boot > 0:
            bb_bd = self.rob_engine.run_bootstrap_test(
                TBSP, derived["mmf_ext"], n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                filter_modes_override=["ma"], 
                X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
                fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
                use_atr_stop=False
            )
            analyze_bootstrap(bb_bd, compute_metrics(wf_bd_eq), thresholds=BOND_THRESHOLDS_BOOTSTRAP)

        # Level 3 - Perturbacja Wag
        if self.run_weights_perturb:
            logging.info("\n" + "="*80 + "\n--- Level 3: Allocation Weight Perturbation Test (PENSION) ---\n" + "="*80)
            
            wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
                WIG, derived["mmf_ext"], train_y, test_y, 
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr_eq, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None,
                n_jobs=get_n_jobs(), fast_mode=True
            )
            sig_eq_full = build_signal_series(wf_eq, wf_tr_eq)
            sig_bd_full = build_signal_series(wf_bd_eq, _)
            
            oos_start = max(wf_res_eq["TestStart"].min(), wf_bd_res["TestStart"].min())
            oos_end = min(wf_res_eq["TestEnd"].max(), wf_bd_res["TestEnd"].max())
            sig_eq_oos = sig_eq_full.loc[oos_start:oos_end]
            sig_bd_oos = sig_bd_full.loc[oos_start:oos_end]

            port_eq, _, _, alloc_df = allocation_walk_forward(
                derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], sig_eq_full, sig_bd_full,
                sig_eq_oos, sig_bd_oos, wf_res_eq, wf_bd_res
            )
            
            baseline_metrics = compute_metrics(port_eq)

            robust_df = allocation_weight_robustness(
                alloc_results_df=alloc_df,
                equity_returns=derived["ret_eq"],
                bond_returns=derived["ret_bd"],
                mmf_returns=derived["ret_mmf"],
                sig_equity_oos=sig_eq_oos,
                sig_bond_oos=sig_bd_oos,
                baseline_metrics=baseline_metrics
            )
            print_allocation_robustness_report(robust_df)

    def validate_global(self, variant, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING GLOBAL {variant} | Train: {train_y} | Stop: {stop_type_eq}")
        
        cfg = ASSET_REGISTRY[variant]
        mode, fx_hedged = cfg["mode"], cfg.get("fx_hedged", True)
        use_atr = (stop_type_eq == "atr")

        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        MMF = load_local_csv("fund_2720", "MMF")
        if WIBOR1M is not None:
            MMF = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02")
            
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        fx_map = {c: load_local_csv(f"{c.lower()}pln", f"{c}PLN")["Zamkniecie"] for c in ["USD", "EUR", "JPY"]}
        
        if mode == "global_equity":
            stoxx = build_and_upload(self.folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", self.creds_path)
            assets = {
                "WIG": (WIG, None), 
                "SP500": (load_local_csv("sp500", "SP500"), fx_map["USD"]), 
                "STOXX600": (stoxx, fx_map["EUR"]), 
                "Nikkei225": (load_local_csv("nikkei225", "Nikkei225"), fx_map["JPY"])
            }
        else:
            msciw = build_and_upload(self.folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", self.creds_path, is_msci_world=True)
            assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

        rets_dict, sigs_full = {}, {}
        
        # Testy komponentów: MC + Bootstrap na danych syntetycznych (skorygowanych o FX)
        for lbl, (px_df, fx_s) in assets.items():
            logging.info("\n" + "="*60 + f"\n--- Component Robustness: {lbl} ---\n" + "="*60)
            
            ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret_s, lbl)
            
            wf_e, wf_r, wf_t = walk_forward(
                proc_px, MMF, train_y, test_y, 
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None, 
                n_jobs=get_n_jobs(), fast_mode=True
            )
            sigs_full[lbl] = build_signal_series(wf_e, wf_t)

            if self.n_mc > 0:
                mc_res = self.rob_engine.run_mc_test(wf_r, proc_px, MMF, n_samples=self.n_mc)
                analyze_robustness(mc_res, compute_metrics(wf_e), thresholds=EQUITY_THRESHOLDS_MC)

            if self.n_boot > 0:
                bb_res = self.rob_engine.run_bootstrap_test(
                    proc_px, MMF, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                    X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                    fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                    use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None
                )
                analyze_bootstrap(bb_res, compute_metrics(wf_e), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

        # Komponent obligacyjny
        logging.info("\n" + "="*60 + "\n--- Component Robustness: TBSP ---\n" + "="*60)
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(), fast_mode=True
        )
        rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)

        if self.n_mc > 0:
            mc_bd = self.rob_engine.run_mc_test(wf_res_bd, TBSP, MMF, n_samples=self.n_mc)
            analyze_robustness(mc_bd, compute_metrics(wf_bd), thresholds=BOND_THRESHOLDS_MC)

        if self.n_boot > 0:
            bb_bd = self.rob_engine.run_bootstrap_test(
                TBSP, MMF, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                filter_modes_override=["ma"], 
                X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
                fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
                use_atr_stop=False
            )
            analyze_bootstrap(bb_bd, compute_metrics(wf_bd), thresholds=BOND_THRESHOLDS_BOOTSTRAP)

        # 3. Alokacja N-Asset i wyliczenie linii bazowej
        logging.info("\n" + "-"*60 + "\nRunning N-Asset allocation walk-forward...\n" + "-"*60)
        asset_keys = list(rets_dict.keys())
        
        port_eq, _, _, alloc_df = allocation_walk_forward_n(
            returns_dict=rets_dict,
            signals_full_dict=sigs_full,
            signals_oos_dict=sigs_full, 
            mmf_returns=MMF["Zamkniecie"].pct_change().dropna(),
            wf_results_ref=wf_res_bd, 
            asset_keys=asset_keys,
            train_years=train_y
        )
        
        baseline_metrics = compute_metrics(port_eq)

        # 4. Perturbacja wag (Level 3 - focus: WIG)
        if self.run_weights_perturb:
            logging.info("\n" + "="*80 + "\n--- Level 3: Allocation Weight Perturbation Test (GLOBAL) ---\n" + "="*80)
            
            oos_start = wf_res_bd["TestStart"].min()
            oos_end = wf_res_bd["TestEnd"].max()
            sigs_oos = {k: v.loc[oos_start:oos_end] for k, v in sigs_full.items()}

            focus_asset = "WIG"
            logging.info(f"Running Weight Perturbation Test (focus: {focus_asset})...")
            
            robust_df = allocation_weight_robustness_n(
                alloc_results_df=alloc_df,
                returns_dict=rets_dict,
                mmf_returns=MMF["Zamkniecie"].pct_change().dropna(),
                signals_oos_dict=sigs_oos,
                asset_keys=asset_keys,
                baseline_metrics=baseline_metrics,
                focus_asset=focus_asset
            )
            
            print_allocation_robustness_report_n(robust_df, focus_asset=focus_asset)


def main():
    parser = argparse.ArgumentParser(description="Full Robustness Validator")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL"], required=True)
    parser.add_argument("--asset", help="Dla mode SINGLE lub GLOBAL (np. WIG20TR, GLOBAL_A)")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--stop", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--n_mc", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--weights_perturb", action="store_true", help="Uruchom dodatkowy test perturbacji wag alokacji (PENSION/GLOBAL).")
    args = parser.parse_args()
    
    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(get_funds=False)    
    
    validator = ValidationManager(args.n_mc, args.n_boot, args.weights_perturb)

    if args.mode == "SINGLE":
        if not args.asset:
            sys.exit("Dla trybu SINGLE wymagany jest argument --asset.")
        df = load_local_csv(args.asset.lower(), args.asset)
        cash_df = load_local_csv("fund_2720", "MMF")
        validator.validate_single(args.asset, args.train, args.test, args.stop, df, cash_df)

    elif args.mode == "PENSION":
        validator.validate_pension(args.train, args.test, args.stop)

    elif args.mode == "GLOBAL":
        if not args.asset or args.asset not in ["GLOBAL_A", "GLOBAL_B"]:
            sys.exit("Dla trybu GLOBAL wymagany jest --asset GLOBAL_A lub GLOBAL_B.")
        validator.validate_global(args.asset, args.train, args.test, args.stop)

if __name__ == "__main__":
    main()