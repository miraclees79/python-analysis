# moj_system/scripts/validate_robustness.py

# -*- coding: utf-8 -*-
"""
moj_system/scripts/validate_robustness.py
=========================================
Deep Validation Engine. Performs Level 1 (MC), Level 2 (Bootstrap),
and optionally Level 3 (Allocation Weight Perturbation) testing for
chosen configurations of Single, Pension, or Global strategies.
Includes automated OOS chart generation.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')

# --- PATH SETUP ---
from moj_system.config import OUTPUT_DIR

# --- CORE ENGINE IMPORTS ---
from moj_system.core.strategy_engine import (
    get_n_jobs, walk_forward, compute_metrics, compute_buy_and_hold
)
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

    def _save_validation_chart(self, strategy_equity, bh_equity, title, filename):
        """Generates a 2-panel OOS validation chart (Equity + Drawdown)."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        chart_path = OUTPUT_DIR / filename

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.2)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        # Panel 1: Equity
        ax1.plot(strategy_equity.index, strategy_equity.values, label="Strategy (OOS)", color="steelblue", linewidth=2)
        if bh_equity is not None:
            # Align and normalize BH
            bh_aligned = bh_equity.reindex(index=strategy_equity.index).ffill()
            bh_norm = bh_aligned / bh_aligned.iloc[0]
            ax1.plot(bh_norm.index, bh_norm.values, label="Buy & Hold", color="grey", linestyle="--", alpha=0.7)

        ax1.set_title(label=title, fontsize=14, fontweight="bold")
        ax1.set_ylabel(ylabel="Normalised Equity")
        ax1.legend(loc="upper left")
        ax1.grid(visible=True, alpha=0.3)

        # Panel 2: Drawdown
        dd_strat = strategy_equity / strategy_equity.cummax() - 1
        ax2.fill_between(dd_strat.index, dd_strat.values, 0, color="steelblue", alpha=0.3, label="Strategy DD")
        
        if bh_equity is not None:
            dd_bh = bh_norm / bh_norm.cummax() - 1
            ax2.plot(dd_bh.index, dd_bh.values, color="grey", linestyle="--", alpha=0.5)

        ax2.set_ylabel(ylabel="Drawdown")
        ax2.grid(visible=True, alpha=0.3)
        ax2.set_ylim(top=0)

        plt.savefig(fname=chart_path, dpi=72, bbox_inches="tight")
        plt.close(fig=fig)
        logging.info(msg=f"Validation chart saved to: {chart_path}")

    def validate_single(self, asset_name, train_y, test_y, stop_type, df, cash_df):
        logging.info(f"VALIDATING SINGLE ASSET: {asset_name} | {train_y}+{test_y} | {stop_type}")
        use_atr = (stop_type == "atr")

        # 1. Base Walk-Forward
        wf_eq, wf_res, wf_tr = walk_forward(
            df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
            X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs(), fast_mode=True
        )
        
        # Generowanie wykresu
        bh_eq, bh_m = compute_buy_and_hold(df=df, price_col="Zamkniecie", start=wf_eq.index.min(), end=wf_eq.index.max())
        self._save_validation_chart(
            strategy_equity=wf_eq, 
            bh_equity=bh_eq, 
            title=f"OOS Validation: {asset_name} ({train_y}+{test_y} {stop_type})",
            filename=f"validate_{asset_name.lower()}_{train_y}_{test_y}.png"
        )

        # 2. MC Test
        if self.n_mc > 0:
            mc_results = self.rob_engine.run_mc_test(wf_results=wf_res, df=df, cash_df=cash_df, n_samples=self.n_mc)
            analyze_robustness(results_df=mc_results, baseline_metrics=compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)

        # 3. Bootstrap Test
        if self.n_boot > 0:
            bb_results = self.rob_engine.run_bootstrap_test(
                df=df, cash_df=cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None
            )
            analyze_bootstrap(results_df=bb_results, baseline_metrics=compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

    def validate_pension(self, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING PENSION PORTFOLIO | EQ Stop: {stop_type_eq}")
        
        WIG = load_local_csv(ticker="wig", label="WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv(ticker="fund_2720", label="MMF")
        TBSP = build_and_upload(folder_id=self.folder_id, raw_filename="tbsp_extended_full.csv", combined_filename="tbsp_extended_combined.csv", extension_ticker="^tbsp", extension_source="stooq", credentials_path=self.creds_path)
        WIBOR1M = load_local_csv(ticker="wibor1m", label="WIBOR1M", mandatory=False)
        PL10Y, DE10Y = load_local_csv(ticker="pl10y", label="PL10Y"), load_local_csv(ticker="de10y", label="DE10Y")
        derived = build_standard_two_asset_data(wig=WIG, tbsp=TBSP, mmf=MMF, wibor1m=WIBOR1M, pl10y=PL10Y, de10y=DE10Y, mmf_floor="1995-01-02")
        use_atr_eq = (stop_type_eq == 'atr')
        
        # PENSION składa się z komponentów, ale potrzebujemy też krzywej portfela do wykresu
        logging.info("\n" + "="*60 + "\n--- Component 1: WIG Robustness ---\n" + "="*60)
        self.validate_single(asset_name="WIG", train_y=train_y, test_y=test_y, stop_type=stop_type_eq, df=WIG, cash_df=derived["mmf_ext"])
        
        logging.info("\n" + "="*60 + "\n--- Component 2: TBSP Robustness ---\n" + "="*60)
        wf_bd_eq, wf_bd_res, wf_bd_tr = walk_forward(
            df=TBSP, cash_df=derived["mmf_ext"], train_years=train_y, test_years=test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(), fast_mode=True
        )
        
        # Symulacja portfela bazowego dla wykresu
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
            df=WIG, cash_df=derived["mmf_ext"], train_years=train_y, test_years=test_y, 
            X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr_eq, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None,
            n_jobs=get_n_jobs(), fast_mode=True
        )
        
        sig_eq = build_signal_series(wf_equity=wf_eq, wf_trades=wf_tr_eq)
        sig_bd = build_signal_series(wf_equity=wf_bd_eq, wf_trades=wf_bd_tr)
        
        port_eq, port_wgts, realloc_log, alloc_df = allocation_walk_forward(
            equity_returns=derived["ret_eq"], bond_returns=derived["ret_bd"], mmf_returns=derived["ret_mmf"], 
            sig_equity_full=sig_eq, sig_bond_full=sig_bd,
            sig_equity_oos=sig_eq.loc[wf_eq.index.min():], sig_bond_oos=sig_bd.loc[wf_eq.index.min():], 
            wf_results_eq=wf_res_eq, wf_results_bd=wf_bd_res
        )

        bh_wig, _ = compute_buy_and_hold(df=WIG, price_col="Zamkniecie", start=port_eq.index.min(), end=port_eq.index.max())
        self._save_validation_chart(
            strategy_equity=port_eq, 
            bh_equity=bh_wig, 
            title=f"OOS Validation: PENSION Portfolio ({train_y}+{test_y})",
            filename=f"validate_pension_{train_y}_{test_y}.png"
        )

        # Level 3 - Perturbacja Wag
        if self.run_weights_perturb:
            logging.info("\n" + "="*80 + "\n--- Level 3: Allocation Weight Perturbation Test (PENSION) ---\n" + "="*80)
            baseline_metrics = compute_metrics(equity=port_eq)
            robust_df = allocation_weight_robustness(
                alloc_results_df=alloc_df,
                equity_returns=derived["ret_eq"],
                bond_returns=derived["ret_bd"],
                mmf_returns=derived["ret_mmf"],
                sig_equity_oos=sig_eq.loc[wf_eq.index.min():],
                sig_bond_oos=sig_bd.loc[wf_eq.index.min():],
                baseline_metrics=baseline_metrics
            )
            print_allocation_robustness_report(results_df=robust_df)

    def validate_global(self, variant, train_y, test_y, stop_type_eq):
        logging.info(f"VALIDATING GLOBAL {variant} | Train: {train_y} | Stop: {stop_type_eq}")
        
        cfg = ASSET_REGISTRY[variant]
        mode, fx_hedged = cfg["mode"], cfg.get("fx_hedged", True)
        use_atr = (stop_type_eq == "atr")

        WIG = load_local_csv(ticker="wig", label="WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv(ticker="fund_2720", label="MMF")
        WIBOR1M = load_local_csv(ticker="wibor1m", label="WIBOR1M", mandatory=False)
        mmf_ext = build_mmf_extended(mmf_df=MMF, wibor1m_df=WIBOR1M, floor_date="1995-01-02")
            
        TBSP = build_and_upload(folder_id=self.folder_id, raw_filename="tbsp_extended_full.csv", combined_filename="tbsp_extended_combined.csv", extension_ticker="^tbsp", extension_source="stooq", credentials_path=self.creds_path)
        fx_map = {c: load_local_csv(ticker=f"{c.lower()}pln", label=f"{c}PLN")["Zamkniecie"] for c in ["USD", "EUR", "JPY"]}
        
        if mode == "global_equity":
            stoxx = build_and_upload(folder_id=self.folder_id, raw_filename="stoxx600.csv", combined_filename="stoxx600_combined.csv", extension_ticker="^STOXX", extension_source="yfinance", credentials_path=self.creds_path)
            assets = {
                "WIG": (WIG, None), 
                "SP500": (load_local_csv(ticker="sp500", label="SP500"), fx_map["USD"]), 
                "STOXX600": (stoxx, fx_map["EUR"]), 
                "Nikkei225": (load_local_csv(ticker="nikkei225", label="Nikkei225"), fx_map["JPY"])
            }
        else:
            msciw = build_and_upload(folder_id=self.folder_id, raw_filename="msci_world_wsj_raw.csv", combined_filename="msci_world_combined.csv", extension_ticker="URTH", extension_source="yfinance", credentials_path=self.creds_path, is_msci_world=True)
            assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

        rets_dict, sigs_full = {}, {}
        
        for lbl, (px_df, fx_s) in assets.items():
            logging.info("\n" + "="*60 + f"\n--- Component Robustness: {lbl} ---\n" + "="*60)
            ret_s = build_return_series(price_df=px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret=ret_s, label=lbl)
            
            wf_e, wf_r, wf_t = walk_forward(
                df=proc_px, cash_df=mmf_ext, train_years=train_y, test_years=test_y, 
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None, 
                n_jobs=get_n_jobs(), fast_mode=True
            )
            sigs_full[lbl] = build_signal_series(wf_equity=wf_e, wf_trades=wf_t)

            if self.n_mc > 0:
                mc_res = self.rob_engine.run_mc_test(wf_results=wf_r, df=proc_px, cash_df=mmf_ext, n_samples=self.n_mc)
                analyze_robustness(results_df=mc_res, baseline_metrics=compute_metrics(equity=wf_e), thresholds=EQUITY_THRESHOLDS_MC)

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=TBSP, cash_df=mmf_ext, train_years=train_y, test_years=test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(), fast_mode=True
        )
        rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(wf_equity=wf_bd, wf_trades=wf_tr_bd)

        # Alokacja N-Asset i Wykres
        port_eq, _, _, alloc_df = allocation_walk_forward_n(
            returns_dict=rets_dict, signals_full_dict=sigs_full, signals_oos_dict=sigs_full, 
            mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(), wf_results_ref=wf_res_bd, 
            asset_keys=list(rets_dict.keys()), train_years=train_y
        )
        
        bh_wig, _ = compute_buy_and_hold(df=WIG, price_col="Zamkniecie", start=port_eq.index.min(), end=port_eq.index.max())
        self._save_validation_chart(
            strategy_equity=port_eq, 
            bh_equity=bh_wig, 
            title=f"OOS Validation: GLOBAL {variant} ({train_y}+{test_y})",
            filename=f"validate_{variant.lower()}_{train_y}_{test_y}.png"
        )

        if self.run_weights_perturb:
            logging.info("\n" + "="*80 + "\n--- Level 3: Allocation Weight Perturbation Test (GLOBAL) ---\n" + "="*80)
            baseline_metrics = compute_metrics(equity=port_eq)
            robust_df = allocation_weight_robustness_n(
                alloc_results_df=alloc_df, returns_dict=rets_dict, mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(),
                signals_oos_dict={k: v.loc[port_eq.index.min():] for k, v in sigs_full.items()},
                asset_keys=list(rets_dict.keys()), baseline_metrics=baseline_metrics, focus_asset="WIG"
            )
            print_allocation_robustness_report_n(results_df=robust_df, focus_asset="WIG")


def main():
    parser = argparse.ArgumentParser(description="Full Robustness Validator")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL"], required=True)
    parser.add_argument("--asset", help="Dla mode SINGLE lub GLOBAL (np. WIG20TR, GLOBAL_A)")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--stop", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--n_mc", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--weights_perturb", action="store_true", help="Uruchom dodatkowy test perturbacji wag alokacji.")
    args = parser.parse_args()
    
    # Setup logging
    log_file = OUTPUT_DIR / f"validate_{args.mode.lower()}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    DataUpdater().run_full_update(get_funds=False)    
    
    validator = ValidationManager(n_mc=args.n_mc, n_boot=args.n_boot, run_weights_perturb=args.weights_perturb)

    if args.mode == "SINGLE":
        df = load_local_csv(ticker=args.asset.lower(), label=args.asset)
        cash_df = load_local_csv(ticker="fund_2720", label="MMF")
        validator.validate_single(asset_name=args.asset, train_y=args.train, test_y=args.test, stop_type=args.stop, df=df, cash_df=cash_df)
    elif args.mode == "PENSION":
        validator.validate_pension(train_y=args.train, test_y=args.test, stop_type_eq=args.stop)
    elif args.mode == "GLOBAL":
        validator.validate_global(variant=args.asset, train_y=args.train, test_y=args.test, stop_type_eq=args.stop)

if __name__ == "__main__":
    main()