# -*- coding: utf-8 -*-
"""
moj_system/scripts/sweep_optimizer.py
======================================
Advanced Strategy Sweeper. Supports:
- SINGLE assets (e.g., WIG20TR, MWIG40TR) with Common OOS and individual regime analysis.
- PENSION Portfolio (WIG+TBSP) with WIG-based regime analysis.
- GLOBAL Portfolios (Mode A & Mode B) with WIG-based regime analysis.
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
from moj_system.core.strategy_engine import (
    get_n_jobs, walk_forward, compute_metrics, compute_buy_and_hold
)
from moj_system.core.pension_engine import (
    build_signal_series, allocation_walk_forward, build_standard_two_asset_data
)
from moj_system.core.global_engine import (
    build_return_series, build_price_df_from_returns, allocation_walk_forward_n
)
from moj_system.core.robustness_engine import analyze_robustness
from moj_system.core.utils import build_mmf_extended
from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, SWEEP_WINDOW_CONFIGS, EQUITY_THRESHOLDS_MC, EQUITY_THRESHOLDS_BOOTSTRAP, BOND_THRESHOLDS_MC, BOND_THRESHOLDS_BOOTSTRAP
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater
from moj_system.data.builder import build_and_upload
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.robustness_engine import analyze_bootstrap
from moj_system.core.research import (
    get_common_oos_start, prepare_regime_inputs, run_regime_decomposition, extract_flat_regime_stats
)


class SweepManager:
    def __init__(self, n_mc):
        self.n_mc = n_mc
        self.n_boot = n_boot # <-- DODANY n_boot
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    def run_single_asset_iteration(self, asset_name, train_y, test_y, stop_type, common_start):
        """Standard Walk-Forward + Lite MC for one asset with INDIVIDUAL regime analysis."""
        df = load_local_csv(asset_name.lower(), asset_name)
        cash_df = load_local_csv("fund_2720", "MMF")
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        if WIBOR1M is not None:
            
            cash_df = build_mmf_extended(cash_df, WIBOR1M, floor_date="1995-01-02")
        
        use_atr = (stop_type == "atr")
        
        wf_equity, wf_results, wf_trades = walk_forward(
            df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
            X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs(), fast_mode=True
        )
        if wf_equity.empty: return None

        trimmed = wf_equity.loc[wf_equity.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        # --- REGIME DECOMPOSITION (Względem samego analizowanego waloru) ---
        bh_equity, bh_metrics = compute_buy_and_hold(df, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(df, wf_results, trimmed, bh_equity)
        raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
        regime_metrics = extract_flat_regime_stats(raw_regimes)

        mc_verdict = "N/A"
        mc_p05 = float('nan')
        if self.n_mc > 0:
            logging.info(f"Running MC for Single Asset ({self.n_mc} iterations)...")
            mc_df = self.rob_engine.run_mc_test(wf_results, df, cash_df, n_samples=self.n_mc)
            mc_sum = analyze_robustness(mc_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_MC)
            mc_verdict = mc_sum["verdict"]
            mc_p05 = mc_sum.get("CAGR", {}).get("p05", float('nan'))

        bb_verdict = "N/A"
        bb_p05 = float('nan')
        if self.n_boot > 0:
            logging.info(f"Running Bootstrap for Single Asset ({self.n_boot} iterations)...")
            bb_df = self.rob_engine.run_bootstrap_test(
                df=df, cash_df=cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"]
            )
            bb_sum = analyze_bootstrap(bb_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)
            bb_verdict = bb_sum["verdict"]
            bb_p05 = bb_sum.get("CAGR", {}).get("p05", float('nan'))
        return {
            "Strategy": asset_name, "Config": f"{train_y}+{test_y}", "Stop": stop_type,
            "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], 
            "MC": mc_verdict, "MC_p05": mc_p05,
            "BB": bb_verdict, "BB_p05": bb_p05,
            **regime_metrics
        }

    def run_pension_iteration(self, train_y, test_y, stop_type_eq, common_start):
        """Standard Walk-Forward for Pension Portfolio + Lite MC on WIG component."""
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR1M, PL10Y, DE10Y, "1995-01-02")
        
        use_atr = (stop_type_eq == "atr")
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
            WIG, derived["mmf_ext"], train_y, test_y, 
            use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs(), fast_mode=True
        )
        
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            TBSP, derived["mmf_ext"], train_y, test_y, 
            filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"], 
            n_jobs=get_n_jobs(), entry_gate_series=derived["bond_gate"], fast_mode=True
        )
        
        sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
        port_eq, wgt_ser, all_realloc_log, alloc_results_df = allocation_walk_forward(
            derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], 
            sig_eq, sig_bd, sig_eq, sig_bd, wf_res_eq, wf_res_bd
        )
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        # --- REGIME DECOMPOSITION (Względem WIG B&H) ---
        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(WIG, wf_res_eq, trimmed, bh_equity)
        raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
        regime_metrics = extract_flat_regime_stats(raw_regimes)

        mc_verdict = "N/A"
        mc_p05 = float('nan')
        if self.n_mc > 0 and not wf_res_eq.empty:
            logging.info(f"Running MC for Pension WIG component ({self.n_mc} iterations)...")
            mc_df = self.rob_engine.run_mc_test(wf_res_eq, WIG, derived["mmf_ext"], n_samples=self.n_mc)
            mc_sum = analyze_robustness(mc_df, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)
            mc_verdict = f"WIG:{mc_sum['verdict']}"
        bb_verdict = "N/A"
        bb_p05 = float('nan')
        if self.n_boot > 0:
            logging.info(f"Running Bootstrap for Pension WIG ({self.n_boot} iterations)...")
            bb_results = self.rob_engine.run_bootstrap_test(
                df=df, cash_df=cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"]
            )
            bb_sum = analyze_bootstrap(bb_results, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)
            bb_verdict = bb_sum["verdict"]
            bb_p05 = bb_sum.get("CAGR", {}).get("p05", float('nan'))

        mc_verdict_bd = "N/A"
        mc_p05_bd = float('nan')
        if self.n_mc > 0 and not wf_res_bd.empty:
            logging.info(f"Running MC for Pension TBSP component ({self.n_mc} iterations)...")
            mc_df_bd = self.rob_engine.run_mc_test(wf_res_bd, TBSP, derived["mmf_ext"], n_samples=self.n_mc)
            mc_sum_bd = analyze_robustness(mc_df, compute_metrics(wf_bd), thresholds=BOND_THRESHOLDS_MC)
            mc_verdict_bd = f"TBSP:{mc_sum['verdict']}"
        bb_verdict_bd = "N/A"
        bb_p05_bd = float('nan')
        if self.n_boot > 0:
            logging.info(f"Running Bootstrap for Pension TBSP ({self.n_boot} iterations)...")
            bb_results_bd = self.rob_engine.run_bootstrap_test(
                df=df, cash_df=cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"]
            )
            bb_sum_bd = analyze_bootstrap(bb_results_bd, compute_metrics(wf_equity), thresholds=BOND_THRESHOLDS_BOOTSTRAP)
            bb_verdict_bd = bb_sum_bd["verdict"]
            bb_p05_bd = bb_sum_bd.get("CAGR", {}).get("p05", float('nan'))

        return {
            "Strategy": "PENSION", "Config": f"{train_y}+{test_y}", "Stop": stop_type_eq, 
            "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], 
            "MC": mc_verdict, "MC_p05": mc_p05,
            "BB": bb_verdict, "BB_p05": bb_p05,
            **regime_metrics
        }

    def run_global_iteration(self, variant_key, train_y, test_y, stop_type_eq, common_start):
        """Standard Walk-Forward for Global Portfolio + Lite MC on ALL equity components."""
        cfg = ASSET_REGISTRY[variant_key]
        mode, fx_hedged = cfg["mode"], cfg["fx_hedged"]
        
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        MMF = load_local_csv("fund_2720", "MMF")
        if WIBOR1M is not None:
            
            MMF_EXT = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02")
            MMF = MMF_EXT
            
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        fx_map = {c: load_local_csv(f"{c.lower()}pln", f"{c}PLN")["Zamkniecie"] for c in ["USD", "EUR", "JPY"]}
        
        if mode == "global_equity":
            stoxx = build_and_upload(self.folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", self.creds_path)
            assets = {"WIG": (WIG, None), "SP500": (load_local_csv("sp500", "SP500"), fx_map["USD"]), "STOXX600": (stoxx, fx_map["EUR"]), "Nikkei225": (load_local_csv("nikkei225", "Nikkei225"), fx_map["JPY"])}
        else:
            msciw = build_and_upload(self.folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", self.creds_path, is_msci_world=True)
            assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

        rets_dict, sigs_full = {}, {}
        use_atr = (stop_type_eq == "atr")
        
        mc_data_queue = {}
        mc_verdicts = []
        bb_verdicts = []
        
        wig_wf_res = None
        wig_wf_eq = None
        
        for lbl, (px_df, fx_s) in assets.items():
            ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret_s, lbl)
            
            wf_e, wf_r, wf_t = walk_forward(
                proc_px, MMF, train_y, test_y, 
                use_atr_stop=use_atr, 
                N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None, 
                n_jobs=get_n_jobs(), fast_mode=True
            )
            sigs_full[lbl] = build_signal_series(wf_e, wf_t)
            mc_data_queue[lbl] = (wf_r, proc_px, wf_e)
            
            if lbl == "WIG":
                wig_wf_res = wf_r
                wig_wf_eq = wf_e

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], n_jobs=get_n_jobs(), fast_mode=True
        )
        rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)
        
        port_eq, wgt_ser, all_realloc_log, alloc_results_df = allocation_walk_forward_n(
            rets_dict, sigs_full, sigs_full, MMF["Zamkniecie"].pct_change().dropna(), 
            wf_res_bd, list(rets_dict.keys()), train_years=train_y
        )
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)
        
        # --- REGIME DECOMPOSITION (Global Portfolio względem WIG B&H) ---
        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(WIG, wig_wf_res, trimmed, bh_equity)
        raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
        regime_metrics = extract_flat_regime_stats(raw_regimes)

        # --- MC EXECUTION FOR ALL EQUITY ASSETS ---
        if self.n_mc > 0:
            for lbl, (wf_res, proc_px, wf_eq) in mc_data_queue.items():
                if wf_res is not None and not wf_res.empty:
                    logging.info(f"Running MC for Global {lbl} component ({self.n_mc} iterations)...")
                    mc_df = self.rob_engine.run_mc_test(wf_res, proc_px, MMF, n_samples=self.n_mc)
                    mc_sum = analyze_robustness(mc_df, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)
                    mc_verdicts.append(f"{lbl[:3]}:{mc_sum['verdict'][:3]}")
        
        final_mc_string = "|".join(mc_verdicts) if mc_verdicts else "N/A"
        # bootstrap execution
        if self.n_boot > 0:
            for lbl, (wf_res, proc_px, wf_eq) in mc_data_queue.items():
                if wf_res is not None and not wf_res.empty:
                    logging.info(f"Running Bootstrap for Global {lbl} component ({self.n_boot} iterations)...")
                    bb_df = self.rob_engine.run_bootstrap_test(
                    df=df, cash_df=cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                    use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                    X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                    fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"]
                    )
                    bb_sum = analyze_bootstrap(bb_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)
                    bb_verdict = bb_sum["verdict"]
                    bb_p05 = bb_sum.get("CAGR", {}).get("p05", float('nan'))
                    bb_verdicts.append(f"{lbl[:3]}:{bb_sum['verdict'][:3]}")
        final_bb_string = "|".join(bb_verdicts) if bb_verdicts else "N/A"
        return {
            "Strategy": variant_key, "Config": f"{train_y}+{test_y}", "Stop": stop_type_eq, 
            "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], "MC": final_mc_string, 
            "BB": final_BB_string,
            **regime_metrics
        }
    # ==============================================================================
# REPORT FORMATTING (Funkcja pomocnicza do drukowania ładnych tabel w logach)
# ==============================================================================

def print_sweep_report(results_df: pd.DataFrame, common_start: dt.date):
    """
    Drukuje sformatowany raport w konsoli, dzieląc wyniki na sekcje:
    1. Leaderboard (Główne metryki + Robustness)
    2. Regime Decomposition (Jak strategia zachowuje się w hossie/bessie)
    """
    if results_df.empty:
        logging.warning("Brak wyników do zaraportowania.")
        return

    sep = "=" * 120
    logging.info(f"\n{sep}")
    logging.info(f"SWEEP OPTIMIZER REPORT (Common OOS Start: {common_start})")
    logging.info(sep)

    # --- SEKCJA 1: LEADERBOARD & ROBUSTNESS ---
    logging.info("\n--- 1. PERFORMANCE & ROBUSTNESS LEADERBOARD ---")
    
    # Wybieramy i formatujemy kolumny do pokazania
    display_cols = ["Strategy", "Config", "Stop", "CAGR", "CalMAR", "MaxDD", "MC", "BB"]
    
    # Dodajemy p05 jeśli istnieje
    if "MC_p05" in results_df.columns: display_cols.append("MC_p05")
    if "BB_p05" in results_df.columns: display_cols.append("BB_p05")

    # Upewniamy się, że kolumny istnieją
    existing_cols = [c for c in display_cols if c in results_df.columns]
    
    format_df = results_df[existing_cols].copy()
    format_df["CAGR"] = (format_df["CAGR"] * 100).round(2).astype(str) + "%"
    format_df["MaxDD"] = (format_df["MaxDD"] * 100).round(2).astype(str) + "%"
    format_df["CalMAR"] = format_df["CalMAR"].round(3)
    
    if "MC_p05" in format_df.columns:
        format_df["MC_p05"] = (format_df["MC_p05"] * 100).round(2).astype(str) + "%"
    if "BB_p05" in format_df.columns:
        format_df["BB_p05"] = (format_df["BB_p05"] * 100).round(2).astype(str) + "%"

    logging.info("\n" + format_df.to_string(index=False))

    # --- SEKCJA 2: REGIME DECOMPOSITION ---
    # Sprawdzamy, czy w ogóle mamy kolumny reżimowe (np. adx_uptrend_strat_cagr)
    regime_cols = [c for c in results_df.columns if "adx_" in c or "vol_" in c]
    
    if regime_cols:
        logging.info(f"\n{sep}")
        logging.info("--- 2. REGIME DECOMPOSITION (CAGR in Specific Market Conditions) ---")
        
        # Wybieramy tylko najważniejsze kolumny reżimowe do podglądu (żeby tabela się zmieściła)
        key_regime_cols = [
            "Strategy", "Config", "Stop",
            "adx_uptrend_strat_cagr", "adx_uptrend_bh_cagr",
            "adx_downtrend_strat_cagr", "adx_downtrend_bh_cagr",
            "vol_high_strat_cagr"
        ]
        
        avail_regime_cols = [c for c in key_regime_cols if c in results_df.columns]
        regime_df = results_df[avail_regime_cols].copy()
        
        # Formatowanie na procenty (w słowniku masz np. 15.4 co oznacza 15.4%)
        for col in avail_regime_cols:
            if "cagr" in col:
                regime_df[col] = regime_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

        # Krótsze nazwy kolumn dla czytelności w logu
        rename_map = {
            "adx_uptrend_strat_cagr": "UP_Strat",
            "adx_uptrend_bh_cagr": "UP_B&H",
            "adx_downtrend_strat_cagr": "DOWN_Strat",
            "adx_downtrend_bh_cagr": "DOWN_B&H",
            "vol_high_strat_cagr": "VOL_HI_Strat"
        }
        regime_df = regime_df.rename(columns=rename_map)

        logging.info("\n" + regime_df.to_string(index=False))
        logging.info("\n* Interpretation:")
        logging.info("  UP_Strat vs UP_B&H     : Does the strategy capture bull markets? (Expected: Strat <= B&H)")
        logging.info("  DOWN_Strat vs DOWN_B&H : Does the strategy protect capital? (Expected: Strat >> B&H)")
    
    logging.info(sep)

def main():
    parser = argparse.ArgumentParser(description="Professional Multi-Strategy Sweeper")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL", "ALL"], required=True)
    parser.add_argument("--assets", nargs="+", help="Dla trybu SINGLE (np. WIG20TR SP500)")
    parser.add_argument("--n_mc", type=int, default=500, help="Liczba probek MC (0 = pomin)")
    parser.add_argument("--n_boot", type=int, default=0, help="Liczba probek Bootstrap (0 = pomin, domyslnie wylaczone dla szybkosci)")
    args = parser.parse_args()

    os.chdir(project_root)
    
    log_file = "outputs/sweep_optimizer_run.log"
    os.makedirs("outputs", exist_ok=True)
    
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
        
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    manager = SweepManager(args.n_mc, args.n_boot)
    results = []

    # 1. Update Data
    updater = DataUpdater()
    updater.run_full_update(get_funds=False)

    # 2. Universal OOS Start Calculation
    logging.info("Preparing extended data series for OOS calculation...")
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    tbsp_df = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    msciw_df = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
    stoxx_df = build_and_upload(folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path)

    logging.info("Calculating common OOS start for all possible assets...")
    
    data_map = {}
    check_list = ["WIG", "SP500", "STOXX600", "Nikkei225"]
    if args.assets: check_list.extend(args.assets)
    
    for a in set(check_list):
        df = load_local_csv(a.lower(), a, mandatory=False)
        if df is not None: data_map[a] = df

    if tbsp_df is not None: 
        data_map["TBSP"] = tbsp_df
        logging.info(f"Using Extended TBSP for OOS calc: {tbsp_df.index.min().date()}")
    if msciw_df is not None: 
        data_map["MSCI_World"] = msciw_df
        logging.info(f"Using Extended MSCI World for OOS calc: {msciw_df.index.min().date()}")
    if stoxx_df is not None:
        data_map["STOXX600"] = stoxx_df
        logging.info(f"Using Extended STOXX600 for OOS calc: {stoxx_df.index.min().date()}")

    common_start = get_common_oos_start(data_map, SWEEP_WINDOW_CONFIGS)
    
    logging.info("OOS Constraint Analysis:")
    for name, df in data_map.items():
        earliest_oos = df.index.min() + pd.DateOffset(years=9)
        logging.info(f"  {name:<12} Earliest OOS: {earliest_oos.date()} (Data starts: {df.index.min().date()})")

    # 3. Execution Loops
    if args.mode in ["SINGLE", "ALL"] and args.assets:
        for asset in args.assets:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info("=" * 80)
                    logging.info(f"SINGLE SWEEP: {asset} | {ty}+{te} | {st}")
                    logging.info("=" * 80)
                    res = manager.run_single_asset_iteration(asset, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["GLOBAL", "ALL"]:
        for var in ["GLOBAL_A", "GLOBAL_B"]:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info("=" * 80)
                    logging.info(f"GLOBAL SWEEP: {var} | {ty}+{te} | {st}")
                    logging.info("=" * 80)
                    res = manager.run_global_iteration(var, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["PENSION", "ALL"]:
        for ty, te in SWEEP_WINDOW_CONFIGS:
            for st in ["fixed", "atr"]:
                logging.info("=" * 80)
                logging.info(f"PENSION SWEEP: {ty}+{te} | {st}")
                logging.info("=" * 80)
                res = manager.run_pension_iteration(ty, te, st, common_start)
                if res: results.append(res)

    if results:
        final_df = pd.DataFrame(results).sort_values("CalMAR", ascending=False)
        
        # Wywołanie dedykowanej funkcji drukującej tabele do logów
        print_sweep_report(final_df, common_start.date())
        
        # Zapis pełnych danych do CSV (wszystkie 40+ kolumn)
        final_df.to_csv(f"outputs/sweep_{args.mode.lower()}_results.csv", index=False)
        logging.info(f"\nPełne wyniki zapisano do: outputs/sweep_{args.mode.lower()}_results.csv")

if __name__ == "__main__":
    main()
