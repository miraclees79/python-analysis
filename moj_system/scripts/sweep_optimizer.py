# -*- coding: utf-8 -*-
"""
moj_system/scripts/sweep_optimizer.py
======================================
Advanced Strategy Sweeper. Supports:
- SINGLE assets (e.g., WIG20TR, MWIG40TR) with Common OOS
- PENSION Portfolio (WIG+TBSP)
- GLOBAL Portfolios (Mode A & Mode B)
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
    get_n_jobs, walk_forward, compute_metrics, analyze_trades, 
    compute_buy_and_hold, print_backtest_report, annual_cagr_by_year, count_year_wins
)
from moj_system.core.pension_engine import (
    build_signal_series, allocation_walk_forward, print_multiasset_report,
    build_standard_two_asset_data
)
from moj_system.core.global_engine import (
    build_return_series, build_price_df_from_returns, 
    allocation_walk_forward_n, print_global_equity_report
)
from moj_system.core.robustness_engine import analyze_robustness, analyze_bootstrap
from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, SWEEP_WINDOW_CONFIGS, EQUITY_THRESHOLDS_MC
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater
from moj_system.data.builder import build_and_upload
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.research import get_common_oos_start

# Strategy Core Imports





class SweepManager:
    def __init__(self, n_mc):
        self.n_mc = n_mc
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    def run_single_asset_iteration(self, asset_name, train_y, test_y, stop_type, common_start):
        """Standard Walk-Forward + Lite MC for one asset."""
        df = load_local_csv(asset_name.lower(), asset_name)
        cash_df = load_local_csv("fund_2720", "MMF")
        use_atr = (stop_type == "atr")
        
        wf_equity, wf_results, _ = walk_forward(
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

        mc_verdict = "N/A"
        if self.n_mc > 0:
            mc_df = self.rob_engine.run_mc_test(wf_results, df, cash_df, n_samples=self.n_mc)
            mc_sum = analyze_robustness(mc_df, m, thresholds=EQUITY_THRESHOLDS_MC)
            mc_verdict = mc_sum["verdict"]

        return {
            "Strategy": asset_name, "Config": f"{train_y}+{test_y}", "Stop": stop_type,
            "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], "MC": mc_verdict
        }

    def run_pension_iteration(self, train_y, test_y, stop_type_eq, common_start):
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
        derived = build_standard_two_asset_data(WIG, TBSP, MMF, None, PL10Y, DE10Y, "1995-01-02")
        
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(WIG, derived["mmf_ext"], train_y, test_y, use_atr_stop=(stop_type_eq=="atr"), n_jobs=get_n_jobs())
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, derived["mmf_ext"], train_y, test_y, filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"], n_jobs=get_n_jobs(), entry_gate_series=derived["bond_gate"])
        
        sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
        port_eq, _, _, _ = allocation_walk_forward(derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], sig_eq, sig_bd, sig_eq, sig_bd, wf_res_eq, wf_res_bd)
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)
        return {"Strategy": "PENSION", "Config": f"{train_y}+{test_y}", "Stop": stop_type_eq, "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], "MC": "N/A"}

    def run_global_iteration(self, variant_key, train_y, test_y, stop_type_eq, common_start):
        cfg = ASSET_REGISTRY[variant_key]
        mode, fx_hedged = cfg["mode"], cfg["fx_hedged"]
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
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
        for lbl, (px_df, fx_s) in assets.items():
            ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret_s, lbl)
            wf_e, _, wf_t = walk_forward(proc_px, MMF, train_y, test_y, use_atr_stop=use_atr, n_jobs=get_n_jobs())
            sigs_full[lbl] = build_signal_series(wf_e, wf_t)

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"], n_jobs=get_n_jobs())
        rets_dict["TBSP"], sigs_full["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna(), build_signal_series(wf_bd, _)
        
        port_eq, _, _, _ = allocation_walk_forward_n(rets_dict, sigs_full, sigs_full, MMF["Zamkniecie"].pct_change().dropna(), wf_res_bd, list(rets_dict.keys()), train_years=train_y)
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)
        return {"Strategy": variant_key, "Config": f"{train_y}+{test_y}", "Stop": stop_type_eq, "CAGR": m["CAGR"], "CalMAR": m["CalMAR"], "MaxDD": m["MaxDD"], "MC": "N/A"}

def main():
    parser = argparse.ArgumentParser(description="Professional Multi-Strategy Sweeper")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL", "ALL"], required=True)
    parser.add_argument("--assets", nargs="+", help="Dla trybu SINGLE (np. WIG20TR SP500)")
    parser.add_argument("--n_mc", type=int, default=0, help="Liczba probek MC (0 = pomin)")
    args = parser.parse_args()

    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    manager = SweepManager(args.n_mc)
    results = []

    # 1. Update Data
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(get_funds=False)

    # 2. Universal OOS Start Calculation
    logging.info("Preparing extended data series for OOS calculation...")
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    # [KROK A] BUDOWA SERII ROZSZERZONYCH (MSCI, STOXX, TBSP)
    # Wymuszamy budowę tych trzech, bo one są najdłuższe
    tbsp_df = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    msciw_df = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
    stoxx_df = build_and_upload(folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path)

    logging.info("Calculating common OOS start for all possible assets...")
    
    # [KROK B] BUDOWANIE MAPY DANYCH DLA SILNIKA RESEARCH
    data_map = {}
    
    # 1. Dodajemy to, co updater ściągnął do raw_csv (WIG, SP500, Nikkei)
    for a in ["WIG", "SP500", "Nikkei225"]:
        df = load_local_csv(a.lower(), a, mandatory=False)
        if df is not None: data_map[a] = df

    # 2. Dodajemy nasze rozszerzone serie (nadpisujemy jeśli updater pobrał krótkie wersje)
    if tbsp_df is not None: 
        data_map["TBSP"] = tbsp_df
        logging.info(f"Using Extended TBSP for OOS calc: {tbsp_df.index.min().date()}")
    if msciw_df is not None: 
        data_map["MSCI_World"] = msciw_df
        logging.info(f"Using Extended MSCI World for OOS calc: {msciw_df.index.min().date()}")
    if stoxx_df is not None:
        data_map["STOXX600"] = stoxx_df
        logging.info(f"Using Extended STOXX600 for OOS calc: {stoxx_df.index.min().date()}")

    # [KROK C] WYLICZENIE WSPÓLNEGO STARTU
    # Silnik sprawdzi każdą serię i każdą konfigurację okien (6..9 lat)
    common_start = get_common_oos_start(data_map, SWEEP_WINDOW_CONFIGS)
    
    # LOGOWANIE "DOWODU" - sprawdzamy co nas blokuje
    logging.info("OOS Constraint Analysis:")
    for name, df in data_map.items():
        earliest_oos = df.index.min() + pd.DateOffset(years=9) # Najdłuższy train
        logging.info(f"  {name:<12} Earliest OOS: {earliest_oos.date()} (Data starts: {df.index.min().date()})")

    # 3. Execution Loops
    if args.mode == "SINGLE" and args.assets:
        for asset in args.assets:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info(f"SINGLE SWEEP: {asset} | {ty}+{te} | {st}")
                    res = manager.run_single_asset_iteration(asset, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["GLOBAL", "ALL"]:
        for var in ["GLOBAL_A", "GLOBAL_B"]:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info(f"GLOBAL SWEEP: {var} | {ty}+{te} | {st}")
                    res = manager.run_global_iteration(var, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["PENSION", "ALL"]:
        for ty, te in SWEEP_WINDOW_CONFIGS:
            for st in ["fixed", "atr"]:
                logging.info(f"PENSION SWEEP: {ty}+{te} | {st}")
                res = manager.run_pension_iteration(ty, te, st, common_start)
                if res: results.append(res)

    if results:
        final_df = pd.DataFrame(results).sort_values("CalMAR", ascending=False)
        print(f"\n=== LEADERBOARD (OOS Start: {common_start.date()}) ===")
        print(final_df.head(20).to_string(index=False))
        final_df.to_csv(f"outputs/sweep_{args.mode.lower()}_results.csv", index=False)

if __name__ == "__main__":
    main()