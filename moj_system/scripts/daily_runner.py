# -*- coding: utf-8 -*-
"""
moj_system/scripts/daily_runner.py
==================================
Universal Entry Point for all investment strategies.
Supports Single-Asset, Pension (Multi-Asset), and Global Equity (Mode A/B).
"""

import argparse
import os
import sys
import logging
import datetime as dt
import pandas as pd
import tempfile

# --- IMPORT PATH REPAIR ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
legacy_dir = os.path.join(project_root, 'current_development')

sys.path.append(project_root)
sys.path.append(legacy_dir)

# --- LEGACY LIBRARY IMPORTS ---
from current_development.strategy_test_library import (
    get_n_jobs, walk_forward, compute_metrics, analyze_trades, 
    compute_buy_and_hold, print_backtest_report
)
from current_development.multiasset_library import (
    build_signal_series, allocation_walk_forward, print_multiasset_report,
    build_standard_two_asset_data
)
from current_development.global_equity_library import (
    build_return_series, build_price_df_from_returns, 
    allocation_walk_forward_n, print_global_equity_report
)

# --- NEW REFACTORED MODULES ---
from moj_system.data.updater import DataUpdater
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload
from moj_system.reporting.daily_output import build_daily_outputs
from moj_system.reporting.multiasset_daily_output import build_daily_outputs as build_multiasset_outputs
from moj_system.reporting.global_equity_daily_output import build_daily_outputs as build_global_outputs

# --- GRID CONFIGURATIONS and ASSET REGISTRY ---
from moj_system.config import BASE_GRIDS, BOND_GRIDS, ASSET_REGISTRY



def setup_logging(output_prefix):
    log_file = f"outputs/{output_prefix}_DailyRun.log"
    os.makedirs("outputs", exist_ok=True)
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])

def run_single_asset(asset_name, stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY[asset_name]
    output_prefix = asset_name.lower()
    selected_stop_mode = cfg["default_stop"] if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_stop = (selected_stop_mode == "atr")

    logging.info(f"SINGLE ASSET ENGINE: {asset_name} | Stop Mode: {selected_stop_mode}")

    # Load Cash & Indices
    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    from current_development.global_equity_library import build_mmf_extended
    cash_df = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02") if WIBOR1M is not None else MMF

    if cfg["source"] == "drive":
        is_msci = (asset_name == "MSCI_World")
        df = build_and_upload(os.environ.get("GDRIVE_FOLDER_ID"), f"{output_prefix}_wsj_raw.csv" if is_msci else f"{output_prefix}.csv",
                              f"{output_prefix}_combined.csv", cfg["ticker"], "yfinance", creds_path, is_msci_world=is_msci)
    else:
        df = load_local_csv(output_prefix, asset_name)

    # Process Walk-Forward
    grids = BASE_GRIDS.copy()
    if "grids" in cfg: grids.update(cfg["grids"])
    
    wf_equity, wf_results, wf_trades = walk_forward(
        df=df, cash_df=cash_df, train_years=cfg["train"], test_years=cfg["test"],
        X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"], fast_grid=grids["FAST_GRID"], 
        slow_grid=grids["SLOW_GRID"], tv_grid=grids["TV_GRID"], sl_grid=grids["SL_GRID"], 
        mom_lookback_grid=grids["MOM_LB_GRID"], objective="calmar", n_jobs=get_n_jobs(), fast_mode=True,
        use_atr_stop=use_atr_stop, N_atr_grid=grids["N_ATR_GRID"] if use_atr_stop else None, atr_window=grids["ATR_WINDOW"]
    )

    # Reporting
    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    print_backtest_report(wf_metrics, wf_trades, analyze_trades(wf_trades), wf_results, "full", None)
    
    oos_start, oos_end = wf_results["TestStart"].min(), wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, price_col="Zamkniecie", start=oos_start, end=oos_end)

    build_daily_outputs(wf_equity, wf_trades, wf_metrics, wf_results, bh_equity, bh_metrics, df, 
                        f"outputs/{output_prefix}", asset_name, f"{output_prefix}_signal_log.csv",
                        os.environ.get("GDRIVE_FOLDER_ID"), creds_path if os.path.exists(creds_path) else None)

def run_pension_strategy(stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY["PENSION"]
    selected_stop = cfg["default_stop_eq"] if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_eq = (selected_stop == "atr")

    logging.info(f"PENSION PORTFOLIO ENGINE (WIG+TBSP+MMF) | WIG Stop: {selected_stop}")

    # 1. Dane
    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    TBSP = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")

    # 2. Serie pochodne
    derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR, PL10Y, DE10Y, "1995-01-02", 50.0)
    
    # 3. WF Equity (WIG)
    wf_eq, wf_res_eq, wf_tr_eq = walk_forward(WIG, derived["mmf_ext"], cfg["train"], cfg["test"], 
        X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"], fast_grid=BASE_GRIDS["FAST_GRID"], 
        slow_grid=BASE_GRIDS["SLOW_GRID"], tv_grid=BASE_GRIDS["TV_GRID"], sl_grid=BASE_GRIDS["SL_GRID"], 
        mom_lookback_grid=BASE_GRIDS["MOM_LB_GRID"], objective="calmar", n_jobs=get_n_jobs(), fast_mode=True,
        use_atr_stop=use_atr_eq, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None, 
        atr_window=BASE_GRIDS["ATR_WINDOW"])
    
    # 4. WF Bond (TBSP)
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, derived["mmf_ext"], cfg["train"], cfg["test"], filter_modes_override=["ma"], 
        X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"], fast_grid=BOND_GRIDS["FAST_GRID"], 
        slow_grid=BOND_GRIDS["SLOW_GRID"], tv_grid=BOND_GRIDS["TV_GRID"], sl_grid=BOND_GRIDS["SL_GRID"], 
        n_jobs=get_n_jobs(), fast_mode=True, entry_gate_series=derived["bond_gate"])

    # 5. Allocation
    sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
    oos_s = max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min())
    oos_e = min(wf_res_eq["TestEnd"].max(), wf_res_bd["TestEnd"].max())
    
    def _trim(s, a, b): return s.loc[(s.index >= a) & (s.index <= b)]
    sig_eq_oos, sig_bd_oos = _trim(sig_eq, oos_s, oos_e), _trim(sig_bd, oos_s, oos_e)

    portfolio_equity, weights_series, realloc_log, alloc_df = allocation_walk_forward(
        derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], sig_eq, sig_bd, 
        sig_eq_oos, sig_bd_oos, wf_res_eq, wf_res_bd)

    # 6. Obliczanie metryk B&H (Zabezpieczone przed lukami w datach)
    port_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}
    
    # Używamy reindexacji zwrotów dla pełnej spójności z portfelem
    bh_eq_rets = derived["ret_eq"].reindex(portfolio_equity.index).fillna(0.0)
    bh_eq_curve = (1 + bh_eq_rets).cumprod()
    bh_m_eq = compute_metrics(bh_eq_curve)

    bh_bd_rets = derived["ret_bd"].reindex(portfolio_equity.index).fillna(0.0)
    bh_bd_curve = (1 + bh_bd_rets).cumprod()
    bh_m_bd = compute_metrics(bh_bd_curve)

    # Raport konsolowy
    print_multiasset_report(port_metrics, bh_m_eq, bh_m_bd, alloc_df, realloc_log, sig_eq_oos, sig_bd_oos, oos_s, oos_e, sig_bd)

    # 7. Output plików (NAPRAWIONE)
    build_multiasset_outputs(
        wf_equity_eq=wf_eq, wf_trades_eq=wf_tr_eq, wf_results_eq=wf_res_eq,
        wf_equity_bd=wf_bd, wf_trades_bd=wf_tr_bd, wf_results_bd=wf_res_bd,
        portfolio_equity=portfolio_equity, portfolio_metrics=port_metrics,
        weights_series=weights_series, reallocation_log=realloc_log,
        bh_eq_equity=bh_eq_curve, bh_eq_metrics=bh_m_eq, 
        bh_bd_equity=bh_bd_curve, bh_bd_metrics=bh_m_bd, 
        WIG=WIG, TBSP=TBSP, sig_eq_oos=sig_eq_oos, sig_bd_oos=sig_bd_oos, 
        output_dir="outputs/multiasset", gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"), 
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None
    )
def run_global_strategy(asset_key, stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY[asset_key]
    mode, train_y, test_y, fx_h = cfg["mode"], cfg["train"], cfg["test"], cfg["fx_hedged"]
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    logging.info(f"GLOBAL PORTFOLIO ENGINE: {mode} | FX Hedged: {fx_h}")

    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    TBSP = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    MMF = load_local_csv("fund_2720", "MMF")
    
    fx_map = {curr: load_local_csv(f"{curr.lower()}pln", f"{curr}PLN")["Zamkniecie"] for curr in ["USD", "EUR", "JPY"]}

    if mode == "global_equity":
        stoxx = build_and_upload(folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path)
        assets = {"WIG": (WIG, None), "SP500": (load_local_csv("sp500", "SP500"), fx_map["USD"]), 
                  "STOXX600": (stoxx, fx_map["EUR"]), "Nikkei225": (load_local_csv("nikkei225", "Nikkei225"), fx_map["JPY"])}
    else: # msci_world
        msciw = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
        assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

    rets_dict, sigs_full, bh_metrics_all = {}, {}, {}
    for lbl, (px_df, fx_s) in assets.items():
        ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_h)
        rets_dict[lbl] = ret_s.dropna()
        proc_px = px_df if fx_h or fx_s is None else build_price_df_from_returns(ret_s, lbl)
        
        logging.info(f"Running WF for {lbl}...")
        wf_e, wf_r, wf_t = walk_forward(proc_px, MMF, train_y, test_y, X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"], 
                                        fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"], n_jobs=get_n_jobs(), fast_mode=True)
        sigs_full[lbl] = build_signal_series(wf_e, wf_t)

    # Bonds
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], 
                                              X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"], n_jobs=get_n_jobs(), fast_mode=True)
    rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
    sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)

    # Allocation
    p_e, w_s, realloc, a_df = allocation_walk_forward_n(rets_dict, sigs_full, sigs_full, MMF["Zamkniecie"].pct_change().dropna(), wf_res_bd, list(rets_dict.keys()), train_years=train_y)
    
    # Metrics & Reporting (NAPRAWIONE)
    m = {k: float(v) for k, v in compute_metrics(p_e).items()}
    
    # Obliczamy B&H dla każdego assetu do raportu (wyrównując kalendarze)
    for lbl in rets_dict.keys():
        # reindex zamiast loc wypełnia luki zerami (brak zwrotu w święta danego rynku)
        aligned_rets = rets_dict[lbl].reindex(p_e.index).fillna(0.0)
        bh_curve = (1 + aligned_rets).cumprod()
        bh_metrics_all[lbl] = compute_metrics(bh_curve)

    print_global_equity_report(m, bh_metrics_all, a_df, realloc, sigs_full, p_e.index.min(), p_e.index.max(), mode, fx_h)
    
    # Output (NAPRAWIONE)
    build_global_outputs(
        wf_results_dict={}, portfolio_equity=p_e, portfolio_metrics=m, weights_series=w_s,
        reallocation_log=realloc, bh_metrics_dict=bh_metrics_all, returns_dict=rets_dict,
        signals_oos_dict=sigs_full, asset_keys=list(rets_dict.keys()), portfolio_mode=mode, fx_hedged=fx_h,
        output_dir=f"outputs/{asset_key.lower()}", gdrive_folder_id=folder_id, gdrive_credentials=creds_path
    )
def main():
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument("--asset", type=str, required=True, help="Asset key: WIG20TR, SP500, PENSION, GLOBAL_A, GLOBAL_B")
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto")
    args = parser.parse_args()

    os.chdir(project_root)
    cfg = ASSET_REGISTRY.get(args.asset)
    if not cfg: sys.exit(f"Error: Unknown asset '{args.asset}'.")
    setup_logging(args.asset.lower())

    # 1. Hybrid Data Update
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(get_funds=False)

    # 2. Execution Routing
    if cfg["type"] == "portfolio_pension":
        run_pension_strategy(args.stop_mode, creds_path)
    elif cfg["type"] == "portfolio_global":
        run_global_strategy(args.asset, args.stop_mode, creds_path)
    else:
        run_single_asset(args.asset, args.stop_mode, creds_path)

    logging.info(f"Execution of {args.asset} completed successfully.")

if __name__ == "__main__":
    main()