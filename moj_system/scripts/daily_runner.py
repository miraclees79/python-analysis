# -*- coding: utf-8 -*-
"""
moj_system/scripts/daily_runner.py
==================================
Universal Entry Point for all investment strategies.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import tempfile
from datetime import datetime as dt
import matplotlib
matplotlib.use('Agg')

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- New, Clean Imports ---
from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS
from moj_system.data.updater import DataUpdater
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload
from moj_system.reporting.daily_output import build_daily_outputs
from moj_system.reporting.multiasset_daily_output import build_daily_outputs as build_multiasset_outputs
from moj_system.reporting.global_equity_daily_output import build_daily_outputs as build_global_outputs

# --- Core Engine Imports (Using the legacy, verified engines) ---
sys.path.append(os.path.join(project_root, 'current_development'))
from moj_system.core.strategy_engine import get_n_jobs, walk_forward, compute_metrics, analyze_trades, compute_buy_and_hold, print_backtest_report
from moj_system.core.pension_engine import build_signal_series, allocation_walk_forward, print_multiasset_report, build_standard_two_asset_data
from moj_system.core.global_engine import build_return_series, build_price_df_from_returns, allocation_walk_forward_n, print_global_equity_report, build_mmf_extended

def setup_logging(output_prefix):
    log_file = f"outputs/{output_prefix}.log"
    os.makedirs("outputs", exist_ok=True)
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])

def run_single_asset(asset_name, stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY[asset_name]
    output_prefix = asset_name.lower()
    selected_stop_mode = cfg.get("default_stop", "fixed") if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_stop = (selected_stop_mode == "atr")
    logging.info(f"SINGLE ASSET ENGINE: {asset_name} | Stop Mode: {selected_stop_mode}")

    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    cash_df = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02") if WIBOR1M is not None else MMF

    if cfg["source"] == "drive":
        is_msci = (asset_name == "MSCI_World")
        df = build_and_upload(os.environ.get("GDRIVE_FOLDER_ID"), f"{output_prefix}_wsj_raw.csv" if is_msci else f"{output_prefix}.csv",
                              f"{output_prefix}_combined.csv", cfg["ticker"], "yfinance", creds_path, is_msci_world=is_msci)
    else:
        df = load_local_csv(output_prefix, asset_name)
    if df is None: sys.exit(f"Failed to load data for {asset_name}")

    grids = BASE_GRIDS.copy()
    if "grids" in cfg: grids.update(cfg["grids"])
    
    wf_equity, wf_results, wf_trades = walk_forward(
        df=df, cash_df=cash_df, train_years=cfg["train"], test_years=cfg["test"],
        X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"], fast_grid=grids["FAST_GRID"], 
        slow_grid=grids["SLOW_GRID"], tv_grid=grids["TV_GRID"], sl_grid=grids["SL_GRID"], 
        mom_lookback_grid=grids["MOM_LB_GRID"], objective="calmar", n_jobs=get_n_jobs(), fast_mode=True,
        use_atr_stop=use_atr_stop, N_atr_grid=grids["N_ATR_GRID"] if use_atr_stop else None, atr_window=grids["ATR_WINDOW"]
    )
    if wf_equity.empty: sys.exit(f"Walk-forward returned no results for {asset_name}.")

    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    trade_stats = analyze_trades(wf_trades)
    
    # CORRECTED CALL
    print_backtest_report(metrics=wf_metrics, trades=wf_trades, trade_stats=trade_stats, 
                          wf_results=wf_results, position_mode="full", filter_modes_override=None)
    
    oos_start, oos_end = wf_results["TestStart"].min(), wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, "Zamkniecie", oos_start, oos_end)

    build_daily_outputs(
        wf_equity=wf_equity, 
        wf_trades=wf_trades, 
        wf_metrics=wf_metrics,
        wf_results=wf_results, 
        bh_equity=bh_equity, 
        bh_metrics=bh_metrics, 
        df=df, 
        output_dir=f"outputs/{output_prefix}", 
        asset_name=asset_name,
        price_col="Zamkniecie", # <--- UPEWNIJ SIĘ, ŻE TU JEST "Zamkniecie"
        logfile_name=f"{output_prefix}_signal_log.csv",
        gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None
    )

def run_pension_portfolio(stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY["PENSION"]
    selected_stop = cfg.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_eq = (selected_stop == "atr")
    logging.info(f"PENSION PORTFOLIO ENGINE (WIG+TBSP+MMF) | WIG Stop: {selected_stop}")

    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    TBSP = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
    
    derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR, PL10Y, DE10Y, "1995-01-02")
    n_jobs = get_n_jobs()

    wf_eq, wf_res_eq, wf_tr_eq = walk_forward(WIG, derived["mmf_ext"], cfg["train"], cfg["test"], use_atr_stop=use_atr_eq, n_jobs=n_jobs)
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, derived["mmf_ext"], cfg["train"], cfg["test"], filter_modes_override=["ma"], n_jobs=n_jobs, entry_gate_series=derived["bond_gate"])
    
    sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
    oos_s, oos_e = max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min()), min(wf_res_eq["TestEnd"].max(), wf_res_bd["TestEnd"].max())
    sig_eq_oos, sig_bd_oos = sig_eq.loc[oos_s:oos_e], sig_bd.loc[oos_s:oos_e]
    
    port_eq, w_s, realloc, alloc_df = allocation_walk_forward(derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], sig_eq, sig_bd, sig_eq_oos, sig_bd_oos, wf_res_eq, wf_res_bd)

    m_p = compute_metrics(port_eq)
    bh_eq, bh_m_eq = compute_buy_and_hold(WIG, "Zamkniecie", oos_s, oos_e)
    bh_bd, bh_m_bd = compute_buy_and_hold(TBSP, "Zamkniecie", oos_s, oos_e)
    
    print_multiasset_report(m_p, bh_m_eq, bh_m_bd, alloc_df, realloc, sig_eq_oos, sig_bd_oos, oos_s, oos_e, sig_bd)
    build_multiasset_outputs(wf_eq, wf_tr_eq, wf_res_eq, wf_bd, wf_tr_bd, wf_res_bd, port_eq, m_p, w_s, realloc, bh_eq, bh_m_eq, bh_bd, bh_m_bd, 
                             WIG, TBSP, sig_eq_oos, sig_bd_oos, "outputs/pension", None, folder_id, creds_path)

def run_global_portfolio(asset_key, stop_mode_arg, creds_path):
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
                  "STOXX600": (stoxx, fx_map["EUR"]), "Nikkei225": (load_local_csv("nk225", "Nikkei225"), fx_map["JPY"])}
    else: # msci_world
        msciw = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
        assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

    rets_dict, sigs_full, bh_metrics_all = {}, {}, {}
    n_jobs = get_n_jobs()

    for lbl, (px_df, fx_s) in assets.items():
        ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_h)
        rets_dict[lbl] = ret_s.dropna()
        proc_px = px_df if fx_h or fx_s is None else build_price_df_from_returns(ret_s, lbl)
        wf_e, wf_r, wf_t = walk_forward(proc_px, MMF, train_y, test_y, n_jobs=n_jobs)
        sigs_full[lbl] = build_signal_series(wf_e, wf_t)

    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], n_jobs=n_jobs)
    rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
    sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)

    p_e, w_s, realloc, a_df = allocation_walk_forward_n(rets_dict, sigs_full, sigs_full, MMF["Zamkniecie"].pct_change().dropna(), wf_res_bd, list(rets_dict.keys()), train_years=train_y)
    
    m = {k: float(v) for k, v in compute_metrics(p_e).items()}
    for lbl in rets_dict.keys():
        aligned_rets = rets_dict[lbl].reindex(p_e.index).fillna(0.0)
        bh_curve = (1 + aligned_rets).cumprod()
        bh_metrics_all[lbl] = compute_metrics(bh_curve)

    print_global_equity_report(m, bh_metrics_all, a_df, realloc, sigs_full, p_e.index.min(), p_e.index.max(), mode, fx_h)
    
    build_global_outputs({}, p_e, m, w_s, realloc, bh_metrics_all, rets_dict, sigs_full, list(rets_dict.keys()), mode, fx_h, f"outputs/{asset_key.lower()}", folder_id, creds_path)

def main():
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument("--asset", type=str, required=True, help="Asset key: WIG20TR, PENSION, GLOBAL_A, etc.")
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto")
    args = parser.parse_args()

    os.chdir(project_root)
    cfg = ASSET_REGISTRY.get(args.asset)
    if not cfg: sys.exit(f"Error: Unknown asset '{args.asset}'.")
    setup_logging(args.asset.lower())

    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(get_funds=False)

    if cfg["type"] == "portfolio_pension":
        run_pension_portfolio(args.stop_mode, creds_path)
    elif cfg["type"] == "portfolio_global":
        run_global_portfolio(args.asset, args.stop_mode, creds_path)
    else:
        run_single_asset(args.asset, args.stop_mode, creds_path)

    logging.info(f"Execution of {args.asset} completed successfully.")

if __name__ == "__main__":
    main()