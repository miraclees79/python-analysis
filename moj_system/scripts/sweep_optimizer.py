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
import numpy as np
import tempfile
from collections import Counter
import matplotlib
import datetime as dt
matplotlib.use('Agg')

# --- PATH SETUP ---
from moj_system.config import OUTPUT_DIR

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
from moj_system.core.robustness_engine import analyze_robustness, analyze_bootstrap
from moj_system.config import (
    ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, SWEEP_WINDOW_CONFIGS, 
    EQUITY_THRESHOLDS_MC, EQUITY_THRESHOLDS_BOOTSTRAP, BOND_THRESHOLDS_MC, BOND_THRESHOLDS_BOOTSTRAP
)
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater
from moj_system.data.builder import build_and_upload
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.research import (
    get_common_oos_start, prepare_regime_inputs, run_regime_decomposition, extract_flat_regime_stats, print_live_regime_report
)
from moj_system.core.utils import build_mmf_extended


# ==============================================================================
# HELPER FUNCTIONS 
# ==============================================================================

def _extract_window_rows(
    asset_name:  str, train_years: int, test_years:  int,
    wf_results:  pd.DataFrame, stop_mode:   str,
) -> list[dict]:
    """Extracts parameters and performance metrics for individual walk-forward windows."""
    rows = []
    prev_params = None
    use_atr = stop_mode == "atr"
    stop_col = "N_atr" if use_atr else "X"

    for idx, row in wf_results.iterrows():
        cur_params = {
            "filter_mode": str(row.get("filter_mode", "")),
            "stop_param":  float(row.get(stop_col, pd.NA)),
            "Y":           float(row.get("Y", pd.NA)),
            "fast":        int(row.get("fast", 0)),
            "slow":        int(row.get("slow", 0)),
            "stop_loss":   float(row.get("stop_loss", pd.NA)),
        }

        param_changed = False
        if prev_params is not None:
            param_changed = any(cur_params[k] != prev_params[k] for k in cur_params)

        rows.append({
            "Strategy":      asset_name,
            "train_years":   train_years,
            "test_years":    test_years,
            "stop_mode":     stop_mode,
            "window_idx":    idx,
            "train_start":   str(pd.Timestamp(row["TrainStart"]).date()),
            "test_start":    str(pd.Timestamp(row["TestStart"]).date()),
            "test_end":      str(pd.Timestamp(row["TestEnd"]).date()),
            "win_cagr":      round(float(row.get("CAGR", pd.NA)) * 100, 2),
            "win_sharpe":    round(float(row.get("Sharpe", pd.NA)), 3),
            "win_maxdd":     round(float(row.get("MaxDD", pd.NA)) * 100, 2),
            "win_calmar":    round(float(row.get("CalMAR", pd.NA)), 3),
            "filter_mode":   cur_params["filter_mode"],
            "stop_param":    cur_params["stop_param"],
            "Y":             cur_params["Y"],
            "fast":          cur_params["fast"],
            "slow":          cur_params["slow"],
            "stop_loss":     cur_params["stop_loss"],
            "param_changed": param_changed,
        })
        prev_params = cur_params
    return rows

def _compile_window_stats(window_rows: list) -> dict:
    """Calculates aggregate statistics over the extracted windows."""
    summary = {}
    if window_rows:
        win_cagrs = [w["win_cagr"] for w in window_rows if pd.notna(w["win_cagr"])]
        summary["window_cagr_mean"]    = round(np.mean(win_cagrs),  2) if win_cagrs else pd.NA
        summary["window_cagr_std"]     = round(np.std(win_cagrs),   2) if len(win_cagrs) > 1 else pd.NA
        summary["window_cagr_min"]     = round(np.min(win_cagrs),   2) if win_cagrs else pd.NA
        summary["window_cagr_max"]     = round(np.max(win_cagrs),   2) if win_cagrs else pd.NA
        summary["param_change_count"]  = sum(1 for w in window_rows if w["param_changed"])

        filter_modes = [w["filter_mode"] for w in window_rows if w["filter_mode"]]
        if filter_modes:
            dominant_mode, dominant_count = Counter(filter_modes).most_common(1)[0]
            summary["dominant_filter"]    = dominant_mode
            summary["filter_consistency"] = round((dominant_count / len(filter_modes)) * 100, 1)
        else:
            summary["dominant_filter"]    = "unknown"
            summary["filter_consistency"] = pd.NA
    else:
        for col in ("window_cagr_mean", "window_cagr_std", "window_cagr_min",
                    "window_cagr_max", "param_change_count",
                    "dominant_filter", "filter_consistency"):
            summary[col] = pd.NA
    
    return summary




# ==============================================================================
# SWEEP MANAGER
# ==============================================================================

class SweepManager:
    def __init__(self, n_mc, n_boot):
        self.n_mc = n_mc
        self.n_boot = n_boot
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        self.all_windows = [] 

    def _compile_full_result(self, strat_name, train_y, test_y, stop_type, common_start, 
                             wf_results, wf_equity_trimmed, m_trimmed, bh_metrics, 
                             regime_metrics, mc_verdicts_dict, bb_verdicts_dict):
        """Unified result compiler for all 3 modes."""
        
        # 1. Window Stats
        window_rows = _extract_window_rows(strat_name, train_y, test_y, wf_results, stop_type)
        self.all_windows.extend(window_rows)
        win_stats = _compile_window_stats(window_rows)

        # 2. Stringify Robustness verdicts (e.g. "WIG:ROB|TBS:FRA")
        mc_verdict = "|".join([f"{k[:3]}:{v['verdict'][:3]}" for k,v in mc_verdicts_dict.items()]) if mc_verdicts_dict else "N/A"
        bb_verdict = "|".join([f"{k[:3]}:{v['verdict'][:3]}" for k,v in bb_verdicts_dict.items()]) if bb_verdicts_dict else "N/A"
        
        # Pull p05 for the primary asset (usually the first key, e.g. WIG or the Single Asset)
        mc_p05 = pd.NA
        if mc_verdicts_dict:
            first_mc_key = list(mc_verdicts_dict.keys())[0]
            mc_p05 = mc_verdicts_dict[first_mc_key].get("CAGR", {}).get("p05", pd.NA)
            
        bb_p05 = pd.NA
        if bb_verdicts_dict:
            first_bb_key = list(bb_verdicts_dict.keys())[0]
            bb_p05 = bb_verdicts_dict[first_bb_key].get("CAGR", {}).get("p05", pd.NA)

        
        
        
        
        return {
            "Strategy": strat_name, "train_years": train_y, "test_years": test_y, "stop_mode": stop_type,
            "oos_cagr": m_trimmed.get("CAGR", pd.NA), "oos_calmar": m_trimmed.get("CalMAR", pd.NA), 
            "oos_maxdd": m_trimmed.get("MaxDD", pd.NA), "oos_sharpe": m_trimmed.get("Sharpe", pd.NA), 
            "oos_sortino": m_trimmed.get("Sortino", pd.NA),
            "bh_cagr": bh_metrics.get("CAGR", pd.NA), "bh_calmar": bh_metrics.get("CalMAR", pd.NA), 
            "bh_maxdd": bh_metrics.get("MaxDD", pd.NA), "bh_sharpe": bh_metrics.get("Sharpe", pd.NA),
            "cagr_vs_bh": (m_trimmed.get("CAGR", 0) - bh_metrics.get("CAGR", 0)) * 100 if pd.notna(m_trimmed.get("CAGR")) and pd.notna(bh_metrics.get("CAGR")) else pd.NA,
            "MC_Verdict": mc_verdict, "MC_p05_CAGR": mc_p05,
            "BB_Verdict": bb_verdict, "BB_p05_CAGR": bb_p05,
            **win_stats,
            **regime_metrics
        }

    def run_single_asset_iteration(self, asset_name, train_y, test_y, stop_type, common_start):
        df = load_local_csv(asset_name.lower(), asset_name)
        cash_df = load_local_csv("fund_2720", "MMF")
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        if WIBOR1M is not None:
            cash_df = build_mmf_extended(cash_df, WIBOR1M, floor_date="1995-01-02")
        
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

        bh_equity, bh_metrics = compute_buy_and_hold(df, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(df, wf_results, trimmed, bh_equity)
        
        if regime_inputs: # <--- DODAJ TO SPRAWDZENIE
            raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
            regime_metrics = extract_flat_regime_stats(raw_regimes)
            print_live_regime_report(regime_metrics)
        else:
            logging.warning(f"Skipping regime analysis for {asset_name} due to insufficient data overlap.")
            regime_metrics = extract_flat_regime_stats({}) # Zwróci słownik z NaN

        mc_res, bb_res = {}, {}
        if self.n_mc > 0:
            logging.info(f"  Running MC for Single Asset ({self.n_mc} iterations)...")
            mc_df = self.rob_engine.run_mc_test(wf_results, df, cash_df, n_samples=self.n_mc)
            mc_res[asset_name] = analyze_robustness(mc_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_MC)
            
        if self.n_boot > 0:
            logging.info(f"  Running Bootstrap for Single Asset ({self.n_boot} iterations)...")
            bb_df = self.rob_engine.run_bootstrap_test(df, cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                                                       use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                                                       X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                                                       fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"])
            bb_res[asset_name] = analyze_bootstrap(bb_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

        return self._compile_full_result(asset_name, train_y, test_y, stop_type, common_start, 
                                         wf_results, trimmed, m, bh_metrics, regime_metrics, mc_res, bb_res)

    def run_pension_iteration(self, train_y, test_y, stop_type_eq, common_start):
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        MMF = load_local_csv("fund_2720", "MMF")
        TBSP = build_and_upload(self.folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", self.creds_path)
        PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR1M, PL10Y, DE10Y, "1995-01-02")
        
        use_atr = (stop_type_eq == "atr")
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(WIG, derived["mmf_ext"], train_y, test_y, use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None, n_jobs=get_n_jobs(), fast_mode=True)
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, derived["mmf_ext"], train_y, test_y, filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"], n_jobs=get_n_jobs(), entry_gate_series=derived["bond_gate"], fast_mode=True)
        
        sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
        port_eq, port_wgts, realloc_log, alloc_df = allocation_walk_forward(derived["ret_eq"], derived["ret_bd"], derived["ret_mmf"], sig_eq, sig_bd, sig_eq, sig_bd, wf_res_eq, wf_res_bd)
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(WIG, wf_res_eq, trimmed, bh_equity)
        

        if regime_inputs: # <--- DODAJ TO SPRAWDZENIE
            raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
            regime_metrics = extract_flat_regime_stats(raw_regimes)
            print_live_regime_report(regime_metrics)
        else:
            logging.warning("Skipping regime analysis for PENSION due to insufficient data overlap.")
            regime_metrics = extract_flat_regime_stats({}) # Zwróci słownik z NaN


        mc_res, bb_res = {}, {}
        if self.n_mc > 0:
            logging.info(f"  Running MC for WIG...")
            mc_df = self.rob_engine.run_mc_test(wf_res_eq, WIG, derived["mmf_ext"], n_samples=self.n_mc)
            mc_res["WIG"] = analyze_robustness(mc_df, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_MC)
            
            logging.info(f"  Running MC for TBSP...")
            mc_df_bd = self.rob_engine.run_mc_test(wf_res_bd, TBSP, derived["mmf_ext"], n_samples=self.n_mc)
            mc_res["TBSP"] = analyze_robustness(mc_df_bd, compute_metrics(wf_bd), thresholds=BOND_THRESHOLDS_MC)

        if self.n_boot > 0:
            logging.info(f"  Running Bootstrap for WIG...")
            bb_df = self.rob_engine.run_bootstrap_test(WIG, derived["mmf_ext"], n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                                                       use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                                                       X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                                                       fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"])
            bb_res["WIG"] = analyze_bootstrap(bb_df, compute_metrics(wf_eq), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)
            
            logging.info(f"  Running Bootstrap for TBSP...")
            bb_df_bd = self.rob_engine.run_bootstrap_test(TBSP, derived["mmf_ext"], n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                                                          filter_modes_override=["ma"], X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
                                                          fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"])
            bb_res["TBSP"] = analyze_bootstrap(bb_df_bd, compute_metrics(wf_bd), thresholds=BOND_THRESHOLDS_BOOTSTRAP)

        return self._compile_full_result("PENSION", train_y, test_y, stop_type_eq, common_start, 
                                         wf_res_eq, trimmed, m, bh_metrics, regime_metrics, mc_res, bb_res)

    def run_global_iteration(self, variant_key, train_y, test_y, stop_type_eq, common_start):
        cfg = ASSET_REGISTRY[variant_key]
        mode, fx_hedged = cfg["mode"], cfg["fx_hedged"]
        
        WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
        WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
        MMF = load_local_csv("fund_2720", "MMF")
        if WIBOR1M is not None:
            MMF = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02")
            
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
        wig_wf_res = None
        
        for lbl, (px_df, fx_s) in assets.items():
            ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret_s, lbl)
            
            wf_e, wf_r, wf_t = walk_forward(
                proc_px, MMF, train_y, test_y, 
                use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None, 
                n_jobs=get_n_jobs(), fast_mode=True
            )
            sigs_full[lbl] = build_signal_series(wf_e, wf_t)
            mc_data_queue[lbl] = (wf_r, proc_px, wf_e, EQUITY_THRESHOLDS_MC, EQUITY_THRESHOLDS_BOOTSTRAP)
            if lbl == "WIG": wig_wf_res = wf_r

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], 
            X_grid=BOND_GRIDS["X_GRID"], n_jobs=get_n_jobs(), fast_mode=True
        )
        rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)
        mc_data_queue["TBSP"] = (wf_res_bd, TBSP, wf_bd, BOND_THRESHOLDS_MC, BOND_THRESHOLDS_BOOTSTRAP)
        
        port_eq, port_wgts, realloc_log, alloc_df = allocation_walk_forward_n(
            rets_dict, sigs_full, sigs_full, MMF["Zamkniecie"].pct_change().dropna(), 
            wf_res_bd, list(rets_dict.keys()), train_years=train_y
        )
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(
            df=WIG, 
            wf_results=wig_wf_res, 
            wf_equity=trimmed, 
            bh_equity=bh_equity
            )
     
        if regime_inputs: # <--- DODAJ TO SPRAWDZENIE
            raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
            regime_metrics = extract_flat_regime_stats(raw_regimes)
            print_live_regime_report(regime_metrics)
        else:
            logging.warning(f"Skipping regime analysis for {variant_key} due to insufficient data overlap.")
            regime_metrics = extract_flat_regime_stats({}) # Zwróci słownik z NaN

        mc_res, bb_res = {}, {}
        for lbl, (res_df, px_df, base_eq, mc_thr, bb_thr) in mc_data_queue.items():
            if self.n_mc > 0 and res_df is not None and not res_df.empty:
                logging.info(f"  Running MC for {lbl}...")
                mc_df = self.rob_engine.run_mc_test(res_df, px_df, MMF, n_samples=self.n_mc)
                mc_res[lbl] = analyze_robustness(mc_df, compute_metrics(base_eq), thresholds=mc_thr)
            if self.n_boot > 0 and res_df is not None and not res_df.empty:
                logging.info(f"  Running Bootstrap for {lbl}...")
                bb_df = self.rob_engine.run_bootstrap_test(
                    px_df, MMF, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                    use_atr_stop=use_atr if lbl != "TBSP" else False, 
                    N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if (use_atr and lbl != "TBSP") else None,
                    filter_modes_override=["ma"] if lbl == "TBSP" else None,
                    X_grid=BOND_GRIDS["X_GRID"] if lbl == "TBSP" else BASE_GRIDS["X_GRID"]
                )
                bb_res[lbl] = analyze_bootstrap(bb_df, compute_metrics(base_eq), thresholds=bb_thr)

        return self._compile_full_result(variant_key, train_y, test_y, stop_type_eq, common_start, 
                                         wig_wf_res, trimmed, m, bh_metrics, regime_metrics, mc_res, bb_res)


def print_sweep_report(results_df: pd.DataFrame, common_start):
    """Outputs a formatted text report to logs."""
    if results_df.empty: return
    sep = "=" * 120
    logging.info(f"\n{sep}\nSWEEP OPTIMIZER REPORT (Common OOS Start: {common_start})\n{sep}")
    logging.info("\n--- 1. PERFORMANCE & ROBUSTNESS LEADERBOARD ---")
    
    display_cols = ["Strategy", "train_years", "test_years", "stop_mode", "oos_cagr", "oos_calmar", "oos_maxdd", "MC_Verdict", "BB_Verdict"]
    if "MC_p05_CAGR" in results_df.columns: display_cols.append("MC_p05_CAGR")
    
    existing_cols = [c for c in display_cols if c in results_df.columns]
    format_df = results_df[existing_cols].copy()
    format_df["oos_cagr"] = (format_df["oos_cagr"] * 100).round(2).astype(str) + "%"
    format_df["oos_maxdd"] = (format_df["oos_maxdd"] * 100).round(2).astype(str) + "%"
    format_df["oos_calmar"] = format_df["oos_calmar"].round(3)
    if "MC_p05_CAGR" in format_df.columns: format_df["MC_p05_CAGR"] = (format_df["MC_p05_CAGR"] * 100).round(2).astype(str) + "%"

    logging.info("\n" + format_df.to_string(index=False))

    regime_cols = [c for c in results_df.columns if "adx_" in c or "vol_" in c]
    if regime_cols:
        logging.info(f"\n{sep}\n--- 2. REGIME DECOMPOSITION: CAGR in Specific Market Conditions, \n as defined by ADX regime - WIG for PENSION and GLOBAL, specific asset for SINGLE) ---")
        key_regime_cols = [
            "Strategy", "train_years", "test_years", "stop_mode",
            "adx_uptrend_strat_cagr", "adx_uptrend_bh_cagr",
            "adx_downtrend_strat_cagr", "adx_downtrend_bh_cagr",
            "adx_sideways_strat_cagr", "adx_sideways_bh_cagr",
        ]
        avail_regime_cols = [c for c in key_regime_cols if c in results_df.columns]
        regime_df = results_df[avail_regime_cols].copy()
        
        for col in avail_regime_cols:
            if "cagr" in col: regime_df[col] = regime_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

        rename_map = {
            "adx_uptrend_strat_cagr": "UP_Strat", "adx_uptrend_bh_cagr": "UP_B&H",
            "adx_downtrend_strat_cagr": "DOWN_Strat", "adx_downtrend_bh_cagr": "DOWN_B&H",
            "adx_sideways_strat_cagr": "RGBOUND_Strat", "adx_sideways_bh_cagr": "RGBOUND_B&H"
        }
        regime_df = regime_df.rename(columns=rename_map)

        logging.info("\n" + regime_df.to_string(index=False))
    logging.info(sep)

def main():
    parser = argparse.ArgumentParser(description="Professional Multi-Strategy Sweeper")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL", "ALL"], required=True)
    parser.add_argument("--assets", nargs="+", help="Dla trybu SINGLE (np. WIG20TR SP500)")
    parser.add_argument("--n_mc", type=int, default=0, help="Liczba probek MC (0 = pomin)")
    parser.add_argument("--n_boot", type=int, default=0, help="Liczba probek Bootstrap (0 = pomin)")
    args = parser.parse_args()

    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"sweep_{args.mode.lower()}_run.log"
    
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    
    manager = SweepManager(args.n_mc, args.n_boot)
    results = []

    updater = DataUpdater(); updater.run_full_update(get_funds=False)

# --- PRZYGOTOWANIE DANYCH DO OOS ---
    logging.info("Preparing extended data series for OOS calculation...")
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    
    # 1. Budujemy/pobieramy serie o długiej historii (Klucze zgodne z ASSET_REGISTRY)
    tbsp_df = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    msciw_df = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
    stoxx_df = build_and_upload(folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path)

    # 2. Inicjalizujemy mapę danych - używamy wyłącznie wielkich liter dla bezpieczeństwa porównań
    data_map = {}
    if tbsp_df is not None: 
        data_map["TBSP"] = tbsp_df
    if msciw_df is not None: 
        data_map["MSCI_WORLD"] = msciw_df
    if stoxx_df is not None: 
        data_map["STOXX600"] = stoxx_df

    # 3. Lista aktywów do doładowania
    check_list = ["WIG", "SP500", "NIKKEI225", "WIG20TR", "MWIG40TR", "SWIG80TR", "NASDAQ100"]
    if args.assets: 
        check_list.extend([a.upper() for a in args.assets])
    
    unique_assets = set([a.upper() for a in check_list])
    
    for asset_key in unique_assets:
        # Jeśli klucz już jest (np. MSCI_WORLD), pomijamy
        if asset_key in data_map:
            continue
            
        # Specjalna obsługa nazw plików (ticker musi być mały)
        ticker_name = asset_key.lower()
        
        # Próba ładowania
        df = load_local_csv(
            ticker=ticker_name, 
            label=asset_key, 
            mandatory=False
        )
        
        if df is not None: 
            data_map[asset_key] = df

    # 5. Debugowanie zawartości przed obliczeniem startu
    for k, v in data_map.items():
        logging.info(f"Final Data Map: {k:<12} | Start: {v.index.min().date()} | End: {v.index.max().date()}")

    # 6. Obliczenie Common OOS Start
    common_start = get_common_oos_start(
        assets_data=data_map, 
        window_configs=SWEEP_WINDOW_CONFIGS
    )
    
    if args.mode in ["SINGLE", "ALL"] and args.assets:
        for asset in args.assets:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info(f"\n{'='*80}\nSINGLE SWEEP: {asset} | {ty}+{te} | {st}\n{'='*80}")
                    res = manager.run_single_asset_iteration(asset, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["GLOBAL", "ALL"]:
        for var in ["GLOBAL_A", "GLOBAL_B"]:
            for ty, te in SWEEP_WINDOW_CONFIGS:
                for st in ["fixed", "atr"]:
                    logging.info(f"\n{'='*80}\nGLOBAL SWEEP: {var} | {ty}+{te} | {st}\n{'='*80}")
                    res = manager.run_global_iteration(var, ty, te, st, common_start)
                    if res: results.append(res)

    if args.mode in ["PENSION", "ALL"]:
        for ty, te in SWEEP_WINDOW_CONFIGS:
            for st in ["fixed", "atr"]:
                logging.info(f"\n{'='*80}\nPENSION SWEEP: {ty}+{te} | {st}\n{'='*80}")
                res = manager.run_pension_iteration(ty, te, st, common_start)
                if res: results.append(res)

    if results:
        final_df = pd.DataFrame(results).sort_values("oos_calmar", ascending=False)
        print_sweep_report(final_df, common_start.date())
        
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        res_path = OUTPUT_DIR / f"sweep_results_{ts}.csv"
        res_path_latest = OUTPUT_DIR / f"sweep_results_{ts}.csv"
      
        win_path =  OUTPUT_DIR / f"sweep_windows_{ts}.csv"
        win_path_latest =  OUTPUT_DIR / f"sweep_windows_{ts}.csv"
        
        final_df.to_csv(res_path, index=False, sep=";")
        final_df.to_csv(res_path_latest, index=False, sep=";")
        
        if manager.all_windows:
            win_df = pd.DataFrame(manager.all_windows)
            win_df.to_csv(win_path, index=False, sep=";")
            win_df.to_csv(win_path_latest , index=False, sep=";")
            logging.info(f"Saved individual windows stats to: {win_path}")

        logging.info(f"Full results saved to: {res_path}")

if __name__ == "__main__":
    main()
