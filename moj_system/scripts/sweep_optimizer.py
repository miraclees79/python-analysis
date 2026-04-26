# -*- coding: utf-8 -*-
"""
moj_system/scripts/sweep_optimizer.py
======================================
Advanced Strategy Sweeper. Optimized for extended data handling.
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
    asset_name: str, 
    train_years: int, 
    test_years: int,
    wf_results: pd.DataFrame, 
    stop_mode: str,
    alloc_df: pd.DataFrame = None,    # Optymalne wagi (IS)
    weights_series: pd.Series = None  # Faktyczne wagi (OOS)
) -> list[dict]:
    rows = []
    prev_params = None
    use_atr = (stop_mode == "atr")
    stop_col = "N_atr" if use_atr else "X"

    for idx, row in wf_results.iterrows():
        # Podstawowe parametry
        cur_params = {
            "filter_mode": str(row.get("filter_mode", "")),
            "stop_param":  float(row.get(stop_col, np.nan)),
            "Y":           float(row.get("Y", np.nan)),
            "fast":        int(row.get("fast", 0)),
            "slow":        int(row.get("slow", 0)),
            "stop_loss":   float(row.get("stop_loss", np.nan)),
        }

        param_changed = False
        if prev_params is not None:
            param_changed = any(cur_params[k] != prev_params[k] for k in cur_params)

        window_data = {
            "Strategy":      asset_name,
            "train_years":   train_years,
            "test_years":    test_years,
            "stop_mode":     stop_mode,
            "window_idx":    idx,
            "train_start":   str(pd.Timestamp(row["TrainStart"]).date()),
            "test_start":    str(pd.Timestamp(row["TestStart"]).date()),
            "test_end":      str(pd.Timestamp(row["TestEnd"]).date()),
            "win_cagr":      round(float(row.get("CAGR", np.nan)) * 100, 2),
            "win_sharpe":    round(float(row.get("Sharpe", np.nan)), 3),
            "win_maxdd":     round(float(row.get("MaxDD", np.nan)) * 100, 2),
            "win_calmar":    round(float(row.get("CalMAR", np.nan)), 3),
            **cur_params,
            "param_changed": param_changed,
        }

        # --- DODAWANIE WAG OPTYMALNYCH (IS) ---
        if alloc_df is not None:
            # Szukamy wiersza w alloc_df pasującego do daty testowej okienka
            match = alloc_df[alloc_df["TestStart"] == row["TestStart"]]
            if not match.empty:
                alloc_row = match.iloc[0]
                # Pobieramy wszystkie kolumny zaczynające się od 'w_'
                for col in alloc_df.columns:
                    if col.startswith("w_"):
                        window_data[f"opt_{col}"] = round(float(alloc_row[col]), 3)

        # --- DODAWANIE ŚREDNICH FAKTYCZNYCH WAG (OOS) ---
        if weights_series is not None:
            # Wycinamy faktyczne wagi dzienne dla zakresu dat tego okienka
            ts = pd.Timestamp(row["TestStart"])
            te = pd.Timestamp(row["TestEnd"])
            mask = (weights_series.index >= ts) & (weights_series.index <= te)
            win_weights = weights_series.loc[mask]
            
            if not win_weights.empty:
                # win_weights to seria słowników, zamieniamy na DataFrame do średniej
                weights_df = pd.DataFrame(data=list(win_weights.values))
                for asset_col in weights_df.columns:
                    mean_w = weights_df[asset_col].mean()
                    window_data[f"actual_avg_w_{asset_col}"] = round(float(mean_w), 3)

        rows.append(window_data)
        prev_params = cur_params
    return rows

def _compile_window_stats(window_rows: list) -> dict:
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
    def __init__(self, n_mc, n_boot, data_map):
        self.n_mc = n_mc
        self.n_boot = n_boot
        self.data_map = data_map
        self.rob_engine = RobustnessEngine(n_jobs=get_n_jobs())
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        self.all_windows = [] 
        self.wf_cache = {} 
        self.mc_cache = {}
        self.boot_cache = {}

    def get_cached_wf(self, asset_name, df, train_y, test_y, stop_type, grid_type="EQUITY", entry_gate=None):
        gate_id = "GATED" if entry_gate is not None else "RAW"
        cache_key = (asset_name, train_y, test_y, stop_type, gate_id)
        if cache_key in self.wf_cache:
            logging.info(f"  [WF CACHE HIT] {asset_name} ")
            return self.wf_cache[cache_key]

        logging.info(f"  [WF CACHE MISS] Calculating WF: {asset_name} ...")
        cash_df = self.data_map.get("MMF_EXT")
        grids = BOND_GRIDS if grid_type == "BOND" else BASE_GRIDS
        use_atr = (stop_type == "atr")
        
        wf_equity, wf_results, wf_trades = walk_forward(
            df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
            X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"],
            fast_grid=grids["FAST_GRID"], slow_grid=grids["SLOW_GRID"],
            use_atr_stop=use_atr, N_atr_grid=grids["N_ATR_GRID"] if use_atr else None,
            entry_gate_series=entry_gate, n_jobs=get_n_jobs(), fast_mode=True
        )
        self.wf_cache[cache_key] = (wf_equity, wf_results, wf_trades)
        return wf_equity, wf_results, wf_trades

    def get_cached_mc(self, asset_name, wf_results, df, cash_df, n_samples, thresholds, train_y, test_y, stop_type, gate_id, base_equity):
        cache_key = (asset_name, train_y, test_y, stop_type, gate_id, n_samples)
        if cache_key in self.mc_cache:
            logging.info(f"  [MC CACHE HIT] {asset_name}")
            return self.mc_cache[cache_key]

        logging.info(f"  [MC CACHE MISS] Running MC: {asset_name}...")
        mc_df = self.rob_engine.run_mc_test(wf_results=wf_results, df=df, cash_df=cash_df, n_samples=n_samples)
        # POPRAWKA: baseline_metrics liczone z krzywej kapitału, nie z wyników okienek
        result = analyze_robustness(results_df=mc_df, baseline_metrics=compute_metrics(base_equity), thresholds=thresholds)
        self.mc_cache[cache_key] = result
        return result

    def get_cached_boot(self, asset_name, df, cash_df, n_samples, train_y, test_y, stop_type, grid_type, entry_gate, thresholds, base_equity):
        gate_id = "GATED" if entry_gate is not None else "RAW"
        cache_key = (asset_name, train_y, test_y, stop_type, gate_id, n_samples)
        if cache_key in self.boot_cache:
            logging.info(f"  [BOOT CACHE HIT] {asset_name}")
            return self.boot_cache[cache_key]

        logging.info(f"  [BOOT CACHE MISS] Running Boot: {asset_name}...")
        use_atr = (stop_type == "atr")
        grids = BOND_GRIDS if grid_type == "BOND" else BASE_GRIDS
        bb_df = self.rob_engine.run_bootstrap_test(
            df=df, cash_df=cash_df, n_samples=n_samples, train_years=train_y, test_years=test_y,
            use_atr_stop=use_atr, N_atr_grid=grids["N_ATR_GRID"] if use_atr else None,
            X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"],
            fast_grid=grids["FAST_GRID"], slow_grid=grids["SLOW_GRID"],
            filter_modes_override=["ma"] if grid_type == "BOND" else None, fast_mode=True
        )
        result = analyze_bootstrap(results_df=bb_df, baseline_metrics=compute_metrics(base_equity), thresholds=thresholds)
        self.boot_cache[cache_key] = result
        return result

    def _prepare_pension_data(self):
        return build_standard_two_asset_data(
            wig=self.data_map["WIG"], tbsp=self.data_map["TBSP"], mmf=self.data_map["MMF_EXT"],
            wibor1m=self.data_map.get("WIBOR1M"), pl10y=self.data_map["PL10Y"], de10y=self.data_map["DE10Y"], mmf_floor="1995-01-02"
        )

    def _compile_full_result(self, strat_name, train_y, test_y, stop_type, common_start, 
                             wf_results, wf_equity_trimmed, m_trimmed, bh_metrics, 
                             regime_metrics, mc_verdicts_dict, bb_verdicts_dict,
                             alloc_df=None, weights_series=None): # <--- DODANO
        
        # Przekazujemy wagi do ekstrakcji
        window_rows = _extract_window_rows(
            asset_name=strat_name, train_years=train_y, test_years=test_y, 
            wf_results=wf_results, stop_mode=stop_type,
            alloc_df=alloc_df, weights_series=weights_series # <--- DODANO
        )
        self.all_windows.extend(window_rows)
        win_stats = _compile_window_stats(window_rows)

        mc_verdict = "|".join([f"{k[:3]}:{v['verdict'][:3]}" for k,v in mc_verdicts_dict.items()]) if mc_verdicts_dict else "N/A"
        bb_verdict = "|".join([f"{k[:3]}:{v['verdict'][:3]}" for k,v in bb_verdicts_dict.items()]) if bb_verdicts_dict else "N/A"
        
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
        asset_key = asset_name.upper()
        df = self.data_map.get(asset_key)
        cash_df = self.data_map.get("MMF_EXT")
 
        
        if df is None or cash_df is None:
            logging.error(f"Missing pre-loaded data for {asset_name}")
            return None

        use_atr = (stop_type == "atr")
        wf_equity, wf_results, wf_trades = self.get_cached_wf(
                    asset_name=asset_name, 
                    df=df, 
                    train_y=train_y, 
                    test_y=test_y, 
                    stop_type=stop_type
                    )   
        if wf_equity.empty: return None

        trimmed = wf_equity.loc[wf_equity.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        bh_equity, bh_metrics = compute_buy_and_hold(df, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(df=df, wf_results=wf_results, wf_equity=trimmed, bh_equity=bh_equity)
        
        if regime_inputs:
            raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
            regime_metrics = extract_flat_regime_stats(raw_regimes)
            print_live_regime_report(regime_metrics)
        else:
            regime_metrics = extract_flat_regime_stats({})

        mc_res, bb_res = {}, {}
        if self.n_mc > 0:
            mc_df = self.rob_engine.run_mc_test(wf_results, df, cash_df, n_samples=self.n_mc)
            mc_res[asset_name] = analyze_robustness(mc_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_MC)
            
        if self.n_boot > 0:
            bb_df = self.rob_engine.run_bootstrap_test(df, cash_df, n_samples=self.n_boot, train_years=train_y, test_years=test_y,
                                                       use_atr_stop=use_atr, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                                                       X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                                                       fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"])
            bb_res[asset_name] = analyze_bootstrap(bb_df, compute_metrics(wf_equity), thresholds=EQUITY_THRESHOLDS_BOOTSTRAP)

        return self._compile_full_result(asset_name, train_y, test_y, stop_type, common_start, 
                                         wf_results, trimmed, m, bh_metrics, regime_metrics, mc_res, bb_res)

    def run_pension_iteration(self, train_y, test_y, stop_type_eq, common_start):
        
        WIG = self.data_map.get("WIG")
        TBSP = self.data_map.get("TBSP")
        MMF_EXT = self.data_map.get("MMF_EXT")
        
        derived = self._prepare_pension_data()
        
        # 1. WF (Signals)
        wf_eq, wf_res_eq, wf_tr_eq = self.get_cached_wf(asset_name="WIG", 
                                                        df=WIG, 
                                                        train_y=train_y, 
                                                        test_y=test_y, 
                                                        stop_type=stop_type_eq)
        wf_bd, wf_res_bd, wf_tr_bd = self.get_cached_wf(asset_name="TBSP", 
                                                        df=TBSP, 
                                                        train_y=train_y, 
                                                        test_y=test_y, 
                                                        stop_type="fixed", 
                                                        grid_type="BOND", 
                                                        entry_gate=derived["bond_gate"])
        


        # 2. MC Robustness
        mc_res = {}
        if self.n_mc > 0:
            mc_res["WIG"] = self.get_cached_mc(asset_name="WIG", 
                                               wf_results=wf_res_eq, 
                                               df=WIG, 
                                               cash_df=MMF_EXT, 
                                               n_samples=self.n_mc, 
                                               thresholds=EQUITY_THRESHOLDS_MC, 
                                               train_y=train_y, 
                                               test_y=test_y, 
                                               stop_type=stop_type_eq, 
                                               gate_id="RAW",
                                               base_equity=wf_eq)
            mc_res["TBSP"] = self.get_cached_mc(asset_name="TBSP", 
                                                wf_results=wf_res_bd, 
                                                df=TBSP, 
                                                cash_df=MMF_EXT, 
                                                n_samples=self.n_mc, 
                                                thresholds=BOND_THRESHOLDS_MC, 
                                                train_y=train_y, 
                                                test_y=test_y, 
                                                stop_type="fixed", 
                                                gate_id="GATED",
                                                base_equity=wf_bd
                                                )

        # 3. Bootstrap Robustness
        bb_res = {}
        if self.n_boot > 0:
            bb_res["WIG"] = self.get_cached_boot(asset_name="WIG", 
                                                 df=WIG, 
                                                 cash_df=MMF_EXT, 
                                                 n_samples=self.n_boot, 
                                                 train_y=train_y, 
                                                 test_y=test_y, 
                                                 stop_type=stop_type_eq, 
                                                 grid_type="EQUITY", 
                                                 entry_gate=None, 
                                                 thresholds=EQUITY_THRESHOLDS_BOOTSTRAP, 
                                                 base_equity= wf_eq)
            bb_res["TBSP"] = self.get_cached_boot(asset_name="TBSP", 
                                                  df=TBSP, 
                                                  cash_df=MMF_EXT, 
                                                  n_samples=self.n_boot, 
                                                  train_y=train_y, 
                                                  test_y=test_y, 
                                                  stop_type="fixed", 
                                                  grid_type="BOND", 
                                                  entry_gate=derived["bond_gate"], 
                                                  thresholds=BOND_THRESHOLDS_BOOTSTRAP,
                                                  base_equity=wf_bd)

         # 4. Allocation OOS
        sig_eq, sig_bd = build_signal_series(wf_equity=wf_eq, wf_trades=wf_tr_eq), build_signal_series(wf_equity=wf_bd, wf_trades=wf_tr_bd)
        port_eq, port_wgts, realloc_log, alloc_df = allocation_walk_forward(equity_returns=derived["ret_eq"], bond_returns=derived["ret_bd"], mmf_returns=derived["ret_mmf"], sig_equity_full=sig_eq, sig_bond_full=sig_bd, sig_equity_oos=sig_eq, sig_bond_oos=sig_bd, wf_results_eq=wf_res_eq, wf_results_bd=wf_res_bd)
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(df=WIG, wf_results=wf_res_eq, wf_equity=trimmed, bh_equity=bh_equity)
                
        if regime_inputs:
             raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
             regime_metrics = extract_flat_regime_stats(raw_regimes)
             print_live_regime_report(regime_metrics)
        else:
            regime_metrics = extract_flat_regime_stats({})
            
        return self._compile_full_result(
            strat_name="PENSION", train_y=train_y, test_y=test_y, stop_type=stop_type_eq, 
            common_start=common_start, wf_results=wf_res_eq, wf_equity_trimmed=trimmed, 
            m_trimmed=m, bh_metrics=bh_metrics, regime_metrics=regime_metrics, 
            mc_verdicts_dict=mc_res, bb_verdicts_dict=bb_res,
            alloc_df=alloc_df, weights_series=port_wgts 
            )

    def run_global_iteration(self, variant_key, train_y, test_y, stop_type_eq, common_start):
        cfg = ASSET_REGISTRY[variant_key]
        mode, fx_hedged = cfg["mode"], cfg["fx_hedged"]
        
        WIG = self.data_map.get("WIG")
        MMF_EXT = self.data_map.get("MMF_EXT")
        TBSP = self.data_map.get("TBSP")
        derived = self._prepare_pension_data()
        fx_map = {c: self.data_map.get(f"{c}PLN")["Zamkniecie"] for c in ["USD", "EUR", "JPY"]}
        
        if mode == "global_equity":
            assets = {
                "WIG": (WIG, None), 
                "SP500": (self.data_map.get("SP500"), fx_map["USD"]), 
                "STOXX600": (self.data_map.get("STOXX600"), fx_map["EUR"]), 
                "NIKKEI225": (self.data_map.get("NIKKEI225"), fx_map["JPY"])
            }
        else:
            assets = {
                "WIG": (WIG, None), 
                "MSCI_WORLD": (self.data_map.get("MSCI_WORLD"), fx_map["USD"])
            }

        rets_dict, sigs_full, mc_res, bb_res = {}, {}, {}, {}
        
        
        
        
        for lbl, (px_df, fx_s) in assets.items():
            ret_s = build_return_series(price_df=px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = px_df if fx_hedged or fx_s is None else build_price_df_from_returns(ret=ret_s, label=lbl)
            
            
            
            # POPRAWKA: Dodano df=proc_px i jawne nazewnictwo
            wf_e, wf_r, wf_t = self.get_cached_wf(
                asset_name=lbl, 
                df=proc_px, 
                train_y=train_y, 
                test_y=test_y, 
                stop_type=stop_type_eq
            )
            
            
            sigs_full[lbl] = build_signal_series(wf_equity=wf_e, wf_trades=wf_t)
            
            # MC & Boot (Global components)
            if self.n_mc > 0:
                mc_res[lbl] = self.get_cached_mc(
                    asset_name=lbl, 
                    wf_results=wf_r, 
                    df=proc_px, 
                    cash_df=MMF_EXT, 
                    n_samples=self.n_mc, 
                    thresholds=EQUITY_THRESHOLDS_MC, 
                    train_y=train_y, 
                    test_y=test_y, 
                    stop_type=stop_type_eq, 
                    gate_id="RAW",
                    base_equity=wf_e)
            if self.n_boot > 0:
                bb_res[lbl] = self.get_cached_boot(asset_name=lbl, 
                                                   df=proc_px, 
                                                   cash_df=MMF_EXT, 
                                                   n_samples=self.n_boot, 
                                                   train_y=train_y, 
                                                   test_y=test_y, 
                                                   stop_type=stop_type_eq, 
                                                   grid_type="EQUITY", 
                                                   entry_gate=None, 
                                                   thresholds=EQUITY_THRESHOLDS_BOOTSTRAP, 
                                                   base_equity=wf_e)

        # TBSP (Shared Gated)
        wf_bd_e, wf_bd_r, wf_bd_t = self.get_cached_wf(asset_name="TBSP", 
                                                        df=TBSP, 
                                                        train_y=train_y, 
                                                        test_y=test_y, 
                                                        stop_type="fixed", 
                                                        grid_type="BOND", 
                                                        entry_gate=derived["bond_gate"])
        if self.n_mc > 0:
            mc_res["TBSP"] = self.get_cached_mc(asset_name="TBSP", 
                                                wf_results=wf_bd_r, 
                                                df=TBSP, 
                                                cash_df=MMF_EXT, 
                                                n_samples=self.n_mc, 
                                                thresholds=BOND_THRESHOLDS_MC, 
                                                train_y=train_y, 
                                                test_y=test_y, 
                                                stop_type="fixed", 
                                                gate_id="GATED",
                                                base_equity=wf_bd_e)
        if self.n_boot > 0:
            bb_res["TBSP"] = self.get_cached_boot(asset_name="TBSP", 
                                                  df=TBSP, 
                                                  cash_df=MMF_EXT, 
                                                  n_samples=self.n_boot, 
                                                  train_y=train_y, 
                                                  test_y=test_y, 
                                                  stop_type="fixed", 
                                                  grid_type="BOND", 
                                                  entry_gate=derived["bond_gate"], 
                                                  thresholds=BOND_THRESHOLDS_BOOTSTRAP, 
                                                  base_equity=wf_bd_e)
        
        port_eq, port_wgts, realloc_log, alloc_df = allocation_walk_forward_n(
            rets_dict, sigs_full, sigs_full, MMF_EXT["Zamkniecie"].pct_change().dropna(), 
            wf_bd_r, list(rets_dict.keys()), train_years=train_y
        )
        
        trimmed = port_eq.loc[port_eq.index >= common_start]
        if trimmed.empty: return None
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)

        bh_equity, bh_metrics = compute_buy_and_hold(WIG, "Zamkniecie", common_start, trimmed.index.max())
        regime_inputs = prepare_regime_inputs(df=WIG, wf_results=None, wf_equity=trimmed, bh_equity=bh_equity)
     
        if regime_inputs:
            raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
            regime_metrics = extract_flat_regime_stats(raw_regimes)
            print_live_regime_report(regime_metrics)
        else:
            regime_metrics = extract_flat_regime_stats({})

        portfolio_wf_results = wf_bd_r.copy()

        for row_index, row in portfolio_wf_results.iterrows():
            w_start = row["TestStart"]
            w_end   = row["TestEnd"]
            
            # Wycinamy kawałek krzywej kapitału portfela dla tego okienka
            # port_eq to pełna krzywa OOS wyliczona przez allocation_walk_forward_n
            window_equity_slice = port_eq.loc[w_start:w_end]
            
            if not window_equity_slice.empty:
                # Normalizujemy plasterek do 1.0 na starcie okienka
                window_equity_norm = window_equity_slice / window_equity_slice.iloc[0]
                
                # Liczymy metryki dla tego konkretnego okienka
                m_win = compute_metrics(equity=window_equity_norm)
                
                # Nadpisujemy metryki TBSP metrykami całego portfela
                portfolio_wf_results.loc[row_index, "CAGR"]   = m_win["CAGR"]
                portfolio_wf_results.loc[row_index, "Sharpe"] = m_win["Sharpe"]
                portfolio_wf_results.loc[row_index, "MaxDD"]  = m_win["MaxDD"]
                portfolio_wf_results.loc[row_index, "CalMAR"] = m_win["CalMAR"]
                
                # Parametry techniczne (fast, slow itp.) dla portfela nie mają 
                # sensu w jednym polu, więc możemy je oznaczyć jako NaN lub 0
                portfolio_wf_results.loc[row_index, "filter_mode"] = "PORTFOLIO"

        return self._compile_full_result(
            strat_name=variant_key, train_y=train_y, test_y=test_y, stop_type=stop_type_eq, 
            common_start=common_start, wf_results=portfolio_wf_results, wf_equity_trimmed=trimmed, 
            m_trimmed=m, bh_metrics=bh_metrics, regime_metrics=regime_metrics, 
            mc_verdicts_dict=mc_res, bb_verdicts_dict=bb_res,
            alloc_df=alloc_df, weights_series=port_wgts
        )


def print_sweep_report(results_df: pd.DataFrame, common_start):
    if results_df.empty: return
    sep = "=" * 120
    logging.info(f"\n{sep}\nSWEEP OPTIMIZER REPORT (Common OOS Start: {common_start})\n{sep}")
    logging.info("\n--- 1. PERFORMANCE & ROBUSTNESS LEADERBOARD ---")
    
    display_cols = ["Strategy", "train_years", "test_years", "stop_mode", "oos_cagr", "oos_calmar", "oos_maxdd", "MC_Verdict", "BB_Verdict"]
    if "MC_p05_CAGR" in results_df.columns: display_cols.append("MC_p05_CAGR")
    
    existing_cols = [c for c in display_cols if c in results_df.columns]
    format_df = results_df[existing_cols].copy()
    # Słownik kolumn do sformatowania jako % (mnożymy przez 100 i dodajemy %)
    pct_cols = ["oos_cagr", "oos_maxdd", "MC_p05_CAGR"]

    for col in pct_cols:
        if col in format_df.columns:
            format_df[col] = format_df[col].apply(
                func=lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A"
                )

    # Kolumny numeryczne (tylko zaokrąglenie)
    if "oos_calmar" in format_df.columns:
        format_df["oos_calmar"] = format_df["oos_calmar"].apply(
            func=lambda x: round(x, 3) if pd.notna(x) else "N/A"
            )

    

    logging.info("\n" + format_df.to_string(index=False))

    regime_cols = [c for c in results_df.columns if "adx_" in c or "vol_" in c]
    if regime_cols:
        logging.info(f"\n{sep}\n--- 2. REGIME DECOMPOSITION: CAGR in Specific Market Conditions ---")
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
        logging.info("\n" + regime_df.to_string(index=False))
    logging.info(sep)

def main():
    parser = argparse.ArgumentParser(description="Professional Multi-Strategy Sweeper")
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL", "ALL"], required=True)
    parser.add_argument("--assets", nargs="+", help="Dla trybu SINGLE (np. WIG20TR SP500)")
    parser.add_argument("--n_mc", type=int, default=0, help="Liczba probek MC")
    parser.add_argument("--n_boot", type=int, default=0, help="Liczba probek Bootstrap")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / f"sweep_{args.mode.lower()}_run.log"
    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    
    # 1. Update data
    DataUpdater().run_full_update(get_funds=False)

    logging.info("Preparing data map for all assets...")
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    
    # Przechowujemy wszystkie dane w jednej mapie (UPPERCASE)
    data_map = {}
    
    # Ładowanie walut dla GLOBAL
    for c in ["USD", "EUR", "JPY"]:
        df = load_local_csv(f"{c.lower()}pln", f"{c}PLN")
        if df is not None: data_map[f"{c}PLN"] = df

    # Ładowanie serii przedłużonych
    data_map["TBSP"] = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    data_map["MSCI_WORLD"] = build_and_upload(folder_id, "msci_world_wsj_raw.csv", "msci_world_combined.csv", "URTH", "yfinance", creds_path, is_msci_world=True)
    data_map["STOXX600"] = build_and_upload(folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path)

    # Ładowanie reszty
    check_list = ["WIG", "SP500", "NIKKEI225", "WIG20TR", "NASDAQ100", "PL10Y", "DE10Y", "WIBOR1M"]
    if args.assets: check_list.extend([a.upper() for a in args.assets])
    
    for asset_key in set(check_list):
        if asset_key in data_map: continue
        df = load_local_csv(asset_key.lower(), asset_key, mandatory=False)
        if df is not None: data_map[asset_key] = df

    # Przygotowanie przedłużonego MMF (raz dla wszystkich)
    mmf_raw = load_local_csv("fund_2720", "MMF")
    wibor = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    data_map["MMF_EXT"] = build_mmf_extended(mmf_raw, wibor, floor_date="1995-01-02")

    # 2. Calculate Common OOS Start
    common_start = get_common_oos_start(data_map, SWEEP_WINDOW_CONFIGS)

    # 3. Init Manager with the map
    manager = SweepManager(args.n_mc, args.n_boot, data_map)
    results = []

    # 4. Iteration loops
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

    # 5. Saving
    if results:
        final_df = pd.DataFrame(results).sort_values("oos_calmar", ascending=False)
        print_sweep_report(final_df, common_start.date())
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        final_df.to_csv(OUTPUT_DIR / f"sweep_results_{ts}.csv", index=False, sep=";")
        final_df.to_csv(OUTPUT_DIR / "sweep_results_latest.csv", index=False, sep=";")
        if manager.all_windows:
            win_df = pd.DataFrame(manager.all_windows)
            win_df.to_csv(OUTPUT_DIR / f"sweep_windows_{ts}.csv", index=False, sep=";")
            win_df.to_csv(OUTPUT_DIR / "sweep_windows_latest.csv", index=False, sep=";")

if __name__ == "__main__":
    main()