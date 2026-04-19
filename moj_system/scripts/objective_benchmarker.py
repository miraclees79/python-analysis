# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:27:02 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/scripts/objective_benchmarker.py
===========================================
Performs annual objective function review.
Compares different optimization targets (CalMAR, Sharpe, etc.) 
over the last 5 years to determine the best incumbent.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import tempfile
from datetime import datetime as dt

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)


from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, SWEEP_WINDOW_CONFIGS
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload
from moj_system.core.research import get_common_oos_start

# Core Engine Imports
from moj_system.core.strategy_engine  import (
    walk_forward, compute_metrics, get_n_jobs, annual_cagr_by_year, count_year_wins
)
from moj_system.core.pension_engine import build_signal_series, allocation_walk_forward

OBJECTIVES = ["calmar", "sharpe", "sortino", "calmar_sharpe", "calmar_sortino"]

def main():
    parser = argparse.ArgumentParser(description="Annual Objective Benchmarker")
    parser.add_argument("--strategy", choices=["PENSION", "GLOBAL_B"], default="PENSION")
    parser.add_argument("--years", type=int, default=5, help="Number of recent years to compare wins")
    args = parser.parse_args()

    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    # 1. Load Data
    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    TBSP = build_and_upload(folder_id, "tbsp_extended_full.csv", "tbsp_extended_combined.csv", "^tbsp", "stooq", creds_path)
    MMF = load_local_csv("fund_2720", "MMF")
    
    # 2. Setup Comparison
    common_start = pd.Timestamp("2008-01-04") # Based on previous sweep success
    results = []
    
    logging.info(f"BENCHMARKING OBJECTIVES FOR {args.strategy} STARTING FROM {common_start.date()}")

    for obj in OBJECTIVES:
        logging.info(f">>> Testing Objective: {obj.upper()}")
        
        # Simple Pension simulation for benchmarking
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(WIG, MMF, 8, 2, objective=obj, n_jobs=get_n_jobs())
        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(TBSP, MMF, 8, 2, filter_modes_override=["ma"], objective=obj, n_jobs=get_n_jobs())

        sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
        port_eq, _port_weights, realloc, _alloc_df = allocation_walk_forward(
        WIG["Zamkniecie"].pct_change().dropna(), 
        TBSP["Zamkniecie"].pct_change().dropna(), 
        MMF["Zamkniecie"].pct_change().dropna(), 
        sig_eq, sig_bd, sig_eq, sig_bd, wf_res_eq, wf_res_bd, objective=obj
        )

        # Calculate metrics for the common window
        trimmed = port_eq.loc[port_eq.index >= common_start]
        trimmed = trimmed / trimmed.iloc[0]
        m = compute_metrics(trimmed)
        
        results.append({
            "Objective": obj,
            "CAGR": m["CAGR"],
            "CalMAR": m["CalMAR"],
            "Sharpe": m["Sharpe"],
            "MaxDD": m["MaxDD"],
            "Reallocs": len(realloc)
        })

    # 3. Decision Table Output
    bench_df = pd.DataFrame(results).sort_values("CalMAR", ascending=False)
    print("\n=== OBJECTIVE FUNCTION REVIEW TABLE ===")
    print(bench_df.to_string(index=False))
    
    bench_df.to_csv(f"outputs/objective_benchmark_{args.strategy.lower()}.csv", index=False)

if __name__ == "__main__":
    main()