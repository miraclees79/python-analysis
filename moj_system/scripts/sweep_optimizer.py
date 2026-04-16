# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'current_development'))

# Imports from new architecture
from moj_system.config import BASE_GRIDS, SWEEP_WINDOW_CONFIGS, EQUITY_THRESHOLDS_MC
from moj_system.data.data_manager import load_local_csv
from moj_system.core.robustness import RobustnessEngine
from moj_system.core.research import get_common_oos_start # <-- IMPORTED RESEARCH UTILS

# Imports from legacy library
from current_development.strategy_test_library import walk_forward, compute_metrics, get_n_jobs
from current_development.mc_robustness import analyze_robustness

def main():
    parser = argparse.ArgumentParser(description="Strategy Configuration Sweeper with Common OOS")
    parser.add_argument("--asset", type=str, required=True, help="e.g. WIG20TR, SP500")
    parser.add_argument("--n_mc", type=int, default=100, help="Number of MC samples per config")
    args = parser.parse_args()

    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Load Data
    asset_lower = args.asset.lower()
    df = load_local_csv(asset_lower, args.asset)
    cash_df = load_local_csv("fund_2720", "MMF")

    # 2. Calculate Common OOS Start Date
    # This ensures that a 6y train config and a 9y train config are evaluated on the same days
    common_start = get_common_oos_start({args.asset: df}, SWEEP_WINDOW_CONFIGS)
    logging.info(f"Leaderboard will be calculated starting from: {common_start.date()}")

    results = []
    rob_engine = RobustnessEngine(n_jobs=get_n_jobs())

    # 3. Outer Loop: Window Configs
    for train_y, test_y in SWEEP_WINDOW_CONFIGS:
        for stop_type in ["fixed", "atr"]:
            logging.info(f"--- TESTING: {train_y}+{test_y} | {stop_type} ---")
            
            use_atr = (stop_type == "atr")
            
            # Run Walk-Forward
            wf_equity, wf_results, _ = walk_forward(
                df=df, cash_df=cash_df, train_years=train_y, test_years=test_y,
                X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
                tv_grid=BASE_GRIDS["TV_GRID"], sl_grid=BASE_GRIDS["SL_GRID"],
                mom_lookback_grid=BASE_GRIDS["MOM_LB_GRID"],
                use_atr_stop=use_atr, 
                N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                atr_window=BASE_GRIDS["ATR_WINDOW"],
                n_jobs=get_n_jobs()
            )

            if wf_equity.empty: continue

            # --- KEY STEP: Trim equity to common OOS start ---
            # We only evaluate performance from the date where ALL configs have OOS data
            wf_equity_trimmed = wf_equity.loc[wf_equity.index >= common_start]
            if wf_equity_trimmed.empty:
                logging.warning(f"No data for {train_y}+{test_y} after common start {common_start}")
                continue
            
            # Re-normalize trimmed equity to 1.0 at start of evaluation
            wf_equity_trimmed = wf_equity_trimmed / wf_equity_trimmed.iloc[0]
            # -------------------------------------------------

            # Run Lite MC Perturbation on the FULL available OOS for better stats
            mc_df = rob_engine.run_mc_test(wf_results, df, cash_df, n_samples=args.n_mc)
            
            # We analyze MC results against the trimmed baseline to be consistent
            baseline_metrics = compute_metrics(wf_equity_trimmed)
            mc_summary = analyze_robustness(mc_df, baseline_metrics, thresholds=EQUITY_THRESHOLDS_MC)

            results.append({
                "Config": f"{train_y}+{test_y}",
                "StopType": stop_type,
                "CAGR": round(baseline_metrics["CAGR"], 4),
                "CalMAR": round(baseline_metrics["CalMAR"], 3),
                "MaxDD": round(baseline_metrics["MaxDD"], 4),
                "MC_Verdict": mc_summary["verdict"],
                "MC_p05_CAGR": round(mc_summary["CAGR"]["p05"], 4),
                "OOS_Start": str(wf_equity_trimmed.index.min().date())
            })

    # 4. Save & Display
    if results:
        sweep_df = pd.DataFrame(results).sort_values("CalMAR", ascending=False)
        os.makedirs("outputs", exist_ok=True)
        sweep_df.to_csv(f"outputs/sweep_{asset_lower}.csv", index=False)
        
        print(f"\nLEADERBOARD (Evaluated from {common_start.date()}):")
        print(sweep_df.head(10).to_string(index=False))
    else:
        logging.error("No results generated.")

if __name__ == "__main__":
    main()