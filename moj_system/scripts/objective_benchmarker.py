# -*- coding: utf-8 -*-
"""
moj_system/scripts/objective_benchmarker.py
===========================================
Performs annual objective function review for the PENSION strategy.
Compares different optimization targets (CalMAR, Sharpe, etc.)
using the validated 7+1 ATR configuration.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# --- PATH SETUP ---
from moj_system.config import OUTPUT_DIR, SWEEP_WINDOW_CONFIGS
from moj_system.core.pension_engine import (
    allocation_walk_forward,
    build_signal_series,
    build_standard_two_asset_data,
)
from moj_system.core.research import get_common_oos_start

# --- CORE ENGINE IMPORTS ---
from moj_system.core.strategy_engine import (
    compute_metrics,
    get_n_jobs,
    walk_forward,
)
from moj_system.data.builder import build_and_upload
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater

# Definicja testowanych funkcji celu
OBJECTIVES = ["calmar", "sharpe", "sortino", "calmar_sharpe", "calmar_sortino"]

def run_benchmark() -> None:
    # 1. Setup i logowanie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = OUTPUT_DIR / "objective_benchmark.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file_path, mode="w", encoding='utf-8'),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )

    logging.info(msg="=== STARTING ANNUAL OBJECTIVE FUNCTION REVIEW (PENSION 7+1 ATR) ===")

    # 2. Ładowanie i przygotowanie danych (Standardowy potok)
    DataUpdater().run_full_update(get_funds=False)

    creds_path = Path(tempfile.gettempdir()) / "credentials.json"
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")

    # Pobieramy pełne serie (identycznie jak w Sweep)
    WIG = load_local_csv(ticker="wig", label="WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    TBSP = build_and_upload(
        folder_id=folder_id,
        raw_filename="tbsp_extended_full.csv",
        combined_filename="tbsp_extended_combined.csv",
        extension_ticker="^tbsp",
        extension_source="stooq",
        credentials_path=str(creds_path),
    )
    MMF = load_local_csv(ticker="fund_2720", label="MMF")
    WIBOR = load_local_csv(ticker="wibor1m", label="WIBOR1M", mandatory=False)
    PL10Y = load_local_csv(ticker="pl10y", label="PL10Y")
    DE10Y = load_local_csv(ticker="de10y", label="DE10Y")

    derived = build_standard_two_asset_data(
        wig=WIG, tbsp=TBSP, mmf=MMF, wibor1m=WIBOR,
        pl10y=PL10Y, de10y=DE10Y, mmf_floor="1995-01-02",
    )

    # Obliczamy Common Start (2013), aby porównanie było rzetelne
    data_map = {"WIG": WIG, "TBSP": TBSP}
    common_start = get_common_oos_start(assets_data=data_map, window_configs=SWEEP_WINDOW_CONFIGS)

    results = []
    n_jobs = get_n_jobs()

    # 3. Pętla przez funkcje celu
    for obj in OBJECTIVES:
        logging.info(msg=f"\n>>> TESTING OBJECTIVE: {obj.upper()}")

        # Walk-forward dla komponentów z użyciem testowanego celu (PENSION 7+1 ATR)
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
            df=WIG,
            cash_df=derived["mmf_ext"],
            train_years=7,
            test_years=1,
            use_atr_stop=True,
            objective=obj,
            n_jobs=n_jobs,
        )

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=TBSP,
            cash_df=derived["mmf_ext"],
            train_years=7,
            test_years=1,
            filter_modes_override=["ma"],
            objective=obj,
            n_jobs=n_jobs,
            entry_gate_series=derived["bond_gate"],
        )

        # Rebranding sygnałów
        sig_eq = build_signal_series(wf_equity=wf_eq, wf_trades=wf_tr_eq)
        sig_bd = build_signal_series(wf_equity=wf_bd, wf_trades=wf_tr_bd)

        # Alokacja z użyciem testowanego celu
        port_eq, weights_series, reallocation_log, alloc_df = allocation_walk_forward(
            equity_returns=derived["ret_eq"],
            bond_returns=derived["ret_bd"],
            mmf_returns=derived["ret_mmf"],
            sig_equity_full=sig_eq,
            sig_bond_full=sig_bd,
            sig_equity_oos=sig_eq,
            sig_bond_oos=sig_bd,
            wf_results_eq=wf_res_eq,
            wf_results_bd=wf_res_bd,
            objective=obj,
        )

        # Przycięcie do wspólnego okna OOS dla porównania
        trimmed = port_eq.loc[port_eq.index >= common_start]
        trimmed_norm = trimmed / trimmed.iloc[0]
        m = compute_metrics(equity=trimmed_norm)

        results.append({
            "Objective": obj,
            "CAGR": m["CAGR"],
            "CalMAR": m["CalMAR"],
            "Sharpe": m["Sharpe"],
            "MaxDD": m["MaxDD"],
            "Reallocs": len(reallocation_log),
        })

    # 4. Tabela Decyzyjna
    bench_df = pd.DataFrame(data=results).sort_values(by="CalMAR", ascending=False)

    logging.info(msg="\n" + "="*50)
    logging.info(msg="FINAL OBJECTIVE COMPARISON TABLE")
    logging.info(msg="="*50)
    logging.info(msg="\n" + bench_df.to_string(index=False))

    # Zapis wyniku
    output_path = OUTPUT_DIR / "objective_benchmark_results.csv"
    bench_df.to_csv(path_or_buf=output_path, index=False, sep=";")
    logging.info(msg=f"\nBenchmark results saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
