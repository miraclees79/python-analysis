import argparse
import os
import sys
import logging
import datetime as dt
import pandas as pd
import tempfile

# --- ŚCIEŻKI IMPORTU ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
legacy_dir = os.path.join(project_root, 'current_development')
sys.path.append(project_root)
sys.path.append(legacy_dir)

# --- IMPORTUJEMY ORYGINALNĄ BIBLIOTEKĘ ---
from current_development.strategy_test_library import (
    load_stooq_local, get_n_jobs, walk_forward, compute_metrics, 
    analyze_trades, compute_buy_and_hold, print_backtest_report
)
from moj_system.reporting.daily_output import build_daily_outputs
from current_development.stooq_hybrid_updater import run_update

# --- UNIWERSALNE SIATKI PARAMETRÓW (Grids) ---
BASE_GRIDS = {
    "X_GRID":        [0.08, 0.10, 0.12, 0.15, 0.20],
    "Y_GRID":        [0.02, 0.03, 0.05, 0.07, 0.10],
    "FAST_GRID":     [50, 75, 100],
    "SLOW_GRID":     [150, 200, 250],
    "TV_GRID":       [0.08, 0.10, 0.12, 0.15, 0.20],
    "SL_GRID":       [0.05, 0.08, 0.10, 0.15],
    "MOM_LB_GRID":   [126, 252],
    "ATR_WINDOW":    20,
    "N_ATR_GRID":    [0.08, 0.10, 0.12, 0.15, 0.20],
}

# --- REJESTR ASSETÓW ---
ASSET_REGISTRY = {
    "WIG20TR":   {"source": "stooq", "ticker": "wig20tr",  "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}},
    "MWIG40TR":  {"source": "stooq", "ticker": "mwig40tr", "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}},
    "SWIG80TR":  {"source": "stooq", "ticker": "swig80tr", "train": 9, "test": 2, "default_stop": "atr"},
    "SP500":     {"source": "stooq", "ticker": "^spx",     "train": 6, "test": 2, "default_stop": "fixed"},
    "NASDAQ100": {"source": "stooq", "ticker": "ndq100",   "train": 6, "test": 2, "default_stop": "fixed"},
    "MSCI_World":{"source": "drive", "ticker": "URTH",     "train": 9, "test": 1, "default_stop": "atr"},
    "STOXX600":  {"source": "drive", "ticker": "^STOXX",   "train": 7, "test": 1, "default_stop": "atr"},
}

def setup_logging(output_prefix):
    log_file = f"outputs/{output_prefix}_DailyRun.log"
    os.makedirs("outputs", exist_ok=True)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])

def main():
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument("--asset", type=str, required=True, help="Nazwa assetu, np. WIG20TR, SP500")
    parser.add_argument("--force_filter", type=str, default=None, help="Wymuś filtr np. 'ma' lub 'mom'")
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto", 
                        help="Rodzaj trailing stopa: 'fixed' (stały %%), 'atr' (Chandelier exit) lub 'auto' (zgodnie z rejestrem).")
    args = parser.parse_args()

    asset_name = args.asset
    cfg = ASSET_REGISTRY.get(asset_name)
    if not cfg: sys.exit(f"Błąd: Nieznany asset '{asset_name}'.")
    output_prefix = asset_name.lower()
    train_years, test_years = cfg["train"], cfg["test"]
    filter_mode = [args.force_filter] if args.force_filter else None
    
    selected_stop_mode = cfg.get("default_stop", "fixed") if args.stop_mode == "auto" else args.stop_mode
    use_atr_stop = (selected_stop_mode == "atr")

    setup_logging(output_prefix)
    logging.info("=" * 80)
    logging.info(f"DAILY RUNFILE START: {asset_name.upper()} | Train: {train_years}y | Test: {test_years}y")
    logging.info("=" * 80)

    # 1. Update Data
    run_update(get_funds=False)

    # 2. Pobranie danych
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'current_development', 'data'))
    
    MMF = load_stooq_local("fund_2720", "MMF", data_dir, "1990-01-01")
    WIBOR1M = load_stooq_local("wibor1m", "WIBOR1M", data_dir, "1990-01-01", mandatory=False)

    if WIBOR1M is not None:
        from current_development.global_equity_library import build_mmf_extended
        cash_df = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02")
    else:
        cash_df = MMF
        
    df = load_stooq_local(cfg["ticker"], asset_name, data_dir, "1990-01-01")
    if df is None or cash_df is None: sys.exit("Błąd wczytywania danych.")

    grids = BASE_GRIDS.copy()
    if "grids" in cfg: grids.update(cfg["grids"])
    
    n_jobs = get_n_jobs()
    
    # 3. WYWOŁANIE ORYGINALNEGO WALK_FORWARD
    wf_equity, wf_results, wf_trades = walk_forward(
        df=df, cash_df=cash_df, train_years=train_years, test_years=test_years,
        vol_window=20, selected_mode="full",
        filter_modes_override=filter_mode,
        X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"], fast_grid=grids["FAST_GRID"], 
        slow_grid=grids["SLOW_GRID"], tv_grid=grids["TV_GRID"], sl_grid=grids["SL_GRID"], 
        mom_lookback_grid=grids["MOM_LB_GRID"],
        objective="calmar", n_jobs=n_jobs, fast_mode=True,
        use_atr_stop=use_atr_stop,
        N_atr_grid=grids["N_ATR_GRID"] if use_atr_stop else None,
        atr_window=grids["ATR_WINDOW"]
    )

    if wf_equity.empty: sys.exit("Walk-forward nie zwrócił wyników!")

    # 4. Generowanie oryginalnego raportu
    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    trade_stats = analyze_trades(wf_trades, boundary_exits={"CARRY", "SAMPLE_END"})
    
    print_backtest_report(
        metrics=wf_metrics, trades=wf_trades, trade_stats=trade_stats,
        wf_results=wf_results, position_mode="full", filter_modes_override=filter_mode
    )
    
    oos_start = wf_results["TestStart"].min()
    oos_end   = wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, price_col="Zamkniecie", start=oos_start, end=oos_end)
    logging.info("B&H (%s to %s):  CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%",
                 oos_start.date(), oos_end.date(),
                 bh_metrics["CAGR"]*100, bh_metrics["Sharpe"], bh_metrics["MaxDD"]*100)

    # Pobieramy ścieżkę do credentials (tak samo jak w GDriveClient)
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    
    build_daily_outputs(
        wf_equity=wf_equity, wf_trades=wf_trades, wf_metrics=wf_metrics,
        wf_results=wf_results, bh_equity=bh_equity, bh_metrics=bh_metrics, df=df,
        output_dir=f"outputs/{output_prefix}", asset_name=asset_name,
        logfile_name=f"{output_prefix}_signal_log.csv",
        
        # --- DODANE PARAMETRY DLA GDRIVE ---
        gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None
    )

if __name__ == "__main__":
    main()