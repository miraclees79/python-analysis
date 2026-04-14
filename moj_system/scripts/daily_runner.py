import argparse
import os
import sys
import logging
import datetime as dt

# --- NAPRAWA ŚCIEŻEK IMPORTU ---
# Pobieramy ścieżkę główną projektu (dwa foldery wyżej niż ten skrypt)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Ścieżka do starego folderu (aby 'daily_output' widziało 'daily_output_base')
legacy_dir = os.path.join(project_root, 'current_development')

sys.path.append(project_root)
sys.path.append(legacy_dir) # <--- To eliminuje ModuleNotFoundError!

# Importy z naszych nowych, czystych modułów:
from moj_system.core.optimizer import walk_forward
from moj_system.core.indicators import compute_metrics, compute_buy_and_hold

# Importy ze starego kodu (będziemy je sukcesywnie przenosić w kolejnych krokach):
from current_development.strategy_test_library import load_stooq_local, get_n_jobs, analyze_trades, print_backtest_report
from current_development.daily_output import build_daily_outputs
from current_development.stooq_hybrid_updater import run_update

# --- PARAMETRY STRATEGII (Siatki poszukiwań) ---
POSITION_MODE = "full"
VOL_WINDOW    = 20
X_GRID        =[0.08, 0.10, 0.12, 0.15, 0.20]
Y_GRID        =[0.02, 0.03, 0.05, 0.07, 0.10]
FAST_GRID     =[50, 75, 100]
SLOW_GRID     =[150, 200, 250]
TV_GRID       =[0.08, 0.10, 0.12, 0.15, 0.20]
SL_GRID       =[0.05, 0.08, 0.10, 0.15]
MOM_LB_GRID   = [126, 252]

# --- PARAMETRY DLA ATR ---
ATR_WINDOW    = 20
N_ATR_GRID    = [0.08, 0.10, 0.12, 0.15, 0.20]

# --- REJESTR ASSETÓW ---
ASSET_REGISTRY = {
    "WIG20TR":   {"source": "stooq", "ticker": "wig20tr",  "train": 8, "test": 2, "default_stop": "fixed"},
    "MWIG40TR":  {"source": "stooq", "ticker": "mwig40tr", "train": 8, "test": 2, "default_stop": "fixed"},
    "SWIG80TR":  {"source": "stooq", "ticker": "swig80tr", "train": 9, "test": 2, "default_stop": "atr"},
    "SP500":     {"source": "stooq", "ticker": "^spx",     "train": 6, "test": 2, "default_stop": "fixed"},
    "NASDAQ100": {"source": "stooq", "ticker": "ndq100",   "train": 6, "test": 2, "default_stop": "fixed"},
    "MSCI_World":{"source": "drive", "ticker": "URTH",     "train": 9, "test": 1, "default_stop": "atr"},
    "STOXX600":  {"source": "drive", "ticker": "^STOXX",   "train": 7, "test": 1, "default_stop": "atr"},
}

# (Dalsza część kodu bez zmian, zaczynając od: def setup_logging...)

def setup_logging(output_prefix):
    log_file = f"outputs/{output_prefix}_DailyRun.log"
    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()]
    )

def main():
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument("--asset", type=str, required=True, help="Nazwa assetu, np. WIG20TR, SP500")
    parser.add_argument("--force_filter", type=str, default=None, help="Wymuś filtr np. 'ma' lub 'mom'")
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto", 
                        help="Rodzaj trailing stopa: 'fixed' (stały %%), 'atr' (Chandelier exit) lub 'auto' (zgodnie z rejestrem).")
    args = parser.parse_args()

    asset_name = args.asset
    if asset_name not in ASSET_REGISTRY:
        print(f"Błąd: Nieznany asset '{asset_name}'. Dostępne: {list(ASSET_REGISTRY.keys())}")
        sys.exit(1)

    cfg = ASSET_REGISTRY[asset_name]
    output_prefix = asset_name.lower()
    train_years = cfg["train"]
    test_years = cfg["test"]
    filter_mode = [args.force_filter] if args.force_filter else None
    
    # Wybór trybu Stop Loss
    selected_stop_mode = cfg["default_stop"] if args.stop_mode == "auto" else args.stop_mode
    use_atr_stop = (selected_stop_mode == "atr")

    setup_logging(output_prefix)
    logging.info("=" * 80)
    logging.info(f"DAILY RUNFILE START: {asset_name.upper()} | Train: {train_years}y | Test: {test_years}y")
    logging.info(f"Stop Mode: {'ATR-scaled (Chandelier)' if use_atr_stop else 'Fixed percentage'}")
    logging.info("=" * 80)

    # 1. Update Data (pobieranie z API/Drive)
    run_update(get_funds=False)

    # 2. Wczytanie danych lokalnych
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'current_development', 'data'))
    
    if cfg["source"] == "stooq":
        df = load_stooq_local(cfg["ticker"], asset_name, data_dir, "1990-01-01")
    else:
        logging.error("Pobieranie wprost z GDrive dla MSCI/STOXX skonfigurujemy po wydzieleniu modułu danych. "
                      "Tymczasowo runner obsługuje tylko indeksy Stooq (WIG20, SWIG80, SP500 itp.).")
        sys.exit(1)
        
    cash_df = load_stooq_local("fund_2720", "MMF", data_dir, "1990-01-01")

    if df is None or cash_df is None:
        logging.error("Nie udało się wczytać danych ze Stooq. Sprawdź folder data/.")
        sys.exit(1)

    # 3. Walk Forward (teraz z uwzględnieniem wybranego trybu ATR / Fixed)
    n_jobs = get_n_jobs()
    logging.info(f"Rozpoczynam Walk-Forward ({n_jobs} procesów)...")

    wf_equity, wf_results, wf_trades = walk_forward(
        df=df, 
        cash_df=cash_df, 
        train_years=train_years, 
        test_years=test_years,
        vol_window=VOL_WINDOW,
        selected_mode=POSITION_MODE,
        filter_modes_override=filter_mode,
        # --- Jawne przekazanie wszystkich siatek ---
        X_grid=X_GRID, 
        Y_grid=Y_GRID, 
        fast_grid=FAST_GRID, 
        slow_grid=SLOW_GRID,
        tv_grid=TV_GRID, 
        sl_grid=SL_GRID, 
        mom_lookback_grid=MOM_LB_GRID,
        objective="calmar",  # Dodajmy jawnie, żeby było pewne
        n_jobs=n_jobs, 
        fast_mode=True,
        # --- Parametry ATR (tak jak były) ---
        use_atr_stop=use_atr_stop,
        N_atr_grid=N_ATR_GRID if use_atr_stop else None,
        atr_window=ATR_WINDOW
        )

    if wf_equity.empty or wf_results.empty:
        logging.error("Walk-forward nie zwrócił wyników! (Puste DataFrame)")
        sys.exit(1)

    # 4. Generowanie wyników
    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    logging.info(f"Stitched OOS Metrics: CAGR={wf_metrics['CAGR']*100:.2f}% | "
                 f"Sharpe={wf_metrics['Sharpe']:.2f} | "
                 f"MaxDD={wf_metrics['MaxDD']*100:.2f}% | "
                 f"CalMAR={wf_metrics['CalMAR']:.2f}")

    # B&H do raportu
    oos_start = wf_results["TestStart"].min()
    oos_end   = wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, price_col="Zamkniecie", start=oos_start, end=oos_end)

    # Wywołanie istniejącego modułu Daily Output
    logging.info("Generowanie plików wyjściowych (Output)...")
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
        logfile_name=f"{output_prefix}_signal_log.csv"
    )

    logging.info("Gotowe! Pliki wygenerowane w folderze outputs/")

if __name__ == "__main__":
    main()