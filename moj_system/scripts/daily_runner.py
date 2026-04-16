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

# --- IMPORTY Z ORYGINALNEJ BIBLIOTEKI ---
from current_development.strategy_test_library import (
    load_stooq_local, get_n_jobs, walk_forward, compute_metrics, 
    analyze_trades, compute_buy_and_hold, print_backtest_report
)
from current_development.multiasset_library import (
    build_signal_series, allocation_walk_forward, print_multiasset_report,
    build_standard_two_asset_data
)
# --- NOWE, CZYSTE IMPORTY DANYCH ---
from moj_system.data.updater import DataUpdater
from moj_system.data.data_manager import load_local_csv
from moj_system.data.builder import build_and_upload

# --- IMPORTY Z NOWEJ STRUKTURY RAPORTOWANIA ---
from moj_system.reporting.daily_output import build_daily_outputs
from moj_system.reporting.multiasset_daily_output import build_daily_outputs as build_multiasset_outputs

# --- KONFIGURACJA BAZOWYCH SIATEK ---
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

# Konfiguracja dla obligacji (TBSP) w trybie Multi-Asset
BOND_GRIDS = {
    "X_GRID": [0.05, 0.08, 0.10, 0.15],
    "Y_GRID": [0.001],
    "FAST_GRID": [50, 100, 150],
    "SLOW_GRID": [200, 300, 400, 500],
    "TV_GRID": [0.08, 0.10, 0.12],
    "SL_GRID": [0.01, 0.02, 0.03],
    "MOM_LB_GRID": [252],
    "N_ATR_GRID": [0.05, 0.08, 0.10, 0.15]
}

# --- REJESTR ASSETÓW ---
# --- REJESTR ASSETÓW Z PEŁNĄ KONFIGURACJĄ (Stooq + yFinance + GDrive) ---
ASSET_REGISTRY = {
    # Indeksy polskie (głównie Stooq)
    "WIG20TR":  {
        "type": "single", "source": "stooq", "ticker": "wig20tr", "yf_ticker": "WIG20TR.WA", 
        "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}
    },
    "MWIG40TR": {
        "type": "single", "source": "stooq", "ticker": "mwig40tr", "yf_ticker": "MWIG40TR.WA", 
        "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}
    },
    "SWIG80TR": {
        "type": "single", "source": "stooq", "ticker": "swig80tr", "yf_ticker": "SWIG80TR.WA", 
        "train": 9, "test": 2, "default_stop": "atr"
    },
    # Indeksy globalne (Stooq + yFinance)
    "SP500":    {
        "type": "single", "source": "stooq", "ticker": "^spx", "yf_ticker": "^GSPC", 
        "train": 6, "test": 2, "default_stop": "fixed"
    },
    "NASDAQ100":{
        "type": "single", "source": "stooq", "ticker": "ndq100", "yf_ticker": "^NDX", 
        "train": 6, "test": 2, "default_stop": "fixed"
    },
    "Nikkei225":{
        "type": "single", "source": "stooq", "ticker": "nk225", "yf_ticker": "^N225", 
        "train": 6, "test": 2, "default_stop": "fixed"
    },
    # Aktywa oparte o bazę na Google Drive + przedłużenie yFinance
    "MSCI_World":{
        "type": "single", "source": "drive", "ticker": "URTH", "yf_ticker": "URTH", 
        "train": 9, "test": 1, "default_stop": "atr"
    },
    "STOXX600": {
        "type": "single", "source": "drive", "ticker": "^STOXX", "yf_ticker": "^STOXX", 
        "train": 7, "test": 1, "default_stop": "atr"
    },
    # --- Tryb Portfelowy ---
    "MULTIASSET": {
        "type": "portfolio", "train_eq": 7, "test_eq": 2, "train_bd": 7, "test_bd": 2, 
        "default_stop_eq": "atr"
    } 
}


def setup_logging(output_prefix):
    log_file = f"outputs/{output_prefix}_DailyRun.log"
    logging.info(f"KATALOG ROBOCZY (CWD): {os.getcwd()}")
    os.makedirs("outputs", exist_ok=True)
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding='utf-8'), logging.StreamHandler(sys.stdout)])


def run_multiasset(stop_mode_arg):
    """Główna logika portfela Multi-Asset (WIG + TBSP + MMF)"""
    cfg = ASSET_REGISTRY["MULTIASSET"]
    
    # Wybór trybu stopu dla WIG w ramach portfela
    selected_stop_mode = cfg.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_stop_eq = (selected_stop_mode == "atr")
    
    logging.info(f"Stop Mode dla portfela (WIG): {'ATR-scaled' if use_atr_stop_eq else 'Fixed percentage'}")

    # 1. Update Danych przez DataUpdater (bez KNF)
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(ASSET_REGISTRY, get_funds=False)

    # 2. Ładowanie i budowanie złożonych Assetów
    logging.info("Ładowanie danych...")
    WIG = load_local_csv("wig", "WIG", "1990-01-01")
    WIG = WIG.loc[WIG.index >= pd.Timestamp("1995-01-02")]
    MMF = load_local_csv("fund_2720", "MMF", "1990-01-01")
    W1M = load_local_csv("wibor1m", "WIBOR1M", "1990-01-01", mandatory=False)
    PL10Y = load_local_csv("PL10Y", "PL10Y", "1990-01-01")
    DE10Y = load_local_csv("DE10Y", "DE10Y", "1990-01-01")
    
    # Budowa rozszerzonego TBSP przy pomocy nowego Buildera!
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    TBSP = build_and_upload(
        folder_id=folder_id,
        raw_filename="tbsp_extended_full.csv",
        combined_filename="tbsp_extended_combined.csv",
        extension_ticker="^tbsp",
        extension_source="stooq",
        credentials_path=creds_path
    )
    if TBSP is None:
        TBSP = load_local_csv("tbsp", "TBSP","1990-01-01")
        if TBSP is None:
            logging.error("FAIL: TBSP build/update failed.")
            sys.exit(1)
            
    logging.info("OK  %-14s  %5d rows  %s to %s",
                 "TBSP", len(TBSP), TBSP.index.min().date(), TBSP.index.max().date())
    for col in ["Najwyzszy", "Najnizszy"]:
        if col not in TBSP.columns: TBSP[col] = TBSP["Zamkniecie"]

    # 2. Serie pochodne (WIBOR extension, bramki, spread)
    derived = build_standard_two_asset_data(
        wig=WIG, tbsp=TBSP, mmf=MMF, wibor1m=W1M, pl10y=PL10Y, de10y=DE10Y,
        mmf_floor="1995-01-02", yield_prefilter_bp=50.0
    )
    
    n_jobs = get_n_jobs()

    # 3. Walk-Forward: Akcje (WIG)
    logging.info("=" * 80)
    logging.info(f"PHASE 3: EQUITY SIGNAL WALK-FORWARD (WIG, train={cfg['train_eq']}y test={cfg['test_eq']}y)")
    
    wf_equity_eq, wf_results_eq, wf_trades_eq = walk_forward(
        df=WIG, cash_df=derived["mmf_ext"], train_years=cfg["train_eq"], test_years=cfg["test_eq"],
        vol_window=20, selected_mode="full", filter_modes_override=None,
        X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"], fast_grid=BASE_GRIDS["FAST_GRID"], 
        slow_grid=BASE_GRIDS["SLOW_GRID"], tv_grid=BASE_GRIDS["TV_GRID"], sl_grid=BASE_GRIDS["SL_GRID"], 
        mom_lookback_grid=BASE_GRIDS["MOM_LB_GRID"],
        objective="calmar", n_jobs=n_jobs, fast_mode=True,
        use_atr_stop=use_atr_stop_eq, N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_stop_eq else None, 
        atr_window=BASE_GRIDS["ATR_WINDOW"]
    )
    sig_eq = build_signal_series(wf_equity_eq, wf_trades_eq)

    # 4. Walk-Forward: Obligacje (TBSP)
    logging.info("=" * 80)
    logging.info(f"PHASE 4: BOND SIGNAL WALK-FORWARD (TBSP, train={cfg['train_bd']}y test={cfg['test_bd']}y)")

    wf_equity_bd, wf_results_bd, wf_trades_bd = walk_forward(
        df=TBSP, cash_df=derived["mmf_ext"], train_years=cfg["train_bd"], test_years=cfg["test_bd"],
        vol_window=20, selected_mode="full", filter_modes_override=["ma"],
        X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"], fast_grid=BOND_GRIDS["FAST_GRID"], 
        slow_grid=BOND_GRIDS["SLOW_GRID"], tv_grid=BOND_GRIDS["TV_GRID"], sl_grid=BOND_GRIDS["SL_GRID"], 
        mom_lookback_grid=BOND_GRIDS["MOM_LB_GRID"],
        objective="calmar", n_jobs=n_jobs, fast_mode=True, entry_gate_series=derived["bond_gate"],
        use_atr_stop=False # TBSP zawsze na Fixed
    )
    sig_bd = build_signal_series(wf_equity_bd, wf_trades_bd)

    # 5. Allocation Walk-Forward
    logging.info("=" * 80)
    logging.info("PHASE 5: ALLOCATION WALK-FORWARD")
    
    oos_start = max(wf_results_eq["TestStart"].min(), wf_results_bd["TestStart"].min())
    oos_end   = min(wf_results_eq["TestEnd"].max(),   wf_results_bd["TestEnd"].max())
    
    def _trim(s, a, b): return s.loc[(s.index >= a) & (s.index <= b)]
    sig_eq_oos = _trim(sig_eq, oos_start, oos_end)
    sig_bd_oos = _trim(sig_bd, oos_start, oos_end)

    portfolio_equity, weights_series, realloc_log, alloc_df = allocation_walk_forward(
        equity_returns=derived["ret_eq"], bond_returns=derived["ret_bd"], mmf_returns=derived["ret_mmf"],
        sig_equity_full=sig_eq, sig_bond_full=sig_bd, sig_equity_oos=sig_eq_oos, sig_bond_oos=sig_bd_oos,
        wf_results_eq=wf_results_eq, wf_results_bd=wf_results_bd,
        step=0.10, objective="calmar", cooldown_days=10, annual_cap=12
    )

    port_metrics = {k: float(v) for k, v in compute_metrics(portfolio_equity).items()}
    bh_eq, bh_m_eq = compute_buy_and_hold(WIG, price_col="Zamkniecie", start=oos_start, end=oos_end)
    bh_bd, bh_m_bd = compute_buy_and_hold(TBSP, price_col="Zamkniecie", start=oos_start, end=oos_end)

    print_multiasset_report(
        portfolio_metrics=port_metrics, bh_equity_metrics=bh_m_eq, bh_bond_metrics=bh_m_bd,
        alloc_results_df=alloc_df, reallocation_log=realloc_log, sig_equity=sig_eq_oos, sig_bond=sig_bd_oos,
        oos_start=oos_start, oos_end=oos_end, sig_bond_raw=sig_bd, 
        spread_prefilter=derived["spread_pf"], yield_prefilter=derived["yield_pf"], sig_bond_post_spread=sig_bd
    )

    # 6. Generowanie plików wyjściowych z wykorzystaniem GDriveClient
    logging.info("=" * 80)
    logging.info("PHASE 6: DAILY OUTPUT")
    logging.info("=" * 80)
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    
    build_multiasset_outputs(
        wf_equity_eq=wf_equity_eq, wf_trades_eq=wf_trades_eq, wf_results_eq=wf_results_eq,
        wf_equity_bd=wf_equity_bd, wf_trades_bd=wf_trades_bd, wf_results_bd=wf_results_bd,
        portfolio_equity=portfolio_equity, portfolio_metrics=port_metrics,
        weights_series=weights_series, reallocation_log=realloc_log,
        bh_eq_equity=bh_eq, bh_eq_metrics=bh_m_eq, bh_bd_equity=bh_bd, bh_bd_metrics=bh_m_bd,
        WIG=WIG, TBSP=TBSP, sig_eq_oos=sig_eq_oos, sig_bd_oos=sig_bd_oos,
        output_dir="outputs/multiasset", 
        gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None
    )


def main():
    
    # Pobierz ścieżkę do folderu, w którym jest ten skrypt i cofnij się o 2 poziomy do projektu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_home = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Zmień katalog roboczy na główny folder projektu
    os.chdir(project_home)

    # Teraz wszystko co zapiszesz w "outputs/" trafi do F:\.../python-analysis/outputs/
    
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument("--asset", type=str, required=True, help="Nazwa assetu, np. WIG20TR, MULTIASSET, SP500")
    parser.add_argument("--force_filter", type=str, default=None, help="Wymuś filtr np. 'ma' lub 'mom' (tylko Single-Asset)")
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto", 
                        help="Rodzaj trailing stopa: 'fixed', 'atr' lub 'auto'.")
    args = parser.parse_args()

    asset_name = args.asset
    if asset_name not in ASSET_REGISTRY:
        sys.exit(f"Błąd: Nieznany asset '{asset_name}'. Dostępne: {list(ASSET_REGISTRY.keys())}")

    cfg = ASSET_REGISTRY[asset_name]
    output_prefix = asset_name.lower()
    setup_logging(output_prefix)

    # 1. Update Data (Hybrydowy system z moj_system/data)
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(ASSET_REGISTRY, get_funds=False)

    # Nowy folder danych
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_csv'))

    # BRAMKA: PORTFEL (Multi-Asset) vs POJEDYNCZY ASSET
    if cfg.get("type") == "portfolio":
        run_multiasset(args.stop_mode)
        logging.info("Zakończono wykonywanie portfela MULTIASSET.")
        sys.exit(0)

    # --- LOGIKA DLA POJEDYNCZYCH ASSETÓW (Single-Asset) ---
    train_years, test_years = cfg["train"], cfg["test"]
    filter_mode = [args.force_filter] if args.force_filter else None
    selected_stop_mode = cfg.get("default_stop", "fixed") if args.stop_mode == "auto" else args.stop_mode
    use_atr_stop = (selected_stop_mode == "atr")

    logging.info("=" * 80)
    logging.info(f"DAILY RUNFILE START: {asset_name.upper()} | Train: {train_years}y | Test: {test_years}y")
    logging.info(f"Stop Mode: {'ATR-scaled (Chandelier)' if use_atr_stop else 'Fixed percentage'}")
    logging.info("=" * 80)
    
    # 2. Ładowanie
    # Używamy ujednoliconych nazw, dokładnie takich jakie zapisał DataUpdater
    MMF = load_local_csv("fund_2720", "MMF", "1990-01-01")
    WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", "1990-01-01", mandatory=False)
    
    if WIBOR1M is not None:
        from current_development.global_equity_library import build_mmf_extended
        cash_df = build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02")
    else:
        cash_df = MMF
        
    # Wczytanie konkretnego assetu
    if cfg["source"] == "drive":
        from moj_system.data.builder import build_and_upload
        
        is_msci = (asset_name == "MSCI_World") # <-- Sprawdzamy, czy to nasz specjalny przypadek

        df = build_and_upload(
            folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
            raw_filename=f"{asset_name.lower()}_wsj_raw.csv" if is_msci else f"{asset_name.lower()}.csv",
            combined_filename=f"{asset_name.lower()}_combined.csv",
            extension_ticker=cfg["ticker"],
            extension_source="yfinance",
            credentials_path=creds_path,
            is_msci_world=is_msci # <-- Włączamy specjalną logikę
        )
    else:
        df = load_local_csv(asset_name.lower(), asset_name, "1990-01-01")

    if df is None or cash_df is None: sys.exit("Błąd wczytywania danych.")

    grids = BASE_GRIDS.copy()
    if "grids" in cfg: grids.update(cfg["grids"])
    
    n_jobs = get_n_jobs()
    
    wf_equity, wf_results, wf_trades = walk_forward(
        df=df, cash_df=cash_df, train_years=train_years, test_years=test_years,
        vol_window=20, selected_mode="full", filter_modes_override=filter_mode,
        X_grid=grids["X_GRID"], Y_grid=grids["Y_GRID"], fast_grid=grids["FAST_GRID"], 
        slow_grid=grids["SLOW_GRID"], tv_grid=grids["TV_GRID"], sl_grid=grids["SL_GRID"], 
        mom_lookback_grid=grids["MOM_LB_GRID"], objective="calmar", n_jobs=n_jobs, fast_mode=True,
        use_atr_stop=use_atr_stop, N_atr_grid=grids["N_ATR_GRID"] if use_atr_stop else None,
        atr_window=grids["ATR_WINDOW"]
    )

    if wf_equity.empty: sys.exit("Walk-forward nie zwrócił wyników!")

    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    trade_stats = analyze_trades(wf_trades, boundary_exits={"CARRY", "SAMPLE_END"})
    print_backtest_report(metrics=wf_metrics, trades=wf_trades, trade_stats=trade_stats,
                          wf_results=wf_results, position_mode="full", filter_modes_override=filter_mode)
    
    oos_start, oos_end = wf_results["TestStart"].min(), wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, price_col="Zamkniecie", start=oos_start, end=oos_end)

    logging.info("=" * 80)
    logging.info("PHASE 6: DAILY OUTPUT")
    logging.info("=" * 80)
    
    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    build_daily_outputs(
        wf_equity=wf_equity, wf_trades=wf_trades, wf_metrics=wf_metrics,
        wf_results=wf_results, bh_equity=bh_equity, bh_metrics=bh_metrics, df=df,
        output_dir=f"outputs/{output_prefix}", asset_name=asset_name,
        logfile_name=f"{output_prefix}_signal_log.csv",
        gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None
    )

if __name__ == "__main__":
    main()