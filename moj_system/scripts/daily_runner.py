# -*- coding: utf-8 -*-
"""
moj_system/scripts/daily_runner.py
==================================
Universal Entry Point for all investment strategies.
"""

import argparse
import logging
import os
import sys
import tempfile

import matplotlib
import pandas as pd

matplotlib.use("Agg")

# --- Path Setup ---
# --- New, Clean Imports ---
from moj_system.config import ASSET_REGISTRY, BASE_GRIDS, BOND_GRIDS, OUTPUT_DIR
from moj_system.core.global_engine import (
    allocation_walk_forward_n,
    build_price_df_from_returns,
    build_return_series,
    print_global_equity_report,
)
from moj_system.core.pension_engine import (
    allocation_walk_forward,
    build_signal_series,
    build_standard_two_asset_data,
    print_multiasset_report,
)
from moj_system.core.research import (
    extract_flat_regime_stats,
    prepare_regime_inputs,
    print_live_regime_report,
    run_regime_decomposition,
)
from moj_system.core.strategy_engine import (
    analyze_trades,
    compute_buy_and_hold,
    compute_metrics,
    get_n_jobs,
    print_backtest_report,
    walk_forward,
)
from moj_system.core.utils import build_mmf_extended
from moj_system.data.builder import build_and_upload
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater
from moj_system.reporting.daily_output import build_daily_outputs
from moj_system.reporting.global_equity_daily_output import (
    build_daily_outputs as build_global_outputs,
)
from moj_system.reporting.multiasset_daily_output import (
    build_daily_outputs as build_multiasset_outputs,
)


def setup_logging(output_prefix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Użycie operatora / z biblioteki pathlib
    log_file = OUTPUT_DIR / f"{output_prefix}.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_single_asset(asset_name, stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY[asset_name]
    output_prefix = asset_name.lower()
    selected_stop_mode = (
        cfg.get("default_stop", "fixed") if stop_mode_arg == "auto" else stop_mode_arg
    )
    use_atr_stop = selected_stop_mode == "atr"
    logging.info(f"SINGLE ASSET ENGINE: {asset_name} | Stop Mode: {selected_stop_mode}")

    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR1M = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    cash_df = (
        build_mmf_extended(MMF, WIBOR1M, floor_date="1995-01-02") if WIBOR1M is not None else MMF
    )

    if cfg["source"] == "drive":
        is_msci = asset_name == "MSCI_World"
        df = build_and_upload(
            os.environ.get("GDRIVE_FOLDER_ID"),
            f"{output_prefix}_wsj_raw.csv" if is_msci else f"{output_prefix}.csv",
            f"{output_prefix}_combined.csv",
            cfg["ticker"],
            "yfinance",
            creds_path,
            is_msci_world=is_msci,
        )
    else:
        df = load_local_csv(output_prefix, asset_name)
    if df is None:
        sys.exit(f"Failed to load data for {asset_name}")

    grids = BASE_GRIDS.copy()
    if "grids" in cfg:
        grids.update(cfg["grids"])

    wf_equity, wf_results, wf_trades = walk_forward(
        df=df,
        cash_df=cash_df,
        train_years=cfg["train"],
        test_years=cfg["test"],
        X_grid=grids["X_GRID"],
        Y_grid=grids["Y_GRID"],
        fast_grid=grids["FAST_GRID"],
        slow_grid=grids["SLOW_GRID"],
        tv_grid=grids["TV_GRID"],
        sl_grid=grids["SL_GRID"],
        mom_lookback_grid=grids["MOM_LB_GRID"],
        objective="calmar",
        n_jobs=get_n_jobs(),
        fast_mode=True,
        use_atr_stop=use_atr_stop,
        N_atr_grid=grids["N_ATR_GRID"] if use_atr_stop else None,
        atr_window=grids["ATR_WINDOW"],
    )
    if wf_equity.empty:
        sys.exit(f"Walk-forward returned no results for {asset_name}.")

    wf_equity = wf_equity.loc[~wf_equity.index.duplicated(keep="last")]

    wf_metrics = {k: float(v) for k, v in compute_metrics(wf_equity).items()}
    trade_stats = analyze_trades(wf_trades)

    print_backtest_report(
        metrics=wf_metrics,
        trades=wf_trades,
        trade_stats=trade_stats,
        wf_results=wf_results,
        position_mode="full",
        filter_modes_override=None,
    )

    oos_start, oos_end = wf_results["TestStart"].min(), wf_results["TestEnd"].max()
    bh_equity, bh_metrics = compute_buy_and_hold(df, "Zamkniecie", oos_start, oos_end)

    # Raportowanie reżimów
    regime_inputs = prepare_regime_inputs(df, wf_results, wf_equity, bh_equity)

    if regime_inputs:  # <--- DODAJ TO SPRAWDZENIE
        raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
        regime_metrics = extract_flat_regime_stats(raw_regimes)
        print_live_regime_report(regime_metrics)
    else:
        logging.warning(
            f"Skipping regime analysis for {asset_name} due to insufficient data overlap.",
        )
        regime_metrics = extract_flat_regime_stats({})  # Zwróci słownik z NaN

    # Wysyłanie (wewnętrznie wywołuje upload na GDrive)
    build_daily_outputs(
        wf_equity=wf_equity,
        wf_trades=wf_trades,
        wf_metrics=wf_metrics,
        wf_results=wf_results,
        bh_equity=bh_equity,
        bh_metrics=bh_metrics,
        df=df,
        output_dir=str(OUTPUT_DIR / output_prefix),
        asset_name=asset_name,
        price_col="Zamkniecie",
        logfile_name=f"{output_prefix}_signal_log.csv",
        gdrive_folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        gdrive_credentials=creds_path if os.path.exists(creds_path) else None,
    )

def run_pension_portfolio(stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY["PENSION"]
    selected_stop = cfg.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_eq = selected_stop == "atr"
    logging.info(f"PENSION PORTFOLIO ENGINE (WIG+TBSP+MMF) | WIG Stop: {selected_stop} | Train years: {cfg['train']} | Test years: {cfg['test']}")

    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    MMF = load_local_csv("fund_2720", "MMF")
    WIBOR = load_local_csv("wibor1m", "WIBOR1M", mandatory=False)
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    TBSP = build_and_upload(
        folder_id,
        "tbsp_extended_full.csv",
        "tbsp_extended_combined.csv",
        "^tbsp",
        "stooq",
        creds_path,
    )
    PL10Y, DE10Y = load_local_csv("pl10y", "PL10Y"), load_local_csv("de10y", "DE10Y")

    derived = build_standard_two_asset_data(WIG, TBSP, MMF, WIBOR, PL10Y, DE10Y, "1995-01-02")
    n_jobs = get_n_jobs()

    logging.info("================ WIG ================")
    # --- POPRAWKA: Jawne przekazanie BASE_GRIDS oraz fast_mode=True ---
    wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
        df=WIG, 
        cash_df=derived["mmf_ext"], 
        train_years=cfg["train"], 
        test_years=cfg["test"], 
        X_grid=BASE_GRIDS["X_GRID"], 
        Y_grid=BASE_GRIDS["Y_GRID"],
        fast_grid=BASE_GRIDS["FAST_GRID"], 
        slow_grid=BASE_GRIDS["SLOW_GRID"],
        tv_grid=BASE_GRIDS.get("TV_GRID"),
        sl_grid=BASE_GRIDS.get("SL_GRID"),
        mom_lookback_grid=BASE_GRIDS.get("MOM_LB_GRID"),
        use_atr_stop=use_atr_eq, 
        N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None,
        atr_window=BASE_GRIDS.get("ATR_WINDOW", 20),
        n_jobs=n_jobs,
        fast_mode=True
    )
    
    logging.info("================ TBSP ================")
    # --- POPRAWKA: Jawne przekazanie BOND_GRIDS oraz fast_mode=True ---
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
        df=TBSP,
        cash_df=derived["mmf_ext"],
        train_years=cfg["train"],
        test_years=cfg["test"],
        filter_modes_override=["ma"],
        X_grid=BOND_GRIDS["X_GRID"], 
        Y_grid=BOND_GRIDS["Y_GRID"],
        fast_grid=BOND_GRIDS["FAST_GRID"], 
        slow_grid=BOND_GRIDS["SLOW_GRID"],
        sl_grid=BOND_GRIDS.get("SL_GRID"),
        use_atr_stop=False, # Obligacje testowaliśmy zawsze na Fixed Stop
        n_jobs=n_jobs,
        entry_gate_series=derived["bond_gate"],
        fast_mode=True
    )

    sig_eq, sig_bd = build_signal_series(wf_eq, wf_tr_eq), build_signal_series(wf_bd, wf_tr_bd)
    oos_s, oos_e = (
        max(wf_res_eq["TestStart"].min(), wf_res_bd["TestStart"].min()),
        min(wf_res_eq["TestEnd"].max(), wf_res_bd["TestEnd"].max()),
    )
    sig_eq_oos, sig_bd_oos = sig_eq.loc[oos_s:oos_e], sig_bd.loc[oos_s:oos_e]

    port_eq, w_s, realloc, alloc_df = allocation_walk_forward(
        derived["ret_eq"],
        derived["ret_bd"],
        derived["ret_mmf"],
        sig_eq,
        sig_bd,
        sig_eq_oos,
        sig_bd_oos,
        wf_res_eq,
        wf_res_bd,
    )
    # --- PANCERNE USUWANIE DUPLIKATÓW DAT ---
    port_eq = port_eq.loc[~port_eq.index.duplicated(keep="last")]
    m_p = compute_metrics(port_eq)
    bh_eq, bh_m_eq = compute_buy_and_hold(WIG, "Zamkniecie", oos_s, oos_e)
    bh_bd, bh_m_bd = compute_buy_and_hold(TBSP, "Zamkniecie", oos_s, oos_e)

    print_multiasset_report(
        m_p, bh_m_eq, bh_m_bd, alloc_df, realloc, sig_eq_oos, sig_bd_oos, oos_s, oos_e, sig_bd,
    )

    # Raportowanie reżimów
    regime_inputs = prepare_regime_inputs(WIG, wf_res_eq, port_eq, bh_eq)
    if regime_inputs:
        raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
        regime_metrics = extract_flat_regime_stats(raw_regimes)
        print_live_regime_report(regime_metrics)
    else:
        regime_metrics = extract_flat_regime_stats({})

    # Wysyłanie (wewnętrznie wywołuje upload na GDrive)
    build_multiasset_outputs(
        wf_equity_eq=wf_eq,
        wf_trades_eq=wf_tr_eq,
        wf_results_eq=wf_res_eq,
        wf_equity_bd=wf_bd,
        wf_trades_bd=wf_tr_bd,
        wf_results_bd=wf_res_bd,
        portfolio_equity=port_eq,
        portfolio_metrics=m_p,
        weights_series=w_s,
        reallocation_log=realloc,
        bh_eq_equity=bh_eq,
        bh_eq_metrics=bh_m_eq,
        bh_bd_equity=bh_bd,
        bh_bd_metrics=bh_m_bd,
        WIG=WIG,
        TBSP=TBSP,
        sig_eq_oos=sig_eq_oos,
        sig_bd_oos=sig_bd_oos,
        output_dir=str(OUTPUT_DIR / "pension"),
        asset_name="PENSION",  # [Ważne]
        run_date=None,
        gdrive_folder_id=folder_id,
        gdrive_credentials=creds_path,
    )

def run_global_portfolio(asset_key, stop_mode_arg, creds_path):
    cfg = ASSET_REGISTRY[asset_key]
    mode, train_y, test_y, fx_h = cfg["mode"], cfg["train"], cfg["test"], cfg.get("fx_hedged", True)
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    
    #[POPRAWKA 1] Ustalenie trybu stopu (podobnie jak w PENSION)
    selected_stop = cfg.get("default_stop_eq", "atr") if stop_mode_arg == "auto" else stop_mode_arg
    use_atr_eq = (selected_stop == "atr")
    
    # [POPRAWKA 2] Naprawione cudzysłowy w f-stringu, aby uniknąć SyntaxError
    logging.info(f"GLOBAL PORTFOLIO ENGINE: {mode} | FX Hedged: {fx_h} | Stop mode: {selected_stop} | Train years: {train_y} | Test years: {test_y}")

    WIG = load_local_csv("wig", "WIG").loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]
    TBSP = build_and_upload(
        folder_id,
        "tbsp_extended_full.csv",
        "tbsp_extended_combined.csv",
        "^tbsp",
        "stooq",
        creds_path,
    )
    MMF = load_local_csv("fund_2720", "MMF")

    fx_map = {
        curr: load_local_csv(f"{curr.lower()}pln", f"{curr}PLN")["Zamkniecie"]
        for curr in ["USD", "EUR", "JPY"]
    }

    if mode == "global_equity":
        stoxx = build_and_upload(
            folder_id, "stoxx600.csv", "stoxx600_combined.csv", "^STOXX", "yfinance", creds_path,
        )
        assets = {
            "WIG": (WIG, None),
            "SP500": (load_local_csv("sp500", "SP500"), fx_map["USD"]),
            "STOXX600": (stoxx, fx_map["EUR"]),
            "Nikkei225": (load_local_csv("nikkei225", "Nikkei225"), fx_map["JPY"]),
        }
    else:  # msci_world
        msciw = build_and_upload(
            folder_id,
            "msci_world_wsj_raw.csv",
            "msci_world_combined.csv",
            "URTH",
            "yfinance",
            creds_path,
            is_msci_world=True,
        )
        assets = {"WIG": (WIG, None), "MSCI_World": (msciw, fx_map["USD"])}

    rets_dict, sigs_full, bh_metrics_all = {}, {}, {}
    n_jobs = get_n_jobs()

    wig_wf_res = None
    for lbl, (px_df, fx_s) in assets.items():
        ret_s = build_return_series(px_df, fx_series=fx_s, hedged=fx_h)
        rets_dict[lbl] = ret_s.dropna()
        proc_px = px_df if fx_h or fx_s is None else build_price_df_from_returns(ret_s, lbl)
        logging.info(f"================ {lbl} ================")
        #[POPRAWKA 3] Jawne przekazanie parametrów stopu, siatek i fast_mode
        wf_e, wf_r, wf_t = walk_forward(
            proc_px, MMF, train_y, test_y, 
            X_grid=BASE_GRIDS["X_GRID"], Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"], slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr_eq, 
            N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None,
            n_jobs=n_jobs, fast_mode=True
        )
        
        sigs_full[lbl] = build_signal_series(wf_e, wf_t)
        if lbl == "WIG":
            wig_wf_res = wf_r
    logging.info("================ TBSP ================")
    # [POPRAWKA 4] Dodanie odpowiednich siatek również dla obligacji
    wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
        TBSP, MMF, train_y, test_y, filter_modes_override=["ma"], 
        X_grid=BOND_GRIDS["X_GRID"], Y_grid=BOND_GRIDS["Y_GRID"],
        fast_grid=BOND_GRIDS["FAST_GRID"], slow_grid=BOND_GRIDS["SLOW_GRID"],
        n_jobs=n_jobs, fast_mode=True
    )
    rets_dict["TBSP"] = TBSP["Zamkniecie"].pct_change().dropna()
    sigs_full["TBSP"] = build_signal_series(wf_bd, wf_tr_bd)

    p_e, w_s, realloc, a_df = allocation_walk_forward_n(
        rets_dict,
        sigs_full,
        sigs_full,
        MMF["Zamkniecie"].pct_change().dropna(),
        wf_res_bd,
        list(rets_dict.keys()),
        train_years=train_y,
    )

    # --- PANCERNE USUWANIE DUPLIKATÓW DAT ---
    p_e = p_e.loc[~p_e.index.duplicated(keep="last")]

    m = {k: float(v) for k, v in compute_metrics(p_e).items()}
    for lbl in rets_dict.keys():
        aligned_rets = rets_dict[lbl].reindex(p_e.index).fillna(0.0)
        bh_curve = (1 + aligned_rets).cumprod()
        bh_metrics_all[lbl] = compute_metrics(bh_curve)

    print_global_equity_report(
        m, bh_metrics_all, a_df, realloc, sigs_full, p_e.index.min(), p_e.index.max(), mode, fx_h,
    )

    # Raportowanie reżimów (Z użyciem WIG jako rynkowego kontekstu odniesienia)
    bh_wig = (1 + rets_dict["WIG"].reindex(p_e.index).fillna(0.0)).cumprod()
    regime_inputs = prepare_regime_inputs(WIG, wig_wf_res, p_e, bh_wig)
    raw_regimes = run_regime_decomposition(regime_inputs, generate_plots=False)
    regime_metrics = extract_flat_regime_stats(raw_regimes)
    print_live_regime_report(regime_metrics)

    # Wysyłanie (wewnętrznie wywołuje upload na GDrive)
    build_global_outputs(
        wf_results_dict={},
        portfolio_equity=p_e,
        portfolio_metrics=m,
        weights_series=w_s,
        reallocation_log=realloc,
        bh_metrics_dict=bh_metrics_all,
        returns_dict=rets_dict,
        signals_oos_dict=sigs_full,
        asset_keys=list(rets_dict.keys()),
        portfolio_mode=mode,
        fx_hedged=fx_h,
        output_dir=str(OUTPUT_DIR / asset_key.lower()),
        asset_name=asset_key,  # [Ważne] - czyli np. GLOBAL_A lub GLOBAL_B
        run_date=None,
        gdrive_folder_id=folder_id,
        gdrive_credentials=creds_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument(
        "--asset", type=str, required=True, help="Asset key: WIG20TR, PENSION, GLOBAL_A, etc.",
    )
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto")
    args = parser.parse_args()

    cfg = ASSET_REGISTRY.get(args.asset)
    if not cfg:
        sys.exit(f"Error: Unknown asset '{args.asset}'.")
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
