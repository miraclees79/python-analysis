# -*- coding: utf-8 -*-
"""
moj_system/scripts/sharded_robustness.py
========================================
Distributed Deep Validation Engine.
Designed specifically for GitHub Actions sharding.
Modes:
  - 'worker': Runs Walk-Forward, generates MC and Boot raw data (with unique seeds), saves to CSV.
  - 'merge' : Re-runs WF, loads CSVs from all shards, concatenates, evaluates verdicts, runs allocation perturbations.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")

from moj_system.config import (
    ASSET_REGISTRY,
    BASE_GRIDS,
    BOND_GRIDS,
    BOND_THRESHOLDS_BOOTSTRAP,
    BOND_THRESHOLDS_MC,
    EQUITY_THRESHOLDS_BOOTSTRAP,
    EQUITY_THRESHOLDS_MC,
    OUTPUT_DIR,
)
from moj_system.core.global_engine import (
    allocation_walk_forward_n,
    allocation_weight_robustness_n,
    build_price_df_from_returns,
    build_return_series,
    print_allocation_robustness_report_n,
)
from moj_system.core.pension_engine import (
    allocation_walk_forward,
    allocation_weight_robustness,
    build_signal_series,
    build_standard_two_asset_data,
    print_allocation_robustness_report,
)
from moj_system.core.robustness_engine import (
    analyze_bootstrap,
    analyze_robustness,
    block_bootstrap_history,
    extract_best_params_from_wf_results,
    extract_windows_from_wf_results,
    run_monte_carlo_robustness,
)
from moj_system.core.strategy_engine import (
    compute_buy_and_hold,
    compute_metrics,
    get_n_jobs,
    walk_forward,
)
from moj_system.core.utils import build_mmf_extended
from moj_system.data.builder import build_and_upload
from moj_system.data.data_manager import load_local_csv
from moj_system.data.updater import DataUpdater

# =========================================================================
# CUSTOM SHARDED BOOTSTRAP LOGIC
# =========================================================================

def _sharded_bootstrap_single_sample(
    iteration:      int,
    base_seed:      int,
    combined:       pd.DataFrame,
    df:             pd.DataFrame,
    cash_df:        pd.DataFrame,
    price_col:      str,
    cash_price_col: str,
    block_size:     int,
    wf_kwargs:      dict,
) -> dict | None:

    seed = base_seed + iteration
    wf_kwargs_inner = {**wf_kwargs, "n_jobs": 1}

    try:
        synthetic = block_bootstrap_history(
            df=combined,
            price_col=price_col,
            cash_col="cash_price",
            block_size=block_size,
            seed=seed,
        )

        n_synth = len(synthetic)
        synthetic_df = combined.iloc[:n_synth, [combined.columns.get_loc(price_col)]].copy()
        synthetic_df[price_col] = synthetic[price_col].values

        synthetic_cash = combined.iloc[:n_synth][["cash_price"]].copy()
        synthetic_cash = synthetic_cash.rename(columns={"cash_price": cash_price_col})
        synthetic_cash[cash_price_col] = synthetic["cash_price"].values

        equity, wf_res, _ = walk_forward(
            df=synthetic_df,
            cash_df=synthetic_cash,
            **wf_kwargs_inner,
        )

        if equity is None or equity.empty:
            return None

        m = compute_metrics(
            equity=equity,
            risk_free_rate=0.0,
        )
        return {
            "sample": seed,
            "CAGR": m["CAGR"],
            "Sharpe": m["Sharpe"],
            "MaxDD": m["MaxDD"],
            "CalMAR": m["CalMAR"],
            "Sortino": m.get("Sortino", np.nan),
        }
    except Exception as e:
        logging.warning(
            msg=f"Sharded Bootstrap sample {seed} failed: {e}",
        )
        return None

def run_sharded_block_bootstrap(
    df:             pd.DataFrame,
    cash_df:        pd.DataFrame,
    base_seed:      int,
    n_samples:      int,
    block_size:     int = 250,
    price_col:      str = "Zamkniecie",
    cash_price_col: str = "Zamkniecie",
    **wf_kwargs:    object,
) -> pd.DataFrame:

    n_jobs = get_n_jobs()
    common_idx = df.index.intersection(other=cash_df.index)
    combined = df.loc[common_idx, [price_col]].copy()
    combined["cash_price"] = cash_df.loc[common_idx, cash_price_col]

    valid = []

    try:
        source = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
        )(
            delayed(
                function=_sharded_bootstrap_single_sample,
            )(
                iteration=i,
                base_seed=base_seed,
                combined=combined,
                df=df,
                cash_df=cash_df,
                price_col=price_col,
                cash_price_col=cash_price_col,
                block_size=block_size,
                wf_kwargs=wf_kwargs,
            )
            for i in range(n_samples)
        )

        for idx, r in enumerate(source):
            if r is not None:
                valid.append(r)

    except Exception as e:
        logging.error(
            msg=f"Parallel bootstrap failed: {e}",
        )

    return pd.DataFrame(data=valid)

# =========================================================================
# SHARDED VALIDATION MANAGER
# =========================================================================

class ShardedValidationManager:
    def __init__(
        self,
        run_mode:            str,
        shard_id:            int,
        n_mc:                int,
        n_boot:              int,
        run_weights_perturb: bool,
    ) -> None:
        self.run_mode = run_mode
        self.shard_id = shard_id
        self.n_mc = n_mc
        self.n_boot = n_boot
        self.run_weights_perturb = run_weights_perturb
        self.creds_path = str(Path(tempfile.gettempdir()) / "credentials.json")
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _save_validation_chart(
        self,
        strategy_equity: pd.Series,
        bh_equity:       pd.Series | None,
        title:           str,
        filename:        str,
    ) -> None:
        chart_path = OUTPUT_DIR / filename
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        ax1.plot(
            strategy_equity.index,
            strategy_equity.values,
            label="Strategy (OOS)",
            color="steelblue",
            linewidth=2,
        )
        if bh_equity is not None:
            bh_aligned = bh_equity.reindex(index=strategy_equity.index).ffill()
            bh_norm = bh_aligned / bh_aligned.iloc[0]
            ax1.plot(
                bh_norm.index,
                bh_norm.values,
                label="Buy & Hold",
                color="grey",
                linestyle="--",
                alpha=0.7,
            )

        ax1.set_title(label=title, fontsize=14, fontweight="bold")
        ax1.set_ylabel(ylabel="Normalised Equity")
        ax1.legend(loc="upper left")
        ax1.grid(visible=True, alpha=0.3)

        dd_strat = strategy_equity / strategy_equity.cummax() - 1.0
        ax2.fill_between(
            dd_strat.index, dd_strat.values, 0.0, color="steelblue", alpha=0.3, label="Strategy DD",
        )

        if bh_equity is not None:
            dd_bh = bh_norm / bh_norm.cummax() - 1.0
            ax2.plot(dd_bh.index, dd_bh.values, color="grey", linestyle="--", alpha=0.5)

        ax2.set_ylabel(ylabel="Drawdown")
        ax2.grid(visible=True, alpha=0.3)
        ax2.set_ylim(top=0.0)

        plt.savefig(fname=chart_path, dpi=72, bbox_inches="tight")
        plt.close(fig=fig)
        logging.info(msg=f"Validation chart saved to: {chart_path}")

    def handle_mc(
        self,
        asset_name:  str,
        wf_results:  pd.DataFrame,
        df:          pd.DataFrame,
        cash_df:     pd.DataFrame,
        thresholds:  dict,
        base_equity: pd.Series,
    ) -> None:

        if self.run_mode == "worker":
            if self.n_mc > 0:
                logging.info(
                    msg=f"Worker {self.shard_id} running {self.n_mc} MC samples for {asset_name}...",
                )
                seed_offset = 42 + (self.shard_id * 10000)
                windows = extract_windows_from_wf_results(
                    wf_results=wf_results,
                )
                best_params = extract_best_params_from_wf_results(
                    wf_results=wf_results,
                )
                mc_df = run_monte_carlo_robustness(
                    best_params=best_params,
                    windows=windows,
                    df=df,
                    cash_df=cash_df,
                    vol_window=20,
                    selected_mode="full",
                    n_samples=self.n_mc,
                    n_jobs=get_n_jobs(),
                    perturb_pct=0.20,
                    seed=seed_offset,
                    price_col="Zamkniecie",
                )
                out_path = OUTPUT_DIR / f"shard_{self.shard_id}_mc_{asset_name}.csv"
                mc_df.to_csv(path_or_buf=out_path, index=False)

        elif self.run_mode == "merge":
            files = list(OUTPUT_DIR.glob(pattern=f"shard_*_mc_{asset_name}.csv"))
            if files:
                mc_df = pd.concat(
                    objs=[pd.read_csv(filepath_or_buffer=f) for f in files],
                    ignore_index=True,
                )
                logging.info(
                    msg=f"Merge: Found {len(files)} shards, total {len(mc_df)} MC samples for {asset_name}.",
                )
                analyze_robustness(
                    results_df=mc_df,
                    baseline_metrics=compute_metrics(equity=base_equity),
                    thresholds=thresholds,
                )
            else:
                logging.warning(
                    msg=f"No MC shard files found for {asset_name}",
                )

    def handle_boot(
        self,
        asset_name:  str,
        df:          pd.DataFrame,
        cash_df:     pd.DataFrame,
        train_y:     int,
        test_y:      int,
        use_atr:     bool,
        grid_type:   str,
        thresholds:  dict,
        base_equity: pd.Series,
        entry_gate:  pd.Series | None = None,
    ) -> None:

        grids = BOND_GRIDS if grid_type == "BOND" else BASE_GRIDS

        if self.run_mode == "worker":
            if self.n_boot > 0:
                logging.info(
                    msg=f"Worker {self.shard_id} running {self.n_boot} Bootstrap samples for {asset_name}...",
                )
                seed_offset = 1000 + (self.shard_id * 10000)

                bb_df = run_sharded_block_bootstrap(
                    df=df,
                    cash_df=cash_df,
                    base_seed=seed_offset,
                    n_samples=self.n_boot,
                    block_size=250,
                    train_years=train_y,
                    test_years=test_y,
                    use_atr_stop=use_atr,
                    N_atr_grid=grids["N_ATR_GRID"] if use_atr else None,
                    X_grid=grids["X_GRID"],
                    Y_grid=grids["Y_GRID"],
                    fast_grid=grids["FAST_GRID"],
                    slow_grid=grids["SLOW_GRID"],
                    filter_modes_override=["ma"] if grid_type == "BOND" else None,
                    fast_mode=True,
                    entry_gate_series=entry_gate,
                )
                out_path = OUTPUT_DIR / f"shard_{self.shard_id}_boot_{asset_name}.csv"
                bb_df.to_csv(path_or_buf=out_path, index=False)

        elif self.run_mode == "merge":
            files = list(OUTPUT_DIR.glob(pattern=f"shard_*_boot_{asset_name}.csv"))
            if files:
                bb_df = pd.concat(
                    objs=[pd.read_csv(filepath_or_buffer=f) for f in files],
                    ignore_index=True,
                )
                logging.info(
                    msg=f"Merge: Found {len(files)} shards, total {len(bb_df)} Bootstrap samples for {asset_name}.",
                )
                analyze_bootstrap(
                    results_df=bb_df,
                    baseline_metrics=compute_metrics(equity=base_equity),
                    thresholds=thresholds,
                )
            else:
                logging.warning(
                    msg=f"No Bootstrap shard files found for {asset_name}",
                )

    def validate_single(
        self,
        asset_name: str,
        train_y:    int,
        test_y:     int,
        stop_type:  str,
        df:         pd.DataFrame,
        cash_df:    pd.DataFrame,
    ) -> None:

        logging.info(
            msg=f"VALIDATING SINGLE ASSET: {asset_name} | {train_y}+{test_y} | {stop_type}",
        )
        use_atr = stop_type == "atr"

        wf_eq, wf_res, _ = walk_forward(
            df=df,
            cash_df=cash_df,
            train_years=train_y,
            test_years=test_y,
            X_grid=BASE_GRIDS["X_GRID"],
            Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"],
            slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr,
            N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
            n_jobs=get_n_jobs(),
            fast_mode=True,
        )

        self.handle_mc(
            asset_name=asset_name,
            wf_results=wf_res,
            df=df,
            cash_df=cash_df,
            thresholds=EQUITY_THRESHOLDS_MC,
            base_equity=wf_eq,
        )

        self.handle_boot(
            asset_name=asset_name,
            df=df,
            cash_df=cash_df,
            train_y=train_y,
            test_y=test_y,
            use_atr=use_atr,
            grid_type="EQUITY",
            thresholds=EQUITY_THRESHOLDS_BOOTSTRAP,
            base_equity=wf_eq,
        )

        if self.run_mode == "merge":
            bh_eq, _ = compute_buy_and_hold(
                df=df,
                price_col="Zamkniecie",
                start=wf_eq.index.min(),
                end=wf_eq.index.max(),
            )
            self._save_validation_chart(
                strategy_equity=wf_eq,
                bh_equity=bh_eq,
                title=f"OOS Validation: {asset_name} ({train_y}+{test_y} {stop_type})",
                filename=f"validate_{asset_name.lower()}_{train_y}_{test_y}.png",
            )

    def validate_pension(
        self,
        train_y:      int,
        test_y:       int,
        stop_type_eq: str,
    ) -> None:

        logging.info(
            msg=f"VALIDATING PENSION PORTFOLIO | Train: {train_y} | Test: {test_y} | EQ Stop: {stop_type_eq}",
        )

        wig_df = load_local_csv(
            ticker="wig",
            label="WIG",
        ).loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]

        mmf_df = load_local_csv(
            ticker="fund_2720",
            label="MMF",
        )

        tbsp_df = build_and_upload(
            folder_id=self.folder_id,
            raw_filename="tbsp_extended_full.csv",
            combined_filename="tbsp_extended_combined.csv",
            extension_ticker="^tbsp",
            extension_source="stooq",
            credentials_path=self.creds_path,
        )

        wibor1m_df = load_local_csv(ticker="wibor1m", label="WIBOR1M", mandatory=False)
        pl10y_df = load_local_csv(ticker="pl10y", label="PL10Y")
        de10y_df = load_local_csv(ticker="de10y", label="DE10Y")

        derived = build_standard_two_asset_data(
            wig=wig_df,
            tbsp=tbsp_df,
            mmf=mmf_df,
            wibor1m=wibor1m_df,
            pl10y=pl10y_df,
            de10y=de10y_df,
            mmf_floor="1995-01-02",
        )
        use_atr_eq = stop_type_eq == "atr"

        # --- COMPONENT 1: WIG ---
        wf_eq, wf_res_eq, wf_tr_eq = walk_forward(
            df=wig_df,
            cash_df=derived["mmf_ext"],
            train_years=train_y,
            test_years=test_y,
            X_grid=BASE_GRIDS["X_GRID"],
            Y_grid=BASE_GRIDS["Y_GRID"],
            fast_grid=BASE_GRIDS["FAST_GRID"],
            slow_grid=BASE_GRIDS["SLOW_GRID"],
            use_atr_stop=use_atr_eq,
            N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr_eq else None,
            n_jobs=get_n_jobs(),
            fast_mode=True,
        )

        self.handle_mc(
            asset_name="WIG",
            wf_results=wf_res_eq,
            df=wig_df,
            cash_df=derived["mmf_ext"],
            thresholds=EQUITY_THRESHOLDS_MC,
            base_equity=wf_eq,
        )

        self.handle_boot(
            asset_name="WIG",
            df=wig_df,
            cash_df=derived["mmf_ext"],
            train_y=train_y,
            test_y=test_y,
            use_atr=use_atr_eq,
            grid_type="EQUITY",
            thresholds=EQUITY_THRESHOLDS_BOOTSTRAP,
            base_equity=wf_eq,
        )

        # --- COMPONENT 2: TBSP ---
        wf_bd_eq, wf_bd_res, wf_bd_tr = walk_forward(
            df=tbsp_df,
            cash_df=derived["mmf_ext"],
            train_years=train_y,
            test_years=test_y,
            filter_modes_override=["ma"],
            X_grid=BOND_GRIDS["X_GRID"],
            Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"],
            slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(),
            fast_mode=True,
            entry_gate_series=derived["bond_gate"],
        )

        self.handle_mc(
            asset_name="TBSP",
            wf_results=wf_bd_res,
            df=tbsp_df,
            cash_df=derived["mmf_ext"],
            thresholds=BOND_THRESHOLDS_MC,
            base_equity=wf_bd_eq,
        )

        self.handle_boot(
            asset_name="TBSP",
            df=tbsp_df,
            cash_df=derived["mmf_ext"],
            train_y=train_y,
            test_y=test_y,
            use_atr=False,
            grid_type="BOND",
            thresholds=BOND_THRESHOLDS_BOOTSTRAP,
            base_equity=wf_bd_eq,
            entry_gate=derived["bond_gate"],
        )

        if self.run_mode == "worker":
            return

        # --- MERGE SPECIFIC: Portfolio & Allocation ---
        sig_eq = build_signal_series(wf_equity=wf_eq, wf_trades=wf_tr_eq)
        sig_bd = build_signal_series(wf_equity=wf_bd_eq, wf_trades=wf_bd_tr)

        port_eq, _, _, alloc_df = allocation_walk_forward(
            equity_returns=derived["ret_eq"],
            bond_returns=derived["ret_bd"],
            mmf_returns=derived["ret_mmf"],
            sig_equity_full=sig_eq,
            sig_bond_full=sig_bd,
            sig_equity_oos=sig_eq.loc[wf_eq.index.min() :],
            sig_bond_oos=sig_bd.loc[wf_eq.index.min() :],
            wf_results_eq=wf_res_eq,
            wf_results_bd=wf_bd_res,
        )

        bh_wig, _ = compute_buy_and_hold(
            df=wig_df,
            price_col="Zamkniecie",
            start=port_eq.index.min(),
            end=port_eq.index.max(),
        )

        self._save_validation_chart(
            strategy_equity=port_eq,
            bh_equity=bh_wig,
            title=f"OOS Validation: PENSION Portfolio ({train_y}+{test_y})",
            filename=f"validate_pension_{train_y}_{test_y}.png",
        )

        if self.run_weights_perturb:
            logging.info(
                msg="\n" + "=" * 80 + "\n--- Level 3: Allocation Weight Perturbation Test (PENSION) ---\n" + "=" * 80,
            )
            robust_df = allocation_weight_robustness(
                alloc_results_df=alloc_df,
                equity_returns=derived["ret_eq"],
                bond_returns=derived["ret_bd"],
                mmf_returns=derived["ret_mmf"],
                sig_equity_oos=sig_eq.loc[wf_eq.index.min() :],
                sig_bond_oos=sig_bd.loc[wf_eq.index.min() :],
                baseline_metrics=compute_metrics(equity=port_eq),
            )
            print_allocation_robustness_report(results_df=robust_df)


    def validate_global(
        self,
        variant:      str,
        train_y:      int,
        test_y:       int,
        stop_type_eq: str,
    ) -> None:

        logging.info(
            msg=f"VALIDATING GLOBAL - {variant} | Train: {train_y} | Test: {test_y} | Stop: {stop_type_eq}",
        )

        cfg = ASSET_REGISTRY[variant]
        mode = cfg["mode"]
        fx_hedged = cfg.get("fx_hedged", True)
        use_atr = stop_type_eq == "atr"

        wig_df = load_local_csv(
            ticker="wig",
            label="WIG",
        ).loc[lambda x: x.index >= pd.Timestamp("1995-01-02")]

        mmf_df = load_local_csv(ticker="fund_2720", label="MMF")
        wibor1m_df = load_local_csv(ticker="wibor1m", label="WIBOR1M", mandatory=False)
        mmf_ext = build_mmf_extended(mmf_df=mmf_df, wibor1m_df=wibor1m_df, floor_date="1995-01-02")

        tbsp_df = build_and_upload(
            folder_id=self.folder_id,
            raw_filename="tbsp_extended_full.csv",
            combined_filename="tbsp_extended_combined.csv",
            extension_ticker="^tbsp",
            extension_source="stooq",
            credentials_path=self.creds_path,
        )

        fx_map = {
            c: load_local_csv(ticker=f"{c.lower()}pln", label=f"{c}PLN")["Zamkniecie"]
            for c in ["USD", "EUR", "JPY"]
        }

        if mode == "global_equity":
            stoxx = build_and_upload(
                folder_id=self.folder_id,
                raw_filename="stoxx600.csv",
                combined_filename="stoxx600_combined.csv",
                extension_ticker="^STOXX",
                extension_source="yfinance",
                credentials_path=self.creds_path,
            )
            assets = {
                "WIG": (wig_df, None),
                "SP500": (load_local_csv(ticker="sp500", label="SP500"), fx_map["USD"]),
                "STOXX600": (stoxx, fx_map["EUR"]),
                "Nikkei225": (load_local_csv(ticker="nikkei225", label="Nikkei225"), fx_map["JPY"]),
            }
        else:
            msciw = build_and_upload(
                folder_id=self.folder_id,
                raw_filename="msci_world_wsj_raw.csv",
                combined_filename="msci_world_combined.csv",
                extension_ticker="URTH",
                extension_source="yfinance",
                credentials_path=self.creds_path,
                is_msci_world=True,
            )
            assets = {"WIG": (wig_df, None), "MSCI_World": (msciw, fx_map["USD"])}

        rets_dict = {}
        sigs_full = {}

        # 1. Equity Components
        for lbl, (px_df, fx_s) in assets.items():
            ret_s = build_return_series(price_df=px_df, fx_series=fx_s, hedged=fx_hedged)
            rets_dict[lbl] = ret_s.dropna()
            proc_px = (
                px_df if fx_hedged or fx_s is None
                else build_price_df_from_returns(ret=ret_s, label=lbl)
            )

            wf_e, wf_r, wf_t = walk_forward(
                df=proc_px,
                cash_df=mmf_ext,
                train_years=train_y,
                test_years=test_y,
                X_grid=BASE_GRIDS["X_GRID"],
                Y_grid=BASE_GRIDS["Y_GRID"],
                fast_grid=BASE_GRIDS["FAST_GRID"],
                slow_grid=BASE_GRIDS["SLOW_GRID"],
                use_atr_stop=use_atr,
                N_atr_grid=BASE_GRIDS["N_ATR_GRID"] if use_atr else None,
                n_jobs=get_n_jobs(),
                fast_mode=True,
            )
            sigs_full[lbl] = build_signal_series(wf_equity=wf_e, wf_trades=wf_t)

            self.handle_mc(
                asset_name=lbl,
                wf_results=wf_r,
                df=proc_px,
                cash_df=mmf_ext,
                thresholds=EQUITY_THRESHOLDS_MC,
                base_equity=wf_e,
            )

            self.handle_boot(
                asset_name=lbl,
                df=proc_px,
                cash_df=mmf_ext,
                train_y=train_y,
                test_y=test_y,
                use_atr=use_atr,
                grid_type="EQUITY",
                thresholds=EQUITY_THRESHOLDS_BOOTSTRAP,
                base_equity=wf_e,
            )

        # 2. Bond Component
        pl10y_df = load_local_csv(ticker="pl10y", label="PL10Y")
        de10y_df = load_local_csv(ticker="de10y", label="DE10Y")
        derived = build_standard_two_asset_data(
            wig=wig_df, tbsp=tbsp_df, mmf=mmf_df, wibor1m=wibor1m_df,
            pl10y=pl10y_df, de10y=de10y_df, mmf_floor="1995-01-02",
        )

        wf_bd, wf_res_bd, wf_tr_bd = walk_forward(
            df=tbsp_df,
            cash_df=mmf_ext,
            train_years=train_y,
            test_years=test_y,
            filter_modes_override=["ma"],
            X_grid=BOND_GRIDS["X_GRID"],
            Y_grid=BOND_GRIDS["Y_GRID"],
            fast_grid=BOND_GRIDS["FAST_GRID"],
            slow_grid=BOND_GRIDS["SLOW_GRID"],
            n_jobs=get_n_jobs(),
            fast_mode=True,
            entry_gate_series=derived["bond_gate"],
        )
        rets_dict["TBSP"] = tbsp_df["Zamkniecie"].pct_change().dropna()
        sigs_full["TBSP"] = build_signal_series(wf_equity=wf_bd, wf_trades=wf_tr_bd)

        self.handle_mc(
            asset_name="TBSP",
            wf_results=wf_res_bd,
            df=tbsp_df,
            cash_df=mmf_ext,
            thresholds=BOND_THRESHOLDS_MC,
            base_equity=wf_bd,
        )

        self.handle_boot(
            asset_name="TBSP",
            df=tbsp_df,
            cash_df=mmf_ext,
            train_y=train_y,
            test_y=test_y,
            use_atr=False,
            grid_type="BOND",
            thresholds=BOND_THRESHOLDS_BOOTSTRAP,
            base_equity=wf_bd,
            entry_gate=derived["bond_gate"],
        )

        if self.run_mode == "worker":
            return

        # 3. Portfolio Allocation & Chart
        logging.info(
            msg="\n" + "-" * 60 + "\nRunning N-Asset allocation walk-forward...\n" + "-" * 60,
        )
        port_eq, _, _, alloc_df = allocation_walk_forward_n(
            returns_dict=rets_dict,
            signals_full_dict=sigs_full,
            signals_oos_dict=sigs_full,
            mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(),
            wf_results_ref=wf_res_bd,
            asset_keys=list(rets_dict.keys()),
            train_years=train_y,
        )

        bh_wig, _ = compute_buy_and_hold(
            df=wig_df,
            price_col="Zamkniecie",
            start=port_eq.index.min(),
            end=port_eq.index.max(),
        )

        self._save_validation_chart(
            strategy_equity=port_eq,
            bh_equity=bh_wig,
            title=f"OOS Validation: GLOBAL {variant} ({train_y}+{test_y})",
            filename=f"validate_{variant.lower()}_{train_y}_{test_y}.png",
        )

        # 4. Weight Perturbation
        if self.run_weights_perturb:
            logging.info(
                msg="\n" + "=" * 80 + "\n--- Level 3: Allocation Weight Perturbation Test (GLOBAL) ---\n" + "=" * 80,
            )
            robust_df = allocation_weight_robustness_n(
                alloc_results_df=alloc_df,
                returns_dict=rets_dict,
                mmf_returns=mmf_ext["Zamkniecie"].pct_change().dropna(),
                signals_oos_dict={k: v.loc[port_eq.index.min() :] for k, v in sigs_full.items()},
                asset_keys=list(rets_dict.keys()),
                baseline_metrics=compute_metrics(equity=port_eq),
                focus_asset="WIG",
            )
            print_allocation_robustness_report_n(
                results_df=robust_df,
                focus_asset="WIG",
            )

# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Sharded Deep Robustness Validator")
    parser.add_argument("--run_mode", choices=["worker", "merge"], required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--mode", choices=["SINGLE", "PENSION", "GLOBAL"], required=True)
    parser.add_argument("--asset", help="Dla mode SINGLE lub GLOBAL (np. WIG20TR, GLOBAL_A)")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--stop", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--n_mc", type=int, default=100)
    parser.add_argument("--n_boot", type=int, default=50)
    parser.add_argument("--weights_perturb", action="store_true")
    args = parser.parse_args()

    # Logging setup
    log_file = OUTPUT_DIR / f"sharded_{args.run_mode}_{args.shard_id}_{args.mode.lower()}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )

    DataUpdater().run_full_update(get_funds=False)

    manager = ShardedValidationManager(
        run_mode=args.run_mode,
        shard_id=args.shard_id,
        n_mc=args.n_mc,
        n_boot=args.n_boot,
        run_weights_perturb=args.weights_perturb,
    )

    if args.mode == "SINGLE":
        df = load_local_csv(ticker=args.asset.lower(), label=args.asset)
        cash_df = load_local_csv(ticker="fund_2720", label="MMF")
        manager.validate_single(
            asset_name=args.asset,
            train_y=args.train,
            test_y=args.test,
            stop_type=args.stop,
            df=df,
            cash_df=cash_df,
        )
    elif args.mode == "PENSION":
        manager.validate_pension(
            train_y=args.train,
            test_y=args.test,
            stop_type_eq=args.stop,
        )
    elif args.mode == "GLOBAL":
        manager.validate_global(
            variant=args.asset,
            train_y=args.train,
            test_y=args.test,
            stop_type_eq=args.stop,
        )


if __name__ == "__main__":
    main()
