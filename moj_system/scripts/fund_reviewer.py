# -*- coding: utf-8 -*-
"""
moj_system/scripts/fund_reviewer.py
===================================
Master script for TFI Fund Analysis.
Full metadata merge from KNF Summary, performance metrics, and ranking generation.
"""

import argparse
import logging
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
import pandas as pd

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from moj_system.core.fund_analytics import BenchmarkComparator, FundPerformanceEngine
from moj_system.data.data_manager import load_local_csv
from moj_system.data.gdrive import GDriveClient
from moj_system.data.updater import DataUpdater

# --- CONFIGURATION ---
WINDOWS = {"1y": 252, "3y": 756, "5y": 1260, "all": None}
MIN_CATEGORY_PEERS = 3

# --- EXACT COLUMN ORDER AS REQUESTED (35 Columns) ---
FINAL_COLS = [
    "subfundId",
    "stooq_id",
    "knf_name",
    "tfi_name",
    "category",
    "classification",
    "riskLevel",
    "managementAndOtherFeesRate",
    "maxEntryFeeRate",
    "hasBenchmark",
    "benchmark",
    "window",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "data_years",
    "n_days",
    "insufficient",
    "fetch_ok",
    "alpha",
    "beta",
    "tracking_error",
    "ir",
    "pct_ir_positive",
    "n_ir_windows",
    "category_funds",
    "category_peers_ok",
    "rank_ir",
    "rank_sharpe",
    "rank_sortino",
    "rank_calmar",
    "rank_return",
]


class FundReviewer:
    def __init__(self, update_data=True):
        self.creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        self.gdrive = GDriveClient(credentials_path=self.creds_path)
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")

        if update_data:
            logging.info("Updating all KNF fund data (API + ZIP)...")
            updater = DataUpdater(credentials_path=self.creds_path)
            updater.run_full_update(get_funds=True)

        logging.info("Loading Benchmarks for Hit Rate calculation...")
        self.benchmarks = {}
        for name in ["WIG20TR", "MWIG40TR", "SWIG80TR", "WBBW"]:
            df = load_local_csv(name.lower(), name, mandatory=False)
            if df is not None and "Zamkniecie" in df.columns:
                self.benchmarks[name] = df["Zamkniecie"].squeeze()

    def load_confirmed_funds(self):
        """Loads confirmed list and merges with ALL metadata from knf_summary.csv"""
        if not self.folder_id:
            sys.exit("GDRIVE_FOLDER_ID not set.")

        df_conf = self.gdrive.download_csv(self.folder_id, "knf_stooq_confirmed.csv", sep=",")
        if df_conf is None:
            sys.exit("Could not load confirmed list.")
        df_conf["subfundId"] = pd.to_numeric(df_conf["subfundId"], errors="coerce")

        df_sum = self.gdrive.download_csv(self.folder_id, "knf_summary.csv", sep=";")
        if df_sum is not None:
            df_sum["subfundId"] = pd.to_numeric(df_sum["subfundId"], errors="coerce")
            meta_cols = [
                "subfundId",
                "tfi_name",
                "classification",
                "category",
                "riskLevel",
                "managementAndOtherFeesRate",
                "maxEntryFeeRate",
                "hasBenchmark",
                "benchmark",
            ]

            # --- POPRAWKA: Bezpieczne wybieranie tylko ISTNIEJĄCYCH kolumn ---
            # Dzięki temu nie wywali błędu, jeśli KNF nie podał którejś z metryk w API
            available_meta_cols = [col for col in meta_cols if col in df_sum.columns]

            # Merge while keeping confirmed names as primary
            df_funds = df_conf.merge(
                df_sum[available_meta_cols], on="subfundId", how="left", suffixes=("", "_knf"),
            )
            # ----------------------------------------------------------------
            logging.info("Metadata from knf_summary.csv merged successfully.")
        else:
            logging.warning("knf_summary.csv not found. Metadata will be missing!")
            df_funds = df_conf

        df_funds = df_funds.dropna(subset=["stooq_id"]).copy()

        fund_prices = {}
        for row_index, fund_row in df_funds.iterrows():
            sid = "Unknown"
            try:
                # Konwersja ID
                sid = str(int(float(fund_row["stooq_id"])))
                knf_label = fund_row.get("knf_name", "Unknown")

                # Jawne przekazanie argumentów
                prices_df = load_local_csv(
                    ticker=f"fund_{sid}",
                    label=knf_label,
                    mandatory=False,
                )

                if prices_df is not None:
                    fund_prices[sid] = prices_df["Zamkniecie"].squeeze()

            except Exception as exc:
                logging.warning(
                    f"Skipping price load for fund index {row_index} (SID: {sid}): {exc}",
                )
                continue

        logging.info(f"Loaded {len(fund_prices)} price histories.")
        return df_funds, fund_prices

    def build_category_benchmarks(self, df_funds, fund_prices):
        """Builds synthetic benchmarks for each KNF category."""
        logging.info("Building synthetic category benchmarks...")
        category_rets = defaultdict(list)

        for row_index, fund_row in df_funds.iterrows():
            sid = "Unknown"
            try:
                sid = str(int(float(fund_row["stooq_id"])))
                category = fund_row.get("category")
                prices = fund_prices.get(sid)

                if category and prices is not None:
                    # Obliczanie log-zwrotów
                    log_ret = np.log(prices / prices.shift(1)).dropna()
                    log_ret.name = sid
                    category_rets[category].append(log_ret)

            except Exception as exc:
                logging.warning(
                    f"Error processing returns for benchmark at index {row_index} (SID: {sid}): {exc}",
                )
                continue

        benchmarks = {}
        for cat, rets in category_rets.items():
            if rets:
                # Jawne przekazanie argumentów do concat
                benchmarks[cat] = pd.concat(objs=rets, axis=1, sort=True).mean(axis=1)
        return benchmarks

    def evaluate_fund(self, fund_row, prices, cat_benchmarks):
        """Calculates performance and regression metrics including Hit Rates."""
        category = fund_row.get("category")
        cat_bench_ret = cat_benchmarks.get(category)

        # Base benchmark for Alpha/Beta calculation if no category peers
        wig20 = self.benchmarks.get("WIG20TR")
        base_bench_ret = np.log(wig20 / wig20.shift(1)).dropna() if wig20 is not None else None

        results = []
        for win_name, win_days in WINDOWS.items():
            abs_m = FundPerformanceEngine.compute_absolute_metrics(prices, win_days)
            if abs_m["insufficient"]:
                continue

            fund_log_ret = abs_m.pop("_log_ret")
            rel_m = {}

            # Prefer category benchmark, fall back to WIG20
            target_bench = cat_bench_ret if cat_bench_ret is not None else base_bench_ret

            if target_bench is not None:
                rel_m = FundPerformanceEngine.compute_relative_metrics(fund_log_ret, target_bench)
                rel_m.update(FundPerformanceEngine.compute_rolling_ir(fund_log_ret, target_bench))

            peers_data = cat_benchmarks.get(category)
            n_peers = len(peers_data) if peers_data is not None else 0

            row_data = {
                "subfundId": fund_row.get("subfundId"),
                "stooq_id": str(int(float(fund_row["stooq_id"]))),
                "knf_name": fund_row.get("knf_name"),
                "tfi_name": fund_row.get("tfi_name"),
                "category": category,
                "classification": fund_row.get("classification"),
                "riskLevel": fund_row.get("riskLevel"),
                "managementAndOtherFeesRate": fund_row.get("managementAndOtherFeesRate"),
                "maxEntryFeeRate": fund_row.get("maxEntryFeeRate"),
                "hasBenchmark": fund_row.get("hasBenchmark"),
                "benchmark": fund_row.get("benchmark"),
                "window": win_name,
                **abs_m,
                "fetch_ok": True,
                **rel_m,
                "category_funds": n_peers,
                "category_peers_ok": n_peers >= MIN_CATEGORY_PEERS,
            }

            # Optional Hit Rate calculation for "all" window
            if win_name == "all":
                for b_name, b_prices in self.benchmarks.items():
                    if b_prices is not None:
                        win_rates, _, _ = BenchmarkComparator.compare_fund_to_benchmark(
                            prices, b_prices,
                        )
                        row_data[f"hit_{b_name}_22d"] = round(win_rates.get("22-day", np.nan), 1)
                        row_data[f"hit_{b_name}_66d"] = round(win_rates.get("66-day", np.nan), 1)
                        row_data[f"hit_{b_name}_252d"] = round(win_rates.get("252-day", np.nan), 1)

            results.append(row_data)
        return results

    def generate_reports(self):
        df_funds, fund_prices = self.load_confirmed_funds()
        category_benchmarks = self.build_category_benchmarks(df_funds, fund_prices)

        all_results = []
        # Changed 'index' instead of '_' to satisfy developer instructions
        for row_index, fund_row in df_funds.iterrows():
            sid = "Unknown"
            try:
                sid = str(int(float(fund_row["stooq_id"])))
                prices = fund_prices.get(sid)

                if prices is not None:
                    # Explicit argument passing as per instructions
                    fund_evaluation = self.evaluate_fund(
                        fund_row=fund_row,
                        prices=prices,
                        cat_benchmarks=category_benchmarks,
                    )
                    all_results.extend(fund_evaluation)

            except Exception as exc:
                # Log the error so we know WHICH fund failed and WHY
                fund_name = fund_row.get("knf_name", "Unknown")
                logging.error(
                    f"Failed to evaluate fund '{fund_name}' (SID: {sid}) at index {row_index}: {exc}",
                )
                continue

        if not all_results:
            logging.warning("No results were generated during fund evaluation.")
            return

        final_df = pd.DataFrame(all_results)

        # Ranking Logic
        rank_df = final_df.copy()
        for metric in ["ir", "sharpe", "sortino", "calmar", "ann_return"]:
            rank_col = f"rank_{metric}"
            if metric in rank_df.columns:
                rank_df[rank_col] = rank_df.groupby(["category", "window"])[metric].rank(
                    method="min", ascending=False,
                )
            else:
                rank_df[rank_col] = np.nan

        # Enforce exact columns and order
        for col in FINAL_COLS:
            if col not in rank_df.columns:
                rank_df[col] = np.nan

        # Append hit rate columns at the end of ranking if they exist
        other_cols = [c for c in rank_df.columns if c.startswith("hit_")]
        rank_df = rank_df[FINAL_COLS + other_cols]

        today = dt.today().strftime("%Y%m%d")
        os.makedirs("outputs", exist_ok=True)
        perf_path, rank_path = (
            f"outputs/fund_performance_{today}.csv",
            f"outputs/fund_ranking_{today}.csv",
        )

        final_df.to_csv(perf_path, index=False, sep=";")
        rank_df.to_csv(rank_path, index=False, sep=";")

        logging.info(f"Generated ranking with {len(rank_df)} rows: {rank_path}")

        if self.folder_id:
            self.gdrive.upload_csv(self.folder_id, perf_path)
            self.gdrive.upload_csv(self.folder_id, rank_path)


def main():
    parser = argparse.ArgumentParser(description="TFI Comprehensive Reviewer")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    reviewer = FundReviewer(update_data=args.update)
    reviewer.generate_reports()


if __name__ == "__main__":
    main()
