# -*- coding: utf-8 -*-
"""
moj_system/scripts/fund_reviewer.py
===================================
Master script for TFI Fund Analysis.
Full metadata merge from KNF Summary, performance metrics, and ranking generation.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections.abc import Hashable, Mapping
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path Setup ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from moj_system.config import OUTPUT_DIR, PROJECT_ROOT  # noqa: E402
from moj_system.core.benchmark_builder import BenchmarkBuilder  # noqa: E402
from moj_system.core.benchmark_catalog import INDEX_CATALOG  # noqa: E402
from moj_system.core.benchmark_llm import GeminiBenchmarkResolver  # noqa: E402
from moj_system.core.benchmark_parser import (  # noqa: E402
    BenchmarkComponent,
    ParsedBenchmark,
    parse_benchmark_text,
)
from moj_system.core.fund_analytics import BenchmarkComparator, FundPerformanceEngine  # noqa: E402
from moj_system.data.data_manager import load_local_csv  # noqa: E402
from moj_system.data.gdrive import GDriveClient  # noqa: E402
from moj_system.data.updater import DataUpdater  # noqa: E402

# --- CONFIGURATION ---
WINDOWS = {"1y": 252, "3y": 756, "5y": 1260, "all": None}
MIN_CATEGORY_PEERS = 3

# --- Report column order ---
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
    "active_return",
    "tracking_error",
    "ir",
    "appraisal_ratio",
    "residual_tracking_error",
    "correlation",
    "r_squared",
    "n_relative_observations",
    "pct_ir_positive",
    "n_ir_windows",
    "peer_ir",
    "category_funds",
    "category_peers_ok",
    "benchmark_source",
    "benchmark_series_source",
    "benchmark_quality",
    "benchmark_frequency",
    "benchmark_coverage",
    "benchmark_components",
    "benchmark_unresolved",
    "benchmark_reason",
    "benchmark_is_proxy",
    "benchmark_rankable",
    "rank_ir",
    "rank_appraisal_ratio",
    "rank_sharpe",
    "rank_sortino",
    "rank_calmar",
    "rank_return",
]


class FundReviewer:
    def __init__(
        self,
        update_data: bool = True,
        use_gemini: bool = False,
    ) -> None:

        self.creds_path = str(Path(tempfile.gettempdir()) / "credentials.json")
        self.gdrive = GDriveClient(credentials_path=self.creds_path)
        self.folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        self.benchmark_builder = BenchmarkBuilder()
        self.gemini = GeminiBenchmarkResolver() if use_gemini else None
        self.benchmark_overrides_path = (
            PROJECT_ROOT / "moj_system" / "data" / "fund_benchmark_overrides.csv"
        )

        if update_data:
            logging.info("Updating all KNF fund data (API + ZIP)...")
            updater = DataUpdater(credentials_path=self.creds_path)
            updater.run_full_update(get_funds=True)

        logging.info("Loading fixed market benchmarks for hit-rate columns...")
        self.benchmarks: dict[str, pd.Series] = {}
        for name in ("WIG20TR", "MWIG40TR", "SWIG80TR", "WBBW"):
            frame = load_local_csv(name.lower(), name, mandatory=False)
            if frame is not None and "Zamkniecie" in frame.columns:
                self.benchmarks[name] = frame["Zamkniecie"]

    @staticmethod
    def _normalize_id(value: object) -> str | None:
        if isinstance(value, str):
            try:
                numeric_id = float(value)
            except ValueError:
                return None
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric_id = float(value)
        else:
            return None
        if not np.isfinite(numeric_id) or not numeric_id.is_integer():
            return None
        return str(int(numeric_id))

    def load_confirmed_funds(self) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
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
                df_sum[available_meta_cols],
                on="subfundId",
                how="left",
                suffixes=("", "_knf"),
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

    def build_category_benchmarks(
        self,
        df_funds: pd.DataFrame,
        fund_prices: dict[str, pd.Series],
    ) -> dict[str, dict[str, object]]:
        """Build leave-one-out category peers, excluding the fund under evaluation."""
        category_funds: dict[str, list[tuple[str, pd.Series]]] = {}
        for row in df_funds.to_dict(orient="records"):
            sid = self._normalize_id(row.get("stooq_id"))
            category_value = row.get("category")
            if sid is None:
                continue
            if not isinstance(category_value, str) or not category_value.strip():
                continue
            prices = fund_prices.get(sid)
            if prices is None or prices.empty:
                continue
            category_funds.setdefault(category_value, []).append((sid, prices))

        peer_benchmarks: dict[str, dict[str, object]] = {}
        for row in df_funds.to_dict(orient="records"):
            fund_key = self._normalize_id(row.get("subfundId"))
            sid = self._normalize_id(row.get("stooq_id"))
            if fund_key is None or sid is None:
                continue
            category = row.get("category")
            if not isinstance(category, str):
                continue
            peer_prices = {
                peer_sid: prices
                for peer_sid, prices in category_funds.get(category, [])
                if peer_sid != sid
            }
            peer_returns = [
                pd.Series(
                    np.log(prices.to_numpy() / prices.shift(1).to_numpy()),
                    index=prices.index,
                )
                .dropna()
                .rename(peer_sid)
                for peer_sid, prices in peer_prices.items()
            ]
            peer_series = (
                pd.concat(peer_returns, axis=1, sort=True).mean(axis=1).dropna()
                if peer_returns
                else pd.Series(dtype=float)
            )
            peer_benchmarks[fund_key] = {
                "log_returns": peer_series,
                "n_peers": len(peer_prices),
            }
        return peer_benchmarks

    @staticmethod
    def _category_proxy(
        fund_row: Mapping[Hashable, object] | pd.Series,
    ) -> ParsedBenchmark | None:
        category = str(fund_row.get("category", "")).replace("\xa0", " ").casefold()
        classification = str(fund_row.get("classification", "")).casefold()
        name = str(fund_row.get("knf_name", fund_row.get("name", ""))).casefold()
        if "akcje polskie" in category or category == "akcji polskich":
            components = (BenchmarkComponent("WIG", 1.0),)
        elif "akcje zagraniczne" in category or category == "akcji zagranicznych":
            components = (BenchmarkComponent("MSCI_ACWI", 1.0),)
        elif "surow" in category or "surow" in name:
            components = (BenchmarkComponent("BLOOMBERG_COMMODITY", 1.0),)
        elif "absolutnej stopy zwrotu" in category or "absolutnej stopy zwrotu" in classification:
            components = (BenchmarkComponent("WIBOR_3M", 1.0),)
        elif "dłuż" in classification or category.startswith("obligacji"):
            components = (BenchmarkComponent("TBSP", 1.0),)
        elif category.startswith("mieszane") or "mieszane" in classification:
            raw_risk = fund_row.get("riskLevel")
            try:
                risk_level = float(raw_risk) if isinstance(raw_risk, (str, int, float)) else 3.0
            except ValueError:
                risk_level = 3.0
            if not np.isfinite(risk_level):
                risk_level = 3.0
            equity_weight = min(0.8, max(0.2, (risk_level - 1.0) / 5.0))
            components = (
                BenchmarkComponent("WIG", equity_weight),
                BenchmarkComponent("TBSP", 1.0 - equity_weight),
            )
        else:
            return None
        return ParsedBenchmark(components, (), 1.0)

    @staticmethod
    def _parsed_from_gemini(result: Mapping[str, object]) -> ParsedBenchmark:
        component_data = result.get("components")
        if not isinstance(component_data, list):
            raise ValueError("Benchmark resolution has no component list.")
        components: list[BenchmarkComponent] = []
        for item in component_data:
            if not isinstance(item, Mapping):
                raise ValueError("Benchmark resolution contains a malformed component.")
            index_key = item.get("index_key")
            weight = item.get("weight")
            spread_bps = item.get("spread_bps", 0.0)
            hedged_to_pln = item.get("hedged_to_pln", False)
            if (
                not isinstance(index_key, str)
                or index_key not in INDEX_CATALOG
                or not isinstance(weight, (int, float, np.integer, np.floating))
                or not isinstance(spread_bps, (int, float, np.integer, np.floating))
                or not isinstance(hedged_to_pln, bool)
            ):
                raise ValueError("Benchmark resolution contains invalid component fields.")
            components.append(
                BenchmarkComponent(index_key, float(weight), float(spread_bps), hedged_to_pln)
            )
        portfolio_spread = result.get("portfolio_spread_bps", 0.0)
        if not isinstance(portfolio_spread, (int, float, np.integer, np.floating)):
            portfolio_spread = 0.0
        return ParsedBenchmark(
            tuple(components),
            (),
            sum(component.weight for component in components),
            portfolio_spread_bps=float(portfolio_spread),
        )

    def _load_benchmark_overrides(self) -> dict[str, dict[str, object]]:
        path = getattr(
            self,
            "benchmark_overrides_path",
            PROJECT_ROOT / "moj_system" / "data" / "fund_benchmark_overrides.csv",
        )
        if not path.exists():
            return {}
        try:
            overrides = pd.read_csv(path, sep=";")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            logging.error("Could not read manual benchmark overrides (%s).", type(exc).__name__)
            return {}
        required_columns = {"subfundId", "components_json", "kind"}
        if not required_columns.issubset(overrides.columns):
            logging.error(
                "Manual benchmark overrides require columns: %s.",
                ", ".join(sorted(required_columns)),
            )
            return {}

        parsed_overrides: dict[str, dict[str, object]] = {}
        for row in overrides.to_dict(orient="records"):
            fund_key = self._normalize_id(row.get("subfundId"))
            components_text = row.get("components_json")
            if fund_key is None or not isinstance(components_text, str):
                logging.warning("Skipping malformed manual benchmark override.")
                continue
            try:
                components = json.loads(components_text)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed manual benchmark override.")
                continue
            kind = str(row.get("kind")).strip().casefold()
            if kind not in {"official", "proxy"}:
                logging.warning("Skipping manual benchmark override with unknown kind for %s.", fund_key)
                continue
            portfolio_spread = pd.to_numeric(
                row.get("portfolio_spread_bps", 0.0),
                errors="coerce",
            )
            if pd.isna(portfolio_spread):
                portfolio_spread = 0.0
            validated = GeminiBenchmarkResolver._validate_response(
                {
                    "components": components,
                    "confidence": 1.0,
                    "reason": str(row.get("notes", "Manual override")),
                    "portfolio_spread_bps": portfolio_spread,
                },
            )
            if validated is None:
                logging.warning("Skipping invalid manual benchmark components for %s.", fund_key)
                continue
            parsed_overrides[fund_key] = {**validated, "kind": kind}
        return parsed_overrides

    def build_fund_benchmarks(
        self,
        df_funds: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """Resolve and build each fund's official KID benchmark or explicit proxy."""
        benchmark_data: dict[str, dict[str, object]] = {}
        manual_overrides = self._load_benchmark_overrides()
        for fund_row in df_funds.to_dict(orient="records"):
            fund_key = self._normalize_id(fund_row.get("subfundId"))
            if fund_key is None:
                continue
            raw_text = fund_row.get("benchmark")
            benchmark_text = raw_text.strip() if isinstance(raw_text, str) else ""
            parsed = parse_benchmark_text(benchmark_text)
            source = "kid"
            llm_used = False
            llm_result: dict[str, object] | None = None
            is_category_proxy = False
            resolution_reason = ""

            manual = manual_overrides.get(fund_key)
            if manual is not None:
                parsed = self._parsed_from_gemini(manual)
                source = "manual_override"
                is_category_proxy = manual["kind"] == "proxy"
                resolution_reason = str(manual.get("reason", "Manual override"))
            elif not benchmark_text or parsed.is_no_benchmark:
                proposal = (
                    self.gemini.propose_proxy(
                        name=str(fund_row.get("knf_name", fund_row.get("name", ""))),
                        category=str(fund_row.get("category", "")),
                        classification=str(fund_row.get("classification", "")),
                        risk_level=str(fund_row.get("riskLevel", "")),
                    )
                    if self.gemini is not None
                    else None
                )
                if proposal is not None:
                    llm_result = proposal
                    parsed = self._parsed_from_gemini(proposal)
                    source = "gemini_proxy"
                    is_category_proxy = True
                    resolution_reason = str(proposal.get("reason", ""))
                else:
                    parsed = self._category_proxy(fund_row) or ParsedBenchmark(
                        (),
                        ("no official benchmark or category proxy available",),
                        0.0,
                        True,
                    )
                    source = "category_proxy" if parsed.components else "none"
                    is_category_proxy = bool(parsed.components)
                    resolution_reason = (
                        "Category proxy; KID does not define a benchmark."
                        if is_category_proxy
                        else "No official benchmark or category proxy is available."
                    )
            elif parsed.unresolved and self.gemini is not None:
                resolved = self.gemini.resolve_kid_text(benchmark_text)
                if resolved is not None:
                    llm_result = resolved
                    parsed = self._parsed_from_gemini(resolved)
                    source = "gemini_kid"
                    llm_used = True
                    resolution_reason = str(resolved.get("reason", ""))

            frequency = (
                "weekly"
                if any(
                    INDEX_CATALOG[component.index_key].currency != "PLN"
                    for component in parsed.components
                )
                else "daily"
            )
            built = self.benchmark_builder.build(parsed, frequency=frequency)
            if built.quality in {"unresolved", "none", "unavailable"}:
                quality = "unresolved"
            elif is_category_proxy or built.quality == "proxy":
                quality = "proxy"
            elif llm_used:
                quality = "kid_partial"
            else:
                quality = "kid_full"
            is_proxy = quality == "proxy"
            benchmark_data[fund_key] = {
                "prices": built.prices,
                "source": source,
                "series_source": built.source,
                "quality": quality,
                "frequency": frequency,
                "coverage": built.coverage,
                "components": ";".join(
                    f"{component.index_key}:{component.weight:.1%}"
                    f"{f"+{component.spread_bps:g}bp" if component.spread_bps else ''}"
                    f"{':hedged_to_pln' if component.hedged_to_pln else ''}"
                    for component in parsed.components
                ),
                "unresolved": " | ".join(built.unresolved or parsed.unresolved),
                "is_proxy": is_proxy,
                "rankable": (
                    quality in {"kid_full", "kid_partial"}
                    and not is_proxy
                    and built.coverage >= 0.80
                ),
                "reason": resolution_reason or (str(llm_result.get("reason", "")) if llm_result else ""),
            }
        return benchmark_data

    def evaluate_fund(
        self,
        fund_row: pd.Series,
        prices: pd.Series,
        peer_benchmarks: dict[str, dict[str, object]],
        fund_benchmarks: dict[str, dict[str, object]],
    ) -> list[dict]:
        """Calculate absolute and benchmark-relative metrics for one fund."""
        category = fund_row.get("category")
        fund_key = self._normalize_id(fund_row.get("subfundId")) or ""
        benchmark_data = fund_benchmarks.get(fund_key, {})
        benchmark_prices = benchmark_data.get("prices")
        frequency = str(benchmark_data.get("frequency", "daily"))
        periods_per_year = 52 if frequency == "weekly" else 252
        peer_data = peer_benchmarks.get(fund_key, {})
        peer_log_returns = peer_data.get("log_returns")
        peer_count_value = peer_data.get("n_peers", 0)
        peer_count = (
            int(peer_count_value)
            if isinstance(peer_count_value, (int, np.integer))
            else 0
        )

        results = []
        for win_name, win_days in WINDOWS.items():
            abs_m = FundPerformanceEngine.compute_absolute_metrics(prices, win_days)
            if abs_m["insufficient"]:
                continue
            abs_m.pop("_log_ret")
            window_prices = prices if win_days is None else prices.iloc[-win_days:]
            if frequency == "weekly":
                fund_relative_prices = window_prices.resample("W-FRI").last().dropna()
            else:
                fund_relative_prices = window_prices

            fund_relative_log = pd.Series(dtype=float)
            if isinstance(benchmark_prices, pd.Series):
                relative_index = fund_relative_prices.index.intersection(benchmark_prices.index).sort_values()
                fund_window = fund_relative_prices.loc[relative_index]
                benchmark_window = benchmark_prices.loc[relative_index]
                fund_relative_log = pd.Series(
                    np.log(fund_window.to_numpy()),
                    index=relative_index,
                ).diff().dropna()
                benchmark_relative_log = pd.Series(
                    np.log(benchmark_window.to_numpy()),
                    index=relative_index,
                ).diff().dropna()
                relative_metrics = FundPerformanceEngine.compute_relative_metrics(
                    fund_relative_log,
                    benchmark_relative_log,
                    periods_per_year=periods_per_year,
                )
                rolling_window = periods_per_year
                relative_metrics.update(
                    FundPerformanceEngine.compute_rolling_ir(
                        fund_relative_log,
                        benchmark_relative_log,
                        window=rolling_window,
                        step=max(1, rolling_window // 4),
                        periods_per_year=periods_per_year,
                    ),
                )
            else:
                relative_metrics = FundPerformanceEngine.compute_relative_metrics(
                    pd.Series(dtype=float),
                    pd.Series(dtype=float),
                    periods_per_year=periods_per_year,
                )
                relative_metrics.update({"pct_ir_positive": np.nan, "n_ir_windows": 0})

            peer_ir = np.nan
            if isinstance(peer_log_returns, pd.Series) and not peer_log_returns.empty:
                peer_window = peer_log_returns
                if frequency == "weekly":
                    peer_window = peer_window.resample("W-FRI").sum()
                peer_metrics = FundPerformanceEngine.compute_relative_metrics(
                    fund_relative_log,
                    peer_window,
                    periods_per_year=periods_per_year,
                )
                peer_ir = peer_metrics["ir"]

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
                **relative_metrics,
                "peer_ir": peer_ir,
                "category_funds": peer_count,
                "category_peers_ok": peer_count >= MIN_CATEGORY_PEERS,
                "benchmark_source": benchmark_data.get("source", "none"),
                "benchmark_series_source": benchmark_data.get("series_source", "unavailable"),
                "benchmark_quality": benchmark_data.get("quality", "unresolved"),
                "benchmark_frequency": frequency,
                "benchmark_coverage": benchmark_data.get("coverage", 0.0),
                "benchmark_components": benchmark_data.get("components", ""),
                "benchmark_unresolved": benchmark_data.get("unresolved", ""),
                "benchmark_reason": benchmark_data.get("reason", ""),
                "benchmark_is_proxy": bool(benchmark_data.get("is_proxy", False)),
                "benchmark_rankable": bool(benchmark_data.get("rankable", False)),
            }

            if win_name == "all" and isinstance(benchmark_prices, pd.Series):
                hit_periods = {"1m": 4, "3m": 13, "1y": 52} if frequency == "weekly" else {
                    "1m": 22,
                    "3m": 66,
                    "1y": 252,
                }
                for label, period in hit_periods.items():
                    fund_return = fund_relative_prices.pct_change(periods=period, fill_method=None)
                    benchmark_return = benchmark_prices.reindex(fund_relative_prices.index)
                    benchmark_return = benchmark_return.pct_change(periods=period, fill_method=None)
                    common_returns = fund_return.dropna().index.intersection(
                        benchmark_return.dropna().index,
                    )
                    if len(common_returns):
                        hit_rate = (
                            fund_return.loc[common_returns] > benchmark_return.loc[common_returns]
                        ).mean()
                        row_data[f"hit_fund_benchmark_{label}"] = round(float(hit_rate * 100), 1)
                    else:
                        row_data[f"hit_fund_benchmark_{label}"] = np.nan

            if win_name == "all":
                for b_name, b_prices in self.benchmarks.items():
                    if b_prices is not None:
                        win_rates, _, _ = BenchmarkComparator.compare_fund_to_benchmark(
                            prices,
                            b_prices,
                        )
                        row_data[f"hit_{b_name}_22d"] = round(win_rates.get("22-day", np.nan), 1)
                        row_data[f"hit_{b_name}_66d"] = round(win_rates.get("66-day", np.nan), 1)
                        row_data[f"hit_{b_name}_252d"] = round(win_rates.get("252-day", np.nan), 1)

            results.append(row_data)
        return results

    @staticmethod
    def rank_results(results_df: pd.DataFrame) -> pd.DataFrame:
        rank_df = results_df.copy()
        rankable = rank_df.get("benchmark_rankable", pd.Series(False, index=rank_df.index))
        rankable = rankable.fillna(False).astype(bool)

        for metric in ("ir", "appraisal_ratio"):
            rank_col = f"rank_{metric}"
            rank_df[rank_col] = np.nan
            if metric not in rank_df.columns:
                continue
            eligible = rank_df.loc[rankable & rank_df[metric].notna()]
            if not eligible.empty:
                rank_df.loc[eligible.index, rank_col] = eligible.groupby(
                    ["category", "window"],
                )[metric].rank(method="min", ascending=False)

        for metric in ("sharpe", "sortino", "calmar", "ann_return"):
            rank_col = f"rank_{metric}"
            if metric in rank_df.columns:
                rank_df[rank_col] = rank_df.groupby(["category", "window"])[metric].rank(
                    method="min",
                    ascending=False,
                )
            else:
                rank_df[rank_col] = np.nan
        return rank_df

    def generate_reports(self) -> None:

        df_funds, fund_prices = self.load_confirmed_funds()
        peer_benchmarks = self.build_category_benchmarks(df_funds, fund_prices)
        fund_benchmarks = self.build_fund_benchmarks(df_funds)

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
                        peer_benchmarks=peer_benchmarks,
                        fund_benchmarks=fund_benchmarks,
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

        rank_df = self.rank_results(final_df)

        # Enforce exact columns and order
        for col in FINAL_COLS:
            if col not in rank_df.columns:
                rank_df[col] = np.nan

        # Append hit rate columns at the end of ranking if they exist
        other_cols = [c for c in rank_df.columns if c.startswith("hit_")]
        rank_df = rank_df[FINAL_COLS + other_cols]

        today = dt.today().strftime("%Y%m%d")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        perf_path, rank_path = (
            OUTPUT_DIR / f"fund_performance_{today}.csv",
            OUTPUT_DIR / f"fund_ranking_{today}.csv",
        )
        review_path = OUTPUT_DIR / f"fund_benchmark_review_{today}.csv"
        review_columns = [
            "subfundId",
            "knf_name",
            "category",
            "hasBenchmark",
            "benchmark",
            "benchmark_source",
            "benchmark_series_source",
            "benchmark_quality",
            "benchmark_frequency",
            "benchmark_coverage",
            "benchmark_components",
            "benchmark_unresolved",
            "benchmark_reason",
            "benchmark_is_proxy",
            "benchmark_rankable",
        ]
        review_df = rank_df.loc[~rank_df["benchmark_rankable"].fillna(False), review_columns]
        review_df = review_df.drop_duplicates(subset=["subfundId"], keep="first")

        final_df.to_csv(perf_path, index=False, sep=";")
        rank_df.to_csv(rank_path, index=False, sep=";")
        review_df.to_csv(review_path, index=False, sep=";")

        logging.info("Generated ranking with %d rows: %s", len(rank_df), rank_path)
        logging.info("Benchmark review queue contains %d funds: %s", len(review_df), review_path)

        if self.folder_id:
            self.gdrive.upload_csv(self.folder_id, str(perf_path))
            self.gdrive.upload_csv(self.folder_id, str(rank_path))
            self.gdrive.upload_csv(self.folder_id, str(review_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="TFI Comprehensive Reviewer")
    parser.add_argument("--update", action="store_true")
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use GEMINI_API_KEY for unresolved KID benchmarks and category proxy proposals.",
    )
    args = parser.parse_args()
    os.chdir(project_root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    reviewer = FundReviewer(update_data=args.update, use_gemini=args.use_gemini)
    reviewer.generate_reports()


if __name__ == "__main__":
    main()
