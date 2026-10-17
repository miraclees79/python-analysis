from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from moj_system.core.benchmark_builder import BenchmarkBuilder, BuiltBenchmark
from moj_system.core.benchmark_llm import GeminiBenchmarkResolver
from moj_system.core.benchmark_parser import parse_benchmark_text
from moj_system.core.fund_analytics import FundPerformanceEngine
from moj_system.scripts.fund_reviewer import FundReviewer


class BenchmarkParserTests(unittest.TestCase):
    def test_parses_decimal_comma_and_long_index_aliases(self) -> None:
        parsed = parse_benchmark_text(
            "42,5% WIG20TR + 42,5% MSCI World USD Index + 15% Treasury BondSpot Poland Index (TBSP)",
        )

        self.assertTrue(parsed.is_resolved)
        self.assertEqual(
            [component.index_key for component in parsed.components],
            ["WIG20TR", "MSCI_WORLD", "TBSP"],
        )
        self.assertAlmostEqual(parsed.weight_sum, 1.0)

    def test_does_not_mistake_rate_spread_for_component_weight(self) -> None:
        parsed = parse_benchmark_text("90% WIG + 10% (WIBOR 6M + 0,6%)")

        self.assertTrue(parsed.is_resolved)
        self.assertEqual([component.index_key for component in parsed.components], ["WIG", "WIBOR_6M"])
        self.assertAlmostEqual(parsed.components[1].spread_bps, 60.0)

    def test_unparenthesized_tenored_rate_keeps_its_spread(self) -> None:
        parsed = parse_benchmark_text("90% WIG + 10% WIBOR1M + 0.5%")

        self.assertEqual(parsed.portfolio_spread_bps, 0.0)
        self.assertAlmostEqual(parsed.components[-1].spread_bps, 50.0)

    def test_rate_spread_does_not_hide_following_weighted_component(self) -> None:
        parsed = parse_benchmark_text("90% (WIBOR 6M + 30 bp) + 10% POLSTR")

        self.assertTrue(parsed.is_resolved)
        self.assertEqual([component.index_key for component in parsed.components], ["WIBOR_6M", "POLSTR"])
        self.assertAlmostEqual(parsed.components[0].spread_bps, 30.0)

    def test_parses_portfolio_wide_spread_separately(self) -> None:
        parsed = parse_benchmark_text("75% GPWB-BWZ + 25% ICE BofA 1-3 Year Poland Government Index + 20 bp")

        self.assertTrue(parsed.is_resolved)
        self.assertAlmostEqual(parsed.portfolio_spread_bps, 20.0)
        self.assertEqual(parsed.components[-1].spread_bps, 0.0)

    def test_parses_terminal_percentage_spread_after_full_weight_sum(self) -> None:
        parsed = parse_benchmark_text("80% GPWB-BWZ + 20% GPWB-B1Y3Y + 0,45%")

        self.assertTrue(parsed.is_resolved)
        self.assertEqual([component.index_key for component in parsed.components], ["GPWB_BWZ", "GPWB_B1Y3Y"])
        self.assertAlmostEqual(parsed.portfolio_spread_bps, 45.0)

    def test_parses_single_rate_with_basis_point_spread(self) -> None:
        parsed = parse_benchmark_text("WIBOR 3M (WIBO3M Ask Price) powiększony o 50 punktów bazowych")

        self.assertTrue(parsed.is_resolved)
        self.assertEqual(parsed.components[0].index_key, "WIBOR_3M")
        self.assertAlmostEqual(parsed.components[0].spread_bps, 50.0)

    def test_marks_no_benchmark_explicitly(self) -> None:
        parsed = parse_benchmark_text("NIE POSIADA BENCHMARKU")

        self.assertTrue(parsed.is_no_benchmark)
        self.assertFalse(parsed.components)

    def test_keeps_unrecognized_weighted_component_unresolved(self) -> None:
        parsed = parse_benchmark_text("70% nieznany indeks obligacyjny + 30% WIG")

        self.assertFalse(parsed.is_resolved)
        self.assertEqual(parsed.unresolved[0], "nieznany indeks obligacyjny")


class BenchmarkBuilderTests(unittest.TestCase):
    def setUp(self) -> None:  # noqa: N802
        self.dates = pd.date_range("2025-01-02", periods=2, freq="B")

    def test_composite_converts_foreign_leg_to_pln_and_rebalances_daily(self) -> None:
        series = {
            "wig": [100.0, 110.0],
            "msci_world_combined": [100.0, 110.0],
            "usdpln": [4.0, 4.4],
        }

        def load_local(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame:
            del label, mandatory
            return pd.DataFrame({"Zamkniecie": series[ticker]}, index=self.dates)

        with patch("moj_system.core.benchmark_builder.load_local_csv", side_effect=load_local):
            result = BenchmarkBuilder().build(parse_benchmark_text("50% WIG + 50% MSCI World USD Index"))

        self.assertIsNotNone(result.prices)
        self.assertAlmostEqual(float(result.prices.iloc[-1]), 115.5)
        self.assertEqual(result.quality, "proxy")
        self.assertAlmostEqual(result.coverage, 1.0)

    def test_rate_leg_uses_actual_365_accrual_and_component_spread(self) -> None:
        series = {
            "wig": [100.0, 110.0],
            "wibor1m": [12.0, 12.0],
        }

        def load_local(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame:
            del label, mandatory
            return pd.DataFrame({"Zamkniecie": series[ticker]}, index=self.dates)

        with patch("moj_system.core.benchmark_builder.load_local_csv", side_effect=load_local):
            result = BenchmarkBuilder().build(parse_benchmark_text("90% WIG + 10% WIBOR1M + 0.5%"))

        self.assertIsNotNone(result.prices)
        expected_rate_return = math.expm1(0.125 / 365.0)
        expected = 100.0 * (1.0 + 0.9 * 0.1 + 0.1 * expected_rate_return)
        self.assertAlmostEqual(float(result.prices.iloc[-1]), expected)
        self.assertEqual(result.quality, "exact")

    def test_missing_fx_returns_unresolved_instead_of_partial_benchmark(self) -> None:
        def load_local(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame | None:
            del label, mandatory
            if ticker == "msci_world_combined":
                return pd.DataFrame({"Zamkniecie": [100.0, 101.0]}, index=self.dates)
            return None

        with patch("moj_system.core.benchmark_builder.load_local_csv", side_effect=load_local):
            result = BenchmarkBuilder(yfinance_downloader=Mock(return_value=None)).build(
                parse_benchmark_text("100% MSCI World USD Index"),
            )

        self.assertIsNone(result.prices)
        self.assertIn("no USD/PLN FX series", result.unresolved[0])

    def test_foreign_benchmark_can_be_resampled_to_friday_weekly_prices(self) -> None:
        dates = pd.bdate_range("2025-01-06", periods=10)
        series = {
            "msci_world_combined": [100.0 + value for value in range(10)],
            "usdpln": [4.0 + value * 0.01 for value in range(10)],
        }

        def load_local(ticker: str, label: str, mandatory: bool = True) -> pd.DataFrame:
            del label, mandatory
            return pd.DataFrame({"Zamkniecie": series[ticker]}, index=dates)

        with patch("moj_system.core.benchmark_builder.load_local_csv", side_effect=load_local):
            result = BenchmarkBuilder().build(
                parse_benchmark_text("100% MSCI World USD Index"),
                frequency="weekly",
            )

        self.assertIsNotNone(result.prices)
        self.assertEqual(result.frequency, "weekly")
        self.assertTrue((result.prices.index.dayofweek == 4).all())


class GeminiBenchmarkResolverTests(unittest.TestCase):
    def test_validates_llm_output_and_reuses_cache(self) -> None:
        response = Mock()
        response.text = json.dumps(
            {
                "components": [
                    {
                        "index_key": "WIG",
                        "weight": 0.9,
                        "spread_bps": 0,
                        "hedged_to_pln": False,
                    },
                    {
                        "index_key": "POLSTR",
                        "weight": 0.1,
                        "spread_bps": 0,
                        "hedged_to_pln": False,
                    },
                ],
                "confidence": 0.9,
                "reason": "Local equity with cash leg.",
                "portfolio_spread_bps": 0,
            },
        )
        client = Mock()
        client.models.generate_content.return_value = response

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "benchmark-cache.json"
            first = GeminiBenchmarkResolver(api_key="test-key", cache_path=cache_path, client=client)
            resolved = first.resolve_kid_text("90% WIG and 10% POLSTR")
            second_client = Mock()
            second = GeminiBenchmarkResolver(api_key="test-key", cache_path=cache_path, client=second_client)
            cached = second.resolve_kid_text("90% WIG and 10% POLSTR")

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["components"][0]["index_key"], "WIG")
        self.assertEqual(cached["components"], resolved["components"])
        client.models.generate_content.assert_called_once()
        second_client.models.generate_content.assert_not_called()

    def test_rejects_unknown_catalog_key_and_bad_weight_sum(self) -> None:
        unknown = GeminiBenchmarkResolver._validate_response(
            {
                "components": [
                    {"index_key": "NOT_A_TICKER", "weight": 1.0, "spread_bps": 0, "hedged_to_pln": False},
                ],
                "confidence": 0.9,
                "reason": "unknown index",
            },
        )
        bad_sum = GeminiBenchmarkResolver._validate_response(
            {
                "components": [
                    {"index_key": "WIG", "weight": 0.7, "spread_bps": 0, "hedged_to_pln": False},
                ],
                "confidence": 0.9,
                "reason": "weights do not add up",
            },
        )

        self.assertIsNone(unknown)
        self.assertIsNone(bad_sum)

    def test_disabled_resolver_does_not_call_gemini(self) -> None:
        client = Mock()
        resolver = GeminiBenchmarkResolver(api_key="", client=None)
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            result = resolver.resolve_kid_text("100% WIG")

        self.assertIsNone(result)
        client.models.generate_content.assert_not_called()


class RelativeFundMetricTests(unittest.TestCase):
    def test_information_ratio_uses_active_return_over_tracking_error(self) -> None:
        dates = pd.date_range("2025-01-01", periods=60, freq="B")
        benchmark_returns = pd.Series([0.01, -0.005, 0.002, -0.001] * 15, index=dates)
        active_returns = pd.Series([0.004, -0.001, 0.002, 0.0] * 15, index=dates)
        fund_returns = benchmark_returns + active_returns

        metrics = FundPerformanceEngine.compute_relative_metrics(
            np.log1p(fund_returns),
            np.log1p(benchmark_returns),
        )

        expected_ir = np.sqrt(252) * active_returns.mean() / active_returns.std(ddof=1)
        expected_te = active_returns.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(metrics["ir"], round(float(expected_ir), 4))
        self.assertAlmostEqual(metrics["tracking_error"], round(float(expected_te), 4))
        self.assertIn("appraisal_ratio", metrics)
        self.assertIn("correlation", metrics)
        self.assertEqual(metrics["n_relative_observations"], 60)

    def test_ir_ranking_excludes_proxy_benchmarks(self) -> None:
        results = pd.DataFrame(
            {
                "category": ["akcji polskich", "akcji polskich"],
                "window": ["all", "all"],
                "benchmark_rankable": [True, False],
                "ir": [0.5, 2.0],
                "appraisal_ratio": [0.4, 3.0],
                "sharpe": [1.0, 2.0],
                "sortino": [1.0, 2.0],
                "calmar": [0.5, 1.0],
                "ann_return": [0.05, 0.1],
            },
        )

        ranked = FundReviewer.rank_results(results)

        self.assertEqual(ranked.loc[0, "rank_ir"], 1.0)
        self.assertTrue(pd.isna(ranked.loc[1, "rank_ir"]))
        self.assertEqual(ranked.loc[0, "rank_sharpe"], 2.0)
        self.assertEqual(ranked.loc[1, "rank_sharpe"], 1.0)

    def test_fund_ir_uses_its_own_benchmark_and_reports_peers_separately(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        benchmark_returns = np.tile([0.002, -0.001, 0.001, -0.0005], 75)
        active_returns = np.tile([0.001, -0.0002, 0.0005, -0.0001], 75)
        benchmark_prices = pd.Series(100.0 * np.cumprod(1.0 + benchmark_returns), index=dates)
        fund_prices = pd.Series(
            100.0 * np.cumprod(1.0 + benchmark_returns + active_returns),
            index=dates,
        )
        peer_log_returns = pd.Series(np.log1p(-benchmark_returns), index=dates)
        reviewer = FundReviewer.__new__(FundReviewer)
        reviewer.benchmarks = {}

        rows = reviewer.evaluate_fund(
            fund_row=pd.Series(
                {
                    "subfundId": 1,
                    "stooq_id": 101,
                    "category": "akcji polskich",
                    "benchmark": "100% WIG",
                },
            ),
            prices=fund_prices,
            peer_benchmarks={"1": {"log_returns": peer_log_returns, "n_peers": 4}},
            fund_benchmarks={
                "1": {
                    "prices": benchmark_prices,
                    "source": "kid",
                    "series_source": "local",
                    "quality": "kid_full",
                    "frequency": "daily",
                    "coverage": 1.0,
                    "components": "WIG:100.0%",
                    "unresolved": "",
                    "is_proxy": False,
                    "rankable": True,
                    "reason": "",
                },
            },
        )

        one_year = next(row for row in rows if row["window"] == "1y")
        self.assertEqual(one_year["benchmark_source"], "kid")
        self.assertEqual(one_year["benchmark_quality"], "kid_full")
        self.assertEqual(one_year["category_funds"], 4)
        self.assertEqual(one_year["benchmark_components"], "WIG:100.0%")
        self.assertTrue(np.isfinite(one_year["ir"]))
        self.assertTrue(np.isfinite(one_year["peer_ir"]))

    def test_leave_one_out_category_benchmark_excludes_same_stooq_series(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=3)
        prices = {
            "101": pd.Series([100.0, 120.0, 100.0], index=dates),
            "102": pd.Series([100.0, 90.0, 95.0], index=dates),
            "103": pd.Series([100.0, 110.0, 121.0], index=dates),
        }
        funds = pd.DataFrame(
            {
                "subfundId": [1, 2, 3, 4],
                "stooq_id": [101, 101, 102, 103],
                "category": ["akcji polskich"] * 4,
            },
        )
        reviewer = FundReviewer.__new__(FundReviewer)

        peers = reviewer.build_category_benchmarks(funds, prices)

        self.assertEqual(peers["1"]["n_peers"], 2)
        self.assertEqual(peers["2"]["n_peers"], 2)
        expected = (
            np.log(prices["102"] / prices["102"].shift(1)).dropna()
            + np.log(prices["103"] / prices["103"].shift(1)).dropna()
        ) / 2.0
        pd.testing.assert_series_equal(peers["1"]["log_returns"], expected, check_names=False)

    def test_missing_kid_benchmark_gets_explicit_proxy_but_unknown_kid_stays_unresolved(self) -> None:
        reviewer = FundReviewer.__new__(FundReviewer)
        reviewer.gemini = None
        reviewer.benchmark_builder = Mock(
            build=Mock(
                side_effect=[
                    BuiltBenchmark(
                        prices=pd.Series([100.0, 101.0]),
                        source="local",
                        quality="proxy",
                        frequency="daily",
                        coverage=1.0,
                        component_keys=("WIG",),
                    ),
                    BuiltBenchmark(
                        prices=None,
                        source="unresolved",
                        quality="unresolved",
                        frequency="daily",
                        coverage=0.0,
                        component_keys=(),
                        unresolved=("no series for FUTURE_INDEX",),
                    ),
                ],
            ),
        )
        funds = pd.DataFrame(
            {
                "subfundId": [1, 2],
                "category": ["akcji polskich", "akcji polskich"],
                "classification": ["akcji", "akcji"],
                "riskLevel": [4.0, 4.0],
                "benchmark": ["NIE POSIADA BENCHMARKU", "100% FUTURE_INDEX"],
            },
        )

        results = reviewer.build_fund_benchmarks(funds)

        self.assertEqual(results["1"]["source"], "category_proxy")
        self.assertEqual(results["1"]["quality"], "proxy")
        self.assertFalse(results["1"]["rankable"])
        self.assertEqual(results["2"]["quality"], "unresolved")
        self.assertFalse(results["2"]["is_proxy"])

    def test_manual_override_takes_precedence_over_kid_text(self) -> None:
        reviewer = FundReviewer.__new__(FundReviewer)
        reviewer.gemini = None
        reviewer.benchmark_builder = Mock(
            build=Mock(
                return_value=BuiltBenchmark(
                    prices=pd.Series([100.0, 101.0]),
                    source="local",
                    quality="exact",
                    frequency="daily",
                    coverage=1.0,
                    component_keys=("WIG",),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer.benchmark_overrides_path = Path(temp_dir) / "overrides.csv"
            pd.DataFrame(
                {
                    "subfundId": [7],
                    "components_json": [
                        '[{"index_key":"WIG","weight":1.0,"spread_bps":0,"hedged_to_pln":false}]',
                    ],
                    "kind": ["official"],
                    "notes": ["Verified against prospectus"],
                    "portfolio_spread_bps": [""],
                },
            ).to_csv(reviewer.benchmark_overrides_path, sep=";", index=False)
            result = reviewer.build_fund_benchmarks(
                pd.DataFrame(
                    {
                        "subfundId": [7],
                        "category": ["akcji zagranicznych"],
                        "benchmark": ["100% FUTURE_INDEX"],
                    },
                ),
            )["7"]

        self.assertEqual(result["source"], "manual_override")
        self.assertEqual(result["quality"], "kid_full")
        self.assertEqual(result["reason"], "Verified against prospectus")


if __name__ == "__main__":
    unittest.main()
