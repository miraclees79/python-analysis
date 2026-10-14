from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from moj_system.config import PORTFOLIO_THRESHOLDS_MC
from moj_system.core.portfolio_robustness import (
    PortfolioLeg,
    analyze_portfolio_robustness,
    add_ohlc_columns_for_context,
    bootstrap_joint_ohlc_panel,
    build_portfolio_parameter_grids,
    run_portfolio_block_bootstrap,
    run_portfolio_monte_carlo,
    sample_portfolio_parameter_universe,
)


def _ohlc_frame(index: pd.DatetimeIndex, offset: float) -> pd.DataFrame:
    close = pd.Series(np.arange(len(index), dtype=float) + 100.0 + offset, index=index)
    range_size = close * (pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index) / 100.0)
    return pd.DataFrame(
        {
            "Zamkniecie": close,
            "Najwyzszy": close + range_size,
            "Najnizszy": close - range_size,
        },
        index=index,
    )


def _params() -> dict[int, dict]:
    return {
        0: {
            "filter_mode": "ma",
            "X": 0.10,
            "Y": 0.03,
            "fast": 50,
            "slow": 200,
            "stop_loss": 0.05,
            "use_atr_stop": False,
        },
    }


class PortfolioRobustnessTests(unittest.TestCase):
    def test_joint_bootstrap_is_reproducible_and_keeps_common_dates(self) -> None:
        index = pd.date_range("2020-01-01", periods=20, freq="B")
        frames = {
            "A": _ohlc_frame(index, 0.0),
            "B": _ohlc_frame(index, 50.0),
        }
        cash = _ohlc_frame(index, 100.0)

        first = bootstrap_joint_ohlc_panel(frames, cash, block_size=5, seed=7)
        second = bootstrap_joint_ohlc_panel(frames, cash, block_size=5, seed=7)

        pd.testing.assert_frame_equal(first[0]["A"], second[0]["A"])
        pd.testing.assert_frame_equal(first[0]["B"], second[0]["B"])
        pd.testing.assert_frame_equal(first[1], second[1])
        self.assert_index_equal(first[0]["A"].index, index)
        a_range_ratio = (
            first[0]["A"]["Najwyzszy"] - first[0]["A"]["Najnizszy"]
        ) / first[0]["A"]["Zamkniecie"]
        b_range_ratio = (
            first[0]["B"]["Najwyzszy"] - first[0]["B"]["Najnizszy"]
        ) / first[0]["B"]["Zamkniecie"]
        pd.testing.assert_series_equal(a_range_ratio, b_range_ratio)

    def test_parameter_universe_contains_all_legs(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        legs = [
            PortfolioLeg("A", _params(), _ohlc_frame(index, 0.0)),
            PortfolioLeg("B", _params(), _ohlc_frame(index, 10.0)),
        ]
        grids = build_portfolio_parameter_grids(legs)
        sampled = sample_portfolio_parameter_universe(grids, seed=42)

        self.assertEqual(set(sampled), {"A", "B"})
        self.assertEqual(set(sampled["A"]), {0})
        self.assertEqual(set(sampled["B"]), {0})

    def test_close_only_bond_gate_context_gets_jointly_resampled(self) -> None:
        index = pd.date_range("2020-01-01", periods=20, freq="B")
        assets = {
            "WIG": _ohlc_frame(index, 0.0),
            "TBSP": _ohlc_frame(index, 10.0),
            "PL10Y": add_ohlc_columns_for_context(
                pd.DataFrame(
                    {"Zamkniecie": np.linspace(2.0, 3.0, len(index))},
                    index=index,
                ),
            ),
        }
        cash = _ohlc_frame(index, 100.0)

        synthetic, _ = bootstrap_joint_ohlc_panel(
            frames=assets,
            cash_df=cash,
            block_size=5,
            seed=3,
        )

        self.assertEqual(set(synthetic), set(assets))
        self.assertIn("Najwyzszy", synthetic["PL10Y"].columns)
        self.assertIn("Najnizszy", synthetic["PL10Y"].columns)
        self.assert_index_equal(synthetic["PL10Y"].index, index)

    def test_portfolio_bootstrap_handles_different_source_calendars(self) -> None:
        dates = pd.date_range("2020-01-06", periods=7, freq="B")
        frames = {
            "A": _ohlc_frame(dates[:6], 0.0),
            "B": _ohlc_frame(dates[1:], 50.0),
        }
        cash = _ohlc_frame(dates[::2], 100.0)

        def runner(
            sampled_frames: dict[str, pd.DataFrame],
            sampled_cash: pd.DataFrame,
        ) -> pd.Series:
            index = sampled_frames["A"].index
            return pd.Series(np.linspace(1.0, 1.05, len(index)), index=index)

        results = run_portfolio_block_bootstrap(
            frames=frames,
            cash_df=cash,
            runner=runner,
            n_samples=1,
            block_size=2,
            n_jobs=1,
        )

        self.assertEqual(len(results), 1)
        self.assertIn("MaxDD", results.columns)

    def test_portfolio_robustness_logs_every_configured_criterion(self) -> None:
        results = pd.DataFrame(
            {
                "CAGR": [0.01, 0.02, 0.03],
                "Sharpe": [0.1, 0.2, 0.3],
                "MaxDD": [-0.30, -0.25, -0.22],
            },
        )

        with self.assertLogs(level="INFO") as captured:
            summary = analyze_portfolio_robustness(
                results_df=results,
                baseline_metrics={},
                thresholds=PORTFOLIO_THRESHOLDS_MC,
                label="TEST PORTFOLIO MC",
            )

        report = "\n".join(captured.output)
        self.assertEqual(summary["verdict"], "FRAGILE")
        self.assertIn("PASS portfolio p05 CAGR > 0%", report)
        self.assertIn("PASS portfolio p05 Sharpe > 0", report)
        self.assertIn("FAIL portfolio p05 MaxDD > -20%", report)

    def test_monte_carlo_runs_end_to_end_callback(self) -> None:
        index = pd.date_range("2020-01-01", periods=10, freq="B")
        legs = [
            PortfolioLeg("A", _params(), _ohlc_frame(index, 0.0)),
            PortfolioLeg("B", _params(), _ohlc_frame(index, 10.0)),
        ]

        def runner(universe: dict[str, dict[int, dict]]) -> pd.Series:
            self.assertEqual(set(universe), {"A", "B"})
            return pd.Series(np.linspace(1.0, 1.2, len(index)), index=index)

        result = run_portfolio_monte_carlo(
            legs=legs,
            runner=runner,
            n_samples=3,
            n_jobs=2,
        )

        self.assertEqual(len(result), 3)
        self.assertIn("CAGR", result.columns)
        self.assertIn("MaxDD", result.columns)

    @staticmethod
    def assert_index_equal(
        actual: pd.DatetimeIndex,
        expected: pd.DatetimeIndex,
    ) -> None:
        pd.testing.assert_index_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
