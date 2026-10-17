"""Portfolio-level robustness engines.

The component-level engines remain in ``robustness_engine``.  This module
provides independent orchestration for end-to-end portfolio simulations:

* one Monte Carlo draw samples a parameter universe for every portfolio leg;
* one bootstrap draw samples the same time blocks for every leg and the cash
  series, preserving cross-asset dependence;
* portfolio construction is supplied by a callback so the existing
  allocation and reallocation rules are reused rather than duplicated.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from moj_system.core.pension_engine import build_signal_series
from moj_system.core.robustness_engine import (
    build_perturbation_grid,
    compute_metrics,
    extract_windows_from_wf_results,
    run_universe,
)

PortfolioParams = dict[int, dict]
ParameterUniverses = dict[str, dict[int, dict]]
PortfolioFrames = dict[str, pd.DataFrame]
PortfolioRunner = Callable[[ParameterUniverses], pd.Series | None]
BootstrapRunner = Callable[[PortfolioFrames, pd.DataFrame], pd.Series | None]


@dataclass(frozen=True)
class PortfolioLeg:
    """Input required to perturb one strategy leg."""

    name: str
    best_params: PortfolioParams
    frame: pd.DataFrame
    price_col: str = "Zamkniecie"


def build_portfolio_parameter_grids(
    legs: Sequence[PortfolioLeg],
    perturb_pct: float = 0.20,
) -> dict[str, dict[int, list[dict]]]:
    """Build independent, per-leg perturbation grids."""
    return {
        leg.name: {
            window_id: build_perturbation_grid(params, pct=perturb_pct)
            for window_id, params in leg.best_params.items()
        }
        for leg in legs
    }


def sample_portfolio_parameter_universe(
    grids: Mapping[str, Mapping[int, Sequence[dict]]],
    seed: int,
) -> ParameterUniverses:
    """Sample all legs with one reproducible random stream."""
    rng = random.Random(seed)
    universe: ParameterUniverses = {}
    for leg_name, window_grids in grids.items():
        universe[leg_name] = {
            window_id: dict(rng.choice(list(variants)))
            for window_id, variants in window_grids.items()
        }
    return universe


def add_ohlc_columns_for_context(frame: pd.DataFrame, price_col: str = "Zamkniecie") -> pd.DataFrame:
    """Add flat OHLC columns to a close-only context series for joint resampling."""
    if price_col not in frame.columns:
        raise ValueError(f"Context frame is missing {price_col!r}.")
    result = frame.copy()
    if "Najwyzszy" not in result.columns:
        result["Najwyzszy"] = result[price_col]
    if "Najnizszy" not in result.columns:
        result["Najnizszy"] = result[price_col]
    return result


def run_parameterized_signals(
    universe: ParameterUniverses,
    legs: Sequence[PortfolioLeg],
    wf_results: Mapping[str, pd.DataFrame],
    cash_df: pd.DataFrame,
    entry_gates: Mapping[str, pd.Series | None] | None = None,
) -> dict[str, pd.Series]:
    """Run every perturbed leg and convert its trades into binary signals."""
    gates = entry_gates or {}
    signals: dict[str, pd.Series] = {}
    for leg in legs:
        windows = extract_windows_from_wf_results(wf_results[leg.name])
        equity, trades = run_universe(
            universe=universe[leg.name],
            windows=windows,
            df=leg.frame,
            cash_df=cash_df,
            vol_window=20,
            selected_mode="full",
            price_col=leg.price_col,
            entry_gate_series=gates.get(leg.name),
        )
        if equity is None:
            raise ValueError(f"Parameterized simulation returned no equity for {leg.name}.")
        signals[leg.name] = build_signal_series(
            wf_equity=equity,
            wf_trades=pd.DataFrame(trades),
        )
    return signals


def _metrics_or_none(equity: pd.Series | None, sample: int) -> dict[str, float] | None:
    if equity is None or equity.empty:
        return None
    metrics = compute_metrics(equity=equity)
    return {
        "sample": float(sample),
        **{key: float(value) for key, value in metrics.items()},
    }


def _run_parameter_sample(
    sample: int,
    seed: int,
    grids: Mapping[str, Mapping[int, Sequence[dict]]],
    runner: PortfolioRunner,
) -> dict[str, float] | None:
    universe = sample_portfolio_parameter_universe(grids=grids, seed=seed)
    try:
        return _metrics_or_none(equity=runner(universe), sample=sample)
    except Exception:
        logging.exception("Portfolio MC sample %d failed.", sample)
        return None


def run_portfolio_monte_carlo(
    legs: Sequence[PortfolioLeg],
    runner: PortfolioRunner,
    n_samples: int = 1000,
    perturb_pct: float = 0.20,
    seed: int = 42,
    n_jobs: int = 1,
    sample_offset: int = 0,
) -> pd.DataFrame:
    """Run end-to-end portfolio parameter perturbation.

    ``runner`` must execute the complete portfolio process: component signal
    generation, allocation walk-forward, reallocation gate, and portfolio
    equity construction.  It receives one sampled parameter universe per
    trial.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if not legs:
        raise ValueError("At least one portfolio leg is required.")

    grids = build_portfolio_parameter_grids(legs=legs, perturb_pct=perturb_pct)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_parameter_sample)(
            sample=sample_offset + sample,
            seed=seed + sample_offset + sample,
            grids=grids,
            runner=runner,
        )
        for sample in range(n_samples)
    )
    valid = [result for result in results if result is not None]
    return pd.DataFrame(valid)


def _validate_bootstrap_frames(
    frames: Mapping[str, pd.DataFrame],
    cash_df: pd.DataFrame,
    price_col: str,
) -> pd.DatetimeIndex:
    if not frames:
        raise ValueError("At least one portfolio frame is required.")
    indexes = [frame.index for frame in frames.values()]
    indexes.append(cash_df.index)
    common_index = indexes[0]
    for index in indexes[1:]:
        common_index = common_index.union(index)
    common_index = common_index.sort_values()
    if common_index.empty:
        raise ValueError("Portfolio frames and cash series have no common dates.")
    for name, frame in frames.items():
        missing = {price_col, "Najwyzszy", "Najnizszy"} - set(frame.columns)
        if missing:
            raise ValueError(f"Portfolio frame {name!r} is missing OHLC columns: {sorted(missing)}")
    if price_col not in cash_df.columns:
        raise ValueError(f"Cash frame is missing {price_col!r}.")
    return common_index


def _joint_block_indices(n_rows: int, block_size: int, seed: int) -> np.ndarray:
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if block_size > n_rows:
        raise ValueError("block_size cannot exceed the common panel length.")
    starts = np.arange(0, n_rows - block_size + 1)
    n_blocks = math.ceil(n_rows / block_size)
    rng = np.random.default_rng(seed)
    sampled_starts = rng.choice(starts, size=n_blocks, replace=True)
    offsets = np.arange(block_size)
    indices = np.concatenate([start + offsets for start in sampled_starts])
    return indices[:n_rows]


def bootstrap_joint_ohlc_panel(
    frames: Mapping[str, pd.DataFrame],
    cash_df: pd.DataFrame,
    block_size: int = 250,
    seed: int = 42,
    price_col: str = "Zamkniecie",
) -> tuple[PortfolioFrames, pd.DataFrame]:
    """Resample full OHLC rows using one common sequence of time blocks.

    Prices are normalised to the first source observation before sampling and
    then reconstructed by chaining sampled close-to-close returns.  High and
    low are scaled by the same close ratio, preserving each sampled candle's
    intraday geometry while keeping prices continuous at block boundaries.
    """
    common_index = _validate_bootstrap_frames(frames, cash_df, price_col)
    aligned_frames = {
        name: frame.reindex(common_index).ffill()
        for name, frame in frames.items()
    }
    aligned_cash = cash_df.reindex(common_index).ffill()
    valid_mask = pd.Series(True, index=common_index)
    for frame in [*aligned_frames.values(), aligned_cash]:
        valid_mask &= frame[price_col].notna()
    valid_index = common_index[valid_mask.to_numpy()]
    common_index = valid_index
    if common_index.empty:
        raise ValueError("Portfolio OHLC panel has no fully populated dates.")
    positions = _joint_block_indices(len(common_index), block_size, seed)
    synthetic_frames: PortfolioFrames = {}

    for name in frames:
        source_panel = aligned_frames[name].loc[common_index].copy()
        source_close = source_panel[price_col].astype(float)
        source_returns = source_close.pct_change().fillna(0.0).to_numpy()
        sampled_returns = source_returns[positions]
        close = source_close.iloc[0] * np.cumprod(1.0 + sampled_returns)
        close_ratio = close / source_close.iloc[positions].to_numpy()
        synthetic = source_panel.iloc[positions].copy()
        synthetic[price_col] = close
        synthetic["Najwyzszy"] = source_panel["Najwyzszy"].to_numpy()[positions] * close_ratio
        synthetic["Najnizszy"] = source_panel["Najnizszy"].to_numpy()[positions] * close_ratio
        synthetic.index = common_index
        synthetic_frames[name] = synthetic

    cash_source = aligned_cash.loc[common_index].copy()
    cash_close = cash_source[price_col].astype(float)
    cash_returns = cash_close.pct_change().fillna(0.0).to_numpy()[positions]
    synthetic_cash = cash_source.iloc[positions].copy()
    synthetic_cash[price_col] = cash_close.iloc[0] * np.cumprod(1.0 + cash_returns)
    synthetic_cash.index = common_index
    return synthetic_frames, synthetic_cash


def _run_bootstrap_sample(
    sample: int,
    seed: int,
    frames: PortfolioFrames,
    cash_df: pd.DataFrame,
    block_size: int,
    runner: BootstrapRunner,
) -> dict[str, float] | None:
    try:
        synthetic_frames, synthetic_cash = bootstrap_joint_ohlc_panel(
            frames=frames,
            cash_df=cash_df,
            block_size=block_size,
            seed=seed,
        )
        return _metrics_or_none(
            equity=runner(synthetic_frames, synthetic_cash),
            sample=sample,
        )
    except Exception:
        logging.exception("Portfolio bootstrap sample %d failed.", sample)
        return None


def run_portfolio_block_bootstrap(
    frames: Mapping[str, pd.DataFrame],
    cash_df: pd.DataFrame,
    runner: BootstrapRunner,
    n_samples: int = 500,
    block_size: int = 250,
    seed: int = 1000,
    n_jobs: int = 1,
    price_col: str = "Zamkniecie",
    sample_offset: int = 0,
) -> pd.DataFrame:
    """Run end-to-end bootstrap with jointly sampled full OHLC blocks."""
    common_index = _validate_bootstrap_frames(frames, cash_df, price_col)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_bootstrap_sample)(
            sample=sample_offset + sample,
            seed=seed + sample_offset + sample,
            frames={name: frame.reindex(common_index) for name, frame in frames.items()},
            cash_df=cash_df.reindex(common_index),
            block_size=block_size,
            runner=runner,
        )
        for sample in range(n_samples)
    )
    valid = [result for result in results if result is not None]
    return pd.DataFrame(valid)


def analyze_portfolio_robustness(
    results_df: pd.DataFrame,
    baseline_metrics: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
    label: str,
) -> dict[str, object]:
    """Summarise a portfolio sample distribution without changing component gates."""
    if results_df.empty:
        raise ValueError(f"{label} returned no valid portfolio samples.")

    summary: dict[str, object] = {
        "label": label,
        "n_valid": int(len(results_df)),
        "metrics": {},
    }
    is_bootstrap = "p_loss" in thresholds
    metrics_to_show = ("CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR", "Sortino")
    failures: list[str] = []
    criteria: list[tuple[bool, str]] = []
    metric_summary = summary["metrics"]
    if not isinstance(metric_summary, dict):
        raise TypeError("Internal portfolio summary has an invalid metrics container.")

    logging.info("=" * 96)
    logging.info("%s ROBUSTNESS REPORT (n=%d samples)", label, len(results_df))
    if is_bootstrap:
        logging.info(
            "Note: block reshuffling suppresses multi-year trends; interpret bootstrap percentiles accordingly.",
        )
    logging.info("=" * 96)
    header = f"{'Metric':<10} {'Baseline':>10} {'Mean':>10} {'p05':>10} "
    header += f"{'p25':>10} {'Median':>10} {'p75':>10} {'p95':>10} "
    header += f"{'P(CAGR<0)' if is_bootstrap else 'P(worse)':>10}"
    logging.info(header)
    logging.info("-" * 96)

    p_loss = (
        float((results_df["CAGR"] < 0.0).mean())
        if "p_loss" in thresholds and "CAGR" in results_df.columns
        else float("nan")
    )

    for metric in metrics_to_show:
        if metric not in results_df.columns:
            if metric in thresholds:
                failures.append(f"Missing portfolio metric: {metric}")
                criteria.append((False, f"Missing portfolio metric: {metric}"))
            continue
        values = results_df[metric]
        metric_summary[metric] = {
            "baseline": float(baseline_metrics.get(metric, np.nan)),
            "mean": float(values.mean()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
        }
        baseline = float(baseline_metrics.get(metric, np.nan))
        p_worse = float((values < baseline).mean())
        metric_summary[metric]["p_worse_than_baseline"] = p_worse
        if metric in metrics_to_show:
            def format_metric(value: float, metric_name: str = metric) -> str:
                if metric_name in {"CAGR", "Vol", "MaxDD"}:
                    return f"{value:.1%}"
                return f"{value:.2f}"

            comparison_value = f"{p_loss:.1%}" if is_bootstrap and metric == "CAGR" else (
                f"{p_worse:.1%}" if not is_bootstrap else ""
            )
            logging.info(
                f"{metric:<10} {format_metric(baseline):>10} {format_metric(float(values.mean())):>10} "
                f"{format_metric(float(values.quantile(0.05))):>10} "
                f"{format_metric(float(values.quantile(0.25))):>10} "
                f"{format_metric(float(values.median())):>10} "
                f"{format_metric(float(values.quantile(0.75))):>10} "
                f"{format_metric(float(values.quantile(0.95))):>10} {comparison_value:>10}"
            )
        config = thresholds.get(metric, {})
        p05_min = config.get("p05_min")
        if p05_min is not None:
            p05 = float(values.quantile(0.05))
            passed = p05 >= p05_min
            criterion = str(config.get("label", f"{metric} p05 >= {p05_min}"))
            value_text = f"{p05:.2%}" if metric in {"CAGR", "MaxDD"} else f"{p05:.3f}"
            threshold_text = (
                f"{p05_min:.2%}"
                if metric in {"CAGR", "MaxDD"}
                else f"{p05_min:.3f}"
            )
            criteria.append(
                (passed, f"{criterion} (p05={value_text}; threshold={threshold_text})")
            )
            if not passed:
                failures.append(f"{metric} p05 below {p05_min}")

    if "p_loss" in thresholds:
        p_loss_config = thresholds["p_loss"]
        criterion = str(p_loss_config.get("label", "P(CAGR < 0)"))
        p_loss_max = p_loss_config.get("max")
        if "CAGR" not in results_df.columns:
            failures.append("Missing portfolio metric: CAGR (required for p_loss)")
            criteria.append((False, f"{criterion} (cannot calculate: missing CAGR)"))
        else:
            metric_summary["p_loss"] = p_loss
            if p_loss_max is not None:
                passed = p_loss <= p_loss_max
                criteria.append(
                    (
                        passed,
                        f"{criterion} (value={p_loss:.2%}; threshold={p_loss_max:.2%})",
                    )
                )
                if not passed:
                    failures.append(f"p_loss above {p_loss_max}")

    logging.info("-" * 96)

    summary["failures"] = failures
    summary["verdict"] = "FRAGILE" if failures else "ROBUST"
    logging.info(
        "%s portfolio verdict: %s (%d valid samples)",
        label,
        summary["verdict"],
        len(results_df),
    )
    for passed, criterion in criteria:
        logging.info(
            "%s portfolio criterion: %s %s",
            label,
            "PASS" if passed else "FAIL",
            criterion,
        )
    for failure in failures:
        logging.info("%s portfolio failure: %s", label, failure)
    return summary


def combine_portfolio_verdicts(
    summaries: Sequence[Mapping[str, object]],
) -> str:
    """Return the final portfolio decision after all requested portfolio gates."""
    if not summaries:
        raise ValueError("At least one portfolio robustness summary is required.")
    verdicts = [summary.get("verdict") for summary in summaries]
    if any(verdict != "ROBUST" for verdict in verdicts):
        return "NOT_ACCEPTABLE"
    return "ACCEPTABLE"
