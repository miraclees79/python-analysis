# -*- coding: utf-8 -*-
"""
moj_system/core/fund_analytics.py
=================================
Core mathematical engine for fund performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from moj_system.core.strategy_engine import compute_metrics

TRADING_DAYS_PER_YEAR = 252

class FundPerformanceEngine:
    """Calculates long-term metrics and OLS regression."""
    
    @staticmethod
    def compute_absolute_metrics(prices: pd.Series, window_days: int = None) -> dict:
        """Calculates absolute performance metrics for a given window."""
        if window_days is not None and len(prices) < window_days:
            log_ret = np.log(prices / prices.shift(1)).dropna()
            return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "sortino": np.nan,
                    "max_drawdown": np.nan, "calmar": np.nan, "data_years": round(len(prices) / TRADING_DAYS_PER_YEAR, 1),
                    "n_days": len(prices), "insufficient": True, "_log_ret": log_ret}

        if window_days is not None:
            prices = prices.iloc[-window_days:]
        
        equity_curve = prices / prices.iloc[0]
        base_metrics = compute_metrics(equity_curve)
        log_ret = np.log(prices / prices.shift(1)).dropna()

        return {
            "ann_return": round(base_metrics.get("CAGR", np.nan), 4),
            "ann_vol": round(base_metrics.get("Vol", np.nan), 4),
            "sharpe": round(base_metrics.get("Sharpe", np.nan), 4),
            "sortino": round(base_metrics.get("Sortino", np.nan), 4),
            "max_drawdown": round(base_metrics.get("MaxDD", np.nan), 4),
            "calmar": round(base_metrics.get("CalMAR", np.nan), 4),
            "data_years": round(len(prices) / TRADING_DAYS_PER_YEAR, 1),
            "n_days": len(prices),
            "insufficient": False,
            "_log_ret": log_ret
        }

    @staticmethod
    def compute_relative_metrics(fund_log_ret: pd.Series, benchmark_log_ret: pd.Series) -> dict:
        """Performs OLS regression: fund_ret ~ alpha + beta * benchmark_ret."""
        common = fund_log_ret.index.intersection(benchmark_log_ret.index)
        if len(common) < 30:
            return {"alpha": np.nan, "beta": np.nan, "tracking_error": np.nan, "ir": np.nan}

        y = fund_log_ret.loc[common].values
        x = benchmark_log_ret.loc[common].values
        X = np.column_stack([np.ones(len(x)), x])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_daily, beta = coeffs
        residuals = y - (alpha_daily + beta * x)
        te_daily = residuals.std()
        
        alpha_ann = float(np.exp(alpha_daily * TRADING_DAYS_PER_YEAR) - 1)
        te_ann = float(te_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
        ir = (alpha_ann / te_ann) if te_ann > 0 else np.nan

        return {"alpha": round(alpha_ann, 4), "beta": round(float(beta), 4),
                "tracking_error": round(te_ann, 4), "ir": round(ir, 4)}

    @staticmethod
    def compute_rolling_ir(fund_log_ret: pd.Series, benchmark_log_ret: pd.Series, 
                           window: int = 252, step: int = 63) -> dict:
        """Calculates rolling Information Ratio (IR)."""
        common = fund_log_ret.index.intersection(benchmark_log_ret.index)
        if len(common) < window:
            return {"pct_ir_positive": np.nan, "n_ir_windows": 0}

        y_all = fund_log_ret.loc[common].values
        x_all = benchmark_log_ret.loc[common].values
        n = len(common)

        ir_vals = []
        for start in range(0, n - window + 1, step):
            y, x = y_all[start:start+window], x_all[start:start+window]
            X = np.column_stack([np.ones(window), x])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - (coeffs[0] + coeffs[1] * x)
            te = resid.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            a_ann = float(np.exp(coeffs[0] * TRADING_DAYS_PER_YEAR) - 1)
            ir_vals.append(a_ann / te if te > 0 else np.nan)

        valid = [v for v in ir_vals if not np.isnan(v)]
        pct_pos = sum(1 for v in valid if v > 0) / len(valid) if valid else np.nan

        return {"pct_ir_positive": round(pct_pos, 3), "n_ir_windows": len(valid)}

class BenchmarkComparator:
    """Calculates short-term stats and benchmark Hit Rate."""

    @staticmethod
    def calculate_returns(series: pd.Series, periods: int, tail_limit: int = 760) -> pd.Series:
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        return series.pct_change(periods=periods).iloc[-tail_limit:]

    @staticmethod
    def get_distribution_stats(returns_series: pd.Series, tail_limit: int = 66) -> Tuple[float, float, float]:
        if returns_series is None or returns_series.empty:
            return np.nan, np.nan, np.nan
        last_n = returns_series.iloc[-tail_limit:].dropna()
        if last_n.empty:
            return np.nan, np.nan, np.nan
        return last_n.min(), last_n.max(), np.percentile(last_n, 25)

    @classmethod
    def compare_fund_to_benchmark(cls, fund_prices: pd.Series, benchmark_prices: pd.Series) -> Tuple[Dict[str, float], float, float]:
        if isinstance(fund_prices, pd.DataFrame): fund_prices = fund_prices.squeeze()
        if isinstance(benchmark_prices, pd.DataFrame): benchmark_prices = benchmark_prices.squeeze()

        common_dates = fund_prices.index.intersection(benchmark_prices.index)
        if len(common_dates) < 252:
            return {"22-day": np.nan, "66-day": np.nan, "252-day": np.nan}, np.nan, np.nan

        fund_p = fund_prices.loc[common_dates]
        bench_p = benchmark_prices.loc[common_dates]

        periods = [22, 66, 252]
        win_rates = {}
        last_fund_252, last_bench_252 = np.nan, np.nan

        for p in periods:
            fund_ret = cls.calculate_returns(fund_p, p)
            bench_ret = cls.calculate_returns(bench_p, p)

            if fund_ret.dropna().empty or bench_ret.dropna().empty:
                win_rates[f"{p}-day"] = np.nan
                continue

            common_ret_dates = fund_ret.dropna().index.intersection(bench_ret.dropna().index)
            if len(common_ret_dates) == 0:
                win_rates[f"{p}-day"] = np.nan
                continue
                
            f_r = fund_ret.loc[common_ret_dates]
            b_r = bench_ret.loc[common_ret_dates]
            
            win_rate = (f_r > b_r).mean() * 100
            win_rates[f"{p}-day"] = win_rate
            
            if p == 252 and not f_r.empty and not b_r.empty:
                last_fund_252 = f_r.iloc[-1]
                last_bench_252 = b_r.iloc[-1]

        return win_rates, last_fund_252, last_bench_252