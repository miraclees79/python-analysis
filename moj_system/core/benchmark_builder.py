from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from moj_system.core.benchmark_catalog import INDEX_CATALOG, IndexSpec
from moj_system.core.benchmark_parser import ParsedBenchmark
from moj_system.data.data_manager import load_local_csv

FX_FILES = {"USD": ("usdpln", "USDPLN=X"), "EUR": ("eurpln", "EURPLN=X"), "JPY": ("jpypln", "JPYPLN=X")}
YFinanceDownloader = Callable[[str], pd.Series | None]


@dataclass(frozen=True)
class BuiltBenchmark:
    prices: pd.Series | None
    source: str
    quality: str
    frequency: str
    coverage: float
    component_keys: tuple[str, ...]
    unresolved: tuple[str, ...] = ()


class BenchmarkBuilder:
    def __init__(self, yfinance_downloader: YFinanceDownloader | None = None) -> None:
        self._price_cache: dict[str, tuple[pd.Series | None, str]] = {}
        self._fx_cache: dict[str, pd.Series | None] = {}
        self._yfinance_downloader = yfinance_downloader or self._download_yfinance

    @staticmethod
    def _clean_series(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors="coerce")
        clean.index = pd.to_datetime(clean.index, errors="coerce")
        clean = clean.loc[~clean.index.isna()]
        clean = clean.loc[~clean.index.duplicated(keep="last")].sort_index()
        clean = clean.loc[np.isfinite(clean) & (clean > 0.0)]
        return clean.astype(float)

    @staticmethod
    def _download_yfinance(ticker: str) -> pd.Series | None:
        try:
            data = yf.download(
                tickers=ticker,
                period="max",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data is None or data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(level=1, axis=1)
            column = "Close" if "Close" in data.columns else "Adj Close"
            if column not in data.columns:
                return None
            return data[column]
        except Exception as exc:
            logging.warning("Could not download benchmark ticker %s: %s", ticker, exc)
            return None

    def _load_index(self, spec: IndexSpec) -> tuple[pd.Series | None, str]:
        cached = self._price_cache.get(spec.key)
        if cached is not None:
            return cached

        series: pd.Series | None = None
        source = "unavailable"
        if spec.data_file:
            frame = load_local_csv(ticker=spec.data_file, label=spec.key, mandatory=False)
            if frame is not None and "Zamkniecie" in frame:
                series = self._clean_series(frame["Zamkniecie"])
                if not series.empty:
                    source = "local"

        if (series is None or series.empty) and spec.yfinance_ticker:
            downloaded = self._yfinance_downloader(spec.yfinance_ticker)
            if downloaded is not None:
                series = self._clean_series(downloaded)
                if not series.empty:
                    source = "yfinance_proxy"

        self._price_cache[spec.key] = (series, source)
        return series, source

    def _load_fx(self, currency: str) -> pd.Series | None:
        if currency == "PLN":
            return None
        if currency in self._fx_cache:
            return self._fx_cache[currency]
        spec = FX_FILES.get(currency)
        if spec is None:
            self._fx_cache[currency] = None
            return None

        frame = load_local_csv(ticker=spec[0], label=f"{currency}/PLN", mandatory=False)
        fx: pd.Series | None = None
        if frame is not None and "Zamkniecie" in frame:
            local = self._clean_series(frame["Zamkniecie"])
            if not local.empty:
                fx = local
        if fx is None:
            downloaded = self._yfinance_downloader(spec[1])
            if downloaded is not None:
                clean = self._clean_series(downloaded)
                fx = clean if not clean.empty else None
        self._fx_cache[currency] = fx
        return fx

    @staticmethod
    def _rate_returns(rate_percent: pd.Series, spread_bps: float) -> pd.Series:
        day_counts = rate_percent.index.to_series().diff().dt.days.fillna(0).clip(lower=0)
        annual_rate = rate_percent.shift(1).ffill().to_numpy() / 100.0 + spread_bps / 10000.0
        returns = np.expm1(annual_rate * day_counts.to_numpy() / 365.0)
        result = pd.Series(returns, index=rate_percent.index, dtype=float)
        result.iloc[0] = 0.0
        return result

    @staticmethod
    def _add_spread(returns: pd.Series, spread_bps: float) -> pd.Series:
        if spread_bps == 0.0:
            return returns
        day_counts = returns.index.to_series().diff().dt.days.fillna(0).clip(lower=0)
        spread_returns = np.expm1((spread_bps / 10000.0) * day_counts.to_numpy() / 365.0)
        return (1.0 + returns) * (1.0 + pd.Series(spread_returns, index=returns.index)) - 1.0

    def build(self, parsed: ParsedBenchmark, frequency: str = "daily") -> BuiltBenchmark:
        if parsed.is_no_benchmark:
            return BuiltBenchmark(None, "none", "none", frequency, 0.0, (), parsed.unresolved)
        if parsed.unresolved:
            return BuiltBenchmark(None, "unresolved", "unresolved", frequency, 0.0, (), parsed.unresolved)

        component_returns: dict[str, pd.Series] = {}
        sources: set[str] = set()
        proxy_used = False
        unresolved: list[str] = []

        for component in parsed.components:
            spec = INDEX_CATALOG.get(component.index_key)
            if spec is None:
                unresolved.append(f"unknown catalog key: {component.index_key}")
                continue
            index_prices, source = self._load_index(spec)
            if index_prices is None or index_prices.empty:
                unresolved.append(f"no series available for {component.index_key}")
                continue
            sources.add(source)
            proxy_used = proxy_used or spec.proxy or source == "yfinance_proxy"

            if spec.asset_type == "rate":
                returns = self._rate_returns(index_prices, component.spread_bps)
            else:
                prices = index_prices
                if spec.currency != "PLN" and not component.hedged_to_pln:
                    fx = self._load_fx(spec.currency)
                    if fx is None or fx.empty:
                        unresolved.append(f"no {spec.currency}/PLN FX series for {component.index_key}")
                        continue
                    fx_aligned = fx.reindex(prices.index).ffill(limit=5)
                    prices = (prices * fx_aligned).dropna()
                elif component.hedged_to_pln and spec.currency != "PLN":
                    proxy_used = True
                returns = prices.pct_change(fill_method=None).fillna(0.0)
                returns = self._add_spread(returns, component.spread_bps)
            component_returns[component.index_key] = returns

        if unresolved:
            return BuiltBenchmark(
                None,
                "+".join(sorted(sources)) if sources else "unavailable",
                "unresolved",
                frequency,
                0.0,
                tuple(component_returns),
                tuple(unresolved),
            )
        if not component_returns:
            return BuiltBenchmark(None, "unavailable", "unresolved", frequency, 0.0, (), ("no components",))

        calendar = component_returns[next(iter(component_returns))].index
        for returns in component_returns.values():
            calendar = calendar.union(returns.index)
        calendar = calendar.sort_values()
        observed = pd.DataFrame(
            {key: returns.reindex(calendar) for key, returns in component_returns.items()}
        )
        observed_periods = observed.iloc[1:]
        coverage = float(observed_periods.notna().all(axis=1).mean()) if not observed_periods.empty else 0.0
        aligned = observed.fillna(0.0)
        weights = {component.index_key: component.weight for component in parsed.components}
        portfolio_returns = aligned.mul(pd.Series(weights), axis="columns").sum(axis=1)
        portfolio_returns = self._add_spread(portfolio_returns, parsed.portfolio_spread_bps)
        benchmark_prices = (1.0 + portfolio_returns).cumprod() * 100.0
        benchmark_prices.name = "benchmark"

        if frequency == "weekly":
            benchmark_prices = benchmark_prices.resample("W-FRI").last().dropna()
        return BuiltBenchmark(
            benchmark_prices,
            "+".join(sorted(sources)),
            "proxy" if proxy_used else "exact",
            frequency,
            coverage,
            tuple(component_returns),
        )
