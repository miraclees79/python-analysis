# -*- coding: utf-8 -*-
"""
moj_system/core/universe.py
===========================
Shared construction of the global-portfolio asset universe.
Single source of truth for daily_runner, validate_robustness,
sharded_robustness and sweep_optimizer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from moj_system.config import BASE_GRIDS, CRYPTO_GRIDS, CRYPTO_KEYS
from moj_system.core.global_engine import (
    build_price_df_from_returns,
    build_return_series,
)
from moj_system.data.builder import build_and_upload
from moj_system.data.data_manager import load_local_csv

CLOSE_COL: str = "Zamkniecie"


@dataclass(frozen=True, eq=False)
class AssetSpec:
    price_df: pd.DataFrame
    fx_series: pd.Series | None
    hedged: bool
    is_crypto: bool = False

    @property
    def asset_class(self) -> str:
        return "CRYPTO" if self.is_crypto else "EQUITY"


@dataclass(frozen=True)
class AllocationSettings:
    asset_caps: dict[str, float] | None
    optional_keys: frozenset[str]
    min_delta: float
    delta_tol: float


def get_allocation_settings(cfg: Mapping[str, Any]) -> AllocationSettings:
    caps = cfg.get("asset_caps")
    return AllocationSettings(
        asset_caps=dict(caps) if caps else None,
        optional_keys=CRYPTO_KEYS if cfg.get("mode") == "global_crypto" else frozenset(),
        min_delta=float(cfg.get("min_delta", 0.10)),
        delta_tol=float(cfg.get("delta_tol", 0.0)),
    )


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Collapse a 7-day series onto a business-day calendar.

    Every calendar day is mapped to the first session >= that day, so weekend
    bars fold into Monday: close = last, high = max, low = min.
    """
    src = df.sort_index()
    pos: np.ndarray = np.asarray(calendar.searchsorted(value=src.index))
    valid = pos < len(calendar)
    src = src.loc[valid]
    pos = pos[valid]
    grouped = src.groupby(by=pos)
    out = pd.DataFrame(
        data={
            CLOSE_COL: grouped[CLOSE_COL].last(),
            "Najwyzszy": grouped["Najwyzszy"].max(),
            "Najnizszy": grouped["Najnizszy"].min(),
        },
    )
    out.index = calendar[out.index.to_numpy()]
    out.index.name = "Data"
    return out


def load_crypto_asset(
    ticker: str,
    label: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    raw = load_local_csv(ticker=ticker, label=label)
    if raw is None:
        raise FileNotFoundError(f"Missing data file for {label}")
    return align_to_calendar(df=raw, calendar=calendar)


def load_fx_map() -> dict[str, pd.Series]:
    fx: dict[str, pd.Series] = {}
    for currency in ("USD", "EUR", "JPY"):
        df = load_local_csv(ticker=f"{currency.lower()}pln", label=f"{currency}PLN")
        if df is None:
            raise FileNotFoundError(f"Missing FX data for {currency}PLN")
        fx[currency] = df[CLOSE_COL]
    return fx


def build_global_assets(
    mode: str,
    wig_df: pd.DataFrame,
    fx_map: Mapping[str, pd.Series],
    fx_hedged: bool,
    folder_id: str | None = None,
    credentials_path: str | None = None,
    preloaded: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, AssetSpec]:
    """Return {label: AssetSpec} for the equity legs of a global portfolio.

    TBSP is NOT included – it stays a separate, gated bond component.
    `preloaded` (sweep_optimizer.data_map, UPPERCASE keys) avoids re-downloading
    Drive series on every sweep iteration. Crypto is always read from local CSV.
    """

    def _local(key: str, ticker: str, label: str) -> pd.DataFrame:
        if preloaded is not None and key in preloaded:
            return preloaded[key]
        df = load_local_csv(ticker=ticker, label=label)
        if df is None:
            raise FileNotFoundError(f"Missing data file for {label}")
        return df

    def _drive(
        key: str,
        raw_filename: str,
        combined_filename: str,
        extension_ticker: str,
        is_msci_world: bool = False,
    ) -> pd.DataFrame:
        if preloaded is not None and key in preloaded:
            return preloaded[key]
        df = build_and_upload(
            folder_id=folder_id or "",
            raw_filename=raw_filename,
            combined_filename=combined_filename,
            extension_ticker=extension_ticker,
            extension_source="yfinance",
            credentials_path=credentials_path,
            is_msci_world=is_msci_world,
        )
        if df is None:
            raise ValueError(f"Could not build series {key}")
        return df

    wig = AssetSpec(price_df=wig_df, fx_series=None, hedged=fx_hedged)
    sp500 = AssetSpec(
        price_df=_local(key="SP500", ticker="sp500", label="SP500"),
        fx_series=fx_map["USD"],
        hedged=fx_hedged,
    )

    if mode == "global_equity":
        return {
            "WIG": wig,
            "SP500": sp500,
            "STOXX600": AssetSpec(
                price_df=_drive("STOXX600", "stoxx600.csv", "stoxx600_combined.csv", "^STOXX"),
                fx_series=fx_map["EUR"],
                hedged=fx_hedged,
            ),
            "Nikkei225": AssetSpec(
                price_df=_local(key="NIKKEI225", ticker="nikkei225", label="Nikkei225"),
                fx_series=fx_map["JPY"],
                hedged=fx_hedged,
            ),
        }

    if mode == "msci_world":
        return {
            "WIG": wig,
            "MSCI_World": AssetSpec(
                price_df=_drive(
                    "MSCI_WORLD",
                    "msci_world_wsj_raw.csv",
                    "msci_world_combined.csv",
                    "URTH",
                    is_msci_world=True,
                ),
                fx_series=fx_map["USD"],
                hedged=fx_hedged,
            ),
        }

    if mode == "global_crypto":
        # No PLN-hedged BTC/ETH exist -> always unhedged (USD -> PLN via FX series)
        return {
            "WIG": wig,
            "SP500": sp500,
            "BTC": AssetSpec(
                price_df=load_crypto_asset(ticker="btc", label="BTC", calendar=wig_df.index),
                fx_series=fx_map["USD"],
                hedged=False,
                is_crypto=True,
            ),
            "ETH": AssetSpec(
                price_df=load_crypto_asset(ticker="eth", label="ETH", calendar=wig_df.index),
                fx_series=fx_map["USD"],
                hedged=False,
                is_crypto=True,
            ),
        }

    raise ValueError(f"Unknown global portfolio mode: {mode!r}")


def prepare_asset_series(
    label: str,
    spec: AssetSpec,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (daily PLN returns without NaN, price frame for walk_forward)."""
    ret_s = build_return_series(
        price_df=spec.price_df,
        fx_series=spec.fx_series,
        hedged=spec.hedged,
    )
    proc_px = (
        spec.price_df
        if spec.hedged or spec.fx_series is None
        else build_price_df_from_returns(ret=ret_s, label=label)
    )
    return ret_s.dropna(), proc_px


def resolve_train_years(
    cfg: Mapping[str, Any],
    spec: AssetSpec,
    default_train: int,
) -> int:
    if spec.is_crypto:
        return int(cfg.get("crypto_train", default_train))
    return default_train


def wf_grid_kwargs(
    spec: AssetSpec,
    use_atr: bool,
) -> dict[str, Any]:
    """Grid kwargs for walk_forward on an equity/crypto leg.

    Non-crypto: identical to BASE_GRIDS defaults (baseline unchanged).
    """
    grids: dict[str, Any] = CRYPTO_GRIDS if spec.is_crypto else BASE_GRIDS
    kwargs: dict[str, Any] = {
        "X_grid": grids["X_GRID"],
        "Y_grid": grids["Y_GRID"],
        "fast_grid": grids["FAST_GRID"],
        "slow_grid": grids["SLOW_GRID"],
        "N_atr_grid": grids["N_ATR_GRID"] if use_atr else None,
    }
    if spec.is_crypto:
        kwargs["tv_grid"] = grids["TV_GRID"]
        kwargs["sl_grid"] = grids["SL_GRID"]
        kwargs["mom_lookback_grid"] = grids["MOM_LB_GRID"]
    return kwargs