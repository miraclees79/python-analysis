# -*- coding: utf-8 -*-
"""
moj_system/data/market_data_facade.py
=====================================
Facade over every market-data source used by the strategy runners.

Runners (daily / validation / sweep) ask for ready-to-use DataFrames and never
learn whether a series comes from a local CSV (``load_local_csv``), from the
Drive-backed builder (``build_and_upload``) or from a derived construction
(``build_standard_two_asset_data``, ``build_global_assets``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from moj_system.config import ASSET_REGISTRY
from moj_system.core.pension_engine import StandardTwoAssetData, build_standard_two_asset_data
from moj_system.core.universe import (
    AllocationSettings,
    AssetSpec,
    build_global_assets,
    get_allocation_settings,
    load_fx_map,
)
from moj_system.core.utils import build_mmf_extended
from moj_system.data.builder import build_and_upload
from moj_system.data.data_manager import load_local_csv
from moj_system.data.ppe_manager import build_continuous_ppe_data

HISTORY_FLOOR = "1995-01-02"
SWEEP_FX_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "JPY")
SWEEP_DEFAULT_ASSET_KEYS: tuple[str, ...] = (
    "SP500",
    "NIKKEI225",
    "WIG20TR",
    "NASDAQ100",
    "PL10Y",
    "DE10Y",
    "WIBOR1M",
)


class MarketDataError(RuntimeError):
    """Raised when a requested market series cannot be provided."""


@dataclass(frozen=True, eq=False)
class PensionMarketContext:
    """Inputs of the WIG + TBSP + MMF pension strategy."""

    wig: pd.DataFrame
    tbsp: pd.DataFrame
    mmf: pd.DataFrame
    wibor1m: pd.DataFrame | None
    pl10y: pd.DataFrame
    de10y: pd.DataFrame
    derived: StandardTwoAssetData


@dataclass(frozen=True, eq=False)
class GlobalMarketContext:
    """Pension inputs plus the global-portfolio asset universe."""

    pension: PensionMarketContext
    assets: dict[str, AssetSpec]
    settings: AllocationSettings
    fx_hedged: bool


class MarketDataFacade:
    """Single entry point for market data (see module docstring)."""

    def __init__(
        self,
        folder_id: str | None,
        credentials_path: str | None,
    ) -> None:
        self.folder_id = folder_id.strip() if folder_id else folder_id
        self.credentials_path = credentials_path

    @property
    def drive_folder_id(self) -> str:
        return self.folder_id or ""

    # ------------------------------------------------------------------
    # Elementary series
    # ------------------------------------------------------------------

    def get_wig(self) -> pd.DataFrame:
        wig_df = load_local_csv(
            ticker="wig",
            label="WIG",
        )
        return wig_df.loc[wig_df.index >= pd.Timestamp(HISTORY_FLOOR)]

    def get_tbsp(self) -> pd.DataFrame:
        tbsp_df = build_and_upload(
            folder_id=self.drive_folder_id,
            raw_filename="tbsp_extended_full.csv",
            combined_filename="tbsp_extended_combined.csv",
            extension_ticker="^tbsp",
            extension_source="stooq",
            credentials_path=self.credentials_path,
        )
        if tbsp_df is None:
            raise MarketDataError("Failed to build TBSP series.")
        return tbsp_df

    def get_mmf(self) -> pd.DataFrame:
        return load_local_csv(
            ticker="fund_2720",
            label="MMF",
        )

    def get_wibor1m(self) -> pd.DataFrame | None:
        return load_local_csv(
            ticker="wibor1m",
            label="WIBOR1M",
            mandatory=False,
        )

    def get_cash_series(self) -> pd.DataFrame:
        """MMF, extended backwards with WIBOR1M when that series is available."""
        mmf_df = self.get_mmf()
        wibor_df = self.get_wibor1m()
        if wibor_df is None:
            return mmf_df
        return build_mmf_extended(
            mmf_df=mmf_df,
            wibor1m_df=wibor_df,
            floor_date=HISTORY_FLOOR,
        )

    def get_ppe_data(self) -> pd.DataFrame | None:
        """Continuous PPE price history used by the execution simulation (None if unavailable)."""
        return build_continuous_ppe_data(
            folder_id=self.drive_folder_id,
            credentials_path=self.credentials_path or "",
        )

    # ------------------------------------------------------------------
    # Asset series
    # ------------------------------------------------------------------

    def get_asset_series(
        self,
        asset_key: str,
        local_only: bool = False,
    ) -> pd.DataFrame:
        """Price frame of one registry asset.

        Drive-sourced assets are built through the Drive builder unless
        ``local_only`` forces the local CSV.
        """
        asset_config = ASSET_REGISTRY[asset_key]
        file_prefix = asset_key.lower()

        if asset_config.get("source") == "drive" and not local_only:
            is_msci_world = asset_key == "MSCI_World"
            asset_df = build_and_upload(
                folder_id=self.drive_folder_id,
                raw_filename=f"{file_prefix}_wsj_raw.csv" if is_msci_world else f"{file_prefix}.csv",
                combined_filename=f"{file_prefix}_combined.csv",
                extension_ticker=str(asset_config["ticker"]),
                extension_source="yfinance",
                credentials_path=self.credentials_path,
                is_msci_world=is_msci_world,
            )
        else:
            asset_df = load_local_csv(
                ticker=file_prefix,
                label=asset_key,
            )

        if asset_df is None:
            raise MarketDataError(f"Failed to load data for {asset_key}")
        return asset_df

    # ------------------------------------------------------------------
    # Composite contexts
    # ------------------------------------------------------------------

    def get_pension_context(self) -> PensionMarketContext:
        wig_df = self.get_wig()
        mmf_df = self.get_mmf()
        wibor_df = self.get_wibor1m()
        tbsp_df = self.get_tbsp()
        pl10y_df = load_local_csv(
            ticker="pl10y",
            label="PL10Y",
        )
        de10y_df = load_local_csv(
            ticker="de10y",
            label="DE10Y",
        )
        derived = build_standard_two_asset_data(
            wig=wig_df,
            tbsp=tbsp_df,
            mmf=mmf_df,
            wibor1m=wibor_df,
            pl10y=pl10y_df,
            de10y=de10y_df,
            mmf_floor=HISTORY_FLOOR,
        )
        return PensionMarketContext(
            wig=wig_df,
            tbsp=tbsp_df,
            mmf=mmf_df,
            wibor1m=wibor_df,
            pl10y=pl10y_df,
            de10y=de10y_df,
            derived=derived,
        )

    def get_global_context(
        self,
        variant: str,
    ) -> GlobalMarketContext:
        variant_config = ASSET_REGISTRY[variant]
        fx_hedged = bool(variant_config.get("fx_hedged", True))
        pension_context = self.get_pension_context()
        assets = build_global_assets(
            wig_df=pension_context.wig,
            fx_map=load_fx_map(),
            fx_hedged=fx_hedged,
            asset_keys=cast("list[str]", variant_config["assets"]),
            folder_id=self.folder_id,
            credentials_path=self.credentials_path,
            crypto_data_start=str(variant_config.get("crypto_data_start", "1990-01-01")),
        )
        settings = get_allocation_settings(
            cfg=variant_config,
            assets=assets,
        )
        return GlobalMarketContext(
            pension=pension_context,
            assets=assets,
            settings=settings,
            fx_hedged=fx_hedged,
        )

    def get_sweep_data_map(
        self,
        extra_asset_keys: Sequence[str] = (),
    ) -> dict[str, pd.DataFrame]:
        """Upper-case keyed map of every series the parameter sweep needs."""
        data_map: dict[str, pd.DataFrame] = {}

        for currency in SWEEP_FX_CURRENCIES:
            fx_df = load_local_csv(
                ticker=f"{currency.lower()}pln",
                label=f"{currency}PLN",
            )
            if fx_df is not None:
                data_map[f"{currency}PLN"] = fx_df

        extended_series: dict[str, pd.DataFrame | None] = {}
        extended_series["TBSP"] = build_and_upload(
            folder_id=self.drive_folder_id,
            raw_filename="tbsp_extended_full.csv",
            combined_filename="tbsp_extended_combined.csv",
            extension_ticker="^tbsp",
            extension_source="stooq",
            credentials_path=self.credentials_path,
        )
        extended_series["MSCI_WORLD"] = build_and_upload(
            folder_id=self.drive_folder_id,
            raw_filename="msci_world_wsj_raw.csv",
            combined_filename="msci_world_combined.csv",
            extension_ticker="URTH",
            extension_source="yfinance",
            credentials_path=self.credentials_path,
            is_msci_world=True,
        )
        extended_series["STOXX600"] = build_and_upload(
            folder_id=self.drive_folder_id,
            raw_filename="stoxx600.csv",
            combined_filename="stoxx600_combined.csv",
            extension_ticker="^STOXX",
            extension_source="yfinance",
            credentials_path=self.credentials_path,
        )

        for series_key, series_df in extended_series.items():
            if series_df is not None:
                data_map[series_key] = series_df

        requested_keys = set(SWEEP_DEFAULT_ASSET_KEYS)
        requested_keys.update(asset_key.upper() for asset_key in extra_asset_keys)
        for asset_key in requested_keys:
            if asset_key in data_map:
                continue
            asset_df = load_local_csv(
                ticker=asset_key.lower(),
                label=asset_key,
                mandatory=False,
            )
            if asset_df is not None:
                data_map[asset_key] = asset_df

        data_map["WIG"] = self.get_wig()
        data_map["MMF_EXT"] = self.get_cash_series()
        return data_map
