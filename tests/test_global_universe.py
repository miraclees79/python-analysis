from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from moj_system.config import ASSET_REGISTRY, GLOBAL_ASSET_CATALOG
from moj_system.core.universe import build_global_assets, get_allocation_settings


class GlobalAssetConfigurationTests(unittest.TestCase):
    def test_existing_global_variants_keep_their_current_asset_sets(self) -> None:
        expected_asset_sets = {
            "GLOBAL_A": ["WIG", "SP500", "STOXX600", "Nikkei225"],
            "GLOBAL_B": ["WIG", "MSCI_World"],
            "GLOBAL_CRYPTO": ["WIG", "SP500", "BTC"],
            "GLOBAL_B_CRYPTO": ["WIG", "MSCI_World", "BTC"],
        }

        for variant, expected_assets in expected_asset_sets.items():
            with self.subTest(variant=variant):
                self.assertEqual(ASSET_REGISTRY[variant]["assets"], expected_assets)

    def test_global_variant_asset_lists_drive_builder_selection(self) -> None:
        fx_map = {
            currency: pd.Series([1.0])
            for currency in ("USD", "EUR", "JPY")
        }

        with (
            patch(
                "moj_system.core.universe.load_local_csv",
                return_value=pd.DataFrame({"Zamkniecie": [1.0]}),
            ),
            patch(
                "moj_system.core.universe.build_and_upload",
                return_value=pd.DataFrame({"Zamkniecie": [1.0]}),
            ),
            patch(
                "moj_system.core.universe.load_crypto_asset",
                return_value=pd.DataFrame({"Zamkniecie": [1.0]}),
            ),
        ):
            for variant, cfg in ASSET_REGISTRY.items():
                if cfg["type"] != "portfolio_global":
                    continue

                with self.subTest(variant=variant):
                    assets = build_global_assets(
                        wig_df=pd.DataFrame({"Zamkniecie": [1.0]}),
                        fx_map=fx_map,
                        fx_hedged=cfg.get("fx_hedged", True),
                        asset_keys=cfg["assets"],
                        crypto_data_start=cfg.get("crypto_data_start", "1990-01-01"),
                    )
                    self.assertEqual(list(assets), cfg["assets"])

    def test_variant_definitions_reference_catalog_assets(self) -> None:
        required_fields_by_source = {
            "provided": set(),
            "local": {"ticker"},
            "drive": {
                "data_key",
                "raw_filename",
                "combined_filename",
                "extension_ticker",
            },
            "crypto": {"ticker"},
        }

        for variant, cfg in ASSET_REGISTRY.items():
            if cfg["type"] != "portfolio_global":
                continue

            with self.subTest(variant=variant):
                asset_keys = set(cfg["assets"])
                self.assertTrue(asset_keys.issubset(GLOBAL_ASSET_CATALOG))
                cap_keys = set(cfg.get("asset_caps", {}))
                self.assertTrue(cap_keys.issubset(asset_keys | {"TBSP"}))

        for asset_key, spec in GLOBAL_ASSET_CATALOG.items():
            with self.subTest(asset=asset_key):
                self.assertIn(spec["source"], required_fields_by_source)
                self.assertTrue(required_fields_by_source[spec["source"]].issubset(spec))
                self.assertIn(spec["hedge"], {"portfolio", "never"})
                if "fx" in spec:
                    self.assertIn(spec["fx"], {"USD", "EUR", "JPY"})

    def test_catalog_builder_loads_assets_and_applies_fx_policies(self) -> None:
        wig_df = pd.DataFrame({"Zamkniecie": [1.0]})
        local_df = pd.DataFrame({"Zamkniecie": [2.0]})
        drive_df = pd.DataFrame({"Zamkniecie": [3.0]})
        crypto_df = pd.DataFrame({"Zamkniecie": [4.0]})
        fx_map = {
            currency: pd.Series([1.0])
            for currency in ("USD", "EUR", "JPY")
        }

        with (
            patch(
                "moj_system.core.universe.load_local_csv",
                return_value=local_df,
            ) as load_local,
            patch(
                "moj_system.core.universe.build_and_upload",
                return_value=drive_df,
            ) as build_drive,
            patch(
                "moj_system.core.universe.load_crypto_asset",
                return_value=crypto_df,
            ) as load_crypto,
        ):
            assets = build_global_assets(
                wig_df=wig_df,
                fx_map=fx_map,
                fx_hedged=True,
                asset_keys=[
                    "WIG",
                    "SP500",
                    "STOXX600",
                    "Nikkei225",
                    "MSCI_World",
                    "BTC",
                    "ETH",
                ],
                crypto_data_start="2013-01-01",
            )

        self.assertEqual(
            list(assets),
            ["WIG", "SP500", "STOXX600", "Nikkei225", "MSCI_World", "BTC", "ETH"],
        )
        self.assertIs(assets["WIG"].price_df, wig_df)
        self.assertIs(assets["SP500"].price_df, local_df)
        self.assertIs(assets["STOXX600"].price_df, drive_df)
        self.assertIs(assets["BTC"].price_df, crypto_df)
        self.assertIs(assets["ETH"].price_df, crypto_df)
        self.assertIs(assets["SP500"].fx_series, fx_map["USD"])
        self.assertIs(assets["STOXX600"].fx_series, fx_map["EUR"])
        self.assertIs(assets["Nikkei225"].fx_series, fx_map["JPY"])
        self.assertTrue(assets["SP500"].hedged)
        self.assertFalse(assets["BTC"].hedged)
        self.assertFalse(assets["ETH"].hedged)
        self.assertTrue(assets["BTC"].is_crypto)
        self.assertTrue(assets["ETH"].is_crypto)

        self.assertEqual(load_local.call_count, 2)
        self.assertEqual(build_drive.call_count, 2)
        self.assertEqual(
            [call.kwargs["extension_ticker"] for call in build_drive.call_args_list],
            ["^STOXX", "URTH"],
        )
        self.assertTrue(
            all(call.kwargs["extension_source"] == "yfinance" for call in build_drive.call_args_list),
        )
        self.assertTrue(build_drive.call_args_list[1].kwargs["is_msci_world"])
        self.assertEqual(load_crypto.call_count, 2)
        self.assertEqual(
            [call.kwargs["ticker"] for call in load_crypto.call_args_list],
            ["btc", "eth"],
        )
        self.assertEqual(
            [call.kwargs["data_start"] for call in load_crypto.call_args_list],
            ["2013-01-01", "2013-01-01"],
        )
        allocation_settings = get_allocation_settings(cfg={}, assets=assets)
        self.assertEqual(allocation_settings.optional_keys, frozenset({"BTC", "ETH"}))
        self.assertTrue(GLOBAL_ASSET_CATALOG["BTC"]["is_crypto"])
        self.assertTrue(GLOBAL_ASSET_CATALOG["ETH"]["is_crypto"])

    def test_preloaded_data_is_used_for_local_and_drive_assets(self) -> None:
        preloaded = {
            "SP500": pd.DataFrame({"Zamkniecie": [1.0]}),
            "STOXX600": pd.DataFrame({"Zamkniecie": [2.0]}),
            "MSCI_WORLD": pd.DataFrame({"Zamkniecie": [3.0]}),
        }

        with (
            patch("moj_system.core.universe.load_local_csv") as load_local,
            patch("moj_system.core.universe.build_and_upload") as build_drive,
            patch(
                "moj_system.core.universe.load_crypto_asset",
                return_value=pd.DataFrame({"Zamkniecie": [4.0]}),
            ),
        ):
            assets = build_global_assets(
                wig_df=pd.DataFrame({"Zamkniecie": [0.0]}),
                fx_map={
                    currency: pd.Series([1.0])
                    for currency in ("USD", "EUR", "JPY")
                },
                fx_hedged=False,
                asset_keys=["SP500", "STOXX600", "MSCI_World"],
                preloaded=preloaded,
            )

        load_local.assert_not_called()
        build_drive.assert_not_called()
        self.assertIs(assets["SP500"].price_df, preloaded["SP500"])
        self.assertIs(assets["STOXX600"].price_df, preloaded["STOXX600"])
        self.assertIs(assets["MSCI_World"].price_df, preloaded["MSCI_WORLD"])

    def test_unknown_asset_and_missing_fx_fail_explicitly(self) -> None:
        shared_kwargs = {
            "wig_df": pd.DataFrame({"Zamkniecie": [1.0]}),
            "fx_map": {},
            "fx_hedged": True,
        }

        with self.assertRaisesRegex(ValueError, "Unknown global portfolio asset"):
            build_global_assets(**shared_kwargs, asset_keys=["NOT_CONFIGURED"])

        with self.assertRaisesRegex(ValueError, "Missing FX series for USD"):
            build_global_assets(**shared_kwargs, asset_keys=["SP500"])


if __name__ == "__main__":
    unittest.main()
