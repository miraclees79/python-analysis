# -*- coding: utf-8 -*-
"""
moj_system/scripts/daily_runner.py
==================================
Universal Entry Point for all investment strategies.

The script only parses arguments and dispatches to a strategy pipeline
(see ``moj_system.core.daily_pipelines``); data access goes through
``MarketDataFacade``.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from moj_system.config import ASSET_REGISTRY, OUTPUT_DIR
from moj_system.core.daily_pipelines import (
    DailyStrategyPipeline,
    GlobalPipeline,
    PensionPipeline,
    SingleAssetPipeline,
)
from moj_system.data.market_data_facade import MarketDataFacade
from moj_system.data.updater import DataUpdater


def setup_logging(
    output_prefix: str,
) -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Użycie operatora / z biblioteki pathlib
    log_file = OUTPUT_DIR / f"{output_prefix}.log"
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )



def build_pipeline(
    asset_key:   str,
    stop_mode:   str,
    facade:      MarketDataFacade,
    send_email:  bool,
) -> DailyStrategyPipeline:
    asset_type = ASSET_REGISTRY[asset_key]["type"]
    if asset_type == "portfolio_pension":
        return PensionPipeline(
            facade=facade,
            stop_mode_arg=stop_mode,
            send_email=send_email,
        )
    if asset_type == "portfolio_global":
        return GlobalPipeline(
            asset_key=asset_key,
            facade=facade,
            stop_mode_arg=stop_mode,
            send_email=send_email,
        )
    return SingleAssetPipeline(
        asset_name=asset_key,
        facade=facade,
        stop_mode_arg=stop_mode,
        send_email=send_email,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Daily Strategy Runner")
    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        help="Asset key: WIG20TR, PENSION, GLOBAL_A, etc.",
    )
    parser.add_argument("--stop_mode", type=str, choices=["fixed", "atr", "auto"], default="auto")
    parser.add_argument(
        "--email",
        action="store_true",
        help="Wyślij raport e-mailem (wszystkie strategie, w tym PENSION).",
    )
    args = parser.parse_args()

    cfg = ASSET_REGISTRY.get(args.asset)
    if not cfg:
        sys.exit(f"Error: Unknown asset '{args.asset}'.")
    setup_logging(
        output_prefix=args.asset.lower(),
    )

    creds_path = str(Path(tempfile.gettempdir()) / "credentials.json")
    updater = DataUpdater(credentials_path=creds_path)
    updater.run_full_update(get_funds=False)

    facade = MarketDataFacade(
        folder_id=os.environ.get("GDRIVE_FOLDER_ID"),
        credentials_path=creds_path,
    )
    pipeline = build_pipeline(
        asset_key=args.asset,
        stop_mode=args.stop_mode,
        facade=facade,
        send_email=args.email,
    )
    pipeline.execute_pipeline()

    logging.info(f"Execution of {args.asset} completed successfully.")


if __name__ == "__main__":
    main()
