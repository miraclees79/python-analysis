# -*- coding: utf-8 -*-
"""
moj_system/data/data_manager.py
===============================
Loads and processes historical data from raw_csv folder.
Replaces legacy load_stooq_local.
"""

import logging
import os

import pandas as pd

# Target data path: moj_system/data/raw_csv/
from moj_system.config import DATA_DIR as DATA_ROOT

DATA_DIR = DATA_ROOT / "raw_csv"


def ensure_data_dir_exists():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_local_csv(
    ticker: str, label: str, data_start: str = "1990-01-01", mandatory: bool = True,
) -> pd.DataFrame | None:
    """
    Loads a local CSV file from moj_system/data/raw_csv/.
    """
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        if mandatory:
            logging.error(f"Missing data file: {path}")
            import sys

            sys.exit(1)
        return None

    try:
        # utf-8-sig removes BOM if present
        df = pd.read_csv(
            path, on_bad_lines="skip", delimiter=",", decimal=".", encoding="utf-8-sig",
        )
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return None

    if df.empty or "Data" not in df.columns:
        if mandatory:
            logging.error(f"Corrupted or empty file: {path}")
        return None

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df.dropna(subset=["Data"], inplace=True)
    df = df.sort_values(by="Data").set_index("Data")

    # Filter by history start date
    df = df.loc[df.index >= pd.Timestamp(data_start)]

    logging.info(
        f"OK  : {label:<14} {len(df):>5} rows  {df.index.min().date()} to {df.index.max().date()}",
    )
    return df
