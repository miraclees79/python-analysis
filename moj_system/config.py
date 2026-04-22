# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:28:04 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/config.py
====================
Centralized configuration for strategy grids and system paths.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "moj_system" / "data" / "raw_csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Default Strategy Grids
BASE_GRIDS = {
    "X_GRID": [0.08, 0.10, 0.12, 0.15, 0.20],
    "Y_GRID": [0.02, 0.03, 0.05, 0.07, 0.10],
    "FAST_GRID": [50, 75, 100],
    "SLOW_GRID": [150, 200, 250],
    "TV_GRID": [0.08, 0.10, 0.12, 0.15, 0.20],
    "SL_GRID": [0.05, 0.08, 0.10, 0.15],
    "MOM_LB_GRID": [126, 252],
    "ATR_WINDOW": 20,
    "N_ATR_GRID": [0.08, 0.10, 0.12, 0.15, 0.20],
}

BOND_GRIDS = {
    "X_GRID": [0.05, 0.08, 0.10, 0.15],
    "Y_GRID": [0.001],
    "FAST_GRID": [50, 100, 150],
    "SLOW_GRID": [200, 300, 400, 500],
    "TV_GRID": [0.08, 0.10, 0.12],
    "SL_GRID": [0.01, 0.02, 0.03],
    "MOM_LB_GRID": [252],
    "N_ATR_GRID": [0.05, 0.08, 0.10, 0.15],
}

# Research Window Configurations

SWEEP_WINDOW_CONFIGS = [
    (6, 1), (6, 2),
    (7, 1), (7, 2),
    (8, 1), (8, 2),
    (9, 1), (9, 2),
]

# Smoke test run only
SWEEP_WINDOW_CONFIGS_TEST = [(6,2)]


# ---------------------------------------------------------------------------
# Asset-class threshold presets
# ---------------------------------------------------------------------------

EQUITY_THRESHOLDS_MC = {
    "CAGR":   {"p05_min":  0.00, "label": "p05 CAGR > 0%"},
    "Sharpe": {"p05_min":  0.00, "label": "p05 Sharpe > 0"},
    "MaxDD":  {"p05_min": -0.35, "label": "p05 MaxDD > -35%"},
}

BOND_THRESHOLDS_MC = {
    "CAGR":   {"p05_min":  0.00, "label": "p05 CAGR > 0%"},
    "Sharpe": {"p05_min":  0.00, "label": "p05 Sharpe > 0"},
    "MaxDD":  {"p05_min": -0.10, "label": "p05 MaxDD > -10%"},
}

EQUITY_THRESHOLDS_BOOTSTRAP = {
    "CAGR":   {"p05_min": -0.01, "label": "p05 CAGR > -1%"},
    "Sharpe": {"p05_min": -0.10, "label": "p05 Sharpe > -0.10"},
    "MaxDD":  {"p05_min": -0.40, "label": "p05 MaxDD > -40%"},
    "p_loss": {"max":      0.20, "label": "P(CAGR < 0) < 20%"},
}

BOND_THRESHOLDS_BOOTSTRAP = {
    "CAGR":   {"p05_min": -0.01, "label": "p05 CAGR > -1%"},
    "Sharpe": {"p05_min": -0.10, "label": "p05 Sharpe > -0.10"},
    "MaxDD":  {"p05_min": -0.10, "label": "p05 MaxDD > -10%"},
    "p_loss": {"max":      0.20, "label": "P(CAGR < 0) < 20%"},
}

# moj_system/config.py (dodaj do istniejącej treści)

ASSET_REGISTRY = {
    "WIG20TR":  {"type": "single", "source": "stooq", "ticker": "wig20tr", "yf_ticker": "WIG20TR.WA", "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}},
    "MWIG40TR": {"type": "single", "source": "stooq", "ticker": "mwig40tr", "yf_ticker": "MWIG40TR.WA", "train": 8, "test": 2, "default_stop": "fixed", "grids": {"MOM_LB_GRID": [252]}},
    "SWIG80TR": {"type": "single", "source": "stooq", "ticker": "swig80tr", "yf_ticker": "SWIG80TR.WA", "train": 9, "test": 2, "default_stop": "atr"},
    "SP500":    {"type": "single", "source": "stooq", "ticker": "^spx", "yf_ticker": "^GSPC", "train": 6, "test": 2, "default_stop": "fixed"},
    "NASDAQ100":{"type": "single", "source": "stooq", "ticker": "ndq100", "yf_ticker": "^NDX", "train": 6, "test": 2, "default_stop": "fixed"},
    "Nikkei225":{"type": "single", "source": "stooq", "ticker": "nk225", "yf_ticker": "^N225", "train": 6, "test": 2, "default_stop": "fixed"},
    "MSCI_World":{"type": "single", "source": "drive", "ticker": "URTH", "yf_ticker": "URTH", "train": 9, "test": 1, "default_stop": "atr"},
    "STOXX600": {"type": "single", "source": "drive", "ticker": "^STOXX", "yf_ticker": "^STOXX", "train": 7, "test": 1, "default_stop": "atr"},
    # Portfolio Templates
    "PENSION":  {"type": "portfolio_pension", "train": 7, "test": 2, "default_stop_eq": "atr"},
    "GLOBAL_A": {"type": "portfolio_global", "mode": "global_equity", "train": 7, "test": 2, "fx_hedged": True},
    "GLOBAL_B": {"type": "portfolio_global", "mode": "msci_world", "train": 7, "test": 2, "fx_hedged": True}
}
