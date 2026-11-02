# -*- coding: utf-8 -*-
"""
moj_system/core/state_analytics.py
==================================
Shared, side-effect-free analytics describing the *current* state of a
strategy (ATR volatility, MA/momentum filters, active walk-forward window,
open position). Used by every daily report builder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CLOSE_COL: str = "Zamkniecie"
HIGH_COL: str = "Najwyzszy"
LOW_COL: str = "Najnizszy"


def compute_atr_val(
    df: pd.DataFrame,
    atr_window: int,
) -> float:
    """Calculates the current ATR volatility percentage (shifted by 1 for next day logic)."""
    if len(df) < atr_window + 1:
        return 0.0

    df_copy = df.copy()
    has_hl = HIGH_COL in df_copy.columns and LOW_COL in df_copy.columns

    if has_hl:
        prev_close = df_copy[CLOSE_COL].shift(periods=1)
        tr = np.maximum(df_copy[HIGH_COL], prev_close) - np.minimum(df_copy[LOW_COL], prev_close)
        atr_s = (tr / prev_close).rolling(window=atr_window).mean().shift(periods=1) * 100.0
    else:
        atr_s = (df_copy[CLOSE_COL].diff().abs() / df_copy[CLOSE_COL].shift(periods=1)).rolling(
            window=atr_window
        ).mean().shift(periods=1) * 100.0

    val = float(atr_s.iloc[-1])
    return val if np.isfinite(val) else 0.0


def compute_ma_filter_state(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    price_col: str = CLOSE_COL,
    gap_digits: int = 2,
) -> dict[str, Any]:
    """Recompute fast/slow MAs from the full price series.

    ``gap_digits`` is the rounding precision of ``gap_pct`` (single-asset
    reports use 2, portfolio reports 3).
    """
    prices = df[price_col].dropna()
    if len(prices) < slow:
        return {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}

    fast_ma = float(prices.rolling(window=fast).mean().iloc[-1])
    slow_ma = float(prices.rolling(window=slow).mean().iloc[-1])
    gap_ratio = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
    return {
        "fast_ma": round(fast_ma, 2),
        "slow_ma": round(slow_ma, 2),
        "filter_on": fast_ma > slow_ma,
        "gap_pct": round(gap_ratio * 100.0, gap_digits),
    }


def compute_mom_filter_state(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    price_col: str = CLOSE_COL,
    min_history: int | None = None,
) -> dict[str, Any]:
    """Recompute momentum filter state from the full price series.

    ``min_history`` is the minimum number of prices required (defaults to
    ``lookback``; portfolio reports require ``lookback + skip``).
    """
    prices = df[price_col].dropna()
    required_history = lookback if min_history is None else min_history
    if len(prices) < required_history:
        return {"mom_value": None, "filter_on": None}

    mom_series = prices.shift(skip) / prices.shift(lookback) - 1
    last_value = mom_series.iloc[-1]
    mom_value = None if pd.isna(last_value) else float(last_value)
    return {
        "mom_value": round(mom_value * 100, 2) if mom_value is not None else None,
        "filter_on": (mom_value > 0) if mom_value is not None else None,
    }


def _optional_float(
    row: pd.Series,
    key: str,
    default: float | None = None,
) -> float | None:
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


def get_active_window_params(
    wf_results: pd.DataFrame | None,
    include_window_bounds: bool = False,
) -> dict[str, Any]:
    """Parameters of the last walk-forward window and whether an ATR or fixed stop is used.

    ``include_window_bounds`` adds TestStart/TestEnd/target_vol, as needed by
    the single-asset report.
    """
    if wf_results is None or wf_results.empty:
        return {}
    last = wf_results.iloc[-1]

    use_atr = bool(last.get("use_atr_stop", False))
    stop_label = "N_atr" if use_atr else "X"

    params: dict[str, Any] = {
        "filter_mode": last.get("filter_mode", "ma"),
        "stop_param": _optional_float(row=last, key=stop_label, default=0.10),
        "stop_label": stop_label,
        "use_atr_stop": use_atr,
        "Y": _optional_float(row=last, key="Y", default=0.10),
        "fast": int(_optional_float(row=last, key="fast", default=50)),
        "slow": int(_optional_float(row=last, key="slow", default=200)),
        "stop_loss": _optional_float(row=last, key="stop_loss", default=0.05),
        "mom_lookback": int(_optional_float(row=last, key="mom_lookback", default=252)),
        "atr_window": int(_optional_float(row=last, key="atr_window", default=20)),
    }
    if not include_window_bounds:
        return params

    params["TestStart"] = pd.Timestamp(last["TestStart"])
    params["TestEnd"] = pd.Timestamp(last["TestEnd"])
    target_vol = last.get("target_vol")
    if target_vol is not None and pd.notna(target_vol) and str(target_vol).strip() not in ("", "N/A", "nan"):
        params["target_vol"] = float(target_vol)
    mom_lookback = last.get("mom_lookback")
    if (
        mom_lookback is not None
        and pd.notna(mom_lookback)
        and str(mom_lookback).strip() not in ("", "N/A", "nan")
    ):
        params["mom_lookback"] = int(float(mom_lookback))
    else:
        params["mom_lookback"] = 252
    return params


def extract_open_position(
    wf_trades: pd.DataFrame | None,
    last_row_only: bool = False,
) -> dict[str, Any] | None:
    """Return the most recent CARRY record as a dict, or None if flat.

    With ``last_row_only`` the position counts as open only when the very
    last trade row is the CARRY record.
    """
    if wf_trades is None or wf_trades.empty:
        return None
    if last_row_only:
        last_trade = wf_trades.iloc[-1]
        return last_trade.to_dict() if last_trade["Exit Reason"] == "CARRY" else None
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    if carry.empty:
        return None
    return carry.iloc[-1].to_dict()
