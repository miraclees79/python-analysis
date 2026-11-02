# -*- coding: utf-8 -*-
"""
moj_system/reporting/snapshot_builder.py
========================================
Template-method base for building the machine-readable daily snapshot.

``SnapshotBuilder`` owns what every report shares (run date header, rounded
metric blocks, the stop-level arithmetic of an open position); the
strategy-specific parts live in subclasses next to their report module.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pandas as pd

from moj_system.core.state_analytics import compute_atr_val

ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class StopLevels:
    """Stop-loss levels of one open position."""

    peak_price: float
    trail_stop: float
    abs_stop: float
    binding_stop: float
    atr_val: float | None


def compute_stop_levels(
    price_df: pd.DataFrame,
    params: Mapping[str, Any],
    entry_price: float,
    peak_price: float,
) -> StopLevels:
    """Trailing (fixed or ATR-scaled) and absolute stops; the binding one is the higher."""
    stop_param = params.get("stop_param", 0.10)
    stop_loss = params.get("stop_loss", 0.05)

    if params.get("use_atr_stop"):
        atr_val: float | None = compute_atr_val(
            df=price_df,
            atr_window=params.get("atr_window", 20),
        )
        trail_stop = round(number=peak_price * (1.0 - stop_param * float(atr_val)), ndigits=2)
    else:
        atr_val = None
        trail_stop = round(number=peak_price * (1.0 - stop_param), ndigits=2)

    abs_stop = round(number=entry_price * (1.0 - stop_loss), ndigits=2)
    return StopLevels(
        peak_price=peak_price,
        trail_stop=trail_stop,
        abs_stop=abs_stop,
        binding_stop=max(trail_stop, abs_stop),
        atr_val=atr_val,
    )


def round_metrics(
    metrics: Mapping[str, Any] | None,
    ndigits: int = 4,
) -> dict[str, float]:
    return {key: round(number=float(value), ndigits=ndigits) for key, value in (metrics or {}).items()}


def stop_gap_pct(
    binding_stop: float,
    today_price: float,
) -> float:
    return round(number=(binding_stop - today_price) / today_price * 100.0, ndigits=2)


def build_position_block(
    open_position: Mapping[str, Any],
    stops: StopLevels,
    entry_price: float,
    today_price: float,
) -> dict[str, Any]:
    """Position dictionary shared by the pension and global snapshots."""
    return {
        "entry_date": pd.Timestamp(open_position["EntryDate"]).date().isoformat(),
        "entry_price": round(number=entry_price, ndigits=2),
        "today_price": round(number=today_price, ndigits=2),
        "days_in_trade": int(open_position.get("Days", 0)),
        "unrealised_pct": round(number=float(open_position["Return"]) * 100.0, ndigits=2),
        "peak_price": round(number=stops.peak_price, ndigits=2),
        "trail_stop": stops.trail_stop,
        "abs_stop": stops.abs_stop,
        "binding_stop": stops.binding_stop,
        "stop_gap_pct": stop_gap_pct(binding_stop=stops.binding_stop, today_price=today_price),
    }


class SnapshotBuilder(ABC, Generic[ContextT]):
    """Build the snapshot dict: common header first, then the variant-specific body."""

    def __init__(
        self,
        context: ContextT,
        run_date: dt.date,
    ) -> None:
        self.context = context
        self.run_date = run_date

    def build(self) -> dict[str, Any]:
        snapshot = self.build_header()
        snapshot.update(self.build_body())
        return snapshot

    def build_header(self) -> dict[str, Any]:
        return {"run_date": self.run_date.isoformat()}

    @abstractmethod
    def build_body(self) -> dict[str, Any]:
        """Variant-specific snapshot content."""
