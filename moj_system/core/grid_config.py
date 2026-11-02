# -*- coding: utf-8 -*-
"""
moj_system/core/grid_config.py
==============================
Immutable DTOs describing optimisation search spaces.

Engines receive these objects via dependency injection instead of importing
the global ``moj_system.config`` module. The defaults mirror the historical
engine defaults, so an unconfigured engine behaves exactly as before.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_X_GRID: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20)
DEFAULT_Y_GRID: tuple[float, ...] = (0.02, 0.03, 0.05, 0.07, 0.10)
DEFAULT_FAST_GRID: tuple[int, ...] = (50, 75, 100)
DEFAULT_SLOW_GRID: tuple[int, ...] = (150, 200, 250)
DEFAULT_TV_GRID: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20)
DEFAULT_SL_GRID: tuple[float, ...] = (0.05, 0.08, 0.10, 0.15)
DEFAULT_MOM_LB_GRID: tuple[int, ...] = (126, 252)
DEFAULT_N_ATR_GRID: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20)
DEFAULT_ATR_WINDOW: int = 20

# Config-style key -> OptimizationGridConfig field.
_GRID_KEY_TO_FIELD: dict[str, str] = {
    "X_GRID": "x_grid",
    "Y_GRID": "y_grid",
    "FAST_GRID": "fast_grid",
    "SLOW_GRID": "slow_grid",
    "TV_GRID": "tv_grid",
    "SL_GRID": "sl_grid",
    "MOM_LB_GRID": "mom_lookback_grid",
    "N_ATR_GRID": "n_atr_grid",
    "ATR_WINDOW": "atr_window",
}


@dataclass(frozen=True)
class OptimizationGridConfig:
    """Parameter grids searched by ``walk_forward``."""

    x_grid: tuple[float, ...] = DEFAULT_X_GRID
    y_grid: tuple[float, ...] = DEFAULT_Y_GRID
    fast_grid: tuple[int, ...] = DEFAULT_FAST_GRID
    slow_grid: tuple[int, ...] = DEFAULT_SLOW_GRID
    tv_grid: tuple[float, ...] = DEFAULT_TV_GRID
    sl_grid: tuple[float, ...] = DEFAULT_SL_GRID
    mom_lookback_grid: tuple[int, ...] = DEFAULT_MOM_LB_GRID
    n_atr_grid: tuple[float, ...] = DEFAULT_N_ATR_GRID
    atr_window: int = DEFAULT_ATR_WINDOW

    @classmethod
    def from_mapping(
        cls,
        grids: Mapping[str, Any],
        keys: Collection[str] | None = None,
    ) -> OptimizationGridConfig:
        """Build a config from a config-style mapping (``X_GRID``, ``FAST_GRID``, ...).

        Only ``keys`` (all known keys when None) are read; the remaining
        fields keep the engine defaults.
        """
        selected_keys = _GRID_KEY_TO_FIELD.keys() if keys is None else keys
        values: dict[str, Any] = {}
        for key in selected_keys:
            if key not in grids:
                continue
            value = grids[key]
            values[_GRID_KEY_TO_FIELD[key]] = int(value) if key == "ATR_WINDOW" else tuple(value)
        return cls(**values)

    def with_overrides(
        self,
        grids: Mapping[str, Any],
    ) -> OptimizationGridConfig:
        """Return a copy updated with config-style overrides (e.g. ``cfg["grids"]``)."""
        override_keys = [key for key in grids if key in _GRID_KEY_TO_FIELD]
        overrides = OptimizationGridConfig.from_mapping(
            grids=grids,
            keys=override_keys,
        )
        changes = {
            _GRID_KEY_TO_FIELD[key]: getattr(overrides, _GRID_KEY_TO_FIELD[key]) for key in override_keys
        }
        return replace(self, **changes)

    def as_walk_forward_kwargs(
        self,
        use_atr_stop: bool,
    ) -> dict[str, Any]:
        """Legacy keyword form of this config, for callers not yet DI-aware."""
        return {
            "X_grid": list(self.x_grid),
            "Y_grid": list(self.y_grid),
            "fast_grid": list(self.fast_grid),
            "slow_grid": list(self.slow_grid),
            "tv_grid": list(self.tv_grid),
            "sl_grid": list(self.sl_grid),
            "mom_lookback_grid": list(self.mom_lookback_grid),
            "N_atr_grid": list(self.n_atr_grid) if use_atr_stop else None,
            "atr_window": self.atr_window,
        }


@dataclass(frozen=True)
class AllocationOptimizationConfig:
    """Search/gating settings of the allocation walk-forward optimisers.

    A field left as None keeps the engine's own default for that setting.
    The ``asset_caps``/``optional_keys``/``min_delta``/``delta_tol`` fields
    are only used by the N-asset (global) allocator.
    """

    step: float | None = None
    objective: str | None = None
    cooldown_days: int | None = None
    annual_cap: int | None = None
    asset_caps: Mapping[str, float] | None = None
    optional_keys: Collection[str] | None = None
    min_delta: float | None = None
    delta_tol: float | None = None
