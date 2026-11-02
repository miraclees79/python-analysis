# -*- coding: utf-8 -*-
"""
moj_system/core/pipeline_base.py
================================
Template-Method base class shared by every strategy runner (daily, validation,
sweep). A pipeline owns the fixed order of the three phases; concrete
subclasses only supply the variant-specific steps.

    load_data()  ->  run_simulation(market_data)  ->  generate_reports(market_data, simulation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import pandas as pd

MarketDataT = TypeVar("MarketDataT")
SimulationT = TypeVar("SimulationT")


class BaseStrategyPipeline(ABC, Generic[MarketDataT, SimulationT]):
    """Load market data, simulate the strategy, then report on the result."""

    @abstractmethod
    def load_data(self) -> MarketDataT:
        """Fetch every input the simulation needs (typically through ``MarketDataFacade``)."""

    @abstractmethod
    def run_simulation(
        self,
        market_data: MarketDataT,
    ) -> SimulationT:
        """Run the walk-forward / allocation simulation on ``market_data``."""

    @abstractmethod
    def generate_reports(
        self,
        market_data: MarketDataT,
        simulation: SimulationT,
    ) -> dict[str, Any]:
        """Produce the variant's outputs (files, e-mails, robustness reports ...)."""

    def execute_pipeline(self) -> dict[str, Any]:
        """Template method: the only place that fixes the order of the phases."""
        market_data = self.load_data()
        simulation = self.run_simulation(market_data=market_data)
        return self.generate_reports(
            market_data=market_data,
            simulation=simulation,
        )


def drop_duplicate_dates(
    series: pd.Series,
) -> pd.Series:
    """Keep the last observation of every duplicated index entry."""
    return series.loc[~series.index.duplicated(keep="last")]
