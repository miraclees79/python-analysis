# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:28:31 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/core/research.py
===========================
Core engine for batch research, window generation, and strategy comparison.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple

def get_common_oos_start(assets_data: dict, window_configs: List[Tuple[int, int]]) -> pd.Timestamp:
    """
    Calculates the latest possible start date for Out-Of-Sample period
    to ensure all tested configurations cover the exact same time range.
    """
    latest_starts = []
    
    for df in assets_data.values():
        data_start = df.index.min()
        for train_y, _ in window_configs:
            # Each config needs at least 'train_y' of history
            potential_start = data_start + pd.DateOffset(years=train_y)
            latest_starts.append(potential_start)
            
    # The common start is the latest of all required starts
    common_start = max(latest_starts)
    logging.info(f"Calculated Common OOS Start Date: {common_start.date()}")
    return common_start

def rank_research_results(results_df: pd.DataFrame, objective: str = "CalMAR") -> pd.DataFrame:
    """Ranks sweep results based on primary objective and robustness."""
    if results_df.empty: return results_df
    
    # Simple ranking logic - can be extended with 'Robustness' scores
    sorted_df = results_df.sort_values(by=objective, ascending=False).reset_index(drop=True)
    return sorted_df