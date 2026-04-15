# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:28:51 2026

@author: adamg
"""
import numpy as np
import pandas as pd
import logging

def analyze_trades(trades: pd.DataFrame, boundary_exits={"CARRY", "SAMPLE_END"}) -> dict:
    """Analizuje listę transakcji i zwraca kluczowe statystyki."""
    if trades.empty:
        return None

    trades = trades[~trades["Exit Reason"].isin(boundary_exits)].copy()
    if trades.empty:
        return None

    n_cross = trades["CrossWindow"].sum() if "CrossWindow" in trades.columns else 0
    if n_cross > 0:
        logging.info(f"{n_cross} transakcji zostało przeniesionych między oknami walk-forward.")

    loss = abs(trades.loc[trades["Return"] < 0, "Return"].sum())
    profit_factor = np.inf if loss == 0 else (trades.loc[trades["Return"] > 0, "Return"].sum() / loss)

    return {
        "Trades": len(trades),
        "WinRate": (trades["Return"] > 0).mean(),
        "AvgWin": trades.loc[trades["Return"] > 0, "Return"].mean(),
        "AvgLoss": trades.loc[trades["Return"] < 0, "Return"].mean(),
        "ProfitFactor": profit_factor,
        "AvgDays": trades["Days"].mean(),
        "CrossWindow": int(n_cross)
    }