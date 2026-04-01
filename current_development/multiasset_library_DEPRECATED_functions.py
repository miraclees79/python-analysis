"""
multiasset_library_DEPRECATED_functions.py
===========================================
Documents the two functions removed from multiasset_library.py.

REMOVED: portfolio_returns()
-----------------------------
An early-iteration function that computed daily portfolio returns from
three asset return series and a time-varying weights series.  Superseded
by allocation_walk_forward(), which implements the full simulation inline
(signal-gating, reallocation gate, per-window weight optimisation, and
chain-linked equity slices).

portfolio_returns() was never called from any runfile in the codebase.
Its signature accepted pre-computed weights_series and produced a return
series only — it had no gate logic and could not produce the weights_series
it consumed.

REMOVED: portfolio_equity_from_signals()
-----------------------------------------
An earlier iteration of the multi-asset portfolio simulation that
iterated day-by-day applying reallocation_gate on each date.  The full
walk-forward variant (allocation_walk_forward()) superseded it by adding
per-window IS optimisation of both-on weights and continuous gate state
across window boundaries.

portfolio_equity_from_signals() was never called from any runfile in the
codebase.

Both functions are preserved here for reference until the next
documentation freeze, then this file is deleted.
"""

import pandas as pd
import numpy as np
from multiasset_library import (
    signals_to_target_weights,
    reallocation_gate,
)


# ---------------------------------------------------------------------------
# ARCHIVED: portfolio_returns()
# ---------------------------------------------------------------------------

def portfolio_returns(
    equity_returns: pd.Series,
    bond_returns:   pd.Series,
    mmf_returns:    pd.Series,
    weights_series: pd.Series,
) -> pd.Series:
    """
    ARCHIVED — superseded by allocation_walk_forward(); never called from
    any runfile.

    Compute daily portfolio returns from three asset return series and a
    time-varying weights series.
    """
    common = (equity_returns.index
              .intersection(bond_returns.index)
              .intersection(mmf_returns.index)
              .intersection(weights_series.index))

    eq_r = equity_returns.loc[common]
    bd_r = bond_returns.loc[common]
    mf_r = mmf_returns.loc[common]
    ws   = weights_series.loc[common]

    port_r = pd.Series(index=common, dtype=float)
    for date in common:
        w = ws[date]
        port_r[date] = (
            w.get("equity", 0.0) * eq_r[date]
            + w.get("bond",   0.0) * bd_r[date]
            + w.get("mmf",    1.0) * mf_r[date]
        )
    return port_r


# ---------------------------------------------------------------------------
# ARCHIVED: portfolio_equity_from_signals()
# ---------------------------------------------------------------------------

def portfolio_equity_from_signals(
    equity_returns:   pd.Series,
    bond_returns:     pd.Series,
    mmf_returns:      pd.Series,
    sig_equity:       pd.Series,
    sig_bond:         pd.Series,
    weights_both_on:  dict,
    cooldown_days:    int   = 10,
    min_delta:        float = 0.10,
    annual_cap:       int   = 12,
    w_equity_only:    float = 1.0,
    w_bond_only:      float = 1.0,
) -> tuple:
    """
    ARCHIVED — superseded by allocation_walk_forward(); never called from
    any runfile.

    Full simulation: daily signals → gate → weights → portfolio equity curve.
    Iterates day-by-day applying reallocation_gate on each date.
    """
    common = (equity_returns.index
              .intersection(bond_returns.index)
              .intersection(mmf_returns.index)
              .intersection(sig_equity.index)
              .intersection(sig_bond.index))
    common = common.sort_values()

    eq_r = equity_returns.loc[common]
    bd_r = bond_returns.loc[common]
    mf_r = mmf_returns.loc[common]
    s_eq = sig_equity.reindex(common).fillna(0).astype(int)
    s_bd = sig_bond.reindex(common).fillna(0).astype(int)

    current_weights  = {"equity": 0.0, "bond": 0.0, "mmf": 1.0}
    last_change_date = None
    annual_counter   = {}

    weights_series   = pd.Series(index=common, dtype=object)
    reallocation_log = []
    portfolio_r      = pd.Series(index=common, dtype=float)

    for date in common:
        target = signals_to_target_weights(
            sig_equity=int(s_eq[date]),
            sig_bond=int(s_bd[date]),
            weights_both_on=weights_both_on,
            w_equity_only=w_equity_only,
            w_bond_only=w_bond_only,
        )

        accepted, reallocated, annual_counter = reallocation_gate(
            current_weights  = current_weights,
            target_weights   = target,
            last_change_date = last_change_date,
            current_date     = date,
            cooldown_days    = cooldown_days,
            min_delta        = min_delta,
            annual_cap       = annual_cap,
            annual_counter   = annual_counter,
        )

        if reallocated:
            reallocation_log.append({
                "Date":          date,
                "equity_before": current_weights["equity"],
                "bond_before":   current_weights["bond"],
                "mmf_before":    current_weights["mmf"],
                "equity_after":  accepted["equity"],
                "bond_after":    accepted["bond"],
                "mmf_after":     accepted["mmf"],
            })
            current_weights  = accepted
            last_change_date = date

        weights_series[date] = dict(current_weights)
        w = current_weights
        portfolio_r[date] = (
            w["equity"] * eq_r[date]
            + w["bond"]   * bd_r[date]
            + w["mmf"]    * mf_r[date]
        )

    equity_curve = (1 + portfolio_r).cumprod()
    equity_curve = equity_curve / equity_curve.iloc[0]

    return equity_curve, weights_series, reallocation_log
