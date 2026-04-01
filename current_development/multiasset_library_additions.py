"""
multiasset_library_additions.py
================================
Changes to apply to multiasset_library.py:

ADDITIONS
---------
  build_standard_two_asset_data()  — consolidates the repeated pattern of
    building MMF_EXT, spread_prefilter, yield_prefilter, bond_entry_gate,
    and daily return series that appeared verbatim in:
      multiasset_runfile.py
      multiasset_daily_runfile.py
      twoasset_extended_comparison.py
      twoasset_vs_global_robustness.py
      twoasset_robustness.py
    Eliminates ~150 lines of duplicated setup code.

REMOVALS
--------
  portfolio_returns()               — superseded by allocation_walk_forward().
                                      Never called from any runfile.
  portfolio_equity_from_signals()   — earlier iteration superseded by
                                      allocation_walk_forward().
                                      Never called from any runfile.

  These two functions are marked for deletion.  They are retained in
  multiasset_library.py with a DEPRECATED docstring until the next
  documentation freeze, then removed.
"""

from __future__ import annotations

import logging
import pandas as pd

# These imports are already present in multiasset_library.py
from global_equity_library import build_mmf_extended


# ============================================================
# CONSOLIDATED TWO-ASSET DATA PREPARATION
# ============================================================

def build_standard_two_asset_data(
    wig: pd.DataFrame,
    tbsp: pd.DataFrame,
    mmf: pd.DataFrame,
    wibor1m: pd.DataFrame | None,
    pl10y: pd.DataFrame,
    de10y: pd.DataFrame,
    mmf_floor: str = "1995-01-02",
    yield_prefilter_bp: float = 50.0,
    yield_lookback_days: int = 63,
    spread_change_window: int = 63,
    spread_change_cap_bp: float = 50.0,
) -> dict:
    """
    Build all derived series required by the 2-asset WIG+TBSP+MMF strategy.

    This function consolidates the repeated setup block that appeared in five
    strategy scripts.  It builds the extended MMF, spread and yield pre-filters,
    the combined bond entry gate, and daily return series — everything needed
    before the walk-forward phases begin.

    Parameters
    ----------
    wig        : pd.DataFrame — WIG price DataFrame (stooq format)
    tbsp       : pd.DataFrame — TBSP price DataFrame (stooq format, already
                                loaded with Najwyzszy/Najnizszy columns added)
    mmf        : pd.DataFrame — MMF price DataFrame (stooq format, 2720.n)
    wibor1m    : pd.DataFrame | None — WIBOR 1M price DataFrame; if None the
                                MMF is used without WIBOR backward extension
    pl10y      : pd.DataFrame — PL 10Y yield (stooq format)
    de10y      : pd.DataFrame — DE 10Y yield (stooq format)
    mmf_floor  : str  — ISO date string; MMF backward extension floor date.
                        Should match WIG_DATA_FLOOR in the calling runfile.
    yield_prefilter_bp    : float — yield momentum pre-filter threshold (bp)
    yield_lookback_days   : int   — yield momentum lookback window (trading days)
    spread_change_window  : int   — spread pre-filter rolling window (trading days)
    spread_change_cap_bp  : float — spread pre-filter threshold (bp)

    Returns
    -------
    dict with keys:
        mmf_ext     : pd.DataFrame — extended MMF (WIBOR chain-link backwards)
        bond_gate   : pd.Series    — binary bond entry gate (AND of spread + yield)
        spread_pf   : pd.Series    — raw spread pre-filter series (before alignment)
        yield_pf    : pd.Series    — raw yield pre-filter series (before alignment)
        ret_eq      : pd.Series    — daily WIG returns (dropna'd)
        ret_bd      : pd.Series    — daily TBSP returns (dropna'd)
        ret_mmf     : pd.Series    — daily MMF returns (from plain MMF, not extended)

    Notes
    -----
    ret_mmf uses the un-extended MMF to match the existing runfile behaviour.
    The extended MMF (mmf_ext) is passed to walk_forward as cash_df so that
    in-sample periods starting before 1999 have realistic cash returns.

    The bond_entry_gate is aligned to TBSP's date index (the index of `tbsp`).
    Any dates in pl10y/de10y that do not appear in tbsp are forward-filled.
    """
    # ── Extended MMF (WIBOR chain-link backwards) ─────────────────────────
    if wibor1m is not None:
        mmf_ext = build_mmf_extended(mmf, wibor1m, floor_date=mmf_floor)
        logging.info("MMF extended back to %s", mmf_ext.index.min().date())
    else:
        mmf_ext = mmf
        logging.warning(
            "WIBOR1M unavailable — using standard MMF (starts ~1999). "
            "IS windows before %s will use ret_mmf=0 for cash returns.",
            mmf.index.min().date(),
        )

    # ── Bond entry gate (spread + yield pre-filter AND-ed together) ────────
    spread_pf = build_spread_prefilter(
        pl10y,
        de10y,
        spread_change_window=spread_change_window,
        spread_change_cap_bp=spread_change_cap_bp,
    )
    yield_pf = build_yield_momentum_prefilter(
        pl10y,
        rise_threshold_bp=yield_prefilter_bp,
        lookback_days=yield_lookback_days,
    )

    tbsp_index  = tbsp.index
    spread_gate = (
        spread_pf.reindex(tbsp_index, method="ffill").fillna(1).astype(int)
    )
    yield_gate = (
        yield_pf.reindex(tbsp_index, method="ffill").fillna(1).astype(int)
    )
    bond_gate = (spread_gate & yield_gate).astype(int)
    bond_gate.name = "bond_entry_gate"

    logging.info(
        "Bond entry gate: %.1f%% of days pass  "
        "(spread: %.1f%% pass | yield: %.1f%% pass)",
        bond_gate.mean() * 100,
        spread_gate.mean() * 100,
        yield_gate.mean() * 100,
    )

    # ── Daily return series ────────────────────────────────────────────────
    ret_eq  = wig["Zamkniecie"].pct_change().dropna().rename("ret_eq")
    ret_bd  = tbsp["Zamkniecie"].pct_change().dropna().rename("ret_bd")
    ret_mmf = mmf["Zamkniecie"].pct_change().dropna().rename("ret_mmf")

    logging.info(
        "Return series: equity %d rows | bond %d rows | mmf %d rows",
        len(ret_eq), len(ret_bd), len(ret_mmf),
    )

    return dict(
        mmf_ext   = mmf_ext,
        bond_gate = bond_gate,
        spread_pf = spread_pf,
        yield_pf  = yield_pf,
        ret_eq    = ret_eq,
        ret_bd    = ret_bd,
        ret_mmf   = ret_mmf,
    )
