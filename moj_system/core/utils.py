# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:42:46 2026

@author: adamg
"""

# -*- coding: utf-8 -*-
"""
moj_system/core/utils.py
========================
Neutral module for shared functions to break circular imports.
"""
import logging

import pandas as pd

CLOSE_COL = "Zamkniecie"  # stooq close column name used throughout


def signals_to_target_weights(
    sig_equity: int,
    sig_bond: int,
    weights_both_on: dict,
    w_equity_only: float = 1.0,
    w_bond_only: float = 1.0,
) -> dict:
    """
    Map binary asset signals to a target allocation vector.

    Four states:
      Both ON  — use the optimised split (weights_both_on)
      Equity only ON  — 100% equity (or w_equity_only if < 1)
      Bond only ON    — 100% bond   (or w_bond_only   if < 1)
      Both OFF — 100% money-market fund

    Parameters
    ----------
    sig_equity       : int   — 1 = equity signal on, 0 = off
    sig_bond         : int   — 1 = bond signal on,   0 = off
    weights_both_on  : dict  — {'equity': float, 'bond': float, 'mmf': float}
                               sums to 1.0; used when both signals are on
    w_equity_only    : float — equity weight when only equity signal is on
                               (remainder goes to mmf); default 1.0
    w_bond_only      : float — bond weight when only bond signal is on
                               (remainder goes to mmf); default 1.0

    Returns
    -------
    dict — {'equity': float, 'bond': float, 'mmf': float}  sums to 1.0
    """
    if sig_equity and sig_bond:
        return dict(weights_both_on)

    if sig_equity and not sig_bond:
        return {
            "equity": w_equity_only,
            "bond": 0.0,
            "mmf": 1.0 - w_equity_only,
        }

    if sig_bond and not sig_equity:
        return {
            "equity": 0.0,
            "bond": w_bond_only,
            "mmf": 1.0 - w_bond_only,
        }

    # Both off
    return {"equity": 0.0, "bond": 0.0, "mmf": 1.0}


# ============================================================
# LAYER 3 — REALLOCATION GATE
# ============================================================


def reallocation_gate(
    current_weights: dict,
    target_weights: dict,
    last_change_date,
    current_date,
    cooldown_days: int = 10,
    min_delta: float = 0.10,
    annual_cap: int = 12,
    annual_counter: dict = None,
) -> tuple:
    """
    Decide whether to apply a target reallocation or hold current weights.

    A reallocation is blocked if ANY of the following conditions hold:
      1. Fewer than `cooldown_days` calendar days have elapsed since the
         last reallocation (primary governor — almost always the binding
         constraint given ~2-3 reallocations/year).
      2. No asset weight differs from its current value by >= `min_delta`
         (avoids noise-triggered micro-adjustments).
      3. The annual reallocation counter for the current calendar year
         has already reached `annual_cap` (contractual safety valve).

    Parameters
    ----------
    current_weights  : dict  — {'equity': float, 'bond': float, 'mmf': float}
    target_weights   : dict  — same structure, desired new allocation
    last_change_date : date-like or None — date of the most recent accepted
                       reallocation; None if no reallocation has occurred yet
    current_date     : date-like — today's date in the simulation loop
    cooldown_days    : int   — minimum calendar days between reallocations
    min_delta        : float — minimum weight change to trigger reallocation
    annual_cap       : int   — maximum reallocations permitted per calendar year
    annual_counter   : dict  — mutable {year: count} tracking reallocations;
                       pass the same dict object on every call so it persists
                       across the simulation.  If None, no cap is enforced.

    Returns
    -------
    (accepted_weights, reallocated, annual_counter)
      accepted_weights : dict  — either target_weights (gate passed) or
                                 current_weights (gate blocked)
      reallocated      : bool  — True if the reallocation was accepted
      annual_counter   : dict  — updated counter (same object as input)
    """

    if annual_counter is None:
        annual_counter = {}

    current_date = pd.Timestamp(current_date)
    year = current_date.year

    # --- Guard 1: cooldown ---
    if last_change_date is not None:
        days_since = (current_date - pd.Timestamp(last_change_date)).days
        if days_since < cooldown_days:
            logging.debug(
                "Gate BLOCKED (cooldown): %d days since last change (need %d)",
                days_since,
                cooldown_days,
            )
            return current_weights, False, annual_counter

    # --- Guard 2: minimum delta ---
    assets = set(current_weights) | set(target_weights)
    max_delta = max(abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0)) for a in assets)
    if max_delta < min_delta:
        logging.debug(
            "Gate BLOCKED (min_delta): max weight change %.1f%% < %.1f%%",
            max_delta * 100,
            min_delta * 100,
        )
        return current_weights, False, annual_counter

    # --- Guard 3: annual cap ---
    count_this_year = annual_counter.get(year, 0)
    if count_this_year >= annual_cap:
        logging.warning(
            "Gate BLOCKED (annual cap): %d reallocations already in %d",
            count_this_year,
            year,
        )
        return current_weights, False, annual_counter

    # --- Gate passed ---
    annual_counter[year] = count_this_year + 1
    logging.debug(
        "Gate PASSED: equity %.0f%% → %.0f%%, bond %.0f%% → %.0f%%, "
        "mmf %.0f%% → %.0f%%  [reallocation #%d in %d]",
        current_weights.get("equity", 0) * 100,
        target_weights.get("equity", 0) * 100,
        current_weights.get("bond", 0) * 100,
        target_weights.get("bond", 0) * 100,
        current_weights.get("mmf", 0) * 100,
        target_weights.get("mmf", 0) * 100,
        annual_counter[year],
        year,
    )
    return target_weights, True, annual_counter


# ============================================================
# MMF BACKWARD EXTENSION  (WIBOR1M chain-link)
# ============================================================


def build_mmf_extended(
    mmf_df: pd.DataFrame,
    wibor1m_df: pd.DataFrame,
    floor_date: str = "1995-01-02",
) -> pd.DataFrame:
    """
    Extend the MMF (2720.n) price series backwards using WIBOR 1M.

    The MMF fund (Goldman Sachs Konserwatywny) has NAV data from ~1999.
    For IS windows starting before 1999 (e.g. Mode A / global_equity with
    WIG data from 1994), the MMF series is used as the cash return when
    a signal is OFF.  Without extension, those early days get ret_mmf=NaN,
    which is treated as 0 (no cash return) in the allocation layer.

    Extension method
    ----------------
    WIBOR 1M (PLOPLN1M on stooq) is a daily-published overnight-style
    reference rate expressed as an annual percentage. Each day's accrual is:

        daily_factor = (1 + wibor_rate / 100) ^ (1 / 252)

    where 252 is the assumed number of business days per year.  This is
    consistent with how money-market NAVs accrue — they compound the
    overnight/short-term rate every business day.

    The extension is chain-linked backwards from the first available MMF
    observation: the synthetic level on day t < mmf_start is computed as:

        price_t = price_mmf_start / product( daily_factor_{t+1..mmf_start} )

    so the synthetic series joins seamlessly to the real series on the
    first MMF date (same level, same return going forward).

    WIBOR coverage
    --------------
    PLOPLN1M on stooq is available from approximately 1992.  Any gap days
    within the WIBOR series are forward-filled (rate unchanged) before
    computing daily accruals.

    Floor date
    ----------
    floor_date clips the extended series at a hard lower bound. Default
    is 1994-10-03, the date from which the WIG index has reliable daily
    quotes.  Earlier WIG data has multi-day gaps that add noise; the WIG
    data floor in the runfile should match this value.

    Parameters
    ----------
    mmf_df     : pd.DataFrame  — stooq-format MMF DataFrame (2720.n)
    wibor1m_df : pd.DataFrame  — stooq-format WIBOR 1M DataFrame (PLOPLN1M)
    floor_date : str           — hard lower bound for the extended series
                                 (ISO date string, default "1994-10-03")

    Returns
    -------
    pd.DataFrame  — stooq-format DataFrame with CLOSE_COL, Najwyzszy,
                    Najnizszy; index covers floor_date to latest MMF date.
                    Real MMF data is preserved exactly from mmf_start onwards.

    Notes
    -----
    - WIBOR 1M is an overnight-to-1-month interbank offer rate, not a
      NAV-accrual rate.  It is used here as a proxy for the short-term PLN
      cash return, which is the best available daily proxy for 1994-1999.
    - The chain-link is backward only — the real MMF data is never modified.
    - If WIBOR data does not cover the requested floor_date, the extension
      starts from the earliest available WIBOR date.
    """
    floor_ts = pd.Timestamp(floor_date)
    mmf_close = mmf_df[CLOSE_COL].sort_index().dropna()
    mmf_start = mmf_close.index.min()

    # If MMF already covers the floor, no extension needed
    if mmf_start <= floor_ts:
        logging.info(
            "build_mmf_extended: MMF starts %s, at or before floor %s — no extension needed.",
            mmf_start.date(),
            floor_ts.date(),
        )
        out = mmf_df.copy()
        out = out.loc[out.index >= floor_ts]
        return out

    # ── Prepare WIBOR series ───────────────────────────────────────────────
    wibor_close = wibor1m_df[CLOSE_COL].sort_index().dropna()
    # Clip WIBOR to the extension window [floor_date, mmf_start)
    wibor_ext = wibor_close.loc[(wibor_close.index >= floor_ts) & (wibor_close.index < mmf_start)]

    if wibor_ext.empty:
        logging.warning(
            "build_mmf_extended: WIBOR 1M has no data between %s and %s — "
            "cannot extend. Returning original MMF from its start date.",
            floor_ts.date(),
            mmf_start.date(),
        )
        return mmf_df.loc[mmf_df.index >= floor_ts]

    # Forward-fill any gaps within the WIBOR extension window
    # (rate is published on business days; weekends/holidays carry last rate)
    ext_idx = pd.bdate_range(start=floor_ts, end=mmf_start - pd.Timedelta(days=1))
    wibor_ext_full = wibor_ext.reindex(ext_idx, method="ffill").bfill()

    # ── Compute daily accrual factors ─────────────────────────────────────
    # WIBOR is expressed as annual % (e.g. 25.0 = 25% per annum)
    annual_rate = wibor_ext_full / 100.0
    daily_factors = (1.0 + annual_rate) ** (1.0 / 252)

    # ── Chain-link backwards from first MMF observation ───────────────────
    # cumulative product going FORWARD from floor → mmf_start gives the
    # total growth factor over the extension window.
    # To chain-link backwards: divide mmf_start price by the forward cumprod.
    anchor_price = mmf_close.iloc[0]  # price on mmf_start date

    # daily_factors is indexed floor → (mmf_start - 1 bday)
    # cumprod going forward: at day t, cumprod(t) = product of factors[floor..t]
    forward_cumprod = daily_factors.cumprod()

    # Scale so that cumprod(mmf_start - 1 bday) × next_factor = anchor_price
    # i.e. synthetic[mmf_start] would equal anchor_price
    # synthetic[t] = anchor_price / (total_forward_cumprod / forward_cumprod[t])
    total_forward = forward_cumprod.iloc[-1] * daily_factors.iloc[-1]
    # Actually simpler: work in returns space, then build price backwards
    # Returns during extension: r_t = daily_factor_t - 1
    ext_returns = daily_factors - 1.0

    # Build the synthetic price series backwards from the anchor:
    # price[mmf_start] = anchor_price (known)
    # price[t-1] = price[t] / daily_factor[t]
    # We have factors indexed [floor_date .. mmf_start-1], so reverse:
    factors_reverse = daily_factors.iloc[::-1]  # mmf_start-1 down to floor
    price_reverse = pd.Series(index=factors_reverse.index, dtype=float)
    prev_price = anchor_price
    for date in factors_reverse.index:
        prev_price = prev_price / daily_factors.loc[date]
        price_reverse.loc[date] = prev_price

    synthetic_prices = price_reverse.iloc[::-1]  # back to chronological

    # ── Combine synthetic extension + real MMF ────────────────────────────
    # Jawne przekazanie obiektów do połączenia
    combined = pd.concat(objs=[synthetic_prices, mmf_close], axis=0)
    combined = combined.sort_index()

    # Sanity check zastępujący niebezpieczny 'assert'
    if combined.index.min() < floor_ts:
        error_msg = (
            f"Data integrity error: Combined series starts at {combined.index.min().date()} "
            f"which is before requested floor {floor_ts.date()}"
        )
        logging.error(msg=error_msg)
        raise ValueError(error_msg)

    # Build stooq-format output z jawnym przekazaniem indeksu
    out = pd.DataFrame(index=combined.index)
    out[CLOSE_COL] = combined
    out["Najwyzszy"] = combined
    out["Najnizszy"] = combined
    out.index.name = "Data"

    # Verify the seam: return at mmf_start should match the actual first MMF return
    ext_rows = out.loc[out.index < mmf_start]
    real_rows = out.loc[out.index >= mmf_start]

    logging.info(
        "build_mmf_extended: synthetic extension %s to %s (%d days) "
        "chain-linked to real MMF %s to %s (%d days).  "
        "Anchor price=%.4f.  Extension start price=%.4f.",
        ext_rows.index.min().date() if not ext_rows.empty else "n/a",
        ext_rows.index.max().date() if not ext_rows.empty else "n/a",
        len(ext_rows),
        real_rows.index.min().date() if not real_rows.empty else "n/a",
        real_rows.index.max().date() if not real_rows.empty else "n/a",
        len(real_rows),
        anchor_price,
        out[CLOSE_COL].iloc[0],
    )
    return out
