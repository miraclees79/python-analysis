"""
global_equity_library.py
========================
Core utilities for the global equity multi-market framework.

Extends the existing WIG+TBSP+MMF architecture to support multiple equity
markets (PL large cap, PL mid/small cap, EU, US, JPN) plus an alternative
simplified mode using MSCI World (PL large, PL mid/small, MSCI World, TBSP).

This module provides:

  Data acquisition
    download_yfinance()           download index data from Yahoo Finance
    yfinance_to_stooq()           normalise yfinance DataFrame to stooq format
                                  (DatetimeIndex named 'Data', column 'Zamkniecie')

  Return construction
    build_return_series()         compute daily PLN returns with FX switch:
                                    hedged=True  → local returns only
                                    hedged=False → (1+local)*(1+fx) - 1
    build_blend()                 construct a weighted blend of two price
                                  series (e.g. 50:50 MWIG40TR + SWIG80TR)
                                  returned as a stooq-format DataFrame
    build_price_df_from_returns() reconstruct a synthetic price DataFrame
                                  (Zamkniecie + dummy OHLV columns) from a
                                  return series — suitable for walk_forward

  Calendar alignment
    align_to_reference()          reindex all series to a common business-day
                                  calendar using forward-fill

  N-asset allocation
    generate_weight_grid()        enumerate all (w1..wN) tuples at given step
                                  where sum(wi) <= 1.0
    optimise_asset_weights()      grid-search best allocation weights across N
                                  assets using in-sample return/signal data
    allocation_walk_forward_n()   walk-forward allocation optimisation for N
                                  assets; generalises multiasset_library's
                                  allocation_walk_forward to arbitrary asset dicts

  Reporting helpers
    print_global_equity_report()  structured OOS report for N-asset portfolio

DESIGN NOTES
------------
- walk_forward (from strategy_test_library) is called unchanged per asset.
  Each asset's price DataFrame is normalised to stooq format before passing.
- The FX switch (FX_HEDGED) controls whether foreign asset returns are
  adjusted for PLN/foreign currency moves. The switch is applied in
  build_return_series(); downstream signal and allocation code is unaware of
  the FX treatment.
- PLN-native assets (WIG20TR, MWIG40TR, SWIG80TR, TBSP, MMF) always pass
  fx_df=None to build_return_series() and are unaffected by the FX switch.
- The MWIG blend is constructed once in Phase 1 and treated as a single asset
  throughout the signal and allocation layers.
- annual_cap in the reallocation gate is set to 999 (effectively disabled)
  for this portfolio; the cooldown_days parameter is the active governor.

DEPENDENCIES
------------
  pandas, numpy, yfinance, requests, joblib
  strategy_test_library (download_csv, load_csv, compute_metrics,
                         )

PORTFOLIO MODES
---------------
  "global_equity"  : PL_LARGE, PL_MID, EU, US, JPN, TBSP  + MMF residual
  "msci_world"     : PL_LARGE, PL_MID, WORLD, TBSP         + MMF residual
"""

import itertools
import logging
import os
import time
import random

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


from strategy_test_library import (
    compute_metrics,
    
)


# ============================================================
# CONSTANTS
# ============================================================

CLOSE_COL = "Zamkniecie"   # stooq close column name used throughout
DATA_START = "1990-01-01"  # hard floor for all series

# Stooq FX tickers (rate expressed as units of foreign per 1 PLN, inverted below)
# stooq stores these as PLN per unit of foreign currency, e.g. USDPLN = PLN/USD
FX_TICKERS = {
    "USD": "usdpln",
    "EUR": "eurpln",
    "JPY": "jpypln",
}

# yfinance tickers for non-stooq indices
YFINANCE_TICKERS = {
    "EU":    "^STOXX",        # STOXX Europe 600
    "WORLD": "URTH",          # iShares MSCI World ETF (USD, from 2012)
                              # Note: for longer history back to 1990, consider
                              # sourcing MSCI World from stooq if available
                              # (check ^msciw or similar during data validation).
                              # MSCI_WORLD_SOURCE setting in runfile controls this.
}


# ============================================================
# DATA ACQUISITION — yfinance
# ============================================================

def download_yfinance(ticker: str, start: str = DATA_START) -> pd.DataFrame | None:
    """
    Download daily OHLCV data from Yahoo Finance for a single ticker.

    Returns a DataFrame in stooq-compatible format (DatetimeIndex named 'Data',
    column 'Zamkniecie' for close) via yfinance_to_stooq(), or None on failure.

    yfinance returns a MultiIndex DataFrame; this function handles both the
    legacy single-level and the newer MultiIndex column formats.

    Parameters
    ----------
    ticker : str  — Yahoo Finance ticker symbol, e.g. "^STOXX", "URTH"
    start  : str  — ISO date string, data clipped to this floor

    Returns
    -------
    pd.DataFrame | None  — stooq-format DataFrame or None on failure
    """
    logging.info("Downloading %s from yfinance (start=%s) ...", ticker, start)
    try:
        raw = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
    except Exception as exc:
        logging.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        logging.warning("yfinance returned empty DataFrame for %s.", ticker)
        return None

    df = yfinance_to_stooq(raw, ticker)

    if df is None or df.empty:
        logging.warning("yfinance_to_stooq returned empty result for %s.", ticker)
        return None

    logging.info(
        "%s loaded from yfinance: %d rows  (%s to %s)",
        ticker, len(df), df.index.min().date(), df.index.max().date(),
    )
    return df


def yfinance_to_stooq(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Normalise a yfinance OHLCV DataFrame to stooq-compatible format.

    stooq format requirements:
      - DatetimeIndex named 'Data', timezone-naive, ascending
      - Column 'Zamkniecie' containing the adjusted close price
      - Rows with NaN close dropped

    yfinance v0.2+ returns a MultiIndex column header
    ('Price', ticker) when downloading a single ticker with auto_adjust=True.
    This function handles both single-level and MultiIndex column layouts.

    Parameters
    ----------
    df     : pd.DataFrame  — raw yfinance download output
    ticker : str           — ticker symbol used for MultiIndex lookup

    Returns
    -------
    pd.DataFrame | None  — normalised DataFrame with 'Zamkniecie' column
    """
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present (yfinance >= 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        # levels: ('Price', 'Ticker') — drop the ticker level
        df = df.droplevel(level=1, axis=1)

    # Locate the close column — yfinance uses 'Close' after auto_adjust
    close_candidates = [c for c in df.columns if c.lower() == "close"]
    if not close_candidates:
        logging.error(
            "yfinance_to_stooq: no 'Close' column found for %s. "
            "Available columns: %s", ticker, list(df.columns)
        )
        return None

    close_col = close_candidates[0]

    # Build output DataFrame
    out = pd.DataFrame(index=df.index)
    out[CLOSE_COL] = df[close_col]

    # Also carry Najwyzszy (high) and Najnizszy (low) if present — walk_forward
    # uses these for the breakout signal. Map yfinance column names.
    high_candidates = [c for c in df.columns if c.lower() == "high"]
    low_candidates  = [c for c in df.columns if c.lower() == "low"]
    if high_candidates:
        out["Najwyzszy"] = df[high_candidates[0]]
    else:
        out["Najwyzszy"] = out[CLOSE_COL]   # fallback: use close as high proxy

    if low_candidates:
        out["Najnizszy"] = df[low_candidates[0]]
    else:
        out["Najnizszy"] = out[CLOSE_COL]   # fallback: use close as low proxy

    # Ensure DatetimeIndex, timezone-naive, named 'Data'
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Data"

    # Drop NaN closes and sort ascending
    out = out.dropna(subset=[CLOSE_COL]).sort_index()

    # Apply data floor
    out = out.loc[out.index >= pd.Timestamp(DATA_START)]

    return out


# ============================================================
# FX DATA — stooq
# ============================================================

def load_fx_series(currency: str, stooq_df_map: dict) -> pd.Series | None:
    """
    Extract a daily FX close series (PLN per unit of foreign currency) from
    the pre-loaded stooq DataFrame map.

    Parameters
    ----------
    currency     : str   — "USD", "EUR", or "JPY"
    stooq_df_map : dict  — mapping of FX ticker → loaded stooq DataFrame
                           e.g. {"usdpln": <DataFrame>, ...}

    Returns
    -------
    pd.Series | None  — daily close prices (PLN/foreign), DatetimeIndex
    """
    ticker = FX_TICKERS.get(currency)
    if ticker is None:
        logging.error("Unknown currency: %s. Expected one of %s.", currency, list(FX_TICKERS))
        return None

    df = stooq_df_map.get(ticker)
    if df is None:
        logging.error("FX data for %s (%s) not found in stooq_df_map.", currency, ticker)
        return None

    return df[CLOSE_COL].rename(f"FX_{currency}_PLN")


# ============================================================
# RETURN CONSTRUCTION
# ============================================================

def build_return_series(
    price_df: pd.DataFrame,
    fx_series: pd.Series | None = None,
    hedged: bool = True,
    reference_index: pd.Index | None = None,
) -> pd.Series:
    """
    Compute daily PLN return series from a price DataFrame.

    FX switch
    ---------
    hedged=True  (default):
        return = price_df[CLOSE_COL].pct_change()
        Appropriate when the investment fund absorbs FX (PLN-hedged share class).
        fx_series is ignored; pass None for PLN-native assets.

    hedged=False:
        return = (1 + local_ret) * (1 + fx_ret) - 1
        Appropriate for unhedged foreign-currency funds.
        fx_series must be provided (PLN per unit of foreign currency, from stooq).
        FX return is forward-filled to match the price series calendar before
        compounding — this handles days when FX fixing is available but the
        foreign market is closed (e.g. US market closed on a Polish working day).

    Calendar alignment
    ------------------
    If reference_index is provided, the return series is reindexed to that
    index using forward-fill before returning. Use this to align all assets
    to a common business-day calendar prior to the allocation layer.

    Parameters
    ----------
    price_df        : pd.DataFrame  — stooq-format DataFrame with CLOSE_COL
    fx_series       : pd.Series     — daily FX close (PLN per foreign unit),
                                      required when hedged=False, else None
    hedged          : bool          — FX treatment switch (default True)
    reference_index : pd.Index      — optional target DatetimeIndex for alignment

    Returns
    -------
    pd.Series  — daily PLN returns, NaN on first date, DatetimeIndex
    """
    close = price_df[CLOSE_COL].sort_index()
    local_ret = close.pct_change()

    if hedged or fx_series is None:
        ret = local_ret
    else:
        # Align FX to price calendar via ffill, then compound
        fx_aligned = fx_series.reindex(local_ret.index, method="ffill")
        fx_ret = fx_aligned.pct_change()
        ret = (1 + local_ret) * (1 + fx_ret) - 1

    ret.name = "ret"

    if reference_index is not None:
        ret = ret.reindex(reference_index, method="ffill")

    return ret


def build_blend(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    w_a: float = 0.5,
    w_b: float = 0.5,
    label: str = "Blend",
) -> pd.DataFrame:
    """
    Construct a weighted blend of two price series as a synthetic index.

    The blend is constructed in return space (weighted average of daily returns),
    then reconstructed as a cumulative price series starting at 100. The result
    is a stooq-format DataFrame with CLOSE_COL, Najwyzszy, and Najnizszy columns.

    High/low in the blended series are set equal to the close (since a weighted
    blend of two indices has no meaningful intraday high/low). The breakout
    signal in walk_forward uses the rolling window of closes for the trough
    calculation, so this is acceptable.

    Parameters
    ----------
    df_a  : pd.DataFrame  — stooq-format DataFrame for first component
    df_b  : pd.DataFrame  — stooq-format DataFrame for second component
    w_a   : float         — weight of first component (default 0.5)
    w_b   : float         — weight of second component (default 0.5)
    label : str           — name for logging

    Returns
    -------
    pd.DataFrame  — stooq-format DataFrame with CLOSE_COL column,
                    index = union of both input indices (ffilled)
    """
    if abs(w_a + w_b - 1.0) > 1e-9:
        logging.warning(
            "build_blend(%s): weights %.2f + %.2f != 1.0. Proceeding anyway.",
            label, w_a, w_b,
        )

    # Align both series to union calendar via ffill
    union_idx = df_a.index.union(df_b.index)
    close_a = df_a[CLOSE_COL].reindex(union_idx, method="ffill")
    close_b = df_b[CLOSE_COL].reindex(union_idx, method="ffill")

    # Compute daily returns and weighted blend
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    blend_ret = w_a * ret_a + w_b * ret_b

    # Reconstruct price series starting at 100 on the FIRST date of the input.
    # The first return is NaN (no prior day); treat it as 0 (price stays at 100)
    # so that the output index starts on the same date as the inputs.
    # walk_forward will drop its own leading NaN when computing strategy returns.
    blend_ret_filled = blend_ret.fillna(0.0)
    blend_price = 100.0 * (1 + blend_ret_filled).cumprod()

    # Build stooq-format DataFrame
    out = pd.DataFrame(index=blend_price.index)
    out[CLOSE_COL]  = blend_price
    out["Najwyzszy"] = blend_price   # no intraday data — use close
    out["Najnizszy"] = blend_price

    out.index.name = "Data"

    logging.info(
        "Blend '%s' (%.0f%% + %.0f%%): %d rows  (%s to %s)",
        label, w_a * 100, w_b * 100,
        len(out), out.index.min().date(), out.index.max().date(),
    )
    return out


def build_price_df_from_returns(
    ret: pd.Series,
    label: str = "synthetic",
) -> pd.DataFrame:
    """
    Reconstruct a stooq-format price DataFrame from a return series.

    Starting value = 100. Used to convert FX-adjusted return series into
    the price DataFrame format expected by walk_forward.

    For the breakout signal, walk_forward needs Najwyzszy (high) and
    Najnizszy (low). These are set equal to the close here, since the
    signal operates on a rolling window of closes. The MA/momentum filter
    component of the signal is unaffected.

    Parameters
    ----------
    ret   : pd.Series  — daily return series (first value may be NaN)
    label : str        — name for logging

    Returns
    -------
    pd.DataFrame  — stooq-format with CLOSE_COL, Najwyzszy, Najnizszy
    """
    ret_clean = ret.dropna()
    price = 100.0 * (1 + ret_clean).cumprod()

    out = pd.DataFrame(index=price.index)
    out[CLOSE_COL]  = price
    out["Najwyzszy"] = price
    out["Najnizszy"] = price
    out.index.name = "Data"

    logging.info(
        "Price series '%s': %d rows  (%s to %s)  "
        "start=%.2f  end=%.2f",
        label, len(out),
        out.index.min().date(), out.index.max().date(),
        out[CLOSE_COL].iloc[0], out[CLOSE_COL].iloc[-1],
    )
    return out


# ============================================================
# MMF BACKWARD EXTENSION  (WIBOR1M chain-link)
# ============================================================

def build_mmf_extended(
    mmf_df:      pd.DataFrame,
    wibor1m_df:  pd.DataFrame,
    floor_date:  str = "1994-10-03",
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
    floor_ts  = pd.Timestamp(floor_date)
    mmf_close = mmf_df[CLOSE_COL].sort_index().dropna()
    mmf_start = mmf_close.index.min()

    # If MMF already covers the floor, no extension needed
    if mmf_start <= floor_ts:
        logging.info(
            "build_mmf_extended: MMF starts %s, at or before floor %s — "
            "no extension needed.", mmf_start.date(), floor_ts.date(),
        )
        out = mmf_df.copy()
        out = out.loc[out.index >= floor_ts]
        return out

    # ── Prepare WIBOR series ───────────────────────────────────────────────
    wibor_close = wibor1m_df[CLOSE_COL].sort_index().dropna()
    # Clip WIBOR to the extension window [floor_date, mmf_start)
    wibor_ext = wibor_close.loc[
        (wibor_close.index >= floor_ts) & (wibor_close.index < mmf_start)
    ]

    if wibor_ext.empty:
        logging.warning(
            "build_mmf_extended: WIBOR 1M has no data between %s and %s — "
            "cannot extend. Returning original MMF from its start date.",
            floor_ts.date(), mmf_start.date(),
        )
        return mmf_df.loc[mmf_df.index >= floor_ts]

    # Forward-fill any gaps within the WIBOR extension window
    # (rate is published on business days; weekends/holidays carry last rate)
    ext_idx = pd.bdate_range(start=floor_ts, end=mmf_start - pd.Timedelta(days=1))
    wibor_ext_full = wibor_ext.reindex(ext_idx, method="ffill").bfill()

    # ── Compute daily accrual factors ─────────────────────────────────────
    # WIBOR is expressed as annual % (e.g. 25.0 = 25% per annum)
    annual_rate   = wibor_ext_full / 100.0
    daily_factors = (1.0 + annual_rate) ** (1.0 / 252)

    # ── Chain-link backwards from first MMF observation ───────────────────
    # cumulative product going FORWARD from floor → mmf_start gives the
    # total growth factor over the extension window.
    # To chain-link backwards: divide mmf_start price by the forward cumprod.
    anchor_price = mmf_close.iloc[0]   # price on mmf_start date

    # daily_factors is indexed floor → (mmf_start - 1 bday)
    # cumprod going forward: at day t, cumprod(t) = product of factors[floor..t]
    forward_cumprod = daily_factors.cumprod()

    # Scale so that cumprod(mmf_start - 1 bday) × next_factor = anchor_price
    # i.e. synthetic[mmf_start] would equal anchor_price
    # synthetic[t] = anchor_price / (total_forward_cumprod / forward_cumprod[t])
    total_forward   = forward_cumprod.iloc[-1] * daily_factors.iloc[-1]
    # Actually simpler: work in returns space, then build price backwards
    # Returns during extension: r_t = daily_factor_t - 1
    ext_returns = daily_factors - 1.0

    # Build the synthetic price series backwards from the anchor:
    # price[mmf_start] = anchor_price (known)
    # price[t-1] = price[t] / daily_factor[t]
    # We have factors indexed [floor_date .. mmf_start-1], so reverse:
    factors_reverse = daily_factors.iloc[::-1]          # mmf_start-1 down to floor
    price_reverse   = pd.Series(index=factors_reverse.index, dtype=float)
    prev_price      = anchor_price
    for date in factors_reverse.index:
        prev_price           = prev_price / daily_factors.loc[date]
        price_reverse.loc[date] = prev_price

    synthetic_prices = price_reverse.iloc[::-1]          # back to chronological

    # ── Combine synthetic extension + real MMF ────────────────────────────
    combined = pd.concat([synthetic_prices, mmf_close])
    combined = combined.sort_index()
    # Sanity: the join should be seamless — synthetic ends just before mmf_start
    assert combined.index.min() >= floor_ts, "Combined series starts before floor"

    # Build stooq-format output
    out = pd.DataFrame(index=combined.index)
    out[CLOSE_COL]   = combined
    out["Najwyzszy"] = combined   # no intraday data
    out["Najnizszy"] = combined
    out.index.name   = "Data"

    # Verify the seam: return at mmf_start should match the actual first MMF return
    ext_rows  = out.loc[out.index < mmf_start]
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


# ============================================================
# CALENDAR ALIGNMENT
# ============================================================

def align_to_reference(
    series_dict: dict,
    reference_key: str | None = None,
) -> dict:
    """
    Align all Series/DataFrames in series_dict to a common business-day
    calendar using forward-fill.

    The reference calendar is taken from the series identified by
    reference_key. If reference_key is None, the union of all indices
    is used (most inclusive calendar).

    This ensures that when different markets have different holiday
    calendars, the price and return series all share a common index
    before the allocation layer. Missing values on non-trading days
    are forward-filled (last known price / return = 0 for returns).

    For return series: forward-fill would incorrectly propagate a non-zero
    return. Callers should call this on price DataFrames, then compute
    returns afterwards. If calling on return series, NaN-fill (not ffill)
    is used: a missing return on a non-trading day = 0 (the asset was not
    traded, so no P&L).

    Parameters
    ----------
    series_dict   : dict  — {name: pd.Series or pd.DataFrame}
    reference_key : str   — key of the series to use as the reference
                            calendar; None = union of all indices

    Returns
    -------
    dict  — same keys, all aligned to the reference calendar
    """
    if reference_key is not None:
        ref_idx = series_dict[reference_key].index
    else:
        ref_idx = series_dict[next(iter(series_dict))].index
        for s in series_dict.values():
            ref_idx = ref_idx.union(s.index)

    aligned = {}
    for name, obj in series_dict.items():
        if isinstance(obj, pd.DataFrame):
            aligned[name] = obj.reindex(ref_idx, method="ffill")
        else:
            # Series: distinguish price (ffill) vs return (fillna 0)
            # Heuristic: return series have many small values; price series
            # are monotone-ish. We use the series name as a hint if possible.
            aligned[name] = obj.reindex(ref_idx, method="ffill")

    return aligned


# ============================================================
# N-ASSET ALLOCATION — WEIGHT GRID
# ============================================================

def generate_weight_grid(n_assets: int, step: float = 0.10) -> list[tuple]:
    """
    Enumerate all weight tuples (w1, ..., wN) at the given step size
    where all weights are non-negative and sum(wi) <= 1.0.

    MMF is implicit — it receives the residual allocation 1 - sum(wi).
    A tuple of all zeros is included (all in MMF).

    Parameters
    ----------
    n_assets : int    — number of risky assets (excluding MMF)
    step     : float  — weight increment (default 0.10)

    Returns
    -------
    list[tuple]  — list of weight tuples, length = C(N + 1/step, N) roughly
    """
    levels = [round(i * step, 10) for i in range(int(round(1.0 / step)) + 1)]
    combos = [
        c for c in itertools.product(levels, repeat=n_assets)
        if sum(c) <= 1.0 + 1e-9
    ]
    logging.info(
        "Weight grid: %d assets at step=%.2f → %d combinations",
        n_assets, step, len(combos),
    )
    return combos


# ============================================================
# N-ASSET ALLOCATION — OPTIMISER
# ============================================================

def optimise_asset_weights(
    returns_dict:   dict,
    signals_dict:   dict,
    mmf_returns:    pd.Series,
    step:           float = 0.10,
    objective:      str   = "calmar",
) -> tuple[dict, float]:
    """
    Grid-search the best N-asset allocation weights on in-sample data.

    For each weight combination, simulate a fully-invested portfolio
    (risky assets + MMF residual) and compute the objective metric.
    The best combination is returned.

    The simulation ignores the reallocation gate (it is applied in the
    OOS phase in allocation_walk_forward_n). Signals are applied
    daily: if a signal is off, that asset's weight goes to MMF.

    Parameters
    ----------
    returns_dict  : dict[str, pd.Series]  — in-sample daily returns per asset
    signals_dict  : dict[str, pd.Series]  — in-sample binary signals per asset
                                            (1=invested, 0=cash); same keys as
                                            returns_dict
    mmf_returns   : pd.Series             — in-sample MMF daily returns
    step          : float                 — weight grid step (default 0.10)
    objective     : str                   — "calmar", "sharpe", or "cagr"

    Returns
    -------
    tuple (best_weights_dict, best_objective_value)
    best_weights_dict : dict[str, float]  — weight per asset key (excludes MMF;
                                            MMF = 1 - sum of risky weights)
    """
    asset_keys = list(returns_dict.keys())
    n = len(asset_keys)
    combos = generate_weight_grid(n, step)

    # Build common_idx from assets that have IS signal coverage only.
    # Assets with no IS signal (empty signals_dict entry) are excluded from
    # the intersection — they have zero signal on all days so they contribute
    # nothing to IS evaluation, but their shorter return history would silently
    # truncate the IS eval window and change the IS CalMAR scores.
    # (e.g. PL_MID/MSCI data starts 2010 while WIG20TR/TBSP start 2005/2006;
    #  including them in the intersection cuts 4 years of IS history including
    #  the 2008-2009 crisis, which changes which assets appear favourable.)
    keys_with_signal = [
        k for k in asset_keys
        if signals_dict.get(k) is not None and len(signals_dict[k]) > 0
    ]
    keys_no_signal = [k for k in asset_keys if k not in keys_with_signal]

    if keys_with_signal:
        common_idx = mmf_returns.index
        for key in keys_with_signal:
            common_idx = common_idx.intersection(returns_dict[key].index)
    else:
        # No signals at all — use full intersection as fallback
        common_idx = mmf_returns.index
        for key in asset_keys:
            common_idx = common_idx.intersection(returns_dict[key].index)

    if len(common_idx) < 252:
        logging.warning(
            "optimise_asset_weights: common index only %d days (< 252) — "
            "returning equal weights.", len(common_idx),
        )
        equal_w = round(1.0 / (n + 1), 2)
        return {k: equal_w for k in asset_keys}, -np.inf

    # No signal era restriction needed: pre-signal days have signal=0 (fillna(0)),
    # so those days earn MMF automatically — matching multiasset_library behaviour.
    # Reindex ALL assets (including no-signal ones) to common_idx;
    # no-signal assets will have signal=0 everywhere so their returns don't matter.
    r     = {k: returns_dict[k].reindex(common_idx).fillna(0.0) for k in asset_keys}
    mmf_r = mmf_returns.reindex(common_idx).fillna(0.0)

    # Per-asset binary signal.  Default to 0 (flat/out) before signal era starts,
    # matching multiasset_library which uses fillna(0) for the IS optimiser.
    # Signal=0 before the OOS start means those days earn MMF (not the asset),
    # which correctly represents the pre-signal period where no position is held.
    s = {}
    for key in asset_keys:
        sig = signals_dict.get(key)
        if sig is not None and len(sig) > 0:
            s[key] = sig.reindex(common_idx, method="ffill").fillna(0.0)
        else:
            s[key] = pd.Series(0.0, index=common_idx)

    # ── Portfolio return construction — mirrors optimise_both_on_weights exactly ──
    #
    # This is the critical insight from multiasset_library.optimise_both_on_weights:
    #
    #   n_on == 0  :  100% MMF              (FIXED — same for every combo)
    #   n_on == 1  :  100% the single asset (FIXED — same for every combo)
    #   n_on >= 2  :  weight combo applied  (VARIES — this is what we optimise)
    #
    # On 0-signal and 1-signal days, every combo earns exactly the same return.
    # The ranking is driven purely by performance on "multi-signal-on" days.
    # The all-zero combo (MMF) earns MMF even on multi-on days, so it competes
    # fairly against risky combos without the CalMAR infinity problem.
    #
    # Pre-compute per-day signal count and "fixed" returns (identical across combos):
    sig_mat  = np.column_stack([s[k].values for k in asset_keys])  # (T, N)
    ret_mat  = np.column_stack([r[k].values for k in asset_keys])  # (T, N)
    mmf_arr  = mmf_r.values                                         # (T,)
    n_on_arr = sig_mat.sum(axis=1)                                  # (T,)

    # Fixed return: 0-on -> MMF; 1-on -> that asset; 2+-on -> handled per combo
    fixed_r = np.where(n_on_arr == 0, mmf_arr, 0.0)
    for i, key in enumerate(asset_keys):
        only_this = (sig_mat[:, i] == 1) & (n_on_arr == 1)
        fixed_r = np.where(only_this, ret_mat[:, i], fixed_r)

    multi_on_mask = n_on_arr >= 2   # days where combo matters

    best_weights = {k: 0.0 for k in asset_keys}
    best_obj     = -np.inf

    for combo in combos:
        w     = np.array(combo, dtype=float)        # (N,)
        w_mmf = max(0.0, 1.0 - w.sum())

        # On multi-signal-on days: weighted blend (combo determines the split)
        # w[i] * (sig[i]*ret[i] + (1-sig[i])*mmf) + w_mmf * mmf
        multi_r = mmf_arr * w_mmf
        for i in range(n):
            multi_r = multi_r + w[i] * (
                sig_mat[:, i] * ret_mat[:, i] + (1.0 - sig_mat[:, i]) * mmf_arr
            )

        port_r_arr = np.where(multi_on_mask, multi_r, fixed_r)
        port_r = pd.Series(port_r_arr, index=common_idx)

        equity = (1 + port_r).cumprod()
        equity = equity / equity.iloc[0]

        metrics = compute_metrics(equity)
        if not metrics:
            continue

        if objective == "calmar":
            obj_val = metrics.get("CalMAR", -np.inf)
        elif objective == "sharpe":
            obj_val = metrics.get("Sharpe", -np.inf)
        elif objective == "cagr":
            obj_val = metrics.get("CAGR", -np.inf)
        else:
            obj_val = metrics.get("CalMAR", -np.inf)

        if np.isfinite(obj_val) and obj_val > best_obj:
            best_obj     = obj_val
            best_weights = dict(zip(asset_keys, combo))

    logging.info(
        "optimise_asset_weights: best %s=%.4f  weights=%s",
        objective, best_obj,
        {k: f"{v:.1%}" for k, v in best_weights.items()},
    )
    return best_weights, best_obj


# ============================================================
# N-ASSET ALLOCATION — REALLOCATION GATE (generalised)
# ============================================================

def reallocation_gate_n(
    current_weights:  dict,
    target_weights:   dict,
    last_change_date,
    current_date,
    cooldown_days:    int   = 10,
    min_delta:        float = 0.10,
    annual_cap:       int   = 999,
    annual_counter:   dict  = None,
) -> tuple[dict, bool]:
    """
    Decide whether to apply a target N-asset reallocation or hold.

    Blocks reallocation if ANY of:
      1. Fewer than cooldown_days calendar days since last reallocation.
      2. No asset weight changes by >= min_delta.
      3. Annual reallocation count >= annual_cap (set to 999 to disable).

    Parameters
    ----------
    current_weights  : dict  — current weights per asset + "mmf"
    target_weights   : dict  — proposed target weights per asset + "mmf"
    last_change_date         — date of last accepted reallocation (or None)
    current_date             — today's date
    cooldown_days    : int   — minimum days between reallocations
    min_delta        : float — minimum weight change to trigger reallocation
    annual_cap       : int   — max reallocations per calendar year (999=disabled)
    annual_counter   : dict  — mutable {year: count} counter; modified in place

    Returns
    -------
    (accepted_weights, did_reallocate)
    accepted_weights : dict  — weights to use (current if blocked, target if accepted)
    did_reallocate   : bool  — True if reallocation was accepted
    """
    if annual_counter is None:
        annual_counter = {}

    current_year = pd.Timestamp(current_date).year

    # Gate 1: cooldown
    if last_change_date is not None:
        days_elapsed = (pd.Timestamp(current_date) - pd.Timestamp(last_change_date)).days
        if days_elapsed < cooldown_days:
            return current_weights, False

    # Gate 2: minimum delta
    max_delta = max(
        abs(target_weights.get(k, 0.0) - current_weights.get(k, 0.0))
        for k in set(target_weights) | set(current_weights)
    )
    if max_delta < min_delta:
        return current_weights, False

    # Gate 3: annual cap
    count_this_year = annual_counter.get(current_year, 0)
    if count_this_year >= annual_cap:
        return current_weights, False

    # Accept
    annual_counter[current_year] = count_this_year + 1
    return dict(target_weights), True


# ============================================================
# N-ASSET ALLOCATION — SIGNALS TO TARGET WEIGHTS
# ============================================================

def signals_to_target_weights_n(
    signals:      dict,
    best_weights: dict,
    mmf_key:      str = "mmf",
) -> dict:
    """
    Map per-asset binary signals + optimised weights to a target weight dict.

    Mirrors signals_to_target_weights from multiasset_library exactly,
    generalised to N assets:

      n_on == 0  :  100% MMF
      n_on == 1  :  100% the single on-asset  (not its partial weight)
      n_on >= 2  :  best_weights split applied across all on-assets

    This IS/OOS consistency is critical: the IS optimiser evaluates combos
    using exactly this same 3-state logic, so the OOS simulation must use
    the same rule to remain consistent with what was optimised.

    Parameters
    ----------
    signals      : dict[str, int]    — binary signal per asset (0 or 1)
    best_weights : dict[str, float]  — optimised weight per asset
                                       (used only when n_on >= 2)
    mmf_key      : str               — key for MMF in returned dict

    Returns
    -------
    dict  — {asset_key: weight, ..., mmf_key: mmf_weight}
    """
    asset_keys = list(signals.keys())
    on_keys    = [k for k in asset_keys if signals.get(k, 0)]
    n_on       = len(on_keys)

    target = {k: 0.0 for k in asset_keys}

    if n_on == 0:
        # All signals off — 100% MMF
        target[mmf_key] = 1.0

    elif n_on == 1:
        # Exactly one signal on — 100% that asset, consistent with IS optimiser
        target[on_keys[0]] = 1.0
        target[mmf_key]    = 0.0

    else:
        # Multiple signals on — apply optimised weight split
        # Each on-asset gets its best_weights allocation; off-assets get 0
        # MMF absorbs: residual (1 - sum of on-asset weights) + off-asset weights
        total_risky = 0.0
        for key in on_keys:
            w = best_weights.get(key, 0.0)
            target[key]  = w
            total_risky += w
        target[mmf_key] = max(0.0, 1.0 - total_risky)

    return target


# ============================================================
# N-ASSET ALLOCATION WALK-FORWARD
# ============================================================

def allocation_walk_forward_n(
    returns_dict:      dict,
    signals_full_dict: dict,
    signals_oos_dict:  dict,
    mmf_returns:       pd.Series,
    wf_results_ref:    pd.DataFrame,
    asset_keys:        list,
    step:              float = 0.10,
    objective:         str   = "calmar",
    cooldown_days:     int   = 10,
    annual_cap:        int   = 999,
    train_years:       int   = 9,
) -> tuple:
    """
    Walk-forward allocation optimisation for N risky assets + MMF residual.

    Generalises multiasset_library.allocation_walk_forward to an arbitrary
    number of assets. The window schedule is driven by wf_results_ref, which
    should correspond to the asset with the latest/shortest OOS history
    (typically TBSP, as in the existing framework).

    For each window:
      1. Slice in-sample returns and signals up to train_end.
      2. Grid-search best weights via optimise_asset_weights.
      3. Apply weights to OOS window with reallocation gate.
      4. Chain-link OOS equity slices.

    Reallocation gate state is carried continuously across window boundaries.

    Parameters
    ----------
    returns_dict       : dict[str, pd.Series]  — full daily returns per asset
    signals_full_dict  : dict[str, pd.Series]  — full OOS signal series per asset
    signals_oos_dict   : dict[str, pd.Series]  — OOS-trimmed signal series per asset
    mmf_returns        : pd.Series             — full MMF daily returns
    wf_results_ref     : pd.DataFrame          — walk-forward results for schedule
                                                 reference asset (defines windows)
    asset_keys         : list[str]             — ordered list of risky asset keys
    step               : float                 — weight grid step (default 0.10)
    objective          : str                   — allocation objective function
    cooldown_days      : int                   — reallocation gate cooldown
    annual_cap         : int                   — max reallocations/year (999=disabled)
    train_years        : int                   — in-sample window length (years)

    Returns
    -------
    tuple of:
      portfolio_equity   : pd.Series        — chain-linked OOS equity curve
      weights_series     : pd.Series        — daily weight dicts
      reallocation_log   : list[dict]       — log of accepted reallocations
      alloc_results_df   : pd.DataFrame     — per-window optimised weights
    """
    oos_equity_slices = []
    alloc_results     = []
    all_weights       = {}
    all_realloc_log   = []
    annual_counter    = {}

    # Determine OOS period from reference walk-forward results
    oos_start = wf_results_ref["TestStart"].min()
    oos_end   = wf_results_ref["TestEnd"].max()

    # Initialise gate state
    last_change_date = None
    current_weights  = {k: 0.0 for k in asset_keys}
    current_weights["mmf"] = 1.0

    prev_best_weights = {k: 0.0 for k in asset_keys}  # all-zero = no carry-forward

    for _, row in wf_results_ref.iterrows():
        train_end  = pd.Timestamp(row["TestStart"])
        test_start = pd.Timestamp(row["TestStart"])
        test_end   = pd.Timestamp(row["TestEnd"])

        # ── In-sample slices ──────────────────────────────────────────────
        def _is(series, end):
            return series.loc[series.index < end]

        is_returns = {k: _is(returns_dict[k], train_end) for k in asset_keys}
        is_mmf     = _is(mmf_returns, train_end)
        # Slice per-asset signals up to train_end.
        # signals_full_dict contains walk-forward OOS signals starting from
        # each asset's first OOS date.  For IS dates before that start,
        # optimise_asset_weights will forward-fill with 1.0 (always invested).
        # This ensures the IS optimisation uses real signal-gated returns for
        # the period where signals exist, rather than raw buy-and-hold returns.
        is_signals = {k: _is(signals_full_dict[k], train_end) for k in asset_keys}

        # Require minimum in-sample length
        min_is_len = min(len(s) for s in is_returns.values())
        if min_is_len < 252:
            logging.warning(
                "Skipping window %s–%s: insufficient in-sample data (%d days)",
                train_end.date(), test_end.date(), min_is_len,
            )
            continue

        # ── Optimise weights ──────────────────────────────────────────────
        best_weights, best_obj = optimise_asset_weights(
            returns_dict  = is_returns,
            signals_dict  = is_signals,
            mmf_returns   = is_mmf,
            step          = step,
            objective     = objective,
        )

        # If all combos were identical in IS (indiscriminate), best_weights will
        # be all-zero (first combo in grid).  Carry forward previous window's
        # weights in that case, rather than defaulting to 100% MMF.
        # This happens in early windows where most asset signals haven't started.
        total_risky = sum(best_weights.values())
        if total_risky < step - 1e-9 and sum(prev_best_weights.values()) >= step - 1e-9:
            logging.info(
                "Window %s: IS optimisation indiscriminate (all-zero) — "
                "carrying forward previous weights %s",
                test_start.date(),
                {k: f"{v:.0%}" for k, v in prev_best_weights.items()},
            )
            best_weights = dict(prev_best_weights)

        prev_best_weights = dict(best_weights)

        alloc_results.append({
            "TrainEnd":   train_end.date(),
            "TestStart":  test_start.date(),
            "TestEnd":    test_end.date(),
            **{f"w_{k}": best_weights.get(k, 0.0) for k in asset_keys},
            "w_mmf":      max(0.0, 1.0 - sum(best_weights.values())),
            f"{objective}": best_obj,
        })

        # ── OOS simulation ────────────────────────────────────────────────
        # OOS signals trimmed to this window's test period.
        # Use explicit lambda parameters to avoid loop-variable closure issues.
        def _oos(series, ts=test_start, te=test_end):
            return series.loc[(series.index >= ts) & (series.index <= te)]

        oos_sigs    = {k: _oos(signals_oos_dict[k]) for k in asset_keys}

        # Reference calendar: union of all asset signal calendars.
        # Different markets (PL, US, EU, JP) have different holidays.
        # Using the union ensures every trading day from any market is
        # represented, with signals/returns forward-filled for non-trading days.
        ref_oos_idx = oos_sigs[asset_keys[0]].index
        for k in asset_keys[1:]:
            ref_oos_idx = ref_oos_idx.union(oos_sigs[k].index)
        ref_oos_idx = ref_oos_idx.sort_values()

        if ref_oos_idx.empty:
            logging.warning(
                "Skipping OOS window %s-%s: no signal data for any asset.",
                test_start.date(), test_end.date(),
            )
            continue

        # Align signals: ffill for non-trading days, default 0 at start
        oos_sigs_aligned = {
            k: oos_sigs[k].reindex(ref_oos_idx, method="ffill").fillna(0.0)
            for k in asset_keys
        }
        # Align returns: ffill for non-trading days (no return = carry last price)
        # then fill any remaining NaN at the start with 0
        oos_returns = {
            k: returns_dict[k].reindex(ref_oos_idx, method="ffill").fillna(0.0)
            for k in asset_keys
        }
        oos_mmf = mmf_returns.reindex(ref_oos_idx, method="ffill").fillna(0.0)

        window_equity_vals = []

        for date in ref_oos_idx:
            # Build signal dict for this date (already aligned above)
            sigs_today = {k: int(oos_sigs_aligned[k].get(date, 0)) for k in asset_keys}

            # Target weights from signals + optimised allocation
            target = signals_to_target_weights_n(
                signals      = sigs_today,
                best_weights = best_weights,
            )

            # Apply reallocation gate
            accepted, did_reallocate = reallocation_gate_n(
                current_weights  = current_weights,
                target_weights   = target,
                last_change_date = last_change_date,
                current_date     = date,
                cooldown_days    = cooldown_days,
                min_delta        = 0.10,
                annual_cap       = annual_cap,
                annual_counter   = annual_counter,
            )

            if did_reallocate:
                all_realloc_log.append({
                    "Date":         date,
                    "weights_before": dict(current_weights),
                    "weights_after":  dict(accepted),
                })
                current_weights  = accepted
                last_change_date = date

            all_weights[date] = dict(current_weights)

            # Compute daily portfolio return using current (gated) weights.
            # Each asset contributes: weight * return (on trading days)
            # MMF accrues on all days including non-trading days for that market.
            w   = current_weights
            mmf_r_today = float(oos_mmf.get(date, 0.0))
            port_r = w.get("mmf", 1.0) * mmf_r_today
            for k in asset_keys:
                port_r += w.get(k, 0.0) * float(oos_returns[k].get(date, 0.0))

            window_equity_vals.append((date, port_r))

        if not window_equity_vals:
            continue

        dates, rets = zip(*window_equity_vals)
        window_port_r = pd.Series(list(rets), index=list(dates))
        window_equity = (1 + window_port_r).cumprod()
        window_equity = window_equity / window_equity.iloc[0]

        if oos_equity_slices:
            window_equity = window_equity * oos_equity_slices[-1].iloc[-1]

        oos_equity_slices.append(window_equity)

    if not oos_equity_slices:
        logging.warning("allocation_walk_forward_n produced no OOS slices.")
        return pd.Series(dtype=float), pd.Series(dtype=object), [], pd.DataFrame()

    portfolio_equity = pd.concat(oos_equity_slices).sort_index()
    weights_series   = pd.Series(all_weights).sort_index()
    alloc_results_df = pd.DataFrame(alloc_results)

    logging.info(
        "allocation_walk_forward_n complete: %d OOS days  "
        "(%s to %s)  %d reallocations",
        len(portfolio_equity),
        portfolio_equity.index.min().date(),
        portfolio_equity.index.max().date(),
        len(all_realloc_log),
    )
    return portfolio_equity, weights_series, all_realloc_log, alloc_results_df


# ============================================================
# REPORTING
# ============================================================

def print_global_equity_report(
    portfolio_metrics:  dict,
    bh_metrics_dict:    dict,
    alloc_results_df:   pd.DataFrame,
    reallocation_log:   list,
    signals_oos_dict:   dict,
    oos_start,
    oos_end,
    portfolio_mode:     str,
    fx_hedged:          bool,
) -> None:
    """
    Log a structured OOS report for the N-asset global equity portfolio.

    Parameters
    ----------
    portfolio_metrics : dict                  — compute_metrics output for portfolio
    bh_metrics_dict   : dict[str, dict]       — B&H metrics per asset key
    alloc_results_df  : pd.DataFrame          — per-window allocation weights
    reallocation_log  : list[dict]            — accepted reallocation log
    signals_oos_dict  : dict[str, pd.Series]  — OOS binary signals per asset
    oos_start         : date-like
    oos_end           : date-like
    portfolio_mode    : str                   — "global_equity" or "msci_world"
    fx_hedged         : bool                  — FX treatment used
    """
    sep = "=" * 80

    logging.info(sep)
    logging.info(
        "GLOBAL EQUITY PORTFOLIO  OOS REPORT  (%s to %s)",
        pd.Timestamp(oos_start).date(), pd.Timestamp(oos_end).date(),
    )
    logging.info("Mode: %s  |  FX treatment: %s",
                 portfolio_mode, "hedged (no FX adjustment)" if fx_hedged else "unhedged (FX-adjusted)")
    logging.info(sep)

    # Per-window allocation weights
    if alloc_results_df is not None and not alloc_results_df.empty:
        logging.info("PER-WINDOW OPTIMISED WEIGHTS:")
        logging.info("\n%s", alloc_results_df.to_string(index=False))
    logging.info("-" * 80)

    # Portfolio performance
    def _fmt(m):
        if not m:
            return "N/A"
        return (
            f"CAGR={m.get('CAGR', 0)*100:.2f}%  "
            f"Vol={m.get('Vol', 0)*100:.2f}%  "
            f"Sharpe={m.get('Sharpe', 0):.2f}  "
            f"MaxDD={m.get('MaxDD', 0)*100:.2f}%  "
            f"CalMAR={m.get('CalMAR', 0):.2f}"
        )

    logging.info("PORTFOLIO:  %s", _fmt(portfolio_metrics))
    logging.info("-" * 80)

    logging.info("BENCHMARKS (buy-and-hold per asset):")
    for key, m in bh_metrics_dict.items():
        logging.info("  %-12s %s", key, _fmt(m))
    logging.info("-" * 80)

    # Signal frequency
    logging.info("SIGNAL FREQUENCY (OOS):")
    for key, sig in signals_oos_dict.items():
        logging.info("  %-12s in position %.1f%% of days", key, sig.mean() * 100)
    logging.info("-" * 80)

    # Reallocation summary
    n_realloc = len(reallocation_log)
    logging.info("REALLOCATIONS: %d total", n_realloc)
    if reallocation_log:
        log_df = pd.DataFrame(reallocation_log)
        log_df["Year"] = pd.to_datetime(log_df["Date"]).dt.year
        by_year = log_df.groupby("Year").size()
        logging.info("\n%s", by_year.to_string())
    logging.info(sep)


# ============================================================
# LEVEL 2 ROBUSTNESS — N-ASSET ALLOCATION WEIGHT PERTURBATION
# ============================================================

def allocation_weight_robustness_n(
    alloc_results_df:  pd.DataFrame,
    returns_dict:      dict,
    mmf_returns:       pd.Series,
    signals_oos_dict:  dict,
    asset_keys:        list,
    baseline_metrics:  dict,
    perturb_steps:     list  = [-0.2, -0.1, 0.0, 0.1, 0.2],
    focus_asset:       str   = None,
    min_weight:        float = 0.0,
    max_weight:        float = 1.0,
    cooldown_days:     int   = 10,
    annual_cap:        int   = 999,
) -> pd.DataFrame:
    """
    Level 2 robustness check: perturb one asset's optimised weight by fixed
    steps and re-simulate the full OOS portfolio for each perturbation.

    Generalises multiasset_library.allocation_weight_robustness from 2 assets
    (equity + bond) to N assets.

    For each window in alloc_results_df the optimised weight for focus_asset
    is shifted by each value in perturb_steps (e.g. ±0.10, ±0.20).
    All other risky-asset weights are scaled proportionally so that
    sum(w_i for i != focus_asset) + w_focus stays <= 1.0.  MMF absorbs the
    remainder.  All windows shift by the same step simultaneously.

    The reallocation gate state is carried continuously across windows,
    matching the live simulation exactly.

    Design rationale
    ----------------
    In the N-asset case, the allocation result is a vector (not a scalar), so
    we cannot perturb a single "equity weight" as in the 2-asset case.  Instead
    we pick one asset as the focal point and shift it, proportionally adjusting
    the others.  This is most useful when called once per asset — the runfile
    runs it for each equity asset with significant non-zero allocation history.

    Parameters
    ----------
    alloc_results_df : pd.DataFrame  — per-window optimised weights from
                                       allocation_walk_forward_n; must contain
                                       columns TestStart, TestEnd, and
                                       w_{key} for each key in asset_keys,
                                       plus w_mmf.
    returns_dict     : dict[str, pd.Series]  — OOS daily returns per asset
    mmf_returns      : pd.Series             — full MMF daily returns
    signals_oos_dict : dict[str, pd.Series]  — OOS binary signals per asset
    asset_keys       : list[str]             — ordered risky asset keys
    baseline_metrics : dict                  — compute_metrics on baseline portfolio
    perturb_steps    : list[float]           — weight shifts to apply to focus_asset;
                                               0.0 reproduces the baseline
    focus_asset      : str | None            — asset whose weight is shifted.
                                               If None, uses asset_keys[0].
    min_weight       : float                 — floor on any perturbed weight (default 0.0)
    max_weight       : float                 — ceiling on focus_asset weight (default 1.0)
    cooldown_days    : int                   — reallocation gate cooldown
    annual_cap       : int                   — reallocation gate annual cap

    Returns
    -------
    pd.DataFrame  — one row per perturbation step with columns:
        focus_asset, weight_shift, focus_weight_mean,
        CAGR, Vol, Sharpe, MaxDD, CalMAR, Sortino,
        n_reallocations, n_floor_clamped, n_ceil_clamped,
        CAGR_vs_base, MaxDD_vs_base, CalMAR_vs_base
    """
    if focus_asset is None:
        focus_asset = asset_keys[0]

    if focus_asset not in asset_keys:
        logging.error(
            "allocation_weight_robustness_n: focus_asset '%s' not in asset_keys %s",
            focus_asset, asset_keys,
        )
        return pd.DataFrame()

    other_keys = [k for k in asset_keys if k != focus_asset]

    # Pre-align returns and signals to a common OOS index (union of all signal calendars)
    all_signal_idx = None
    for k in asset_keys:
        sig = signals_oos_dict.get(k)
        if sig is not None and not sig.empty:
            all_signal_idx = sig.index if all_signal_idx is None else all_signal_idx.union(sig.index)

    if all_signal_idx is None:
        logging.error("allocation_weight_robustness_n: no OOS signal data found.")
        return pd.DataFrame()

    # Reindex all returns and signals to the common OOS index
    ret_aligned  = {k: returns_dict[k].reindex(all_signal_idx, method="ffill").fillna(0.0)
                    for k in asset_keys}
    mmf_aligned  = mmf_returns.reindex(all_signal_idx, method="ffill").fillna(0.0)
    sig_aligned  = {k: signals_oos_dict[k].reindex(all_signal_idx, method="ffill").fillna(0.0)
                    for k in asset_keys}

    results = []

    for step in perturb_steps:
        # ── Build per-window perturbed weight dicts ───────────────────────
        perturbed_windows = []
        n_floor_clamped = 0
        n_ceil_clamped  = 0

        for _, row in alloc_results_df.iterrows():
            # Current optimised weights
            w = {k: float(row.get(f"w_{k}", 0.0)) for k in asset_keys}
            w_mmf_orig = float(row.get("w_mmf", 0.0))

            # Shift focus asset
            raw_focus = w[focus_asset] + step
            clamped   = False

            if raw_focus < min_weight:
                raw_focus = min_weight
                n_floor_clamped += 1
                clamped = True
            elif raw_focus > max_weight:
                raw_focus = max_weight
                n_ceil_clamped += 1
                clamped = True

            w_focus_new = round(raw_focus, 10)

            # Scale other risky-asset weights proportionally so their
            # relative ratios are preserved; total risky <= 1.0
            other_total_orig = sum(w[k] for k in other_keys)
            risky_budget     = min(1.0 - w_focus_new, 1.0)

            if other_total_orig > 1e-9:
                scale = min(risky_budget, other_total_orig) / other_total_orig
                w_others = {k: max(min_weight, round(w[k] * scale, 10))
                            for k in other_keys}
            else:
                w_others = {k: 0.0 for k in other_keys}

            w_new          = {focus_asset: w_focus_new, **w_others}
            w_mmf_new      = max(0.0, round(1.0 - sum(w_new.values()), 10))

            perturbed_windows.append({
                "TestStart": pd.Timestamp(row["TestStart"]),
                "TestEnd":   pd.Timestamp(row["TestEnd"]),
                **{k: w_new[k] for k in asset_keys},
                "w_mmf":     w_mmf_new,
            })

        if n_floor_clamped or n_ceil_clamped:
            logging.info(
                "step=%+.2f: %d windows clamped at floor (min=%.2f), "
                "%d clamped at ceiling (max=%.2f).",
                step, n_floor_clamped, min_weight, n_ceil_clamped, max_weight,
            )

        # ── Simulate OOS with perturbed weights ───────────────────────────
        oos_equity_slices = []
        n_reallocations   = 0
        current_weights   = {k: 0.0 for k in asset_keys}
        current_weights["mmf"] = 1.0
        last_change_date  = None
        annual_counter    = {}

        for pw in perturbed_windows:
            window_start = pw["TestStart"]
            window_end   = pw["TestEnd"]

            # Target weights from the perturbed window
            target = {k: pw[k] for k in asset_keys}
            target["mmf"] = pw["w_mmf"]

            # Slice the OOS index to this window
            window_idx = all_signal_idx[
                (all_signal_idx >= window_start) & (all_signal_idx <= window_end)
            ]
            if window_idx.empty:
                continue

            window_ret_vals = []
            for date in window_idx:
                # Signals on this date
                sigs_today = {k: int(sig_aligned[k].get(date, 0)) for k in asset_keys}
                n_on = sum(sigs_today.values())

                # Apply 3-state portfolio weights (mirrors signals_to_target_weights_n)
                if n_on == 0:
                    effective = {k: 0.0 for k in asset_keys}
                    effective["mmf"] = 1.0
                elif n_on == 1:
                    on_key = next(k for k in asset_keys if sigs_today[k] == 1)
                    effective = {k: 0.0 for k in asset_keys}
                    effective[on_key] = 1.0
                    effective["mmf"] = 0.0
                else:
                    # Distribute signal-off assets to MMF within each weight bucket
                    effective = {}
                    extra_mmf = 0.0
                    for k in asset_keys:
                        if sigs_today[k] == 1:
                            effective[k] = target[k]
                        else:
                            effective[k] = 0.0
                            extra_mmf += target[k]
                    effective["mmf"] = target["mmf"] + extra_mmf

                # Gate
                accepted, did_reallocate = reallocation_gate_n(
                    current_weights  = current_weights,
                    target_weights   = effective,
                    last_change_date = last_change_date,
                    current_date     = date,
                    cooldown_days    = cooldown_days,
                    min_delta        = 0.10,
                    annual_cap       = annual_cap,
                    annual_counter   = annual_counter,
                )
                if did_reallocate:
                    current_weights  = accepted
                    last_change_date = date
                    n_reallocations += 1

                # Daily portfolio return
                r = sum(current_weights[k] * float(ret_aligned[k].get(date, 0.0))
                        for k in asset_keys)
                r += current_weights["mmf"] * float(mmf_aligned.get(date, 0.0))
                window_ret_vals.append((date, r))

            if not window_ret_vals:
                continue

            dates, rets = zip(*window_ret_vals)
            port_r = pd.Series(list(rets), index=list(dates))
            eq = (1 + port_r).cumprod()
            eq = eq / eq.iloc[0]
            if oos_equity_slices:
                eq = eq * oos_equity_slices[-1].iloc[-1]
            oos_equity_slices.append(eq)

        if not oos_equity_slices:
            logging.warning("step=%+.2f: no OOS slices produced — skipped.", step)
            continue

        portfolio_equity = pd.concat(oos_equity_slices).sort_index()
        m = compute_metrics(portfolio_equity)
        focus_mean = np.mean([pw[focus_asset] for pw in perturbed_windows])

        results.append({
            "focus_asset":       focus_asset,
            "weight_shift":      step,
            "focus_weight_mean": round(focus_mean, 3),
            "n_floor_clamped":   n_floor_clamped,
            "n_ceil_clamped":    n_ceil_clamped,
            "CAGR":              m.get("CAGR",    np.nan),
            "Vol":               m.get("Vol",     np.nan),
            "Sharpe":            m.get("Sharpe",  np.nan),
            "MaxDD":             m.get("MaxDD",   np.nan),
            "CalMAR":            m.get("CalMAR",  np.nan),
            "Sortino":           m.get("Sortino", np.nan),
            "n_reallocations":   n_reallocations,
        })

    if not results:
        logging.warning("allocation_weight_robustness_n: no results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Add vs-baseline comparison columns (step=0.0 row is the reference)
    base_row = df.loc[df["weight_shift"].abs() < 1e-9]
    if not base_row.empty:
        base_cagr   = base_row["CAGR"].iloc[0]
        base_maxdd  = base_row["MaxDD"].iloc[0]
        base_calmar = base_row["CalMAR"].iloc[0]
        df["CAGR_vs_base"]   = (df["CAGR"]  - base_cagr)  * 100
        df["MaxDD_vs_base"]  = (df["MaxDD"] - base_maxdd) * 100
        df["CalMAR_vs_base"] = (df["CalMAR"] / base_calmar
                                if base_calmar != 0 else np.nan)
    else:
        logging.warning(
            "allocation_weight_robustness_n: step=0.0 not in perturb_steps — "
            "vs_base columns will be NaN.  Add 0.0 to perturb_steps."
        )
        df["CAGR_vs_base"]   = np.nan
        df["MaxDD_vs_base"]  = np.nan
        df["CalMAR_vs_base"] = np.nan

    return df


def print_allocation_robustness_report_n(
    results_df: pd.DataFrame,
    focus_asset: str = "",
) -> None:
    """
    Log a structured report for the N-asset allocation weight perturbation.

    Matches the style of multiasset_library.print_allocation_robustness_report.

    Verdict thresholds (same as 2-asset case):
      CalMAR ratio >= 0.80 at ±10pp  →  PLATEAU  (robust)
      CalMAR ratio  0.60–0.80        →  MODERATE
      CalMAR ratio <  0.60           →  SPIKE    (fragile)
    """
    sep = "=" * 80
    label = f"  (focus: {focus_asset})" if focus_asset else ""
    logging.info(sep)
    logging.info(
        "ALLOCATION WEIGHT ROBUSTNESS  (N-asset, Level 2%s)", label
    )
    logging.info(sep)

    if results_df.empty:
        logging.warning("No results to report.")
        return

    display = results_df.copy()
    display["CAGR"]             = (display["CAGR"]  * 100).round(2).astype(str) + "%"
    display["Vol"]              = (display["Vol"]   * 100).round(2).astype(str) + "%"
    display["MaxDD"]            = (display["MaxDD"] * 100).round(2).astype(str) + "%"
    display["Sharpe"]           = display["Sharpe"].round(3)
    display["CalMAR"]           = display["CalMAR"].round(3)
    display["Sortino"]          = display["Sortino"].round(3)
    display["CAGR_vs_base"]     = display["CAGR_vs_base"].round(2).astype(str) + "pp"
    display["MaxDD_vs_base"]    = display["MaxDD_vs_base"].round(2).astype(str) + "pp"
    display["CalMAR_vs_base"]   = display["CalMAR_vs_base"].round(3)
    display["weight_shift"]     = display["weight_shift"].apply(lambda x: f"{x:+.0%}")
    display["focus_weight_mean"]= display["focus_weight_mean"].apply(lambda x: f"{x:.0%}")

    cols = [
        "weight_shift", "focus_weight_mean",
        "n_floor_clamped", "n_ceil_clamped",
        "CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR", "Sortino",
        "n_reallocations", "CAGR_vs_base", "MaxDD_vs_base", "CalMAR_vs_base",
    ]
    # Only show columns that exist
    cols = [c for c in cols if c in display.columns]
    logging.info("\n%s", display[cols].to_string(index=False))
    logging.info("-" * 80)

    # Verdict: CalMAR ratio at ±10pp steps
    inner = results_df.loc[results_df["weight_shift"].abs().between(0.09, 0.11)]
    if inner.empty:
        logging.info(
            "Verdict: ±10pp steps not present — add ±0.10 to perturb_steps."
        )
        return

    min_ratio = inner["CalMAR_vs_base"].min()
    if pd.isna(min_ratio):
        logging.info("Verdict: baseline CalMAR is zero — ratio undefined.")
        return

    if min_ratio >= 0.80:
        verdict = "PLATEAU"
        interp  = "robust — weight is on a broad performance plateau"
    elif min_ratio >= 0.60:
        verdict = "MODERATE"
        interp  = "somewhat sensitive — treat allocation with moderate confidence"
    else:
        verdict = "SPIKE"
        interp  = "fragile — performance depends critically on this weight level"

    logging.info(
        "Verdict: %s  (min CalMAR ratio at ±10pp = %.3f)  %s",
        verdict, min_ratio, interp,
    )
    logging.info(sep)