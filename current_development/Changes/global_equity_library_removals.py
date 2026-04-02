"""
global_equity_library_removals.py
==================================
Documents the two functions removed from global_equity_library.py.

REMOVED: align_to_reference()
-------------------------------
Defined but never called from any runfile or library function in the
codebase.  The docstring described calendar alignment across multiple
assets, but the actual alignment logic in allocation_walk_forward_n()
uses per-asset reindex/ffill inline.  No callers exist.

REMOVED: load_fx_series()
--------------------------
Defined but never called.  All runfiles extract FX series directly:

    fx_usd = USDPLN[CLOSE_COL] if USDPLN is not None else None
    fx_eur = EURPLN[CLOSE_COL] if EURPLN is not None else None
    fx_jpy = JPYPLN[CLOSE_COL] if JPYPLN is not None else None

The function added a lookup through a `stooq_df_map` dict that was never
constructed in any calling context.

Both functions are preserved here for reference until the next
documentation freeze, then this file is deleted.
"""


# ---------------------------------------------------------------------------
# ARCHIVED: align_to_reference()  (removed from global_equity_library.py)
# ---------------------------------------------------------------------------

def align_to_reference(
    series_dict: dict,
    reference_key=None,
) -> dict:
    """
    ARCHIVED — never called from any runfile or library.

    Align all Series/DataFrames in series_dict to a common business-day
    calendar using forward-fill.

    The reference calendar is taken from the series identified by
    reference_key. If reference_key is None, the union of all indices
    is used (most inclusive calendar).
    """
    import pandas as pd

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
            aligned[name] = obj.reindex(ref_idx, method="ffill")

    return aligned


# ---------------------------------------------------------------------------
# ARCHIVED: load_fx_series()  (removed from global_equity_library.py)
# ---------------------------------------------------------------------------

def load_fx_series(currency: str, stooq_df_map: dict):
    """
    ARCHIVED — never called from any runfile or library.

    Extract a daily FX close series from a pre-loaded stooq DataFrame map.
    The stooq_df_map dict ({ticker: DataFrame}) was never constructed in
    any calling context; all runfiles access FX series directly.
    """
    from global_equity_library import FX_TICKERS, CLOSE_COL
    ticker = FX_TICKERS.get(currency)
    if ticker is None:
        return None
    df = stooq_df_map.get(ticker)
    if df is None:
        return None
    return df[CLOSE_COL].rename(f"FX_{currency}_PLN")
