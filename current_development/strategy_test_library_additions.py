"""
strategy_test_library_additions.py
===================================
New utility functions to be merged into strategy_test_library.py.

Adds:
  load_stooq_local()     — consolidated stooq CSV loader (replaces per-runfile
                           _stooq() definitions; eliminates ~200 lines of
                           duplication across ~13 runfiles)
  get_n_jobs()           — canonical N_JOBS calculation (replaces per-file
                           duplicate of the cpu-count / platform logic)
  annual_cagr_by_year()  — calendar-year CAGR per year (moved from objective_review.py)
  count_year_wins()      — compare annual returns vs incumbent (moved from objective_review.py)

NOTE: This file is an *additions patch* only — it shows the four functions to
insert into strategy_test_library.py.  The full merged file is
strategy_test_library.py in this PR.  The functions below replace all
per-runfile _stooq() helpers and the N_JOBS boilerplate block.
"""

# These imports are already present in strategy_test_library.py — shown here
# for clarity when reviewing the additions in isolation.
import os
import sys
import logging
import pandas as pd
import numpy as np


# ============================================================
# CONSOLIDATED STOOQ LOCAL LOADER
# ============================================================

def load_stooq_local(
    ticker: str,
    label: str,
    data_dir: str,
    data_start: str = "1990-01-01",
    mandatory: bool = True,
) -> pd.DataFrame | None:
    """
    Load a stooq price series from the local /data subfolder.

    The /data directory is populated by stooq_hybrid_updater.run_update()
    at the start of each runfile.  This function replaces the per-runfile
    _stooq() / _stooq_local() helper that was duplicated across ~13 files.

    Parameters
    ----------
    ticker     : str  — file basename without extension
                        (e.g. "wig", "^tbsp", "fund_2720")
    label      : str  — human-readable name for log messages
    data_dir   : str  — path to the local data directory
    data_start : str  — ISO date string; rows before this date are dropped
    mandatory  : bool — if True, calls sys.exit(1) when the file is missing
                        or load_csv returns None; if False, returns None quietly

    Returns
    -------
    pd.DataFrame | None  — stooq-format DataFrame clipped to data_start,
                           or None if unavailable (when mandatory=False)

    Examples
    --------
    >>> from strategy_test_library import load_stooq_local
    >>> DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    >>> WIG = load_stooq_local("wig", "WIG", DATA_DIR)
    >>> MMF = load_stooq_local("fund_2720", "MMF", DATA_DIR)
    >>> WIBOR1M = load_stooq_local("wibor1m", "WIBOR1M", DATA_DIR, mandatory=False)
    """
    path = os.path.join(data_dir, f"{ticker}.csv")
    df = load_csv(path)                          # load_csv already in strategy_test_library

    if df is None:
        if mandatory:
            logging.error(
                "FAIL: load_csv returned None for %s (%s) — exiting.", label, path
            )
            sys.exit(1)
        logging.warning(
            "WARN: %s (%s) — not available, skipping.", label, path
        )
        return None

    df = df.loc[df.index >= pd.Timestamp(data_start)]
    logging.info(
        "OK  : %-14s  %5d rows  %s to %s",
        label, len(df),
        df.index.min().date(), df.index.max().date(),
    )
    return df


# ============================================================
# CANONICAL N_JOBS CALCULATION
# ============================================================

def get_n_jobs() -> int:
    """
    Return the recommended number of parallel jobs for this machine.

    Logic:
      - On Windows with > 3 cores: use (cpu_count - 1) to keep the UI
        responsive (loky/spawn overhead is higher on Windows).
      - All other platforms: use all logical cores.
      - Always returns at least 1.

    Replaces the following block that was duplicated verbatim in ~8 files::

        _cpu_count = os.cpu_count() or 1
        N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    Returns
    -------
    int — recommended n_jobs value for joblib.Parallel / walk_forward

    Example
    -------
    >>> from strategy_test_library import get_n_jobs
    >>> N_JOBS = get_n_jobs()
    """
    cpu_count = os.cpu_count() or 1
    if cpu_count > 3 and sys.platform == "win32":
        return max(1, cpu_count - 1)
    return cpu_count


# ============================================================
# ANNUAL PERFORMANCE UTILITIES  (moved from objective_review.py)
# ============================================================

def annual_cagr_by_year(portfolio_equity: pd.Series) -> dict[int, float]:
    """
    Compute the annualised return for each full calendar year in the equity curve.

    Years with fewer than 50 trading days of data (partial years at the start
    or end of the series) are excluded.

    Parameters
    ----------
    portfolio_equity : pd.Series — portfolio equity curve (DatetimeIndex,
                                   values normalised to an arbitrary base)

    Returns
    -------
    dict[int, float]  — {year: annual_cagr} for all complete years

    Example
    -------
    >>> annual = annual_cagr_by_year(portfolio_equity)
    >>> print(annual)
    {2020: 0.142, 2021: -0.031, 2022: 0.087, 2023: 0.211, 2024: 0.053}
    """
    annual = {}
    df = portfolio_equity.copy()
    df.index = pd.to_datetime(df.index)

    for year in df.index.year.unique():
        yr = df[df.index.year == year]
        if len(yr) < 50:
            continue                           # skip partial years
        start_val = yr.iloc[0]
        end_val   = yr.iloc[-1]
        days      = (yr.index[-1] - yr.index[0]).days
        if days < 1 or start_val <= 0:
            continue
        cagr = (end_val / start_val) ** (365.25 / days) - 1
        annual[year] = cagr

    return annual


def count_year_wins(
    cand_annual: dict[int, float],
    incumb_annual: dict[int, float],
    years: list[int],
) -> int:
    """
    Count how many of the specified calendar years the challenger
    outperforms the incumbent.

    Used in the annual objective function review (objective_review.py) as
    one of the five switching gates.

    Parameters
    ----------
    cand_annual   : dict[int, float] — challenger annual CAGRs by year
    incumb_annual : dict[int, float] — incumbent annual CAGRs by year
    years         : list[int]        — calendar years to compare (e.g. last 5)

    Returns
    -------
    int — number of years in `years` where cand_annual[y] > incumb_annual[y].
          Years missing from either dict are skipped (not counted as a win or
          a loss).

    Example
    -------
    >>> wins = count_year_wins(cand_annual, incumb_annual, [2020, 2021, 2022, 2023, 2024])
    >>> print(f"{wins}/5 year wins")
    """
    wins = 0
    for y in years:
        c = cand_annual.get(y)
        i = incumb_annual.get(y)
        if c is not None and i is not None and c > i:
            wins += 1
    return wins
