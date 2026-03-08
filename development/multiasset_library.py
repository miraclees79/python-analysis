"""
multiasset_library.py
=====================
Multi-asset extension for the WIG/TBSP pension fund strategy.

Imports signal-generation and metrics machinery from strategy_test_library.
Adds three layers that are specific to the multi-asset problem:

  Layer 1 — Per-asset binary signals
    Handled by strategy_test_library.walk_forward, called once per asset
    in the runfile.  No new code needed here — documented below for clarity.

  Layer 2 — Target weight calculation
    signals_to_target_weights()   maps (sig_equity, sig_bond) → weight dict
    optimise_both_on_weights()    grid-searches the equity/bond split when
                                  both signals are simultaneously on, using
                                  combined portfolio CalMAR as objective.

  Layer 3 — Reallocation gate
    reallocation_gate()           enforces cooldown + annual cap + 10% threshold.

  Portfolio simulation
    build_signal_series()         converts per-asset walk-forward equity curves
                                  into a daily binary signal series (1/0).
    portfolio_returns()           stitches daily returns from three assets
                                  according to a time-varying weights series.
    portfolio_equity_from_signals() end-to-end: signals → gated weights →
                                  portfolio equity curve.

  Reporting
    print_multiasset_report()     logs metrics, reallocation count, and
                                  per-asset signal frequency.

DEPENDENCIES
------------
    from strategy_test_library import (
        walk_forward,
        run_strategy_with_trades,
        compute_metrics,
        compute_buy_and_hold,
        download_csv,
        load_csv,
    )
"""

import itertools
import logging
import pandas as pd
import numpy as np

from strategy_test_library import (
    walk_forward,
    run_strategy_with_trades,
    compute_metrics,
    compute_buy_and_hold,
    download_csv,
    load_csv,
)


# ============================================================
# LAYER 3 — REALLOCATION GATE
# ============================================================

def reallocation_gate(
    current_weights: dict,
    target_weights:  dict,
    last_change_date,
    current_date,
    cooldown_days:   int   = 10,
    min_delta:       float = 0.10,
    annual_cap:      int   = 12,
    annual_counter:  dict  = None,
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
                days_since, cooldown_days
            )
            return current_weights, False, annual_counter

    # --- Guard 2: minimum delta ---
    assets = set(current_weights) | set(target_weights)
    max_delta = max(
        abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0))
        for a in assets
    )
    if max_delta < min_delta:
        logging.debug(
            "Gate BLOCKED (min_delta): max weight change %.1f%% < %.1f%%",
            max_delta * 100, min_delta * 100
        )
        return current_weights, False, annual_counter

    # --- Guard 3: annual cap ---
    count_this_year = annual_counter.get(year, 0)
    if count_this_year >= annual_cap:
        logging.warning(
            "Gate BLOCKED (annual cap): %d reallocations already in %d",
            count_this_year, year
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
        annual_counter[year], year,
    )
    return target_weights, True, annual_counter


# ============================================================
# LAYER 2 — WEIGHT MAPPING
# ============================================================

def signals_to_target_weights(
    sig_equity: int,
    sig_bond:   int,
    weights_both_on: dict,
    w_equity_only:   float = 1.0,
    w_bond_only:     float = 1.0,
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
            "bond":   0.0,
            "mmf":    1.0 - w_equity_only,
        }

    if sig_bond and not sig_equity:
        return {
            "equity": 0.0,
            "bond":   w_bond_only,
            "mmf":    1.0 - w_bond_only,
        }

    # Both off
    return {"equity": 0.0, "bond": 0.0, "mmf": 1.0}


# ============================================================
# LAYER 2 — ALLOCATION OPTIMISATION (both-signals-on state)
# ============================================================

def _feasible_weight_grid(step: float = 0.10) -> list:
    """
    Build all feasible (equity_weight, bond_weight) combinations that:
      - are multiples of `step`
      - equity + bond <= 1.0  (remainder is mmf)
      - equity >= step, bond >= step  (both assets get at least one unit)

    Returns list of dicts {'equity': e, 'bond': b, 'mmf': 1-e-b}.
    45 combinations for step=0.10.
    """
    combos = []
    levels = [round(i * step, 10) for i in range(1, int(1 / step) + 1)]
    for e in levels:
        for b in levels:
            if round(e + b, 10) <= 1.0:
                combos.append({
                    "equity": e,
                    "bond":   b,
                    "mmf":    round(1.0 - e - b, 10),
                })
    return combos


def optimise_both_on_weights(
    equity_returns: pd.Series,
    bond_returns:   pd.Series,
    mmf_returns:    pd.Series,
    sig_equity:     pd.Series,
    sig_bond:       pd.Series,
    step:           float = 0.10,
    objective:      str   = "calmar",
) -> dict:
    """
    Grid-search the optimal equity/bond split for the both-signals-on state.

    Called once per walk-forward training window as Phase 4 of the
    sequential optimisation (after per-asset signal parameters are fixed).

    The search only covers days when BOTH signals are simultaneously on,
    because only then does the split matter.  On days when one or both
    signals are off the weights are deterministic (see signals_to_target_weights).

    No reallocation gate is applied during training search — the gate is
    an execution constraint, not a signal-quality criterion.

    Parameters
    ----------
    equity_returns : pd.Series — daily returns of the equity asset
    bond_returns   : pd.Series — daily returns of the bond asset
    mmf_returns    : pd.Series — daily returns of the money-market fund
    sig_equity     : pd.Series — binary signal series for equity (0/1)
    sig_bond       : pd.Series — binary signal series for bond (0/1)
    step           : float     — weight grid step (default 0.10 = 10%)
    objective      : str       — 'calmar' | 'sharpe' | 'sortino'

    Returns
    -------
    dict — best {'equity': float, 'bond': float, 'mmf': float}
    """
    combos = _feasible_weight_grid(step)

    # Align all series to a common index
    common = equity_returns.index \
        .intersection(bond_returns.index) \
        .intersection(mmf_returns.index) \
        .intersection(sig_equity.index) \
        .intersection(sig_bond.index)

    eq_r  = equity_returns.loc[common]
    bd_r  = bond_returns.loc[common]
    mf_r  = mmf_returns.loc[common]
    s_eq  = sig_equity.loc[common]
    s_bd  = sig_bond.loc[common]

    # Days when both signals are on — only these rows differ across combos
    both_on = (s_eq == 1) & (s_bd == 1)

    best_score  = -np.inf
    best_combo  = combos[0]

    for combo in combos:
        # Build daily portfolio returns for the full period
        # When both on: weighted blend
        # When equity only: 100% equity
        # When bond only: 100% bond
        # When both off: 100% mmf
        port_r = pd.Series(0.0, index=common)

        eq_only = (s_eq == 1) & (s_bd == 0)
        bd_only = (s_eq == 0) & (s_bd == 1)
        both_off = (s_eq == 0) & (s_bd == 0)

        port_r[both_on]  = (combo["equity"] * eq_r
                            + combo["bond"]   * bd_r
                            + combo["mmf"]    * mf_r)[both_on]
        port_r[eq_only]  = eq_r[eq_only]
        port_r[bd_only]  = bd_r[bd_only]
        port_r[both_off] = mf_r[both_off]

        equity_curve = (1 + port_r).cumprod()
        if len(equity_curve) < 2:
            continue

        m = compute_metrics(equity_curve)

        if objective == "calmar":
            score = m["CalMAR"]
        elif objective == "sharpe":
            score = m["Sharpe"]
        elif objective == "sortino":
            score = m["Sortino"]
        else:
            raise ValueError(f"Unknown objective: {objective!r}")

        if score > best_score:
            best_score = score
            best_combo = combo

    logging.info(
        "Both-on weight optimisation: best equity=%.0f%% bond=%.0f%% mmf=%.0f%% "
        "(objective=%s score=%.3f)",
        best_combo["equity"] * 100,
        best_combo["bond"]   * 100,
        best_combo["mmf"]    * 100,
        objective, best_score,
    )
    return best_combo


# ============================================================
# SIGNAL EXTRACTION FROM WALK-FORWARD EQUITY CURVES
# ============================================================

def build_signal_series(wf_equity: pd.Series, wf_trades: pd.DataFrame) -> pd.Series:
    """
    Convert a walk-forward equity curve + trade log into a daily binary
    signal series (1 = in position, 0 = flat / in MMF).

    The trade log records entry and exit dates precisely.  We reconstruct
    the daily in/out state by marking every date between Entry Date and
    Exit Date (inclusive) as 1, everything else as 0.

    CARRY records (window-boundary snapshots) are excluded — they do not
    represent real entry/exit events.

    Parameters
    ----------
    wf_equity : pd.Series    — stitched OOS equity curve (DatetimeIndex)
    wf_trades : pd.DataFrame — trade log from walk_forward; must have columns
                               'Entry Date', 'Exit Date', 'Exit Reason'

    Returns
    -------
    pd.Series — binary 0/1 series aligned to wf_equity.index
    """
    signal = pd.Series(0, index=wf_equity.index, dtype=int)

    if wf_trades.empty:
        return signal

    boundary = {"CARRY", "SAMPLE_END"}
    closed_trades = wf_trades[
        ~wf_trades["Exit Reason"].isin(boundary)
    ].copy()

    # Also include the currently open trade if the last record is CARRY
    last = wf_trades.iloc[-1]
    if last["Exit Reason"] in boundary:
        # Position is open to end of data — mark to end
        entry = pd.Timestamp(last["EntryDate"])
        mask  = signal.index >= entry
        signal.loc[mask] = 1

    for _, row in closed_trades.iterrows():
        entry = pd.Timestamp(row["EntryDate"])
        exit_ = pd.Timestamp(row["ExitDate"])
        mask  = (signal.index >= entry) & (signal.index <= exit_)
        signal.loc[mask] = 1

    return signal


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def portfolio_returns(
    equity_returns: pd.Series,
    bond_returns:   pd.Series,
    mmf_returns:    pd.Series,
    weights_series: pd.Series,
) -> pd.Series:
    """
    Compute daily portfolio returns from three asset return series and a
    time-varying weights series.

    Parameters
    ----------
    equity_returns : pd.Series — daily returns of the equity asset
    bond_returns   : pd.Series — daily returns of the bond asset
    mmf_returns    : pd.Series — daily returns of the money-market fund
    weights_series : pd.Series — DatetimeIndex → dict of
                                 {'equity': float, 'bond': float, 'mmf': float}
                                 One entry per date (post-gate weights).

    Returns
    -------
    pd.Series — daily portfolio returns on the common date range
    """
    common = equity_returns.index \
        .intersection(bond_returns.index) \
        .intersection(mmf_returns.index) \
        .intersection(weights_series.index)

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
    Full simulation: daily signals → gate → weights → portfolio equity curve.

    Iterates day-by-day applying reallocation_gate on each date so that
    the cooldown and annual cap operate correctly in calendar time.

    Parameters
    ----------
    equity_returns   : pd.Series  — daily returns of the equity asset
    bond_returns     : pd.Series  — daily returns of the bond asset
    mmf_returns      : pd.Series  — daily returns of the money-market fund
    sig_equity       : pd.Series  — binary signal for equity (0/1)
    sig_bond         : pd.Series  — binary signal for bond (0/1)
    weights_both_on  : dict       — optimised split when both signals on
    cooldown_days    : int        — min calendar days between reallocations
    min_delta        : float      — min weight change to trigger reallocation
    annual_cap       : int        — max reallocations per calendar year
    w_equity_only    : float      — equity weight when only equity is on
    w_bond_only      : float      — bond weight when only bond is on

    Returns
    -------
    (equity_curve, weights_series, reallocation_log)
      equity_curve     : pd.Series  — portfolio equity curve starting at 1.0
      weights_series   : pd.Series  — per-date weight dicts (post-gate)
      reallocation_log : list[dict] — one entry per accepted reallocation
    """
    common = equity_returns.index \
        .intersection(bond_returns.index) \
        .intersection(mmf_returns.index) \
        .intersection(sig_equity.index) \
        .intersection(sig_bond.index)
    common = common.sort_values()

    eq_r = equity_returns.loc[common]
    bd_r = bond_returns.loc[common]
    mf_r = mmf_returns.loc[common]
    s_eq = sig_equity.reindex(common).fillna(0).astype(int)
    s_bd = sig_bond.reindex(common).fillna(0).astype(int)

    # State
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
    equity_curve = equity_curve / equity_curve.iloc[0]   # normalise to 1.0

    return equity_curve, weights_series, reallocation_log


# ============================================================
# ALLOCATION WALK-FORWARD  (Phase 4)
# ============================================================

def allocation_walk_forward(
    equity_returns:  pd.Series,
    bond_returns:    pd.Series,
    mmf_returns:     pd.Series,
    sig_equity_oos:  pd.Series,
    sig_bond_oos:    pd.Series,
    wf_results_eq:   pd.DataFrame,
    wf_results_bd:   pd.DataFrame,
    step:            float = 0.10,
    objective:       str   = "calmar",
    cooldown_days:   int   = 10,
    annual_cap:      int   = 12,
) -> tuple:
    """
    Walk-forward optimisation of the both-signals-on allocation weights.

    Uses the same window schedule as the per-asset signal walk-forwards
    (derived from wf_results_eq).  For each training window:
      1. Extract in-sample signal series for equity and bond.
      2. Grid-search the best equity/bond split via optimise_both_on_weights.
      3. Apply best weights to the OOS window with reallocation gate.
      4. Chain-link OOS equity slices.

    Parameters
    ----------
    equity_returns  : pd.Series      — full daily returns of equity asset
    bond_returns    : pd.Series      — full daily returns of bond asset
    mmf_returns     : pd.Series      — full daily returns of MMF
    sig_equity_oos  : pd.Series      — full OOS binary signal for equity
    sig_bond_oos    : pd.Series      — full OOS binary signal for bond
    wf_results_eq   : pd.DataFrame   — walk_forward results for equity asset;
                                       used to extract window schedule
    wf_results_bd   : pd.DataFrame   — walk_forward results for bond asset
    step            : float          — weight grid step
    objective       : str            — optimisation objective
    cooldown_days   : int            — reallocation gate cooldown
    annual_cap      : int            — reallocation gate annual cap

    Returns
    -------
    (portfolio_equity, weights_series, all_reallocation_log, alloc_results_df)
      portfolio_equity      : pd.Series   — stitched OOS portfolio equity curve
      weights_series        : pd.Series   — per-date weight dicts (post-gate)
      all_reallocation_log  : list[dict]
      alloc_results_df      : pd.DataFrame — per-window best weights
    """
    oos_equity_slices = []
    all_weights       = {}
    all_realloc_log   = []
    alloc_results     = []

    windows = wf_results_eq[["TrainStart", "TrainEnd",
                               "TestStart", "TestEnd"]].drop_duplicates()

    for _, row in windows.iterrows():
        train_start = pd.Timestamp(row["TrainStart"])
        train_end   = pd.Timestamp(row["TrainEnd"])
        test_start  = pd.Timestamp(row["TestStart"])
        test_end    = pd.Timestamp(row["TestEnd"])

        # --- Training: extract in-sample return and signal slices ---
        def _slice(s, start, end):
            return s.loc[(s.index >= start) & (s.index < end)]

        train_eq_r   = _slice(equity_returns, train_start, train_end)
        train_bd_r   = _slice(bond_returns,   train_start, train_end)
        train_mf_r   = _slice(mmf_returns,    train_start, train_end)
        train_s_eq   = _slice(sig_equity_oos, train_start, train_end)
        train_s_bd   = _slice(sig_bond_oos,   train_start, train_end)

        if len(train_eq_r) < 30:
            logging.warning(
                "Allocation WF: training window %s–%s too short (%d rows), skipping.",
                train_start.date(), train_end.date(), len(train_eq_r)
            )
            continue

        best_combo = optimise_both_on_weights(
            equity_returns = train_eq_r,
            bond_returns   = train_bd_r,
            mmf_returns    = train_mf_r,
            sig_equity     = train_s_eq,
            sig_bond       = train_s_bd,
            step           = step,
            objective      = objective,
        )

        alloc_results.append({
            "TrainStart":    train_start,
            "TrainEnd":      train_end,
            "TestStart":     test_start,
            "TestEnd":       test_end,
            "w_equity":      best_combo["equity"],
            "w_bond":        best_combo["bond"],
            "w_mmf":         best_combo["mmf"],
        })

        # --- OOS: simulate with gate ---
        test_eq_r = _slice(equity_returns, test_start, test_end)
        test_bd_r = _slice(bond_returns,   test_start, test_end)
        test_mf_r = _slice(mmf_returns,    test_start, test_end)
        test_s_eq = _slice(sig_equity_oos, test_start, test_end)
        test_s_bd = _slice(sig_bond_oos,   test_start, test_end)

        if len(test_eq_r) < 5:
            continue

        oos_equity, oos_weights, realloc_log = portfolio_equity_from_signals(
            equity_returns  = test_eq_r,
            bond_returns    = test_bd_r,
            mmf_returns     = test_mf_r,
            sig_equity      = test_s_eq,
            sig_bond        = test_s_bd,
            weights_both_on = best_combo,
            cooldown_days   = cooldown_days,
            annual_cap      = annual_cap,
        )

        # Chain-link
        if oos_equity_slices:
            prev_end  = oos_equity_slices[-1].iloc[-1]
            oos_equity = oos_equity * prev_end

        oos_equity_slices.append(oos_equity)
        all_weights.update(oos_weights.to_dict())
        all_realloc_log.extend(realloc_log)

    if not oos_equity_slices:
        logging.warning("Allocation walk-forward produced no OOS slices.")
        return pd.Series(dtype=float), pd.Series(dtype=object), [], pd.DataFrame()

    portfolio_equity = pd.concat(oos_equity_slices).sort_index()
    weights_series   = pd.Series(all_weights).sort_index()
    alloc_results_df = pd.DataFrame(alloc_results)

    return portfolio_equity, weights_series, all_realloc_log, alloc_results_df


# ============================================================
# REPORTING
# ============================================================

def print_multiasset_report(
    portfolio_metrics:  dict,
    bh_equity_metrics:  dict,
    bh_bond_metrics:    dict,
    alloc_results_df:   pd.DataFrame,
    reallocation_log:   list,
    sig_equity:         pd.Series,
    sig_bond:           pd.Series,
    oos_start,
    oos_end,
) -> None:
    """
    Log a structured multi-asset backtest report.

    Sections:
      1. Per-window allocation parameters (equity/bond/mmf split chosen
         in training for both-signals-on state).
      2. Portfolio aggregate metrics vs equity B&H vs bond B&H.
      3. Signal frequency statistics (% time each asset signal is on).
      4. Reallocation log summary (total count, annual breakdown).

    Parameters
    ----------
    portfolio_metrics  : dict         — output of compute_metrics on portfolio equity
    bh_equity_metrics  : dict         — B&H metrics for the equity asset
    bh_bond_metrics    : dict         — B&H metrics for the bond asset
    alloc_results_df   : pd.DataFrame — per-window allocation weights from
                                        allocation_walk_forward
    reallocation_log   : list[dict]   — from portfolio_equity_from_signals
    sig_equity         : pd.Series    — OOS binary signal for equity
    sig_bond           : pd.Series    — OOS binary signal for bond
    oos_start          : date-like
    oos_end            : date-like
    """
    sep = "=" * 80

    logging.info(sep)
    logging.info("MULTI-ASSET WALK-FORWARD OOS REPORT  (%s to %s)",
                 pd.Timestamp(oos_start).date(), pd.Timestamp(oos_end).date())
    logging.info(sep)

    # --- Per-window allocation weights ---
    if alloc_results_df is not None and not alloc_results_df.empty:
        logging.info("PER-WINDOW ALLOCATION PARAMETERS (both-signals-on state):")
        logging.info(
            "\n%s",
            alloc_results_df[["TrainStart", "TestStart", "TestEnd",
                               "w_equity", "w_bond", "w_mmf"]].to_string(index=False)
        )
    logging.info("-" * 80)

    # --- Performance comparison ---
    def _fmt(m):
        return (
            f"CAGR: {m['CAGR']*100:.2f}%  "
            f"Vol: {m['Vol']*100:.2f}%  "
            f"Sharpe: {m['Sharpe']:.2f}  "
            f"MaxDD: {m['MaxDD']*100:.2f}%  "
            f"CalMAR: {m['CalMAR']:.2f}  "
            f"Sortino: {m['Sortino']:.2f}"
        )

    logging.info("PERFORMANCE:")
    logging.info("  Portfolio          : %s", _fmt(portfolio_metrics))
    logging.info("  B&H Equity (WIG)   : %s", _fmt(bh_equity_metrics))
    logging.info("  B&H Bond   (TBSP)  : %s", _fmt(bh_bond_metrics))
    logging.info("-" * 80)

    # --- Signal frequency ---
    n = len(sig_equity)
    pct_eq = sig_equity.sum() / n * 100 if n > 0 else 0
    pct_bd = sig_bond.sum()   / n * 100 if n > 0 else 0
    both_on  = ((sig_equity == 1) & (sig_bond == 1)).sum() / n * 100
    both_off = ((sig_equity == 0) & (sig_bond == 0)).sum() / n * 100

    logging.info("SIGNAL FREQUENCY (OOS period):")
    logging.info("  Equity signal ON : %.1f%% of days", pct_eq)
    logging.info("  Bond signal ON   : %.1f%% of days", pct_bd)
    logging.info("  Both ON          : %.1f%% of days", both_on)
    logging.info("  Both OFF (MMF)   : %.1f%% of days", both_off)
    logging.info("-" * 80)

    # --- Reallocation summary ---
    n_realloc = len(reallocation_log)
    logging.info("REALLOCATION SUMMARY:")
    logging.info("  Total reallocations: %d", n_realloc)

    if n_realloc > 0:
        realloc_df = pd.DataFrame(reallocation_log)
        realloc_df["Year"] = pd.to_datetime(realloc_df["Date"]).dt.year
        annual_counts = realloc_df.groupby("Year").size()
        logging.info("  Annual breakdown:")
        for year, count in annual_counts.items():
            logging.info("    %d : %d", year, count)
        logging.info("  Reallocation log:")
        logging.info("\n%s", realloc_df.to_string(index=False))

    logging.info(sep)
