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
from __future__ import annotations
import itertools
import logging
import pandas as pd
import numpy as np

from strategy_test_library import (
    walk_forward,
    run_strategy_with_trades,
    compute_metrics,
    compute_buy_and_hold,
    
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
    - equity >= 0, bond >= 0  (either asset can be zero)
    Returns list of dicts {'equity': e, 'bond': b, 'mmf': 1-e-b}.
    66 combinations for step=0.10.
    """
    combos = []
    #  levels = [round(i * step, 10) for i in range(1, int(1 / step) + 1)] OLD version, more conservative, both assets get at least 10% when both signals are on
    levels = [round(i * step, 10) for i in range(0, int(1 / step) + 1)]
    for e in levels:
        for b in levels:
            if round(e + b, 10) <= 1.0:
                combos.append({
                    "equity": e,
                    "bond":   b,
                    "mmf":    max(0.0, round(1.0 - e - b, 10)),
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
        elif objective == "calmar_sortino":
            if m["CalMAR"] is None or m["Sortino"] is None:
                return None
            score = 0.5 * m["CalMAR"] + 0.5 * m["Sortino"]
        elif objective == "calmar_sharpe":
            if m["CalMAR"] is None or m["Sharpe"] is None:
                return None
            score = 0.5 * m["CalMAR"] + 0.5 * m["Sharpe"]

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

# ============================================================
# ALLOCATION WALK-FORWARD  (Phase 4)
# ============================================================

def allocation_walk_forward(
    equity_returns:     pd.Series,
    bond_returns:       pd.Series,
    mmf_returns:        pd.Series,
    sig_equity_full:    pd.Series,
    sig_bond_full:      pd.Series,
    sig_equity_oos:     pd.Series,
    sig_bond_oos:       pd.Series,
    wf_results_eq:      pd.DataFrame,
    wf_results_bd:      pd.DataFrame,
    step:               float = 0.10,
    objective:          str   = "calmar",
    cooldown_days:      int   = 10,
    annual_cap:         int   = 12,
) -> tuple:
    """
    Walk-forward optimisation of the both-signals-on allocation weights.

    Window schedule is driven by wf_results_bd (the bond asset, which has
    the shorter/later history and therefore defines the common OOS start).
    For each window in that schedule:
      1. Slice the full in-sample return and signal series up to train_end.
         These slices include the full history available — NOT just the OOS
         portion — so the optimiser has the maximum relevant data.
      2. Grid-search the best equity/bond split via optimise_both_on_weights.
      3. Apply best weights to the OOS test window with reallocation gate.
      4. Chain-link OOS equity slices.

    The reallocation gate state (last_change_date, annual_counter) is carried
    continuously across windows — it does not reset at each window boundary.

    Parameters
    ----------
    equity_returns   : pd.Series      — full daily returns of equity asset
    bond_returns     : pd.Series      — full daily returns of bond asset
    mmf_returns      : pd.Series      — full daily returns of MMF
    sig_equity_full  : pd.Series      — full binary signal for equity
                                        (covers entire equity OOS history)
    sig_bond_full    : pd.Series      — full binary signal for bond
                                        (covers entire bond OOS history)
    sig_equity_oos   : pd.Series      — equity signal trimmed to common OOS period
    sig_bond_oos     : pd.Series      — bond signal trimmed to common OOS period
    wf_results_eq    : pd.DataFrame   — walk_forward results for equity asset
    wf_results_bd    : pd.DataFrame   — walk_forward results for bond asset;
                                        used as the window schedule driver
    step             : float          — weight grid step
    objective        : str            — optimisation objective
    cooldown_days    : int            — reallocation gate cooldown
    annual_cap       : int            — reallocation gate annual cap

    Returns
    -------
    (portfolio_equity, weights_series, all_reallocation_log, alloc_results_df)
      portfolio_equity      : pd.Series   — stitched OOS portfolio equity curve
      weights_series        : pd.Series   — per-date weight dicts (post-gate)
      all_reallocation_log  : list[dict]
      alloc_results_df      : pd.DataFrame — per-window best weights
    """
    def _slice(s, start, end):
        return s.loc[(s.index >= start) & (s.index < end)]

    oos_equity_slices = []
    all_weights       = {}
    alloc_results     = []

    # Reallocation gate state persists across windows
    current_weights  = {"equity": 0.0, "bond": 0.0, "mmf": 1.0}
    last_change_date = None
    annual_counter   = {}
    all_realloc_log  = []

    # Use bond WF schedule — it starts later and defines the common OOS period
    windows = wf_results_bd[["TrainStart", "TrainEnd",
                              "TestStart", "TestEnd"]].drop_duplicates()
    prev_best_combo = None   # tracks best_combo from previous window
    for _, row in windows.iterrows():
        train_start = pd.Timestamp(row["TrainStart"])
        train_end   = pd.Timestamp(row["TrainEnd"])
        test_start  = pd.Timestamp(row["TestStart"])
        test_end    = pd.Timestamp(row["TestEnd"])

        # --- Training: full in-sample history up to train_end ---
        # Use all available return and signal data — not just the OOS window.
        # Both signals default to 0 (flat) outside their available history.
        train_eq_r = _slice(equity_returns,  train_start, train_end)
        train_bd_r = _slice(bond_returns,    train_start, train_end)
        train_mf_r = _slice(mmf_returns,     train_start, train_end)
        train_s_eq = _slice(sig_equity_full, train_start, train_end).reindex(
                         train_eq_r.index).fillna(0).astype(int)
        train_s_bd = _slice(sig_bond_full,   train_start, train_end).reindex(
                         train_bd_r.index).fillna(0).astype(int)

        if len(train_eq_r) < 30 or len(train_bd_r) < 30:
            logging.warning(
                "Allocation WF: training window %s–%s too short "
                "(eq=%d rows, bd=%d rows), skipping.",
                train_start.date(), train_end.date(),
                len(train_eq_r), len(train_bd_r),
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
            "TrainStart": train_start,
            "TrainEnd":   train_end,
            "TestStart":  test_start,
            "TestEnd":    test_end,
            "w_equity":   best_combo["equity"],
            "w_bond":     best_combo["bond"],
            "w_mmf":      best_combo["mmf"],
        })

        # --- OOS: simulate with gate (gate state carried across windows) ---
        test_eq_r = _slice(equity_returns,  test_start, test_end)
        test_bd_r = _slice(bond_returns,    test_start, test_end)
        test_mf_r = _slice(mmf_returns,     test_start, test_end)
        test_s_eq = _slice(sig_equity_oos,  test_start, test_end)
        test_s_bd = _slice(sig_bond_oos,    test_start, test_end)

        if len(test_eq_r) < 5:
            continue

        # Align to common dates for this test window
        common = (test_eq_r.index
                  .intersection(test_bd_r.index)
                  .intersection(test_mf_r.index))
        common = common.sort_values()

        eq_r = test_eq_r.loc[common]
        bd_r = test_bd_r.loc[common]
        mf_r = test_mf_r.loc[common]
        s_eq = test_s_eq.reindex(common).fillna(0).astype(int)
        s_bd = test_s_bd.reindex(common).fillna(0).astype(int)

        window_equity_vals = []
        
        
        for date in common:
            target = signals_to_target_weights(
                sig_equity      = int(s_eq[date]),
                sig_bond        = int(s_bd[date]),
                weights_both_on = best_combo,
            )

            accepted, reallocated, annual_counter = reallocation_gate(
                current_weights  = current_weights,
                target_weights   = target,
                last_change_date = last_change_date,
                current_date     = date,
                cooldown_days    = cooldown_days,
                annual_cap       = annual_cap,
                annual_counter   = annual_counter,
            )

            if reallocated:
                
                param_updated = (
                date == common[0]
                and prev_best_combo is not None
                and prev_best_combo != best_combo
                )

                if param_updated:
                    # Would the target have been the same under the old weights?
                    old_target = signals_to_target_weights(
                    sig_equity      = int(s_eq[date]),
                    sig_bond        = int(s_bd[date]),
                    weights_both_on = prev_best_combo,
                    )   
                    signal_also_changed = (old_target != target)
                    reason = "BOTH" if signal_also_changed else "PARAM_UPDATE"
                else:
                    reason = "SIGNAL_CHANGE"


                all_realloc_log.append({
                    "Date":          date,
                    "equity_before": current_weights["equity"],
                    "bond_before":   current_weights["bond"],
                    "mmf_before":    current_weights["mmf"],
                    "equity_after":  accepted["equity"],
                    "bond_after":    accepted["bond"],
                    "mmf_after":     accepted["mmf"],
                    "reason":        reason,
                })
                current_weights  = accepted
                last_change_date = date

            all_weights[date] = dict(current_weights)

            w = current_weights
            r = (w["equity"] * eq_r[date]
                 + w["bond"]  * bd_r[date]
                 + w["mmf"]   * mf_r[date])
            window_equity_vals.append((date, r))
            
            
            
        if not window_equity_vals:
            continue

        dates, rets = zip(*window_equity_vals)
        window_port_r  = pd.Series(list(rets), index=list(dates))
        window_equity  = (1 + window_port_r).cumprod()
        window_equity  = window_equity / window_equity.iloc[0]

        if oos_equity_slices:
            window_equity = window_equity * oos_equity_slices[-1].iloc[-1]

        oos_equity_slices.append(window_equity)
        prev_best_combo = best_combo   
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
    sig_bond_raw        = None,
    spread_prefilter    = None,
    yield_prefilter     = None,          # NEW
    sig_bond_post_spread = None,         # NEW — intermediate state between filters
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
    logging.info("  Equity signal ON          : %.1f%% of days", pct_eq)
    logging.info("  Bond signal ON (raw)      : %.1f%% of days",
         sig_bond_raw.sum() / n * 100 if sig_bond_raw is not None else pct_bd)
    logging.info("  Bond signal ON (filtered) : %.1f%% of days", pct_bd)

    if sig_bond_raw is not None:
        # Spread filter contribution
        sig_post_spread = sig_bond_post_spread if sig_bond_post_spread is not None else sig_bond
        n_blocked_spread = (sig_bond_raw - sig_post_spread).clip(lower=0).reindex(sig_bond.index).fillna(0).sum()
        logging.info(
            "  Spread pre-filter blocked : %d days  (%.1f%% of raw bond-on days)",
            int(n_blocked_spread),
            n_blocked_spread / sig_bond_raw.sum() * 100 if sig_bond_raw.sum() > 0 else 0,
    )

    # Yield filter contribution (only shown if intermediate state available)
    if sig_bond_post_spread is not None:
        n_blocked_yield = (sig_bond_post_spread - sig_bond).clip(lower=0).reindex(sig_bond.index).fillna(0).sum()
        logging.info(
            "  Yield pre-filter blocked  : %d days  (%.1f%% of post-spread bond-on days)",
            int(n_blocked_yield),
            n_blocked_yield / sig_bond_post_spread.sum() * 100 if sig_bond_post_spread.sum() > 0 else 0,
        )

    # Combined total blocked across both filters
    n_blocked_total = (sig_bond_raw - sig_bond).clip(lower=0).reindex(sig_bond.index).fillna(0).sum()
    if sig_bond_post_spread is not None:
        logging.info(
            "  Both filters blocked      : %d days total  (%.1f%% of raw bond-on days)",
            int(n_blocked_total),
            n_blocked_total / sig_bond_raw.sum() * 100 if sig_bond_raw.sum() > 0 else 0,
        )

    if spread_prefilter is not None:
        pf_aligned = spread_prefilter.reindex(sig_bond.index).fillna(1)
        logging.info(
        "  Spread pre-filter ON      : %.1f%% of OOS days",
        pf_aligned.mean() * 100,
    )

    if yield_prefilter is not None:
        yf_aligned = yield_prefilter.reindex(sig_bond.index).fillna(1)
        logging.info(
        "  Yield pre-filter ON       : %.1f%% of OOS days",
        yf_aligned.mean() * 100,
    )

    logging.info("  Both ON                   : %.1f%% of days", both_on)
    logging.info("  Both OFF (MMF)            : %.1f%% of days", both_off)
    logging.info("-" * 80)

    
    if sig_bond_raw is not None:
        blocked = (sig_bond_raw - sig_bond).clip(lower=0)
        blocked_oos = blocked.reindex(sig_bond.index).fillna(0)
        if blocked_oos.sum() > 0:
            blocked_df = pd.DataFrame({
                "blocked":     blocked_oos,
                "bond_raw_on": sig_bond_raw.reindex(sig_bond.index).fillna(0),
                })
            blocked_df["Year"] = blocked_df.index.year
            annual = blocked_df.groupby("Year").agg(
                blocked_days = ("blocked",     "sum"),
                bond_on_days = ("bond_raw_on", "sum"),
                )
            annual["pct_blocked"] = (
                annual["blocked_days"] / annual["bond_on_days"].replace(0, np.nan) * 100
                ).fillna(0)
            logging.info("  Pre-filter blocking by year:")
            for yr, row in annual[annual["blocked_days"] > 0].iterrows():
                logging.info(
                    "    %d : %3d days blocked  (%.0f%% of bond-on days that year)",
                    yr, int(row["blocked_days"]), row["pct_blocked"],
                    )


    # --- Reallocation summary ---
    n_realloc = len(reallocation_log)
    logging.info("REALLOCATION SUMMARY:")
    logging.info("  Total reallocations: %d", n_realloc)
    if sig_bond_raw is not None:
        logging.info(
            "  Note: bond signal includes spread pre-filter — "
            "transitions to/from bond reflect both MA signal and macro gate."
            )

    if n_realloc > 0:
        realloc_df = pd.DataFrame(reallocation_log)
        realloc_df["Year"] = pd.to_datetime(realloc_df["Date"]).dt.year
        annual_counts = realloc_df.groupby("Year").size()
        logging.info("  Annual breakdown:")
        for year, count in annual_counts.items():
            logging.info("    %d : %d", year, count)
        logging.info("  Reallocation log:")
        logging.info("\n%s", realloc_df[
            ["Date", "equity_before", "bond_before", "mmf_before",
            "equity_after", "bond_after", "mmf_after", "reason", "Year"]
            ].to_string(index=False))

    logging.info(sep)


# ============================================================
# LEVEL 2 MC — ALLOCATION WEIGHT PERTURBATION ROBUSTNESS
# ============================================================

def allocation_weight_robustness(
    alloc_results_df:   pd.DataFrame,
    equity_returns:     pd.Series,
    bond_returns:       pd.Series,
    mmf_returns:        pd.Series,
    sig_equity_oos:     pd.Series,
    sig_bond_oos:       pd.Series,
    baseline_metrics:   dict,
    perturb_steps:      list  = [-0.2, -0.1, 0.0, 0.1, 0.2],
    min_equity:         float = 0.10,
    max_equity:         float = 1.00,
    cooldown_days:      int   = 10,
    annual_cap:         int   = 12,
) -> pd.DataFrame:
    """
    Level 2 robustness check: perturb the optimised both-signals-on equity
    weight by ±10pp and ±20pp steps and re-simulate the full OOS portfolio
    for each perturbation.

    For each window in alloc_results_df the optimised w_equity is shifted by
    each value in perturb_steps (e.g. ±0.10, ±0.20).  Bond weight is held
    fixed at the optimised value (typically 0.10).  MMF absorbs the remainder.
    All windows shift by the same step simultaneously — this tests the
    sensitivity of the combined portfolio to the equity allocation level,
    not individual window choices.

    The reallocation gate state is carried continuously across windows,
    matching the live simulation exactly.

    ---------------------------------------------------------------
    INTERPRETATION GUIDE
    ---------------------------------------------------------------
    Ideal outcome: CalMAR, Sharpe, and MaxDD are stable across the
    ±10–20pp range.  This indicates the both-on equity weight is on a
    broad plateau — the exact optimised value matters less than the
    general direction (equity-heavy vs bond-heavy vs balanced).

    Concerning outcome: metrics drop sharply at ±10pp.  This indicates
    the optimised weight sits on a narrow spike — the portfolio is
    sensitive to the precise allocation and the Phase 4 result should
    be treated with caution.

    Separately review whether the equity_weight direction is stable:
    if most windows chose 90% equity, the portfolio is essentially an
    equity strategy with a bond sleeve, and Phase 4 is adding little
    diversification value.

    Parameters
    ----------
    alloc_results_df : pd.DataFrame — per-window optimised weights from
                                      allocation_walk_forward; must have
                                      columns TestStart, TestEnd,
                                      w_equity, w_bond, w_mmf.
    equity_returns   : pd.Series   — full daily returns of equity asset
    bond_returns     : pd.Series   — full daily returns of bond asset
    mmf_returns      : pd.Series   — full daily returns of MMF
    sig_equity_oos   : pd.Series   — equity signal trimmed to common OOS period
    sig_bond_oos     : pd.Series   — bond signal trimmed to common OOS period
    baseline_metrics : dict        — compute_metrics output for the baseline
                                     portfolio (step=0.0 reference)
    perturb_steps    : list[float] — equity weight shifts to apply; 0.0 is
                                     included to reproduce the baseline as a
                                     sanity check.  Default: ±0.10, ±0.20.
    min_equity       : float       — floor on perturbed equity weight (default 0.10)
    max_equity       : float       — ceiling on perturbed equity weight (default 1.00)
    cooldown_days    : int         — reallocation gate cooldown (must match
                                     the value used in the main run)
    annual_cap       : int         — reallocation gate annual cap

    Returns
    -------
    pd.DataFrame — one row per perturbation step with columns:
        equity_shift    : the applied shift (e.g. -0.20, -0.10, 0.0, ...)
        w_equity_mean   : mean equity weight across windows after perturbation
        CAGR, Vol, Sharpe, MaxDD, CalMAR, Sortino
        n_reallocations : total reallocations over the OOS period
        CAGR_vs_base    : CAGR difference from baseline (pp)
        MaxDD_vs_base   : MaxDD difference from baseline (pp)
        CalMAR_vs_base  : CalMAR ratio vs baseline (perturbed / baseline)
    """
    def _slice(s, start, end):
        return s.loc[(s.index >= start) & (s.index < end)]

    results = []

    for step in perturb_steps:

        # Build perturbed weight set — one dict per window
        perturbed_windows = []
        n_floor_clamped = 0
        n_ceil_clamped  = 0
        for _, row in alloc_results_df.iterrows():
            raw_eq  = float(row["w_equity"]) + step
            w_eq    = round(raw_eq, 10)
            clamped = ""
            if w_eq < min_equity:
                w_eq    = min_equity
                n_floor_clamped += 1
                clamped = f"  [floor-clamped: raw={raw_eq:.3f} → {min_equity:.3f}]"
            elif w_eq > max_equity:
                w_eq    = max_equity
                n_ceil_clamped += 1
                clamped = f"  [ceil-clamped:  raw={raw_eq:.3f} → {max_equity:.3f}]"
            w_bd   = float(row["w_bond"])               # bond weight held fixed
            w_mmf  = max(0.0, round(1.0 - w_eq - w_bd, 10))
            if clamped:
                logging.debug(
                    "step=%+.2f  window %s%s",
                    step, pd.Timestamp(row["TestStart"]).date(), clamped
                )
            perturbed_windows.append({
                "TestStart": pd.Timestamp(row["TestStart"]),
                "TestEnd":   pd.Timestamp(row["TestEnd"]),
                "w_equity":  w_eq,
                "w_bond":    w_bd,
                "w_mmf":     w_mmf,
            })

        total_windows = len(perturbed_windows)
        if n_floor_clamped or n_ceil_clamped:
            logging.info(
                "step=%+.2f: %d/%d windows clamped at floor (min_equity=%.2f), "
                "%d/%d at ceiling (max_equity=%.2f). "
                "Clamped windows did not receive the full shift — "
                "w_equity_mean will diverge from (baseline_mean + step) for this row.",
                step,
                n_floor_clamped, total_windows, min_equity,
                n_ceil_clamped,  total_windows, max_equity,
            )
        else:
            logging.debug(
                "step=%+.2f: all %d windows shifted without clamping.",
                step, total_windows
            )

        # Simulate full OOS — gate state carried continuously across windows
        current_weights   = {"equity": 0.0, "bond": 0.0, "mmf": 1.0}
        last_change_date  = None
        annual_counter    = {}
        n_reallocations   = 0
        oos_equity_slices = []

        for pw in perturbed_windows:
            test_start = pw["TestStart"]
            test_end   = pw["TestEnd"]
            combo      = {"equity": pw["w_equity"],
                          "bond":   pw["w_bond"],
                          "mmf":    pw["w_mmf"]}

            test_eq_r = _slice(equity_returns, test_start, test_end)
            test_bd_r = _slice(bond_returns,   test_start, test_end)
            test_mf_r = _slice(mmf_returns,    test_start, test_end)
            test_s_eq = _slice(sig_equity_oos, test_start, test_end)
            test_s_bd = _slice(sig_bond_oos,   test_start, test_end)

            if len(test_eq_r) < 5:
                continue

            common = (test_eq_r.index
                      .intersection(test_bd_r.index)
                      .intersection(test_mf_r.index))
            common = common.sort_values()

            eq_r = test_eq_r.loc[common]
            bd_r = test_bd_r.loc[common]
            mf_r = test_mf_r.loc[common]
            s_eq = test_s_eq.reindex(common).fillna(0).astype(int)
            s_bd = test_s_bd.reindex(common).fillna(0).astype(int)

            window_rets = []

            for date in common:
                target = signals_to_target_weights(
                    sig_equity      = int(s_eq[date]),
                    sig_bond        = int(s_bd[date]),
                    weights_both_on = combo,
                )

                accepted, reallocated, annual_counter = reallocation_gate(
                    current_weights  = current_weights,
                    target_weights   = target,
                    last_change_date = last_change_date,
                    current_date     = date,
                    cooldown_days    = cooldown_days,
                    annual_cap       = annual_cap,
                    annual_counter   = annual_counter,
                )

                if reallocated:
                    current_weights  = accepted
                    last_change_date = date
                    n_reallocations += 1

                w = current_weights
                window_rets.append((
                    date,
                    w["equity"] * eq_r[date]
                    + w["bond"]  * bd_r[date]
                    + w["mmf"]   * mf_r[date]
                ))

            if not window_rets:
                continue

            dates, rets  = zip(*window_rets)
            port_r       = pd.Series(list(rets), index=list(dates))
            eq_curve     = (1 + port_r).cumprod()
            eq_curve     = eq_curve / eq_curve.iloc[0]

            if oos_equity_slices:
                eq_curve = eq_curve * oos_equity_slices[-1].iloc[-1]
            oos_equity_slices.append(eq_curve)

        if not oos_equity_slices:
            logging.warning(
                "Perturbation step=%.2f produced no OOS slices — skipped.", step
            )
            continue

        portfolio_equity = pd.concat(oos_equity_slices).sort_index()
        m = compute_metrics(portfolio_equity)

        w_eq_mean = np.mean([pw["w_equity"] for pw in perturbed_windows])

        results.append({
            "equity_shift":    step,
            "w_equity_mean":   round(w_eq_mean, 3),
            "n_floor_clamped": n_floor_clamped,
            "n_ceil_clamped":  n_ceil_clamped,
            "CAGR":            m["CAGR"],
            "Vol":             m["Vol"],
            "Sharpe":          m["Sharpe"],
            "MaxDD":           m["MaxDD"],
            "CalMAR":          m["CalMAR"],
            "Sortino":         m["Sortino"],
            "n_reallocations": n_reallocations,
        })

    if not results:
        logging.warning("allocation_weight_robustness: no results produced.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Comparison columns relative to the step=0.0 baseline
    base_row = df.loc[df["equity_shift"].abs() < 1e-9]
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
            "Baseline step (0.0) not found in perturb_steps — "
            "vs_base columns will be NaN. Add 0.0 to perturb_steps."
        )
        df["CAGR_vs_base"]   = np.nan
        df["MaxDD_vs_base"]  = np.nan
        df["CalMAR_vs_base"] = np.nan

    return df


def print_allocation_robustness_report(results_df: pd.DataFrame) -> None:
    """
    Log a structured report for the allocation weight perturbation analysis.

    Sections:
      1. Sensitivity table — all perturb_steps with metrics and vs-baseline
         comparison columns.
      2. Verdict — PLATEAU / MODERATE / SPIKE based on CalMAR ratio at ±10pp.

    The verdict threshold uses CalMAR ratio at the ±10pp steps (the tightest
    perturbation that is still meaningful):
      >= 0.80 : PLATEAU   — robust, weight is on a broad performance plateau
      0.60–0.80: MODERATE  — somewhat sensitive, treat with moderate confidence
      < 0.60  : SPIKE     — fragile, Phase 4 weight should be treated with caution

    Parameters
    ----------
    results_df : pd.DataFrame — output of allocation_weight_robustness
    """
    sep = "=" * 80
    logging.info(sep)
    logging.info("ALLOCATION WEIGHT ROBUSTNESS  (Level 2 — equity weight perturbation)")
    logging.info(sep)

    if results_df.empty:
        logging.warning("No results to report.")
        return

    # Format display table
    display = results_df.copy()
    display["CAGR"]           = (display["CAGR"]  * 100).round(2).astype(str) + "%"
    display["Vol"]            = (display["Vol"]   * 100).round(2).astype(str) + "%"
    display["MaxDD"]          = (display["MaxDD"] * 100).round(2).astype(str) + "%"
    display["Sharpe"]         = display["Sharpe"].round(3)
    display["CalMAR"]         = display["CalMAR"].round(3)
    display["Sortino"]        = display["Sortino"].round(3)
    display["CAGR_vs_base"]   = display["CAGR_vs_base"].round(2).astype(str) + "pp"
    display["MaxDD_vs_base"]  = display["MaxDD_vs_base"].round(2).astype(str) + "pp"
    display["CalMAR_vs_base"] = display["CalMAR_vs_base"].round(3)
    display["equity_shift"]   = display["equity_shift"].apply(lambda x: f"{x:+.0%}")
    display["w_equity_mean"]  = display["w_equity_mean"].apply(lambda x: f"{x:.0%}")

    cols = ["equity_shift", "w_equity_mean",
            "n_floor_clamped", "n_ceil_clamped",
            "CAGR", "Vol", "Sharpe", "MaxDD", "CalMAR", "Sortino",
            "n_reallocations", "CAGR_vs_base", "MaxDD_vs_base", "CalMAR_vs_base"]
    logging.info("\n%s", display[cols].to_string(index=False))
    logging.info("-" * 80)

    # --- Verdict based on minimum CalMAR ratio at ±10pp ---
    inner = results_df.loc[results_df["equity_shift"].abs().between(0.09, 0.11)]

    if inner.empty:
        logging.info(
            "Verdict: ±10pp steps not present in results — "
            "add ±0.10 to perturb_steps for a verdict."
        )
    else:
        min_ratio = inner["CalMAR_vs_base"].min()
        if pd.isna(min_ratio):
            logging.info("Verdict: baseline CalMAR is zero — ratio undefined.")
        elif min_ratio >= 0.80:
            logging.info(
                "Verdict: PLATEAU  (min CalMAR ratio at ±10pp = %.3f >= 0.80). "
                "Phase 4 allocation weight sits on a broad performance plateau. "
                "Result is robust to ±10pp equity shifts.",
                min_ratio
            )
        elif min_ratio >= 0.60:
            logging.info(
                "Verdict: MODERATE SENSITIVITY  (min CalMAR ratio at ±10pp = %.3f, "
                "range 0.60–0.80). Portfolio is somewhat sensitive to the exact "
                "equity weight. Treat Phase 4 result with moderate confidence.",
                min_ratio
            )
        else:
            logging.info(
                "Verdict: SPIKE  (min CalMAR ratio at ±10pp = %.3f < 0.60). "
                "Portfolio is highly sensitive to the exact equity allocation. "
                "Phase 4 optimised weight should be treated with caution.",
                min_ratio
            )

    logging.info(
        "Note: bond weight held fixed at per-window optimised value throughout. "
        "Only equity weight was perturbed; MMF absorbed the remainder."
    )
    logging.info(sep)


def build_spread_prefilter(
    pl10y:              pd.DataFrame,
    de10y:              pd.DataFrame,
    spread_change_window: int   = 63,
    spread_change_cap_bp: float = 50.0,
) -> pd.Series:
    """
    Build a binary pre-filter series for the TBSP bond signal based on
    the 3-month rolling change in the PL-DE 10Y government bond spread.

    A value of 1 means the macro environment is acceptable for holding bonds.
    A value of 0 means the spread has risen sharply — do not hold TBSP.

    Logic: filter = 1  if  (spread_today - spread_63d_ago) < cap_bp
           filter = 0  otherwise

    The spread is computed as PL10Y - DE10Y in percentage points,
    converted to basis points for the threshold comparison.

    Parameters
    ----------
    pl10y                : pd.DataFrame — PL 10Y yield (stooq CSV, close col)
    de10y                : pd.DataFrame — DE 10Y yield (stooq CSV, close col)
    spread_change_window : int          — lookback in trading days (~3 months)
    spread_change_cap_bp : float        — threshold in basis points (default 50)

    Returns
    -------
    pd.Series (int 0/1, DatetimeIndex) — 1 = filter passes, 0 = blocked
    """
    pl_close = pl10y.iloc[:, 1]   # second column = Zamkniecie
    de_close = de10y.iloc[:, 1]

    # Align on union, forward-fill yield gaps (yields have no weekend fixings)
    spread_pp = (pl_close - de_close).sort_index().ffill()
    spread_bp = spread_pp * 100

    # 3-month rolling change
    spread_chg = spread_bp - spread_bp.shift(spread_change_window)

    # Filter: 1 if change below cap, 0 if cap breached
    prefilter = (spread_chg < spread_change_cap_bp).astype(int)
    prefilter.name = "spread_prefilter"

    logging.info(
        "Spread pre-filter: %.1f%% of days ON  |  "
        "threshold=%.0fbp  window=%dd  "
        "(series: %s to %s)",
        prefilter.mean() * 100,
        spread_change_cap_bp,
        spread_change_window,
        prefilter.index.min().date(),
        prefilter.index.max().date(),
    )
    return prefilter



def build_yield_price_proxy(
    pl10y: pd.DataFrame,
    coupon: float = 0.05,
    n_years: int = 10,
    freq: int = 2,
) -> pd.DataFrame:
    """
    Construct a constant-maturity bond price series from PL 10Y yield.

    Models a par-100 fixed coupon bond repriced daily at the current YTM.
    Price is duration-weighted and behaves like a tradeable instrument —
    suitable as a drop-in replacement for TBSP as the signal source in
    walk_forward.

    Parameters
    ----------
    pl10y   : pd.DataFrame — PL 10Y yield (stooq CSV, close col = col index 1)
    coupon  : float        — annual coupon rate (default 5%)
    freq    : int          — coupon frequency per year (2 = semi-annual)
    n_years : int          — bond maturity in years (10)

    Returns
    -------
    pd.DataFrame with same index as pl10y, single column "Zamkniecie"
    containing the synthetic bond price. Ready to pass directly to
    walk_forward as the price input.
    """
    ytm_series = pl10y.iloc[:, 1].copy() / 100   # convert % to decimal

    def _price(ytm):
        if pd.isna(ytm) or ytm <= 0:
            return np.nan
        c = coupon / freq * 100
        r = ytm / freq
        n = n_years * freq
        return c * (1 - (1 + r) ** -n) / r + 100 * (1 + r) ** -n

    prices = ytm_series.apply(_price)

    out = pd.DataFrame({"Zamkniecie": prices}, index=pl10y.index)

    logging.info(
        "Yield price proxy: %d rows  (%s to %s)  "
        "price range [%.1f, %.1f]  at current yield %.2f%% -> price %.1f",
        len(out),
        out.index.min().date(),
        out.index.max().date(),
        out["Zamkniecie"].min(),
        out["Zamkniecie"].max(),
        ytm_series.iloc[-1] * 100,
        prices.iloc[-1],
    )
    return out


def build_yield_momentum_prefilter(
    pl10y: pd.DataFrame,
    rise_threshold_bp: float = 50.0,
    lookback_days: int = 63,
) -> pd.Series:
    """
    Yield momentum pre-filter for the bond signal.

    Blocks new bond entries (and optionally forces exits) when PL 10Y yield
    has risen sharply over the recent lookback window — indicating an adverse
    rate environment for holding bonds.

    Logic: filter = 1  if  (yield_today - yield_63d_ago) * 100 < threshold_bp
           filter = 0  otherwise  (yield rising sharply — block bond entry)

    A threshold of 50bp means: block entry if yield has risen more than 50bp
    over the past 63 trading days (~3 months).

    Parameters
    ----------
    pl10y             : pd.DataFrame — PL 10Y yield (stooq CSV, close col = col 1)
                        Values expected in percent (e.g. 5.5 for 5.5%)
    rise_threshold_bp : float        — threshold in basis points (default 50bp)
    lookback_days     : int          — lookback in trading days (default 63 ~3m)

    Returns
    -------
    pd.Series (int 0/1, DatetimeIndex) — 1 = filter passes, 0 = blocked
    """
    ytm = pl10y.iloc[:, 1].copy().ffill()   # forward-fill weekend gaps

    # Change in basis points over lookback window
    ytm_bp    = ytm * 100
    ytm_chg   = ytm_bp - ytm_bp.shift(lookback_days)

    # Filter: 1 if yield change below threshold, 0 if rising sharply
    prefilter = (ytm_chg < rise_threshold_bp).astype(int)
    prefilter.name = "yield_momentum_prefilter"

    pct_on = prefilter.mean() * 100
    logging.info(
        "Yield momentum pre-filter: %.1f%% of days ON  |  "
        "threshold=%.0fbp  window=%dd  "
        "(series: %s to %s)",
        pct_on,
        rise_threshold_bp,
        lookback_days,
        prefilter.index.min().date(),
        prefilter.index.max().date(),
    )
    return prefilter






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
    ret_mmf = mmf_ext["Zamkniecie"].pct_change().dropna().rename("ret_mmf")

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
