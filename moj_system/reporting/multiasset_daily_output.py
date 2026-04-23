# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:45:18 2026

@author: adamg
"""

"""
multiasset_daily_output.py
==========================
Daily output module for the multi-asset pension fund strategy
(WIG equity + TBSP bond + MMF).

Produces four artefacts into the specified output directory:

  multiasset_signal_status.txt   — human-readable status block (email body)
  multiasset_signal_log.csv      — appended daily history, one row per run
  multiasset_equity_chart.png    — 3-panel chart: portfolio equity,
                                   drawdown, signal + reallocation events
  multiasset_signal_snapshot.json — machine-readable snapshot for GH
                                    Actions parse step and Sheets import

Infrastructure (atomic writes, Drive pre-fetch, log append) is provided by
daily_output_base.py and shared with daily_output.py and
global_equity_daily_output.py.
"""



import json
import logging
import os
import tempfile
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moj_system.reporting.output_base import (
    atomic_write,
    atomic_write_bytes,
    load_existing_log,
    append_log_row,
    fetch_file_from_drive,
)
from moj_system.core.research import get_current_adx_regime
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}



# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _get_signal_from_trades(wf_trades: pd.DataFrame) -> str:
    """Return 'IN' if a CARRY record exists (position open), else 'OUT'."""
    if wf_trades is None or wf_trades.empty:
        return "OUT"
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    return "IN" if not carry.empty else "OUT"


def _get_open_position(wf_trades: pd.DataFrame) -> dict | None:
    """Return the most recent CARRY record as a dict, or None if flat."""
    if wf_trades is None or wf_trades.empty:
        return None
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    if carry.empty:
        return None
    return carry.iloc[-1].to_dict()


def _get_active_window_params(wf_results: pd.DataFrame) -> dict:
    """Return params from the last walk-forward window."""
    if wf_results is None or wf_results.empty:
        return {}
    last = wf_results.iloc[-1]

    def _safe_float(key, default=None):
        v = last.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _safe_int(key, default=None):
        v = _safe_float(key)
        return int(v) if v is not None else default

    return {
        "filter_mode":  last.get("filter_mode", "ma"),
        "X":            _safe_float("X",         0.10),
        "Y":            _safe_float("Y",         0.10),
        "fast":         _safe_int("fast",          50),
        "slow":         _safe_int("slow",         200),
        "stop_loss":    _safe_float("stop_loss",  0.10),
        "TestStart":    pd.Timestamp(last["TestStart"]),
        "TestEnd":      pd.Timestamp(last["TestEnd"]),
        "mom_lookback": _safe_int("mom_lookback", 252),
    }


def _compute_ma_filter_state(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    price_col: str = "Zamkniecie",
) -> dict:
    prices  = df[price_col].dropna()
    fast_ma = float(prices.rolling(fast).mean().iloc[-1]) if len(prices) >= fast else None
    slow_ma = float(prices.rolling(slow).mean().iloc[-1]) if len(prices) >= slow else None
    if fast_ma is not None and slow_ma is not None:
        filter_on = fast_ma > slow_ma
        gap_pct   = round((fast_ma / slow_ma - 1) * 100, 3)
    else:
        filter_on = None
        gap_pct   = None
    return {
        "fast_ma":  round(fast_ma, 2) if fast_ma else None,
        "slow_ma":  round(slow_ma, 2) if slow_ma else None,
        "filter_on": filter_on,
        "gap_pct":   gap_pct,
    }


def _get_current_weights(weights_series: pd.Series | None) -> dict:
    """Return the most recent allocation weight dict from weights_series."""
    if weights_series is None or weights_series.empty:
        return {"equity": None, "bond": None, "mmf": None}
    last = weights_series.iloc[-1]
    if isinstance(last, dict):
        return {
            "equity": round(float(last.get("equity", 0)), 4),
            "bond":   round(float(last.get("bond",   0)), 4),
            "mmf":    round(float(last.get("mmf",    0)), 4),
        }
    return {"equity": None, "bond": None, "mmf": None}


def _realloc_today(reallocation_log: list, run_date: dt.date) -> bool:
    """Return True if a reallocation event occurred on run_date."""
    if not reallocation_log:
        return False
    return any(
        pd.Timestamp(rec["Date"]).date() == run_date
        for rec in reallocation_log
    )


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------

def _determine_action(
    prev_log:       pd.DataFrame | None,
    sig_eq_today:   str,
    sig_bd_today:   str,
    realloc_today:  bool,
) -> str:
    """Return composite action string (HOLD, EQ_ENTER, BD_EXIT, REALLOC, etc.)."""
    parts = []

    if prev_log is None or prev_log.empty:
        if sig_eq_today == "IN":
            parts.append("EQ_ENTER")
        if sig_bd_today == "IN":
            parts.append("BD_ENTER")
        if not parts:
            parts.append("HOLD")
        return "+".join(parts)

    prev_eq = str(prev_log["signal_equity"].iloc[-1]).upper()
    prev_bd = str(prev_log["signal_bond"].iloc[-1]).upper()

    if prev_eq != sig_eq_today:
        parts.append("EQ_ENTER" if sig_eq_today == "IN" else "EQ_EXIT")
    if prev_bd != sig_bd_today:
        parts.append("BD_ENTER" if sig_bd_today == "IN" else "BD_EXIT")

    if realloc_today and not parts:
        parts.append("REALLOC")

    return "+".join(parts) if parts else "HOLD"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    wf_equity_eq:     pd.Series,
    wf_trades_eq:     pd.DataFrame,
    wf_results_eq:    pd.DataFrame,
    wf_equity_bd:     pd.Series,
    wf_trades_bd:     pd.DataFrame,
    wf_results_bd:    pd.DataFrame,
    portfolio_equity: pd.Series,
    portfolio_metrics: dict,
    weights_series:   pd.Series,
    reallocation_log: list,
    bh_eq_metrics:    dict,
    bh_bd_metrics:    dict,
    WIG:              pd.DataFrame,
    TBSP:             pd.DataFrame,
    run_date:         dt.date,
) -> dict:
    snap = {"run_date": run_date.isoformat()}

    # Equity asset state
    sig_eq  = _get_signal_from_trades(wf_trades_eq)
    pos_eq  = _get_open_position(wf_trades_eq)
    par_eq  = _get_active_window_params(wf_results_eq)
    ma_eq   = _compute_ma_filter_state(WIG, par_eq.get("fast", 50), par_eq.get("slow", 200)) if par_eq else {}

    snap["signal_equity"] = sig_eq
    snap["params_equity"] = {k: v for k, v in par_eq.items() if k not in ("TestStart", "TestEnd")}
    snap["ma_state_equity"] = ma_eq

    if pos_eq:
        prices_eq  = WIG["Zamkniecie"].dropna()
        entry_px   = float(pos_eq["EntryPrice"])
        today_px   = float(prices_eq.iloc[-1])
        in_trade   = prices_eq.loc[prices_eq.index >= pd.Timestamp(pos_eq["EntryDate"])]
        peak_px    = float(in_trade.max()) if not in_trade.empty else today_px
        X          = par_eq.get("X", 0.10)
        sl         = par_eq.get("stop_loss", 0.10)
        trail_stop = round(peak_px * (1 - X), 2)
        abs_stop   = round(entry_px * (1 - sl), 2)
        binding    = max(trail_stop, abs_stop)
        snap["equity_position"] = {
            "entry_date":     pd.Timestamp(pos_eq["EntryDate"]).date().isoformat(),
            "entry_price":    round(entry_px, 2),
            "today_price":    round(today_px, 2),
            "days_in_trade":  int(pos_eq.get("Days", 0)),
            "unrealised_pct": round(float(pos_eq["Return"]) * 100, 2),
            "peak_price":     round(peak_px, 2),
            "trail_stop":     trail_stop,
            "abs_stop":       abs_stop,
            "binding_stop":   binding,
            "stop_gap_pct":   round((binding - today_px) / today_px * 100, 2),
        }
    else:
        snap["equity_position"] = None

    # Bond asset state
    sig_bd  = _get_signal_from_trades(wf_trades_bd)
    pos_bd  = _get_open_position(wf_trades_bd)
    par_bd  = _get_active_window_params(wf_results_bd)
    ma_bd   = _compute_ma_filter_state(TBSP, par_bd.get("fast", 100), par_bd.get("slow", 300)) if par_bd else {}

    snap["signal_bond"] = sig_bd
    snap["params_bond"] = {k: v for k, v in par_bd.items() if k not in ("TestStart", "TestEnd")}
    snap["ma_state_bond"] = ma_bd

    if pos_bd:
        prices_bd  = TBSP["Zamkniecie"].dropna()
        entry_px   = float(pos_bd["EntryPrice"])
        today_px   = float(prices_bd.iloc[-1])
        in_trade   = prices_bd.loc[prices_bd.index >= pd.Timestamp(pos_bd["EntryDate"])]
        peak_px    = float(in_trade.max()) if not in_trade.empty else today_px
        X          = par_bd.get("X", 0.08)
        sl         = par_bd.get("stop_loss", 0.02)
        trail_stop = round(peak_px * (1 - X), 2)
        abs_stop   = round(entry_px * (1 - sl), 2)
        binding    = max(trail_stop, abs_stop)
        snap["bond_position"] = {
            "entry_date":     pd.Timestamp(pos_bd["EntryDate"]).date().isoformat(),
            "entry_price":    round(entry_px, 2),
            "today_price":    round(today_px, 2),
            "days_in_trade":  int(pos_bd.get("Days", 0)),
            "unrealised_pct": round(float(pos_bd["Return"]) * 100, 2),
            "peak_price":     round(peak_px, 2),
            "trail_stop":     trail_stop,
            "abs_stop":       abs_stop,
            "binding_stop":   binding,
            "stop_gap_pct":   round((binding - today_px) / today_px * 100, 2),
        }
    else:
        snap["bond_position"] = None

    snap["weights"]    = _get_current_weights(weights_series)
    snap["portfolio_metrics"] = {k: round(float(v), 4) for k, v in (portfolio_metrics or {}).items()}
    snap["bh_equity_metrics"] = {k: round(float(v), 4) for k, v in (bh_eq_metrics    or {}).items()}
    snap["bh_bond_metrics"]   = {k: round(float(v), 4) for k, v in (bh_bd_metrics    or {}).items()}
    snap["realloc_today"]     = _realloc_today(reallocation_log, run_date)

    if reallocation_log:
        last_r = reallocation_log[-1]
        snap["last_realloc"] = {
            "date":          str(pd.Timestamp(last_r["Date"]).date()),
            "equity_before": round(float(last_r["equity_before"]), 2),
            "bond_before":   round(float(last_r["bond_before"]),   2),
            "mmf_before":    round(float(last_r["mmf_before"]),    2),
            "equity_after":  round(float(last_r["equity_after"]),  2),
            "bond_after":    round(float(last_r["bond_after"]),    2),
            "mmf_after":     round(float(last_r["mmf_after"]),     2),
            "reason":        last_r.get("reason", ""),
        }
    else:
        snap["last_realloc"] = None
    snap['current_regime_adx'] = get_current_adx_regime(WIG)
    return snap


# ---------------------------------------------------------------------------
# Status text
# ---------------------------------------------------------------------------

def _build_status_text(snap: dict, action: str) -> str:
    sep  = "=" * 65
    sep2 = "-" * 65
    w  = snap["weights"]
    pm = snap["portfolio_metrics"]

    def _pct(v):
        return f"{v*100:.2f}%" if v is not None else "N/A"

    lines = [
        sep,
        f"  MULTI-ASSET STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Action:         {action}",
        f"  Equity signal:  {snap['signal_equity']}",
        f"  Bond signal:    {snap['signal_bond']}",
        f"  Rynek (ADX):      {snap.get('current_regime_adx', 'N/A').upper()}",
        sep2,
        "  CURRENT ALLOCATION",
        f"  Equity (WIG):   {w['equity']*100:.0f}%" if w['equity'] is not None else "  Equity (WIG):   N/A",
        f"  Bond (TBSP):    {w['bond']*100:.0f}%"   if w['bond']   is not None else "  Bond (TBSP):    N/A",
        f"  MMF:            {w['mmf']*100:.0f}%"    if w['mmf']    is not None else "  MMF:            N/A",
        sep2,
    ]

    lines.append("  EQUITY POSITION (WIG)")
    ep     = snap.get("equity_position")
    par_eq = snap.get("params_equity", {})
    if ep:
        lines += [
            f"  Entry date:     {ep['entry_date']}",
            f"  Entry price:    {ep['entry_price']}",
            f"  Today price:    {ep['today_price']}",
            f"  Days in trade:  {ep['days_in_trade']}",
            f"  Unrealised:     {ep['unrealised_pct']:+.2f}%",
            f"  Trail stop:     {ep['trail_stop']}  (peak {ep['peak_price']} × (1-{par_eq.get('X',0):.0%}))",
            f"  Abs stop:       {ep['abs_stop']}  (entry × (1-{par_eq.get('stop_loss',0):.0%}))",
            f"  Binding stop:   {ep['binding_stop']}  (gap: {ep['stop_gap_pct']:+.1f}%)",
        ]
    else:
        lines.append("  No open equity position.")
    ma_eq  = snap.get("ma_state_equity", {})
    icon   = "(+)" if ma_eq.get("filter_on") else "(-)"
    lines.append(
        f"  MA filter: {icon}  fast({par_eq.get('fast','?')})={ma_eq.get('fast_ma')}  "
        f"slow({par_eq.get('slow','?')})={ma_eq.get('slow_ma')}  "
        f"gap={ma_eq.get('gap_pct')}%"
    )
    lines.append(sep2)

    lines.append("  BOND POSITION (TBSP)")
    bp     = snap.get("bond_position")
    par_bd = snap.get("params_bond", {})
    if bp:
        lines += [
            f"  Entry date:     {bp['entry_date']}",
            f"  Entry price:    {bp['entry_price']}",
            f"  Today price:    {bp['today_price']}",
            f"  Days in trade:  {bp['days_in_trade']}",
            f"  Unrealised:     {bp['unrealised_pct']:+.2f}%",
            f"  Trail stop:     {bp['trail_stop']}  (peak {bp['peak_price']} × (1-{par_bd.get('X',0):.0%}))",
            f"  Abs stop:       {bp['abs_stop']}  (entry × (1-{par_bd.get('stop_loss',0):.0%}))",
            f"  Binding stop:   {bp['binding_stop']}  (gap: {bp['stop_gap_pct']:+.1f}%)",
        ]
    else:
        lines.append("  No open bond position.")
    ma_bd  = snap.get("ma_state_bond", {})
    icon2  = "+" if ma_bd.get("filter_on") else "(-)"
    lines.append(
        f"  MA filter: {icon2}  fast({par_bd.get('fast','?')})={ma_bd.get('fast_ma')}  "
        f"slow({par_bd.get('slow','?')})={ma_bd.get('slow_ma')}  "
        f"gap={ma_bd.get('gap_pct')}%"
    )
    lines.append(sep2)

    lines += [
        "  PORTFOLIO OOS METRICS",
        f"  CAGR:    {_pct(pm.get('CAGR'))}  |  Sharpe:  {pm.get('Sharpe', 'N/A'):.2f}  |  "
        f"MaxDD: {_pct(pm.get('MaxDD'))}  |  CalMAR: {pm.get('CalMAR', 'N/A'):.2f}",
        sep2,
    ]

    lr = snap.get("last_realloc")
    if lr:
        lines += [
            "  LAST REALLOCATION",
            f"  Date:   {lr['date']}  ({lr['reason']})",
            f"  Before: EQ {lr['equity_before']*100:.0f}%  BD {lr['bond_before']*100:.0f}%  MMF {lr['mmf_before']*100:.0f}%",
            f"  After:  EQ {lr['equity_after']*100:.0f}%  BD {lr['bond_after']*100:.0f}%  MMF {lr['mmf_after']*100:.0f}%",
        ]

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row
# ---------------------------------------------------------------------------

def _build_log_row(snap: dict, action: str) -> dict:
    w  = snap["weights"]
    pm = snap["portfolio_metrics"]
    ep = snap.get("equity_position") or {}
    bp = snap.get("bond_position")   or {}
    return {
        "date":              snap["run_date"],
        "action":            action,
        "signal_equity":     snap["signal_equity"],
        "signal_bond":       snap["signal_bond"],
        "regime_adx":        snap.get("current_regime_adx"),
        "w_equity":          w.get("equity"),
        "w_bond":            w.get("bond"),
        "w_mmf":             w.get("mmf"),
        "realloc_today":     snap.get("realloc_today", False),
        "eq_entry_date":     ep.get("entry_date"),
        "eq_entry_price":    ep.get("entry_price"),
        "eq_unrealised_pct": ep.get("unrealised_pct"),
        "eq_binding_stop":   ep.get("binding_stop"),
        "eq_stop_gap_pct":   ep.get("stop_gap_pct"),
        "bd_entry_date":     bp.get("entry_date"),
        "bd_entry_price":    bp.get("entry_price"),
        "bd_unrealised_pct": bp.get("unrealised_pct"),
        "bd_binding_stop":   bp.get("binding_stop"),
        "bd_stop_gap_pct":   bp.get("stop_gap_pct"),
        "portfolio_cagr":    pm.get("CAGR"),
        "portfolio_maxdd":   pm.get("MaxDD"),
        "portfolio_calmar":  pm.get("CalMAR"),
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _build_chart(
    portfolio_equity: pd.Series,
    sig_eq_oos:       pd.Series,
    sig_bd_oos:       pd.Series,
    reallocation_log: list,
    bh_eq_equity:     pd.Series,
    bh_bd_equity:     pd.Series,
    chart_path:       Path,
    action:           str,
    run_date:         dt.date,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Multi-Asset Strategy — {run_date}  [{action}]",
                 fontsize=12, fontweight="bold")

    oos_start = portfolio_equity.index.min()

    ax = axes[0]
    portfolio_equity.plot(ax=ax, label="Portfolio", color="steelblue", linewidth=1.8)
    if bh_eq_equity is not None:
        bh_eq = bh_eq_equity.loc[bh_eq_equity.index >= oos_start] / bh_eq_equity.loc[bh_eq_equity.index >= oos_start].iloc[0]
        bh_eq.plot(ax=ax, label="B&H WIG",  color="darkorange", linewidth=1, linestyle="--", alpha=0.7)
    if bh_bd_equity is not None:
        bh_bd = bh_bd_equity.loc[bh_bd_equity.index >= oos_start] / bh_bd_equity.loc[bh_bd_equity.index >= oos_start].iloc[0]
        bh_bd.plot(ax=ax, label="B&H TBSP", color="seagreen",   linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Portfolio Equity (OOS)")
    ax.set_ylabel("Equity (rebased to 1.0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    dd_port = portfolio_equity / portfolio_equity.cummax() - 1
    dd_port.plot(ax=ax, label="Portfolio", color="steelblue", linewidth=1.5)
    if bh_eq_equity is not None:
        bh_eq = bh_eq_equity.loc[bh_eq_equity.index >= oos_start] / bh_eq_equity.loc[bh_eq_equity.index >= oos_start].iloc[0]
        dd_eq  = bh_eq / bh_eq.cummax() - 1
        dd_eq.plot(ax=ax, label="B&H WIG", color="darkorange", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    if sig_eq_oos is not None and not sig_eq_oos.empty:
        sig_eq_oos.plot(ax=ax, label="Equity signal", color="darkorange", linewidth=0.9, alpha=0.8)
    if sig_bd_oos is not None and not sig_bd_oos.empty:
        sig_bd_oos.plot(ax=ax, label="Bond signal",   color="seagreen",   linewidth=0.9, alpha=0.8)
    if reallocation_log:
        rdates = [pd.Timestamp(r["Date"]) for r in reallocation_log]
        ax.vlines(rdates, ymin=0, ymax=1.05, color="grey",
                  linewidth=0.5, alpha=0.5, label="Reallocation")
    ax.set_title("Asset Signals and Reallocation Events")
    ax.set_ylabel("Signal (0=off, 1=on)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, dpi=72, bbox_inches="tight")
    plt.close(fig)
    buf.close()
    atomic_write_bytes(chart_path, open(buf.name, "rb").read())
    os.unlink(buf.name)
    logging.info("multiasset_daily_output: chart saved to %s", chart_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_daily_outputs(
    wf_equity_eq:      pd.Series,
    wf_trades_eq:      pd.DataFrame,
    wf_results_eq:     pd.DataFrame,
    wf_equity_bd:      pd.Series,
    wf_trades_bd:      pd.DataFrame,
    wf_results_bd:     pd.DataFrame,
    portfolio_equity:  pd.Series,
    portfolio_metrics: dict,
    weights_series:    pd.Series,
    reallocation_log:  list,
    bh_eq_equity:      pd.Series,
    bh_eq_metrics:     dict,
    bh_bd_equity:      pd.Series,
    bh_bd_metrics:     dict,
    WIG:               pd.DataFrame,
    TBSP:              pd.DataFrame,
    sig_eq_oos:        pd.Series,
    sig_bd_oos:        pd.Series,
    output_dir:        str       = "outputs",
    asset_name:        str       = "PENSION", 
    run_date:          dt.date | None = None,
    gdrive_folder_id:   str | None = None,
    gdrive_credentials: str | None = None,
) -> dict:
    """
    Build and write all daily output artefacts for the multi-asset strategy.

    Returns dict with keys: action, signal_equity, signal_bond, status_text,
                            log_row, chart_path, snapshot_path, log_path
    """
    if run_date is None:
        run_date = dt.date.today()

    out_dir       = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [ZMIANA] Dynamiczne nazwy plików
    prefix = asset_name.lower()
    logfile_name  = f"{prefix}_signal_log.csv"
    log_path      = out_dir / logfile_name
    status_path   = out_dir / f"{prefix}_signal_status.txt"
    chart_path    = out_dir / f"{prefix}_equity_chart.png"
    snapshot_path = out_dir / f"{prefix}_signal_snapshot.json"

    if gdrive_folder_id and gdrive_credentials:
        fetch_file_from_drive(log_path, gdrive_folder_id, logfile_name, gdrive_credentials)
    else:
        logging.info("multiasset_daily_output: skipping Drive log fetch.")

    snap = _build_snapshot(
        wf_equity_eq     = wf_equity_eq,
        wf_trades_eq     = wf_trades_eq,
        wf_results_eq    = wf_results_eq,
        wf_equity_bd     = wf_equity_bd,
        wf_trades_bd     = wf_trades_bd,
        wf_results_bd    = wf_results_bd,
        portfolio_equity = portfolio_equity,
        portfolio_metrics= portfolio_metrics,
        weights_series   = weights_series,
        reallocation_log = reallocation_log,
        bh_eq_metrics    = bh_eq_metrics,
        bh_bd_metrics    = bh_bd_metrics,
        WIG              = WIG,
        TBSP             = TBSP,
        run_date         = run_date,
    )

    prev_log = load_existing_log(log_path)
    action   = _determine_action(
        prev_log,
        snap["signal_equity"],
        snap["signal_bond"],
        snap.get("realloc_today", False),
    )
    snap["action"] = action

    status_text = _build_status_text(snap, action)
    logging.info("\n%s", status_text)
    atomic_write(status_path, status_text)
    logging.info("multiasset_daily_output: status written to %s", status_path)

    log_row = _build_log_row(snap, action)
    append_log_row(log_path, log_row)
    logging.info("multiasset_daily_output: log updated at %s", log_path)

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(snapshot_path, json.dumps(snap_clean, indent=2, default=str))
    logging.info("multiasset_daily_output: snapshot written to %s", snapshot_path)

    _build_chart(
        portfolio_equity = portfolio_equity,
        sig_eq_oos       = sig_eq_oos,
        sig_bd_oos       = sig_bd_oos,
        reallocation_log = reallocation_log,
        bh_eq_equity     = bh_eq_equity,
        bh_bd_equity     = bh_bd_equity,
        chart_path       = chart_path,
        action           = action,
        run_date         = run_date,
    )
    # --- NOWA LOGIKA: WYSYŁANIE NA GOOGLE DRIVE ---
    if gdrive_folder_id and gdrive_credentials:
        logging.info("Uploading artefacts to Google Drive...")
        try:
            from moj_system.data.gdrive import GDriveClient
            client = GDriveClient(credentials_path=gdrive_credentials)
            
            if client.service:
                # Wysyłamy wszystkie 4 wygenerowane pliki
                files_to_upload = [log_path, status_path, chart_path, snapshot_path]
                for file_path in files_to_upload:
                    if file_path.exists():
                        client.upload_csv(gdrive_folder_id, str(file_path), file_path.name)
                        # Uwaga: metoda nazywa się 'upload_csv', ale w kodzie GDriveClient używa 
                        # ogólnego mimetypu 'text/csv' lub go ignoruje, więc prześle poprawnie też .txt i .png. 
                        # Dla pewności, można w przyszłości zaktualizować metodę w GDriveClient.
                logging.info("Successfully uploaded all daily artefacts to Google Drive.")
            else:
                logging.warning("Drive service unavailable. Artefacts saved locally only.")
        except Exception as e:
            logging.error(f"Failed to upload artefacts to Drive: {e}")
    else:
        logging.info("No GDrive credentials provided. Artefacts saved locally only.")
    # ---------------------------------------------
    return {
        "action":        action,
        "signal_equity": snap["signal_equity"],
        "signal_bond":   snap["signal_bond"],
        "status_text":   status_text,
        "log_row":       log_row,
        "chart_path":    chart_path,
        "snapshot_path": snapshot_path,
        "log_path":      log_path,
    }
