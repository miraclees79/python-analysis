"""
daily_output.py
===============
Daily operational output module for the WIG20TR single-asset trend-following
strategy (v0.2) and all generic single-asset daily runners.

Called at the end of the main runfile after walk_forward has completed.
Produces four output artefacts:

  1. STATUS BLOCK   — human-readable text summary of today's signal state,
                      printed to stdout and written to signal_status.txt.
                      Designed to be embedded verbatim in an email body.

  2. SIGNAL LOG     — {logfile_name} — one row per run date, appended.
                      Accumulates the living history of daily signal states.
                      Suitable for upload to Google Drive as an audit trail.

  3. CHART          — equity_chart.png — OOS equity curve vs B&H, with the
                      current open trade highlighted and MA filter state shown
                      in a subplot.  Overwritten on each daily run.

  4. SNAPSHOT JSON  — signal_snapshot.json — today's state in machine-readable
                      form.  Overwritten daily.  Suitable for a Google Sheet
                      IMPORTDATA formula or downstream automation.

Infrastructure (atomic writes, Drive pre-fetch, log append) is provided by
daily_output_base.py and shared with multiasset_daily_output.py and
global_equity_daily_output.py.

PUBLIC API
----------
    from daily_output import build_daily_outputs

    outputs = build_daily_outputs(
        wf_equity   = wf_equity,
        wf_trades   = wf_trades,
        wf_metrics  = wf_metrics,
        wf_results  = wf_results,
        bh_equity   = bh_equity,
        bh_metrics  = bh_metrics,
        df          = df,
        output_dir  = "outputs",
        price_col   = "Zamkniecie",
        logfile_name = "signal_log.csv",
        asset_name  = "WIG20TR",
        gdrive_folder_id   = os.getenv("GDRIVE_FOLDER_ID"),
        gdrive_credentials = "/tmp/credentials.json",
    )
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def _get_active_window_params(wf_results: pd.DataFrame) -> dict:
    """Return the parameter dict from the last walk-forward window."""
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

    params = {
        "filter_mode": last.get("filter_mode", "ma"),
        "X":           _safe_float("X",         0.10),
        "Y":           _safe_float("Y",         0.10),
        "fast":        _safe_int("fast",          50),
        "slow":        _safe_int("slow",         200),
        "stop_loss":   _safe_float("stop_loss",  0.10),
        "TestStart":   pd.Timestamp(last["TestStart"]),
        "TestEnd":     pd.Timestamp(last["TestEnd"]),
    }
    tv = last.get("target_vol")
    if tv is not None and pd.notna(tv) and str(tv).strip() not in ("", "N/A", "nan"):
        params["target_vol"] = float(tv)
    ml = last.get("mom_lookback")
    if ml is not None and pd.notna(ml) and str(ml).strip() not in ("", "N/A", "nan"):
        params["mom_lookback"] = int(float(ml))
    else:
        params["mom_lookback"] = 252
    return params


def _extract_open_position(wf_trades: pd.DataFrame) -> dict | None:
    """Return the most recent CARRY record as a dict, or None if flat."""
    if wf_trades is None or wf_trades.empty:
        return None
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    if carry.empty:
        return None
    return carry.iloc[-1].to_dict()


def _compute_ma_filter_state(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    price_col: str = "Zamkniecie",
) -> dict:
    """Recompute fast/slow MAs from the full price series."""
    prices = df[price_col].dropna()
    if len(prices) < slow:
        return {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}
    fast_ma   = float(prices.rolling(fast).mean().iloc[-1])
    slow_ma   = float(prices.rolling(slow).mean().iloc[-1])
    gap_pct   = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
    filter_on = fast_ma > slow_ma
    return {
        "fast_ma":   round(fast_ma, 2),
        "slow_ma":   round(slow_ma, 2),
        "filter_on": filter_on,
        "gap_pct":   round(gap_pct * 100, 2),
    }


def _compute_mom_filter_state(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    price_col: str = "Zamkniecie",
) -> dict:
    """Recompute momentum filter state from the full price series."""
    prices = df[price_col].dropna()
    if len(prices) < lookback:
        return {"mom_value": None, "filter_on": None}
    mom_series = prices.shift(skip) / prices.shift(lookback) - 1
    mom_value  = float(mom_series.iloc[-1]) if not pd.isna(mom_series.iloc[-1]) else None
    filter_on  = (mom_value > 0) if mom_value is not None else None
    return {
        "mom_value": round(mom_value * 100, 2) if mom_value is not None else None,
        "filter_on": filter_on,
    }


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------

def _determine_action(
    prev_signal_log: pd.DataFrame | None,
    signal_today: str,
) -> str:
    """Compare today's signal to yesterday's. Returns ENTER, EXIT, or HOLD."""
    if prev_signal_log is None or prev_signal_log.empty:
        return "ENTER" if signal_today == "IN" else "HOLD"
    prev_signal = str(prev_signal_log["signal"].iloc[-1]).upper()
    if prev_signal == signal_today:
        return "HOLD"
    return "ENTER" if signal_today == "IN" else "EXIT"


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    wf_equity:  pd.Series,
    wf_trades:  pd.DataFrame,
    wf_metrics: dict,
    wf_results: pd.DataFrame,
    bh_metrics: dict,
    df:         pd.DataFrame,
    price_col:  str,
    run_date:   dt.date,
) -> dict:
    """Assemble the full snapshot dict from walk-forward outputs."""
    snap = {"run_date": run_date.isoformat()}

    params      = _get_active_window_params(wf_results)
    snap["params"] = params

    prices      = df[price_col].dropna()
    today_price = float(prices.iloc[-1]) if len(prices) > 0 else None
    snap["today_price"] = today_price

    open_pos = _extract_open_position(wf_trades)
    snap["open_position"] = open_pos

    if open_pos is not None:
        snap["signal"]    = "IN"
        entry_price       = float(open_pos["EntryPrice"])
        entry_date        = pd.Timestamp(open_pos["EntryDate"]).date().isoformat()
        unreal_ret        = float(open_pos["Return"])
        days_in           = int(open_pos.get("Days", 0))
        peak_price        = today_price

        if wf_equity is not None and not wf_equity.empty:
            trade_start     = pd.Timestamp(open_pos["EntryDate"])
            price_in_trade  = prices.loc[prices.index >= trade_start]
            if not price_in_trade.empty:
                peak_price = float(price_in_trade.max())

        X             = params.get("X", 0.10)
        stop_loss     = params.get("stop_loss", 0.10)
        trail_stop    = round(peak_price * (1 - X), 2)
        abs_stop      = round(entry_price * (1 - stop_loss), 2)
        binding_stop  = max(trail_stop, abs_stop)

        snap["entry_date"]     = entry_date
        snap["entry_price"]    = round(entry_price, 2)
        snap["days_in_trade"]  = days_in
        snap["unrealised_pct"] = round(unreal_ret * 100, 2)
        snap["peak_price"]     = round(peak_price, 2)
        snap["trail_stop"]     = trail_stop
        snap["abs_stop"]       = abs_stop
        snap["binding_stop"]   = binding_stop
        if today_price:
            snap["stop_gap_pct"] = round(
                (binding_stop - today_price) / today_price * 100, 2
            )
    else:
        snap["signal"]         = "OUT"
        snap["entry_date"]     = None
        snap["entry_price"]    = None
        snap["days_in_trade"]  = None
        snap["unrealised_pct"] = None
        snap["peak_price"]     = None
        snap["trail_stop"]     = None
        snap["abs_stop"]       = None
        snap["binding_stop"]   = None
        snap["stop_gap_pct"]   = None

    if params:
        snap["ma_state"]  = _compute_ma_filter_state(df, params["fast"], params["slow"], price_col)
        snap["mom_state"] = _compute_mom_filter_state(
            df, lookback=params.get("mom_lookback", 252), skip=21, price_col=price_col
        )
    else:
        snap["ma_state"]  = {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}
        snap["mom_state"] = {"mom_value": None, "filter_on": None}

    filter_mode = params.get("filter_mode", "ma") if params else "ma"
    snap["active_filter_on"] = (
        snap["mom_state"]["filter_on"] if filter_mode == "mom"
        else snap["ma_state"]["filter_on"]
    )

    snap["strategy_metrics"] = {k: round(float(v), 4) for k, v in (wf_metrics or {}).items()}
    snap["bh_metrics"]       = {k: round(float(v), 4) for k, v in (bh_metrics  or {}).items()}

    if wf_results is not None and not wf_results.empty:
        last_w = wf_results.iloc[-1]
        snap["active_window"] = {
            "train_start": str(last_w.get("TrainStart", "")),
            "test_start":  str(last_w.get("TestStart", "")),
            "test_end":    str(last_w.get("TestEnd", "")),
        }
    else:
        snap["active_window"] = {}
    snap['current_regime_adx'] = get_current_adx_regime(df)
    return snap


# ---------------------------------------------------------------------------
# Status text (artefact 1)
# ---------------------------------------------------------------------------

def _build_status_text(snap: dict, action: str, asset_name: str) -> str:
    """Render the human-readable status block from a snapshot dict."""
    sep  = "=" * 60
    sep2 = "-" * 60
    lines = [
        sep,
        f"  {asset_name} STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Signal:      {snap['signal']}",
        f"  Action:      {action}",
        f"  Price today: {snap['today_price']}",
        f"  Rynek (ADX):   {snap.get('current_regime_adx', 'N/A').upper()}",
        sep2,
    ]

    fmode     = snap["params"].get("filter_mode", "ma")
    ma        = snap.get("ma_state", {})
    mom       = snap.get("mom_state", {})
    active_on = snap.get("active_filter_on")
    act_icon  = "✓" if active_on else "✗"
    ma_icon   = "✓" if ma.get("filter_on") else "✗"
    mom_icon  = "✓" if mom.get("filter_on") else "✗"
    active_lbl = "(ACTIVE)" if fmode == "ma"  else ""
    mom_lbl    = "(ACTIVE)" if fmode == "mom" else ""

    if snap["signal"] == "IN":
        lines += [
            f"  Entry date:    {snap['entry_date']}",
            f"  Entry price:   {snap['entry_price']}",
            f"  Days in trade: {snap['days_in_trade']}",
            f"  Unrealised:    {snap['unrealised_pct']:+.2f}%",
            sep2,
            "  STOP LEVELS",
            f"  Trail stop:    {snap['trail_stop']}  "
            f"(peak {snap['peak_price']} × (1 - {snap['params'].get('X', '?'):.0%}))",
            f"  Abs stop:      {snap['abs_stop']}  "
            f"(entry × (1 - {snap['params'].get('stop_loss', '?'):.0%}))",
            f"  Binding stop:  {snap['binding_stop']}  "
            f"(gap from today: {snap.get('stop_gap_pct', '?'):+.1f}%)",
            sep2,
            f"  FILTERS  [active filter: {fmode.upper()}]  {act_icon}",
            f"  MA  cross {active_lbl}  fast({snap['params'].get('fast', '?')})={ma.get('fast_ma')}  "
            f"slow({snap['params'].get('slow', '?')})={ma.get('slow_ma')}  "
            f"gap={ma.get('gap_pct', 0):+.2f}%  {ma_icon}",
            f"  MOM {mom_lbl}  12m-1m={mom.get('mom_value', 0):+.2f}%  {mom_icon}",
        ]
    else:
        lines += [
            "  Position:      FLAT (cash / MMF)",
            sep2,
            f"  FILTERS  [active filter: {fmode.upper()}]  entry requires: {act_icon}",
            f"  MA  cross {active_lbl}  fast({snap['params'].get('fast', '?')})={ma.get('fast_ma')}  "
            f"slow({snap['params'].get('slow', '?')})={ma.get('slow_ma')}  "
            f"gap={ma.get('gap_pct', 0):+.2f}%  {ma_icon}",
            f"  MOM {mom_lbl}  12m-1m={mom.get('mom_value', 0):+.2f}%  {mom_icon}",
            f"  Entry breakout threshold: +{snap['params'].get('Y', '?'):.0%} from trough",
        ]

    sm = snap.get("strategy_metrics", {})
    bh = snap.get("bh_metrics", {})
    aw = snap.get("active_window", {})

    lines += [
        sep2,
        "  OOS PERFORMANCE (from WF start)",
        f"  Strategy:  CAGR {sm.get('CAGR', 0)*100:+.2f}%  "
        f"Sharpe {sm.get('Sharpe', 0):.2f}  "
        f"MaxDD {sm.get('MaxDD', 0)*100:.1f}%  "
        f"CalMAR {sm.get('CalMAR', 0):.2f}",
        f"  B&H:       CAGR {bh.get('CAGR', 0)*100:+.2f}%  "
        f"Sharpe {bh.get('Sharpe', 0):.2f}  "
        f"MaxDD {bh.get('MaxDD', 0)*100:.1f}%  "
        f"CalMAR {bh.get('CalMAR', 0):.2f}",
        sep2,
        "  ACTIVE WINDOW",
        f"  Train start: {aw.get('train_start', 'n/a')}",
        f"  Test start:  {aw.get('test_start', 'n/a')}",
        f"  Test end:    {aw.get('test_end', 'n/a')}",
        f"  Params:      X={snap['params'].get('X', '?'):.0%}  "
        f"Y={snap['params'].get('Y', '?'):.0%}  "
        f"fast={snap['params'].get('fast', '?')}  "
        f"slow={snap['params'].get('slow', '?')}  "
        f"sl={snap['params'].get('stop_loss', '?'):.0%}  "
        f"mode={snap['params'].get('filter_mode', '?')}",
        sep,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row (artefact 2)
# ---------------------------------------------------------------------------

def _build_log_row(snap: dict, action: str) -> dict:
    """Return a flat dict suitable as a single CSV row."""
    ma = snap.get("ma_state", {})
    sm = snap.get("strategy_metrics", {})
    return {
        "date":            snap["run_date"],
        "signal":          snap["signal"],
        "action":          action,
        "price":           snap.get("today_price"),
        "entry_date":      snap.get("entry_date"),
        "entry_price":     snap.get("entry_price"),
        "days_in_trade":   snap.get("days_in_trade"),
        "unrealised_pct":  snap.get("unrealised_pct"),
        "binding_stop":    snap.get("binding_stop"),
        "stop_gap_pct":    snap.get("stop_gap_pct"),
        "trail_stop":      snap.get("trail_stop"),
        "abs_stop":        snap.get("abs_stop"),
        "ma_fast":         ma.get("fast_ma"),
        "ma_slow":         ma.get("slow_ma"),
        "ma_filter_on":    ma.get("filter_on"),
        "ma_gap_pct":      ma.get("gap_pct"),
        "mom_value_pct":   snap.get("mom_state", {}).get("mom_value"),
        "mom_filter_on":   snap.get("mom_state", {}).get("filter_on"),
        "active_filter":   snap.get("params", {}).get("filter_mode", "ma"),
        "active_filter_on": snap.get("active_filter_on"),
        "regime_adx": snap.get("current_regime_adx"),
        "cagr":            round(sm.get("CAGR",   0) * 100, 4),
        "maxdd":           round(sm.get("MaxDD",  0) * 100, 4),
        "calmar":          round(sm.get("CalMAR", 0),       4),
        "sharpe":          round(sm.get("Sharpe", 0),       4),
    }


def _append_log_row_dedup(log_path: Path, row: dict) -> None:
    """
    Append one row to signal_log.csv, deduplicating on run_date.

    Unlike the base append_log_row() which always appends, this version
    reloads the existing CSV and drops any prior row for the same date
    before appending — preventing duplicate rows on re-runs.
    """
    new_row_df = pd.DataFrame([row])
    
    if log_path.exists():
        try:
            existing = pd.read_csv(log_path)
            existing = existing[existing["date"] != row["date"]]
            combined = pd.concat([existing, new_row_df], ignore_index=True)
        except Exception:
            # Jeśli plik jest uszkodzony, zacznij od nowa
            combined = new_row_df
    else:
        combined = new_row_df
        
    tmp = log_path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    
    # PANCERNE NADPISYWANIE:
    # 1. replace() to wyższa warstwa, os.replace jest najbardziej atomowe
    # 2. Jeżeli plik path istnieje, os.replace go nadpisze bez błędu.
    if os.path.exists(log_path):
        os.replace(tmp, log_path)
    else:
        tmp.rename(log_path)


# ---------------------------------------------------------------------------
# Chart (artefact 3)
# ---------------------------------------------------------------------------

def _build_chart(
    wf_equity:  pd.Series,
    bh_equity:  pd.Series,
    wf_trades:  pd.DataFrame,
    wf_results: pd.DataFrame,
    df:         pd.DataFrame,
    snap:       dict,
    chart_path: Path,
    price_col:  str = "Zamkniecie",
) -> None:
    """3-panel chart: equity curve, drawdown, MA filter state."""
    if wf_equity is None or wf_equity.empty:
        logging.warning("daily_output: wf_equity is empty — skipping chart")
        return

    wf_equity.index = pd.to_datetime(wf_equity.index)

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Panel 1: equity curves
    ax1.plot(wf_equity.index, wf_equity.values,
             label="Strategy (OOS)", color="steelblue", linewidth=2)

    if bh_equity is not None and not bh_equity.empty:
        bh_equity.index = pd.to_datetime(bh_equity.index)
        bh_aligned = bh_equity.loc[
            (bh_equity.index >= wf_equity.index.min()) &
            (bh_equity.index <= wf_equity.index.max())
        ]
        ax1.plot(bh_aligned.index, bh_aligned.values,
                 label="Buy & Hold", color="grey",
                 linewidth=1.5, linestyle="--", alpha=0.8)

    if wf_results is not None and not wf_results.empty:
        for _, row in wf_results.iterrows():
            ax1.axvspan(row["TestStart"], row["TestEnd"], color="grey", alpha=0.05)

    BOUNDARY = {"CARRY", "SAMPLE_END"}
    if wf_trades is not None and not wf_trades.empty:
        closed = wf_trades[~wf_trades["Exit Reason"].isin(BOUNDARY)].copy()
        for _, trade in closed.iterrows():
            color = "green" if trade["Return"] > 0 else "red"
            ax1.axvspan(pd.Timestamp(trade["EntryDate"]),
                        pd.Timestamp(trade["ExitDate"]),
                        color=color, alpha=0.12)

    if snap["signal"] == "IN":
        entry_ts = pd.Timestamp(snap["entry_date"])
        ax1.axvspan(entry_ts, wf_equity.index[-1],
                    color="gold", alpha=0.25, label="Open trade")
        if snap.get("binding_stop") and snap.get("today_price"):
            ax1.axhline(
                y=wf_equity.iloc[-1] * (snap["binding_stop"] / snap["today_price"]),
                color="red", linestyle=":", linewidth=1, alpha=0.7,
                label=f"Binding stop ({snap['binding_stop']})"
            )

    sm = snap.get("strategy_metrics", {})
    bh = snap.get("bh_metrics", {})
    summary = (
        f"Strategy: CAGR {sm.get('CAGR',0)*100:+.1f}%  "
        f"Sharpe {sm.get('Sharpe',0):.2f}  "
        f"MaxDD {sm.get('MaxDD',0)*100:.1f}%\n"
        f"B&H:      CAGR {bh.get('CAGR',0)*100:+.1f}%  "
        f"Sharpe {bh.get('Sharpe',0):.2f}  "
        f"MaxDD {bh.get('MaxDD',0)*100:.1f}%"
    )
    ax1.text(0.01, 0.97, summary, transform=ax1.transAxes, fontsize=8.5,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax1.set_title(
        f"Strategy v0.2 — Signal: {snap['signal']} | "
        f"Action: {snap.get('action', '')} | {snap['run_date']}",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Equity (OOS, normalised)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    # Panel 2: drawdown
    dd_strat = wf_equity / wf_equity.cummax() - 1
    ax2.fill_between(dd_strat.index, dd_strat.values, 0,
                     color="steelblue", alpha=0.4, label="Strategy DD")
    if bh_equity is not None and not bh_equity.empty:
        bh_dd = bh_aligned / bh_aligned.cummax() - 1
        ax2.plot(bh_dd.index, bh_dd.values,
                 color="grey", linewidth=1, linestyle="--", alpha=0.7, label="B&H DD")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(top=0)

    # Panel 3: MA filter ratio
    prices = df[price_col].dropna()
    prices.index = pd.to_datetime(prices.index)
    fast      = snap["params"].get("fast", 50)
    slow      = snap["params"].get("slow", 200)
    oos_start = wf_equity.index.min()

    fast_ma = prices.rolling(fast).mean().shift(1)
    slow_ma = prices.rolling(slow).mean().shift(1)
    ratio   = (fast_ma / slow_ma - 1) * 100
    ratio_oos = ratio.loc[ratio.index >= oos_start]

    ax3.plot(ratio_oos.index, ratio_oos.values,
             color="darkorange", linewidth=1,
             label=f"fast({fast})/slow({slow})-1 [%]")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(ratio_oos.index, ratio_oos.values, 0,
                     where=(ratio_oos > 0), color="green", alpha=0.15, label="Filter ON")
    ax3.fill_between(ratio_oos.index, ratio_oos.values, 0,
                     where=(ratio_oos < 0), color="red",   alpha=0.15, label="Filter OFF")
    ax3.set_ylabel("MA ratio (%)")
    ax3.legend(fontsize=7, loc="lower left")
    ax3.grid(True, alpha=0.25)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.tick_params(axis="x", labelrotation=30)

    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.close()
    atomic_write_bytes(chart_path, open(buf.name, "rb").read())
    os.unlink(buf.name)
    logging.info("daily_output: chart saved to %s", chart_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_daily_outputs(
    wf_equity:   pd.Series,
    wf_trades:   pd.DataFrame,
    wf_metrics:  dict,
    wf_results:  pd.DataFrame,
    bh_equity:   pd.Series,
    bh_metrics:  dict,
    df:          pd.DataFrame,
    output_dir:  str       = "outputs",
    price_col:   str       = "Zamkniecie",
    logfile_name: str      = "signal_log.csv",
    asset_name:  str       = "WIG20TR",
    run_date:    dt.date | None = None,
    gdrive_folder_id:   str | None = None,
    gdrive_credentials: str | None = None,
) -> dict:
    """
    Build and write all daily output artefacts for a single-asset strategy.

    Parameters
    ----------
    wf_equity            : stitched OOS equity curve
    wf_trades            : full trade log including CARRY records
    wf_metrics           : dict of OOS metrics from compute_metrics
    wf_results           : per-window DataFrame from walk_forward
    bh_equity            : buy-and-hold equity curve aligned to OOS period
    bh_metrics           : dict of B&H metrics
    df                   : full price DataFrame (must contain price_col column)
    output_dir           : directory to write output files into
    price_col            : price column name in df
    logfile_name         : Drive-matching log filename (e.g. "wig20tr_signal_log.csv")
    asset_name           : asset label for status text header
    run_date             : override today's date (default: dt.date.today())
    gdrive_folder_id     : Drive folder ID — triggers Drive log pre-fetch
    gdrive_credentials   : path to service account JSON

    Returns
    -------
    dict with keys: action, signal, status_text, log_row,
                    chart_path, snapshot_path, log_path
    """
    
    # Jeśli run_date przyszło jako string (błąd w wywołaniu), przekonwertuj je
    if isinstance(run_date, str):
        try:
            run_date = dt.date.fromisoformat(run_date)
        except ValueError:
            run_date = dt.date.today()
    elif run_date is None:
        run_date = dt.date.today()
    # ----------------
    
    # Teraz run_date na 100% ma metodę .isoformat()
    snap = {"run_date": run_date.isoformat()}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # [ZMIANA] Dynamiczne nazwy plików z użyciem asset_name
    prefix = asset_name.lower()
    logfile_name  = f"{prefix}_signal_log.csv"
    log_path      = out_dir / logfile_name
    status_path   = out_dir / f"{prefix}_signal_status.txt"
    chart_path    = out_dir / f"{prefix}_equity_chart.png"
    snapshot_path = out_dir / f"{prefix}_signal_snapshot.json"

    # [ZMIANA] Używamy logfile_name do ściągania poprzedniego logu
    if gdrive_folder_id and gdrive_credentials:
        fetch_file_from_drive(log_path, gdrive_folder_id, logfile_name, gdrive_credentials)
    else:
        logging.info("daily_output: gdrive credentials not supplied — skipping Drive log fetch.")

    # Build snapshot
    snap = _build_snapshot(
        wf_equity  = wf_equity,
        wf_trades  = wf_trades,
        wf_metrics = wf_metrics,
        wf_results = wf_results,
        bh_metrics = bh_metrics,
        df         = df,
        price_col  = price_col,
        run_date   = run_date,
    )

    # Action
    prev_log = load_existing_log(log_path)
    action   = _determine_action(prev_log, snap["signal"])
    snap["_action"] = action   # underscore copy for chart title
    snap["action"]  = action

    # Status text
    status_text = _build_status_text(snap, action, asset_name)
    logging.info("\n%s", status_text)
    atomic_write(status_path, status_text)
    logging.info("daily_output: status written to %s", status_path)

    # Signal log
    log_row = _build_log_row(snap, action)
    _append_log_row_dedup(log_path, log_row)
    logging.info("daily_output: signal log updated at %s", log_path)

    # Snapshot JSON
    snap_serialisable = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(snapshot_path, json.dumps(snap_serialisable, indent=2, default=str))
    logging.info("daily_output: JSON snapshot written to %s", snapshot_path)

    # Chart
    _build_chart(
        wf_equity  = wf_equity,
        bh_equity  = bh_equity,
        wf_trades  = wf_trades,
        wf_results = wf_results,
        df         = df,
        snap       = snap,
        chart_path = chart_path,
        price_col  = price_col,
    )

    logging.info(
        "daily_output: COMPLETE — signal=%s  action=%s  date=%s",
        snap["signal"], action, run_date,
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
        "signal":        snap["signal"],
        "status_text":   status_text,
        "log_row":       log_row,
        "chart_path":    chart_path,
        "snapshot_path": snapshot_path,
        "log_path":      log_path,
    }