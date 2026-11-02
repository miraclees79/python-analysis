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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from moj_system.core.research import get_current_adx_regime
from moj_system.core.state_analytics import (
    compute_ma_filter_state,
    compute_mom_filter_state,
    extract_open_position,
    get_active_window_params,
)
from moj_system.reporting.output_base import (
    append_log_row,
    atomic_write,
    fetch_file_from_drive,
    load_existing_log,
    resolve_artifact_paths,
    resolve_run_date,
    save_figure_atomically,
    upload_artifacts_to_drive,
)
from moj_system.reporting.report_context import SingleAssetReportContext
from moj_system.reporting.snapshot_builder import (
    SnapshotBuilder,
    compute_stop_levels,
    round_metrics,
    stop_gap_pct,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------


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


class SingleAssetSnapshotBuilder(SnapshotBuilder[SingleAssetReportContext]):
    """Snapshot of a single-asset strategy (flat position fields)."""

    def build_body(self) -> dict[str, Any]:
        ctx = self.context
        snap: dict[str, Any] = {}

        freshness = {}
        if ctx.df is not None and not ctx.df.empty:
            freshness[ctx.asset_name] = str(ctx.df.index.max().date())
        if ctx.wf_equity is not None and not ctx.wf_equity.empty:
            freshness["Strategy"] = str(ctx.wf_equity.index.max().date())
        snap["data_freshness"] = freshness

        params = get_active_window_params(wf_results=ctx.wf_results, include_window_bounds=True)
        snap["params"] = params

        prices = ctx.df[ctx.price_col].dropna()
        today_price = float(prices.iloc[-1]) if len(prices) > 0 else None
        snap["today_price"] = today_price

        open_pos = extract_open_position(wf_trades=ctx.wf_trades)
        snap["open_position"] = open_pos

        if open_pos is not None and today_price is not None:
            snap.update(self._open_position_fields(open_pos=open_pos, prices=prices, today_price=today_price))
        else:
            snap.update(self._flat_position_fields(open_pos=open_pos))

        if params:
            snap["ma_state"] = compute_ma_filter_state(
                df=ctx.df,
                fast=params["fast"],
                slow=params["slow"],
                price_col=ctx.price_col,
            )
            snap["mom_state"] = compute_mom_filter_state(
                df=ctx.df,
                lookback=params.get("mom_lookback", 252),
                skip=21,
                price_col=ctx.price_col,
            )
        else:
            snap["ma_state"] = {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}
            snap["mom_state"] = {"mom_value": None, "filter_on": None}

        filter_mode = params.get("filter_mode", "ma") if params else "ma"
        snap["active_filter_on"] = (
            snap["mom_state"]["filter_on"] if filter_mode == "mom" else snap["ma_state"]["filter_on"]
        )

        snap["strategy_metrics"] = round_metrics(metrics=ctx.wf_metrics)
        snap["bh_metrics"] = round_metrics(metrics=ctx.bh_metrics)

        if ctx.wf_results is not None and not ctx.wf_results.empty:
            last_w = ctx.wf_results.iloc[-1]
            snap["active_window"] = {
                "train_start": str(last_w.get("TrainStart", "")),
                "test_start": str(last_w.get("TestStart", "")),
                "test_end": str(last_w.get("TestEnd", "")),
            }
        else:
            snap["active_window"] = {}
        snap["current_regime_adx"] = get_current_adx_regime(ctx.df)
        return snap

    def _open_position_fields(
        self,
        open_pos: dict[str, Any],
        prices: pd.Series,
        today_price: float,
    ) -> dict[str, Any]:
        ctx = self.context
        params = get_active_window_params(wf_results=ctx.wf_results, include_window_bounds=True)
        entry_price = float(open_pos["EntryPrice"])
        peak_price = today_price

        if ctx.wf_equity is not None and not ctx.wf_equity.empty:
            trade_start = pd.Timestamp(open_pos["EntryDate"])
            price_in_trade = prices.loc[prices.index >= trade_start]
            if not price_in_trade.empty:
                peak_price = float(price_in_trade.max())

        stops = compute_stop_levels(
            price_df=ctx.df,
            params=params,
            entry_price=entry_price,
            peak_price=peak_price,
        )
        fields: dict[str, Any] = {
            "signal": "IN",
            "atr_val_equity": round(number=stops.atr_val, ndigits=3) if stops.atr_val is not None else None,
            "entry_date": pd.Timestamp(open_pos["EntryDate"]).date().isoformat(),
            "entry_price": round(number=entry_price, ndigits=2),
            "days_in_trade": int(open_pos.get("Days", 0)),
            "unrealised_pct": round(number=float(open_pos["Return"]) * 100.0, ndigits=2),
            "peak_price": round(number=peak_price, ndigits=2),
            "trail_stop": stops.trail_stop,
            "abs_stop": stops.abs_stop,
            "binding_stop": stops.binding_stop,
        }
        if today_price:
            fields["stop_gap_pct"] = stop_gap_pct(binding_stop=stops.binding_stop, today_price=today_price)
        return fields

    @staticmethod
    def _flat_position_fields(
        open_pos: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "signal": "IN" if open_pos is not None else "OUT",
            "entry_date": None,
            "entry_price": None,
            "days_in_trade": None,
            "unrealised_pct": None,
            "peak_price": None,
            "trail_stop": None,
            "abs_stop": None,
            "binding_stop": None,
            "stop_gap_pct": None,
        }


# ---------------------------------------------------------------------------
# Status text (artefact 1)
# ---------------------------------------------------------------------------


def _build_status_text(
    snap: dict,
    action: str,
    asset_name: str,
) -> str:
    """Render the human-readable status block from a snapshot dict."""
    sep = "=" * 60
    sep2 = "-" * 60
    lines = [
        sep,
        f"  {asset_name} STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Signal:        {snap['signal']}",
        f"  Action:        {action}",
        f"  Price today:   {snap['today_price']}",
        f"  Rynek (ADX):   {snap.get('current_regime_adx', 'N/A').upper()}",
        f"  Dane z dnia:   {snap.get('data_freshness', {})}",
        sep2,
    ]

    fmode = snap["params"].get("filter_mode", "ma")
    ma = snap.get("ma_state", {})
    mom = snap.get("mom_state", {})
    active_on = snap.get("active_filter_on")
    act_icon = "✓" if active_on else "✗"
    ma_icon = "✓" if ma.get("filter_on") else "✗"
    mom_icon = "✓" if mom.get("filter_on") else "✗"
    active_lbl = "(ACTIVE)" if fmode == "ma" else ""
    mom_lbl = "(ACTIVE)" if fmode == "mom" else ""

    if snap["signal"] == "IN":
        # Bezpieczne pobranie parametrów bez '?' dla formatowań liczbowych
        if snap["params"].get("use_atr_stop"):
            trail_str = (
                f"  Trail stop:    {snap['trail_stop']}  (peak {snap['peak_price']} × "
                f"(1 - {snap['params'].get('stop_param', 0):.2f}[N_atr] × "
                f"{snap.get('atr_val_equity', 0.0):.2f}%[ATR]))"
            )
        else:
            trail_str = (
                f"  Trail stop:    {snap['trail_stop']}  (peak {snap['peak_price']} × "
                f"(1 - {snap['params'].get('stop_param', 0):.0%}))"
            )

        lines += [
            f"  Entry date:    {snap['entry_date']}",
            f"  Entry price:   {snap['entry_price']}",
            f"  Days in trade: {snap['days_in_trade']}",
            f"  Unrealised:    {snap.get('unrealised_pct', 0.0):+.2f}%",
            sep2,
            "  STOP LEVELS",
            trail_str,
            (
                f"  Abs stop:      {snap['abs_stop']}  "
                f"(entry × (1 - {snap['params'].get('stop_loss', 0):.0%}))"
            ),
            (
                f"  Binding stop:  {snap['binding_stop']}  "
                f"(gap from today: {snap.get('stop_gap_pct', 0.0):+.1f}%)"
            ),
            sep2,
            f"  FILTERS  [active filter: {fmode.upper()}]  {act_icon}",
            f"  MA  cross {active_lbl}  fast({snap['params'].get('fast', 'N/A')})={ma.get('fast_ma')}  "
            f"slow({snap['params'].get('slow', 'N/A')})={ma.get('slow_ma')}  "
            f"gap={ma.get('gap_pct', 0.0):+.2f}%  {ma_icon}",
            f"  MOM {mom_lbl}  12m-1m={mom.get('mom_value', 0.0):+.2f}%  {mom_icon}",
        ]
    else:
        lines += [
            "  Position:      FLAT (cash / MMF)",
            sep2,
            f"  FILTERS  [active filter: {fmode.upper()}]  entry requires: {act_icon}",
            f"  MA  cross {active_lbl}  fast({snap['params'].get('fast', 'N/A')})={ma.get('fast_ma')}  "
            f"slow({snap['params'].get('slow', 'N/A')})={ma.get('slow_ma')}  "
            f"gap={ma.get('gap_pct', 0.0):+.2f}%  {ma_icon}",
            f"  MOM {mom_lbl}  12m-1m={mom.get('mom_value', 0.0):+.2f}%  {mom_icon}",
            f"  Entry breakout threshold: +{snap['params'].get('Y', 0):.0%} from trough",
        ]

    sm = snap.get("strategy_metrics", {})
    bh = snap.get("bh_metrics", {})
    aw = snap.get("active_window", {})

    # Wykrycie dynamicznej etykiety dla sekcji Params (X lub N_atr)
    stop_label = snap["params"].get("stop_label", "X")
    stop_val = snap["params"].get("stop_param", 0.0)
    stop_str = f"{stop_val:.2f}" if stop_label == "N_atr" else f"{stop_val:.0%}"

    lines += [
        sep2,
        "  OOS PERFORMANCE (from WF start)",
        f"  Strategy:  CAGR {sm.get('CAGR', 0) * 100:+.2f}%  "
        f"Sharpe {sm.get('Sharpe', 0):.2f}  "
        f"MaxDD {sm.get('MaxDD', 0) * 100:.1f}%  "
        f"CalMAR {sm.get('CalMAR', 0):.2f}",
        f"  B&H:       CAGR {bh.get('CAGR', 0) * 100:+.2f}%  "
        f"Sharpe {bh.get('Sharpe', 0):.2f}  "
        f"MaxDD {bh.get('MaxDD', 0) * 100:.1f}%  "
        f"CalMAR {bh.get('CalMAR', 0):.2f}",
        sep2,
        "  ACTIVE WINDOW",
        f"  Train start: {aw.get('train_start', 'n/a')}",
        f"  Test start:  {aw.get('test_start', 'n/a')}",
        f"  Test end:    {aw.get('test_end', 'n/a')}",
        f"  Params:      {stop_label}={stop_str}  "
        f"Y={snap['params'].get('Y', 0):.0%}  "
        f"fast={snap['params'].get('fast', 'N/A')}  "
        f"slow={snap['params'].get('slow', 'N/A')}  "
        f"sl={snap['params'].get('stop_loss', 0):.0%}  "
        f"mode={snap['params'].get('filter_mode', 'N/A')}",
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
        "date": snap["run_date"],
        "signal": snap["signal"],
        "action": action,
        "price": snap.get("today_price"),
        "entry_date": snap.get("entry_date"),
        "entry_price": snap.get("entry_price"),
        "days_in_trade": snap.get("days_in_trade"),
        "unrealised_pct": snap.get("unrealised_pct"),
        "binding_stop": snap.get("binding_stop"),
        "stop_gap_pct": snap.get("stop_gap_pct"),
        "trail_stop": snap.get("trail_stop"),
        "abs_stop": snap.get("abs_stop"),
        "ma_fast": ma.get("fast_ma"),
        "ma_slow": ma.get("slow_ma"),
        "ma_filter_on": ma.get("filter_on"),
        "ma_gap_pct": ma.get("gap_pct"),
        "mom_value_pct": snap.get("mom_state", {}).get("mom_value"),
        "mom_filter_on": snap.get("mom_state", {}).get("filter_on"),
        "active_filter": snap.get("params", {}).get("filter_mode", "ma"),
        "active_filter_on": snap.get("active_filter_on"),
        "regime_adx": snap.get("current_regime_adx"),
        "cagr": round(sm.get("CAGR", 0) * 100, 4),
        "maxdd": round(sm.get("MaxDD", 0) * 100, 4),
        "calmar": round(sm.get("CalMAR", 0), 4),
        "sharpe": round(sm.get("Sharpe", 0), 4),
    }


# ---------------------------------------------------------------------------
# Chart (artefact 3)
# ---------------------------------------------------------------------------


def _build_chart(
    wf_equity: pd.Series,
    bh_equity: pd.Series,
    wf_trades: pd.DataFrame,
    wf_results: pd.DataFrame,
    df: pd.DataFrame,
    snap: dict,
    chart_path: Path,
    price_col: str = "Zamkniecie",
) -> None:
    """3-panel chart: equity curve, drawdown, MA filter state."""
    if wf_equity is None or wf_equity.empty:
        logging.warning("daily_output: wf_equity is empty — skipping chart")
        return

    wf_equity.index = pd.to_datetime(wf_equity.index)

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Panel 1: equity curves
    ax1.plot(
        wf_equity.index,
        wf_equity.values,
        label="Strategy (OOS)",
        color="steelblue",
        linewidth=2,
    )

    if bh_equity is not None and not bh_equity.empty:
        bh_equity.index = pd.to_datetime(bh_equity.index)
        bh_aligned = bh_equity.loc[
            (bh_equity.index >= wf_equity.index.min()) & (bh_equity.index <= wf_equity.index.max())
        ]
        ax1.plot(
            bh_aligned.index,
            bh_aligned.values,
            label="Buy & Hold",
            color="grey",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )

    if wf_results is not None and not wf_results.empty:
        for row in wf_results.to_dict(orient="records"):
            ax1.axvspan(row["TestStart"], row["TestEnd"], color="grey", alpha=0.05)

    boundary_exits = {"CARRY", "SAMPLE_END"}
    if wf_trades is not None and not wf_trades.empty:
        closed = wf_trades[~wf_trades["Exit Reason"].isin(boundary_exits)].copy()
        for trade in closed.to_dict(orient="records"):
            color = "green" if trade["Return"] > 0 else "red"
            ax1.axvspan(
                pd.Timestamp(trade["EntryDate"]),
                pd.Timestamp(trade["ExitDate"]),
                color=color,
                alpha=0.12,
            )

    if snap["signal"] == "IN":
        entry_ts = pd.Timestamp(snap["entry_date"])
        ax1.axvspan(entry_ts, wf_equity.index[-1], color="gold", alpha=0.25, label="Open trade")
        if snap.get("binding_stop") and snap.get("today_price"):
            ax1.axhline(
                y=wf_equity.iloc[-1] * (snap["binding_stop"] / snap["today_price"]),
                color="red",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Binding stop ({snap['binding_stop']})",
            )

    sm = snap.get("strategy_metrics", {})
    bh = snap.get("bh_metrics", {})
    summary = (
        f"Strategy: CAGR {sm.get('CAGR', 0) * 100:+.1f}%  "
        f"Sharpe {sm.get('Sharpe', 0):.2f}  "
        f"MaxDD {sm.get('MaxDD', 0) * 100:.1f}%\n"
        f"B&H:      CAGR {bh.get('CAGR', 0) * 100:+.1f}%  "
        f"Sharpe {bh.get('Sharpe', 0):.2f}  "
        f"MaxDD {bh.get('MaxDD', 0) * 100:.1f}%"
    )
    ax1.text(
        0.01,
        0.97,
        summary,
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax1.set_title(
        f"Strategy v0.2 — Signal: {snap['signal']} | Action: {snap.get('action', '')} | {snap['run_date']}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Equity (OOS, normalised)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    # Panel 2: drawdown
    dd_strat = wf_equity / wf_equity.cummax() - 1
    ax2.fill_between(
        dd_strat.index,
        dd_strat.values,
        0,
        color="steelblue",
        alpha=0.4,
        label="Strategy DD",
    )
    if bh_equity is not None and not bh_equity.empty:
        bh_dd = bh_aligned / bh_aligned.cummax() - 1
        ax2.plot(
            bh_dd.index,
            bh_dd.values,
            color="grey",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label="B&H DD",
        )
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(top=0)

    # Panel 3: MA filter ratio
    prices = df[price_col].dropna()
    prices.index = pd.to_datetime(prices.index)
    fast = snap["params"].get("fast", 50)
    slow = snap["params"].get("slow", 200)
    oos_start = wf_equity.index.min()

    fast_ma = prices.rolling(fast).mean().shift(1)
    slow_ma = prices.rolling(slow).mean().shift(1)
    ratio = (fast_ma / slow_ma - 1) * 100
    ratio_oos = ratio.loc[ratio.index >= oos_start]

    ax3.plot(
        ratio_oos.index,
        ratio_oos.values,
        color="darkorange",
        linewidth=1,
        label=f"fast({fast})/slow({slow})-1 [%]",
    )
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(
        ratio_oos.index,
        ratio_oos.values,
        0,
        where=(ratio_oos > 0),
        color="green",
        alpha=0.15,
        label="Filter ON",
    )
    ax3.fill_between(
        ratio_oos.index,
        ratio_oos.values,
        0,
        where=(ratio_oos < 0),
        color="red",
        alpha=0.15,
        label="Filter OFF",
    )
    ax3.set_ylabel("MA ratio (%)")
    ax3.legend(fontsize=7, loc="lower left")
    ax3.grid(True, alpha=0.25)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3.tick_params(axis="x", labelrotation=30)

    save_figure_atomically(
        figure=fig,
        chart_path=chart_path,
        dpi=70,
    )
    logging.info("daily_output: chart saved to %s", chart_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_daily_outputs(
    context: SingleAssetReportContext,
) -> dict:
    """
    Build and write all daily output artefacts for a single-asset strategy.

    ``context`` bundles the stitched OOS equity/trades/metrics, the per-window
    results, the buy-and-hold benchmark, the price frame and the output / Drive
    settings (see ``SingleAssetReportContext``).

    Returns
    -------
    dict with keys: action, signal, status_text, log_row,
                    chart_path, snapshot_path, log_path
    """
    run_date = resolve_run_date(run_date=context.run_date)
    artifact_paths = resolve_artifact_paths(
        output_dir=context.output_dir,
        asset_name=context.asset_name,
    )
    log_path = artifact_paths.log_path

    if context.gdrive_folder_id and context.gdrive_credentials:
        fetch_file_from_drive(
            local_path=log_path,
            folder_id=context.gdrive_folder_id,
            filename=artifact_paths.log_filename,
            credentials_path=context.gdrive_credentials,
        )
    else:
        logging.info("daily_output: gdrive credentials not supplied — skipping Drive log fetch.")

    snap = SingleAssetSnapshotBuilder(context=context, run_date=run_date).build()

    prev_log = load_existing_log(log_path=log_path)
    action = _determine_action(prev_log, snap["signal"])
    snap["_action"] = action  # underscore copy for chart title
    snap["action"] = action

    status_text = _build_status_text(snap, action, context.asset_name)
    logging.info("\n%s", status_text)
    atomic_write(path=artifact_paths.status_path, content=status_text)
    logging.info("daily_output: status written to %s", artifact_paths.status_path)

    log_row = _build_log_row(snap, action)
    append_log_row(log_path=log_path, row=log_row)
    logging.info("daily_output: signal log updated at %s", log_path)

    snap_serialisable = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(
        path=artifact_paths.snapshot_path,
        content=json.dumps(snap_serialisable, indent=2, default=str),
    )
    logging.info("daily_output: JSON snapshot written to %s", artifact_paths.snapshot_path)

    _build_chart(
        wf_equity=context.wf_equity,
        bh_equity=context.bh_equity,
        wf_trades=context.wf_trades,
        wf_results=context.wf_results,
        df=context.df,
        snap=snap,
        chart_path=artifact_paths.chart_path,
        price_col=context.price_col,
    )

    logging.info(
        "daily_output: COMPLETE — signal=%s  action=%s  date=%s",
        snap["signal"],
        action,
        run_date,
    )
    upload_artifacts_to_drive(
        artifact_paths=artifact_paths.upload_candidates(),
        gdrive_folder_id=context.gdrive_folder_id,
        gdrive_credentials=context.gdrive_credentials,
    )
    return {
        "action": action,
        "signal": snap["signal"],
        "status_text": status_text,
        "log_row": log_row,
        "chart_path": artifact_paths.chart_path,
        "snapshot_path": artifact_paths.snapshot_path,
        "log_path": log_path,
    }
