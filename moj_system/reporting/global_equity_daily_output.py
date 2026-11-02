# -*- coding: utf-8 -*-
"""
global_equity_daily_output.py
==============================
Builds and writes daily output artefacts for the global equity portfolio
strategy (both "global_equity" and "msci_world" modes).

Mirrors the pattern of multiasset_daily_output.py, adapted for an
N-asset portfolio with mode-specific asset labels.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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
from moj_system.reporting.report_context import GlobalReportContext
from moj_system.reporting.snapshot_builder import (
    SnapshotBuilder,
    build_position_block,
    compute_stop_levels,
    round_metrics,
)

# ---------------------------------------------------------------------------
# Signal & State extraction helpers (Ported from Multiasset)
# ---------------------------------------------------------------------------


def _get_signal_from_series(sig_oos: pd.Series | None) -> str:
    if sig_oos is None or sig_oos.empty:
        return "OUT"
    return "IN" if int(sig_oos.iloc[-1]) == 1 else "OUT"


def _get_current_weights(weights_series: pd.Series | None) -> dict:
    if weights_series is None or weights_series.empty:
        return {}
    return {k: round(number=float(v), ndigits=4) for k, v in dict(weights_series.iloc[-1]).items()}


def _log_allocation_changes(weights_series: pd.Series | None, asset_keys: list) -> None:
    if weights_series is None or weights_series.empty:
        return

    weights_df = pd.DataFrame(weights_series.tolist(), index=weights_series.index).sort_index()
    columns = [*asset_keys, "mmf"]
    weights_df = weights_df.reindex(columns=columns, fill_value=0.0).fillna(0.0)

    changed = weights_df.diff().abs().gt(1e-9).any(axis=1)
    changed.iloc[0] = True
    changes = weights_df.loc[changed].mul(100.0)
    changes.index.name = "Date"
    changes = changes.rename(columns={"mmf": "MMF"})

    logging.info(
        "APPLIED ALLOCATION CHANGES (%%):\n%s",
        changes.to_string(float_format=lambda value: f"{value:.1f}"),
    )


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------


def _determine_action(
    prev_log: pd.DataFrame | None,
    signals_today: dict,
    realloc_today: bool,
    asset_keys: list,
) -> str:
    if prev_log is None or prev_log.empty:
        return "HOLD"

    actions = []
    for key in asset_keys:
        col = f"signal_{key}"
        if col not in prev_log.columns:
            continue
        prev_sig = str(prev_log[col].iloc[-1]).strip().upper()
        curr_sig = signals_today.get(key, "OUT").upper()
        if prev_sig == "OUT" and curr_sig == "IN":
            actions.append(f"ENTER_{key}")
        elif prev_sig == "IN" and curr_sig == "OUT":
            actions.append(f"EXIT_{key}")

    if not actions and realloc_today:
        actions.append("REALLOC")

    return "+".join(actions) if actions else "HOLD"


def _value_at_date(series: pd.Series, date: object) -> float:
    timestamp = pd.Timestamp(date)
    value = series.reindex(pd.DatetimeIndex([timestamp]), method="ffill").iloc[0]
    return float(value)


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


class GlobalSnapshotBuilder(SnapshotBuilder[GlobalReportContext]):
    """Snapshot of the N-asset global portfolio (adds mode / FX-hedge header fields)."""

    def build_header(self) -> dict[str, Any]:
        return {
            "run_date": str(self.run_date),
            "portfolio_mode": self.context.portfolio_mode,
            "fx_hedged": self.context.fx_hedged,
        }

    def build_body(self) -> dict[str, Any]:
        ctx = self.context
        snap: dict[str, Any] = {}

        freshness = {k: str(df.index.max().date()) for k, df in ctx.price_df_dict.items() if not df.empty}
        if ctx.portfolio_equity is not None and not ctx.portfolio_equity.empty:
            freshness["Portfolio"] = str(ctx.portfolio_equity.index.max().date())
        snap["data_freshness"] = freshness

        snap["current_regime_adx"] = get_current_adx_regime(df=ctx.price_df_dict.get("WIG"))
        snap["signals"] = {
            k: _get_signal_from_series(sig_oos=ctx.signals_oos_dict.get(k)) for k in ctx.asset_keys
        }
        snap["weights"] = _get_current_weights(weights_series=ctx.weights_series)
        snap["realloc_today"] = any(
            pd.Timestamp(r["Date"]).date() == self.run_date for r in ctx.reallocation_log if "Date" in r
        )
        snap["asset_states"] = {k: self._asset_state(asset_key=k) for k in ctx.asset_keys}

        snap["portfolio_metrics"] = round_metrics(metrics=ctx.portfolio_metrics)
        snap["bh_metrics"] = {
            k: {mk: round(number=float(mv), ndigits=4) for mk, mv in mdict.items()}
            for k, mdict in ctx.bh_metrics_dict.items()
        }

        if ctx.reallocation_log:
            last_r = ctx.reallocation_log[-1]
            snap["last_realloc"] = {
                "date": str(pd.Timestamp(last_r["Date"]).date()),
                "reason": last_r.get("reason", "N/A"),
                "weights_after": last_r.get("weights_after", {}),
            }
            snap["n_reallocations"] = len(ctx.reallocation_log)
        else:
            snap["last_realloc"] = None
            snap["n_reallocations"] = 0

        if ctx.portfolio_equity is not None and not ctx.portfolio_equity.empty:
            snap["oos_start"] = str(ctx.portfolio_equity.index.min().date())
            snap["oos_end"] = str(ctx.portfolio_equity.index.max().date())
            snap["portfolio_level"] = float(ctx.portfolio_equity.iloc[-1])
        return snap

    def _asset_state(
        self,
        asset_key: str,
    ) -> dict[str, Any]:
        ctx = self.context
        state: dict[str, Any] = {}
        params = get_active_window_params(wf_results=ctx.wf_results_dict.get(asset_key))
        state["params"] = params
        df = ctx.price_df_dict.get(asset_key)

        if df is None or df.empty:
            state["position"] = None
            state["ma_state"] = {}
            state["mom_state"] = {}
            return state

        lookback = params.get("mom_lookback", 252)
        state["ma_state"] = compute_ma_filter_state(
            df=df,
            fast=params.get("fast", 50),
            slow=params.get("slow", 200),
            gap_digits=3,
        )
        state["mom_state"] = compute_mom_filter_state(
            df=df,
            lookback=lookback,
            min_history=lookback + 21,
        )

        open_position = extract_open_position(
            wf_trades=ctx.wf_trades_dict.get(asset_key),
            last_row_only=True,
        )
        if not open_position:
            state["position"] = None
            return state

        display_df = ctx.display_price_df_dict.get(asset_key)
        price_df = display_df if display_df is not None else df
        prices = price_df["Zamkniecie"].dropna()
        entry_date = pd.Timestamp(open_position["EntryDate"])
        if display_df is not None:
            entry_price = _value_at_date(prices, entry_date)
        else:
            entry_price = float(open_position["EntryPrice"])
        in_trade = prices.loc[prices.index >= entry_date]
        stops = compute_stop_levels(
            price_df=price_df,
            params=params,
            entry_price=entry_price,
            peak_price=float(in_trade.max()),
        )
        state["atr_val"] = round(number=stops.atr_val, ndigits=3) if stops.atr_val is not None else None
        position = build_position_block(
            open_position=open_position,
            stops=stops,
            entry_price=entry_price,
            today_price=float(prices.iloc[-1]),
        )
        if display_df is not None:
            original_prices = display_df["ZamkniecieOriginalCurrency"].dropna()
            position.update(
                {
                    "currency": display_df.attrs.get("currency"),
                    "entry_price_original_currency": round(
                        number=_value_at_date(original_prices, entry_date),
                        ndigits=2,
                    ),
                    "today_price_original_currency": round(
                        number=float(original_prices.iloc[-1]),
                        ndigits=2,
                    ),
                },
            )
        state["position"] = position
        return state


# ---------------------------------------------------------------------------
# Status text
# ---------------------------------------------------------------------------


def _build_status_text(snap: dict, action: str, asset_keys: list) -> str:
    sep = "=" * 65
    sep2 = "-" * 65
    w = snap.get("weights", {})
    pm = snap.get("portfolio_metrics", {})

    lines = [
        sep,
        f"  GLOBAL EQUITY STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Mode:           {snap['portfolio_mode']} | FX: {'hedged' if snap['fx_hedged'] else 'unhedged'}",
        f"  Action:         {action}",
        f"  Rynek (ADX):    {snap.get('current_regime_adx', 'N/A').upper()}",
        f"  Dane z dnia:    {snap.get('data_freshness', {})}",  # <--- DODANA LINIA
        sep2,
        "  CURRENT ALLOCATION",
    ]

    for k in asset_keys:
        wt = w.get(k, 0.0)
        lines.append(f"  {k:<14} {wt * 100.0:.0f}%")
    wt_mmf = w.get("mmf", 1.0 - sum(w.get(k, 0.0) for k in asset_keys))
    lines.append(f"  {'MMF':<14} {wt_mmf * 100.0:.0f}%")
    lines.append(sep2)

    # Detailed per-asset blocks
    for k in asset_keys:
        lines.append(f"[{k}] COMPONENT POSITION")
        state = snap["asset_states"].get(k, {})
        pos = state.get("position")
        par = state.get("params", {})

        if pos:
            entry_price_text = (
                f"{pos['entry_price']} PLN ({pos['entry_price_original_currency']} {pos['currency']})"
                if pos.get("currency")
                else str(pos["entry_price"])
            )
            today_price_text = (
                f"{pos['today_price']} PLN ({pos['today_price_original_currency']} {pos['currency']})"
                if pos.get("currency")
                else str(pos["today_price"])
            )
            if par.get("use_atr_stop"):
                trail_str = (
                    f"  Trail stop:     {pos['trail_stop']}  (peak {pos['peak_price']} × "
                    f"(1 - {par.get('stop_param', 0):.2f}[N_atr] × "
                    f"{state.get('atr_val', 0.0):.2f}%[ATR]))"
                )
            else:
                trail_str = (
                    f"  Trail stop:     {pos['trail_stop']}  (peak {pos['peak_price']} × "
                    f"(1 - {par.get('stop_param', 0):.2f}[{par.get('stop_label', 'X')}]))"
                )

            lines += [
                f"  Entry date:     {pos['entry_date']}",
                f"  Entry price:    {entry_price_text}",
                f"  Today price:    {today_price_text}",
                f"  Days in trade:  {pos['days_in_trade']}",
                f"  Unrealised:     {pos['unrealised_pct']:+.2f}%",
                trail_str,
                f"  Abs stop:       {pos['abs_stop']}  (entry × (1 - {par.get('stop_loss', 0):.0%}))",
                f"  Binding stop:   {pos['binding_stop']}  (gap: {pos['stop_gap_pct']:+.1f}%)",
            ]
        else:
            lines.append("  No open position.")

        ma_state = state.get("ma_state", {})
        mom_state = state.get("mom_state", {})
        fmode = par.get("filter_mode", "ma").upper()

        lines.append(f"  Filter (Active: {fmode}):")
        lines.append(
            f"    MA:  {ma_state.get('fast_ma')} / {ma_state.get('slow_ma')} "
            f"(gap {ma_state.get('gap_pct', 0.0):+.2f}%) "
            f"-> {'ON' if ma_state.get('filter_on') else 'OFF'}",
        )
        if mom_state.get("mom_value") is not None:
            lines.append(
                f"    MOM: {mom_state.get('mom_value'):+.2f}% "
                f"-> {'ON' if mom_state.get('filter_on') else 'OFF'}",
            )
        lines.append(sep2)

    lines += [
        "  PORTFOLIO OOS METRICS",
        f"  CAGR: {pm.get('CAGR', 0.0) * 100.0:+.2f}% | "
        f"Sharpe: {pm.get('Sharpe', 0.0):.2f} | "
        f"MaxDD: {pm.get('MaxDD', 0.0) * 100.0:+.2f}%",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row
# ---------------------------------------------------------------------------


def _build_log_row(snap: dict, action: str, asset_keys: list) -> dict:
    row = {
        "date": snap["run_date"],  # ZMIENIONE Z "Date" NA "date"
        "action": action,  # ZMIENIONE Z "Action" NA "action"
        "mode": snap["portfolio_mode"],  # ZMIENIONE Z "Mode" NA "mode"
        "regime_adx": snap.get("current_regime_adx"),
        "realloc_today": snap.get("realloc_today", False),
    }
    for k in asset_keys:
        row[f"signal_{k}"] = snap["signals"].get(k, "OUT")
        row[f"weight_{k}"] = snap["weights"].get(k, 0.0)

        pos = snap["asset_states"].get(k, {}).get("position")
        if pos:
            row[f"{k}_entry_date"] = pos["entry_date"]
            row[f"{k}_unrealised_pct"] = pos["unrealised_pct"]
            if pos.get("currency"):
                row[f"{k}_entry_price_pln"] = pos["entry_price"]
                row[f"{k}_today_price_pln"] = pos["today_price"]
                row[f"{k}_entry_price_original_currency"] = pos["entry_price_original_currency"]
                row[f"{k}_today_price_original_currency"] = pos["today_price_original_currency"]
                row[f"{k}_currency"] = pos["currency"]
        else:
            row[f"{k}_entry_date"] = None
            row[f"{k}_unrealised_pct"] = None

    row["weight_mmf"] = snap["weights"].get(
        "mmf",
        max(0.0, 1.0 - sum(snap["weights"].get(k, 0.0) for k in asset_keys)),
    )
    pm = snap.get("portfolio_metrics", {})
    row["portfolio_cagr"] = pm.get("CAGR")
    row["portfolio_sharpe"] = pm.get("Sharpe")
    row["portfolio_maxdd"] = pm.get("MaxDD")
    return row


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def _build_chart(
    portfolio_equity: pd.Series,
    signals_oos_dict: dict,
    reallocation_log: list,
    returns_dict: dict,
    chart_path: Path,
    action: str,
    asset_keys: list,
    run_date: dt.date,
    portfolio_mode: str,
    fx_hedged: bool,
) -> None:
    if portfolio_equity is None or portfolio_equity.empty:
        logging.warning("_build_chart: portfolio_equity is empty, skipping.")
        return

    oos_start = portfolio_equity.index.min()
    oos_end = portfolio_equity.index.max()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))
    fig.suptitle(
        t=f"Global Equity Portfolio[{portfolio_mode}]  "
        f"{'Hedged' if fx_hedged else 'Unhedged'}  "
        f"{oos_start.date()} → {oos_end.date()}  |  Action: {action}",
        fontsize=11,
    )

    ax1 = axes[0]
    ax1.set_title(label="Portfolio OOS equity (black) vs buy-and-hold per asset")
    ax1.set_yscale(value="log")

    port_norm = portfolio_equity / portfolio_equity.iloc[0] * 100.0
    ax1.plot(
        port_norm.index,
        port_norm.values,
        color="black",
        linewidth=2.0,
        label="Portfolio",
        zorder=10,
    )

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, (key, ret) in enumerate(returns_dict.items()):
        ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
        if ret_oos.empty:
            continue
        bh = (1.0 + ret_oos).cumprod() * 100.0
        ax1.plot(
            bh.index,
            bh.values,
            color=colors[i % len(colors)],
            linewidth=0.9,
            alpha=0.65,
            label=key,
        )

    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    ax1.set_ylabel(ylabel="Cumulative return (log, base=100)")
    ax1.grid(visible=True, alpha=0.25)

    ax2 = axes[1]
    ax2.set_title(label="Per-asset signal state (shaded = in position)")

    y_offset = 0.0
    ytick_pos, ytick_labels = [], []
    for key in asset_keys:
        sig = signals_oos_dict.get(key)
        if sig is None or sig.empty:
            continue
        ax2.fill_between(
            sig.index,
            y_offset,
            y_offset + sig.values * 0.8,
            alpha=0.55,
            step="post",
            label=key,
        )
        ytick_pos.append(y_offset + 0.4)
        ytick_labels.append(key)
        y_offset += 1.0

    for r in reallocation_log:
        d = pd.Timestamp(r["Date"])
        if oos_start <= d <= oos_end:
            ax2.axvline(x=d, color="red", alpha=0.3, linewidth=0.6)

    ax2.set_yticks(ticks=ytick_pos)
    ax2.set_yticklabels(labels=ytick_labels, fontsize=8)
    ax2.set_ylabel(ylabel="Signal (shaded=IN)")
    ax2.set_xlabel(xlabel=f"Run date: {run_date}")
    ax2.grid(visible=True, alpha=0.2)

    plt.tight_layout()

    save_figure_atomically(
        figure=fig,
        chart_path=chart_path,
        dpi=72,
    )
    logging.info(msg=f"_build_chart: saved to {chart_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_daily_outputs(
    context: GlobalReportContext,
) -> dict:
    run_date = resolve_run_date(run_date=context.run_date)
    artifact_paths = resolve_artifact_paths(
        output_dir=context.output_dir,
        asset_name=context.asset_name,
    )
    log_path = artifact_paths.log_path
    asset_keys = context.asset_keys

    if context.gdrive_folder_id and context.gdrive_credentials:
        fetch_file_from_drive(
            local_path=log_path,
            folder_id=context.gdrive_folder_id,
            filename=artifact_paths.log_filename,
            credentials_path=context.gdrive_credentials,
        )
    else:
        logging.info(msg="global_equity_daily_output: skipping log pre-fetch.")

    snap = GlobalSnapshotBuilder(context=context, run_date=run_date).build()

    prev_log = load_existing_log(log_path=log_path)
    action = _determine_action(
        prev_log=prev_log,
        signals_today=snap["signals"],
        realloc_today=snap.get("realloc_today", False),
        asset_keys=asset_keys,
    )
    snap["action"] = action

    status_text = _build_status_text(
        snap=snap,
        action=action,
        asset_keys=asset_keys,
    )
    logging.info(msg=f"\n{status_text}")
    _log_allocation_changes(weights_series=context.weights_series, asset_keys=asset_keys)
    atomic_write(path=artifact_paths.status_path, content=status_text)
    logging.info(msg=f"build_daily_outputs: status written to {artifact_paths.status_path}")

    log_row = _build_log_row(
        snap=snap,
        action=action,
        asset_keys=asset_keys,
    )
    append_log_row(log_path=log_path, row=log_row)
    logging.info(msg=f"build_daily_outputs: log updated at {log_path}")

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(
        path=artifact_paths.snapshot_path,
        content=json.dumps(snap_clean, indent=2, default=str),
    )
    logging.info(msg=f"build_daily_outputs: snapshot written to {artifact_paths.snapshot_path}")

    _build_chart(
        portfolio_equity=context.portfolio_equity,
        signals_oos_dict=context.signals_oos_dict,
        reallocation_log=context.reallocation_log,
        returns_dict=context.returns_dict,
        chart_path=artifact_paths.chart_path,
        action=action,
        asset_keys=asset_keys,
        run_date=run_date,
        portfolio_mode=context.portfolio_mode,
        fx_hedged=context.fx_hedged,
    )

    upload_artifacts_to_drive(
        artifact_paths=artifact_paths.upload_candidates(),
        gdrive_folder_id=context.gdrive_folder_id,
        gdrive_credentials=context.gdrive_credentials,
    )

    return {
        "action": action,
        "status_text": status_text,
        "log_row": log_row,
        "chart_path": str(artifact_paths.chart_path),
        "snapshot_path": str(artifact_paths.snapshot_path),
        "log_path": str(log_path),
        "signals": snap["signals"],
        "weights": snap.get("weights", {}),
    }
