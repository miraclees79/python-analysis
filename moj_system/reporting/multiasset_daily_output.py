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
from moj_system.reporting.report_context import PensionReportContext
from moj_system.reporting.snapshot_builder import (
    SnapshotBuilder,
    build_position_block,
    compute_stop_levels,
    round_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------


def _get_signal_from_series(sig_oos: pd.Series | None) -> str:
    if sig_oos is None or sig_oos.empty:
        return "OUT"
    return "IN" if int(sig_oos.iloc[-1]) == 1 else "OUT"


def _get_signal_from_trades(wf_trades: pd.DataFrame | None) -> str:
    if wf_trades is None or wf_trades.empty:
        return "OUT"
    last_trade = wf_trades.iloc[-1]
    return "IN" if last_trade["Exit Reason"] == "CARRY" else "OUT"


def _get_current_weights(weights_series: pd.Series | None) -> dict:
    if weights_series is None or weights_series.empty:
        return {"equity": None, "bond": None, "mmf": None}
    last = weights_series.iloc[-1]
    return {
        "equity": round(number=float(last.get("equity", 0)), ndigits=4),
        "bond": round(number=float(last.get("bond", 0)), ndigits=4),
        "mmf": round(number=float(last.get("mmf", 0)), ndigits=4),
    }


def _log_allocation_changes(weights_series: pd.Series | None) -> None:
    if weights_series is None or weights_series.empty:
        return

    columns = ["equity", "bond", "mmf"]
    weights_df = pd.DataFrame(weights_series.tolist(), index=weights_series.index).sort_index()
    weights_df = weights_df.reindex(columns=columns, fill_value=0.0).fillna(0.0)

    changed = weights_df.diff().abs().gt(1e-9).any(axis=1)
    changed.iloc[0] = True
    changes = weights_df.loc[changed].mul(100.0)
    changes.index.name = "Date"

    logging.info(
        "APPLIED ALLOCATION CHANGES (%%):\n%s",
        changes.to_string(float_format=lambda value: f"{value:.1f}"),
    )


def _realloc_today(reallocation_log: list, run_date: dt.date) -> bool:
    if not reallocation_log:
        return False
    return any(pd.Timestamp(ts_val["Date"]).date() == run_date for ts_val in reallocation_log)


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------


def _determine_action(
    prev_log: pd.DataFrame | None,
    sig_eq_today: str,
    sig_bd_today: str,
    realloc_today: bool,
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


class PensionSnapshotBuilder(SnapshotBuilder[PensionReportContext]):
    """Snapshot of the WIG equity + TBSP bond + MMF pension portfolio."""

    def build_body(self) -> dict[str, Any]:
        ctx = self.context
        snap: dict[str, Any] = {}

        snap["data_freshness"] = {
            "WIG": str(ctx.wig_df.index.max().date()),
            "TBSP": str(ctx.tbsp_df.index.max().date()),
            "Portfolio": str(ctx.portfolio_equity.index.max().date()),
        }

        par_eq = get_active_window_params(wf_results=ctx.wf_results_eq)
        snap["signal_equity"] = (
            _get_signal_from_series(sig_oos=ctx.sig_eq_oos)
            if ctx.sig_eq_oos is not None and not ctx.sig_eq_oos.empty
            else _get_signal_from_trades(wf_trades=ctx.wf_trades_eq)
        )
        snap["params_equity"] = par_eq
        snap["ma_state_equity"] = compute_ma_filter_state(
            df=ctx.wig_df,
            fast=par_eq.get("fast"),
            slow=par_eq.get("slow"),
            gap_digits=3,
        )
        snap["mom_state_equity"] = compute_mom_filter_state(
            df=ctx.wig_df,
            lookback=par_eq.get("mom_lookback"),
            min_history=par_eq.get("mom_lookback") + 21,
        )
        equity_position, atr_val_equity = self._position_block(
            price_df=ctx.wig_df,
            params=par_eq,
            wf_trades=ctx.wf_trades_eq,
        )
        if equity_position is not None:
            snap["atr_val_equity"] = atr_val_equity
        snap["equity_position"] = equity_position

        par_bd = get_active_window_params(wf_results=ctx.wf_results_bd)
        snap["signal_bond"] = (
            _get_signal_from_series(sig_oos=ctx.sig_bd_oos)
            if ctx.sig_bd_oos is not None and not ctx.sig_bd_oos.empty
            else _get_signal_from_trades(wf_trades=ctx.wf_trades_bd)
        )
        snap["params_bond"] = par_bd
        snap["ma_state_bond"] = compute_ma_filter_state(
            df=ctx.tbsp_df,
            fast=par_bd.get("fast"),
            slow=par_bd.get("slow"),
            gap_digits=3,
        )
        bond_position, atr_val_bond = self._position_block(
            price_df=ctx.tbsp_df,
            params=par_bd,
            wf_trades=ctx.wf_trades_bd,
        )
        if bond_position is not None:
            snap["atr_val_bond"] = atr_val_bond
        snap["bond_position"] = bond_position

        snap["weights"] = _get_current_weights(weights_series=ctx.weights_series)
        snap["portfolio_metrics"] = round_metrics(metrics=ctx.portfolio_metrics)
        snap["bh_equity_metrics"] = round_metrics(metrics=ctx.bh_eq_metrics)
        snap["bh_bond_metrics"] = round_metrics(metrics=ctx.bh_bd_metrics)
        snap["current_regime_adx"] = get_current_adx_regime(df=ctx.wig_df)
        realloc_today = _realloc_today(reallocation_log=ctx.reallocation_log, run_date=self.run_date)
        snap["realloc_today"] = realloc_today
        snap["action"] = _determine_action(
            prev_log=None,  # the real action is derived in build_daily_outputs once the log is loaded
            sig_eq_today=snap["signal_equity"],
            sig_bd_today=snap["signal_bond"],
            realloc_today=realloc_today,
        )

        if ctx.reallocation_log:
            last_r = ctx.reallocation_log[-1]
            snap["last_realloc"] = {
                "date": str(pd.Timestamp(last_r["Date"]).date()),
                "equity_after": round(number=float(last_r["equity_after"]), ndigits=2),
                "bond_after": round(number=float(last_r["bond_after"]), ndigits=2),
                "mmf_after": round(number=float(last_r["mmf_after"]), ndigits=2),
                "reason": last_r.get("reason", ""),
            }
        snap["execution_check"] = self._execution_check()
        return snap

    @staticmethod
    def _position_block(
        price_df: pd.DataFrame,
        params: dict[str, Any],
        wf_trades: pd.DataFrame,
    ) -> tuple[dict[str, Any] | None, float | None]:
        open_position = extract_open_position(
            wf_trades=wf_trades,
            last_row_only=True,
        )
        if not open_position:
            return None, None

        prices = price_df["Zamkniecie"].dropna()
        entry_price = float(open_position["EntryPrice"])
        today_price = float(prices.iloc[-1])
        in_trade = prices.loc[prices.index >= pd.Timestamp(open_position["EntryDate"])]
        stops = compute_stop_levels(
            price_df=price_df,
            params=params,
            entry_price=entry_price,
            peak_price=float(in_trade.max()),
        )
        position = build_position_block(
            open_position=open_position,
            stops=stops,
            entry_price=entry_price,
            today_price=today_price,
        )
        atr_val = round(number=stops.atr_val, ndigits=3) if stops.atr_val is not None else None
        return position, atr_val

    def _execution_check(self) -> dict[str, Any] | None:
        ctx = self.context
        if ctx.exec_equity is None or not ctx.exec_metrics:
            return None
        exec_decomp = ctx.exec_decomp
        portfolio_metrics = ctx.portfolio_metrics

        ret_theory = ctx.portfolio_equity.pct_change().dropna()
        ret_exec = ctx.exec_equity.pct_change().dropna()
        common_ret_idx = ret_theory.index.intersection(other=ret_exec.index)
        correlation = ret_theory.loc[common_ret_idx].corr(other=ret_exec.loc[common_ret_idx])

        th_port_cagr = exec_decomp.get("theory_port_cagr", portfolio_metrics.get("CAGR", 0.0) * 100.0)
        cagr_drag = (ctx.exec_metrics.get("CAGR", 0.0) * 100.0) - th_port_cagr

        th_port_maxdd = exec_decomp.get("theory_port_maxdd", portfolio_metrics.get("MaxDD", 0.0) * 100.0)
        maxdd_drag = (ctx.exec_metrics.get("MaxDD", 0.0) * 100.0) - th_port_maxdd

        return {
            "cagr": round(number=ctx.exec_metrics.get("CAGR", 0.0) * 100.0, ndigits=2),
            "maxdd": round(number=ctx.exec_metrics.get("MaxDD", 0.0) * 100.0, ndigits=2),
            "cagr_drag_pp": round(number=cagr_drag, ndigits=2),
            "maxdd_drag_pp": round(number=maxdd_drag, ndigits=2),
            "correlation": round(number=correlation, ndigits=3),
            "latest_date": str(ctx.exec_equity.index.max().date()),
            "decomp": {k: round(number=v, ndigits=2) for k, v in exec_decomp.items()} if exec_decomp else {},
        }


# ---------------------------------------------------------------------------
# Status text
# ---------------------------------------------------------------------------
def _build_status_text(snap: dict, action: str) -> str:
    sep = "=" * 65
    sep2 = "-" * 65
    w = snap["weights"]

    lines = [
        sep,
        f"  MULTI-ASSET STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Action:         {action}",
        f"  Equity signal:  {snap['signal_equity']}",
        f"  Bond signal:    {snap['signal_bond']}",
        f"  Rynek (ADX):      {snap.get('current_regime_adx', 'N/A').upper()}",
        f"  Data freshness:    {snap.get('data_freshness', {})}",
        sep2,
        "  CURRENT ALLOCATION",
        f"  Equity (WIG):   {w['equity'] * 100:.0f}%" if w["equity"] is not None else "  Equity (WIG):   N/A",
        f"  Bond (TBSP):    {w['bond'] * 100:.0f}%" if w["bond"] is not None else "  Bond (TBSP):    N/A",
        f"  MMF:            {w['mmf'] * 100:.0f}%" if w["mmf"] is not None else "  MMF:            N/A",
        sep2,
    ]

    # Sekcja Akcji (WIG)
    lines.append("  EQUITY POSITION (WIG)")
    ep = snap.get("equity_position")
    pe = snap.get("params_equity", {})
    if ep:
        # BUDOWANIE ETYKIETY STOPA
        if pe.get("use_atr_stop"):
            trail_str = (
                f"  Trail stop:     {ep['trail_stop']}  (peak {ep['peak_price']} × "
                f"(1 - {pe.get('stop_param', 0):.2f}[N_atr] × "
                f"{snap.get('atr_val_equity', 0.0):.2f}%[ATR]))"
            )
        else:
            trail_str = (
                f"  Trail stop:     {ep['trail_stop']}  (peak {ep['peak_price']} × "
                f"(1 - {pe.get('stop_param', 0):.2f}[{pe.get('stop_label', 'X')}]))"
            )

        lines += [
            f"  Entry date:     {ep['entry_date']}",
            f"  Entry price:    {ep['entry_price']}",
            f"  Today price:    {ep['today_price']}",
            f"  Days in trade:  {ep['days_in_trade']}",
            f"  Unrealised:     {ep['unrealised_pct']:+.2f}%",
            trail_str,
            f"  Abs stop:       {ep['abs_stop']}  (entry × (1 - {pe.get('stop_loss', 0):.0%}))",
            f"  Binding stop:   {ep['binding_stop']}  (gap: {ep['stop_gap_pct']:+.1f}%)",
        ]
    else:
        lines.append("  No open equity position.")

    ma_eq = snap.get("ma_state_equity", {})
    mom_eq = snap.get("mom_state_equity", {})
    fmode_eq = pe.get("filter_mode", "ma").upper()
    lines.append(f"  Filter (Active: {fmode_eq}):")
    lines.append(
        f"    MA:  {ma_eq.get('fast_ma')} / {ma_eq.get('slow_ma')} "
        f"(gap {ma_eq.get('gap_pct'):+.2f}%) -> "
        f"{'ON' if ma_eq.get('filter_on') else 'OFF'}",
    )
    lines.append(
        f"    MOM: {mom_eq.get('mom_value'):+.2f}% -> {'ON' if mom_eq.get('filter_on') else 'OFF'}",
    )
    lines.append(sep2)

    # Sekcja Obligacji (TBSP)
    lines.append("  BOND POSITION (TBSP)")
    bp = snap.get("bond_position")
    pb = snap.get("params_bond", {})
    if bp:
        if pb.get("use_atr_stop"):
            trail_str_bd = (
                f"  Trail stop:     {bp['trail_stop']}  (peak {bp['peak_price']} × "
                f"(1 - {pb.get('stop_param', 0):.2f}[N_atr] × "
                f"{snap.get('atr_val_bond', 0.0):.2f}%[ATR]))"
            )
        else:
            trail_str_bd = (
                f"  Trail stop:     {bp['trail_stop']}  (peak {bp['peak_price']} × "
                f"(1 - {pb.get('stop_param', 0):.2f}[{pb.get('stop_label', 'X')}]))"
            )

        lines += [
            f"  Entry date:     {bp['entry_date']}",
            f"  Entry price:    {bp['entry_price']}",
            f"  Today price:    {bp['today_price']}",
            f"  Days in trade:  {bp['days_in_trade']}",
            f"  Unrealised:     {bp['unrealised_pct']:+.2f}%",
            trail_str_bd,
            f"  Abs stop:       {bp['abs_stop']}  (entry × (1 - {pb.get('stop_loss', 0):.0%}))",
            f"  Binding stop:   {bp['binding_stop']}  (gap: {bp['stop_gap_pct']:+.1f}%)",
        ]
    else:
        lines.append("  No open bond position.")

    ma_bd = snap.get("ma_state_bond", {})
    lines.append(
        f"  MA Filter: {ma_bd.get('fast_ma')} / {ma_bd.get('slow_ma')} "
        f"(gap {ma_bd.get('gap_pct'):+.2f}%) -> "
        f"{'ON' if ma_bd.get('filter_on') else 'OFF'}",
    )
    lines.append(sep2)
    ec = snap.get("execution_check")
    if ec:
        drag_sign = "+" if ec["cagr_drag_pp"] > 0 else ""
        drag_sign_mdd = "+" if ec.get("maxdd_drag_pp", 0.0) > 0 else ""
        th_cagr = ec["decomp"].get("theory_port_cagr", 0.0)
        th_maxdd = ec["decomp"].get("theory_port_maxdd", 0.0)

        lines += [
            "  EXECUTION REALITY CHECK (PPE FUNDS)",
            f"  Simulated to:   {ec['latest_date']}",
            f"  Exec CAGR:      {ec['cagr']:+.2f}%  "
            f"(vs Matched Theory {th_cagr:+.2f}% -> Drag: {drag_sign}{ec['cagr_drag_pp']} pp)",
            f"  Exec MaxDD:     {ec['maxdd']:+.2f}%  "
            f"(vs Matched Theory {th_maxdd:+.2f}% -> "
            f"Drag: {drag_sign_mdd}{ec.get('maxdd_drag_pp', 0.0)} pp)",
            f"  Correlation:    {ec['correlation']:.2f}  (1.0 = perfect match)",
        ]

        d = ec.get("decomp")
        if d:
            lines += [
                "  -- Drag Decomposition --",
                f"  Equity (WIG):  Fund {d.get('eq_fund_cagr', 0):+.2f}% "
                f"(MDD: {d.get('eq_fund_maxdd', 0):.2f}%) vs Idx "
                f"{d.get('eq_idx_cagr', 0):+.2f}% "
                f"(MDD: {d.get('eq_idx_maxdd', 0):.2f}%) -> "
                f"Port impact: {d.get('port_impact_eq', 0):+.2f} pp",
                f"  Bond (TBSP):   Fund {d.get('bd_fund_cagr', 0):+.2f}% "
                f"(MDD: {d.get('bd_fund_maxdd', 0):.2f}%) vs Idx "
                f"{d.get('bd_idx_cagr', 0):+.2f}% "
                f"(MDD: {d.get('bd_idx_maxdd', 0):.2f}%) -> "
                f"Port impact: {d.get('port_impact_bd', 0):+.2f} pp",
                f"  MMF:           Fund {d.get('mmf_fund_cagr', 0):+.2f}% "
                f"vs Idx {d.get('mmf_idx_cagr', 0):+.2f}% -> "
                f"Port impact: {d.get('port_impact_mmf', 0):+.2f} pp",
            ]
        lines.append(sep2)
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row
# ---------------------------------------------------------------------------


def _build_log_row(snap: dict, action: str) -> dict:
    w = snap["weights"]
    pm = snap["portfolio_metrics"]
    ep = snap.get("equity_position") or {}
    bp = snap.get("bond_position") or {}
    return {
        "date": snap["run_date"],
        "action": action,
        "signal_equity": snap["signal_equity"],
        "signal_bond": snap["signal_bond"],
        "regime_adx": snap.get("current_regime_adx"),
        "w_equity": w.get("equity"),
        "w_bond": w.get("bond"),
        "w_mmf": w.get("mmf"),
        "realloc_today": snap.get("realloc_today", False),
        "eq_entry_date": ep.get("entry_date"),
        "eq_entry_price": ep.get("entry_price"),
        "eq_unrealised_pct": ep.get("unrealised_pct"),
        "eq_binding_stop": ep.get("binding_stop"),
        "eq_stop_gap_pct": ep.get("stop_gap_pct"),
        "bd_entry_date": bp.get("entry_date"),
        "bd_entry_price": bp.get("entry_price"),
        "bd_unrealised_pct": bp.get("unrealised_pct"),
        "bd_binding_stop": bp.get("binding_stop"),
        "bd_stop_gap_pct": bp.get("stop_gap_pct"),
        "portfolio_cagr": pm.get("CAGR"),
        "portfolio_maxdd": pm.get("MaxDD"),
        "portfolio_calmar": pm.get("CalMAR"),
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def _build_chart(
    portfolio_equity: pd.Series,
    sig_eq_oos: pd.Series,
    sig_bd_oos: pd.Series,
    reallocation_log: list,
    bh_eq_equity: pd.Series,
    bh_bd_equity: pd.Series,
    chart_path: Path,
    action: str,
    run_date: dt.date,
    exec_equity: pd.Series | None = None,
) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle(t=f"Multi-Asset Strategy — {run_date}  [{action}]", fontsize=12, fontweight="bold")

    oos_start = portfolio_equity.index.min()

    # --- PANEL 1: EQUITY ---
    ax = axes[0]
    portfolio_equity.plot(ax=ax, label="Theory Portfolio", color="steelblue", linewidth=1.8)

    # Dodanie krzywej Execution PPE
    if exec_equity is not None and not exec_equity.empty:
        align_date = exec_equity.index.min()
        # "Zaczepiamy" krzywą PPE o poziom teoretycznego portfela z tego samego dnia
        align_factor = (
            float(portfolio_equity.asof(align_date)) if pd.notna(portfolio_equity.asof(align_date)) else 1.0
        )
        exec_line = exec_equity * align_factor
        exec_line.plot(ax=ax, label="Execution PPE", color="purple", linewidth=1.5, alpha=0.9)

    if bh_eq_equity is not None:
        bh_eq = (
            bh_eq_equity.loc[bh_eq_equity.index >= oos_start]
            / bh_eq_equity.loc[bh_eq_equity.index >= oos_start].iloc[0]
        )
        bh_eq.plot(
            ax=ax,
            label="B&H WIG",
            color="darkorange",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    if bh_bd_equity is not None:
        bh_bd = (
            bh_bd_equity.loc[bh_bd_equity.index >= oos_start]
            / bh_bd_equity.loc[bh_bd_equity.index >= oos_start].iloc[0]
        )
        bh_bd.plot(
            ax=ax,
            label="B&H TBSP",
            color="seagreen",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_title(label="Portfolio Equity (OOS)")
    ax.set_ylabel(ylabel="Equity (rebased to 1.0)")
    ax.legend(fontsize=8)
    ax.grid(visible=True, alpha=0.25)

    # --- PANEL 2: DRAWDOWN ---
    ax = axes[1]
    dd_port = portfolio_equity / portfolio_equity.cummax() - 1.0
    dd_port.plot(ax=ax, label="Theory DD", color="steelblue", linewidth=1.5)

    # Dodanie Drawdownu dla Execution PPE
    if exec_equity is not None and not exec_equity.empty:
        dd_exec = exec_equity / exec_equity.cummax() - 1.0
        dd_exec.plot(ax=ax, label="Execution DD", color="purple", linewidth=1.2, alpha=0.8)

    if bh_eq_equity is not None:
        bh_eq = (
            bh_eq_equity.loc[bh_eq_equity.index >= oos_start]
            / bh_eq_equity.loc[bh_eq_equity.index >= oos_start].iloc[0]
        )
        dd_eq = bh_eq / bh_eq.cummax() - 1.0
        dd_eq.plot(
            ax=ax,
            label="B&H WIG",
            color="darkorange",
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
    ax.set_title(label="Drawdown")
    ax.set_ylabel(ylabel="Drawdown")
    ax.legend(fontsize=8)
    ax.grid(visible=True, alpha=0.25)

    # --- PANEL 3: SIGNALS ---
    ax = axes[2]
    if sig_eq_oos is not None and not sig_eq_oos.empty:
        sig_eq_oos.plot(ax=ax, label="Equity signal", color="darkorange", linewidth=0.9, alpha=0.8)
    if sig_bd_oos is not None and not sig_bd_oos.empty:
        sig_bd_oos.plot(ax=ax, label="Bond signal", color="seagreen", linewidth=0.9, alpha=0.8)
    if reallocation_log:
        rdates = [pd.Timestamp(r["Date"]) for r in reallocation_log]
        ax.vlines(
            x=rdates,
            ymin=0,
            ymax=1.05,
            color="grey",
            linewidth=0.5,
            alpha=0.5,
            label="Reallocation",
        )
    ax.set_title(label="Asset Signals and Reallocation Events")
    ax.set_ylabel(ylabel="Signal (0=off, 1=on)")
    ax.set_ylim(bottom=-0.05, top=1.15)
    ax.legend(fontsize=8)
    ax.grid(visible=True, alpha=0.25)

    plt.tight_layout()
    save_figure_atomically(
        figure=fig,
        chart_path=chart_path,
        dpi=72,
    )
    logging.info(msg=f"multiasset_daily_output: chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_daily_outputs(
    context: PensionReportContext,
) -> dict:
    """
    Build and write all daily output artefacts for the multi-asset strategy.

    Returns dict with keys: action, signal_equity, signal_bond, status_text,
                            log_row, chart_path, snapshot_path, log_path
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
        logging.info("multiasset_daily_output: skipping Drive log fetch.")

    snap = PensionSnapshotBuilder(context=context, run_date=run_date).build()

    prev_log = load_existing_log(log_path=log_path)
    action = _determine_action(
        prev_log=prev_log,
        sig_eq_today=snap["signal_equity"],
        sig_bd_today=snap["signal_bond"],
        realloc_today=snap.get("realloc_today", False),
    )
    snap["action"] = action

    status_text = _build_status_text(snap, action)
    logging.info("\n%s", status_text)
    _log_allocation_changes(weights_series=context.weights_series)
    atomic_write(path=artifact_paths.status_path, content=status_text)
    logging.info("multiasset_daily_output: status written to %s", artifact_paths.status_path)

    log_row = _build_log_row(snap, action)
    append_log_row(log_path=log_path, row=log_row)
    logging.info("multiasset_daily_output: log updated at %s", log_path)

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(
        path=artifact_paths.snapshot_path,
        content=json.dumps(snap_clean, indent=2, default=str),
    )
    logging.info("multiasset_daily_output: snapshot written to %s", artifact_paths.snapshot_path)

    _build_chart(
        portfolio_equity=context.portfolio_equity,
        sig_eq_oos=context.sig_eq_oos,
        sig_bd_oos=context.sig_bd_oos,
        reallocation_log=context.reallocation_log,
        bh_eq_equity=context.bh_eq_equity,
        bh_bd_equity=context.bh_bd_equity,
        chart_path=artifact_paths.chart_path,
        action=action,
        run_date=run_date,
        exec_equity=context.exec_equity,
    )
    upload_artifacts_to_drive(
        artifact_paths=artifact_paths.upload_candidates(),
        gdrive_folder_id=context.gdrive_folder_id,
        gdrive_credentials=context.gdrive_credentials,
    )
    return {
        "action": action,
        "signal_equity": snap["signal_equity"],
        "signal_bond": snap["signal_bond"],
        "status_text": status_text,
        "log_row": log_row,
        "chart_path": artifact_paths.chart_path,
        "snapshot_path": artifact_paths.snapshot_path,
        "log_path": log_path,
    }
