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
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moj_system.core.research import get_current_adx_regime
from moj_system.reporting.output_base import (
    append_log_row,
    atomic_write,
    atomic_write_bytes,
    fetch_file_from_drive,
    load_existing_log,
)


# ---------------------------------------------------------------------------
# Signal & State extraction helpers (Ported from Multiasset)
# ---------------------------------------------------------------------------

def _get_signal_from_series(sig_oos: pd.Series | None) -> str:
    if sig_oos is None or sig_oos.empty:
        return "OUT"
    return "IN" if int(sig_oos.iloc[-1]) == 1 else "OUT"


def _get_open_position(wf_trades: pd.DataFrame | None) -> dict | None:
    if wf_trades is None or wf_trades.empty:
        return None
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    if carry.empty:
        return None
    return carry.iloc[-1].to_dict()


def _get_active_window_params(wf_results: pd.DataFrame | None) -> dict:
    if wf_results is None or wf_results.empty:
        return {}
    last = wf_results.iloc[-1]

    def _safe_float(key, default=None):
        val = last.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    use_atr = bool(last.get("use_atr_stop", False))
    stop_val = _safe_float("N_atr" if use_atr else "X", 0.10)
    stop_label = "N_atr" if use_atr else "X"

    return {
        "filter_mode": last.get("filter_mode", "ma"),
        "stop_param": stop_val,
        "stop_label": stop_label,
        "use_atr_stop": use_atr,
        "Y": _safe_float("Y", 0.10),
        "fast": int(_safe_float("fast", 50)),
        "slow": int(_safe_float("slow", 200)),
        "stop_loss": _safe_float("stop_loss", 0.05),
        "mom_lookback": int(_safe_float("mom_lookback", 252)),
    }


def _compute_ma_filter_state(df: pd.DataFrame, fast: int, slow: int) -> dict:
    prices = df["Zamkniecie"].dropna()
    if len(prices) < slow:
        return {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}

    fast_ma = float(prices.rolling(window=fast).mean().iloc[-1])
    slow_ma = float(prices.rolling(window=slow).mean().iloc[-1])
    filter_on = fast_ma > slow_ma
    gap_pct = round(number=(fast_ma / slow_ma - 1.0) * 100.0, ndigits=3)

    return {
        "fast_ma": round(number=fast_ma, ndigits=2),
        "slow_ma": round(number=slow_ma, ndigits=2),
        "filter_on": filter_on,
        "gap_pct": gap_pct,
    }


def _compute_mom_filter_state(df: pd.DataFrame, lookback: int) -> dict:
    prices = df["Zamkniecie"].dropna()
    if len(prices) < lookback + 21:
        return {"mom_value": None, "filter_on": None}

    mom_val = (prices.shift(periods=21).iloc[-1] / prices.shift(periods=lookback).iloc[-1]) - 1.0
    filter_on = mom_val > 0.0

    return {
        "mom_value": round(number=float(mom_val * 100.0), ndigits=2),
        "filter_on": filter_on,
    }


def _get_current_weights(weights_series: pd.Series | None) -> dict:
    if weights_series is None or weights_series.empty:
        return {}
    return {k: round(number=float(v), ndigits=4) for k, v in dict(weights_series.iloc[-1]).items()}


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

    actions =[]
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


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    wf_results_dict: dict,
    wf_trades_dict: dict,
    price_df_dict: dict,
    portfolio_equity: pd.Series,
    portfolio_metrics: dict,
    weights_series: pd.Series,
    reallocation_log: list,
    bh_metrics_dict: dict,
    signals_oos_dict: dict,
    asset_keys: list,
    portfolio_mode: str,
    fx_hedged: bool,
    run_date: dt.date,
) -> dict:
    
    snap = {
        "run_date": str(run_date),
        "portfolio_mode": portfolio_mode,
        "fx_hedged": fx_hedged,
    }

    # Data freshness
    freshness = {k: str(df.index.max().date()) for k, df in price_df_dict.items() if not df.empty}
    if portfolio_equity is not None and not portfolio_equity.empty:
        freshness["Portfolio"] = str(portfolio_equity.index.max().date())
    snap["data_freshness"] = freshness

    snap["current_regime_adx"] = get_current_adx_regime(df=price_df_dict.get("WIG"))

    snap["signals"] = {k: _get_signal_from_series(sig_oos=signals_oos_dict.get(k)) for k in asset_keys}
    snap["weights"] = _get_current_weights(weights_series=weights_series)
    snap["realloc_today"] = any(pd.Timestamp(r["Date"]).date() == run_date for r in reallocation_log if "Date" in r)

    # Detailed states per asset (mirrors WIG/TBSP from multiasset)
    asset_states = {}
    for k in asset_keys:
        state = {}
        par = _get_active_window_params(wf_results=wf_results_dict.get(k))
        state["params"] = par
        df = price_df_dict.get(k)
        
        if df is not None and not df.empty:
            state["ma_state"] = _compute_ma_filter_state(
                df=df,
                fast=par.get("fast", 50),
                slow=par.get("slow", 200),
            )
            state["mom_state"] = _compute_mom_filter_state(
                df=df,
                lookback=par.get("mom_lookback", 252),
            )
            
            pos = _get_open_position(wf_trades=wf_trades_dict.get(k))
            if pos:
                prices = df["Zamkniecie"].dropna()
                entry_px = float(pos["EntryPrice"])
                today_px = float(prices.iloc[-1])
                in_trade = prices.loc[prices.index >= pd.Timestamp(pos["EntryDate"])]
                peak_px = float(in_trade.max())

                trail_stop = round(number=peak_px * (1.0 - par.get("stop_param", 0.10)), ndigits=2)
                abs_stop = round(number=entry_px * (1.0 - par.get("stop_loss", 0.05)), ndigits=2)
                binding = max(trail_stop, abs_stop)

                state["position"] = {
                    "entry_date": pd.Timestamp(pos["EntryDate"]).date().isoformat(),
                    "entry_price": round(number=entry_px, ndigits=2),
                    "today_price": round(number=today_px, ndigits=2),
                    "days_in_trade": int(pos.get("Days", 0)),
                    "unrealised_pct": round(number=float(pos["Return"]) * 100.0, ndigits=2),
                    "peak_price": round(number=peak_px, ndigits=2),
                    "trail_stop": trail_stop,
                    "abs_stop": abs_stop,
                    "binding_stop": binding,
                    "stop_gap_pct": round(number=(binding - today_px) / today_px * 100.0, ndigits=2),
                }
            else:
                state["position"] = None
        else:
            state["position"] = None
            state["ma_state"] = {}
            state["mom_state"] = {}

        asset_states[k] = state

    snap["asset_states"] = asset_states

    # Metrics
    snap["portfolio_metrics"] = {k: round(number=float(v), ndigits=4) for k, v in (portfolio_metrics or {}).items()}
    snap["bh_metrics"] = {k: {mk: round(number=float(mv), ndigits=4) for mk, mv in mdict.items()} for k, mdict in bh_metrics_dict.items()}

    # Reallocation log
    if reallocation_log:
        last_r = reallocation_log[-1]
        snap["last_realloc"] = {
            "date": str(pd.Timestamp(last_r["Date"]).date()),
            "reason": last_r.get("reason", "N/A"),
            "weights_after": last_r.get("weights_after", {}),
        }
        snap["n_reallocations"] = len(reallocation_log)
    else:
        snap["last_realloc"] = None
        snap["n_reallocations"] = 0

    if portfolio_equity is not None and not portfolio_equity.empty:
        snap["oos_start"] = str(portfolio_equity.index.min().date())
        snap["oos_end"] = str(portfolio_equity.index.max().date())
        snap["portfolio_level"] = float(portfolio_equity.iloc[-1])

    return snap


# ---------------------------------------------------------------------------
# Status text
# ---------------------------------------------------------------------------

def _build_status_text(snap: dict, action: str, asset_keys: list) -> str:
    sep = "=" * 65
    sep2 = "-" * 65
    w = snap.get("weights", {})
    pm = snap.get("portfolio_metrics", {})

    lines =[
        sep,
        f"  GLOBAL EQUITY STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Mode:           {snap['portfolio_mode']} | FX: {'hedged' if snap['fx_hedged'] else 'unhedged'}",
        f"  Action:         {action}",
        f"  Rynek (ADX):    {snap.get('current_regime_adx', 'N/A').upper()}",
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
            lines += [
                f"  Entry date:     {pos['entry_date']}",
                f"  Entry price:    {pos['entry_price']}",
                f"  Today price:    {pos['today_price']}",
                f"  Days in trade:  {pos['days_in_trade']}",
                f"  Unrealised:     {pos['unrealised_pct']:+.2f}%",
                f"  Trail stop:     {pos['trail_stop']}  (peak {pos['peak_price']} × (1-{par.get('stop_param', 0):.2f}[{par.get('stop_label', 'X')}]))",
                f"  Abs stop:       {pos['abs_stop']}  (entry × (1-{par.get('stop_loss', 0):.0%}))",
                f"  Binding stop:   {pos['binding_stop']}  (gap: {pos['stop_gap_pct']:+.1f}%)",
            ]
        else:
            lines.append("  No open position.")

        ma_state = state.get("ma_state", {})
        mom_state = state.get("mom_state", {})
        fmode = par.get("filter_mode", "ma").upper()
        
        lines.append(f"  Filter (Active: {fmode}):")
        lines.append(
            f"    MA:  {ma_state.get('fast_ma')} / {ma_state.get('slow_ma')} (gap {ma_state.get('gap_pct', 0.0):+.2f}%) -> {'ON' if ma_state.get('filter_on') else 'OFF'}"
        )
        if mom_state.get("mom_value") is not None:
            lines.append(
                f"    MOM: {mom_state.get('mom_value'):+.2f}% -> {'ON' if mom_state.get('filter_on') else 'OFF'}"
            )
        lines.append(sep2)

    lines +=[
        "  PORTFOLIO OOS METRICS",
        f"  CAGR: {pm.get('CAGR', 0.0) * 100.0:+.2f}% | Sharpe: {pm.get('Sharpe', 0.0):.2f} | MaxDD: {pm.get('MaxDD', 0.0) * 100.0:+.2f}%",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row
# ---------------------------------------------------------------------------

def _build_log_row(snap: dict, action: str, asset_keys: list) -> dict:
    row = {
        "Date": snap["run_date"],
        "Action": action,
        "Mode": snap["portfolio_mode"],
        "Regime_ADX": snap.get("current_regime_adx"),
        "Realloc_Today": snap.get("realloc_today", False),
    }
    for k in asset_keys:
        row[f"signal_{k}"] = snap["signals"].get(k, "OUT")
        row[f"weight_{k}"] = snap["weights"].get(k, 0.0)
        
        # Add basic position tracking to CSV to match multiasset behavior
        pos = snap["asset_states"].get(k, {}).get("position")
        if pos:
            row[f"{k}_entry_date"] = pos["entry_date"]
            row[f"{k}_unrealised_pct"] = pos["unrealised_pct"]
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

    colors =["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, (key, ret) in enumerate(returns_dict.items()):
        ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
        if ret_oos.empty:
            continue
        bh = (1.0 + ret_oos).cumprod() * 100.0
        ax1.plot(
            bh.index, bh.values, color=colors[i % len(colors)], linewidth=0.9, alpha=0.65, label=key,
        )

    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    ax1.set_ylabel(ylabel="Cumulative return (log, base=100)")
    ax1.grid(visible=True, alpha=0.25)

    ax2 = axes[1]
    ax2.set_title(label="Per-asset signal state (shaded = in position)")

    y_offset = 0.0
    ytick_pos, ytick_labels = [],[]
    for key in asset_keys:
        sig = signals_oos_dict.get(key)
        if sig is None or sig.empty:
            continue
        ax2.fill_between(
            sig.index, y_offset, y_offset + sig.values * 0.8, alpha=0.55, step="post", label=key,
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

    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(fname=buf.name, dpi=72, bbox_inches="tight")
    plt.close(fig=fig)
    buf.close()
    atomic_write_bytes(path=chart_path, data=open(buf.name, "rb").read())
    os.unlink(path=buf.name)
    logging.info(msg=f"_build_chart: saved to {chart_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_daily_outputs(
    wf_results_dict: dict,
    wf_trades_dict: dict,
    price_df_dict: dict,
    portfolio_equity: pd.Series,
    portfolio_metrics: dict,
    weights_series: pd.Series,
    reallocation_log: list,
    bh_metrics_dict: dict,
    returns_dict: dict,
    signals_oos_dict: dict,
    asset_keys: list,
    portfolio_mode: str,
    fx_hedged: bool,
    output_dir: str = "outputs",
    asset_name: str = "GLOBAL_EQUITY",
    run_date: dt.date | None = None,
    gdrive_folder_id: str | None = None,
    gdrive_credentials: str | None = None,
) -> dict:
    
    if run_date is None:
        run_date = dt.date.today()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = asset_name.lower()
    logfile_name = f"{prefix}_signal_log.csv"
    log_path = out_dir / logfile_name
    status_path = out_dir / f"{prefix}_signal_status.txt"
    chart_path = out_dir / f"{prefix}_equity_chart.png"
    snapshot_path = out_dir / f"{prefix}_signal_snapshot.json"

    if gdrive_folder_id and gdrive_credentials:
        fetch_file_from_drive(
            local_path=log_path, 
            folder_id=gdrive_folder_id, 
            filename=logfile_name, 
            credentials_path=gdrive_credentials
        )
    else:
        logging.info(msg="global_equity_daily_output: skipping log pre-fetch.")

    snap = _build_snapshot(
        wf_results_dict=wf_results_dict,
        wf_trades_dict=wf_trades_dict,
        price_df_dict=price_df_dict,
        portfolio_equity=portfolio_equity,
        portfolio_metrics=portfolio_metrics,
        weights_series=weights_series,
        reallocation_log=reallocation_log,
        bh_metrics_dict=bh_metrics_dict,
        signals_oos_dict=signals_oos_dict,
        asset_keys=asset_keys,
        portfolio_mode=portfolio_mode,
        fx_hedged=fx_hedged,
        run_date=run_date,
    )

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
        asset_keys=asset_keys
    )
    logging.info(msg=f"\n{status_text}")
    atomic_write(path=status_path, content=status_text)
    logging.info(msg=f"build_daily_outputs: status written to {status_path}")

    log_row = _build_log_row(
        snap=snap, 
        action=action, 
        asset_keys=asset_keys
    )
    append_log_row(log_path=log_path, row=log_row)
    logging.info(msg=f"build_daily_outputs: log updated at {log_path}")

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    atomic_write(path=snapshot_path, content=json.dumps(snap_clean, indent=2, default=str))
    logging.info(msg=f"build_daily_outputs: snapshot written to {snapshot_path}")

    _build_chart(
        portfolio_equity=portfolio_equity,
        signals_oos_dict=signals_oos_dict,
        reallocation_log=reallocation_log,
        returns_dict=returns_dict,
        chart_path=chart_path,
        action=action,
        asset_keys=asset_keys,
        run_date=run_date,
        portfolio_mode=portfolio_mode,
        fx_hedged=fx_hedged,
    )

    if gdrive_folder_id and gdrive_credentials:
        logging.info(msg="Uploading artefacts to Google Drive...")
        try:
            from moj_system.data.gdrive import GDriveClient

            client = GDriveClient(credentials_path=gdrive_credentials)

            if client.service:
                files_to_upload =[log_path, status_path, chart_path, snapshot_path]
                for file_path in files_to_upload:
                    if file_path.exists():
                        client.upload_csv(
                            folder_id=gdrive_folder_id, 
                            local_path=str(file_path), 
                            filename=file_path.name
                        )
                logging.info(msg="Successfully uploaded all daily artefacts to Google Drive.")
            else:
                logging.warning(msg="Drive service unavailable. Artefacts saved locally only.")
        except Exception as e:
            logging.error(msg=f"Failed to upload artefacts to Drive: {e}")
    else:
        logging.info(msg="No GDrive credentials provided. Artefacts saved locally only.")

    return {
        "action": action,
        "status_text": status_text,
        "log_row": log_row,
        "chart_path": str(chart_path),
        "snapshot_path": str(snapshot_path),
        "log_path": str(log_path),
        "signals": snap["signals"],
        "weights": snap.get("weights", {}),
    }
