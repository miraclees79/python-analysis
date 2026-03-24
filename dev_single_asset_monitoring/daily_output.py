"""
daily_output.py
===============
Daily operational output module for the WIG20TR single-asset trend-following
strategy (v0.2).

Called at the end of the main runfile after walk_forward has completed.
Produces three output artefacts:

  1. STATUS BLOCK   — human-readable text summary of today's signal state,
                      printed to stdout and written to signal_status.txt.
                      Designed to be embedded verbatim in an email body.

  2. SIGNAL LOG     — signal_log.csv — one row per run date, appended.
                      Accumulates the living history of daily signal states.
                      Suitable for upload to Google Drive as an audit trail.

  3. CHART          — equity_chart.png — OOS equity curve vs B&H, with the
                      current open trade highlighted and MA filter state shown
                      in a subplot.  Overwritten on each daily run.

  4. SNAPSHOT JSON  — signal_snapshot.json — today's state in machine-readable
                      form.  Overwritten daily.  Suitable for a Google Sheet
                      IMPORTDATA formula or downstream automation.

PUBLIC API
----------
    from daily_output import build_daily_outputs

    # Call after walk_forward completes in the runfile:
    outputs = build_daily_outputs(
        wf_equity   = wf_equity,
        wf_trades   = wf_trades,
        wf_metrics  = wf_metrics,
        wf_results  = wf_results,
        bh_equity   = bh_equity,
        bh_metrics  = bh_metrics,
        df          = df,             # full price DataFrame with 'Zamkniecie'
        output_dir  = "outputs",      # directory to write files into
        price_col   = "Zamkniecie",
        # Supply these in production so signal_log.csv is fetched from Drive
        # before action determination — required on GitHub Actions runners
        # where outputs/ is always empty at the start of each run.
        gdrive_folder_id   = os.getenv("GDRIVE_FOLDER_ID"),
        gdrive_credentials = "/tmp/credentials.json",
    )
    # outputs["action"] is "ENTER" | "EXIT" | "HOLD" — use in GH Actions env
    # outputs["status_text"] is the full human-readable block

DESIGN NOTES
------------
- No external dependencies beyond pandas, numpy, matplotlib, and json — all
  already present in the strategy environment.
- All file writes are atomic: written to a temp path then renamed, so a
  mid-write crash never leaves a corrupt file.
- The CARRY record in wf_trades (last row, Exit Reason == "CARRY") is the
  authoritative source for open-position state.  If absent the strategy is
  currently flat (OUT).
- MA filter state (fast MA vs slow MA) is recomputed from the raw price
  series using the parameters in the last walk-forward window.  This gives
  an early-warning indicator of an imminent signal flip even before the
  trailing stop or breakout conditions trigger.
- The signal log CSV uses ISO dates (YYYY-MM-DD) and floats rounded to 4dp
  so it loads cleanly into Google Sheets without locale issues.
- Stop level is the absolute stop: entry_price * (1 - stop_loss).
  Trailing stop level is computed as current_peak * (1 - X) where X is the
  trailing stop parameter from the active window.  Both are reported so the
  user always knows the binding exit trigger.
"""

import json
import logging
import os
import tempfile
import datetime as dt
from pathlib import Path

# Google Drive imports — optional; only required when gdrive_folder_id is supplied.
# Wrapped in try/except so the module loads cleanly in environments without the
# google-api packages installed (e.g. local dev without Drive credentials).
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload
    import io as _io
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}


def _atomic_write(path: Path, content: str) -> None:
    """Write text to a temp file then rename — avoids partial-write corruption."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def _fetch_log_from_drive(
    log_path: Path,
    folder_id: str,
    credentials_path: str,
    logfile_name = "signal_log.csv"
) -> None:
    """
    Download signal_log.csv from Google Drive into log_path, overwriting any
    local copy.  Called at the start of build_daily_outputs so the action
    comparison (ENTER / EXIT / HOLD) always uses the authoritative Drive copy
    rather than whatever is (or isn't) in the ephemeral GitHub Actions workspace.

    If the file does not yet exist in Drive (first-ever run) the function logs
    a warning and returns without error — _load_existing_log will then return
    an empty DataFrame and the initial run will record ENTER if IN, HOLD if OUT.

    Requires google-auth and google-api-python-client in the environment.
    """
    if not _GDRIVE_AVAILABLE:
        logging.warning(
            "_fetch_log_from_drive: google-api packages not available; "
            "skipping Drive download.  Install google-auth and "
            "google-api-python-client to enable."
        )
        return

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = _gdrive_build("drive", "v3", credentials=creds)

        # Search for signal_log.csv in the target folder
        query = (
            f"name='{logfile_name}' "
            f"and '{folder_id}' in parents "
            f"and trashed=false"
        )
        results = service.files().list(
            q=query, spaces="drive", fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if not files:
            logging.warning(
                "_fetch_log_from_drive: signal_log.csv not found in Drive folder %s — "
                "this is expected on the first ever run.",
                folder_id,
            )
            return

        file_id = files[0]["id"]
        request  = service.files().get_media(fileId=file_id)
        buf      = _io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(buf.getvalue())
        logging.info(
            "_fetch_log_from_drive: downloaded signal_log.csv from Drive (%d bytes)",
            len(buf.getvalue()),
        )

    except Exception as exc:
        logging.error(
            "_fetch_log_from_drive: failed to download signal_log.csv — %s. "
            "Action determination will fall back to local file (may be stale or absent).",
            exc,
        )


def _get_active_window_params(wf_results: pd.DataFrame) -> dict:
    """
    Return the parameter dict from the last walk-forward window.
    This is the parameter set currently live — used to compute MA filter state
    and to display the stop-loss parameter.
    """
    if wf_results is None or wf_results.empty:
        return {}
    last = wf_results.iloc[-1]
    params = {
        "filter_mode": last.get("filter_mode", "ma"),
        "X":           float(last.get("X",         0.10)),
        "Y":           float(last.get("Y",         0.10)),
        "fast":        int(last.get("fast",         50)),
        "slow":        int(last.get("slow",        200)),
        "stop_loss":   float(last.get("stop_loss", 0.10)),
        "TestStart":   pd.Timestamp(last["TestStart"]),
        "TestEnd":     pd.Timestamp(last["TestEnd"]),
    }
    tv = last.get("target_vol")
    if tv is not None and pd.notna(tv) and str(tv).strip() not in ("", "N/A", "nan"):
        params["target_vol"] = float(tv)
    # mom_lookback — stored in wf_results when filter_mode is "mom"
    ml = last.get("mom_lookback")
    if ml is not None and pd.notna(ml) and str(ml).strip() not in ("", "N/A", "nan"):
        params["mom_lookback"] = int(float(ml))
    else:
        params["mom_lookback"] = 252   # library default
    return params


def _extract_open_position(wf_trades: pd.DataFrame) -> dict | None:
    """
    Return the most recent CARRY record as a dict, or None if flat.

    The CARRY record is appended by run_strategy_with_trades whenever a
    position is still open at the end of the last walk-forward window.
    Its ExitPrice is the last known price (i.e. today's close).
    """
    if wf_trades is None or wf_trades.empty:
        return None
    carry = wf_trades[wf_trades["Exit Reason"] == "CARRY"]
    if carry.empty:
        return None
    rec = carry.iloc[-1].to_dict()
    return rec


def _compute_ma_filter_state(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    price_col: str = "Zamkniecie",
) -> dict:
    """
    Recompute fast and slow MAs from the full price series.
    Returns today's values and the filter state (on / off).
    """
    prices = df[price_col].dropna()
    if len(prices) < slow:
        return {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}

    fast_ma = float(prices.rolling(fast).mean().iloc[-1])
    slow_ma = float(prices.rolling(slow).mean().iloc[-1])
    gap_pct = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0.0
    filter_on = fast_ma > slow_ma
    return {
        "fast_ma":   round(fast_ma, 2),
        "slow_ma":   round(slow_ma, 2),
        "filter_on": filter_on,
        "gap_pct":   round(gap_pct * 100, 2),   # in percent
    }


def _compute_mom_filter_state(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    price_col: str = "Zamkniecie",
) -> dict:
    """
    Recompute momentum filter state from the full price series.
    Momentum = price.shift(skip) / price.shift(lookback) - 1
    Filter is ON when MOM > 0.
    Uses the same formula as compute_momentum() in strategy_test_library.
    """
    prices = df[price_col].dropna()
    if len(prices) < lookback:
        return {"mom_value": None, "filter_on": None}

    # Replicate strategy_test_library.compute_momentum with shift(1) lag
    mom_series = prices.shift(skip) / prices.shift(lookback) - 1
    mom_value  = float(mom_series.iloc[-1]) if not pd.isna(mom_series.iloc[-1]) else None
    filter_on  = (mom_value > 0) if mom_value is not None else None
    return {
        "mom_value":  round(mom_value * 100, 2) if mom_value is not None else None,  # in %
        "filter_on":  filter_on,
    }


def _determine_action(
    prev_signal_log: pd.DataFrame,
    signal_today: str,
) -> str:
    """
    Compare today's signal to yesterday's to determine the required action.
    Returns "ENTER", "EXIT", or "HOLD".
    """
    if prev_signal_log is None or prev_signal_log.empty:
        return "ENTER" if signal_today == "IN" else "HOLD"

    prev_signal = str(prev_signal_log["signal"].iloc[-1]).upper()
    if prev_signal == signal_today:
        return "HOLD"
    return "ENTER" if signal_today == "IN" else "EXIT"


def _load_existing_log(log_path: Path) -> pd.DataFrame:
    """Load existing signal log CSV, or return empty DataFrame."""
    if not log_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(log_path, parse_dates=["date"])
        return df
    except Exception as exc:
        logging.warning("Could not load existing signal log: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Core snapshot builder
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
    """
    Assemble the full snapshot dict that drives all output artefacts.
    All derived values are computed exactly once here.
    """
    snap = {"run_date": run_date.isoformat()}

    # --- Active window params ---
    params = _get_active_window_params(wf_results)
    snap["params"] = params

    # --- Price context ---
    prices = df[price_col].dropna()
    today_price = float(prices.iloc[-1]) if len(prices) > 0 else None
    snap["today_price"] = today_price

    # --- Open position ---
    open_pos = _extract_open_position(wf_trades)
    snap["open_position"] = open_pos

    if open_pos is not None:
        snap["signal"] = "IN"
        entry_price  = float(open_pos["EntryPrice"])
        entry_date   = pd.Timestamp(open_pos["EntryDate"]).date().isoformat()
        unreal_ret   = float(open_pos["Return"])
        days_in      = int(open_pos.get("Days", 0))
        peak_price   = today_price  # conservative: peak is at least current price
        # Attempt to recover running peak from equity curve in-trade slice
        if wf_equity is not None and not wf_equity.empty:
            trade_start = pd.Timestamp(open_pos["EntryDate"])
            in_trade    = wf_equity.loc[wf_equity.index >= trade_start]
            if not in_trade.empty:
                # Peak of equity implies peak of price (full position)
                # Use price series directly for accuracy
                price_in_trade = prices.loc[prices.index >= trade_start]
                if not price_in_trade.empty:
                    peak_price = float(price_in_trade.max())

        X          = params.get("X", 0.10)
        stop_loss  = params.get("stop_loss", 0.10)
        trail_stop = round(peak_price * (1 - X), 2)
        abs_stop   = round(entry_price * (1 - stop_loss), 2)
        # Binding stop is the higher of the two (less adverse)
        binding_stop = max(trail_stop, abs_stop)

        snap["entry_date"]    = entry_date
        snap["entry_price"]   = round(entry_price, 2)
        snap["days_in_trade"] = days_in
        snap["unrealised_pct"]= round(unreal_ret * 100, 2)
        snap["peak_price"]    = round(peak_price, 2)
        snap["trail_stop"]    = trail_stop
        snap["abs_stop"]      = abs_stop
        snap["binding_stop"]  = binding_stop
        if today_price:
            snap["stop_gap_pct"] = round((binding_stop - today_price) / today_price * 100, 2)
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

    # --- MA filter state ---
    if params:
        ma_state = _compute_ma_filter_state(df, params["fast"], params["slow"], price_col)
    else:
        ma_state = {"fast_ma": None, "slow_ma": None, "filter_on": None, "gap_pct": None}
    snap["ma_state"] = ma_state

    # --- Momentum filter state ---
    if params:
        mom_state = _compute_mom_filter_state(
            df,
            lookback  = params.get("mom_lookback", 252),
            skip      = 21,
            price_col = price_col,
        )
    else:
        mom_state = {"mom_value": None, "filter_on": None}
    snap["mom_state"] = mom_state

    # --- Active filter: which filter is actually governing entry/exit today ---
    # "ma" uses ma_state, "mom" uses mom_state; both modes report both for info.
    filter_mode = params.get("filter_mode", "ma") if params else "ma"
    if filter_mode == "mom":
        snap["active_filter_on"] = mom_state["filter_on"]
    else:
        snap["active_filter_on"] = ma_state["filter_on"]

    # --- OOS performance metrics ---
    snap["strategy_metrics"] = {
        k: round(float(v), 4) for k, v in (wf_metrics or {}).items()
    }
    snap["bh_metrics"] = {
        k: round(float(v), 4) for k, v in (bh_metrics or {}).items()
    }

    # --- Window schedule context ---
    if wf_results is not None and not wf_results.empty:
        last_w = wf_results.iloc[-1]
        snap["active_window"] = {
            "train_start": str(last_w.get("TrainStart", "")),
            "test_start":  str(last_w.get("TestStart", "")),
            "test_end":    str(last_w.get("TestEnd", "")),
        }
    else:
        snap["active_window"] = {}

    return snap


# ---------------------------------------------------------------------------
# Artefact 1: Status text block
# ---------------------------------------------------------------------------

def _build_status_text(snap: dict, action: str, asset_name) -> str:
    """
    Render the human-readable status block from a snapshot dict.
    Designed to be pasted verbatim into an email body.
    """
    sep  = "=" * 60
    sep2 = "-" * 60
    lines = [
        sep,
        f"  {asset_name} STRATEGY SIGNAL — {snap['run_date']}",
        sep,
        f"  Signal:      {snap['signal']}",
        f"  Action:      {action}",
        f"  Price today: {snap['today_price']}",
        sep2,
    ]

    fmode      = snap["params"].get("filter_mode", "ma")
    ma         = snap.get("ma_state", {})
    mom        = snap.get("mom_state", {})
    active_on  = snap.get("active_filter_on")
    act_icon   = "✓" if active_on else "✗"
    ma_icon    = "✓" if ma.get("filter_on") else "✗"
    mom_icon   = "✓" if mom.get("filter_on") else "✗"
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
# Artefact 2: Signal log CSV row
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
        "cagr":            round(sm.get("CAGR", 0) * 100, 4),
        "maxdd":           round(sm.get("MaxDD", 0) * 100, 4),
        "calmar":          round(sm.get("CalMAR", 0), 4),
        "sharpe":          round(sm.get("Sharpe", 0), 4),
    }


def _append_log_row(log_path: Path, row: dict) -> None:
    """Append one row to signal_log.csv, creating the file with headers if needed."""
    new_row_df = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        # Avoid duplicate entries for the same run_date
        existing = existing[existing["date"] != row["date"]]
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df
    tmp = log_path.with_suffix(".tmp")
    combined.to_csv(tmp, index=False)
    tmp.rename(log_path)


# ---------------------------------------------------------------------------
# Artefact 3: Equity chart
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
    """
    3-panel chart:
      Panel 1 (large): OOS equity curve vs B&H, trade shading, open trade annotation
      Panel 2:         Drawdown comparison
      Panel 3:         MA filter state (fast MA / slow MA ratio) with signal overlay
    """
    if wf_equity is None or wf_equity.empty:
        logging.warning("daily_output: wf_equity is empty — skipping chart")
        return

    wf_equity.index = pd.to_datetime(wf_equity.index)

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ---- Panel 1: equity curves ----
    ax1.plot(wf_equity.index, wf_equity.values,
             label="Strategy (OOS)", color="steelblue", linewidth=2)

    if bh_equity is not None and not bh_equity.empty:
        bh_equity.index = pd.to_datetime(bh_equity.index)
        bh_aligned = bh_equity.loc[
            (bh_equity.index >= wf_equity.index.min()) &
            (bh_equity.index <= wf_equity.index.max())
        ]
        ax1.plot(bh_aligned.index, bh_aligned.values,
                 label="Buy & Hold WIG20TR", color="grey",
                 linewidth=1.5, linestyle="--", alpha=0.8)

    # Shade WF windows
    if wf_results is not None and not wf_results.empty:
        for _, row in wf_results.iterrows():
            ax1.axvspan(row["TestStart"], row["TestEnd"], color="grey", alpha=0.05)

    # Shade closed trades
    BOUNDARY = {"CARRY", "SAMPLE_END"}
    if wf_trades is not None and not wf_trades.empty:
        closed = wf_trades[~wf_trades["Exit Reason"].isin(BOUNDARY)].copy()
        for _, trade in closed.iterrows():
            color = "green" if trade["Return"] > 0 else "red"
            ax1.axvspan(pd.Timestamp(trade["EntryDate"]),
                        pd.Timestamp(trade["ExitDate"]),
                        color=color, alpha=0.12)

    # Highlight open trade
    if snap["signal"] == "IN":
        entry_ts = pd.Timestamp(snap["entry_date"])
        ax1.axvspan(entry_ts, wf_equity.index[-1],
                    color="gold", alpha=0.25, label="Open trade")
        # Annotate binding stop as horizontal line
        if snap.get("binding_stop"):
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
    ax1.text(0.01, 0.97, summary, transform=ax1.transAxes,
             fontsize=8.5, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax1.set_title(
        f"WIG20TR Strategy v0.2 — Signal: {snap['signal']} | "
        f"Action: {snap.get('_action', '')} | {snap['run_date']}",
        fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("Equity (OOS, normalised)")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    # ---- Panel 2: drawdown ----
    dd_strat = wf_equity / wf_equity.cummax() - 1
    ax2.fill_between(dd_strat.index, dd_strat.values, 0,
                     color="steelblue", alpha=0.4, label="Strategy DD")
    if bh_equity is not None and not bh_equity.empty:
        dd_bh = bh_aligned / bh_aligned.cummax() - 1
        ax2.plot(dd_bh.index, dd_bh.values,
                 color="grey", linewidth=1, linestyle="--", alpha=0.7, label="B&H DD")
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(top=0)

    # ---- Panel 3: MA filter (fast/slow ratio) ----
    prices = df[price_col].dropna()
    prices.index = pd.to_datetime(prices.index)
    fast = snap["params"].get("fast", 50)
    slow = snap["params"].get("slow", 200)
    oos_start = wf_equity.index.min()

    fast_ma = prices.rolling(fast).mean().shift(1)
    slow_ma = prices.rolling(slow).mean().shift(1)
    ratio   = (fast_ma / slow_ma - 1) * 100  # in percent
    ratio_oos = ratio.loc[ratio.index >= oos_start]

    ax3.plot(ratio_oos.index, ratio_oos.values,
             color="darkorange", linewidth=1, label=f"fast({fast})/slow({slow})-1 [%]")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(ratio_oos.index, ratio_oos.values, 0,
                     where=(ratio_oos > 0), color="green", alpha=0.15, label="Filter ON")
    ax3.fill_between(ratio_oos.index, ratio_oos.values, 0,
                     where=(ratio_oos < 0), color="red", alpha=0.15, label="Filter OFF")
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
    _atomic_write_bytes(chart_path, open(buf.name, "rb").read())
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
    output_dir:  str  = "outputs",
    price_col:   str  = "Zamkniecie",
    logfile_name:   str  = "signal_log.csv",
    asset_name: str = 'WIG20TR',
    run_date:    dt.date | None = None,
    gdrive_folder_id:   str | None = None,
    gdrive_credentials: str | None = None,
) -> dict:
    """
    Build and write all daily output artefacts.

    Parameters
    ----------
    wf_equity   : stitched OOS equity curve (pd.Series, DatetimeIndex)
    wf_trades   : full trade log including CARRY records (pd.DataFrame)
    wf_metrics  : dict of OOS metrics from compute_metrics
    wf_results  : per-window DataFrame from walk_forward
    bh_equity   : buy-and-hold equity curve aligned to OOS period
    bh_metrics  : dict of B&H metrics
    df          : full price DataFrame (must contain price_col column)
    output_dir  : directory to write output files into (created if absent)
    price_col   : price column name in df
    logfile_name : correct asset-specific signal log name in Drive
    asset_name  : asset name for formatting
    run_date    : override today's date (default: dt.date.today())
    gdrive_folder_id   : Drive folder ID containing signal_log.csv.
                         When supplied (together with gdrive_credentials),
                         the existing log is downloaded from Drive before
                         action determination, ensuring ENTER/EXIT/HOLD is
                         correct on a fresh GitHub Actions runner.
    gdrive_credentials : path to service account JSON file (e.g. /tmp/credentials.json)

    Returns
    -------
    dict with keys:
        action        : "ENTER" | "EXIT" | "HOLD"
        signal        : "IN" | "OUT"
        status_text   : full human-readable status block (str)
        log_row       : dict of the CSV row appended today
        chart_path    : Path to the PNG chart
        snapshot_path : Path to the JSON snapshot
        log_path      : Path to the signal log CSV
    """
    if run_date is None:
        run_date = dt.date.today()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path      = out_dir / logfile_name

    # --- Fetch yesterday's signal log from Drive (before any local read) ---
    # Ensures ENTER/EXIT/HOLD is correct on a fresh GitHub Actions runner
    # where outputs/ is always empty at the start of each run.
    if gdrive_folder_id and gdrive_credentials:
        _fetch_log_from_drive(log_path, gdrive_folder_id, gdrive_credentials, logfile_name)
    else:
        logging.info(
            "daily_output: gdrive_folder_id/gdrive_credentials not supplied — "
            "skipping Drive log fetch; action determined from local file only."
        )

    status_path   = out_dir / "signal_status.txt"
    chart_path    = out_dir / "equity_chart.png"
    snapshot_path = out_dir / "signal_snapshot.json"

    # --- Build snapshot ---
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

    # --- Determine action (compare to yesterday's log entry) ---
    prev_log = _load_existing_log(log_path)
    action   = _determine_action(prev_log, snap["signal"])
    snap["_action"] = action   # underscore copy for chart title (stripped from JSON)
    snap["action"]  = action   # clean copy written to snapshot JSON for GH Actions

    # --- Status text ---
    status_text = _build_status_text(snap, action, asset_name)
    logging.info("\n%s", status_text)
    _atomic_write(status_path, status_text)
    logging.info("daily_output: status text written to %s", status_path)

    # --- Signal log ---
    log_row = _build_log_row(snap, action)
    _append_log_row(log_path, log_row)
    logging.info("daily_output: signal log updated at %s", log_path)

    # --- Snapshot JSON ---
    # Strip internal keys (leading underscore) but keep "action" which is
    # needed by the GitHub Actions parse step.
    snap_serialisable = {
        k: v for k, v in snap.items()
        if not k.startswith("_")
    }
    _atomic_write(snapshot_path, json.dumps(snap_serialisable, indent=2, default=str))
    logging.info("daily_output: JSON snapshot written to %s", snapshot_path)

    # --- Chart ---
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
        snap["signal"], action, run_date
    )

    return {
        "action":        action,
        "signal":        snap["signal"],
        "status_text":   status_text,
        "log_row":       log_row,
        "chart_path":    chart_path,
        "snapshot_path": snapshot_path,
        "log_path":      log_path,
    }
