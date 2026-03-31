"""
global_equity_daily_output.py
==============================
Builds and writes daily output artefacts for the global equity portfolio
strategy (both "global_equity" and "msci_world" modes).

Mirrors the pattern of multiasset_daily_output.py, adapted for an
N-asset portfolio with mode-specific asset labels.

Artefacts produced (all prefixed "global_equity_"):
  outputs/global_equity_signal_status.txt   — human-readable status block
  outputs/global_equity_signal_log.csv      — appended daily signal history
  outputs/global_equity_chart.png           — equity + signal chart
  outputs/global_equity_signal_snapshot.json

All file writes are atomic (write to .tmp then rename).

Drive pre-fetch:
  Before determining today's action (ENTER / EXIT / HOLD / REALLOC),
  the existing global_equity_signal_log.csv is downloaded from Drive
  so that ENTER/EXIT is always based on the authoritative persistent log
  rather than the empty ephemeral GitHub Actions workspace.

Public API:
  build_daily_outputs(...)  — call after Phase 5 (reporting) in the runfile.
  Returns a dict with keys: action, status_text, log_row,
                            chart_path, snapshot_path, log_path.
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

# Google Drive imports — optional; only required when gdrive_folder_id supplied.
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload
    import io as _io
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_FILE = "global_equity_signal_status.txt"
LOG_FILE    = "global_equity_signal_log.csv"
CHART_FILE  = "global_equity_chart.png"
SNAP_FILE   = "global_equity_signal_snapshot.json"


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Google Drive log pre-fetch
# ---------------------------------------------------------------------------

def _fetch_log_from_drive(
    log_path:         Path,
    folder_id:        str,
    credentials_path: str,
) -> None:
    """
    Download global_equity_signal_log.csv from Drive into log_path before
    action determination, so ENTER/EXIT is always based on the authoritative
    Drive copy rather than the empty ephemeral GH Actions workspace.
    """
    if not _GDRIVE_AVAILABLE:
        logging.warning("_fetch_log_from_drive: google-api packages not available.")
        return

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = _gdrive_build("drive", "v3", credentials=creds)

        q = (
            f"name='{LOG_FILE}' "
            f"and '{folder_id}' in parents "
            f"and trashed=false"
        )
        results = service.files().list(
            q=q, spaces="drive", fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if not files:
            logging.info(
                "_fetch_log_from_drive: %s not found in Drive — "
                "expected on first ever run.", LOG_FILE,
            )
            return

        file_id    = files[0]["id"]
        request    = service.files().get_media(fileId=file_id)
        buf        = _io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(buf.getvalue())
        logging.info(
            "_fetch_log_from_drive: downloaded %s (%d bytes).",
            LOG_FILE, len(buf.getvalue()),
        )

    except Exception as exc:
        logging.error(
            "_fetch_log_from_drive: failed — %s. "
            "Action determination will fall back to local file.", exc,
        )


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _get_signal_from_series(sig_oos: pd.Series) -> str:
    """Return 'IN' if the last OOS signal value is 1, else 'OUT'."""
    if sig_oos is None or sig_oos.empty:
        return "OUT"
    return "IN" if int(sig_oos.iloc[-1]) == 1 else "OUT"


def _get_current_weights(weights_series: pd.Series) -> dict:
    """Return the last weight dict from the weights_series."""
    if weights_series is None or weights_series.empty:
        return {}
    return dict(weights_series.iloc[-1])


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------

def _load_existing_log(log_path: Path) -> pd.DataFrame | None:
    """Load existing signal log CSV, return None if not found or empty."""
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
        return df if not df.empty else None
    except Exception:
        return None


def _determine_action(
    prev_log:        pd.DataFrame | None,
    signals_today:   dict,
    realloc_today:   bool,
    asset_keys:      list,
) -> str:
    """
    Determine today's action string.

    Returns one of:
      "HOLD"        — no signal or allocation change
      "ENTER_{key}" — asset `key` signal turned ON
      "EXIT_{key}"  — asset `key` signal turned OFF
      "REALLOC"     — allocation weights changed (signals unchanged)
      Compound actions separated by "+" e.g. "ENTER_WIG20TR+EXIT_TBSP"
    """
    if prev_log is None or prev_log.empty:
        return "HOLD"

    actions = []

    # Check per-asset signal changes
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
    portfolio_equity:  pd.Series,
    portfolio_metrics: dict,
    weights_series:    pd.Series,
    reallocation_log:  list,
    signals_oos_dict:  dict,
    asset_keys:        list,
    portfolio_mode:    str,
    fx_hedged:         bool,
    run_date:          dt.date,
) -> dict:
    """Build the snapshot dict for JSON serialisation."""
    current_weights = _get_current_weights(weights_series)

    # Per-asset current signal
    signals_today = {
        k: _get_signal_from_series(signals_oos_dict.get(k))
        for k in asset_keys
    }

    # Reallocation today?
    realloc_today = any(
        pd.Timestamp(r["Date"]).date() == run_date
        for r in reallocation_log
        if "Date" in r
    )

    # Portfolio level today
    port_today = None
    if portfolio_equity is not None and not portfolio_equity.empty:
        last_idx = portfolio_equity.index.max()
        port_today = float(portfolio_equity.iloc[-1])

    snap = {
        "run_date":        str(run_date),
        "portfolio_mode":  portfolio_mode,
        "fx_hedged":       fx_hedged,
        "signals":         signals_today,
        "weights":         current_weights,
        "realloc_today":   realloc_today,
        "portfolio_level": port_today,
        "portfolio_cagr":  portfolio_metrics.get("CAGR"),
        "portfolio_sharpe":portfolio_metrics.get("Sharpe"),
        "portfolio_maxdd": portfolio_metrics.get("MaxDD"),
        "portfolio_calmar":portfolio_metrics.get("CalMAR"),
        "oos_start":       str(portfolio_equity.index.min().date())
                           if portfolio_equity is not None and not portfolio_equity.empty
                           else None,
        "oos_end":         str(portfolio_equity.index.max().date())
                           if portfolio_equity is not None and not portfolio_equity.empty
                           else None,
        "n_reallocations": len(reallocation_log),
    }
    return snap


# ---------------------------------------------------------------------------
# Status text
# ---------------------------------------------------------------------------

def _build_status_text(
    snap:       dict,
    action:     str,
    asset_keys: list,
) -> str:
    """Build the human-readable status block."""
    lines = [
        "=" * 60,
        f"GLOBAL EQUITY PORTFOLIO  —  {snap['run_date']}",
        f"Mode: {snap['portfolio_mode']}  |  "
        f"FX: {'hedged' if snap['fx_hedged'] else 'unhedged'}",
        "=" * 60,
        f"ACTION: {action}",
        "-" * 60,
        "CURRENT SIGNALS:",
    ]
    for k in asset_keys:
        sig = snap["signals"].get(k, "OUT")
        lines.append(f"  {k:<14} {sig}")
    lines.append("-" * 60)
    lines.append("CURRENT WEIGHTS:")
    w = snap.get("weights", {})
    for k in asset_keys:
        wt = w.get(k, 0.0)
        lines.append(f"  {k:<14} {wt:.0%}")
    wt_mmf = w.get("mmf", 1.0 - sum(w.get(k, 0.0) for k in asset_keys))
    lines.append(f"  {'MMF':<14} {wt_mmf:.0%}")
    lines.append("-" * 60)
    cagr   = snap.get("portfolio_cagr")
    sharpe = snap.get("portfolio_sharpe")
    maxdd  = snap.get("portfolio_maxdd")
    calmar = snap.get("portfolio_calmar")
    lines.append("OOS PORTFOLIO METRICS:")
    lines.append(
        f"  CAGR   {cagr*100:.2f}%  |  "
        f"Sharpe {sharpe:.2f}  |  "
        f"MaxDD {maxdd*100:.2f}%  |  "
        f"CalMAR {calmar:.2f}"
        if all(v is not None for v in [cagr, sharpe, maxdd, calmar])
        else "  N/A"
    )
    lines.append(
        f"  OOS: {snap.get('oos_start')} to {snap.get('oos_end')}  "
        f"({snap.get('n_reallocations', 0)} reallocations)"
    )
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log row
# ---------------------------------------------------------------------------

def _build_log_row(
    snap:       dict,
    action:     str,
    asset_keys: list,
) -> dict:
    row = {
        "Date":   snap["run_date"],
        "Action": action,
        "Mode":   snap["portfolio_mode"],
    }
    for k in asset_keys:
        row[f"signal_{k}"] = snap["signals"].get(k, "OUT")
        row[f"weight_{k}"] = snap["weights"].get(k, 0.0)
    row["weight_mmf"] = snap["weights"].get(
        "mmf",
        max(0.0, 1.0 - sum(snap["weights"].get(k, 0.0) for k in asset_keys)),
    )
    row["portfolio_cagr"]   = snap.get("portfolio_cagr")
    row["portfolio_sharpe"] = snap.get("portfolio_sharpe")
    row["portfolio_maxdd"]  = snap.get("portfolio_maxdd")
    return row


def _append_log_row(log_path: Path, row: dict) -> None:
    """Append one row to the signal log CSV, creating headers if absent."""
    import csv
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _build_chart(
    portfolio_equity:  pd.Series,
    signals_oos_dict:  dict,
    reallocation_log:  list,
    bh_metrics_dict:   dict,
    returns_dict:      dict,
    chart_path:        Path,
    action:            str,
    asset_keys:        list,
    run_date:          dt.date,
    portfolio_mode:    str,
    fx_hedged:         bool,
) -> None:
    """Build and save the 2-panel equity + signal chart."""
    if portfolio_equity is None or portfolio_equity.empty:
        logging.warning("_build_chart: portfolio_equity is empty, skipping chart.")
        return

    oos_start = portfolio_equity.index.min()
    oos_end   = portfolio_equity.index.max()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        f"Global Equity Portfolio  [{portfolio_mode}]  "
        f"{'Hedged' if fx_hedged else 'Unhedged'}  "
        f"{oos_start.date()} → {oos_end.date()}  |  Action: {action}",
        fontsize=11,
    )

    # Panel 1: portfolio equity vs B&H
    ax1 = axes[0]
    ax1.set_title("Portfolio OOS equity (black) vs buy-and-hold per asset")
    ax1.set_yscale("log")

    port_norm = portfolio_equity / portfolio_equity.iloc[0] * 100
    ax1.plot(port_norm.index, port_norm.values, color="black", linewidth=2.0,
             label="Portfolio", zorder=10)

    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i, (key, ret) in enumerate(returns_dict.items()):
        ret_oos = ret.loc[(ret.index >= oos_start) & (ret.index <= oos_end)]
        if ret_oos.empty:
            continue
        bh = (1 + ret_oos).cumprod() * 100
        ax1.plot(bh.index, bh.values, color=colors[i % len(colors)],
                 linewidth=0.9, alpha=0.65, label=key)

    ax1.legend(fontsize=8, ncol=3, loc="upper left")
    ax1.set_ylabel("Cumulative return (log, base=100)")
    ax1.grid(True, alpha=0.25)

    # Panel 2: signal occupancy
    ax2 = axes[1]
    ax2.set_title("Per-asset signal state (shaded = in position)")

    y_offset = 0
    ytick_pos, ytick_labels = [], []
    for key in asset_keys:
        sig = signals_oos_dict.get(key)
        if sig is None or sig.empty:
            continue
        ax2.fill_between(sig.index, y_offset, y_offset + sig.values * 0.8,
                         alpha=0.55, step="post", label=key)
        ytick_pos.append(y_offset + 0.4)
        ytick_labels.append(key)
        y_offset += 1.0

    # Mark reallocation dates
    for r in reallocation_log:
        d = pd.Timestamp(r["Date"])
        if oos_start <= d <= oos_end:
            ax2.axvline(d, color="red", alpha=0.3, linewidth=0.6)

    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels(ytick_labels, fontsize=8)
    ax2.set_ylabel("Signal (shaded=IN)")
    ax2.set_xlabel(f"Run date: {run_date}")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    import tempfile as _tmpfile
    buf = _tmpfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(buf.name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.close()
    _atomic_write_bytes(chart_path, open(buf.name, "rb").read())
    os.unlink(buf.name)
    logging.info("_build_chart: saved to %s", chart_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_daily_outputs(
    # Walk-forward outputs per asset
    wf_results_dict:   dict,          # {asset_key: wf_results_df}
    # Portfolio outputs
    portfolio_equity:  pd.Series,
    portfolio_metrics: dict,
    weights_series:    pd.Series,
    reallocation_log:  list,
    # B&H benchmarks
    bh_metrics_dict:   dict,          # {asset_key: metrics_dict}
    # Return series (for chart)
    returns_dict:      dict,          # {asset_key: pd.Series}
    # Signals (OOS)
    signals_oos_dict:  dict,          # {asset_key: pd.Series}
    # Asset configuration
    asset_keys:        list,
    portfolio_mode:    str,
    fx_hedged:         bool,
    # Config
    output_dir:        str          = "outputs",
    run_date:          dt.date | None = None,
    gdrive_folder_id:  str | None   = None,
    gdrive_credentials: str | None  = None,
) -> dict:
    """
    Build and write all daily output artefacts for the global equity strategy.

    Called at the end of the daily runfile after Phase 5 (reporting).

    Parameters
    ----------
    wf_results_dict   : per-asset walk-forward results DataFrames
    portfolio_equity  : OOS portfolio equity curve
    portfolio_metrics : compute_metrics output for portfolio
    weights_series    : time-varying weight dicts
    reallocation_log  : list of reallocation event dicts
    bh_metrics_dict   : buy-and-hold metrics per asset
    returns_dict      : daily return series per asset
    signals_oos_dict  : OOS binary signals per asset
    asset_keys        : ordered list of asset keys
    portfolio_mode    : "global_equity" or "msci_world"
    fx_hedged         : FX treatment flag
    output_dir        : local directory for output files
    run_date          : override today's date
    gdrive_folder_id  : Drive folder ID for log pre-fetch
    gdrive_credentials: path to credentials.json

    Returns
    -------
    dict with keys: action, status_text, log_row,
                    chart_path, snapshot_path, log_path
    """
    if run_date is None:
        run_date = dt.date.today()

    out_dir       = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path      = out_dir / LOG_FILE
    status_path   = out_dir / STATUS_FILE
    chart_path    = out_dir / CHART_FILE
    snapshot_path = out_dir / SNAP_FILE

    # Pre-fetch Drive log before action determination
    if gdrive_folder_id and gdrive_credentials:
        _fetch_log_from_drive(log_path, gdrive_folder_id, gdrive_credentials)
    else:
        logging.info(
            "build_daily_outputs: Drive credentials not supplied — "
            "skipping log pre-fetch."
        )

    # Build snapshot
    snap = _build_snapshot(
        portfolio_equity  = portfolio_equity,
        portfolio_metrics = portfolio_metrics,
        weights_series    = weights_series,
        reallocation_log  = reallocation_log,
        signals_oos_dict  = signals_oos_dict,
        asset_keys        = asset_keys,
        portfolio_mode    = portfolio_mode,
        fx_hedged         = fx_hedged,
        run_date          = run_date,
    )

    # Determine action
    prev_log = _load_existing_log(log_path)
    action   = _determine_action(
        prev_log       = prev_log,
        signals_today  = snap["signals"],
        realloc_today  = snap.get("realloc_today", False),
        asset_keys     = asset_keys,
    )
    snap["action"] = action

    # Status text
    status_text = _build_status_text(snap, action, asset_keys)
    logging.info("\n%s", status_text)
    _atomic_write(status_path, status_text)
    logging.info("build_daily_outputs: status written to %s", status_path)

    # Signal log
    log_row = _build_log_row(snap, action, asset_keys)
    _append_log_row(log_path, log_row)
    logging.info("build_daily_outputs: log updated at %s", log_path)

    # Snapshot JSON
    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    _atomic_write(snapshot_path, json.dumps(snap_clean, indent=2, default=str))
    logging.info("build_daily_outputs: snapshot written to %s", snapshot_path)

    # Chart
    _build_chart(
        portfolio_equity = portfolio_equity,
        signals_oos_dict = signals_oos_dict,
        reallocation_log = reallocation_log,
        bh_metrics_dict  = bh_metrics_dict,
        returns_dict     = returns_dict,
        chart_path       = chart_path,
        action           = action,
        asset_keys       = asset_keys,
        run_date         = run_date,
        portfolio_mode   = portfolio_mode,
        fx_hedged        = fx_hedged,
    )

    return {
        "action":         action,
        "status_text":    status_text,
        "log_row":        log_row,
        "chart_path":     str(chart_path),
        "snapshot_path":  str(snapshot_path),
        "log_path":       str(log_path),
        "signals":        snap["signals"],
        "weights":        snap.get("weights", {}),
    }
