# -*- coding: utf-8 -*-
"""
moj_system/reporting/pension_email.py
=====================================
Raport HTML dla portfela PENSION zbudowany z pliku pension_signal_snapshot.json
i wysyłany e-mailem z wykresem (PNG inline + osadzony w załączonym raporcie HTML).
Do wysyłki używa tych samych zmiennych środowiskowych co raport KSL (moj_system.core.mailer).
"""

from __future__ import annotations

import base64
import html
import json
import logging
from collections.abc import Callable
from pathlib import Path

from moj_system.core.mailer import email_is_configured, send_email

CHART_CID = "pension-chart@report"
ACTION_COLORS = {"HOLD": "#2e7d32", "NO_CHANGE": "#2e7d32"}
DEFAULT_ACTION_COLOR = "#c62828"


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: object, fmt: str = "{:.2f}", suffix: str = "") -> str:
    try:
        return fmt.format(float(value)) + suffix
    except (TypeError, ValueError):
        return "—"


def _pct_weight(value: object) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


PERCENT_METRICS = ("CAGR", "MaxDD", "Vol")


def _metrics_table(metrics: object) -> str:
    """CAGR, MaxDD i Vol (ułamki w snapshocie) jako procenty; pozostałe metryki bez zmian."""
    if not isinstance(metrics, dict) or not metrics:
        return ""
    rows = [
        (k, _num(float(v) * 100, "{:.2f}", "%"))
        if k in PERCENT_METRICS and _is_number(v)
        else (k, _num(v, "{:.4f}"))
        for k, v in metrics.items()
    ]
    return _kv_table(rows)


def _is_number(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _kv_table(pairs: list[tuple[str, str]]) -> str:
    return _table(["Parametr", "Wartość"], [[_esc(k), v] for k, v in pairs])


def _position_section(title: str, position: dict | None, params: dict, ma: dict, atr: object) -> str:
    if not position:
        body = "<p>Brak otwartej pozycji.</p>"
    else:
        trail = f"{_num(position.get('trail_stop'))} (szczyt {_num(position.get('peak_price'))})"
        if params.get("use_atr_stop") and atr is not None:
            trail += f", ATR {_num(atr, '{:.2f}', '%')}"
        body = _kv_table(
            [
                ("Data wejścia", _esc(position.get("entry_date"))),
                ("Cena wejścia", _num(position.get("entry_price"))),
                ("Cena bieżąca", _num(position.get("today_price"))),
                ("Dni w pozycji", _esc(position.get("days_in_trade"))),
                ("Wynik niezrealizowany", _num(position.get("unrealised_pct"), "{:+.2f}", "%")),
                ("Trailing stop", _esc(trail)),
                ("Stop bezwzględny", _num(position.get("abs_stop"))),
                (
                    "Stop wiążący",
                    f"{_num(position.get('binding_stop'))} "
                    f"(odległość {_num(position.get('stop_gap_pct'), '{:+.1f}', '%')})",
                ),
            ]
        )
    if ma:
        state = "ON" if ma.get("filter_on") else "OFF"
        body += (
            f"<p>Filtr MA: {_num(ma.get('fast_ma'))} / {_num(ma.get('slow_ma'))} "
            f"(różnica {_num(ma.get('gap_pct'), '{:+.2f}', '%')}) → <b>{state}</b></p>"
        )
    return f"<h3>{_esc(title)}</h3>{body}"


def build_subject(snapshot: dict) -> str:
    return (
        f"[PENSION] {snapshot.get('action', 'brak akcji')} | "
        f"equity {snapshot.get('signal_equity', '?')} / bond {snapshot.get('signal_bond', '?')} — "
        f"{snapshot.get('run_date', '')}"
    )


def build_html(snapshot: dict, chart_src: str | None) -> str:
    """Buduje raport HTML; chart_src to np. 'cid:...' albo data URI, None = bez wykresu."""
    action = str(snapshot.get("action", ""))
    color = ACTION_COLORS.get(action.upper(), DEFAULT_ACTION_COLOR)
    weights = snapshot.get("weights") or {}
    freshness = snapshot.get("data_freshness") or {}
    pe = snapshot.get("params_equity") or {}
    pb = snapshot.get("params_bond") or {}
    mom = snapshot.get("mom_state_equity") or {}

    sections = [
        f"<h2>PENSION — raport dzienny {_esc(snapshot.get('run_date'))}</h2>",
        f"<p>Akcja: <b style='color:{color}'>{_esc(action)}</b> &middot; "
        f"sygnał equity: <b>{_esc(snapshot.get('signal_equity'))}</b> &middot; "
        f"sygnał bond: <b>{_esc(snapshot.get('signal_bond'))}</b> &middot; "
        f"reżim ADX: <b>{_esc(str(snapshot.get('current_regime_adx', 'N/A')).upper())}</b></p>",
        "<h3>Aktualna alokacja</h3>",
        _table(
            ["Equity (WIG)", "Bond (TBSP)", "MMF"],
            [[_pct_weight(weights.get("equity")), _pct_weight(weights.get("bond")),
              _pct_weight(weights.get("mmf"))]],
        ),
    ]
    if chart_src:
        sections.append(
            "<h3>Wykres</h3>"
            f"<img src='{chart_src}' alt='Wykres PENSION' width='640' "
            "style='display:block;width:640px;max-width:100%'>"
        )
    sections.append(
        _position_section(
            "Pozycja equity (WIG)", snapshot.get("equity_position"), pe,
            snapshot.get("ma_state_equity") or {}, snapshot.get("atr_val_equity"),
        )
    )
    if mom:
        state = "ON" if mom.get("filter_on") else "OFF"
        mom_text = _num(mom.get("mom_value"), "{:+.2f}", "%")
        sections.append(f"<p>Filtr momentum: {mom_text} → <b>{state}</b></p>")
    sections.append(
        _position_section(
            "Pozycja obligacji (TBSP)", snapshot.get("bond_position"), pb,
            snapshot.get("ma_state_bond") or {}, snapshot.get("atr_val_bond"),
        )
    )

    last = snapshot.get("last_realloc")
    if last:
        sections.append(
            "<h3>Ostatnia realokacja</h3>"
            + _kv_table(
                [
                    ("Data", _esc(last.get("date"))),
                    (
                        "Equity / Bond / MMF po",
                        f"{_pct_weight(last.get('equity_after'))} / {_pct_weight(last.get('bond_after'))}"
                        f" / {_pct_weight(last.get('mmf_after'))}",
                    ),
                    ("Powód", _esc(last.get("reason"))),
                ]
            )
        )

    for title, key in (
        ("Metryki portfela (OOS)", "portfolio_metrics"),
        ("Buy &amp; hold equity (OOS)", "bh_equity_metrics"),
    ):
        block = _metrics_table(snapshot.get(key))
        if block:
            sections.append(f"<h3>{title}</h3>{block}")

    ec = snapshot.get("execution_check")
    if ec:
        sections.append(
            "<h3>Execution reality check (fundusze PPE)</h3>"
            + _kv_table(
                [
                    ("Symulacja do", _esc(ec.get("latest_date"))),
                    ("CAGR", _num(ec.get("cagr"), "{:+.2f}", "%")),
                    ("MaxDD", _num(ec.get("maxdd"), "{:+.2f}", "%")),
                    ("Drag CAGR", _num(ec.get("cagr_drag_pp"), "{:+.2f}", " pp")),
                    ("Drag MaxDD", _num(ec.get("maxdd_drag_pp"), "{:+.2f}", " pp")),
                    ("Korelacja", _num(ec.get("correlation"))),
                ]
            )
        )

    sections.append(
        "<h3>Świeżość danych</h3>"
        + _kv_table([(k, _esc(v)) for k, v in freshness.items()])
    )
    return (
        "<!DOCTYPE html><html lang='pl'><head><meta charset='utf-8'><title>PENSION — raport</title>"
        "<style>body{font-family:Arial,sans-serif;font-size:13px}table{border-collapse:collapse;"
        "margin-bottom:6px}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "th{background:#f2f2f2}</style></head><body>" + "".join(sections) + "</body></html>"
    )


def build_text(snapshot: dict) -> str:
    weights = snapshot.get("weights") or {}
    return "\n".join(
        [
            build_subject(snapshot),
            f"Alokacja: equity {_pct_weight(weights.get('equity'))}, "
            f"bond {_pct_weight(weights.get('bond'))}, mmf {_pct_weight(weights.get('mmf'))}",
            "",
            "Pełny raport w wiadomości HTML i załączniku.",
        ]
    )


def send_snapshot_report(
    snapshot_path: Path,
    chart_path: Path,
    *,
    label: str,
    attachment_name: str,
    subject_fn: Callable[[dict], str],
    text_fn: Callable[[dict], str],
    html_fn: Callable[[dict, str | None], str],
) -> bool:
    """Wysyła raport ze snapshotu; zwraca False (bez wyjątku) przy braku konfiguracji lub błędzie."""
    if not email_is_configured():
        logging.info("%s: brak SMTP_HOST/EMAIL_TO — pomijam wysyłkę e-mail.", label)
        return False
    try:
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        chart = Path(chart_path).read_bytes() if Path(chart_path).is_file() else None
        inline = {CHART_CID: chart} if chart else None
        data_uri = f"data:image/png;base64,{base64.b64encode(chart).decode()}" if chart else None
        send_email(
            subject_fn(snapshot),
            text_fn(snapshot),
            html_fn(snapshot, f"cid:{CHART_CID}" if chart else None),
            inline,
            attachment_name=attachment_name,
            attachment_html=html_fn(snapshot, data_uri),
        )
    except Exception as exc:  # noqa: BLE001 - błąd odczytu/SMTP nie może przerwać przebiegu strategii
        logging.error("%s: wysyłka e-mail nie powiodła się: %s", label, exc)
        return False
    return True


def send_pension_report(snapshot_path: Path, chart_path: Path) -> bool:
    return send_snapshot_report(
        snapshot_path,
        chart_path,
        label="PENSION",
        attachment_name="pension_report.html",
        subject_fn=build_subject,
        text_fn=build_text,
        html_fn=build_html,
    )
