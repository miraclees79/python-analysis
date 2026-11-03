# -*- coding: utf-8 -*-
"""
moj_system/scripts/ksl_report.py
================================
Dzienny, statyczny raport KSL (uruchamiany np. w GitHub Actions).

Przepływ: konfiguracja pozycji z Google Drive -> pobranie historii tylko
monitorowanych instrumentów do ulotnego katalogu -> obliczenia -> raport HTML
(zapis w outputs/ i na Drive) -> e-mail z raportem; temat zawiera status ALERT/WARNING.

Uruchomienie:
    python -m moj_system.scripts.ksl_report [--no-email] [--no-upload]

Zmienne środowiskowe e-mail: SMTP_HOST, SMTP_PORT (587 = STARTTLS, 465 = SSL),
SMTP_USER, SMTP_PASSWORD, EMAIL_FROM (opcjonalnie), EMAIL_TO (lista po przecinku).
"""

from __future__ import annotations

import argparse
import html
import io
import logging
import smtplib
import sys
from datetime import datetime, timezone

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from moj_system.config import OUTPUT_DIR
from moj_system.core import mailer
from moj_system.core.ksl_monitor import (
    PositionResult,
    RefreshResult,
    active_positions,
    default_data_dir,
    evaluate_positions,
    get_drive_client,
    overall_status,
    read_positions_from_drive,
    refresh_instruments,
    result_to_row,
)

REPORT_FILENAME = "ksl_report.html"
STATUS_COLORS = {
    "ALERT": "#c62828",
    "WARNING": "#ef6c00",
    "PROBLEM DANYCH": "#6d4c41",
    "OK": "#2e7d32",
}
TABLE_COLUMNS = [
    "Rachunek",
    "Instrument",
    "Status",
    "Data wejścia",
    "Ostatnie dane",
    "Wynik od wejścia (%)",
    "Wynik od maksimum (%)",
    "1D return (%)",
    "1M return (%)",
    "3M return (%)",
    "Świeżość",
    "Błąd",
]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.2f}" if value == value else ""
    return str(value)


def _label(result: PositionResult) -> str:
    return result.instrument.label if result.instrument else str(result.position["instrument_id"])


def build_svg(result: PositionResult, width: int = 480, height: int = 90) -> str:
    """Mały wykres SVG z ceną, poziomem KSL, ceną wejścia i skalą pionową."""
    if result.metrics is None:
        return ""
    full_history = result.metrics["price_history"]
    if len(full_history) < 2:
        return ""
    step = max(1, len(full_history) // 200)
    history = full_history.iloc[::step]
    if history.index[-1] != full_history.index[-1]:
        history = pd.concat([history, full_history.iloc[[-1]]])
    ksl_line = history.cummax() * (1 - float(result.position["ksl_pct"]))
    low = float(min(history.min(), ksl_line.min()))
    high = float(history.max())
    span = (high - low) or 1.0
    count = len(history) - 1
    plot_height = height - 18  # miejsce na daty pod wykresem
    plot_top = 12
    plot_bottom = plot_height - 2
    left = 48
    plot_width = width - left
    entry_price = float(result.metrics["entry_price"])

    def points(values: list[float]) -> str:
        return " ".join(
            f"{left + index / count * plot_width:.1f},"
            f"{plot_bottom - (v - low) / span * (plot_bottom - plot_top):.1f}"
            for index, v in enumerate(values)
        )

    def y_coordinate(value: float) -> float:
        return plot_bottom - (value - low) / span * (plot_bottom - plot_top)

    start = full_history.index[0].date().isoformat()
    end = full_history.index[-1].date().isoformat()
    text_y = height - 4
    scale = "".join(
        f'<line x1="{left}" y1="{y_coordinate(value):.1f}" x2="{width}" '
        f'y2="{y_coordinate(value):.1f}" stroke="#eee"/>'
        f'<text x="{left - 4}" y="{y_coordinate(value) + 3:.1f}" font-size="9" '
        f'fill="#555" text-anchor="end">{value:.2f}</text>'
        for value in (low + span * step / 4 for step in range(5))
    )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f"{scale}"
        f'<polyline fill="none" stroke="#1565c0" stroke-width="1.5" points="{points(history.tolist())}"/>'
        f'<polyline fill="none" stroke="#c62828" stroke-width="1" stroke-dasharray="3,2" '
        f'points="{points(ksl_line.tolist())}"/>'
        f'<line x1="{left}" y1="{y_coordinate(entry_price):.1f}" x2="{width}" '
        f'y2="{y_coordinate(entry_price):.1f}" stroke="#2e7d32" stroke-width="1.5"/>'
        f'<line x1="{left}" y1="{plot_height}" x2="{width}" y2="{plot_height}" stroke="#bbb"/>'
        f'<text x="{left}" y="{text_y}" font-size="11" fill="#555">{start}</text>'
        f'<text x="{width}" y="{text_y}" font-size="11" fill="#555" text-anchor="end">{end}</text></svg>'
    )


def build_png(result: PositionResult) -> bytes | None:
    """Renderuje wykres PNG, format obsługiwany przez klientów poczty, w tym Gmail."""
    if result.metrics is None:
        return None
    history = result.metrics["price_history"]
    if len(history) < 2:
        return None

    figure = Figure(figsize=(6.4, 2.4), dpi=120)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    ksl_line = history.cummax() * (1 - float(result.position["ksl_pct"]))
    axis.plot(history.index, history.to_numpy(), color="#1565c0", label="Cena")
    axis.plot(ksl_line.index, ksl_line.to_numpy(), color="#c62828", linestyle="--", label="KSL")
    axis.axhline(
        float(result.metrics["entry_price"]), color="#2e7d32", linewidth=1.5, label="Cena wejścia"
    )
    axis.margins(y=0.15)
    axis.set_ylabel("Cena")
    axis.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    axis.set_xticks([history.index[0], history.index[-1]])
    axis.grid(axis="y", color="#ddd", linewidth=0.7)
    axis.legend(loc="upper left", ncol=3, fontsize=8)
    figure.autofmt_xdate(rotation=0, ha="left")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


def build_email_charts(results: list[PositionResult]) -> tuple[dict[int, str], dict[str, bytes]]:
    """Buduje CID-y i obrazy inline dla wykresów w treści e-maila."""
    chart_cids: dict[int, str] = {}
    images: dict[str, bytes] = {}
    for index, result in enumerate(results):
        image = build_png(result)
        if image is not None:
            cid = f"ksl-chart-{index}@report"
            chart_cids[index] = cid
            images[cid] = image
    return chart_cids, images


def build_subject(overall: str, results: list[PositionResult], today: str) -> str:
    flagged = [r for r in results if r.status in ("ALERT", "WARNING")]
    if overall in ("ALERT", "WARNING") and flagged:
        names = ", ".join(f"{r.position['account_label']}/{_label(r)}" for r in flagged)
        return f"[KSL {overall}] {names} — {today}"
    if overall == "PROBLEM DANYCH":
        return f"[KSL] Problem z danymi — {today}"
    return f"[KSL] Raport dzienny: OK — {today}"


def build_html(
    results: list[PositionResult],
    overall: str,
    refresh_results: list[RefreshResult],
    generated_at: datetime,
    chart_cids: dict[int, str] | None = None,
) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in TABLE_COLUMNS)
    body: list[str] = []
    for result in results:
        row = result_to_row(result)
        cells = []
        for column in TABLE_COLUMNS:
            text = html.escape(_fmt(row.get(column)))
            if column == "Status":
                color = STATUS_COLORS.get(str(row.get(column)), STATUS_COLORS["PROBLEM DANYCH"])
                text = f'<b style="color:{color}">{text}</b>'
            cells.append(f"<td>{text}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")

    charts_list = []
    for index, result in enumerate(results):
        chart = (
            f"<img src='cid:{chart_cids[index]}' alt='Wykres: "
            f"{html.escape(_label(result))}' width='480' style='display:block;width:480px;max-width:100%'>"
            if chart_cids is not None and index in chart_cids
            else build_svg(result)
        )
        charts_list.append(
            f"<div style='margin:10px 0 18px'>"
            f"<div style='font-size:12px;line-height:18px;padding-bottom:10px'>"
            f"{html.escape(str(result.position['account_label']))} — {html.escape(_label(result))}"
            f"</div><div style='display:block;padding-top:4px'>{chart}</div></div>"
        )
    charts = "".join(charts_list)
    failures = [r for r in refresh_results if not r.ok]
    failures_html = (
        "<h3>Problemy z pobraniem danych</h3><ul>"
        + "".join(f"<li>{html.escape(f.instrument_id)}: {html.escape(f.detail)}</li>" for f in failures)
        + "</ul>"
        if failures
        else ""
    )
    color = STATUS_COLORS.get(overall, "#333")
    return (
        "<!DOCTYPE html><html lang='pl'><head><meta charset='utf-8'><title>KSL — raport dzienny</title>"
        "<style>body{font-family:Arial,sans-serif;font-size:13px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "th{background:#f2f2f2}</style></head><body>"
        "<h2>KSL — raport dzienny</h2>"
        f"<p>Status zbiorczy: <b style='color:{color}'>{html.escape(overall)}</b>"
        f" &middot; wygenerowano {generated_at:%Y-%m-%d %H:%M} UTC</p>"
        f"<table><tr>{head}</tr>{''.join(body)}</table>{failures_html}"
        f"<h3>Wykresy od wejścia (niebieski: cena, czerwony: KSL, zielony: cena wejścia)</h3>{charts}"
        "<p style='color:#777'>Prototyp: nie odtwarza wszystkich formuł KSL_Overview.</p></body></html>"
    )


def build_text(results: list[PositionResult], overall: str, subject: str) -> str:
    lines = [subject, f"Status zbiorczy: {overall}", ""]
    for result in results:
        row = result_to_row(result)
        line = (
            f"- {row['Rachunek']} | {row['Instrument']} | {row['Status']} | "
            f"od wejścia {_fmt(row.get('Wynik od wejścia (%)'))}% | "
            f"od maksimum {_fmt(row.get('Wynik od maksimum (%)'))}%"
        )
        if row.get("Błąd"):
            line += f" | {row['Błąd']}"
        lines.append(line)
    lines += ["", "Pełny raport w załączniku."]
    return "\n".join(lines)


def send_email(
    subject: str,
    text: str,
    html_body: str,
    inline_images: dict[str, bytes] | None = None,
) -> None:
    mailer.send_email(subject, text, html_body, inline_images, attachment_name=REPORT_FILENAME)


def run(send: bool, upload: bool) -> int:
    client, folder_id = get_drive_client()
    config = read_positions_from_drive(client, folder_id)
    positions = active_positions(config)

    data_dir = default_data_dir()
    needed_ids = {str(p["instrument_id"]) for p in positions}
    refresh_results = refresh_instruments(needed_ids, data_dir, client, folder_id) if needed_ids else []
    for failure in (r for r in refresh_results if not r.ok):
        logging.warning("Odświeżenie %s: %s", failure.instrument_id, failure.detail)

    results = evaluate_positions(positions, data_dir)
    overall = overall_status(results)
    generated_at = datetime.now(timezone.utc)
    subject = build_subject(overall, results, generated_at.strftime("%Y-%m-%d"))
    chart_cids, chart_images = build_email_charts(results) if send else ({}, {})
    html_body = build_html(
        results, overall, refresh_results, generated_at, chart_cids if send else None
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / REPORT_FILENAME
    report_path.write_text(html_body, encoding="utf-8")
    if sys.platform.startswith("win"):
        logging.info("%s | pozycje: %d | raport: %s", subject, len(results), report_path)
    else:
        # CI: log only the status tag, without position details
        logging.info("%s", subject.split("]")[0] + "]" if subject.startswith("[") else subject)

    if upload and not client.upload_file(folder_id, str(report_path), REPORT_FILENAME):
        logging.error("Nie udało się wgrać %s na Google Drive.", REPORT_FILENAME)
        return 1
    if send:
        send_email(subject, build_text(results, overall, subject), html_body, chart_images)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dzienny raport KSL (HTML + e-mail).")
    parser.add_argument("--no-email", action="store_true", help="Nie wysyłaj e-maila.")
    parser.add_argument("--no-upload", action="store_true", help="Nie wgrywaj raportu na Drive.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        return run(send=not args.no_email, upload=not args.no_upload)
    except (RuntimeError, FileNotFoundError, ValueError, OSError, smtplib.SMTPException) as exc:
        logging.error("Raport KSL nie powiódł się: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
