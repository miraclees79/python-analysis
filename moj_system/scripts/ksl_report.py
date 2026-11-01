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
import logging
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage

from moj_system.config import OUTPUT_DIR
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


def build_svg(result: PositionResult, width: int = 320, height: int = 70) -> str:
    """Mały wykres SVG: cena od wejścia i poziom KSL liczony od kroczącego maksimum."""
    if result.metrics is None:
        return ""
    history = result.metrics["price_history"]
    if len(history) < 2:
        return ""
    history = history.iloc[:: max(1, len(history) // 200)]
    if len(history) < 2:
        return ""
    ksl_line = history.cummax() * (1 - float(result.position["ksl_pct"]))
    low = float(min(history.min(), ksl_line.min()))
    span = (float(history.max()) - low) or 1.0
    count = len(history) - 1

    def points(values: list[float]) -> str:
        return " ".join(
            f"{index / count * width:.1f},{height - (v - low) / span * (height - 4) - 2:.1f}"
            for index, v in enumerate(values)
        )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="#1565c0" stroke-width="1.5" points="{points(history.tolist())}"/>'
        f'<polyline fill="none" stroke="#c62828" stroke-width="1" stroke-dasharray="3,2" '
        f'points="{points(ksl_line.tolist())}"/></svg>'
    )


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

    charts = "".join(
        f"<div style='display:inline-block;margin:6px'><div style='font-size:12px'>"
        f"{html.escape(str(r.position['account_label']))} — {html.escape(_label(r))}"
        f"</div>{build_svg(r)}</div>"
        for r in results
    )
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
        f"<h3>Wykresy od wejścia (niebieski: cena, czerwony: poziom KSL)</h3>{charts}"
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


def send_email(subject: str, text: str, html_body: str) -> None:
    host = os.environ.get("SMTP_HOST", "").strip()
    recipients = [a.strip() for a in os.environ.get("EMAIL_TO", "").split(",") if a.strip()]
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    if not host or not recipients:
        raise RuntimeError("Brak SMTP_HOST lub EMAIL_TO; ustaw sekrety albo użyj --no-email.")
    port = int(os.environ.get("SMTP_PORT", "").strip() or "587")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ.get("EMAIL_FROM", "").strip() or user
    message["To"] = ", ".join(recipients)
    message.set_content(text)
    message.add_alternative(html_body, subtype="html")
    message.add_attachment(
        html_body.encode("utf-8"), maintype="text", subtype="html", filename=REPORT_FILENAME
    )

    smtp_class = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with smtp_class(host, port, timeout=60) as smtp:
        if port != 465:
            smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.send_message(message)
    logging.info("Wysłano e-mail do %d odbiorców.", len(recipients))


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
    html_body = build_html(results, overall, refresh_results, generated_at)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / REPORT_FILENAME
    report_path.write_text(html_body, encoding="utf-8")
    logging.info("%s | pozycje: %d | raport: %s", subject, len(results), report_path)

    if upload and not client.upload_file(folder_id, str(report_path), REPORT_FILENAME):
        logging.error("Nie udało się wgrać %s na Google Drive.", REPORT_FILENAME)
        return 1
    if send:
        send_email(subject, build_text(results, overall, subject), html_body)
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
