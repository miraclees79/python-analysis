# -*- coding: utf-8 -*-
"""
moj_system/core/mailer.py
=========================
Wspólna wysyłka e-maili HTML (raport KSL, raport PENSION) przez SMTP.

Zmienne środowiskowe: SMTP_HOST, SMTP_PORT (587 = STARTTLS, 465 = SSL),
SMTP_USER, SMTP_PASSWORD, EMAIL_FROM (opcjonalnie), EMAIL_TO (lista po przecinku).
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage


def email_is_configured() -> bool:
    recipients = [a for a in os.environ.get("EMAIL_TO", "").split(",") if a.strip()]
    return bool(os.environ.get("SMTP_HOST", "").strip() and recipients)


def send_email(
    subject: str,
    text: str,
    html_body: str,
    inline_images: dict[str, bytes] | None = None,
    attachment_name: str = "report.html",
    attachment_html: str | None = None,
) -> None:
    """Wysyła wiadomość multipart: tekst + HTML z obrazami PNG inline (CID) + załącznik HTML."""
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
    html_part = message.get_payload()[-1]
    for cid, image in (inline_images or {}).items():
        html_part.add_related(
            image,
            maintype="image",
            subtype="png",
            cid=f"<{cid}>",
            disposition="inline",
        )
    message.add_attachment(
        (attachment_html if attachment_html is not None else html_body).encode("utf-8"),
        maintype="text",
        subtype="html",
        filename=attachment_name,
    )

    smtp_class = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with smtp_class(host, port, timeout=60) as smtp:
        if port != 465:
            smtp.starttls()
        if user:
            smtp.login(user, password)
        smtp.send_message(message)
    logging.info("Wysłano e-mail do %d odbiorców.", len(recipients))
