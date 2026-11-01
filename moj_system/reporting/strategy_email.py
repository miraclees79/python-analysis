# -*- coding: utf-8 -*-
"""
moj_system/reporting/strategy_email.py
======================================
Raporty e-mail (HTML + wykres) dla pojedynczych aktywów i portfeli globalnych,
budowane ze snapshotów JSON (<prefix>_signal_snapshot.json) tak jak raport PENSION.
Wysyłka używa tych samych zmiennych środowiskowych co raport KSL (moj_system.core.mailer).

Funkcje nie są podpięte do codziennego uruchomienia; daily_runner wywołuje je wyłącznie z flagą --email.
"""

from __future__ import annotations

from pathlib import Path

from moj_system.reporting.pension_email import (
    ACTION_COLORS,
    DEFAULT_ACTION_COLOR,
    _esc,
    _kv_table,
    _num,
    _pct_weight,
    _position_section,
    _table,
    send_snapshot_report,
)

STYLE = (
    "<style>body{font-family:Arial,sans-serif;font-size:13px}table{border-collapse:collapse;"
    "margin-bottom:6px}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
    "th{background:#f2f2f2}</style>"
)


def _page(title: str, sections: list[str]) -> str:
    return (
        f"<!DOCTYPE html><html lang='pl'><head><meta charset='utf-8'><title>{_esc(title)}</title>"
        f"{STYLE}</head><body>{''.join(sections)}</body></html>"
    )


def _chart_section(chart_src: str | None, label: str) -> str:
    if not chart_src:
        return ""
    return (
        f"<h3>Wykres</h3><img src='{chart_src}' alt='Wykres {_esc(label)}' width='640' "
        "style='display:block;width:640px;max-width:100%'>"
    )


def _action_color(action: str) -> str:
    return ACTION_COLORS.get(action.upper(), DEFAULT_ACTION_COLOR)


def _on_off(state: dict, key: str = "filter_on") -> str:
    value = state.get(key)
    return "—" if value is None else ("ON" if value else "OFF")


def _metrics_section(title: str, metrics: dict) -> str:
    if not metrics:
        return ""
    return f"<h3>{_esc(title)}</h3>" + _kv_table([(k, _num(v, "{:.4f}")) for k, v in metrics.items()])


def _freshness_section(snapshot: dict) -> str:
    freshness = snapshot.get("data_freshness") or {}
    return "<h3>Świeżość danych</h3>" + _kv_table([(k, _esc(v)) for k, v in freshness.items()])


# --- Single asset ---------------------------------------------------------------------


def _single_position(snapshot: dict) -> dict | None:
    if snapshot.get("signal") != "IN":
        return None
    keys = (
        "entry_date", "entry_price", "days_in_trade", "unrealised_pct",
        "peak_price", "trail_stop", "abs_stop", "binding_stop", "stop_gap_pct",
    )
    return {key: snapshot.get(key) for key in keys} | {"today_price": snapshot.get("today_price")}


def single_subject(asset_name: str, snapshot: dict) -> str:
    return (
        f"[{asset_name}] {snapshot.get('action', 'brak akcji')} | "
        f"sygnał {snapshot.get('signal', '?')} — {snapshot.get('run_date', '')}"
    )


def build_single_html(asset_name: str, snapshot: dict, chart_src: str | None) -> str:
    action = str(snapshot.get("action", ""))
    ma = snapshot.get("ma_state") or {}
    mom = snapshot.get("mom_state") or {}
    filter_mode = str((snapshot.get("params") or {}).get("filter_mode", "ma")).upper()
    sections = [
        f"<h2>{_esc(asset_name)} — raport dzienny {_esc(snapshot.get('run_date'))}</h2>",
        f"<p>Akcja: <b style='color:{_action_color(action)}'>{_esc(action)}</b> &middot; "
        f"sygnał: <b>{_esc(snapshot.get('signal'))}</b> &middot; "
        f"cena: <b>{_num(snapshot.get('today_price'))}</b> &middot; "
        f"reżim ADX: <b>{_esc(str(snapshot.get('current_regime_adx', 'N/A')).upper())}</b></p>",
        _chart_section(chart_src, asset_name),
        _position_section(
            f"Pozycja ({asset_name})", _single_position(snapshot), snapshot.get("params") or {},
            ma, snapshot.get("atr_val_equity"),
        ),
        f"<p>Filtr aktywny: <b>{_esc(filter_mode)}</b> ({_on_off(snapshot, 'active_filter_on')}); "
        f"momentum: {_num(mom.get('mom_value'), '{:+.2f}', '%')} → <b>{_on_off(mom)}</b></p>",
        _metrics_section("Metryki strategii (OOS)", snapshot.get("strategy_metrics") or {}),
        _metrics_section("Metryki buy & hold", snapshot.get("bh_metrics") or {}),
        _freshness_section(snapshot),
    ]
    return _page(f"{asset_name} — raport", sections)


def build_single_text(asset_name: str, snapshot: dict) -> str:
    return "\n".join(
        [single_subject(asset_name, snapshot), "", "Pełny raport w wiadomości HTML i załączniku."]
    )


def send_single_asset_report(asset_name: str, snapshot_path: Path, chart_path: Path) -> bool:
    return send_snapshot_report(
        snapshot_path,
        chart_path,
        label=asset_name,
        attachment_name=f"{asset_name.lower()}_report.html",
        subject_fn=lambda snap: single_subject(asset_name, snap),
        text_fn=lambda snap: build_single_text(asset_name, snap),
        html_fn=lambda snap, src: build_single_html(asset_name, snap, src),
    )


# --- Global portfolio -----------------------------------------------------------------


def global_subject(asset_name: str, snapshot: dict) -> str:
    signals = snapshot.get("signals") or {}
    summary = ", ".join(f"{k}: {v}" for k, v in signals.items())
    action = snapshot.get("action", "brak akcji")
    return f"[{asset_name}] {action} | {summary} — {snapshot.get('run_date', '')}"


def build_global_html(asset_name: str, snapshot: dict, chart_src: str | None) -> str:
    action = str(snapshot.get("action", ""))
    weights = snapshot.get("weights") or {}
    signals = snapshot.get("signals") or {}
    states = snapshot.get("asset_states") or {}
    keys = list(signals) or list(states)

    sections = [
        f"<h2>{_esc(asset_name)} — raport dzienny {_esc(snapshot.get('run_date'))}</h2>",
        f"<p>Akcja: <b style='color:{_action_color(action)}'>{_esc(action)}</b> &middot; "
        f"tryb: <b>{_esc(snapshot.get('portfolio_mode'))}</b> &middot; "
        f"zabezpieczenie FX: <b>{_esc(snapshot.get('fx_hedged'))}</b> &middot; "
        f"reżim ADX: <b>{_esc(str(snapshot.get('current_regime_adx', 'N/A')).upper())}</b></p>",
        "<h3>Sygnały i alokacja</h3>",
        _table(
            ["Składnik", "Sygnał", "Waga"],
            [[_esc(k), _esc(signals.get(k)), _pct_weight(weights.get(k))] for k in keys],
        ),
        _chart_section(chart_src, asset_name),
    ]
    for key in keys:
        state = states.get(key) or {}
        sections.append(
            _position_section(
                f"Pozycja: {key}", state.get("position"), state.get("params") or {},
                state.get("ma_state") or {}, state.get("atr_val"),
            )
        )
        position = state.get("position") or {}
        if position.get("currency"):
            sections.append(
                f"<p>Cena w {_esc(position['currency'])}: wejście "
                f"{_num(position.get('entry_price_original_currency'))}, "
                f"bieżąca {_num(position.get('today_price_original_currency'))}</p>"
            )

    last = snapshot.get("last_realloc")
    if last:
        weights_after = ", ".join(
            f"{k}: {_pct_weight(v)}" for k, v in (last.get("weights_after") or {}).items()
        )
        sections.append(
            "<h3>Ostatnia realokacja</h3>"
            + _kv_table(
                [
                    ("Data", _esc(last.get("date"))),
                    ("Powód", _esc(last.get("reason"))),
                    ("Wagi po", _esc(weights_after)),
                    ("Liczba realokacji", _esc(snapshot.get("n_reallocations"))),
                ]
            )
        )
    sections.append(_metrics_section("Metryki portfela (OOS)", snapshot.get("portfolio_metrics") or {}))
    for key, metrics in (snapshot.get("bh_metrics") or {}).items():
        sections.append(_metrics_section(f"Buy & hold: {key}", metrics))
    sections.append(_freshness_section(snapshot))
    return _page(f"{asset_name} — raport", sections)


def build_global_text(asset_name: str, snapshot: dict) -> str:
    return "\n".join(
        [global_subject(asset_name, snapshot), "", "Pełny raport w wiadomości HTML i załączniku."]
    )


def send_global_report(asset_name: str, snapshot_path: Path, chart_path: Path) -> bool:
    return send_snapshot_report(
        snapshot_path,
        chart_path,
        label=asset_name,
        attachment_name=f"{asset_name.lower()}_report.html",
        subject_fn=lambda snap: global_subject(asset_name, snap),
        text_fn=lambda snap: build_global_text(asset_name, snap),
        html_fn=lambda snap, src: build_global_html(asset_name, snap, src),
    )
