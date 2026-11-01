"""Private Streamlit dashboard for fund entry positions and KSL monitoring.

Run locally from any directory with:
    streamlit run moj_system/scripts/ksl_dashboard.py --server.address 127.0.0.1

The app reads and writes ``ksl_positions.json`` in the Google Drive folder
specified by ``GDRIVE_FOLDER_ID`` and downloads price histories of the monitored
instruments (and only them) into a volatile working directory (``KSL_DATA_DIR`` or
a temp folder). Keep the app private; do not deploy it with service-account
credentials to a public hosting provider.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

# Streamlit puts only the script directory on sys.path, so the project root is added explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from moj_system.config import OUTPUT_DIR  # noqa: E402
from moj_system.core.ksl_monitor import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    default_data_dir,
    evaluate_positions,
    get_drive_client,
    get_instrument_catalog,
    list_drive_instrument_ids,
    read_positions_from_drive,
    read_refresh_status,
    refresh_instruments,
    refresh_is_due,
    result_to_row,
    validate_positions_config,
    write_positions_to_drive,
)
from moj_system.data.gdrive import GDriveClient  # noqa: E402

REFRESH_MAX_AGE_HOURS = 6.0


def _load_streamlit() -> ModuleType:
    try:
        return importlib.import_module("streamlit")
    except ImportError as exc:
        raise RuntimeError("Brak Streamlit. Zainstaluj zależności z requirements.txt.") from exc


def main() -> None:
    st = _load_streamlit()
    st.set_page_config(page_title="KSL — pozycje funduszy", layout="wide")
    st.title("KSL — monitoring pozycji")
    st.caption(
        "Konfiguracja pozycji i historie cen są pobierane z prywatnego folderu Google Drive "
        "do ulotnego katalogu roboczego; wyniki liczone są lokalnie."
    )
    st.warning(
        "Prototyp oblicza stopę od wejścia i obsunięcie od maksimum od daty wejścia. "
        "Nie odtwarza jeszcze wszystkich kolumn/formuł KSL_Overview."
    )

    @st.cache_resource(show_spinner=False)
    def cached_client() -> tuple[GDriveClient, str]:
        return get_drive_client()

    @st.cache_data(ttl=300, show_spinner=False)
    def cached_config(_client: GDriveClient, folder_id: str) -> dict[str, Any]:
        return read_positions_from_drive(_client, folder_id)

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_drive_ids(_client: GDriveClient, folder_id: str) -> list[str]:
        return sorted(list_drive_instrument_ids(_client, folder_id))

    try:
        client, folder_id = cached_client()
    except RuntimeError as exc:
        cached_client.clear()
        st.error(f"Nie udało się połączyć z Google Drive: {exc}")
        return

    try:
        config = cached_config(client, folder_id)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            f"Przykładowy plik znajduje się w {OUTPUT_DIR.name}/ksl_positions.example.json. "
            "Po sprawdzeniu i skopiowaniu go do prywatnego folderu Drive odśwież dashboard."
        )
        return
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Nie udało się odczytać konfiguracji: {exc}")
        return

    data_dir = default_data_dir()
    positions = validate_positions_config(config)
    active_positions = [position for position in positions if position.get("enabled", True)]
    needed_ids = {str(position["instrument_id"]) for position in active_positions}

    with st.sidebar:
        st.header("Dane")
        force_refresh = st.button("Odśwież dane z Drive", use_container_width=True)
        status_placeholder = st.empty()

    auto_refresh = not st.session_state.get("auto_refresh_done") and refresh_is_due(
        data_dir, needed_ids, REFRESH_MAX_AGE_HOURS
    )
    if needed_ids and (force_refresh or auto_refresh):
        with st.spinner("Pobieranie danych z Google Drive…"):
            refresh_instruments(needed_ids, data_dir, client, folder_id)
            cached_drive_ids.clear()
        st.session_state["auto_refresh_done"] = True

    refresh_status = read_refresh_status(data_dir)
    if refresh_status:
        status_placeholder.caption(f"Ostatnie pobranie: {refresh_status['refreshed_at'][:19]} UTC")
        failed = [r for r in refresh_status["results"] if not r["ok"]]
        for item in failed:
            st.sidebar.warning(f"{item['instrument_id']}: {item['detail']}")
    else:
        status_placeholder.caption("Dane nie zostały jeszcze pobrane.")

    catalog = get_instrument_catalog(data_dir, extra_ids=frozenset(cached_drive_ids(client, folder_id)))

    if active_positions:
        results = evaluate_positions(active_positions, data_dir)
        rows = [result_to_row(result) for result in results]
        st.subheader("Pozycje")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        charted = {str(r.position["position_id"]): r for r in results if r.metrics is not None}
        if charted:
            st.subheader("Historia ceny/NAV od wejścia")
            chart_position_id = st.selectbox(
                "Pozycja na wykresie",
                list(charted),
                format_func=lambda key: (
                    f"{charted[key].position['account_label']} — {charted[key].instrument.label}"  # type: ignore[union-attr]
                ),
            )
            selected_result = charted[chart_position_id]
            history = selected_result.metrics["price_history"]  # type: ignore[index]
            rolling_peak = history.cummax()
            chart = pd.DataFrame(
                {
                    "Cena/NAV": history,
                    "Poziom KSL": rolling_peak * (1 - float(selected_result.position["ksl_pct"])),
                    "Poziom warning": rolling_peak * (1 - float(selected_result.position["warning_pct"])),
                }
            )
            st.line_chart(chart)

    st.subheader("Dodaj lub edytuj pozycję")
    position_ids = ["➕ Nowa pozycja"] + [str(item["position_id"]) for item in positions]
    selected_id = st.selectbox("Pozycja", position_ids)
    selected = next((item for item in positions if str(item["position_id"]) == selected_id), None)
    instrument_ids = sorted(catalog, key=lambda key: catalog[key].label.casefold())
    if selected and selected["instrument_id"] not in catalog:
        instrument_ids.append(selected["instrument_id"])
    labels = {
        key: catalog[key].label if key in catalog else f"{key} (nieznany instrument)"
        for key in instrument_ids
    }

    with st.form("position_form"):
        current_account = selected or {}
        account_id = st.text_input(
            "Id rachunku (stabilny klucz)",
            value=str(current_account.get("account_id", "")),
        )
        account_label = st.text_input(
            "Nazwa rachunku",
            value=str(current_account.get("account_label", "")),
        )
        chosen_instrument_id = st.selectbox(
            "Instrument",
            instrument_ids,
            index=(
                instrument_ids.index(selected["instrument_id"])
                if selected and selected["instrument_id"] in instrument_ids
                else 0
            ),
            format_func=lambda key: labels[key],
        )
        default_entry = date.fromisoformat(selected["entry_date"]) if selected else date.today()
        entry_date = st.date_input("Data wejścia", value=default_entry)
        ksl_pct = st.number_input(
            "Próg KSL (%)",
            min_value=0.1,
            max_value=99.9,
            value=float(current_account.get("ksl_pct", 0.08) * 100),
            step=0.5,
        )
        warning_pct = st.number_input(
            "Próg warning (%)",
            min_value=0.1,
            max_value=99.8,
            value=float(current_account.get("warning_pct", 0.05) * 100),
            step=0.5,
        )
        enabled = st.checkbox("Aktywna", value=bool(current_account.get("enabled", True)))
        submitted = st.form_submit_button("Zapisz konfigurację na Google Drive")

    if submitted:
        if not account_id.strip() or not account_label.strip():
            st.error("Uzupełnij identyfikator i nazwę rachunku.")
            return
        updated_positions = [dict(item) for item in positions]
        if selected:
            target_index = next(
                index
                for index, item in enumerate(updated_positions)
                if str(item["position_id"]) == selected_id
            )
            position_id = str(selected["position_id"])
        else:
            target_index = -1
            position_id = f"pos-{uuid.uuid4().hex[:12]}"

        updated_position = {
            "position_id": position_id,
            "account_id": account_id.strip(),
            "account_label": account_label.strip(),
            "instrument_id": chosen_instrument_id,
            "entry_date": entry_date.isoformat(),
            "ksl_pct": ksl_pct / 100,
            "warning_pct": warning_pct / 100,
            "enabled": enabled,
        }
        for index, item in enumerate(updated_positions):
            if (
                index != target_index
                and item.get("enabled", True)
                and enabled
                and item["account_id"] == updated_position["account_id"]
                and item["instrument_id"] == updated_position["instrument_id"]
            ):
                st.error(
                    "Ten rachunek ma już aktywną pozycję w tym instrumencie. "
                    "Wybierz ją do edycji albo użyj innego rachunku."
                )
                return

        if target_index >= 0:
            updated_positions[target_index] = updated_position
        else:
            updated_positions.append(updated_position)

        updated_config = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "positions": updated_positions,
        }
        try:
            write_positions_to_drive(updated_config)
        except (RuntimeError, ValueError) as exc:
            st.error(f"Nie udało się zapisać konfiguracji: {exc}")
            return
        cached_config.clear()
        st.session_state["auto_refresh_done"] = False
        st.success("Konfiguracja została zapisana na Google Drive.")
        st.rerun()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
