import argparse
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with config_path.open(
            mode="r",
            encoding="utf-8",
        ) as config_file:
            config_data: dict[str, Any] = yaml.safe_load(stream=config_file)
            return config_data
    except FileNotFoundError:
        print(f"BŁĄD: Plik konfiguracyjny '{config_path}' nie został znaleziony.")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"BŁĄD: Błąd podczas parsowania pliku YAML: {exc}")
        exit(1)

def create_repo_dump(
    config: dict[str, Any],
) -> None:
    """
    Przechodzi przez drzewo katalogów i tworzy jeden plik tekstowy
    z zawartością wybranych plików.
    """
    output_file_name: str = config.get("output_filename", "repo_do_analizy.txt")
    include_extensions: tuple[str, ...] = tuple(config.get("include_extensions", []))
    exclude_dirs: set[str] = set(config.get("exclude_dirs", []))
    exclude_files: set[str] = set(config.get("exclude_files", []))

    file_count: int = 0
    current_working_dir: Path = Path.cwd()
    output_path: Path = current_working_dir / output_file_name

    print(f"Rozpoczynam tworzenie zrzutu do pliku: {output_file_name}...")

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as outfile:
        for root, dirs, files in os.walk(top=current_working_dir):
            # Modyfikujemy dirs w miejscu, aby os.walk ignorował foldery
            dirs[:] = [
                d for d in dirs
                if d not in exclude_dirs
            ]

            for filename in files:
                if filename in exclude_files:
                    continue

                if filename.endswith(include_extensions):
                    file_to_process_path: Path = Path(root) / filename

                    print(f"  -> Dodaję plik: {file_to_process_path.relative_to(current_working_dir)}")

                    outfile.write(f"\n\n{'='*80}\n")
                    outfile.write(f"--- PLIK: {file_to_process_path.relative_to(current_working_dir)} ---\n")
                    outfile.write(f"{'='*80}\n\n")

                    try:
                        file_content: str = file_to_process_path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        )
                        outfile.write(file_content)
                        file_count += 1
                    except Exception as exc:
                        error_message: str = f"# BŁĄD: Nie udało się odczytać pliku '{filename}': {exc}\n"
                        outfile.write(error_message)
                        print(f"  [!] {error_message.strip()}")

    print("\n" + "="*30)
    print("ZAKOŃCZONO!")
    print(f"Przetworzono {file_count} plików.")
    print(f"Cały kod znajduje się w pliku: {output_file_name}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tworzy zrzut kodu repozytorium do jednego pliku tekstowego.",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Ścieżka do pliku konfiguracyjnego YAML (domyślnie: config.yml)",
    )
    args: argparse.Namespace = parser.parse_args()

    path_to_config: Path = Path(args.config)

    loaded_config_dict: dict[str, Any] = load_config(
        config_path=path_to_config,
    )

    create_repo_dump(
        config=loaded_config_dict,
    )
