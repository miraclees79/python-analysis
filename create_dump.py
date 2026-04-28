import argparse
import os

import yaml


def load_config(config_path):
    """Wczytuje konfigurację z pliku YAML."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"BŁĄD: Plik konfiguracyjny '{config_path}' nie został znaleziony.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"BŁĄD: Błąd podczas parsowania pliku YAML: {e}")
        exit(1)


def create_repo_dump(config):
    """
    Przechodzi przez drzewo katalogów i tworzy jeden plik tekstowy
    z zawartością wybranych plików.
    """
    output_file = config.get("output_filename", "repo_do_analizy.txt")
    extensions = tuple(config.get("include_extensions", []))
    exclude_dirs = set(config.get("exclude_dirs", []))
    exclude_files = set(config.get("exclude_files", []))

    file_count = 0

    print(f"Rozpoczynam tworzenie zrzutu do pliku: {output_file}...")

    with open(output_file, "w", encoding="utf-8") as outfile:
        # Przechodzimy przez wszystkie foldery i pliki, zaczynając od bieżącego katalogu
        for root, dirs, files in os.walk("."):
            # Modyfikujemy listę `dirs` w miejscu, aby os.walk ignorował te foldery
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                # Sprawdzamy, czy plik nie jest na liście wykluczonych
                if file in exclude_files:
                    continue

                # Sprawdzamy, czy plik ma odpowiednie rozszerzenie
                if file.endswith(extensions):
                    filepath = os.path.join(root, file)
                    print(f"  -> Dodaję plik: {filepath}")

                    outfile.write(f"\n\n{'=' * 80}\n")
                    outfile.write(f"--- PLIK: {filepath} ---\n")
                    outfile.write(f"{'=' * 80}\n\n")

                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
                            outfile.write(infile.read())
                        file_count += 1
                    except Exception as e:
                        error_message = f"# BŁĄD: Nie udało się odczytać pliku '{filepath}': {e}\n"
                        outfile.write(error_message)
                        print(f"  [!] {error_message.strip()}")

    print("\n" + "=" * 30)
    print("ZAKOŃCZONO!")
    print(f"Przetworzono {file_count} plików.")
    print(f"Cały kod znajduje się w pliku: {output_file}")
    print("=" * 30)


if __name__ == "__main__":
    # Używamy argparse, aby móc łatwo zmienić ścieżkę do pliku konfiguracyjnego
    parser = argparse.ArgumentParser(
        description="Tworzy zrzut kodu repozytorium do jednego pliku tekstowego.",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Ścieżka do pliku konfiguracyjnego YAML (domyślnie: config.yml)",
    )
    args = parser.parse_args()

    # Wczytujemy konfigurację i uruchamiamy główną funkcję
    config = load_config(args.config)
    create_repo_dump(config)
