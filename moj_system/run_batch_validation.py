import logging
import subprocess
import sys
from pathlib import Path

# Konfiguracja logowania dla skryptu sterującego
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [MASTER] - %(message)s")


def run_batch_validation() -> None:
    python_exe = sys.executable
    # Upewniamy się, że ścieżka robocza to root projektu
    current_dir = Path.cwd()

    # Lista konfiguracji (każda jako lista stringów)
    configurations = [
        [
            "--mode",
            "GLOBAL",
            "--asset",
            "GLOBAL_B",
            "--train",
            "7",
            "--test",
            "2",
            "--stop",
            "atr",
            "--n_mc",
            "1000",
            "--n_boot",
            "500",
            "--weights_perturb",
        ]
    ]

    for config_args in configurations:
        # Budujemy pełną komendę do uruchomienia modułu
        command = [python_exe, "-m", "moj_system.scripts.validate_robustness"] + config_args

        logging.info(msg=f"--- STARTING: {' '.join(command)} ---")

        # Używamy Popen, aby mieć pełną kontrolę nad strumieniem wyjściowym
        process = subprocess.Popen(
            args=command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Łączymy błędy ze standardowym wyjściem
            text=True,
            bufsize=1,  # Buforowanie liniowe
            cwd=str(current_dir),
        )

        # Czytamy wyjście linia po linii i wypisujemy je natychmiast
        # To gwarantuje widoczność logów w konsoli Spydera
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()  # Wymuszamy opróżnienie bufora konsoli

        # Czekamy na zakończenie procesu
        return_code = process.wait()

        if return_code == 0:
            logging.info(msg="--- COMPLETED SUCCESSFULLY ---")
        else:
            logging.error(msg=f"--- FAILED WITH RETURN CODE {return_code} ---")


if __name__ == "__main__":
    run_batch_validation()
