# -*- coding: utf-8 -*-
"""
moj_system/scripts/refresh_knf.py
=================================
Executable script to refresh KNF subfunds list, find new funds,
and match them automatically with Stooq records using Price Verification.
Can also run in a fast 'verify_only' mode.
"""

import argparse
import logging
import os
import sys
import tempfile

# Set project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from moj_system.data.knf_tools import KNFTools


def main():
    parser = argparse.ArgumentParser(description="Refresh and Match KNF Subfunds")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Pobierz i sprawdz WSZYSTKIE TFI, ignorujac liste TFI_IN_SCOPE",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Pobiera tylko liste z knf_stooq_confirmed.csv i weryfikuje ich ceny. Nie szuka nowych.",
    )
    args = parser.parse_args()

    os.chdir(project_root)

    log_file = "outputs/refresh_knf.log"
    os.makedirs("outputs", exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    tools = KNFTools(credentials_path=creds_path)

    if args.verify_only:
        logging.info("!!! TRYB SZYBKIEJ WERYFIKACJI !!!")
        logging.info("Pobieram plik knf_stooq_confirmed.csv i sprawdzam dopasowanie cen...")
        tools.verify_confirmed_matches()
    else:
        if args.all:
            logging.info("!!! TRYB PEŁNY: Skanowanie wszystkich dostępnych TFI na rynku !!!")
        else:
            logging.info("Tryb ograniczony: Skanowanie tylko wybranych TFI (TFI_IN_SCOPE).")

        # Uruchamiamy normalny rurociąg poszukiwania
        tools.run_update_pipeline(use_tfi_scope=not args.all)


if __name__ == "__main__":
    main()
