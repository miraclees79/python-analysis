# -*- coding: utf-8 -*-
"""
moj_system/scripts/fund_data_refresher.py
=========================================
Script to refresh KNF fund data.
"""

import os
import sys
import logging
import tempfile

# --- Path Setup ---
from moj_system.config import OUTPUT_DIR
from moj_system.data.updater import DataUpdater


def main() -> None:
    """Główna funkcja odświeżająca dane funduszy."""
    
    # 1. Tworzenie ścieżki logu za pomocą pathlib
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = OUTPUT_DIR / "fund_refresh.log"
    
    # Czyszczenie handlerów, by zapobiec podwójnemu logowaniu
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(hdlr=handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file_path, mode="w", encoding="utf-8"), 
            logging.StreamHandler(stream=sys.stdout)
        ]
    )

    creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    
    logging.info(msg="Updating all KNF fund data (API + ZIP)...")
    
    # Jawne przekazywanie argumentów do DataUpdater
    updater = DataUpdater(
        gdrive_folder_id=folder_id, 
        credentials_path=creds_path
    )
    updater.run_full_update(get_funds=True)


# --- STRAŻNIK (MAIN GUARD) ---
if __name__ == "__main__":
    main()