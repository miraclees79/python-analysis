# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:05:08 2026

@author: adamg
"""

import logging
import os
import sys
import tempfile

# --- Path Setup ---
from moj_system.config import OUTPUT_DIR
from moj_system.data.gdrive import GDriveClient
from moj_system.data.updater import DataUpdater

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUTPUT_DIR / "fund_refresh.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
)

creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
gdrive = GDriveClient(credentials_path=creds_path)
folder_id = os.environ.get("GDRIVE_FOLDER_ID")

logging.info("Updating all KNF fund data (API + ZIP)...")
updater = DataUpdater(credentials_path=creds_path)
updater.run_full_update(get_funds=True)
