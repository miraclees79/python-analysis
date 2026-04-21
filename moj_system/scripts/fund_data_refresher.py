# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:05:08 2026

@author: adamg
"""

import os
import sys
import logging

import tempfile
import pandas as pd
import numpy as np
from datetime import datetime as dt



# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from moj_system.data.updater import DataUpdater
from moj_system.data.gdrive import GDriveClient

os.chdir(project_root)

log_file = "fund_refresh.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)

creds_path = os.path.join(tempfile.gettempdir(), "credentials.json")
gdrive = GDriveClient(credentials_path=creds_path)
folder_id = os.environ.get("GDRIVE_FOLDER_ID")

logging.info("Updating all KNF fund data (API + ZIP)...")
updater = DataUpdater(credentials_path=creds_path)
updater.run_full_update(get_funds=True)