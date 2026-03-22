# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
from strategy_test_library import (download_csv, 
                                   load_csv, 
                                   run_strategy_with_trades)
import tempfile
import datetime as dt
import time
import logging 

# Setup logging
LOG_FILE = "speedup.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Remove any existing handlers — critical in notebook environments
# where the kernel may have pre-attached handlers, and where re-running
# cells accumulates duplicate handlers
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logging.info("Logging initialised — file: %s", LOG_FILE)

# Get the temporary directory
tmp_dir = tempfile.gettempdir()
logging.info(f"Temporary directory: {tmp_dir}")
csv_filename_w20tr = os.path.join(tmp_dir, "w20tr.csv")

#download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr)
download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr)
INDEX_W20 = load_csv(csv_filename_w20tr)

df = INDEX_W20

# CASH series - Goldman Sachs Konserwatywny
csv_filename_cash = os.path.join(tmp_dir, "GS_konserw2720.csv")

# Download money market / bond fund
download_csv(
    "https://stooq.pl/q/d/l/?s=2720.n&i=d",   
    csv_filename_cash    
)

CASH = load_csv(csv_filename_cash)

kwargs = dict(
    price_col="Zamkniecie", X=0.10, Y=0.05,
    stop_loss=0.08, fast=75, slow=200,
    cash_df=CASH, position_mode="full",
)


logging.info("Timing single sample...")
t0 = time.time()
 
 
slow = run_strategy_with_trades(df, **kwargs, fast_mode=False)

single_time = time.time() - t0

logging.info(
        "Single sample SLOW: %.0fms ",
        single_time * 1000, 
        )

logging.info("Timing single sample...")
t0 = time.time()
 
 
fast = run_strategy_with_trades(df, **kwargs, fast_mode=True)


single_time = time.time() - t0

logging.info(
        "Single sample FAST: %.0fms ",
        single_time * 1000, 
        )
eq_slow = slow[0]["equity"].values
eq_fast = fast[0]["equity"].values
assert np.allclose(eq_slow, eq_fast, rtol=1e-10), "Equity curve MISMATCH"

tr_slow, tr_fast = slow[2], fast[2]
assert len(tr_slow) == len(tr_fast), "Trade count MISMATCH"
assert (tr_slow["Return"].values == tr_fast["Return"].values).all(), "Returns MISMATCH"
print("PASS — fast_mode results identical to slow_mode")


 
 
