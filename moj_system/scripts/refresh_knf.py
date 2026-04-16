# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:32:17 2026

@author: adamg
"""

import os
import sys
import logging

# Set project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from moj_system.data.knf_tools import KNFTools

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    tools = KNFTools()
    
    # 1. Fetch current list from KNF
    df_new = tools.run_update_pipeline()
    
    if df_new is not None and not df_new.empty:
        print("\nNOWE SUBFUNDUSZE DO PRZEGLĄDU:")
        print(df_new[['subfundId', 'name', 'category']].head(20).to_string(index=False))
        # Tutaj w przyszłości dodasz tools.match_against_stooq
    else:
        print("\nBrak nowych subfunduszy do dopasowania.")

if __name__ == "__main__":
    main()