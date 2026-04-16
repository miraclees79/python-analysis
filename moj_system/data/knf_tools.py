# -*- coding: utf-8 -*-
"""
moj_system/data/knf_tools.py
============================
Unified toolkit for KNF API interaction and Stooq name matching.
Handles API hydration, TFI scoping, and fuzzy matching.
"""

import os
import re
import logging
import unicodedata
import time
import random
import pandas as pd
import requests
from pathlib import Path
from rapidfuzz import process, fuzz
from moj_system.data.gdrive import GDriveClient

# --- KNF CONFIGURATION ---
KNF_API_BASE = "https://wybieramfundusze-api.knf.gov.pl"
FUZZY_THRESHOLD = 75

# --- TFI REGISTRY (Wide scope for comprehensive matching) ---
TFI_IN_SCOPE = {
    "tfi allianz polska s.a.", "generali investments tfi s.a.", "goldman sachs tfi s.a.",
    "investors tfi s.a.", "pko tfi s.a.", "tfi pzu sa", "quercus tfi s.a.",
    "skarbiec tfi s.a.", "uniqa tfi s.a.", "vig/c-quadrat tfi s.a.", "agiofunds tfi s.a.",
    "opoka tfi s.a.", "alior tfi s.a.", "rockbridge tfi s.a.", "amundi polska tfi s.a.",
    "bnp paribas tfi s.a.", "caspar tfi s.a.", "santander tfi s.a.", "eques investment tfi s.a.",
    "esaliens tfi s.a.", "mtfi s.a.", "best tfi s.a.", "mci capital tfi s.a.",
    "millennium tfi s.a.", "pekao tfi s.a.", "ipopema tfi s.a.", "velofunds tfi s.a."
}

class KNFTools:
    def __init__(self, credentials_path=None):
        self.gdrive = GDriveClient(credentials_path)
        self.root_folder = self.gdrive.root_folder_id

    # -------------------------------------------------------------------------
    # Text Normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def to_ascii(s: str) -> str:
        if not isinstance(s, str): return ""
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")

    def normalize_name(self, name: str) -> str:
        """Removes legal forms and technical terms for better matching."""
        s = self.to_ascii(name.lower())
        legal_terms = [
            r"\bfundusz inwestycyjny otwarty\b", r"\bsfio\b", r"\bfio\b", r"\bfiz\b",
            r"\bspecjalistyczny\b", r"\bparasolowy\b", r"\bsubfundusz\b",
            r"\btowarzystwo funduszy inwestycyjnych\b", r"\btfi\b"
        ]
        for term in legal_terms:
            s = re.sub(term, " ", s)
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # -------------------------------------------------------------------------
    # KNF API Core Methods
    # -------------------------------------------------------------------------

    def _fetch_all_pages(self, path: str, params: dict = None) -> list:
        """Helper to fetch all records from any paged KNF endpoint."""
        items = []
        page = 0
        if params is None: params = {}
        
        while True:
            current_params = {**params, "size": 200, "page": page}
            resp = requests.get(f"{KNF_API_BASE}{path}", params=current_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            content = data.get('content', [])
            items.extend(content)
            
            page_info = data.get('page', {})
            if page + 1 >= page_info.get('totalPages', 1):
                break
            page += 1
            time.sleep(0.05)
        return items

    def fetch_active_subfunds(self) -> pd.DataFrame:
        """
        Fetches subfunds and hydrates them with details (category, classification).
        Required because the main list endpoint returns minimal data.
        """
        logging.info("Step 1: Building TFI and Fund lookup maps...")
        
        # 1. Map Company IDs to Names
        companies = self._fetch_all_pages("/v1/companies")
        company_map = {c['companyId']: c['name'] for c in companies if 'companyId' in c}
        
        # 2. Map Fund IDs to Company IDs
        funds = self._fetch_all_pages("/v1/funds")
        fund_to_company = {f['fundId']: f['companyId'] for f in funds if 'fundId' in f}

        # 3. Get basic Subfunds list
        logging.info("Step 2: Fetching subfund list...")
        subfunds_raw = self._fetch_all_pages("/v1/subfunds", {"includeInactive": "false"})
        
        # 4. Hydration loop (fetching details for each subfundId)
        logging.info(f"Step 3: Hydrating {len(subfunds_raw)} subfunds (fetching category/tfi details)...")
        hydrated_rows = []
        
        for i, s in enumerate(subfunds_raw):
            sfid = s['subfundId']
            if i > 0 and i % 50 == 0:
                logging.info(f"  ... hydrated {i}/{len(subfunds_raw)} subfunds")
            
            try:
                # Visiting each subfund individual endpoint to get full metadata
                detail_resp = requests.get(f"{KNF_API_BASE}/v1/subfunds/{sfid}", timeout=15)
                detail = detail_resp.json()
                
                # Attach TFI name using our lookup maps
                fund_id = detail.get('fundId')
                comp_id = fund_to_company.get(fund_id)
                tfi_name = company_map.get(comp_id, "")
                
                detail['tfi_name'] = tfi_name
                hydrated_rows.append(detail)
                time.sleep(0.05) # Polite delay to respect API
            except Exception as e:
                logging.warning(f"  Could not hydrate subfund {sfid}: {e}")
                hydrated_rows.append(s)

        df = pd.DataFrame(hydrated_rows)
        
        # 5. Filter by TFI Scope (defined at the top of the file)
        if 'tfi_name' in df.columns:
            df['tfi_key'] = df['tfi_name'].apply(lambda x: self.to_ascii(str(x).lower().strip()))
            df = df[df['tfi_key'].isin(TFI_IN_SCOPE)]
            logging.info(f"Successfully filtered to {len(df)} subfunds within scope.")
        
        return df

    # -------------------------------------------------------------------------
    # Matching and Reporting
    # -------------------------------------------------------------------------

    def match_against_stooq(self, knf_df: pd.DataFrame, stooq_titles_df: pd.DataFrame) -> pd.DataFrame:
        """Performs high-speed fuzzy matching using RapidFuzz."""
        if knf_df.empty or stooq_titles_df.empty: return pd.DataFrame()
        
        logging.info(f"Matching {len(knf_df)} subfunds against stooq records...")
        
        # Prepare pool
        stooq_titles_df['norm'] = stooq_titles_df['stooq_title'].apply(self.normalize_name)
        stooq_pool = stooq_titles_df['norm'].tolist()
        
        results = []
        for _, knf_row in knf_df.iterrows():
            knf_name = str(knf_row['name'])
            norm_knf = self.normalize_name(knf_name)
            
            match = process.extractOne(norm_knf, stooq_pool, scorer=fuzz.token_sort_ratio)
            
            if match and match[1] >= FUZZY_THRESHOLD:
                stooq_row = stooq_titles_df.iloc[match[2]]
                results.append({
                    "subfundId": knf_row.get('subfundId'),
                    "knf_name": knf_name,
                    "stooq_id": stooq_row['stooq_id'],
                    "stooq_title": stooq_row['stooq_title'],
                    "match_score": round(match[1], 1),
                    "category": knf_row.get('category')
                })
        
        return pd.DataFrame(results)

    def run_update_pipeline(self, confirmed_file="knf_stooq_confirmed.csv"):
        """Downloads confirmed list and fetches new candidates from KNF."""
        if not self.root_folder:
            logging.error("GDRIVE_FOLDER_ID not set. Cannot access confirmed list.")
            return pd.DataFrame()
            
        # 1. Get confirmed list from Drive
        df_confirmed = self.gdrive.download_csv(self.root_folder, confirmed_file)
        if df_confirmed is not None and 'subfundId' in df_confirmed.columns:
            confirmed_ids = set(pd.to_numeric(df_confirmed['subfundId'], errors='coerce').dropna().tolist())
        else:
            confirmed_ids = set()
        
        # 2. Fetch fresh, fully hydrated data from KNF
        df_fresh = self.fetch_active_subfunds()
        if df_fresh.empty: 
            return pd.DataFrame()
        
        # 3. Filter out those already confirmed
        df_new = df_fresh[~df_fresh['subfundId'].isin(confirmed_ids)].copy()
        logging.info(f"Found {len(df_new)} new subfunds to process.")
        
        return df_new