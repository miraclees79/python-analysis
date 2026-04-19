# -*- coding: utf-8 -*-
"""
moj_system/data/knf_tools.py
============================
Unified toolkit for KNF API interaction and Stooq name matching.
Features: 
- Brand-aware TFI filtering
- Advanced fuzzy matching via RapidFuzz
- GDrive integration
- PRICE VERIFICATION: Validates matches by comparing KNF NAV with Stooq NAV.
"""

import os
import re
import logging
import unicodedata
import time
import requests
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

from moj_system.data.gdrive import GDriveClient
from moj_system.data.data_manager import load_local_csv

# --- CONFIGURATION ---
KNF_API_BASE = "https://wybieramfundusze-api.knf.gov.pl"
FUZZY_THRESHOLD = 75  # Minimum score to consider a fuzzy match
PRICE_TOLERANCE = 0.05 # 5% tolerance for NAV comparison

SCRIPT_DIR = Path(__file__).resolve().parent
FUND_NAMES_DIR = SCRIPT_DIR / "fund_names_stooq"

# --- TFI REGISTRY (Wide Scope) ---
TFI_IN_SCOPE = {
    "tfi allianz polska s.a.", "generali investments tfi s.a.", "goldman sachs tfi s.a.",
    "investors tfi s.a.", "pko tfi s.a.", "tfi pzu sa", "quercus tfi s.a.",
    "skarbiec tfi s.a.", "uniqa tfi s.a.", "vig/c-quadrat tfi s.a.", "agiofunds tfi s.a.",
    "opoka tfi s.a.", "alior tfi s.a.", "rockbridge tfi s.a.", "amundi polska tfi s.a.",
    "bnp paribas tfi s.a.", "caspar tfi s.a.", "santander tfi s.a.", "eques investment tfi s.a.",
    "esaliens tfi s.a.", "mtfi s.a.", "best tfi s.a.", "mci capital tfi s.a.",
    "millennium tfi s.a.", "pekao tfi s.a.", "ipopema tfi s.a.", "velofunds tfi s.a."
}

# --- TFI BRAND OVERRIDES ---
TFI_BRAND_OVERRIDES = {
    "vig/c-quadrat tfi s.a.": "vig cq",
    "tfi allianz polska s.a.": "allianz",
    "agiofunds tfi s.a.": "agio",
    "agio funds tfi s.a.": "agio",
    "amundi polska tfi s.a.": "amundi",
    "generali investments tfi s.a.": "generali",
    "investors tfi s.a.": "investor",
}

class KNFTools:
    def __init__(self, credentials_path=None):
        self.gdrive = GDriveClient(credentials_path)
        self.root_folder = self.gdrive.root_folder_id

    # =========================================================================
    # TEXT NORMALIZATION
    # =========================================================================

    @staticmethod
    def to_ascii(s: str) -> str:
        if not isinstance(s, str): return ""
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")

    def tfi_brand_token(self, company_name: str) -> str:
        if not isinstance(company_name, str) or not company_name.strip(): return ""
        raw_key = self.to_ascii(company_name.strip().lower())
        if raw_key in TFI_BRAND_OVERRIDES: return TFI_BRAND_OVERRIDES[raw_key]
        
        s = raw_key
        structural = [
            r"\btowarzystwo funduszy inwestycyjnych\b", r"\bfunduszy inwestycyjnych\b",
            r"\binvestment(?:\s+partners)?\b", r"\basset\s+management\b", r"\btfi\b",
            r"\bs\.?\s*a\.?\b", r"\bsp\.?\s*z\.?\s*o\.?\s*o\.?\b"
        ]
        for term in structural: s = re.sub(term, " ", s)
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def residual_name(self, name: str, brand_token: str = "") -> str:
        if not isinstance(name, str): return ""
        s = self.to_ascii(name.lower())
        legal_terms = [
            r"\bfundusz inwestycyjny otwarty\b", r"\bfundusz inwestycyjny zamkniety\b",
            r"\bspecjalistyczny fundusz inwestycyjny otwarty\b", r"\bfundusz inwestycyjny\b",
            r"\bsfio\b", r"\bfio\b", r"\bfiz\b", r"\bsubfundusz\b", r"\bparasolowy\b", r"\bparasol\b",
            r"\btowarzystwo funduszy inwestycyjnych\b", r"\btfi\b"
        ]
        for term in legal_terms: s = re.sub(term, " ", s)
        if brand_token: s = re.sub(r"\b" + re.escape(brand_token) + r"\b", " ", s)
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # =========================================================================
    # DATA LOADING (KNF API & STOOQ FILES)
    # =========================================================================

    def load_stooq_txt_files(self) -> pd.DataFrame:
        if not FUND_NAMES_DIR.exists():
            logging.error(f"Directory not found: {FUND_NAMES_DIR}")
            return pd.DataFrame()

        txt_files = sorted(FUND_NAMES_DIR.glob("*.txt"))
        if not txt_files:
            logging.error(f"No .txt files found in {FUND_NAMES_DIR}")
            return pd.DataFrame()

        line_re = re.compile(r"^\s*(\d+)\.n\s+(.+)$", re.IGNORECASE)
        rows = []
        
        for txt_path in txt_files:
            category = txt_path.stem
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    m = line_re.match(line)
                    if m:
                        rows.append({
                            "stooq_id": int(m.group(1)),
                            "stooq_ticker": f"{m.group(1)}.N",
                            "stooq_title": m.group(2).strip(),
                            "stooq_category": category,
                        })
            except Exception as e:
                logging.warning(f"Could not read {txt_path.name}: {e}")

        df = pd.DataFrame(rows)
        if df.empty: return df
        
        df = df.drop_duplicates(subset=["stooq_id"], keep="first")
        
        def strip_prefix(title, sid):
            prefix = f"{sid}.n - "
            if title.lower().startswith(prefix): title = title[len(prefix):]
            title = re.sub(r"^\d+\.n\s*-\s*", "", title, flags=re.IGNORECASE)
            return re.sub(r"\s*-\s*stooq\s*$", "", title, flags=re.IGNORECASE).strip()
            
        df["stooq_name"] = df.apply(lambda r: strip_prefix(r["stooq_title"], r["stooq_id"]), axis=1)
        logging.info(f"Loaded {len(df)} Stooq names from {len(txt_files)} files.")
        return df

    def _fetch_all_pages(self, path: str, params: dict = None) -> list:
        items = []
        page = 0
        if params is None: params = {}
        while True:
            current_params = {**params, "size": 200, "page": page}
            resp = requests.get(f"{KNF_API_BASE}{path}", params=current_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get('content', []))
            if page + 1 >= data.get('page', {}).get('totalPages', 1): break
            page += 1
            time.sleep(0.05)
        return items

    def fetch_knf_subfunds(self, confirmed_ids: set, use_tfi_scope: bool = True) -> pd.DataFrame:
        """
        Fetches active subfunds from KNF, hydrates them, and excludes confirmed.
        If use_tfi_scope is True, it filters only the TFI brands listed in TFI_IN_SCOPE.
        If False, it returns ALL active subfunds from the KNF API.
        """
        logging.info("Fetching lookups from KNF...")
        companies = self._fetch_all_pages("/v1/companies")
        company_map = {c['companyId']: c['name'] for c in companies if 'companyId' in c}
        funds = self._fetch_all_pages("/v1/funds")
        fund_to_company = {f['fundId']: f['companyId'] for f in funds if 'fundId' in f}

        logging.info("Fetching subfund list...")
        subfunds_raw = self._fetch_all_pages("/v1/subfunds", {"includeInactive": "false", "includeLiquidating": "false"})
        
        unconfirmed_raw = [s for s in subfunds_raw if s['subfundId'] not in confirmed_ids]
        logging.info(f"Hydrating {len(unconfirmed_raw)} unconfirmed subfunds...")
        
        hydrated_rows = []
        for i, s in enumerate(unconfirmed_raw):
            sfid = s['subfundId']
            if i > 0 and i % 100 == 0: logging.info(f"  ... hydrated {i}/{len(unconfirmed_raw)}")
            try:
                detail = requests.get(f"{KNF_API_BASE}/v1/subfunds/{sfid}", timeout=10).json()
                fund_id = detail.get('fundId')
                tfi_name = company_map.get(fund_to_company.get(fund_id), "")
                detail['tfi_name'] = tfi_name
                hydrated_rows.append(detail)
                time.sleep(0.05)
            except Exception as e:
                logging.warning(f"Could not hydrate {sfid}: {e}")

        df = pd.DataFrame(hydrated_rows)
        if df.empty: return df
        
        df['knf_tfi_key'] = df['tfi_name'].apply(self.tfi_brand_token)
        df['tfi_key_raw'] = df['tfi_name'].apply(lambda x: self.to_ascii(str(x).lower().strip()))
        
        # --- ZMIANA TUTAJ: Warunkowe filtrowanie TFI ---
        if use_tfi_scope:
            df = df[df['tfi_key_raw'].isin(TFI_IN_SCOPE)].copy()
            logging.info(f"Filtered to {len(df)} subfunds within TFI scope.")
        else:
            logging.info(f"TFI scope filter disabled. Processing ALL {len(df)} subfunds.")
        # ----------------------------------------------

        # Exclude PPK (zawsze wykluczamy te programy emerytalne, bo ich i tak nie ma na Stooq)
        df = df[~df['category'].astype(str).str.startswith("PPK")]
        
        df['knf_norm'] = df.apply(lambda r: self.residual_name(str(r['name']), r['knf_tfi_key']), axis=1)
        logging.info(f"KNF list ready: {len(df)} subfunds to match.")
        return df


    def fetch_subfund_history(self, subfund_id: int) -> list:
        url = f"{KNF_API_BASE}/v1/subfunds/{subfund_id}/history?size=50&sort=validFrom,desc"
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
            if resp.status_code == 404: return []
            resp.raise_for_status()
            content = resp.json().get("content", [])
            former = set()
            for entry in content:
                if str(entry.get("validTo", "")) == "9999-12-31": continue
                name = entry.get("data", {}).get("name", "")
                if name: former.add(name)
            return list(former)
        except Exception:
            return []

    # =========================================================================
    # PRICE VERIFICATION
    # =========================================================================

    def _verify_price_match(self, subfund_id: int, stooq_id: int) -> bool:
        from moj_system.data.updater import DataUpdater # Import here to avoid circular dependency
        
        try:
            # 1. KNF: Get latest valuation (could be 1 month old)
            url = f"{KNF_API_BASE}/v1/valuations"
            resp = requests.get(url, params={"subfundId": subfund_id, "size": 1, "sort": "date,desc"}, timeout=10)
            resp.raise_for_status()
            knf_data = resp.json().get('content', [])
            
            if not knf_data: return False
            item = knf_data[0]
            if item.get('valuation') is None or item.get('date') is None: return False
            
            knf_date = pd.to_datetime(item['date']).normalize()
            knf_nav = float(item['valuation'])

            # 2. STOOQ: Force extraction of candidate's history from local ZIP
            logging.info(f"    [Verification] Extracting candidate {stooq_id} from ZIP...")
            updater = DataUpdater(credentials_path=self.gdrive.credentials_path)
            success = updater.update_ticker(
                label=f"fund_{stooq_id}", 
                stooq_ticker=f"{stooq_id}.n", 
                knf_id=None, # Stooq history only
                zip_type="fund_pl", 
                upload_to_drive=False
            )
            
            if not success:
                logging.warning(f"    [Rejected] Could not extract {stooq_id}.n from ZIP.")
                return False

            # 3. Load the newly extracted CSV
            stooq_df = load_local_csv(f"fund_{stooq_id}", f"Stooq_{stooq_id}", mandatory=False)
            if stooq_df is None or stooq_df.empty: return False 
            stooq_df.index = pd.to_datetime(stooq_df.index).tz_localize(None)

            # 4. DATE ALIGNMENT
            if knf_date in stooq_df.index:
                stooq_date = knf_date
            else:
                past_dates = stooq_df.index[stooq_df.index <= knf_date]
                if past_dates.empty:
                    logging.warning(f"    [Rejected] Stooq history starts after KNF date ({knf_date.date()})")
                    return False
                
                stooq_date = past_dates[-1] 
                if (knf_date - stooq_date).days > 4:
                    logging.warning(f"    [Rejected] Gap too large (KNF: {knf_date.date()}, Stooq: {stooq_date.date()})")
                    return False

            # 5. COMPARE
            stooq_nav = float(stooq_df.loc[stooq_date, "Zamkniecie"])
            diff_pct = abs(knf_nav - stooq_nav) / knf_nav
            is_match = diff_pct <= PRICE_TOLERANCE
            
            if is_match:
                logging.info(f"    [+] PRICE VERIFIED! KNF: {knf_nav:.2f} | Stooq: {stooq_nav:.2f} on {stooq_date.date()} (Diff: {diff_pct*100:.2f}%)")
            else:
                logging.info(f"    [-] PRICE MISMATCH! KNF: {knf_nav:.2f} | Stooq: {stooq_nav:.2f} on {stooq_date.date()} (Diff: {diff_pct*100:.2f}%)")
                
            return is_match

        except Exception as e:
            logging.warning(f"    [Rejected] Exception during verification {subfund_id} vs {stooq_id}: {e}")
            return False

    # =========================================================================
    # MATCHING ENGINE
    # =========================================================================

    def match_funds(self, knf_df: pd.DataFrame, stooq_df: pd.DataFrame) -> pd.DataFrame:
        knf_tfi_keys = set(knf_df["knf_tfi_key"].dropna().unique())
        
        sorted_tokens = sorted(knf_tfi_keys - {""}, key=len, reverse=True)
        def detect_tfi(name: str):
            s = self.to_ascii(name.lower())
            for t in sorted_tokens:
                if t and t in s: return t
            return ""
            
        stooq_df["stooq_tfi_key"] = stooq_df["stooq_name"].apply(detect_tfi)
        stooq_df["stooq_norm"] = stooq_df.apply(lambda r: self.residual_name(r["stooq_name"], r["stooq_tfi_key"]), axis=1)

        tfi_groups = {str(k): g.reset_index(drop=True) for k, g in stooq_df.groupby("stooq_tfi_key")}
        results = []

        for _, knf_row in knf_df.iterrows():
            sfid = knf_row.get("subfundId")
            knf_norm = knf_row["knf_norm"]
            knf_tfi_key = str(knf_row.get("knf_tfi_key", ""))
            
            result = {
                "subfundId": sfid, "name": knf_row.get("name"), "tfi_name": knf_row.get("tfi_name"),
                "category": knf_row.get("category"), "match_tier": "none", "match_score": 0,
                "price_verified": False, "stooq_id": None, "stooq_title": None, "stooq_name": None
            }

            if not knf_norm:
                results.append(result); continue

            pool = tfi_groups.get(knf_tfi_key, stooq_df)
            pool_norms = pool["stooq_norm"].tolist()

            def attempt_match(row, tier, score):
                logging.info(f"  Candidate: {knf_row['name'][:40]} -> {row['stooq_title']} ({tier}, score {score})")
                if self._verify_price_match(sfid, row["stooq_id"]):
                    result.update({
                        "match_tier": tier, "match_score": score, "price_verified": True,
                        "stooq_id": row["stooq_id"], "stooq_title": row["stooq_title"], "stooq_name": row["stooq_name"]
                    })
                    return True
                return False

            # Pass 1: Exact
            exact = pool[pool["stooq_norm"] == knf_norm]
            if not exact.empty and attempt_match(exact.iloc[0], "exact", 100):
                results.append(result); continue

            # Pass 2: Fuzzy
            best_match = process.extractOne(knf_norm, pool_norms, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
            if best_match and attempt_match(pool.iloc[best_match[2]], "fuzzy", round(best_match[1], 1)):
                results.append(result); continue

            # Pass 3: Historical Names
            former_names = self.fetch_subfund_history(int(sfid))
            matched_history = False
            for fname in former_names:
                fnorm = self.residual_name(fname, knf_tfi_key)
                if not fnorm: continue
                
                exact_h = pool[pool["stooq_norm"] == fnorm]
                if not exact_h.empty and attempt_match(exact_h.iloc[0], "historical_exact", 100):
                    matched_history = True; break
                    
                best_h = process.extractOne(fnorm, pool_norms, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
                if best_h and attempt_match(pool.iloc[best_h[2]], "historical_fuzzy", round(best_h[1], 1)):
                    matched_history = True; break
            
            if matched_history:
                results.append(result); continue

            results.append(result)
            time.sleep(0.1)

        return pd.DataFrame(results)

    def run_update_pipeline(self, confirmed_file="knf_stooq_confirmed.csv", use_tfi_scope: bool = True):
        """Full pipeline: Load confirmed -> Fetch KNF -> Load Stooq -> Match -> Save."""
        if not self.root_folder:
            logging.error("GDRIVE_FOLDER_ID not set.")
            return
            
        df_confirmed = self.gdrive.download_csv(self.root_folder, confirmed_file, sep=",")
        confirmed_ids = set(pd.to_numeric(df_confirmed['subfundId'], errors='coerce').dropna().tolist()) if df_confirmed is not None else set()
        
        # --- Przekazujemy parametr do fetch_knf_subfunds ---
        knf_df = self.fetch_knf_subfunds(confirmed_ids, use_tfi_scope=use_tfi_scope)
        
        if knf_df.empty:
            logging.info("No new subfunds to match.")
            return
            
        stooq_df = self.load_stooq_txt_files()
        if stooq_df.empty: return

        logging.info(f"\n--- Starting Ironclad Matching ({len(knf_df)} KNF vs {len(stooq_df)} Stooq) ---")
        match_df = self.match_funds(knf_df, stooq_df)

        matched = match_df[match_df["match_tier"] != "none"]
        unmatched = match_df[match_df["match_tier"] == "none"]
        
        logging.info(f"\nMatch Results: {len(matched)} successful (Price Verified), {len(unmatched)} failed.")
        
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        
        match_path = out_dir / "knf_stooq_matches.csv"
        unmatched_path = out_dir / "knf_stooq_unmatched.csv"
        review_path = out_dir / "knf_stooq_review.csv"
        
        match_df.to_csv(match_path, index=False, sep=";")
        unmatched.to_csv(unmatched_path, index=False, sep=";")
        
        review_mask = (match_df["match_tier"].isin(["fuzzy", "historical_fuzzy"])) & (match_df["match_score"] < 90)
        match_df[review_mask].sort_values("match_score").to_csv(review_path, index=False, sep=";")
        
        self.gdrive.upload_csv(self.root_folder, str(match_path))
        self.gdrive.upload_csv(self.root_folder, str(unmatched_path))
        self.gdrive.upload_csv(self.root_folder, str(review_path))
        logging.info("Matches uploaded to Google Drive.")
        
        # Zapisujemy summary wszystkich pobranych funduszy (jeśli use_tfi_scope=False to wyślemy GIGANTYCZNY słownik)
        summary_path = out_dir / "knf_summary.csv"
        knf_df.to_csv(summary_path, index=False, sep=";")
        self.gdrive.upload_csv(self.root_folder, str(summary_path))