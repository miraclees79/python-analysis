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

import logging
import re
import time
import unicodedata
from numbers import Real
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from moj_system.config import PROJECT_ROOT
from moj_system.data.gdrive import GDriveClient

# --- CONFIGURATION ---
KNF_API_BASE = "https://wybieramfundusze-api.knf.gov.pl"
FUZZY_THRESHOLD = 75  # Minimum score to consider a fuzzy match
PRICE_TOLERANCE = 0.05  # 5% tolerance for NAV comparison

FUND_NAMES_DIR = PROJECT_ROOT / "moj_system" / "data" / "fund_names_stooq"

# --- TFI REGISTRY (Wide Scope) ---
TFI_IN_SCOPE = {
    "tfi allianz polska s.a.",
    "generali investments tfi s.a.",
    "goldman sachs tfi s.a.",
    "investors tfi s.a.",
    "pko tfi s.a.",
    "tfi pzu sa",
    "quercus tfi s.a.",
    "skarbiec tfi s.a.",
    "uniqa tfi s.a.",
    "vig/c-quadrat tfi s.a.",
    "agiofunds tfi s.a.",
    "opoka tfi s.a.",
    "alior tfi s.a.",
    "rockbridge tfi s.a.",
    "amundi polska tfi s.a.",
    "bnp paribas tfi s.a.",
    "caspar tfi s.a.",
    "santander tfi s.a.",
    "eques investment tfi s.a.",
    "esaliens tfi s.a.",
    "mtfi s.a.",
    "best tfi s.a.",
    "mci capital tfi s.a.",
    "millennium tfi s.a.",
    "pekao tfi s.a.",
    "ipopema tfi s.a.",
    "velofunds tfi s.a.",
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
    def __init__(
        self,
        credentials_path: str | None = None,
    ) -> None:

        self.gdrive = GDriveClient(credentials_path)
        self.root_folder = self.gdrive.root_folder_id

    @staticmethod
    def _parse_positive_id(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (str, Real)):
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not pd.notna(numeric_value) or not numeric_value.is_integer() or numeric_value <= 0:
            return None
        return int(numeric_value)

    @staticmethod
    def _log_duplicate_ids(ids: pd.Series, label: str, level: int = logging.ERROR) -> bool:
        """Log every duplicated ID with the data rows (0-based) where it occurs."""
        duplicated = ids[ids.duplicated(keep=False)]
        if duplicated.empty:
            return False
        for value, rows in duplicated.groupby(duplicated).groups.items():
            logging.log(
                level,
                "Duplicate %s %s in confirmed file at rows %s",
                label,
                value,
                [int(row) for row in rows],
            )
        return True

    @staticmethod
    def _request_json(
        url: str,
        params: dict | None = None,
        timeout: int = 10,
    ) -> dict:
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object from {url}.")
                return payload
            except (requests.RequestException, ValueError) as exc:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                retryable = (
                    not isinstance(exc, requests.HTTPError)
                    or status_code == 429
                    or (status_code is not None and status_code >= 500)
                )
                if attempt == 2 or not retryable:
                    raise
                delay = 0.5 * (2**attempt)
                logging.warning(
                    "KNF request failed (%s); retrying in %.1fs (%d/3).",
                    exc,
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)
        raise RuntimeError(f"KNF request failed after retries: {url}")

    # =========================================================================
    # TEXT NORMALIZATION
    # =========================================================================

    @staticmethod
    def to_ascii(
        s: str,
    ) -> str:

        if not isinstance(s, str):
            return ""
        return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")

    def tfi_brand_token(
        self,
        company_name: str,
    ) -> str:

        if not isinstance(company_name, str) or not company_name.strip():
            return ""
        raw_key = self.to_ascii(company_name.strip().lower())
        if raw_key in TFI_BRAND_OVERRIDES:
            return TFI_BRAND_OVERRIDES[raw_key]

        s = raw_key
        structural = [
            r"\btowarzystwo funduszy inwestycyjnych\b",
            r"\bfunduszy inwestycyjnych\b",
            r"\binvestment(?:\s+partners)?\b",
            r"\basset\s+management\b",
            r"\btfi\b",
            r"\bs\.?\s*a\.?\b",
            r"\bsp\.?\s*z\.?\s*o\.?\s*o\.?\b",
        ]
        for term in structural:
            s = re.sub(term, " ", s)
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def residual_name(
        self,
        name: str,
        brand_token: str = "",
    ) -> str:

        if not isinstance(name, str):
            return ""
        s = self.to_ascii(name.lower())
        legal_terms = [
            r"\bfundusz inwestycyjny otwarty\b",
            r"\bfundusz inwestycyjny zamkniety\b",
            r"\bspecjalistyczny fundusz inwestycyjny otwarty\b",
            r"\bfundusz inwestycyjny\b",
            r"\bsfio\b",
            r"\bfio\b",
            r"\bfiz\b",
            r"\bsubfundusz\b",
            r"\bparasolowy\b",
            r"\bparasol\b",
            r"\btowarzystwo funduszy inwestycyjnych\b",
            r"\btfi\b",
        ]
        for term in legal_terms:
            s = re.sub(term, " ", s)
        if brand_token:
            s = re.sub(r"\b" + re.escape(brand_token) + r"\b", " ", s)
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
                        rows.append(
                            {
                                "stooq_id": int(m.group(1)),
                                "stooq_ticker": f"{m.group(1)}.N",
                                "stooq_title": m.group(2).strip(),
                                "stooq_category": category,
                            },
                        )
            except Exception as e:
                logging.warning(f"Could not read {txt_path.name}: {e}")

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["stooq_id"], keep="first")

        def strip_prefix(title: str, sid: int) -> str:
            prefix = f"{sid}.n - "
            if title.lower().startswith(prefix):
                title = title[len(prefix) :]
            title = re.sub(r"^\d+\.n\s*-\s*", "", title, flags=re.IGNORECASE)
            return re.sub(r"\s*-\s*stooq\s*$", "", title, flags=re.IGNORECASE).strip()

        df["stooq_name"] = df.apply(lambda r: strip_prefix(r["stooq_title"], r["stooq_id"]), axis=1)
        logging.info(f"Loaded {len(df)} Stooq names from {len(txt_files)} files.")
        return df

    def _fetch_all_pages(
        self,
        path: str,
        params: dict | None = None,
    ) -> list[dict]:

        items = []
        page = 0
        if params is None:
            params = {}
        while True:
            current_params = {**params, "size": 200, "page": page}
            data = self._request_json(f"{KNF_API_BASE}{path}", params=current_params, timeout=30)
            content = data.get("content")
            page_data = data.get("page", {})
            if (
                not isinstance(content, list)
                or any(not isinstance(item, dict) for item in content)
                or not isinstance(page_data, dict)
            ):
                raise ValueError(f"Invalid paginated response from {path}.")
            items.extend(content)
            total_pages = int(page_data.get("totalPages", 1))
            if page + 1 >= total_pages:
                break
            page += 1
            time.sleep(0.05)
        return items

    def fetch_knf_subfunds(
        self,
        use_tfi_scope: bool = True,
        exclude_subfund_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Deep Hydration: Fetches subfunds and crawls through KIDs to get
        Fees, Risk Levels and Benchmarks.
        """
        logging.info("Building TFI and Fund lookup maps...")
        companies = self._fetch_all_pages("/v1/companies")
        company_map = {c["companyId"]: c["name"] for c in companies if "companyId" in c}
        funds = self._fetch_all_pages("/v1/funds")
        fund_to_company = {f["fundId"]: f["companyId"] for f in funds if "fundId" in f}

        logging.info("Fetching subfund list...")
        all_subfunds = self._fetch_all_pages("/v1/subfunds", {"includeInactive": "false"})

        # Filter scope first to save API calls
        excluded_ids = exclude_subfund_ids or set()
        scoped_raw = []
        for s in all_subfunds:
            subfund_id = self._parse_positive_id(s.get("subfundId"))
            if subfund_id in excluded_ids:
                continue
            comp_id = fund_to_company.get(s.get("fundId"))
            tfi_name = company_map.get(comp_id, "")
            tfi_key = self.to_ascii(tfi_name.lower().strip())
            if not use_tfi_scope or tfi_key in TFI_IN_SCOPE:
                s["tfi_name"] = tfi_name
                scoped_raw.append(s)

        logging.info(f"Hydrating {len(scoped_raw)} subfunds with KID metadata...")
        hydrated_rows = []

        for i, s in enumerate(scoped_raw):
            sfid = s["subfundId"]
            if i > 0 and i % 20 == 0:
                logging.info(f"  ... hydrated {i}/{len(scoped_raw)}")

            try:
                # 1. Basic subfund detail
                detail = self._request_json(f"{KNF_API_BASE}/v1/subfunds/{sfid}", timeout=10)
                s.update(detail)

                # 2. Get latest Primary KID UUID
                kid_list = self._request_json(
                    f"{KNF_API_BASE}/v1/key-information/units",
                    params={"subfundId": sfid, "isPrimary": "true", "size": 1},
                    timeout=10,
                )
                content = kid_list.get("content", [])
                if not isinstance(content, list) or any(not isinstance(item, dict) for item in content):
                    raise ValueError(f"Invalid KID list response for subfund {sfid}.")

                if content:
                    uuid = content[0].get("documentUuid")
                    if not uuid:
                        raise ValueError(f"Primary KID UUID missing for subfund {sfid}.")
                    # 3. Get Deep KID Metadata (Fees, Risk, Benchmark)
                    kid_detail = self._request_json(
                        f"{KNF_API_BASE}/v1/key-information/units/{uuid}",
                        timeout=10,
                    )
                    s.update(
                        {
                            "riskLevel": kid_detail.get("riskLevel"),
                            "managementAndOtherFeesRate": kid_detail.get(
                                "managementAndOtherFeesRate",
                            ),
                            "maxEntryFeeRate": kid_detail.get("maxEntryFeeRate"),
                            "hasBenchmark": kid_detail.get("hasBenchmark"),
                            "benchmark": kid_detail.get("benchmark"),
                        },
                    )

                hydrated_rows.append(s)
                time.sleep(0.05)
            except Exception as e:
                logging.warning(f"Could not fully hydrate {sfid}: {e}")
                hydrated_rows.append(s)

        df = pd.DataFrame(hydrated_rows)
        if df.empty:
            return df

        df = df[~df["category"].astype(str).str.startswith("PPK")]
        df["knf_tfi_key"] = df["tfi_name"].apply(self.tfi_brand_token)
        df["knf_norm"] = df.apply(
            lambda r: self.residual_name(str(r["name"]), r["knf_tfi_key"]),
            axis=1,
        )

        logging.info(f"Deep hydration complete. {len(df)} subfunds ready.")
        return df

    def fetch_subfund_history(
        self,
        subfund_id: int,
    ) -> list[str]:

        url = f"{KNF_API_BASE}/v1/subfunds/{subfund_id}/history"
        try:
            payload = self._request_json(
                url,
                params={"size": 50, "sort": "validFrom,desc"},
                timeout=10,
            )
            content = payload.get("content", [])
            if not isinstance(content, list):
                raise ValueError(f"Invalid history response for subfund {subfund_id}.")
            former = set()
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("validTo", "")) == "9999-12-31":
                    continue
                data = entry.get("data")
                if not isinstance(data, dict):
                    continue
                name = data.get("name", "")
                if name:
                    former.add(name)
            return list(former)
        except (requests.RequestException, ValueError) as exc:
            logging.warning("Could not fetch name history for subfund %s: %s", subfund_id, exc)
            return []

    # =========================================================================
    # PRICE VERIFICATION
    # =========================================================================

    def _verify_price_match(
        self,
        subfund_id: int,
        stooq_id: int,
    ) -> bool:

        from moj_system.data.updater import DataUpdater  # Import here to avoid circular dependency

        try:
            # 1. KNF: Get latest valuation (could be 1 month old)
            url = f"{KNF_API_BASE}/v1/valuations"
            payload = self._request_json(
                url,
                params={"subfundId": subfund_id, "size": 1, "sort": "date,desc"},
                timeout=10,
            )
            knf_data = payload.get("content", [])

            if not isinstance(knf_data, list) or not knf_data or not isinstance(knf_data[0], dict):
                return False
            item = knf_data[0]
            if item.get("valuation") is None or item.get("date") is None:
                return False

            knf_date = pd.to_datetime(item["date"]).normalize()
            knf_nav = float(item["valuation"])

            # 2. STOOQ: Extract candidate history without changing the canonical fund CSV
            logging.info(f"    [Verification] Extracting candidate {stooq_id} from ZIP...")
            updater = DataUpdater(credentials_path=self.gdrive.credentials_path)
            stooq_df = updater.fetch_stooq_history(
                stooq_ticker=f"{stooq_id}.n",
                zip_type="fund_pl",
                label=f"fund_{stooq_id}",
            )
            if stooq_df is None or stooq_df.empty:
                return False
            stooq_df.index = pd.to_datetime(stooq_df.index).tz_localize(None)

            # 4. DATE ALIGNMENT
            if knf_date in stooq_df.index:
                stooq_date = knf_date
            else:
                past_dates = stooq_df.index[stooq_df.index <= knf_date]
                if past_dates.empty:
                    logging.warning(
                        f"    [Rejected] Stooq history starts after KNF date ({knf_date.date()})",
                    )
                    return False

                stooq_date = past_dates[-1]
                if (knf_date - stooq_date).days > 4:
                    logging.warning(
                        f"    [Rejected] Gap too large (KNF: {knf_date.date()}, Stooq: {stooq_date.date()})",
                    )
                    return False

            # 5. COMPARE
            stooq_nav = float(stooq_df.loc[stooq_date, "Zamkniecie"])
            diff_pct = abs(knf_nav - stooq_nav) / knf_nav
            is_match = diff_pct <= PRICE_TOLERANCE

            if is_match:
                logging.info(
                    f"    [+] PRICE VERIFIED! KNF: {knf_nav:.2f} | "
                    f"Stooq: {stooq_nav:.2f} on {stooq_date.date()} "
                    f"(Diff: {diff_pct * 100:.2f}%)",
                )
            else:
                logging.info(
                    f"    [-] PRICE MISMATCH! KNF: {knf_nav:.2f} | "
                    f"Stooq: {stooq_nav:.2f} on {stooq_date.date()} "
                    f"(Diff: {diff_pct * 100:.2f}%)",
                )

            return is_match

        except Exception as e:
            logging.warning(
                f"    [Rejected] Exception during verification {subfund_id} vs {stooq_id}: {e}",
            )
            return False

    # =========================================================================
    # MATCHING ENGINE
    # =========================================================================

    def match_funds(
        self,
        knf_df: pd.DataFrame,
        stooq_df: pd.DataFrame,
    ) -> pd.DataFrame:

        knf_tfi_keys = set(knf_df["knf_tfi_key"].dropna().unique())

        sorted_tokens = sorted(knf_tfi_keys - {""}, key=len, reverse=True)

        def detect_tfi(name: str) -> str:
            s = self.to_ascii(name.lower())
            for t in sorted_tokens:
                if t and t in s:
                    return t
            return ""

        stooq_df = stooq_df.copy()
        stooq_df["stooq_tfi_key"] = stooq_df["stooq_name"].apply(detect_tfi)
        stooq_df["stooq_norm"] = stooq_df.apply(
            lambda r: self.residual_name(r["stooq_name"], r["stooq_tfi_key"]),
            axis=1,
        )

        tfi_groups = {str(k): g.reset_index(drop=True) for k, g in stooq_df.groupby("stooq_tfi_key")}
        results = []
        used_stooq_ids: set[int] = set()

        for knf_row in knf_df.to_dict(orient="records"):
            sfid_value = knf_row.get("subfundId")
            result = {
                "subfundId": sfid_value,
                "name": knf_row.get("name"),
                "tfi_name": knf_row.get("tfi_name"),
                "category": knf_row.get("category"),
                "match_tier": "none",
                "match_score": 0,
                "price_verified": False,
                "stooq_id": None,
                "stooq_title": None,
                "stooq_name": None,
            }
            sfid = self._parse_positive_id(sfid_value)
            if sfid is None:
                logging.warning("Skipping KNF row with invalid subfundId: %r", sfid_value)
                results.append(result)
                continue
            result["subfundId"] = sfid
            candidate_subfund_id = sfid

            knf_name_value = knf_row.get("name")
            knf_name = knf_name_value.strip() if isinstance(knf_name_value, str) else ""
            knf_norm_value = knf_row.get("knf_norm")
            knf_norm = knf_norm_value.strip() if isinstance(knf_norm_value, str) else ""
            knf_tfi_key = str(knf_row.get("knf_tfi_key", ""))

            if not knf_norm:
                results.append(result)
                continue
            if not knf_name:
                logging.warning("Skipping subfund %s with no usable name.", sfid)
                results.append(result)
                continue

            pool = tfi_groups.get(knf_tfi_key)
            if pool is None or pool.empty:
                logging.info("No Stooq candidates for TFI key %r; skipping cross-TFI matching.", knf_tfi_key)
                results.append(result)
                continue
            candidate_pool = pool
            attempted_stooq_ids: set[int] = set()

            def attempt_match(
                row: pd.Series,
                tier: str,
                score: float,
                *,
                current_attempted_ids: set[int] = attempted_stooq_ids,
                current_candidate_subfund_id: int = candidate_subfund_id,
                current_name: str = knf_name,
                current_result: dict[str, object] = result,
                current_used_ids: set[int] = used_stooq_ids,
            ) -> bool:
                stooq_id = self._parse_positive_id(row["stooq_id"])
                if (
                    stooq_id is None
                    or stooq_id in current_used_ids
                    or stooq_id in current_attempted_ids
                ):
                    return False
                current_attempted_ids.add(stooq_id)
                logging.info(
                    f"  Candidate: {current_name[:40]} -> {row['stooq_title']} ({tier}, score {score})",
                )
                if self._verify_price_match(current_candidate_subfund_id, stooq_id):
                    current_result.update(
                        {
                            "match_tier": tier,
                            "match_score": score,
                            "price_verified": True,
                            "stooq_id": row["stooq_id"],
                            "stooq_title": row["stooq_title"],
                            "stooq_name": row["stooq_name"],
                        },
                    )
                    current_used_ids.add(stooq_id)
                    return True
                return False

            # Pass 1: Exact
            exact = candidate_pool[candidate_pool["stooq_norm"] == knf_norm]
            matched = any(
                attempt_match(row, "exact", 100)
                for _, row in exact.iterrows()
            )
            if matched:
                results.append(result)
                continue

            def fuzzy_candidates(
                query: str,
                current_pool: pd.DataFrame = candidate_pool,
                current_attempted_ids: set[int] = attempted_stooq_ids,
                current_used_ids: set[int] = used_stooq_ids,
            ) -> list[tuple[pd.Series, float]]:
                eligible = current_pool[
                    ~current_pool["stooq_id"]
                    .map(self._parse_positive_id)
                    .isin(current_used_ids | current_attempted_ids)
                ]
                if eligible.empty:
                    return []
                matches = process.extract(
                    query,
                    eligible["stooq_norm"].tolist(),
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=FUZZY_THRESHOLD,
                    limit=5,
                )
                candidates = []
                for match in matches:
                    score = match[1]
                    index = match[2]
                    if index is not None:
                        candidates.append((eligible.iloc[index], round(score, 1)))
                return candidates

            # Pass 2: Fuzzy
            matched = any(
                attempt_match(row, "fuzzy", score)
                for row, score in fuzzy_candidates(knf_norm)
            )
            if matched:
                results.append(result)
                continue

            # Pass 3: Historical Names
            former_names = self.fetch_subfund_history(int(sfid))
            matched_history = False
            for fname in former_names:
                fnorm = self.residual_name(fname, knf_tfi_key)
                if not fnorm:
                    continue

                exact_h = candidate_pool[candidate_pool["stooq_norm"] == fnorm]
                if any(
                    attempt_match(row, "historical_exact", 100)
                    for _, row in exact_h.iterrows()
                ):
                    matched_history = True
                    break

                if any(
                    attempt_match(row, "historical_fuzzy", score)
                    for row, score in fuzzy_candidates(fnorm)
                ):
                    matched_history = True
                    break

            if matched_history:
                results.append(result)
                continue

            results.append(result)
            time.sleep(0.1)

        return pd.DataFrame(results)

    def run_update_pipeline(
        self,
        confirmed_file: str = "knf_stooq_confirmed.csv",
        use_tfi_scope: bool = True,
    ) -> None:
        """Full pipeline: Load confirmed -> Fetch KNF -> Load Stooq -> Match -> Save."""
        if not self.root_folder:
            logging.error("GDRIVE_FOLDER_ID not set.")
            return

        df_confirmed = self.gdrive.download_csv(self.root_folder, confirmed_file, sep=",")
        if df_confirmed is None:
            logging.error("Cannot load authoritative confirmed mappings; aborting automatic matching.")
            return
        required_columns = {"subfundId", "stooq_id"}
        missing_columns = required_columns - set(df_confirmed.columns)
        if missing_columns:
            logging.error(
                "Confirmed mapping file is missing required columns %s; aborting automatic matching.",
                sorted(missing_columns),
            )
            return

        parsed_knf_ids = df_confirmed["subfundId"].map(self._parse_positive_id)
        parsed_stooq_ids = df_confirmed["stooq_id"].map(self._parse_positive_id)
        invalid_rows = parsed_knf_ids.isna() | parsed_stooq_ids.isna()
        if invalid_rows.any():
            invalid_indices = df_confirmed.index[invalid_rows].tolist()
            logging.error(
                "Confirmed mapping file contains invalid or missing IDs at rows %s; "
                "aborting automatic matching.",
                invalid_indices,
            )
            return
        has_duplicate_knf = self._log_duplicate_ids(parsed_knf_ids, "subfundId")
        has_duplicate_stooq = self._log_duplicate_ids(parsed_stooq_ids, "stooq_id")
        if has_duplicate_knf or has_duplicate_stooq:
            logging.error(
                "Confirmed mapping file contains duplicate subfundId or stooq_id values; "
                "aborting automatic matching.",
            )
            return

        confirmed_knf_ids = set(parsed_knf_ids.astype(int))
        confirmed_stooq_ids = set(parsed_stooq_ids.astype(int))
        knf_df = self.fetch_knf_subfunds(
            use_tfi_scope=use_tfi_scope,
            exclude_subfund_ids=confirmed_knf_ids,
        )

        if knf_df.empty:
            logging.info("No new subfunds to match.")
            return

        # Summary covers subfunds not already present in the authoritative mapping.
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        summary_path = out_dir / "knf_summary.csv"
        knf_df.to_csv(summary_path, index=False, sep=";")
        if not self.gdrive.upload_csv(self.root_folder, str(summary_path)):
            raise RuntimeError(f"Failed to upload {summary_path.name} to Google Drive.")

        knf_id_values = pd.to_numeric(knf_df["subfundId"], errors="coerce")
        new_knf_mask = (
            knf_id_values.notna()
            & (knf_id_values % 1 == 0)
            & ~knf_id_values.isin(confirmed_knf_ids)
        )
        knf_candidates = knf_df.loc[new_knf_mask].copy()
        if knf_candidates.empty:
            logging.info("No unconfirmed KNF subfunds require matching.")
            return

        stooq_df = self.load_stooq_txt_files()
        if stooq_df.empty:
            return

        stooq_id_values = pd.to_numeric(stooq_df["stooq_id"], errors="coerce")
        available_stooq_mask = (
            stooq_id_values.notna()
            & (stooq_id_values % 1 == 0)
            & ~stooq_id_values.isin(confirmed_stooq_ids)
        )
        stooq_candidates = stooq_df.loc[available_stooq_mask].copy()

        logging.info(
            f"\n--- Matching {len(knf_candidates)} unconfirmed KNF subfunds against "
            f"{len(stooq_candidates)} unassigned Stooq names ---",
        )
        match_df = self.match_funds(knf_candidates, stooq_candidates)

        matched = match_df[match_df["match_tier"] != "none"]
        unmatched = match_df[match_df["match_tier"] == "none"]

        logging.info(
            f"\nMatch Results: {len(matched)} successful (Price Verified), {len(unmatched)} failed.",
        )

        match_path = out_dir / "knf_stooq_matches.csv"
        unmatched_path = out_dir / "knf_stooq_unmatched.csv"
        review_path = out_dir / "knf_stooq_review.csv"

        match_df.to_csv(match_path, index=False, sep=";")
        unmatched.to_csv(unmatched_path, index=False, sep=";")

        review_mask = (match_df["match_tier"].isin(["fuzzy", "historical_fuzzy"])) & (
            match_df["match_score"] < 90
        )
        match_df[review_mask].sort_values("match_score").to_csv(review_path, index=False, sep=";")

        failed_uploads = [
            path.name
            for path in (match_path, unmatched_path, review_path)
            if not self.gdrive.upload_csv(self.root_folder, str(path))
        ]
        if failed_uploads:
            raise RuntimeError(
                "Failed to upload result files to Google Drive: " + ", ".join(failed_uploads),
            )
        logging.info("Matches uploaded to Google Drive.")

    def verify_confirmed_matches(
        self,
        confirmed_file: str = "knf_stooq_confirmed.csv",
    ) -> None:
        """
        Pobiera potwierdzoną listę funduszy z GDrive i tylko dla nich uruchamia weryfikację cen.
        Nie szuka nowych dopasowań, nie skanuje pełnego API KNF.
        """
        if not self.root_folder:
            logging.error("GDRIVE_FOLDER_ID not set. Cannot download confirmed list.")
            return

        logging.info(f"Downloading confirmed matches from {confirmed_file}...")
        df_confirmed = self.gdrive.download_csv(self.root_folder, confirmed_file, sep=",")

        if df_confirmed is None or df_confirmed.empty:
            logging.error("Confirmed list is empty or could not be downloaded.")
            return

        required_columns = {"subfundId", "stooq_id"}
        missing_columns = required_columns - set(df_confirmed.columns)
        if missing_columns:
            logging.error(
                "Confirmed mapping file is missing required columns %s; cannot verify prices.",
                sorted(missing_columns),
            )
            return

        valid_rows = []
        for index, row in df_confirmed.iterrows():
            subfund_id = self._parse_positive_id(row["subfundId"])
            stooq_id = self._parse_positive_id(row["stooq_id"])
            if subfund_id is None or stooq_id is None:
                logging.warning("Skipping confirmed row %s with invalid or missing IDs.", index)
                continue
            valid_rows.append((index, row, subfund_id, stooq_id))

        row_indices = [row[0] for row in valid_rows]
        self._log_duplicate_ids(
            pd.Series([row[2] for row in valid_rows], index=row_indices, dtype="int64"),
            "subfundId",
            logging.WARNING,
        )
        self._log_duplicate_ids(
            pd.Series([row[3] for row in valid_rows], index=row_indices, dtype="int64"),
            "stooq_id",
            logging.WARNING,
        )
        knf_counts = pd.Series([row[2] for row in valid_rows]).value_counts()
        stooq_counts = pd.Series([row[3] for row in valid_rows]).value_counts()
        duplicate_rows = [
            row for row in valid_rows
            if knf_counts[row[2]] > 1 or stooq_counts[row[3]] > 1
        ]
        for duplicate_row in duplicate_rows:
            index = duplicate_row[0]
            logging.warning("Skipping confirmed row %s because its mapping uses a duplicate ID.", index)
        duplicate_indices = {row[0] for row in duplicate_rows}
        df_to_verify = [
            row for row in valid_rows if row[0] not in duplicate_indices
        ]
        if not df_to_verify:
            logging.info("No valid unique mappings found in the confirmed list.")
            return

        total_funds = len(df_to_verify)
        logging.info(
            f"\n--- Starting Fast Price Verification for {total_funds} confirmed funds ---",
        )

        success_count = 0
        failed_funds = []

        for verification_row in df_to_verify:
            row = verification_row[1]
            subfund_id = verification_row[2]
            stooq_id = verification_row[3]
            fund_name = row.get("name", f"Subfund {subfund_id}")

            logging.info(f"Verifying: {fund_name} (KNF: {subfund_id}, Stooq: {stooq_id})")

            is_match = self._verify_price_match(subfund_id, stooq_id)

            if is_match:
                success_count += 1
            else:
                failed_funds.append(f"KNF: {subfund_id} | Stooq: {stooq_id} | Name: {fund_name}")

            time.sleep(0.1)  # Drobne opóźnienie dla stabilności

        logging.info("\n" + "=" * 50)
        logging.info("VERIFICATION SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Total checked: {total_funds}")
        logging.info(f"Passed:        {success_count}")
        logging.info(f"Failed:        {total_funds - success_count}")

        if failed_funds:
            logging.warning("Failed funds:")
            for f in failed_funds:
                logging.warning(f"  - {f}")
        logging.info("=" * 50)
