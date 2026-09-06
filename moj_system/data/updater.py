# -*- coding: utf-8 -*-
"""
moj_system/data/updater.py
==========================
Hybrid data updater module. Handles default indices, ETFs, and KNF funds.
"""

import io
import logging
import zipfile
from typing import Any, cast

import pandas as pd
import requests
import yfinance as yf

from moj_system.config import DATA_DIR
from moj_system.data.gdrive import GDriveClient

# RAW_DIR to bezpośrednio DATA_DIR z config.py (czyli .../data/raw_csv)
RAW_DIR = DATA_DIR 
# ZIP_DIR ląduje obok raw_csv, czyli w .../data/zips
ZIP_DIR = DATA_DIR.parent / "zips"

RAW_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DIR.mkdir(parents=True, exist_ok=True)

ZIP_MAPPING = {
    "index_pl": "d_pl_txt.zip",
    "index_world": "d_world_txt.zip",
    "currencies": "d_world_txt.zip",
    "fund_pl": "d_pl_txt.zip",
    "etf_pl": "d_pl_txt.zip",
    "bonds": "d_world_txt.zip",
}

CONFIRMED_FUNDS_FILE = "knf_stooq_confirmed.csv"
GDRIVE_DATA_FOLDER_NAME = "Dane"

# =========================================================================
# TICKER LISTS
# =========================================================================

DEFAULT_TICKERS = [
    {"label": "wig", "stooq": "wig", "yf": "WIG.WA", "gpw_isin": "PL9999999995", "type": "index_pl"},
    {"label": "wig20tr", "stooq": "wig20tr", "yf": "WIG20TR.WA", "gpw_isin": "PL9999999425", "type": "index_pl"},
    {"label": "mwig40tr", "stooq": "mwig40tr", "yf": "MWIG40TR.WA", "gpw_isin": "PL9999999078", "type": "index_pl"},
    {"label": "swig80tr", "stooq": "swig80tr", "yf": "SWIG80TR.WA", "gpw_isin": "PL9999999060", "type": "index_pl"},
    {"label": "tbsp", "stooq": "^tbsp", "yf": None, "gpw_isin": "PL9999999474", "type": "index_pl"},
    {"label": "sp500", "stooq": "^spx", "yf": "^GSPC", "type": "index_world"},
    {"label": "nikkei225", "stooq": "^nkx", "yf": "^N225", "type": "index_world"},
    {"label": "nasdaq100", "stooq": "^ndx", "yf": "^NDX", "type": "index_world"},
    {"label": "wibor1m", "stooq": "plopln1m", "yf": None, "type": "currencies"},
    {"label": "usdpln", "stooq": "usdpln", "yf": "USDPLN=X", "type": "currencies"},
    {"label": "eurpln", "stooq": "eurpln", "yf": "EURPLN=X", "type": "currencies"},
    {"label": "jpypln", "stooq": "jpypln", "yf": "JPYPLN=X", "type": "currencies"},
    {"label": "de10y", "stooq": "10ydey.b", "yf": None, "type": "bonds"},
    {"label": "pl10y", "stooq": "10yply.b", "yf": None, "type": "bonds"},
    {"label": "fund_2720", "stooq": "2720.n", "yf": None, "type": "fund_pl", "knf": "195983"},
    {"label": "wbbw", "stooq": "^gpwbbwz", "yf": None, "type": "index_pl"},
]

ETF_TICKERS = [
    {"label": "etfbdivpl", "stooq": "etfbdivpl.pl", "yf": "ETFBDIVPL.WA", "type": "etf_pl"},
    {"label": "etfbm40tr", "stooq": "etfbm40tr.pl", "yf": "ETFBM40TR.WA", "type": "etf_pl"},
    {"label": "etfbs80tr", "stooq": "etfbs80tr.pl", "yf": "ETFBS80TR.WA", "type": "etf_pl"},
    {"label": "etfbw20lv", "stooq": "etfbw20lv.pl", "yf": "ETFBW20LV.WA", "type": "etf_pl"},
    {"label": "etfbw20st", "stooq": "etfbw20st.pl", "yf": "ETFBW20ST.WA", "type": "etf_pl"},
    {"label": "etfbw20tr", "stooq": "etfbw20tr.pl", "yf": "ETFBW20TR.WA", "type": "etf_pl"},
    {"label": "etfpzuw20m40", "stooq": "etfpzuw20m40.pl", "yf": "ETFPZUW20M40.WA", "type": "etf_pl"},
    {"label": "etfbcash", "stooq": "etfbcash.pl", "yf": "ETFBCASH.WA", "type": "etf_pl"},
    {"label": "etfbtbsp", "stooq": "etfbtbsp.pl", "yf": "ETFBTBSP.WA", "type": "etf_pl"},
    {"label": "etfbndxpl", "stooq": "etfbndxpl.pl", "yf": "ETFBNDXPL.WA", "type": "etf_pl"},
    {"label": "etfbnq2st", "stooq": "etfbnq2st.pl", "yf": "ETFBNQ2ST.WA", "type": "etf_pl"},
    {"label": "etfbspxpl", "stooq": "etfbspxpl.pl", "yf": "ETFBSPXPL.WA", "type": "etf_pl"},
    {"label": "etfsp500", "stooq": "etfsp500.pl", "yf": "ETFSP500.WA", "type": "etf_pl"},
    {"label": "etfdax", "stooq": "etfdax.pl", "yf": "ETFDAX.WA", "type": "etf_pl"},
]

class DataUpdater:
    def __init__(
        self, 
        gdrive_folder_id: str | None = None, 
        credentials_path: str | None = None,
    ) -> None:
        self.gdrive = GDriveClient(credentials_path=credentials_path)
        self.root_folder_id = gdrive_folder_id or os.environ.get("GDRIVE_FOLDER_ID")
        self.data_folder_id = None
        
        service: Any = self.gdrive.service
        if service and self.root_folder_id:
            self.data_folder_id = self._get_or_create_subfolder(folder_name=GDRIVE_DATA_FOLDER_NAME)

    def _get_or_create_subfolder(
        self, 
        folder_name: str,
    ) -> str | None:
        service: Any = self.gdrive.service
        if not service:
            return None
        
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id)").execute()
        items = results.get("files", [])
        if items:
            return items[0]["id"]
            
        metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self.root_folder_id],
        }
        folder = service.files().create(body=metadata, fields="id").execute()
        return folder.get("id")

    def _get_zip_content(
        self, 
        zip_type: str,
    ) -> bytes | None:
        zip_name = ZIP_MAPPING.get(zip_type)
        if not zip_name:
            return None
        
        local_path = ZIP_DIR / zip_name
        if local_path.exists():
            return local_path.read_bytes()
            
        service: Any = self.gdrive.service
        if self.root_folder_id and service:
            logging.info(msg=f"Downloading {zip_name} from GDrive...")
            file_id = self.gdrive.find_file_id(parent_id=self.root_folder_id, filename=zip_name)
            if file_id:
                try:
                    request = service.files().get_media(fileId=file_id)
                    fh = io.BytesIO()
                    from googleapiclient.http import MediaIoBaseDownload
                    downloader = MediaIoBaseDownload(fd=fh, request=request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    content = fh.getvalue()
                    local_path.write_bytes(data=content)
                    return content
                except Exception as e:
                    logging.error(msg=f"GDrive ZIP error: {e}")
        return None

    def _extract_from_zip(
        self, 
        zip_data:     bytes, 
        stooq_ticker: str,
    ) -> pd.DataFrame | None:
        if not zip_data:
            return None
        try:
            with zipfile.ZipFile(file=io.BytesIO(initial_bytes=zip_data)) as z:
                search_name = f"{stooq_ticker.lower()}.txt"
                target_file = next(
                    (f for f in z.namelist() if f.lower().endswith(search_name)), None,
                )
                if target_file:
                    with z.open(name=target_file) as f:
                        file_bytes = f.read()
                        df = pd.read_csv(filepath_or_buffer=io.BytesIO(initial_bytes=file_bytes))
                        
                        col_map = {
                            "<DATE>": "Data",
                            "<OPEN>": "Otwarcie",
                            "<HIGH>": "Najwyzszy",
                            "<LOW>": "Najnizszy",
                            "<CLOSE>": "Zamkniecie",
                        }
                        cols_to_keep = [c for c in col_map.keys() if c in df.columns]
                        df = df[cols_to_keep].rename(columns={k: col_map[k] for k in cols_to_keep})
                        df["Data"] = pd.to_datetime(arg=df["Data"], format="%Y%m%d")
                        return df
        except Exception as e:
            logging.error(msg=f"ZIP error for {stooq_ticker}: {e}")
        return None

    def _fetch_yfinance_data(
        self, 
        ticker_yf:  str, 
        start_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        try:
            df = yf.download(tickers=ticker_yf, start=start_date, progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(level=1, axis=1)
            
            df.index.name = "Data"
            df = df.reset_index().rename(
                columns={
                    "Open": "Otwarcie",
                    "High": "Najwyzszy",
                    "Low": "Najnizszy",
                    "Close": "Zamkniecie",
                },
            )
            
            df["Data"] = pd.to_datetime(arg=df["Data"]).dt.tz_localize(tz=None)
            cols = ["Data", "Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie"]
            return df[[c for c in cols if c in df.columns]]
            
        except Exception as e:
            logging.warning(msg=f"yfinance error ({ticker_yf}): {e}")
            return None

    def _fetch_gpwbenchmark_data(
        self, 
        isin:       str, 
        start_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        import json
        import time
        import urllib.parse

        payload = [{"isin": isin, "mode": "14D"}] 
        encoded_payload = urllib.parse.quote(string=json.dumps(obj=payload))
        t_param = int(time.time() * 1000.0)
        url = f"https://gpwbenchmark.pl/chart-json.php?req={encoded_payload}&t={t_param}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://gpwbenchmark.pl/karta-indeksu?isin={isin}",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        }

        try:
            response = requests.get(url=url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                dataset = data[0].get("data", [])
            else:
                dataset = []

            if not dataset:
                logging.warning(msg=f"GPW Benchmark returned empty dataset for {isin}")
                return None

            records = []
            for item in dataset:
                if "t" in item and "c" in item:
                    date_val = pd.to_datetime(arg=item["t"], unit="s").tz_localize(tz=None).normalize()
                    if date_val > start_date:
                        records.append(
                            {
                                "Data": date_val,
                                "Otwarcie": float(item.get("o", item["c"])),
                                "Najwyzszy": float(item.get("h", item["c"])),
                                "Najnizszy": float(item.get("l", item["c"])),
                                "Zamkniecie": float(item["c"]),
                            }
                        )

            if not records:
                return None

            df = pd.DataFrame(data=records).sort_values(by="Data")
            return df

        except Exception as e:
            logging.warning(msg=f"GPW Benchmark API error ({isin}): {e}")
            return None

    def _fetch_knf_data(
        self, 
        subfund_id: str | None, 
        start_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        if not subfund_id or pd.isna(obj=subfund_id):
            return None
        
        url = "https://wybieramfundusze-api.knf.gov.pl/v1/valuations"
        params = {
            "subfundId": int(float(subfund_id)),
            "dateFrom": pd.to_datetime(arg=start_date).normalize().strftime(format="%Y-%m-%d"),
        }
        
        try:
            response = requests.get(url=url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            items = data.get("content", []) if isinstance(data, dict) else data
            records = []
            
            for item in items:
                v_date = pd.to_datetime(arg=item.get("date")).normalize()
                if v_date > start_date:
                    records.append(
                        {
                            "Data": v_date,
                            "Otwarcie": item.get("valuation"),
                            "Najwyzszy": item.get("valuation"),
                            "Najnizszy": item.get("valuation"),
                            "Zamkniecie": item.get("valuation"),
                        }
                    )
            return pd.DataFrame(data=records).sort_values(by="Data") if records else None
            
        except Exception as e:
            logging.warning(msg=f"KNF API Error: {e}")
            return None

    def _validate_and_clean(
        self, 
        df:    pd.DataFrame, 
        label: str,
    ) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None

        df["Data"] = pd.to_datetime(arg=df["Data"])
        df = df.sort_values(
            by="Data"
        ).drop_duplicates(
            subset="Data", 
            keep="last"
        ).set_index(
            keys="Data"
        )
        
        date_diffs = df.index.to_series().diff().dt.days
        breaks = date_diffs[date_diffs > 30].index
        if not breaks.empty:
            df = df.loc[df.index > breaks[-1]]

        if label.startswith("fund_"):
            price_cols = ["Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie"]
            available_cols = [c for c in price_cols if c in df.columns]

            if "Zamkniecie" in df.columns and len(df) > 1:
                pct_change = df["Zamkniecie"].pct_change()
                split_indices = pct_change[pct_change.abs() > 0.40].index

                for split_date in split_indices:
                    loc = df.index.get_loc(key=split_date)
                    
                    if isinstance(loc, int) and loc > 0:
                        prev_date = df.index[loc - 1]
                        
                        # Pobranie bezpieczne typologicznellement
                        prev_val = cast(float, df.at[prev_date, "Zamkniecie"])
                        curr_val = cast(float, df.at[split_date, "Zamkniecie"])
                        
                        prev_price = float(prev_val)
                        curr_price = float(curr_val)
                        
                        if prev_price != 0.0:
                            factor = curr_price / prev_price
                            
                            logging.info(
                                msg=f"[{label}] REBASING EVENT on {split_date.date()}: Price jumped from {prev_price:.4f} to {curr_price:.4f} (Factor: {factor:.4f}). Adjusting older data."
                            )
                            
                            for col in available_cols:
                                col_idx = df.columns.get_loc(key=col)
                                if isinstance(col_idx, int):
                                    df.iloc[:loc, col_idx] = df.iloc[:loc, col_idx] * factor

        return df.reset_index()

    def update_ticker(
        self,
        label:           str,
        stooq_ticker:    str,
        yf_ticker:       str | None = None,
        knf_id:          str | None = None,
        gpw_isin:        str | None = None,
        zip_type:        str        = "index_pl",
        upload_to_drive: bool       = False,
    ) -> bool:
        logging.info(msg=f"--- Updating: {label} ({stooq_ticker}) ---")
        
        zip_data = self._get_zip_content(zip_type=zip_type)
        if not zip_data:
            logging.error(msg=f"   [ZIP] Failed to retrieve ZIP content for type '{zip_type}'.")
            df_hist = None
        else:
            df_hist = self._extract_from_zip(zip_data=zip_data, stooq_ticker=stooq_ticker)
            if df_hist is None or df_hist.empty:
                logging.warning(msg=f"   [ZIP] No historical data extracted for '{stooq_ticker}' from ZIP.")
            else:
                logging.info(msg=f"   [ZIP] Extracted {len(df_hist)} rows. Last date: {df_hist['Data'].max().date()}")

        last_date = df_hist["Data"].max() if df_hist is not None and not df_hist.empty else pd.Timestamp("1990-01-01")

        df_new = None
        if gpw_isin:
            logging.info(msg=f"   [API] Fetching missing data from GPW Benchmark ({gpw_isin}) since {last_date.date()}...")
            df_new = self._fetch_gpwbenchmark_data(isin=gpw_isin, start_date=last_date)
            
        if yf_ticker and (df_new is None or df_new.empty):
            if gpw_isin:
                logging.warning(msg=f"   [API] GPW Benchmark failed or returned no data. Falling back to YFinance ({yf_ticker})...")
            else:
                logging.info(msg=f"   [API] Fetching missing data from YFinance ({yf_ticker}) since {last_date.date()}...")
            df_new = self._fetch_yfinance_data(ticker_yf=yf_ticker, start_date=last_date)
            
        if knf_id and (df_new is None or df_new.empty):
            logging.info(msg=f"   [API] Fetching missing data from KNF API (Fund ID: {knf_id}) since {last_date.date()}...")
            df_new = self._fetch_knf_data(subfund_id=knf_id, start_date=last_date)
            
        if not (gpw_isin or yf_ticker or knf_id):
            logging.info(msg="   [API] No external API configured. Relying solely on Stooq ZIP history.")

        if df_new is not None and not df_new.empty:
            logging.info(msg=f"   [API] Successfully retrieved {len(df_new)} new rows.")
        elif (yf_ticker or gpw_isin or knf_id):
            logging.warning(msg=f"   [API] External API returned no new data (or an error occurred) for {label}.")

        if df_hist is not None and not df_hist.empty and df_new is not None and not df_new.empty:
            df_final = pd.concat(objs=[df_hist, df_new], ignore_index=True)
        elif df_hist is not None and not df_hist.empty:
            df_final = df_hist
        elif df_new is not None and not df_new.empty:
            df_final = df_new
        else:
            df_final = None

        if df_final is not None and not df_final.empty:
            logging.info(msg=f"   [DATA] {label} combined range: {df_final['Data'].min().date()} to {df_final['Data'].max().date()}")
        else:
            logging.error(msg=f"   [DATA] Final dataset for {label} is empty. Update failed.")
            return False
                
        df_validated = self._validate_and_clean(df=df_final, label=label)
        
        if df_validated is not None and not df_validated.empty:
            safe_name = label.replace(" ", "_").lower()
            out_path = RAW_DIR / f"{safe_name}.csv"
            
            df_validated.to_csv(path_or_buf=out_path, index=False)
            logging.info(msg=f"   [SAVE] Saved {len(df_validated)} rows to {out_path.name}")
            
            service: Any = self.gdrive.service
            if upload_to_drive and service and self.data_folder_id:
                fname = f"historia{stooq_ticker[:4] if zip_type == 'fund_pl' else stooq_ticker}.csv"
                logging.info(msg=f"   [DRIVE] Uploading {fname} to Google Drive...")
                self.gdrive.upload_csv(folder_id=self.data_folder_id, local_path=str(out_path), filename=fname)
            
            return True
            
        logging.error(msg=f"   [DATA] Validation dropped all data for {label}. Update failed.")
        return False

    def run_full_update(
        self, 
        get_funds: bool = True,
    ) -> None:
        logging.info(msg=f"Full Update Started. Funds/ETFs: {get_funds}")

        for item in DEFAULT_TICKERS:
            self.update_ticker(
                label=item["label"],
                stooq_ticker=item["stooq"],
                yf_ticker=item.get("yf"),
                knf_id=item.get("knf"),
                gpw_isin=item.get("gpw_isin"),
                zip_type=item["type"],
            )

        if get_funds:
            for item in ETF_TICKERS:
                self.update_ticker(
                    label=item["label"],
                    stooq_ticker=item["stooq"],
                    yf_ticker=item.get("yf"),
                    zip_type=item["type"],
                    upload_to_drive=True,
                )

            service: Any = self.gdrive.service
            if service and self.root_folder_id:
                df_c = self.gdrive.download_csv(
                    folder_id=self.root_folder_id, 
                    filename=CONFIRMED_FUNDS_FILE
                )
                if df_c is not None and not df_c.empty:
                    for _, row in df_c.dropna(subset=["stooq_id"]).iterrows():
                        sid = str(row["stooq_id"]).lower()
                        self.update_ticker(
                            label=f"fund_{sid}",
                            stooq_ticker=f"{sid}.n",
                            knf_id=str(row["subfundId"]),
                            zip_type="fund_pl",
                            upload_to_drive=True,
                        )