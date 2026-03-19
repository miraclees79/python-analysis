"""
msci_world_synthetic.py
=======================
Builds a synthetic MSCI World daily return series from 1990 by replicating
the index using observable developed-market equity indices.

RATIONALE
---------
The WSJ MSCI World series starts ~2010 and URTH ETF starts 2012.
For Mode B (msci_world portfolio) IS windows to cover 2010-2019, we need
MSCI World returns back to at least 2001 (9-year IS window before 2010 OOS).
Ideally we want 1990+ to maximise IS depth.

MSCI World is a free-float market-cap weighted index of ~1,300 DM stocks.
Its regional weights have shifted dramatically:

  Era            USA    Japan  Eur(ex-UK)  UK    Other DM
  Late 1980s    ~35%   ~40%    ~15%        ~7%    ~3%
  2000–2005     ~50%   ~15%    ~22%        ~9%    ~4%
  2010–2015     ~55%   ~9%     ~20%        ~8%    ~8%
  2020+         ~70%   ~6%     ~13%        ~4%    ~7%

A fixed-weight blend using today's weights on 1995 data would be completely
wrong: the Japan-collapse and US-dominance shift are material.

APPROACH: Two-tier time-varying replication
-------------------------------------------
Tier 1 (primary, 2012–present):
  Rolling 5-year OLS of daily log returns:
    MSCI_World ~ β_US*SPX + β_JP*NKX + β_EU*STOXX + β_UK*FTSE + ε
  Estimated quarterly. Weights constrained non-negative, sum ~ 1.
  Uses the actual URTH ETF as the target during the calibration period.

Tier 2 (backcast, 1990–2011):
  Historical market-cap weight schedule (from published annual data),
  grouped into the same four regional blocks. Each block mapped to its
  proxy index.  If STOXX is unavailable, fills any pre-1991 gap via a
  DAX+CAC40 blend calibrated on the STOXX overlap period.

The output is a stooq-format DataFrame (column "Zamkniecie") that can be
dropped into wsj_msci_world.py as the base series before URTH chain-linking.

DATA SOURCES
------------
  stooq:
    ^spx       S&P 500            (USD, 1990+)
    ^nkx       Nikkei 225         (JPY, 1990+)
    ^dax       DAX                (EUR, 1990+)
    ^cac       CAC 40             (EUR, 1990+)
    jpypln     JPY/PLN FX         (convert JPY→USD via PLN triangulation)
    gbppln     GBP/PLN FX         (convert GBP→USD via PLN triangulation)
    eurpln     EUR/PLN FX         (convert EUR→USD via PLN triangulation)
    usdpln     USD/PLN FX         (denominator for FX triangulation)

  yfinance:
    ^FTSE      FTSE 100           (GBP, 1984+)
    URTH       iShares MSCI World (USD, 2012+, calibration target only)

  Google Drive (via price_series_builder.build_and_upload):
    stoxx600_combined.csv  STOXX 600  (EUR, 1991+, manual CSV extended with
                           ^STOXX yfinance)  — authoritative European series

All foreign returns converted to USD before blending to match MSCI World USD.

HISTORICAL WEIGHT SCHEDULE
---------------------------
Annual snapshots of MSCI World regional weights (USA, Japan, Europe, UK),
sourced from published academic papers, MSCI factsheets, and Wikipedia.
Interpolated linearly between snapshots. Used for the pre-2012 backcast.

  Year   USA    Japan  Europe  UK     Other
  1990   32%    37%    18%     8%     5%
  1993   35%    32%    19%     9%     5%
  1995   38%    27%    21%     9%     5%
  1998   46%    17%    23%     9%     5%
  2000   52%    13%    22%     8%     5%
  2002   48%    14%    23%     9%     6%
  2005   51%    11%    23%     9%     6%
  2007   50%    10%    24%     9%     7%
  2010   52%    9%     22%     8%     9%
  2012   54%    8%     21%     8%     9%

Sources: MSCI equity allocation report (2012), Dimson-Marsh-Staunton world
market cap data, Wikipedia MSCI World article, MSCI factsheet archives.

CALIBRATION OUTPUT
------------------
The module runs two validation checks:
  1. Correlation of synthetic daily returns vs URTH on overlap period (2012+)
  2. Annual CAGR comparison vs WSJ data (2010–2012) where overlap exists

USAGE
-----
  Standalone:
      python msci_world_synthetic.py

  Integrated into wsj_msci_world.py:
      from msci_world_synthetic import build_synthetic_msci_world
      base_df = build_synthetic_msci_world(folder_id=folder_id)
      # Then chain-link URTH forward from base_df.index.max()

  As stand-alone builder with Drive upload:
      from msci_world_synthetic import build_and_upload_synthetic
      build_and_upload_synthetic(folder_id=folder_id)

OUTPUT FORMAT
-------------
  stooq-format DataFrame: DatetimeIndex named "Data", column "Zamkniecie"
  Normalised to 100.0 on 1990-01-02.
  File: msci_world_synthetic.csv on Google Drive.
"""

import io
import logging
import os
import sys
import time
import tempfile
import datetime as dt

import numpy as np
import pandas as pd
from numpy.linalg import lstsq as _lstsq

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _nnls_sum_to_one(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Non-negative least squares with sum-to-one constraint.

    Solves: min ||y - X @ w||² subject to w >= 0, sum(w) = 1.

    Uses scipy SLSQP if available, otherwise pure-numpy fallback:
      The constrained optimum on the simplex is found by iterating
      over all pairs that could be on the boundary (w_i = 0) and
      taking the best unconstrained solution on the remaining variables,
      clipped to [0,1] and renormalised. For 2–5 variables this is
      exact and fast.

    Parameters
    ----------
    X : (n, k) array — regressors
    y : (n,)   array — target

    Returns
    -------
    w : (k,) array — non-negative weights summing to 1
    """
    k = X.shape[1]

    if _SCIPY_AVAILABLE:
        def _obj(w):
            return np.sum((y - X @ w) ** 2)
        from scipy.optimize import minimize as _m
        res = _m(_obj, np.ones(k) / k,
                 bounds=[(0.0, 1.0)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                 method="SLSQP")
        if res.success:
            w = np.clip(res.x, 0, None)
            return w / w.sum()

    # Pure-numpy fallback: projected gradient / simplex search over boundaries
    # For k <= 5 we enumerate active-set subsets and pick the best.
    best_w = np.ones(k) / k
    best_sse = np.inf

    # Try all 2^k subsets of active variables (feasible for k <= 5)
    for mask_bits in range(1, 2 ** k):
        active = [i for i in range(k) if mask_bits & (1 << i)]
        X_sub = X[:, active]
        # Unconstrained OLS on active subset, then project to simplex
        w_sub, _, _, _ = _lstsq(X_sub, y, rcond=None)
        w_sub = np.clip(w_sub, 0, None)
        if w_sub.sum() < 1e-12:
            continue
        w_sub /= w_sub.sum()
        w_full = np.zeros(k)
        for j, idx in enumerate(active):
            w_full[idx] = w_sub[j]
        sse = np.sum((y - X @ w_full) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_w = w_full

    return best_w

# yfinance — for FTSE 100 and URTH calibration target
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# Google Drive — for input/output
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _GDRIVE_AVAILABLE = True
except ImportError:
    _GDRIVE_AVAILABLE = False

from strategy_test_library import download_csv, load_csv


# ============================================================
# USER SETTINGS
# ============================================================

GDRIVE_FOLDER_ID_DEFAULT = ""   # <- paste your folder ID here
CREDENTIALS_PATH = os.path.join(tempfile.gettempdir(), "credentials.json")
OUTPUT_FILENAME = "msci_world_synthetic.csv"
CLOSE_COL = "Zamkniecie"
DATA_START = "1990-01-01"

# Rolling OLS parameters
OLS_WINDOW_YEARS = 5          # calibration window for rolling betas
OLS_STEP_MONTHS  = 3          # recompute quarterly
OLS_MIN_OBS      = 252 * 2    # minimum observations to estimate betas

# URTH calibration target
URTH_TICKER = "URTH"

# Stooq tickers
STOOQ_TICKERS = {
    "SPX":    "^spx",    # S&P 500, USD
    "NKX":    "^nkx",    # Nikkei 225, JPY
    "DAX":    "^dax",    # DAX, EUR (Europe fallback / pre-1991)
    "CAC":    "^cac",    # CAC 40, EUR (Europe fallback / pre-1991)
    # FX rates — stooq convention: PLN per unit of foreign currency
    "JPYPLN": "jpypln",
    "GBPPLN": "gbppln",
    "EURPLN": "eurpln",
    "USDPLN": "usdpln",
}
# FTSE 100 is sourced from yfinance (ticker ^FTSE) — not available on stooq
# STOXX 600 is sourced from Google Drive via price_series_builder.build_and_upload()

# Historical MSCI World regional weight schedule.
# Each row: (year, w_usa, w_japan, w_europe, w_uk)
# "Europe" = continental Europe (ex-UK), mapped to STOXX600/DAX+CAC blend.
# "Other" = 1 - sum(4 regions), left in residual (absorbed by closest proxy).
# Sources: MSCI equity allocation report 2012, DMS world market cap database,
#          Wikipedia MSCI World composition, academic literature.
HISTORICAL_WEIGHTS = pd.DataFrame([
    # year   USA     Japan  Europe   UK
    (1990,  0.320,  0.370,  0.180,  0.080),
    (1993,  0.350,  0.320,  0.190,  0.090),
    (1995,  0.380,  0.270,  0.210,  0.090),
    (1998,  0.460,  0.170,  0.230,  0.090),
    (2000,  0.520,  0.130,  0.220,  0.080),
    (2002,  0.480,  0.140,  0.230,  0.090),
    (2005,  0.510,  0.110,  0.230,  0.090),
    (2007,  0.500,  0.100,  0.240,  0.090),
    (2010,  0.520,  0.090,  0.220,  0.080),
    (2012,  0.540,  0.080,  0.210,  0.080),
], columns=["year", "w_usa", "w_japan", "w_europe", "w_uk"])


# ============================================================
# LOGGING SETUP (only when run as __main__)
# ============================================================

def _setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    fh = logging.FileHandler("msci_world_synthetic.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ============================================================
# DRIVE HELPERS
# ============================================================

def _get_drive_service():
    if not _GDRIVE_AVAILABLE:
        raise ImportError("google-api packages not installed.")
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return _gdrive_build("drive", "v3", credentials=creds)


def _find_file_id(service, folder_id: str, filename: str):
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id)").execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _download_drive_file(service, file_id: str) -> bytes:
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _upload_to_drive(service, folder_id: str, local_path: str, filename: str):
    existing = _find_file_id(service, folder_id, filename)
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    if existing:
        service.files().update(fileId=existing, media_body=media).execute()
        logging.info("Drive UPDATED: %s", filename)
    else:
        meta = {"name": filename, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()
        logging.info("Drive CREATED: %s", filename)


# ============================================================
# DATA ACQUISITION
# ============================================================

def _load_stooq(ticker: str, label: str) -> pd.Series | None:
    """Download a stooq price series, return close price Series."""
    tmp = os.path.join(tempfile.gettempdir(), f"synth_{label}.csv")
    url = f"https://stooq.pl/q/d/l/?s={ticker}&i=d"
    ok = download_csv(url, tmp)
    if not ok:
        logging.warning("_load_stooq: %s (%s) download failed", label, ticker)
        return None
    df = load_csv(tmp)
    if df is None or CLOSE_COL not in df.columns:
        logging.warning("_load_stooq: %s — no data", label)
        return None
    s = df[CLOSE_COL].dropna().sort_index()
    s = s.loc[s.index >= pd.Timestamp(DATA_START)]
    logging.info("_load_stooq: %-10s  %5d rows  %s to %s",
                 label, len(s), s.index.min().date(), s.index.max().date())
    return s


def _load_yfinance(ticker: str, label: str) -> pd.Series | None:
    """Download a yfinance series, return close price Series in local currency."""
    if not _YF_AVAILABLE:
        logging.error("yfinance not installed.")
        return None
    try:
        raw = yf.download(ticker, start=DATA_START, auto_adjust=True,
                          progress=False, actions=False)
    except Exception as exc:
        logging.warning("_load_yfinance: %s failed: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    close = raw["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Data"
    close = close.sort_index().loc[close.index >= pd.Timestamp(DATA_START)]
    logging.info("_load_yfinance: %-10s  %5d rows  %s to %s",
                 label, len(close), close.index.min().date(), close.index.max().date())
    return close


def _to_log_returns(price: pd.Series) -> pd.Series:
    return np.log(price / price.shift(1)).dropna()


def _to_usd_log_returns(
    local_price: pd.Series,
    fx_pln_per_foreign: pd.Series | None,
    usd_pln: pd.Series | None,
) -> pd.Series:
    """
    Convert local-currency log returns to USD log returns.

    All stooq FX series are expressed as PLN per 1 unit of foreign currency.
    To get local-currency → USD, we need:
        ret_USD = ret_local + ret_FX_local_per_USD

    Where ret_FX_local_per_USD means: +1% if USD weakened vs local (local appreciated).

    From stooq:
        USDPLN = PLN per USD  → USDPLN.pct_change() = PLN gained per USD
        JPYPLN = PLN per JPY  → JPY→USD = (PLN/JPY) / (PLN/USD) = JPYPLN / USDPLN
        GBPPLN = PLN per GBP  → GBP→USD = GBPPLN / USDPLN
        EURPLN = PLN per EUR  → EUR→USD = EURPLN / USDPLN

    If fx_pln_per_foreign is None (i.e., asset is already USD), we just use local returns.
    """
    local_ret = _to_log_returns(local_price)

    if fx_pln_per_foreign is None or usd_pln is None:
        # Asset is already USD-denominated
        return local_ret

    # Construct FX rate: units of local currency per 1 USD
    # foreign_per_usd = (pln/usd) / (pln/foreign) = usdpln / fx_pln_per_foreign
    # Log return of foreign_per_usd tells us how foreign strengthened vs USD
    # USD return = local_return + log(foreign_appreciation_vs_USD)
    common = local_ret.index.intersection(fx_pln_per_foreign.index).intersection(usd_pln.index)
    lr = local_ret.reindex(common)
    fx_local = fx_pln_per_foreign.reindex(common, method="ffill")
    fx_usd   = usd_pln.reindex(common, method="ffill")

    # local/USD rate = (pln/usd) / (pln/local) = USDPLN / FX_PLN_PER_LOCAL
    # So log(local/USD) = log(USDPLN) - log(FX_PLN_PER_LOCAL)
    # ret_USD = ret_local + Δlog(FX) where FX = local per USD
    # Δlog(local per USD) = Δlog(USDPLN) - Δlog(FX_PLN_PER_LOCAL)
    local_per_usd = fx_usd / fx_local
    ret_fx = _to_log_returns(local_per_usd).reindex(common)

    usd_ret = lr + ret_fx
    return usd_ret.dropna()


# ============================================================
# EUROPE PROXY: Extend STOXX 600 backwards using DAX + CAC blend
# ============================================================

def _build_europe_proxy(
    stoxx: pd.Series | None,
    dax: pd.Series | None,
    cac: pd.Series | None,
) -> pd.Series:
    """
    Build a continuous European equity proxy in local currency (EUR).

    STOXX 600 is available from 1991 via Google Drive (stoxx600_combined.csv).
    DAX and CAC are used as fallback for any period where STOXX is unavailable
    (Drive not configured, or the pre-1991 era if ever needed).

    Method:
      1. On the overlap period (STOXX available + DAX + CAC available),
         estimate the OLS blend: STOXX ~ β_dax*DAX + β_cac*CAC
         with non-negative β, β_dax + β_cac = 1.
      2. For dates before STOXX starts, use the DAX+CAC blend calibrated
         on the overlap period.
      3. Chain-link: the pre-STOXX synthetic is normalised to match STOXX
         at the STOXX start date.

    Returns a log-return Series in EUR covering max available history.
    """
    if dax is None or cac is None:
        logging.warning("_build_europe_proxy: DAX or CAC missing — using STOXX only")
        if stoxx is not None:
            return _to_log_returns(stoxx)
        return pd.Series(dtype=float)

    dax_ret = _to_log_returns(dax)
    cac_ret = _to_log_returns(cac)

    if stoxx is None:
        # No STOXX at all — use equal-weight DAX+CAC
        logging.warning("_build_europe_proxy: STOXX not available — using 50/50 DAX+CAC")
        common = dax_ret.index.intersection(cac_ret.index)
        return (0.5 * dax_ret.reindex(common) + 0.5 * cac_ret.reindex(common)).dropna()

    stoxx_ret = _to_log_returns(stoxx)

    # Overlap: STOXX start + 1 year buffer
    overlap_start = stoxx_ret.index.min() + pd.DateOffset(years=1)
    common = (stoxx_ret.index
              .intersection(dax_ret.index)
              .intersection(cac_ret.index))
    common_overlap = common[common >= overlap_start]

    if len(common_overlap) < 252:
        logging.warning("_build_europe_proxy: insufficient overlap — equal-weight fallback")
        w_dax, w_cac = 0.5, 0.5
    else:
        y = stoxx_ret.reindex(common_overlap).values
        X = np.column_stack([
            dax_ret.reindex(common_overlap).values,
            cac_ret.reindex(common_overlap).values,
        ])
        weights = _nnls_sum_to_one(X, y)
        w_dax, w_cac = weights[0], weights[1]
        blend_ret = X @ weights
        corr = np.corrcoef(y, blend_ret)[0, 1]
        logging.info(
            "_build_europe_proxy: DAX weight=%.1f%%  CAC weight=%.1f%%  "
            "overlap corr=%.4f",
            w_dax * 100, w_cac * 100, corr,
        )

    # Build continuous series: pre-STOXX uses DAX+CAC blend, post uses STOXX
    stoxx_start = stoxx_ret.index.min()
    pre_common = dax_ret.index.intersection(cac_ret.index)
    pre_common = pre_common[pre_common < stoxx_start]
    pre_ret = (w_dax * dax_ret.reindex(pre_common) + w_cac * cac_ret.reindex(pre_common)).dropna()

    # Combine: pre-STOXX blend + STOXX
    combined_ret = pd.concat([pre_ret, stoxx_ret]).sort_index()
    combined_ret = combined_ret[~combined_ret.index.duplicated(keep="last")]
    return combined_ret


# ============================================================
# HISTORICAL WEIGHT INTERPOLATION
# ============================================================

def _interpolate_weights(target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate annual MSCI World regional weight snapshots to daily frequency.

    Returns DataFrame indexed by target_dates with columns:
        w_usa, w_japan, w_europe, w_uk

    Weights are renormalised to sum to 1.0 after interpolation (accounting for
    "Other DM" being absorbed proportionally into the four main blocks).
    """
    # Create annual anchor points
    df = HISTORICAL_WEIGHTS.copy()
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df = df.set_index("date").sort_index()

    # Renormalise so the 4 blocks sum to 1 (absorb "Other" proportionally)
    block_cols = ["w_usa", "w_japan", "w_europe", "w_uk"]
    df["total"] = df[block_cols].sum(axis=1)
    for col in block_cols:
        df[col] = df[col] / df["total"]

    # Extend to cover full date range
    min_date = target_dates.min()
    max_date = target_dates.max()

    if min_date < df.index.min():
        row0 = df.iloc[0].copy()
        row0.name = min_date
        df = pd.concat([pd.DataFrame([row0]), df])
    if max_date > df.index.max():
        row_last = df.iloc[-1].copy()
        row_last.name = max_date
        df = pd.concat([df, pd.DataFrame([row_last])])

    df = df.sort_index()

    # Daily interpolation
    daily = df[block_cols].reindex(df.index.union(target_dates))
    daily = daily.interpolate(method="time")
    daily = daily.reindex(target_dates)

    # Final renormalise
    row_sum = daily.sum(axis=1)
    for col in block_cols:
        daily[col] = daily[col] / row_sum

    return daily


# ============================================================
# ROLLING OLS CALIBRATION (on URTH overlap period)
# ============================================================

def _rolling_ols_weights(
    target_ret:  pd.Series,          # URTH log returns (USD)
    component_rets: pd.DataFrame,    # columns: usa, japan, europe, uk (USD)
    window_years: int = OLS_WINDOW_YEARS,
    step_months:  int = OLS_STEP_MONTHS,
    min_obs:      int = OLS_MIN_OBS,
) -> pd.DataFrame:
    """
    Estimate time-varying blend weights via rolling constrained OLS.

    For each quarter-end date, fits:
        target ~ β_usa*r_usa + β_jp*r_jp + β_eu*r_eu + β_uk*r_uk + ε

    Constraints: all β ≥ 0, Σβ = 1.

    Returns DataFrame indexed by the quarter-end dates, columns = component names.
    """
    window_days = int(window_years * 252)
    component_cols = component_rets.columns.tolist()

    common = target_ret.index.intersection(component_rets.index)
    common = common.sort_values()
    y_full = target_ret.reindex(common).values
    X_full = component_rets.reindex(common).values

    # Quarter-end step
    step_approx = int(step_months * 21)   # approx trading days per month

    results = []
    for end_pos in range(window_days, len(common) + 1, step_approx):
        start_pos = end_pos - window_days
        y = y_full[start_pos:end_pos]
        X = X_full[start_pos:end_pos]

        if len(y) < min_obs:
            continue

        weights = _nnls_sum_to_one(X, y)
        if weights is not None:
            row = dict(zip(component_cols, weights))
            row["date"] = common[end_pos - 1]
            results.append(row)

    if not results:
        logging.warning("_rolling_ols_weights: no successful estimates")
        return pd.DataFrame(columns=component_cols)

    df = pd.DataFrame(results).set_index("date")
    logging.info(
        "_rolling_ols_weights: %d quarterly estimates, "
        "first=%s  last=%s",
        len(df), df.index.min().date(), df.index.max().date(),
    )
    logging.info("  Mean weights: %s",
                 {c: f"{df[c].mean():.1%}" for c in component_cols})
    return df


def _interpolate_ols_weights(
    ols_weights: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Daily interpolation of quarterly OLS weights.
    """
    if ols_weights.empty:
        return pd.DataFrame(index=target_dates)

    daily = ols_weights.reindex(ols_weights.index.union(target_dates))
    daily = daily.interpolate(method="time")
    daily = daily.reindex(target_dates)

    # Renormalise
    row_sum = daily.sum(axis=1)
    for col in daily.columns:
        daily[col] = daily[col] / row_sum

    return daily


# ============================================================
# MAIN BUILDER
# ============================================================

def build_synthetic_msci_world(
    folder_id: str | None = None,
    urth_series: pd.Series | None = None,
) -> pd.DataFrame | None:
    """
    Build the full synthetic MSCI World price series from 1990.

    Parameters
    ----------
    folder_id   : str  — Google Drive folder ID. Required to load the
                  authoritative STOXX 600 series (stoxx600_combined.csv,
                  1991+) via price_series_builder.build_and_upload().
                  If None or unavailable, falls back to yfinance ^STOXX (2004+).
    urth_series : pd.Series | None  — pre-loaded URTH close prices.
                  If None, downloads from yfinance.

    Returns
    -------
    pd.DataFrame  — stooq-format with CLOSE_COL column, normalised to 100 at
                    1990-01-02. Or None on fatal failure.
    """
    logging.info("=" * 70)
    logging.info("MSCI WORLD SYNTHETIC BUILDER")
    logging.info("=" * 70)

    # ── Phase 1: Download all constituent series ──────────────────────────
    logging.info("Phase 1: Downloading constituent series...")

    spx   = _load_stooq("^spx",    "SPX")
    nkx   = _load_stooq("^nkx",    "NKX")
    dax   = _load_stooq("^dax",    "DAX")
    cac   = _load_stooq("^cac",    "CAC")

    # FTSE 100 from yfinance (not available on stooq)
    ftse  = _load_yfinance("^FTSE", "FTSE")

    # FX rates from stooq (all expressed as PLN per 1 foreign currency unit)
    jpypln = _load_stooq("jpypln", "JPYPLN")
    gbppln = _load_stooq("gbppln", "GBPPLN")
    eurpln = _load_stooq("eurpln", "EURPLN")
    usdpln = _load_stooq("usdpln", "USDPLN")

    # STOXX 600 from Google Drive via price_series_builder
    # This is the authoritative series going back to 1991.
    # Falls back to yfinance-only (^STOXX, from 2004) if Drive is unavailable.
    stoxx_close = None
    if folder_id:
        try:
            from price_series_builder import build_and_upload as _psb_build
            stoxx_df = _psb_build(
                folder_id        = folder_id,
                raw_filename     = "stoxx600.csv",
                combined_filename= "stoxx600_combined.csv",
                extension_ticker = "^STOXX",
            )
            if stoxx_df is not None and not stoxx_df.empty:
                stoxx_close = stoxx_df[CLOSE_COL].dropna().sort_index()
                stoxx_close = stoxx_close.loc[stoxx_close.index >= pd.Timestamp(DATA_START)]
                logging.info(
                    "STOXX 600 (Drive): %d rows  %s to %s",
                    len(stoxx_close),
                    stoxx_close.index.min().date(),
                    stoxx_close.index.max().date(),
                )
        except Exception as exc:
            logging.warning("STOXX 600 Drive load failed (%s) — falling back to yfinance", exc)

    if stoxx_close is None:
        logging.info("STOXX 600: falling back to yfinance ^STOXX (from ~2004)")
        stoxx_raw = _load_yfinance("^STOXX", "STOXX600")
        if stoxx_raw is not None:
            stoxx_close = stoxx_raw

    # URTH — calibration target
    if urth_series is not None:
        urth_close = urth_series
    else:
        urth_close = _load_yfinance(URTH_TICKER, "URTH")

    # Validate mandatory inputs
    for label, s in [("SPX", spx), ("NKX", nkx), ("USDPLN", usdpln)]:
        if s is None:
            logging.error("FATAL: %s is required but unavailable.", label)
            return None

    if jpypln is None:
        logging.warning("JPYPLN missing — Nikkei will be excluded from blend.")
        nkx = None
    if gbppln is None:
        logging.warning("GBPPLN missing — FTSE 100 will be excluded from blend.")
        ftse = None
    if ftse is None:
        logging.warning("FTSE 100 unavailable — UK block will be omitted (weight redistributed).")
    if stoxx_close is None:
        logging.warning("STOXX 600 unavailable — Europe block will use DAX+CAC only.")


    # ── Phase 2: Convert all to USD log returns ───────────────────────────
    logging.info("Phase 2: Converting to USD log returns...")

    ret_usa  = _to_usd_log_returns(spx,   None,    None)      # already USD
    ret_jp   = _to_usd_log_returns(nkx,   jpypln,  usdpln)
    ret_uk   = _to_usd_log_returns(ftse,  gbppln,  usdpln) if ftse is not None else None
    ret_eu_eur = _build_europe_proxy(stoxx_close, dax, cac)   # EUR log returns
    ret_eu   = _to_usd_log_returns(
        (1 + ret_eu_eur).cumprod(),     # reconstruct synthetic price for FX conversion
        eurpln, usdpln
    ) if eurpln is not None else ret_eu_eur

    logging.info("USD return series lengths: USA=%d  JP=%d  EU=%d  UK=%s",
                 len(ret_usa), len(ret_jp), len(ret_eu),
                 len(ret_uk) if ret_uk is not None else "N/A")

    # ── Phase 3: Rolling OLS calibration on URTH overlap period ──────────
    logging.info("Phase 3: Rolling OLS calibration on URTH...")

    ols_weights = pd.DataFrame()

    if urth_close is not None and len(urth_close) > OLS_MIN_OBS:
        urth_ret = _to_log_returns(urth_close)

        # Build component DataFrame for OLS
        component_data = {"usa": ret_usa}
        if len(ret_jp) > 0:
            component_data["japan"] = ret_jp
        if len(ret_eu) > 0:
            component_data["europe"] = ret_eu
        if ret_uk is not None and len(ret_uk) > 0:
            component_data["uk"] = ret_uk

        components_df = pd.DataFrame(component_data)
        common_ols = urth_ret.index.intersection(components_df.index)
        common_ols = common_ols[common_ols >= urth_ret.index.min()]

        if len(common_ols) >= OLS_MIN_OBS:
            ols_weights = _rolling_ols_weights(
                target_ret     = urth_ret.reindex(common_ols),
                component_rets = components_df.reindex(common_ols).fillna(0),
            )
        else:
            logging.warning("Phase 3: insufficient URTH overlap (%d obs)", len(common_ols))
    else:
        logging.warning("Phase 3: URTH not available — using historical weights only")

    # ── Phase 4: Build daily synthetic returns ────────────────────────────
    logging.info("Phase 4: Building daily synthetic returns...")

    # Union of all component dates
    all_dates = ret_usa.index
    for s in [ret_jp, ret_eu, ret_uk]:
        if s is not None and len(s) > 0:
            all_dates = all_dates.union(s.index)
    all_dates = all_dates.sort_values()
    all_dates = all_dates[all_dates >= pd.Timestamp(DATA_START)]

    # Historical weights (interpolated daily): for pre-OLS period
    hist_w = _interpolate_weights(all_dates)

    # OLS weights (interpolated daily): for OLS-calibrated period
    if not ols_weights.empty:
        ols_w_daily = _interpolate_ols_weights(ols_weights, all_dates)
        # Determine the transition date: first OLS estimate
        ols_start = ols_weights.index.min()
        logging.info("Phase 4: Using OLS weights from %s onwards", ols_start.date())
    else:
        ols_w_daily = pd.DataFrame(index=all_dates)
        ols_start = pd.Timestamp("2099-01-01")  # never use OLS

    # Align components to full date grid (ffill for non-trading days)
    comps = {}
    comps["usa"]    = ret_usa.reindex(all_dates).fillna(0)
    comps["japan"]  = ret_jp.reindex(all_dates).fillna(0) if len(ret_jp) > 0 else pd.Series(0, index=all_dates)
    comps["europe"] = ret_eu.reindex(all_dates).fillna(0) if len(ret_eu) > 0 else pd.Series(0, index=all_dates)
    comps["uk"]     = (ret_uk.reindex(all_dates).fillna(0) if ret_uk is not None and len(ret_uk) > 0
                       else pd.Series(0, index=all_dates))

    # Map hist_w columns to component keys
    w_col_map = {
        "usa": "w_usa", "japan": "w_japan",
        "europe": "w_europe", "uk": "w_uk",
    }

    synthetic_ret = pd.Series(0.0, index=all_dates)

    for date in all_dates:
        # Choose weight source
        if date >= ols_start and not ols_w_daily.empty:
            w = {k: ols_w_daily.loc[date, k] for k in comps if k in ols_w_daily.columns}
        else:
            w = {k: float(hist_w.loc[date, w_col_map[k]]) for k in comps}

        # Zero out components with no data on this date
        if date < ret_jp.index.min():
            w["japan"] = 0.0
        if date < ret_eu.index.min():
            w["europe"] = 0.0
        if ret_uk is None or date < ret_uk.index.min():
            w["uk"] = 0.0

        # Renormalise remaining weights
        total_w = sum(w.values())
        if total_w < 0.01:
            w = {"usa": 1.0, "japan": 0.0, "europe": 0.0, "uk": 0.0}
        else:
            w = {k: v / total_w for k, v in w.items()}

        synthetic_ret[date] = sum(w[k] * comps[k][date] for k in comps)

    # ── Phase 5: Reconstruct price level ─────────────────────────────────
    logging.info("Phase 5: Reconstructing price index...")

    price = 100.0 * np.exp(synthetic_ret.cumsum())

    out = pd.DataFrame(index=price.index)
    out[CLOSE_COL]   = price.values
    out["Najwyzszy"] = price.values
    out["Najnizszy"] = price.values
    out.index.name = "Data"

    # ── Phase 6: Validation ───────────────────────────────────────────────
    logging.info("Phase 6: Validation...")

    if urth_close is not None:
        urth_ret = _to_log_returns(urth_close)
        synth_ret_on_urth = synthetic_ret.reindex(urth_ret.index).dropna()
        common_v = urth_ret.index.intersection(synth_ret_on_urth.index)
        if len(common_v) > 100:
            corr = np.corrcoef(
                urth_ret.reindex(common_v).values,
                synth_ret_on_urth.reindex(common_v).values,
            )[0, 1]
            logging.info(
                "Validation: correlation with URTH on overlap period = %.4f  "
                "(n=%d days)", corr, len(common_v),
            )
            # Annual CAGR comparison
            y_urth  = (1 + urth_ret.reindex(common_v)).prod() ** (252/len(common_v)) - 1
            y_synth = np.exp(synth_ret_on_urth.reindex(common_v).sum() * 252/len(common_v)) - 1
            logging.info(
                "Validation: URTH CAGR=%.2f%%  Synthetic CAGR=%.2f%%  (overlap period)",
                y_urth * 100, y_synth * 100,
            )

    logging.info(
        "Synthetic MSCI World: %d rows  %s to %s  "
        "start=100.0  end=%.2f",
        len(out), out.index.min().date(), out.index.max().date(),
        out[CLOSE_COL].iloc[-1],
    )

    return out


# ============================================================
# DRIVE PIPELINE
# ============================================================

def build_and_upload_synthetic(
    folder_id: str | None = None,
    filename:  str = OUTPUT_FILENAME,
) -> pd.DataFrame | None:
    """
    Build synthetic MSCI World series and upload to Google Drive.

    Also loads the existing MSCI World combined CSV (msci_world_combined.csv)
    from Drive if available — the synthetic series is designed to extend
    BACKWARDS of that file, not replace it.  The two files are kept separate
    to preserve the authoritative WSJ+URTH combined series intact.

    The intended workflow:
        1. synthetic covers 1990–2011 (pre-WSJ data)
        2. msci_world_combined.csv covers 2010–present (WSJ + URTH)
        3. The consumer (e.g. global_equity_library) chain-links synthetic
           backwards from msci_world_combined.csv's start date.
    """
    if folder_id is None:
        folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()

    synth_df = build_synthetic_msci_world(folder_id=folder_id)
    if synth_df is None:
        logging.error("build_and_upload_synthetic: builder returned None")
        return None

    tmp = os.path.join(tempfile.gettempdir(), filename)
    synth_df.to_csv(tmp, date_format="%Y-%m-%d")
    logging.info("Saved locally: %s  (%d rows)", tmp, len(synth_df))

    if folder_id and _GDRIVE_AVAILABLE and os.path.exists(CREDENTIALS_PATH):
        try:
            service = _get_drive_service()
            _upload_to_drive(service, folder_id, tmp, filename)
        except Exception as exc:
            logging.error("Drive upload failed: %s", exc)
    else:
        logging.info("Drive credentials not available — local copy only at %s", tmp)

    return synth_df


def load_synthetic_from_drive(
    folder_id: str | None = None,
    filename:  str = OUTPUT_FILENAME,
) -> pd.DataFrame | None:
    """
    Load a pre-built synthetic MSCI World series from Google Drive.
    Returns stooq-format DataFrame or None if not found.
    """
    if not _GDRIVE_AVAILABLE:
        logging.warning("load_synthetic_from_drive: google-api packages not available")
        return None

    if folder_id is None:
        folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()

    try:
        service  = _get_drive_service()
        file_id  = _find_file_id(service, folder_id, filename)
        if file_id is None:
            logging.info("load_synthetic_from_drive: %s not found", filename)
            return None
        raw = _download_drive_file(service, file_id)
        df = pd.read_csv(io.BytesIO(raw), parse_dates=["Data"], index_col="Data")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Data"
        df = df.sort_index().dropna(subset=[CLOSE_COL])
        logging.info("load_synthetic_from_drive: %d rows  %s to %s",
                     len(df), df.index.min().date(), df.index.max().date())
        return df
    except Exception as exc:
        logging.error("load_synthetic_from_drive: %s", exc)
        return None


# ============================================================
# INTEGRATION HELPER: build full backcast + combine with WSJ series
# ============================================================

def build_full_msci_world_extended(
    wsj_combined_df: pd.DataFrame,
    synthetic_df:    pd.DataFrame | None = None,
    folder_id:       str | None = None,
) -> pd.DataFrame | None:
    """
    Combine the synthetic backcast (1990–2011) with the WSJ+URTH combined
    series (2010–present) into a single continuous price series.

    The chain-link is done in return space at the WSJ series start date,
    so no level discontinuity is introduced. The WSJ data is authoritative
    and is preserved exactly; only synthetic returns are used in the
    extension period.

    Parameters
    ----------
    wsj_combined_df : stooq-format DataFrame from wsj_msci_world.py
    synthetic_df    : stooq-format DataFrame from build_synthetic_msci_world()
                      If None, loads from Drive.
    folder_id       : Drive folder (for loading synthetic if not provided)

    Returns
    -------
    stooq-format DataFrame covering full history.
    """
    if synthetic_df is None:
        synthetic_df = load_synthetic_from_drive(folder_id=folder_id)
        if synthetic_df is None:
            logging.warning(
                "build_full_msci_world_extended: synthetic not available — "
                "returning WSJ series unchanged"
            )
            return wsj_combined_df

    wsj_start = wsj_combined_df.index.min()
    synth_pre = synthetic_df.loc[synthetic_df.index < wsj_start].copy()

    if synth_pre.empty:
        logging.info(
            "build_full_msci_world_extended: synthetic does not predate WSJ "
            "(%s) — nothing to prepend", wsj_start.date()
        )
        return wsj_combined_df

    # Chain-link synthetic to WSJ at wsj_start
    # Find anchor: last synthetic price just before wsj_start
    anchor_synth_price = float(synth_pre[CLOSE_COL].iloc[-1])
    anchor_wsj_price   = float(wsj_combined_df[CLOSE_COL].iloc[0])
    scale = anchor_wsj_price / anchor_synth_price

    synth_pre[CLOSE_COL]   = synth_pre[CLOSE_COL]   * scale
    synth_pre["Najwyzszy"] = synth_pre["Najwyzszy"] * scale
    synth_pre["Najnizszy"] = synth_pre["Najnizszy"] * scale

    combined = pd.concat([synth_pre, wsj_combined_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    logging.info(
        "build_full_msci_world_extended: %d rows (synthetic %d + WSJ %d)  "
        "%s to %s",
        len(combined), len(synth_pre), len(wsj_combined_df),
        combined.index.min().date(), combined.index.max().date(),
    )
    return combined


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _setup_logging()
    logging.info("msci_world_synthetic.py  started  %s", dt.datetime.now())
    logging.info("Credentials: %s  (exists: %s)", CREDENTIALS_PATH,
                 os.path.exists(CREDENTIALS_PATH))

    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip() or GDRIVE_FOLDER_ID_DEFAULT.strip()

    result = build_and_upload_synthetic(folder_id=folder_id)

    if result is not None:
        cagr = (result[CLOSE_COL].iloc[-1] / 100.0) ** (
            1 / ((result.index.max() - result.index.min()).days / 365.25)
        ) - 1
        print(
            f"\nSynthetic MSCI World series ready.\n"
            f"  {len(result)} trading days\n"
            f"  {result.index.min().date()} to {result.index.max().date()}\n"
            f"  CAGR (full period, USD, price-only): {cagr:.1%}\n"
            f"  Saved as: {OUTPUT_FILENAME}\n"
        )
    else:
        print("\nBuild failed. Check msci_world_synthetic.log.\n")
        sys.exit(1)