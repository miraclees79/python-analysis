"""
wsj_msci_world.py
=================
Downloads and processes MSCI World index historical price data from the
Wall Street Journal (WSJ) market data page, index code 990100.

URL: https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices

PURPOSE
-------
The WSJ carries MSCI World (price-only, USD) data back to around 2010.
This module provides a longer pre-2012 history that supplements the
yfinance URTH ETF series (available only from 2012) for the msci_world
portfolio mode.

The combined series is:
  1990–[wsj_end]  : WSJ MSCI World price index (USD, price return only)
  [urth_start]+   : yfinance URTH ETF (USD, total return, includes dividends)

Because WSJ provides price-only and URTH is total-return (includes dividends),
the two series are chain-linked at the overlap date using a scalar multiplier
so that they form a continuous relative return series. Absolute level is
normalised to 100 at the splice point. The slight return difference in the
overlap (dividends) is documented in the log but not corrected — it is
considered immaterial for a trend-following signal which cares about
direction, not absolute level.

TWO DATA ACCESS PATHS
---------------------
Path A — Automatic download (WSJ API):
  WSJ serves historical data via a paginated JSON API. The scraper
  iterates through pages (each returning up to 12 months of data)
  until the requested start date is reached or no more data is found.
  Endpoint: https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices
  with countback / startDate / endDate query parameters.

  NOTE: WSJ uses anti-scraping measures (JS challenge, cookies, rate limits).
  The automated path uses a session with browser-like headers and short
  delays. It may fail if WSJ updates its protection. In that case use Path B.

Path B — Manual CSV load (robust fallback):
  The user downloads the CSV manually from the WSJ historical-prices page
  (click "Download" button on the page) and saves it to a known path.
  This module loads and normalises that CSV.

  WSJ CSV format (as of 2024):
    Date,Open,High,Low,Close,Volume
    12/31/24,3745.52,3748.10,3730.22,3741.88,0
  Dates are in MM/DD/YY or MM/DD/YYYY format, comma-separated.

OUTPUTS
-------
  wsj_msci_world_raw.csv     — raw data as downloaded/parsed, stooq format
  msci_world_combined.csv    — stitched WSJ + URTH series, stooq format

Both CSVs use the stooq column convention (Data, Zamkniecie) and are
suitable for direct use with load_csv() from strategy_test_library.

USAGE
-----
  # Automatic download attempt:
  from wsj_msci_world import download_wsj_msci_world, stitch_msci_world
  wsj_df = download_wsj_msci_world(start="1990-01-01", output_csv="wsj_msci_world_raw.csv")

  # Manual CSV path (preferred for reliability):
  wsj_df = load_wsj_manual_csv("path/to/wsj_download.csv")

  # Stitch with URTH:
  combined = stitch_msci_world(wsj_df, urth_df, output_csv="msci_world_combined.csv")
"""

import io
import os
import time
import random
import logging
import datetime as dt

import numpy as np
import pandas as pd
import requests

# ============================================================
# CONSTANTS
# ============================================================

CLOSE_COL   = "Zamkniecie"
DATA_START  = "1990-01-01"

WSJ_BASE_URL = (
    "https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices"
)

# WSJ JSON API endpoint — returns paginated JSON with daily OHLC.
# Parameters: startDate=YYYY-MM-DD, endDate=YYYY-MM-DD, num=<rows>
WSJ_API_URL = (
    "https://markets.wsj.com/market-data/quotes/"
    "historical/USINDICES?ticker=990100"
    "&countryCode=US&duration={duration}&startDate={start}&endDate={end}"
    "&pageSize=9999&pageNumber=1"
)

# Alternative endpoint observed in WSJ network traffic (as of 2024)
WSJ_API_URL_V2 = (
    "https://api.wsj.net/api/dylan/markets/v2/quotes/historical"
    "?ticker=XX%2F990100&countryCode=XX&startDate={start}&endDate={end}"
    "&type=Index&id=990100&country=XX&exchange=XX&journal=M"
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
]

MAX_RETRIES = 3
RETRY_DELAY = 2.0      # seconds, doubles on each retry
REQUEST_DELAY = 1.5    # seconds between page requests (polite rate limit)


# ============================================================
# PATH B — MANUAL CSV LOAD
# ============================================================

def load_wsj_manual_csv(filepath: str) -> pd.DataFrame | None:
    """
    Load and normalise a manually downloaded WSJ historical prices CSV.

    WSJ CSV download format (as of 2025):
      - Comma-separated
      - Header row: Date,Open,High,Low,Close,Volume  (exact names may vary)
      - Date format: MM/DD/YY or MM/DD/YYYY (e.g. "01/15/10" or "01/15/2010")
      - No currency symbol; prices are plain floats
      - Volume is typically 0 for indices

    Normalised output: stooq-format DataFrame with DatetimeIndex named 'Data'
    and column 'Zamkniecie' (plus Najwyzszy, Najnizszy for high/low).

    Parameters
    ----------
    filepath : str  — path to the manually saved WSJ CSV file

    Returns
    -------
    pd.DataFrame | None
    """
    if not os.path.exists(filepath):
        logging.error(
            "load_wsj_manual_csv: file not found: %s\n"
            "  Download the CSV manually from:\n"
            "  %s\n"
            "  Click the 'Download' button and save the file.",
            filepath, WSJ_BASE_URL,
        )
        return None

    logging.info("load_wsj_manual_csv: reading %s ...", filepath)

    # Try reading with different encodings — WSJ sometimes uses UTF-8-BOM
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = pd.read_csv(filepath, encoding=encoding, thousands=",")
            break
        except Exception:
            continue
    else:
        logging.error("load_wsj_manual_csv: could not read %s.", filepath)
        return None

    logging.info("  Raw columns: %s", list(raw.columns))
    logging.info("  Raw shape:   %s", raw.shape)

    # Normalise column names — WSJ uses various capitalisations
    col_map = {}
    for col in raw.columns:
        c = col.strip().lower()
        if c == "date":
            col_map[col] = "Date"
        elif c == "open":
            col_map[col] = "Open"
        elif c in ("high", "hi"):
            col_map[col] = "High"
        elif c in ("low", "lo"):
            col_map[col] = "Low"
        elif c in ("close", "price", "last"):
            col_map[col] = "Close"

    raw = raw.rename(columns=col_map)

    if "Date" not in raw.columns:
        logging.error(
            "load_wsj_manual_csv: no date column found. "
            "Columns present: %s", list(raw.columns)
        )
        return None

    if "Close" not in raw.columns:
        logging.error(
            "load_wsj_manual_csv: no close/price column found. "
            "Columns present: %s", list(raw.columns)
        )
        return None

    # Parse dates — WSJ uses MM/DD/YY and MM/DD/YYYY
    # format="mixed" handles both gracefully in pandas >= 2.0;
    # dayfirst=False ensures MM/DD/YY is parsed correctly (not DD/MM/YY)
    raw["Date"] = pd.to_datetime(raw["Date"], format="mixed", dayfirst=False, errors="coerce")
    raw = raw.dropna(subset=["Date"])

    # Build stooq-format output
    out = pd.DataFrame(index=raw["Date"])
    out.index.name = "Data"
    out.index = pd.to_datetime(out.index).tz_localize(None)

    out[CLOSE_COL] = pd.to_numeric(raw["Close"].values, errors="coerce")

    if "High" in raw.columns:
        out["Najwyzszy"] = pd.to_numeric(raw["High"].values, errors="coerce")
    else:
        out["Najwyzszy"] = out[CLOSE_COL]

    if "Low" in raw.columns:
        out["Najnizszy"] = pd.to_numeric(raw["Low"].values, errors="coerce")
    else:
        out["Najnizszy"] = out[CLOSE_COL]

    out = out.dropna(subset=[CLOSE_COL]).sort_index()
    out = out.loc[out.index >= pd.Timestamp(DATA_START)]
    out = out[~out.index.duplicated(keep="last")]

    logging.info(
        "load_wsj_manual_csv: loaded %d rows  %s to %s",
        len(out), out.index.min().date(), out.index.max().date(),
    )
    return out


# ============================================================
# PATH A — AUTOMATIC DOWNLOAD (WSJ JSON API)
# ============================================================

def _make_session() -> requests.Session:
    """Build a requests.Session with browser-like headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.wsj.com/",
        "Origin": "https://www.wsj.com",
        "Connection": "keep-alive",
    })
    return s


def _fetch_wsj_page_v2(
    session: requests.Session,
    start: str,
    end: str,
) -> list[dict] | None:
    """
    Fetch one page of WSJ MSCI World data via the v2 API.

    Returns a list of row dicts with keys: date, open, high, low, close.
    Returns None on failure.

    This endpoint is observed in WSJ browser network traffic as of 2024–2025.
    WSJ may change it without notice — the scraper should fail gracefully
    and instruct the user to use the manual CSV path.
    """
    url = WSJ_API_URL_V2.format(start=start, end=end)
    delay = RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=15)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    logging.warning(
                        "WSJ API: non-JSON response (attempt %d). "
                        "Content starts: %s",
                        attempt, resp.text[:200],
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                # Navigate the JSON structure — WSJ wraps data in nested dicts
                # Observed structure: data["data"]["instrumentTimeSeries"][0]["priceTimeSeries"]
                rows = _extract_rows_from_json(data)
                if rows is not None:
                    return rows

                logging.warning(
                    "WSJ API: unexpected JSON structure (attempt %d). "
                    "Keys at root: %s",
                    attempt, list(data.keys()) if isinstance(data, dict) else type(data),
                )

            elif resp.status_code == 403:
                logging.warning(
                    "WSJ API: 403 Forbidden — anti-scraping protection active. "
                    "Use the manual CSV download path instead."
                )
                return None  # Non-retryable

            elif resp.status_code == 429:
                logging.warning("WSJ API: rate limited (429). Waiting before retry.")
                time.sleep(delay * 3)
                delay *= 2
                continue

            else:
                logging.warning(
                    "WSJ API: HTTP %d (attempt %d)", resp.status_code, attempt
                )

        except requests.exceptions.Timeout:
            logging.warning("WSJ API: timeout (attempt %d)", attempt)
        except requests.exceptions.RequestException as exc:
            logging.warning("WSJ API: request error %s (attempt %d)", exc, attempt)

        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    logging.error("WSJ API: all %d attempts failed for %s to %s.", MAX_RETRIES, start, end)
    return None


def _extract_rows_from_json(data: dict) -> list[dict] | None:
    """
    Navigate WSJ JSON response structure to extract OHLC rows.

    WSJ has changed its JSON structure multiple times. This function
    tries several known structures and returns the first that works.

    Returns list of dicts with keys: date (str YYYY-MM-DD), close (float),
    open (float), high (float), low (float). Returns None if no known
    structure matches.
    """
    if not isinstance(data, dict):
        return None

    # --- Structure 1: data["data"]["instrumentTimeSeries"][0]["priceTimeSeries"] ---
    try:
        ts = data["data"]["instrumentTimeSeries"][0]["priceTimeSeries"]
        rows = []
        for entry in ts:
            date_ms = entry.get("dateTime") or entry.get("date")
            if date_ms is None:
                continue
            # dateTime may be milliseconds epoch or ISO string
            if isinstance(date_ms, (int, float)):
                date_str = pd.Timestamp(date_ms, unit="ms").strftime("%Y-%m-%d")
            else:
                date_str = str(date_ms)[:10]  # take YYYY-MM-DD prefix
            close = entry.get("value") or entry.get("close") or entry.get("price")
            if close is None:
                continue
            rows.append({
                "date":  date_str,
                "close": float(close),
                "open":  float(entry.get("open",  close)),
                "high":  float(entry.get("high",  close)),
                "low":   float(entry.get("low",   close)),
            })
        if rows:
            return rows
    except (KeyError, IndexError, TypeError):
        pass

    # --- Structure 2: data["PricePeriods"] (older WSJ API) ---
    try:
        periods = data["PricePeriods"]
        rows = []
        for p in periods:
            rows.append({
                "date":  str(p.get("Date", p.get("date", "")))[:10],
                "close": float(p.get("Price", p.get("Close", p.get("close", 0)))),
                "open":  float(p.get("Open",  p.get("open",  0))),
                "high":  float(p.get("High",  p.get("high",  0))),
                "low":   float(p.get("Low",   p.get("low",   0))),
            })
        if rows:
            return rows
    except (KeyError, TypeError):
        pass

    # --- Structure 3: flat list of OHLC dicts ---
    try:
        if isinstance(data, list) and len(data) > 0 and "close" in data[0]:
            rows = []
            for p in data:
                rows.append({
                    "date":  str(p.get("date", p.get("Date", "")))[:10],
                    "close": float(p.get("close", p.get("Close", 0))),
                    "open":  float(p.get("open",  p.get("Open",  0))),
                    "high":  float(p.get("high",  p.get("High",  0))),
                    "low":   float(p.get("low",   p.get("Low",   0))),
                })
            if rows:
                return rows
    except (KeyError, TypeError):
        pass

    return None


def download_wsj_msci_world(
    start: str = DATA_START,
    end:   str | None = None,
    output_csv: str | None = "wsj_msci_world_raw.csv",
) -> pd.DataFrame | None:
    """
    Download MSCI World historical price data from WSJ.

    Fetches data from the WSJ API in annual chunks from `start` to `end`,
    building a complete daily price series. Saves the result to `output_csv`
    in stooq format if specified.

    IMPORTANT: WSJ uses anti-scraping protection. If this function returns
    None or logs 403/non-JSON errors, switch to the manual CSV path:
      1. Visit: https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices
      2. Set the date range and click "Download"
      3. Save the file and pass the path to load_wsj_manual_csv()

    Parameters
    ----------
    start      : str       — start date "YYYY-MM-DD" (default DATA_START)
    end        : str|None  — end date "YYYY-MM-DD" (default: today)
    output_csv : str|None  — if set, saves raw result to this path

    Returns
    -------
    pd.DataFrame | None  — stooq-format DataFrame or None on failure
    """
    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    logging.info(
        "download_wsj_msci_world: requesting %s to %s ...", start, end
    )

    session = _make_session()

    # Fetch in annual chunks to reduce per-request size and respect rate limits
    all_rows: list[dict] = []
    current_start = pd.Timestamp(start)
    end_ts        = pd.Timestamp(end)

    while current_start <= end_ts:
        chunk_end = min(
            current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1),
            end_ts,
        )
        start_str = current_start.strftime("%Y-%m-%d")
        end_str   = chunk_end.strftime("%Y-%m-%d")

        logging.info("  Fetching chunk: %s to %s ...", start_str, end_str)
        rows = _fetch_wsj_page_v2(session, start_str, end_str)

        if rows is None:
            logging.error(
                "  WSJ API failed for chunk %s–%s. "
                "Switching to manual CSV path is recommended.",
                start_str, end_str,
            )
            # Return whatever we have so far (partial), or None if nothing
            if not all_rows:
                return None
            logging.warning(
                "  Returning partial data (%d rows) up to last successful chunk.",
                len(all_rows),
            )
            break

        all_rows.extend(rows)
        logging.info("  Chunk returned %d rows (cumulative: %d)", len(rows), len(all_rows))

        current_start = chunk_end + pd.DateOffset(days=1)
        time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))

    if not all_rows:
        logging.error("download_wsj_msci_world: no data retrieved.")
        return None

    df = _rows_to_stooq_df(all_rows)

    if df is None or df.empty:
        logging.error("download_wsj_msci_world: failed to build DataFrame from rows.")
        return None

    if output_csv:
        _save_stooq_csv(df, output_csv)

    logging.info(
        "download_wsj_msci_world: %d rows  %s to %s",
        len(df), df.index.min().date(), df.index.max().date(),
    )
    return df


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _rows_to_stooq_df(rows: list[dict]) -> pd.DataFrame | None:
    """Convert a list of OHLC row dicts to a stooq-format DataFrame."""
    if not rows:
        return None

    raw = pd.DataFrame(rows)

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last")

    out = pd.DataFrame(index=raw["date"])
    out.index.name  = "Data"
    out.index       = pd.to_datetime(out.index).tz_localize(None)
    out[CLOSE_COL]  = pd.to_numeric(raw["close"].values, errors="coerce")
    out["Najwyzszy"] = pd.to_numeric(raw["high"].values,  errors="coerce")
    out["Najnizszy"] = pd.to_numeric(raw["low"].values,   errors="coerce")

    # Fallback: if high/low are zero or NaN, use close
    out["Najwyzszy"] = out["Najwyzszy"].where(out["Najwyzszy"] > 0, out[CLOSE_COL])
    out["Najnizszy"] = out["Najnizszy"].where(out["Najnizszy"] > 0, out[CLOSE_COL])

    out = out.dropna(subset=[CLOSE_COL])
    out = out.loc[out.index >= pd.Timestamp(DATA_START)]

    return out


def _save_stooq_csv(df: pd.DataFrame, path: str) -> None:
    """Save a stooq-format DataFrame to CSV."""
    out = df.copy()
    out.index.name = "Data"
    out.to_csv(path, date_format="%Y-%m-%d")
    logging.info("Saved: %s  (%d rows)", path, len(df))


# ============================================================
# STITCH WSJ + URTH
# ============================================================

def stitch_msci_world(
    wsj_df:     pd.DataFrame,
    urth_df:    pd.DataFrame,
    output_csv: str | None = "msci_world_combined.csv",
) -> pd.DataFrame | None:
    """
    Stitch WSJ MSCI World (price-return, USD) and URTH (total-return, USD)
    into a single continuous price series in stooq format.

    Method
    ------
    1. Find the overlap period between the two series.
    2. Compute a scalar multiplier = mean(wsj_close / urth_close) over the
       overlap window (minimum 5 trading days required).
    3. Rescale WSJ data by the multiplier so levels match at the splice.
    4. Use WSJ for dates before the URTH start; use URTH for dates from
       the URTH start onwards.
    5. Normalise the combined series to start at 100 on the first date.

    The splice introduces a small level discontinuity (the dividend wedge
    between price-return WSJ and total-return URTH over the overlap). This
    is logged but not corrected — it is immaterial for trend signals.

    Parameters
    ----------
    wsj_df     : pd.DataFrame  — stooq-format, WSJ MSCI World (USD)
    urth_df    : pd.DataFrame  — stooq-format, yfinance URTH ETF (USD)
    output_csv : str | None    — if set, saves combined series to this path

    Returns
    -------
    pd.DataFrame | None  — combined stooq-format series, or None on failure
    """
    if wsj_df is None or wsj_df.empty:
        logging.error("stitch_msci_world: wsj_df is None or empty.")
        return None
    if urth_df is None or urth_df.empty:
        logging.error("stitch_msci_world: urth_df is None or empty.")
        return None

    wsj_close  = wsj_df[CLOSE_COL].dropna().sort_index()
    urth_close = urth_df[CLOSE_COL].dropna().sort_index()

    urth_start = urth_close.index.min()
    wsj_end    = wsj_close.index.max()

    # --- Find overlap ---
    overlap_start = max(wsj_close.index.min(), urth_start)
    overlap_end   = min(wsj_end, urth_close.index.max())

    if overlap_start > overlap_end:
        logging.warning(
            "stitch_msci_world: no overlap between WSJ (%s–%s) "
            "and URTH (%s–%s). Concatenating without rescaling.",
            wsj_close.index.min().date(), wsj_end.date(),
            urth_start.date(), urth_close.index.max().date(),
        )
        multiplier = 1.0
    else:
        overlap_wsj  = wsj_close.loc[overlap_start:overlap_end]
        overlap_urth = urth_close.reindex(overlap_wsj.index, method="ffill")

        # Filter to days where both have valid prices
        mask         = overlap_urth.notna() & overlap_wsj.notna() & (overlap_urth > 0)
        overlap_wsj  = overlap_wsj[mask]
        overlap_urth = overlap_urth[mask]

        if len(overlap_wsj) < 5:
            logging.warning(
                "stitch_msci_world: overlap too short (%d days). "
                "Using last WSJ / first URTH prices for splice.",
                len(overlap_wsj),
            )
            # Use the last WSJ price vs first URTH price
            multiplier = wsj_close.iloc[-1] / urth_close.iloc[0]
        else:
            # Scale factor: WSJ levels / URTH levels, averaged over overlap
            ratios     = overlap_wsj.values / overlap_urth.values
            multiplier = float(np.median(ratios))
            logging.info(
                "Splice: overlap=%d days  (%s to %s)  "
                "multiplier=%.6f  (WSJ/URTH ratio: median=%.6f  "
                "min=%.6f  max=%.6f)",
                len(overlap_wsj),
                overlap_start.date(), overlap_end.date(),
                multiplier,
                multiplier,
                float(ratios.min()),
                float(ratios.max()),
            )
            # Warn if the ratio is unstable (suggests series mismatch)
            cv = float(np.std(ratios) / np.mean(ratios))
            if cv > 0.05:
                logging.warning(
                    "Splice: ratio coefficient of variation = %.2f%% > 5%%. "
                    "WSJ and URTH may not track the same index closely. "
                    "Review the overlap period manually.",
                    cv * 100,
                )

    # --- Build combined series ---
    # Pre-splice: WSJ prices rescaled by multiplier (so URTH levels = WSJ scaled)
    # Post-splice: URTH prices as-is
    #
    # We splice AT urth_start: use WSJ for everything strictly before urth_start,
    # then URTH from urth_start onwards.

    wsj_pre  = wsj_close.loc[wsj_close.index < urth_start] * (1.0 / multiplier)
    combined = pd.concat([wsj_pre, urth_close]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Normalise to 100 at first date
    combined = combined / combined.iloc[0] * 100.0
    combined = combined.dropna()

    # Build stooq-format output — high/low set to close (no intraday for stitched series)
    out = pd.DataFrame({
        CLOSE_COL:   combined,
        "Najwyzszy": combined,
        "Najnizszy": combined,
    })
    out.index.name = "Data"

    if output_csv:
        _save_stooq_csv(out, output_csv)

    logging.info(
        "stitch_msci_world: combined series  %d rows  %s to %s  "
        "splice at %s",
        len(out),
        out.index.min().date(), out.index.max().date(),
        urth_start.date(),
    )
    return out


# ============================================================
# CONVENIENCE: LOAD COMBINED FROM SAVED CSV
# ============================================================

def load_combined_csv(filepath: str) -> pd.DataFrame | None:
    """
    Load a previously saved combined MSCI World CSV (stooq format).

    This is the recommended path for daily runners — build the combined
    series once, save it, and load from the saved file on subsequent runs.

    Parameters
    ----------
    filepath : str  — path to msci_world_combined.csv

    Returns
    -------
    pd.DataFrame | None
    """
    if not os.path.exists(filepath):
        logging.error("load_combined_csv: file not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["Data"], index_col="Data")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().dropna(subset=[CLOSE_COL])
        df = df.loc[df.index >= pd.Timestamp(DATA_START)]
        logging.info(
            "load_combined_csv: %d rows  %s to %s",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df
    except Exception as exc:
        logging.error("load_combined_csv: error reading %s: %s", filepath, exc)
        return None


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    """
    Run as a script to build the combined MSCI World series.

    Usage:
      # Automatic download attempt (may be blocked by WSJ):
      python wsj_msci_world.py

      # Manual CSV path (robust):
      python wsj_msci_world.py --manual path/to/wsj_download.csv

      # Specify start/end:
      python wsj_msci_world.py --start 1990-01-01 --end 2012-12-31
    """
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("wsj_msci_world.log", mode="w"),
        ],
    )

    parser = argparse.ArgumentParser(description="Build MSCI World combined series.")
    parser.add_argument("--manual",  default=None,       help="Path to manually saved WSJ CSV")
    parser.add_argument("--start",   default=DATA_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=None,       help="End date YYYY-MM-DD")
    parser.add_argument("--urth",    default=None,       help="Path to saved URTH CSV (stooq format)")
    parser.add_argument("--out-raw", default="wsj_msci_world_raw.csv")
    parser.add_argument("--out-combined", default="msci_world_combined.csv")
    args = parser.parse_args()

    # ── Load or download WSJ data ──────────────────────────────────────────
    if args.manual:
        wsj_df = load_wsj_manual_csv(args.manual)
    else:
        wsj_df = download_wsj_msci_world(
            start=args.start,
            end=args.end,
            output_csv=args.out_raw,
        )

    if wsj_df is None:
        print(
            "\nWSJ data not available. To use the manual path:\n"
            "  1. Visit: https://www.wsj.com/market-data/quotes/index/XX/990100/historical-prices\n"
            "  2. Set the date range to your desired start and today\n"
            "  3. Click 'Download' and save the CSV file\n"
            "  4. Run: python wsj_msci_world.py --manual path/to/file.csv\n"
        )
        sys.exit(1)

    # ── Load URTH (yfinance) ───────────────────────────────────────────────
    if args.urth:
        urth_df = load_combined_csv(args.urth)  # loads stooq format CSV
    else:
        try:
            import yfinance as yf
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from global_equity_library import download_yfinance
            urth_df = download_yfinance("URTH")
            if urth_df is None:
                urth_df = download_yfinance("IWDA.L")
        except ImportError:
            print("yfinance not available. Install with: pip install yfinance")
            urth_df = None

    if urth_df is None:
        print(
            "\nUnable to download URTH from yfinance.\n"
            "Either install yfinance (pip install yfinance) or supply a\n"
            "pre-saved URTH CSV with --urth path/to/urth.csv\n"
        )
        sys.exit(1)

    # ── Stitch and save ────────────────────────────────────────────────────
    combined = stitch_msci_world(wsj_df, urth_df, output_csv=args.out_combined)

    if combined is None:
        print("Failed to build combined series.")
        sys.exit(1)

    print(
        f"\nCombined MSCI World series built successfully:\n"
        f"  {len(combined)} trading days\n"
        f"  {combined.index.min().date()} to {combined.index.max().date()}\n"
        f"  Saved to: {args.out_combined}\n"
    )
