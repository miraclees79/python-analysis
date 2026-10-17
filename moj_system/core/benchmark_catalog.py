from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IndexSpec:
    key: str
    aliases: tuple[str, ...]
    data_file: str | None
    yfinance_ticker: str | None
    currency: str
    asset_type: Literal["price", "rate"]
    proxy: bool = False


INDEX_CATALOG: dict[str, IndexSpec] = {
    "WIG": IndexSpec("WIG", ("WIG",), "wig", "WIG.WA", "PLN", "price"),
    "WIG20TR": IndexSpec(
        "WIG20TR", ("WIG20TR", "WIG 20 TR", "WIG20 TR"), "wig20tr", "WIG20TR.WA", "PLN", "price"
    ),
    "WIG30TR": IndexSpec("WIG30TR", ("WIG30TR", "WIG 30 TR"), None, "WIG30TR.WA", "PLN", "price", True),
    "MWIG40TR": IndexSpec(
        "MWIG40TR",
        ("MWIG40TR", "MWIG40 TR", "MWIG40 Total Return", "MWIG40"),
        "mwig40tr",
        "MWIG40TR.WA",
        "PLN",
        "price",
    ),
    "SWIG80TR": IndexSpec(
        "SWIG80TR",
        ("SWIG80TR", "SWIG80 TR", "SWIG80 Total Return", "SWIG80"),
        "swig80tr",
        "SWIG80TR.WA",
        "PLN",
        "price",
    ),
    "WIGDIV": IndexSpec("WIGDIV", ("WIGDIV", "WIG DIV"), "etfbdivpl", None, "PLN", "price", True),
    "TBSP": IndexSpec(
        "TBSP",
        ("TBSP INDEX", "TBSP", "TREASURY BONDSPOT POLAND", "TREASURY BOND SPOT POLAND"),
        "tbsp",
        None,
        "PLN",
        "price",
    ),
    "GPWB_BWZ": IndexSpec(
        "GPWB_BWZ",
        ("GPWB BWZ", "GPWB POLAND GOVERNMENT FLOATING RATE BOND INDEX"),
        "wbbw",
        None,
        "PLN",
        "price",
        True,
    ),
    "GPWB_B1Y4Y": IndexSpec("GPWB_B1Y4Y", ("GPWB B1Y4Y",), "tbsp", None, "PLN", "price", True),
    "GPWB_B1Y3Y": IndexSpec("GPWB_B1Y3Y", ("GPWB B1Y3Y",), "tbsp", None, "PLN", "price", True),
    "GPWB_B5Y": IndexSpec("GPWB_B5Y", ("GPWB B5Y",), "tbsp", None, "PLN", "price", True),
    "GPWB_CENTR": IndexSpec("GPWB_CENTR", ("GPWB CENTR",), "tbsp", None, "PLN", "price", True),
     "MSCI_WORLD": IndexSpec(
          "MSCI_WORLD",
          ("MSCI WORLD NET TOTAL RETURN", "MSCI WORLD USD INDEX", "MSCI WORLD", "MSCITR", "NDDUWI"),
          "msci_world_combined",
          "URTH",
          "USD",
          "price",
          True,
     ),
    "MSCI_ACWI": IndexSpec(
        "MSCI_ACWI", ("MSCI ACWI", "MSCI WORLD ALL COUNTRIES"), None, "ACWI", "USD", "price", True
    ),
    "MSCI_POLAND_IMI": IndexSpec(
        "MSCI_POLAND_IMI", ("MSCI POLAND IMI LOC GROSS", "MIMLPOLG"), "wig", None, "PLN", "price", True
    ),
    "MSCI_WORLD_GROWTH": IndexSpec(
        "MSCI_WORLD_GROWTH", ("MSCI WORLD GROWTH", "M1WO000G"), None, "IWY", "USD", "price", True
    ),
    "MSCI_WORLD_MID_CAP": IndexSpec(
        "MSCI_WORLD_MID_CAP", ("MSCI WORLD MID CAP",), None, "IWR", "USD", "price", True
    ),
    "MSCI_WORLD_HEALTH_CARE": IndexSpec(
        "MSCI_WORLD_HEALTH_CARE", ("MSCI WORLD HEALTH CARE",), None, "XLV", "USD", "price", True
    ),
    "MSCI_WORLD_TECH": IndexSpec(
        "MSCI_WORLD_TECH",
        ("MSCI WORLD INFORMATION TECHNOLOGY", "MSCI WORLD IMI CYBER SECURITY"),
        None,
        "XLK",
        "USD",
        "price",
        True,
    ),
    "MSCI_WORLD_REAL_ESTATE": IndexSpec(
        "MSCI_WORLD_REAL_ESTATE", ("MSCI WORLD REAL ESTATE",), None, "VNQ", "USD", "price", True
    ),
    "MSCI_WORLD_COMMODITY": IndexSpec(
        "MSCI_WORLD_COMMODITY",
        ("MSCI WORLD COMMODITY PRODUCERS", "MSCI ACWI SELECT GOLD MINERS"),
        None,
        "GDX",
        "USD",
        "price",
        True,
    ),
    "MSCI_USA": IndexSpec(
        "MSCI_USA",
        ("MSCI USA NET TOTAL RETURN", "MSCI USA", "SPTR500N", "S&P 500", "S P 500"),
        "sp500",
        "SPY",
        "USD",
        "price",
        True,
    ),
    "MSCI_EM": IndexSpec(
        "MSCI_EM",
        ("MSCI EMERGING MARKETS NET TOTAL RETURN", "MSCI EMERGING MARKETS", "MSCI EM"),
        None,
        "EEM",
        "USD",
        "price",
        True,
    ),
    "MSCI_INDIA": IndexSpec("MSCI_INDIA", ("MSCI INDIA",), None, "INDA", "USD", "price", True),
    "MSCI_CHINA": IndexSpec("MSCI_CHINA", ("MSCI CHINA",), None, "MCHI", "USD", "price", True),
    "MSCI_TURKEY": IndexSpec("MSCI_TURKEY", ("MSCI TURKEY",), None, "TUR", "USD", "price", True),
    "MSCI_EUROPE": IndexSpec(
        "MSCI_EUROPE",
        ("MSCI EUROPE NET TOTAL RETURN", "MSCI EUROPE"),
        "stoxx600_combined",
        "^STOXX",
        "EUR",
        "price",
        True,
    ),
    "MSCI_EUROPE_IT": IndexSpec("MSCI_EUROPE_IT", ("MSCI EUROPE IT",), None, "EXV3.DE", "EUR", "price", True),
    "MSCI_US_GOVT_BOND": IndexSpec(
        "MSCI_US_GOVT_BOND",
        ("MSCI U.S. GOVERNMENT BOND INDEX", "MSCI USA GOVERNMENT BOND INDEX"),
        None,
        "SHY",
        "USD",
        "price",
        True,
    ),
    "MSCI_EURO_GOVT_BOND": IndexSpec(
        "MSCI_EURO_GOVT_BOND", ("MSCI EUROZONE GOVERNMENT BOND INDEX",), None, "IEGA.AS", "EUR", "price", True
    ),
    "MSCI_EURO_GREEN_BOND": IndexSpec(
        "MSCI_EURO_GREEN_BOND", ("BLOOMBERG MSCI EURO GREEN BOND INDEX",), None, None, "EUR", "price", True
    ),
    "BLOOMBERG_AGGREGATE": IndexSpec(
        "BLOOMBERG_AGGREGATE",
        ("BLOOMBERG GLOBAL AGGREGATE", "H00038PL", "BLOOMBERG GLOBAL AGGREGATE BOND INDEX"),
        None,
        "BNDW",
        "USD",
        "price",
        True,
    ),
    "BLOOMBERG_SERIES_E_PL_GOVT_1_5Y": IndexSpec(
        "BLOOMBERG_SERIES_E_PL_GOVT_1_5Y",
        ("BLOOMBERG SERIES E POLAND GOVT 1 5 YR BOND INDEX", "BEPDG1"),
        "tbsp",
        None,
        "PLN",
        "price",
        True,
    ),
    "BLOOMBERG_US_TREASURY_BILL": IndexSpec(
        "BLOOMBERG_US_TREASURY_BILL", ("BLOOMBERG US TREASURY BILL INDEX",), None, "BIL", "USD", "price", True
    ),
    "BLOOMBERG_COMMODITY": IndexSpec(
        "BLOOMBERG_COMMODITY", ("BLOOMBERG COMMODITY INDEX",), None, "PDBC", "USD", "price", True
    ),
    "MSCI_EUROPE_600": IndexSpec(
        "MSCI_EUROPE_600",
        ("BLOOMBERG EUROPE 600 NET RETURN INDEX", "STOXX EUROPE 600 NET RETURN"),
        "stoxx600_combined",
        "^STOXX",
        "EUR",
        "price",
        True,
    ),
    "STOXX600": IndexSpec(
        "STOXX600", ("STOXX EUROPE 600", "STOXX600"), "stoxx600_combined", "^STOXX", "EUR", "price", True
    ),
    "EUROSTOXX50": IndexSpec(
        "EUROSTOXX50", ("SX5T", "EURO STOXX 50"), None, "^STOXX50E", "EUR", "price", True
    ),
    "CDAX": IndexSpec("CDAX", ("CDAX INDEX", "CDAX"), None, "^CDAXI", "EUR", "price", True),
    "NASDAQ100": IndexSpec(
        "NASDAQ100",
        ("NASDAQ 100 USD INDEX", "NASDAQ100", "NASDAQ 100"),
        "nasdaq100",
        "^NDX",
        "USD",
        "price",
        True,
    ),
    "NIKKEI225": IndexSpec(
        "NIKKEI225", ("NIKKEI225", "NIKKEI 225"), "nikkei225", "^N225", "JPY", "price", True
    ),
    "JPM_EMBI": IndexSpec(
        "JPM_EMBI",
        ("JPMORGAN EMBI GLOBAL DIVERSIFIED", "JPM EMBI GLOBAL DIVERSIFIED"),
        None,
        "EMB",
        "USD",
        "price",
        True,
    ),
    "JPM_GBI_GLOBAL": IndexSpec(
        "JPM_GBI_GLOBAL", ("JPMORGAN GBI GLOBAL",), None, "BNDX", "USD", "price", True
    ),
    "ICE_POLAND_GOVT": IndexSpec(
        "ICE_POLAND_GOVT",
        (
            "ICE BOFA POLAND GOVERNMENT INDEX",
            "ICE BOFAML POLAND GOVERNMENT INDEX",
            "ICE BOFA 1 3 YEAR POLAND GOVERNMENT INDEX",
            "ICE BOFA ALL MATURITY POLISH GOVERNMENT INDEX",
            "G1PL",
            "GOPL",
            "G0PL",
        ),
        "tbsp",
        None,
        "PLN",
        "price",
        True,
    ),
    "ICE_EURO_CORP": IndexSpec(
        "ICE_EURO_CORP",
        ("ICE BOFA EURO CORPORATE INDEX", "ER00", "ICE BOFA 1 5 YEAR EURO CORPORATE INDEX", "EROV"),
        None,
        "IEAC.AS",
        "EUR",
        "price",
        True,
    ),
    "ICE_EURO_HIGH_YIELD": IndexSpec(
        "ICE_EURO_HIGH_YIELD",
        ("ICE BOFA EURO HIGH YIELD INDEX", "HE00"),
        None,
        "IHYG.AS",
        "EUR",
        "price",
        True,
    ),
    "ICE_EURO_GOVT": IndexSpec(
        "ICE_EURO_GOVT",
        ("ICE BOFA 1 5 YEAR EURO GOVERNMENT INDEX", "EGOV"),
        None,
        "IEGA.AS",
        "EUR",
        "price",
        True,
    ),
    "ICE_GLOBAL_GOVT_USD_HEDGED": IndexSpec(
        "ICE_GLOBAL_GOVT_USD_HEDGED",
        ("ICE BOFA 15 YEAR GLOBAL GOVERNMENT EXCLUDING JAPAN", "ICE BOFA GLOBAL GOVERNMENT INDEX", "W0G1"),
        None,
        "IGOV",
        "USD",
        "price",
        True,
    ),
    "LBMA_GOLD": IndexSpec("LBMA_GOLD", ("LBMA GOLD PRICE",), None, "GC=F", "USD", "price", True),
    "BITCOIN": IndexSpec(
        "BITCOIN", ("BLOOMBERG BITCOIN INDEX", "BITCOIN"), "btc", "BTC-USD", "USD", "price", True
    ),
    "WIBOR_ON": IndexSpec(
        "WIBOR_ON", ("WIBOR O N", "WIBOR ON", "WIBOR OVERNIGHT"), "wibor1m", None, "PLN", "rate", True
    ),
    "WIBOR_1M": IndexSpec(
        "WIBOR_1M", ("WIBOR 1M", "WIBOR1M", "WIBO1M", "WIBOR1M ASK PRICE"), "wibor1m", None, "PLN", "rate"
    ),
    "WIBOR_3M": IndexSpec(
        "WIBOR_3M", ("WIBOR 3M", "WIBOR3M", "WIBO3M", "WIBR3M"), "wibor1m", None, "PLN", "rate", True
    ),
    "WIBOR_6M": IndexSpec(
        "WIBOR_6M", ("WIBOR 6M", "WIBOR6M", "WIBO6M"), "wibor1m", None, "PLN", "rate", True
    ),
    "WIBOR_1Y": IndexSpec("WIBOR_1Y", ("WIBOR 1Y", "WIBOR1Y"), "wibor1m", None, "PLN", "rate", True),
    "WIBID_1M": IndexSpec("WIBID_1M", ("WIBID 1M", "WIBID1M"), "wibor1m", None, "PLN", "rate", True),
    "WIBID_3M": IndexSpec("WIBID_3M", ("WIBID 3M", "WIBID3M"), "wibor1m", None, "PLN", "rate", True),
    "WIBID_6M": IndexSpec("WIBID_6M", ("WIBID 6M", "WIBID6M"), "wibor1m", None, "PLN", "rate", True),
    "WIBID_ON": IndexSpec(
        "WIBID_ON", ("WIBID O N", "WIBID ON", "WIBID1M"), "wibor1m", None, "PLN", "rate", True
    ),
    "WIRON_ON": IndexSpec("WIRON_ON", ("WIRON O N", "WIRON ON"), "wibor1m", None, "PLN", "rate", True),
    "WIRON_3M": IndexSpec("WIRON_3M", ("WIRON 3M", "WIRON3M"), "wibor1m", None, "PLN", "rate", True),
    "WIRON_6M": IndexSpec("WIRON_6M", ("WIRON 6M", "WIRON6M"), "wibor1m", None, "PLN", "rate", True),
    "POLONIA": IndexSpec("POLONIA", ("POLONIA",), "wibor1m", None, "PLN", "rate", True),
    "POLSTR": IndexSpec(
        "POLSTR", ("POLSTR", "POLSTR INDEKS JEDNOPODSTAWOWY"), "wibor1m", None, "PLN", "rate", True
    ),
    "EURIBOR_1M": IndexSpec("EURIBOR_1M", ("1M EURIBOR", "EURIBOR 1M"), None, None, "EUR", "rate"),
}


def normalize_index_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", ascii_text.upper()).strip()


def resolve_index_key(text: str) -> str | None:
    normalized_text = f" {normalize_index_text(text)} "
    candidates = sorted(
        ((normalize_index_text(alias), key) for key, spec in INDEX_CATALOG.items() for alias in spec.aliases),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, key in candidates:
        if alias and f" {alias} " in normalized_text:
            return key
    return None
