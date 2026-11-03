# Pension Strategy — Quantitative Investment System

Automated analysis, optimisation, and monitoring system for trend-following strategies across Polish and global equity markets. Structured around a pension fund context where capital protection in downtrends is a primary evaluation criterion alongside return metrics. None of this code constitutes investment advice or any recommendation to buy, sell or refrain from buying and selling any financial instruments whatsoever.

---

## Performance Optimization (May 2026) - Tri-Engine Architecture

To eliminate severe bottlenecks during Deep Block Bootstrap validation (previously taking 12-15 hours per configuration), the core simulation module (`strategy_engine.py`) was redesigned using a **Tri-Engine Architecture**.

- **Ultra-Fast Numba Engine**: A pure numerical simulation loop compiled via LLVM (`@njit(cache=True, nogil=True)`), bypassing the Python interpreter overhead. Operates strictly on pre-extracted NumPy arrays and 64-bit primitives.
- **Performance Gain**: Execution times for strategy evaluation were reduced by **>55%** (e.g., standard Pension daily run dropped from ~8 mins to ~3.5 mins), vastly accelerating Monte Carlo and Bootstrap routines.
- **Strict Code Quality**: The refactoring introduced rigorous coding standards across the engine: mandatory keyword arguments (`kwarg=value`), full static type hinting, and strict elimination of "garbage" (`_`) variables.
- **Mathematical Equivalence**: The Numba engine is perfectly mathematically equivalent to the standard Python loop and can be dynamically toggled via the `USE_NUMBA_ENGINE` flag in `config.py`.

---

## Refactoring Summary (April 2026)

The following consolidation work was completed. All changes preserve backward-compatible strategy logic and OOS results.

**Eliminated duplication**

- `load_stooq_local()` — previously duplicated across ~13 runfiles, now `data_manager.load_local_csv()`
- `get_n_jobs()` — previously duplicated verbatim in ~8 files, now in `strategy_engine.py`
- `build_standard_two_asset_data()` — consolidates the WIG+TBSP+MMF setup block that appeared in five strategy scripts (extended MMF, spread pre-filter, yield pre-filter, bond gate, return series)
- Per-runfile `_stooq()` / `_stooq_local()` helper functions — all removed

**Shared infrastructure**

- `output_base.py` — atomic writes, Drive log pre-fetch, and `append_log_row` shared across all three daily output modules
- `utils.py` — neutral module holding `signals_to_target_weights`, `reallocation_gate`, and `build_mmf_extended` to break the circular import between `pension_engine.py` and `global_engine.py`

**Entry point consolidation**

- ~13 per-asset runfiles replaced by `daily_runner.py --asset <KEY>` with asset behaviour driven by `ASSET_REGISTRY` in `config.py`
- `sweep_optimizer.py` replaces per-asset sweep scripts with a unified `--mode [SINGLE|PENSION|GLOBAL|ALL]` interface and common OOS start enforcement across all configs

**Data layer**

- `DataUpdater` in `updater.py` replaces `stooq_hybrid_updater.py`, consolidating ZIP extraction, yfinance extension, and KNF NAV fetching into one class
- `builder.py` handles MSCI World and STOXX600 synthetic series construction (WSJ base + yfinance extension + synthetic pre-2010 chain-link)

---

## Deployed Monitoring Configs

| Asset | Config | Stop Mode | Robustness Status |
|---|---|---|---|
| WIG20TR | 6+2 | fixed | Monitoring only |
| GLOBAL_B | 7+2 | atr | Monitoring only |
| PENSION (WIG+TBSP+MMF) | 7+1 | atr | Production |

**Deployment gate**: configs must pass both MC parameter perturbation and block bootstrap before deployment consideration. MC alone is insufficient.

---

## Validation Hierarchy

Walk-forward OOS → Monte Carlo parameter perturbation → block bootstrap → assessment of investment results → deployment. Each gate must be passed sequentially.

Portfolio-level robustness is available as a separate diagnostic path through
`RobustnessEngine.run_portfolio_mc_test()` and
`RobustnessEngine.run_portfolio_bootstrap_test()`. The existing component-level
path is unchanged. Portfolio MC samples a complete parameter universe across
all legs and delegates the end-to-end simulation to the existing allocation
engine. Portfolio bootstrap samples common full-OHLC blocks for all legs and
cash, preserving cross-asset timing and supporting ATR calculations. Both
methods accept `sample_offset`, so GitHub Actions shards can run disjoint
sample ranges and merge their CSV outputs without changing the random
sequence.

The sharded validation runner writes portfolio samples separately as
`shard_*_portfolio_mc_*.csv` and `shard_*_portfolio_bootstrap_*.csv`. Merge
evaluates these files independently of the existing component-level shard
files and emits a final portfolio verdict only after both portfolio paths
have been combined.

For the PENSION portfolio bootstrap, `PL10Y`, `DE10Y`, and optional `WIBOR1M`
are included in the same joint time-block resampling panel. The bond spread
and yield-momentum gates are rebuilt from those synthetic series for every
sample.

---


## GitHub Actions Workflows

| Workflow | Schedule | Trigger |
|---|---|---|
| `daily_strategy.yml` | 02:00 UTC daily | All assets in matrix |
| `ocr_download.yml` | 00:00 UTC daily | PPE PDF OCR |
| `fund_data_refresher.yml` | 23:00 UTC daily | TFI history update |
| `fund_reviewer.yml` | Saturday 01:00 UTC | TFI fund ranking |
| `refresh_knf.yml` | Friday 01:00 UTC | KNF subfund matching |
| `research_sweep.yml` | Manual only | Parameter sweeps |
| `keepalive` | 1st of month | Repository activity |
| `sharded_validation.yml` |(PENSION, pt 22:00 UTC) | weekly revalidation |
| `sharded_validation_single.yml` | (WIG20TR, sob 22:00 UTC) | weekly revalidation |
| `sharded_validation_global.yml` | (GLOBAL_B, niedz. 22:00 UTC) | weekly revalidation |

---

## Required Secrets

| Secret | Purpose |
|---|---|
| `GDRIVE_FOLDER_ID` | Google Drive output folder |
| `GOOGLE_CREDENTIALS` | Service account JSON |
| `GEMINI_API_KEY` | Optional Gemini benchmark parsing/proxy proposals; only used when explicitly enabled |
| `ZIP_URL` | PPE data source URL |
| `ZIP_PASSWORD` | PPE archive password |
| `INT_FILE_NAME` | PDF filename inside PPE archive |


---


## E-mail reports (`--email`)

`daily_runner.py` sends no e-mail unless called with `--email`; the flag works for every asset (single, GLOBAL_*, PENSION). The report is built from the signal snapshot JSON and sent with the chart inline, using the SMTP secrets (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM`, `EMAIL_TO`). Without them the send is skipped with a log message. In `daily_strategy.yml` the flag is set only for the `PENSION` matrix entry.

## Quick Start (CLI)

```bash
# Daily strategy run
python moj_system/scripts/daily_runner.py --asset WIG20TR
python moj_system/scripts/daily_runner.py --asset PENSION
python moj_system/scripts/daily_runner.py --asset GLOBAL_B

# Same, plus HTML report e-mailed with the chart (opt-in for every asset, incl. PENSION)
python moj_system/scripts/daily_runner.py --asset PENSION --email

# Parameter sweep (manual trigger)
python moj_system/scripts/sweep_optimizer.py --mode PENSION --n_mc 500
python moj_system/scripts/sweep_optimizer.py --mode SINGLE --assets WIG20TR SWIG80TR

# Deep robustness validation
python moj_system/scripts/validate_robustness.py --mode SINGLE --asset WIG20TR --train 8 --test 2 --stop atr --n_mc 1000 --n_boot 500

# KNF fund refresh
python moj_system/scripts/refresh_knf.py --all

# Fund review with optional Gemini fallback (requires GEMINI_API_KEY)
python moj_system/scripts/fund_reviewer.py --update --use-gemini
```

## Private KSL Dashboard (prototype)

The Streamlit app monitors positions configured in `ksl_positions.json` in the
Google Drive folder identified by `GDRIVE_FOLDER_ID`. The example configuration
in `outputs/ksl_positions.example.json` is based on the Fundusze KSL overview;
review it before copying it to the private Drive folder. Positions are keyed by
account and instrument, so the same fund held in separate accounts remains
separate.

Price data is downloaded from Drive **only for the instruments of active
positions** into a volatile working directory (`KSL_DATA_DIR`, default: a temp
folder), so nothing from `moj_system/data/raw_csv/` is required:

- Stooq funds/ETFs: `historia<id>.csv` / `historia<etf>.pl.csv` from the `Dane`
  folder created by `fund_data_refresher`. It is looked up by name on the whole
  Drive (same query as `DataUpdater`), so it does not have to live inside
  `GDRIVE_FOLDER_ID`, which only needs `ksl_positions.json` and the PPE files,
- PPE: `full_ppe_history.csv` + `filtered_table.csv` (OCR) combined with the same
  `combine_ppe_history` logic as `ppe_manager`, without any upload.

The app refreshes automatically on start when the data is missing or older than
6 hours; the sidebar button **Odśwież dane z Drive** forces a refresh. Run it
locally from any directory with credentials that can access the private folder:

```bash
streamlit run moj_system/scripts/ksl_dashboard.py --server.address 127.0.0.1
```

The prototype reports entry return, drawdown from the peak since entry, KSL and
warning status, and data freshness. It does not yet reproduce every formula or
column in the existing `KSL_Overview` sheet. Do not expose the app or its service
account credentials through a public deployment.

### Daily KSL report (GitHub Actions)

`.github/workflows/ksl_report.yml` runs `python -m moj_system.scripts.ksl_report`
daily (04:00 UTC, after the data refreshers). It refreshes the data into a
temporary directory, builds `ksl_report.html`, uploads it to the Drive folder and
e-mails it; the subject carries `[KSL ALERT]` / `[KSL WARNING]` when a threshold
is breached and `[KSL] Problem z danymi` when data is missing or stale. The report
is deliberately not published as a workflow artifact because it contains private
position data. Required repository secrets: `GDRIVE_FOLDER_ID`,
`GOOGLE_CREDENTIALS`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO`
(comma-separated) and optionally `SMTP_PORT` (587 STARTTLS by default, 465 SSL)
and `EMAIL_FROM`. Locally use `--no-email` / `--no-upload` to skip those steps.

Manual benchmark corrections belong in `moj_system/data/fund_benchmark_overrides.csv`. The semicolon-delimited file uses `subfundId`, `components_json`, `kind` (`official` or `proxy`), `notes`, and optional `portfolio_spread_bps`; component keys must exist in `benchmark_catalog.py` and weights are decimal fractions summing to 1.

Gemini benchmark resolution defaults to `gemini-3.5-flash-lite`, with `gemini-3.1-flash-lite` as fallback for bulk KID parsing. Both use the configured limits of 15 RPM, 250k TPM, and 503 RPD. Proxy proposals use `gemini-3.8-flash` as a limited review model (5 RPM, 250k TPM, 20 RPD), then fall back to the Lite chain. Requests are batched in groups of 25 by default; set the GitHub repository variables `GEMINI_BULK_MODEL`, `GEMINI_REVIEW_MODEL`, `GEMINI_FALLBACK_MODELS` (comma-separated), and `GEMINI_BATCH_SIZE` to override this. The resolver retries HTTP 500/503/504 with backoff, switches models on HTTP 429, and persists its daily usage counter and result cache to Google Drive when `GDRIVE_FOLDER_ID` is available. The configured quota profile also covers `gemma-4-26b-a4b-it` (30 RPM, 16k TPM, 14,400 RPD); Gemma is not in the default chain because benchmark batches can exceed its smaller TPM limit.

The `fund_benchmark_review_*.csv` output includes `benchmark_component_tickers` and `benchmark_component_sources`, with FX series shown separately for foreign-currency components.
---

## Configuration

All strategy parameters are in `moj_system/config.py`:

- `ASSET_REGISTRY` — per-asset source, ticker, train/test windows, default stop mode, grid overrides
- `GLOBAL_ASSET_CATALOG` — global-portfolio data source, FX currency and hedge policy for each selectable asset (including BTC and ETH)
- Global portfolio variants select their equity/crypto legs with `assets`; TBSP and MMF remain shared portfolio components outside that list
- `BASE_GRIDS` / `BOND_GRIDS` — parameter search grids for equity and bond assets
- `SWEEP_WINDOW_CONFIGS` — (train_years, test_years) combinations used in sweeps
- `EQUITY_THRESHOLDS_MC` / `BOND_THRESHOLDS_MC` / `*_BOOTSTRAP` — robustness verdict thresholds per asset class

For global portfolios, `WIG` is supplied to the universe builder by each caller after the normal data update and local load. The `DataUpdater` keeps its GPW Benchmark and YFinance fallback chain for WIG; STOXX600 and MSCI World retain their Drive-history plus YFinance extension through `build_and_upload`.


---
Strategy Configuration & Management

The system utilizes a configuration-driven architecture. You do not need to write new Python logic or modify the execution engines to add or adjust standard strategies. All strategy definitions, asset characteristics, and hyperparameter search spaces are strictly centralized in a single Source of Truth.
1. Where Strategies Are Defined

All configurations live in moj_system/config.py. The system relies on three primary dictionaries:

    ASSET_REGISTRY: Defines the actual executable strategies (run modes). This is where you configure training windows, test windows, stop-loss modes, and portfolio compositions. The keys in this dictionary are the exact arguments you pass to the CLI (e.g., python moj_system/scripts/daily_runner.py --asset WIG20TR).

    GLOBAL_ASSET_CATALOG: A dictionary describing individual financial instruments used as building blocks for Multi-Asset / Global portfolios.

    GRID_SETS / BASE_GRIDS: Define the hyperparameter optimization spaces (moving averages, momentum lookbacks, target volatility, etc.).

2. How Assets Are Described

Individual assets are described in the GLOBAL_ASSET_CATALOG (in config.py). Each entry contains metadata instructing the MarketDataFacade how to fetch and process the data.
Python

"SP500": {
    "source": "local",       # Where to get data: 'local' (CSV), 'drive' (Google Drive), 'crypto'
    "ticker": "sp500",       # Local filename or Stooq ticker
    "fx": "USD",             # Base currency of the asset (for FX adjustments)
    "hedge": "portfolio"     # FX Treatment: 'portfolio' (respects strategy setting) or 'never'
}

3. How to Add a New Strategy

Depending on whether you want to run a Single-Asset trend-following strategy or a Multi-Asset Global Portfolio, follow the steps below.
Scenario A: Adding a Single-Asset Strategy

To add a new standalone index (e.g., German DAX):

Step 1: Ensure data is available (either add it to DEFAULT_TICKERS in moj_system/data/updater.py or manually place the CSV in moj_system/data/raw_csv/).
Step 2: Add a new entry to ASSET_REGISTRY in config.py:
Python

"DAX_TR": {
    "type": "single",            # Tells the pipeline to use SingleAssetPipeline
    "source": "stooq",           
    "ticker": "dax",             # Internal ticker name
    "yf_ticker": "^GDAXI",       # yfinance fallback ticker
    "train": 7,                  # In-sample Walk-Forward window (years)
    "test": 2,                   # Out-of-sample window (years)
    "default_stop": "atr"        # Use ATR-trailing stop (alternatives: 'fixed')
},

Step 3: Run it via CLI: python moj_system/scripts/daily_runner.py --asset DAX_TR
Scenario B: Adding a New Global Portfolio

To create a custom portfolio (e.g., a mix of US Tech, Polish Equities, and Bitcoin):

Step 1: Ensure all components exist in GLOBAL_ASSET_CATALOG.
Step 2: Define the new portfolio in ASSET_REGISTRY by linking the assets:
Python

"MY_CUSTOM_PORTFOLIO": {
    "type": "portfolio_global",   # Tells the pipeline to use GlobalPipeline
    "mode": "custom_tech_crypto", # Display label for reports
    "assets": ["WIG", "NASDAQ100", "BTC"], # Must match keys in GLOBAL_ASSET_CATALOG
    "train": 6,
    "test": 1,
    "fx_hedged": False,           # Set to False to expose portfolio to USD/PLN fluctuations
    "default_stop_eq": "atr",
    "asset_caps": {               # Optional: Maximum weight per asset (e.g., limit BTC to 15%)
        "WIG": 1.0, 
        "NASDAQ100": 1.0, 
        "BTC": 0.15
    },
    "min_delta": 0.05             # Reallocation gate: minimum 5% change required to execute trades
},

Step 3: Run it via CLI: python moj_system/scripts/daily_runner.py --asset MY_CUSTOM_PORTFOLIO

(Note: The bond component (TBSP) and the cash component (MMF) are automatically handled by the Global Pipeline and do not need to be listed in the "assets" array).
4. How to Modify an Existing Strategy

To tweak strategy behavior without changing the core Python codebase, edit its dictionary entry in ASSET_REGISTRY:

    Change the Stop-Loss Mode: Change "default_stop": "fixed" to "default_stop": "atr" to switch from a fixed percentage trailing stop to an ATR-scaled Chandelier exit.

    Change Walk-Forward Windows: Modify the "train" and "test" integers (e.g., from 6+2 to 7+1).

    Override Optimization Grids: If you want a specific strategy to use a different parameter search space than the global defaults, inject a "grids" dictionary:
    Python

    "WIG20TR": {
        "type": "single",
        "train": 6,
        "test": 2,
        "default_stop": "fixed",
        # This overrides the global BASE_GRIDS for this asset only:
        "grids": {
            "MOM_LB_GRID": [63, 126],        # Test shorter momentum lookbacks
            "X_GRID": [0.05, 0.10, 0.15]     # Narrower trailing stop search space
        }
    }
---


## System Requirements

Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-pol poppler-utils libgl1`

Windows: install Poppler and Tesseract binaries, set `POPPLER_PATH` and `TESSERACT_CMD` environment variables.

```bash
pip install -r requirements.txt
```
