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

---


## GitHub Actions Workflows

| Workflow | Schedule | Trigger |
|---|---|---|
| `daily_strategy.yml` | 01:00 UTC daily | All assets in matrix |
| `ocr_download.yml` | 00:00 UTC daily | PPE PDF OCR |
| `fund_data_refresher.yml` | 23:00 UTC daily | TFI history update |
| `fund_reviewer.yml` | Friday 23:00 UTC | TFI fund ranking |
| `refresh_knf.yml` | Friday 01:00 UTC | KNF subfund matching |
| `research_sweep.yml` | Manual only | Parameter sweeps |
| `keepalive` | 1st of month | Repository activity |

---

## Required Secrets

| Secret | Purpose |
|---|---|
| `GDRIVE_FOLDER_ID` | Google Drive output folder |
| `GOOGLE_CREDENTIALS` | Service account JSON |
| `ZIP_URL` | PPE data source URL |
| `ZIP_PASSWORD` | PPE archive password |
| `INT_FILE_NAME` | PDF filename inside PPE archive |


---


## Quick Start (CLI)

```bash
# Daily strategy run
python moj_system/scripts/daily_runner.py --asset WIG20TR
python moj_system/scripts/daily_runner.py --asset PENSION
python moj_system/scripts/daily_runner.py --asset GLOBAL_B

# Parameter sweep (manual trigger)
python moj_system/scripts/sweep_optimizer.py --mode PENSION --n_mc 500
python moj_system/scripts/sweep_optimizer.py --mode SINGLE --assets WIG20TR SWIG80TR

# Deep robustness validation
python moj_system/scripts/validate_robustness.py --mode SINGLE --asset WIG20TR --train 8 --test 2 --stop atr --n_mc 1000 --n_boot 500

# KNF fund refresh
python moj_system/scripts/refresh_knf.py --all
---

## Configuration

All strategy parameters are in `moj_system/config.py`:

- `ASSET_REGISTRY` — per-asset source, ticker, train/test windows, default stop mode, grid overrides
- `BASE_GRIDS` / `BOND_GRIDS` — parameter search grids for equity and bond assets
- `SWEEP_WINDOW_CONFIGS` — (train_years, test_years) combinations used in sweeps
- `EQUITY_THRESHOLDS_MC` / `BOND_THRESHOLDS_MC` / `*_BOOTSTRAP` — robustness verdict thresholds per asset class

---


## System Requirements

Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-pol poppler-utils libgl1`

Windows: install Poppler and Tesseract binaries, set `POPPLER_PATH` and `TESSERACT_CMD` environment variables.

```bash
pip install -r requirements.txt
```
