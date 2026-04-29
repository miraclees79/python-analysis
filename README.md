# Pension Strategy — Quantitative Investment System

Automated analysis, optimisation, and monitoring system for trend-following strategies across Polish and global equity markets. Structured around a pension fund context where capital protection in downtrends is a primary evaluation criterion alongside return metrics. None of this code constitutes investment advice or any recommendation to buy, sell or refrain from buying and selling any financial instruments whatsoever.

---

## Architecture

The codebase was fully refactored in April 2026 from a collection of per-asset runfiles into a modular package structure. The refactoring consolidated duplicated logic, introduced shared infrastructure modules, and added a universal CLI entry point.

```
moj_system/
├── config.py                   # Central registry: asset configs, parameter grids, thresholds
├── core/
│   ├── strategy_engine.py      # Walk-forward, run_strategy_with_trades, compute_metrics
│   ├── pension_engine.py       # Multi-asset WIG+TBSP+MMF allocation layer
│   ├── global_engine.py        # N-asset global equity framework, FX handling
│   ├── robustness_engine.py    # Monte Carlo perturbation, block bootstrap
│   ├── robustness.py           # RobustnessEngine wrapper class
│   ├── fund_analytics.py       # OLS regression, IR, hit rate for fund panel
│   ├── research.py             # Common OOS start calculation, result ranking
│   └── utils.py                # Shared helpers: reallocation gate, MMF extension,
│                               #   signals_to_target_weights (breaks circular imports)
├── data/
│   ├── updater.py              # Hybrid updater: ZIP extraction + yfinance + KNF API
│   ├── data_manager.py         # load_local_csv (replaces load_stooq_local)
│   ├── builder.py              # MSCI World / STOXX600 series construction from Drive
│   ├── gdrive.py               # GDriveClient
│   ├── knf_tools.py            # KNF API, fuzzy matching, price verification
│   └── ocr_processor.py        # PPE PDF OCR pipeline
├── reporting/
│   ├── output_base.py          # Shared infrastructure: atomic writes, Drive pre-fetch
│   ├── daily_output.py         # Single-asset daily artefacts
│   ├── multiasset_daily_output.py   # Pension portfolio daily artefacts
│   └── global_equity_daily_output.py # Global equity daily artefacts
└── scripts/
    ├── daily_runner.py         # Universal entry point for all strategies
    ├── sweep_optimizer.py      # Multi-config parameter sweep with MC
    ├── validate_robustness.py  # Deep MC + bootstrap validation
    ├── objective_benchmarker.py # Annual objective function review
    ├── fund_reviewer.py        # TFI fund ranking pipeline
    └── refresh_knf.py          # KNF subfund refresh and matching
outputs/                        # Run artefacts (git-ignored)
```

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

**Known engineering issues resolved**

- `USE_ATR_STOP = False` discrepancy in `twoasset_robustness.py` — file replaced by `validate_robustness.py`
- Broken merge artifact (`NameError` from concatenated `_stooq()` + `load_stooq_local()`) in `global_equity_runfile_v2.py` — file replaced
- Logging gap in `twoasset_extended_comparison.py` Drive-loaded TBSP path — file replaced
- `build_signal_series(wf_bd, _)` bug in `sweep_optimizer.py` and `objective_benchmarker.py` (wrong trades DataFrame passed due to `_` collision across unpacks) — fixed with explicit variable naming throughout

---

## Deployed Monitoring Configs

| Asset | Config | Stop Mode | Robustness Status |
|---|---|---|---|
| WIG20TR | 6+2 | fixed | Pending verification |
| GLOBAL_B | 7+2 | atr | Pending verification |
| PENSION (WIG+TBSP+MMF) | — | — | Production |

**Deployment gate**: configs must pass both MC parameter perturbation and block bootstrap before deployment consideration. MC alone is insufficient.



---

## Validation Hierarchy

Walk-forward OOS → Monte Carlo parameter perturbation → block bootstrap → deployment. Each gate must be passed sequentially.

A full parameter sweep across train/test window configurations and stop modes (fixed vs ATR) is currently in progress to reconfirm deployed configs under the refactored codebase.

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
```

---

## Configuration

All strategy parameters are in `moj_system/config.py`:

- `ASSET_REGISTRY` — per-asset source, ticker, train/test windows, default stop mode, grid overrides
- `BASE_GRIDS` / `BOND_GRIDS` — parameter search grids for equity and bond assets
- `SWEEP_WINDOW_CONFIGS` — (train_years, test_years) combinations used in sweeps
- `EQUITY_THRESHOLDS_MC` / `BOND_THRESHOLDS_MC` / `*_BOOTSTRAP` — robustness verdict thresholds per asset class

---

## GitHub Actions Workflows

| Workflow | Schedule | Trigger |
|---|---|---|
| `daily_strategy.yml` | 00:00 UTC daily | All assets in matrix |
| `ocr_download.yml` | 23:00 UTC daily | PPE PDF OCR |
| `fund_reviewer.yml` | Monday 04:00 UTC | TFI fund ranking |
| `refresh_knf.yml` | Monday 01:00 UTC | KNF subfund matching |
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
| `FOLDER_NAME` | Drive folder for OCR output |

---

## System Requirements

Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-pol poppler-utils libgl1`

Windows: install Poppler and Tesseract binaries, set `POPPLER_PATH` and `TESSERACT_CMD` environment variables.

```bash
pip install -r requirements.txt
```
