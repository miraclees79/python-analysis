from strategy_test_library import (download_csv,
                                   load_csv,
                                   download_fund_navs,
                                   build_funds_df,
                                   walk_forward,
                                   compute_metrics,
                                   analyze_trades,
                                   compute_buy_and_hold,
                                   prepare_regime_inputs,
                                   run_regime_decomposition,
                                   print_backtest_report    
                                   )
from mc_robustness import (
    run_monte_carlo_robustness,
    analyze_robustness,
    extract_windows_from_wf_results,
    extract_best_params_from_wf_results,
    run_block_bootstrap_robustness,
    analyze_bootstrap
)



import os
import logging
import pandas as pd
import tempfile
import datetime as dt
import matplotlib
matplotlib.use("Agg")   # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import sys


### Initials



# Google Drive API dependencies
#from googleapiclient.discovery import build
#from google.oauth2 import service_account
#from googleapiclient.http import MediaIoBaseUpload


# Setup logging
LOG_FILE = "StratRun.log"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Remove any existing handlers — critical in notebook environments
# where the kernel may have pre-attached handlers, and where re-running
# cells accumulates duplicate handlers
root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

logging.info("Logging initialised — file: %s", LOG_FILE)

# Get the temporary directory
tmp_dir = tempfile.gettempdir()
logging.info(f"Temporary directory: {tmp_dir}")






# csv_base_url = "https://stooq.pl/q/d/l/?s={}.n&i=d"
# Base URLs
#csv_base_url = os.getenv('CSV_BASE_URL')


# List of User-Agent headers for different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/118.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_2) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"
]

# Folder ID for Google Drive upload 

#def get_folder_id(name):
#    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
#    res = drive_service.files().list(q=query, fields='files(id)').execute()
#    items = res.get('files', [])
#    return items[0]['id'] if items else None




## Set global parameters
chosen_mode = "full"
# options: "vol_entry", "vol_dynamic", "full"
VOL_WINDOW = 20
FORCE_FILTER_MODE = ["ma","mom"]
# options ["ma","mom"] ["ma"] ["mom"] ["fund"] None (fully auto)
RUN_MONTE_CARLO = True # MC parameter robustness
RUN_BLOCK_BOOTSTRAP = False # Run bootstrap robustness test

# OBJECTIVE FUNCTION
OBJECTIVE = "calmar"   # or "calmar", "sharpe", "sortino", "calmar_sortino", "calmar_sharpe"

#----------------------------------------------
# Robustness check — shorter sample
#----------------------------------------------
USE_SHORTER_SAMPLE = False
short_sample_start = "2008-01-01"
short_sample_end   = "2023-12-31"





# Main function to process the data
logging.info("RUN START: %s", dt.datetime.now())
logging.info("=" * 80)
logging.info("DOWNLOAD DATA")
logging.info("=" * 80)



# Create a temporary file inside the temp directory # Filepath for CSV
# WIG20TR - target
csv_filename_w20tr = os.path.join(tmp_dir, "w20tr.csv")

download_csv('https://stooq.pl/q/d/l/?s=wig20tr&i=d', csv_filename_w20tr)
INDEX_W20 = load_csv(csv_filename_w20tr)

df = INDEX_W20

# CASH series - Goldman Sachs Konserwatywny
csv_filename_cash = os.path.join(tmp_dir, "GS_konserw2720.csv")

# Download money market / bond fund
download_csv(
    "https://stooq.pl/q/d/l/?s=2720.n&i=d",   
    csv_filename_cash    
)

CASH = load_csv(csv_filename_cash)




# -------------------------------------------------------
# Fund NAV downloads for breadth filter
# -------------------------------------------------------

FUND_CODES = {
    "2718": "GS_Akcji",
    "3872": "Skarbiec_Akcji",
    "2847": "Investor_FundamentalnyDywWzr",
    "1422": "Allianz_Selektywny",
    "1626": "Rockbridge_Akcji",
    "4650": "Uniqa_Selektywny",
    "2869": "Ipopema_MiS",
    "4544": "Uniqa_Akcji",
    "3165": "Rockbridge_NeoAkcji",
    "3199": "Millenium_Akcji",
    "3959": "GenKorona_Akcji",
    "1056": "Superfund_Akcji",
    "3396": "PKO_Akcji",
    "3187": "Rockbridge_NeoAkcjiPL",
    "3360": "Pekao_AkcjiAktywna",
    "1137": "PZU_AkcjiPL",
    "1140": "PZU_AkcjiKrak",
    "1656": "Santander_AkcjiPL",
    "1621": "INPZU_AkcjiPL",
    "2159": "CA_Akcji",
    "1692": "SantanderPR_AkcjiPL",
    "2719": "GS_POI",
    "3151": "Esaliens_Akcji",
    "3166": "Rockbridge_NeoMid",
    "3306": "Velo_AkcjiPL",
    "3441": "Quercus_Agr",
    "1043": "Alior_Akcji"
    }

# Check for duplicate codes before downloading
seen_codes = {}
for code, name in FUND_CODES.items():
    if code in seen_codes:
        logging.warning(
            "Duplicate fund code %s — '%s' overwrites '%s'. Check FUND_CODES.",
            code, name, seen_codes[code]
        )
    seen_codes[code] = name

# Check for duplicate names
seen_names = {}
for code, name in FUND_CODES.items():
    if name in seen_names:
        logging.warning(
            "Duplicate fund name '%s' — code %s overwrites code %s. Check FUND_CODES.",
            name, code, seen_names[name]
        )
    seen_names[name] = code



if FORCE_FILTER_MODE is None or "fund" in FORCE_FILTER_MODE:
    FUND_FILES = download_fund_navs(FUND_CODES, tmp_dir)

    FUNDS = build_funds_df(
        fund_files=FUND_FILES,
        price_col="Zamkniecie",
        min_history_years=10
    ) if FUND_FILES else None

    if FUNDS is None or FUNDS.empty:
        logging.warning(
            "Fund panel unavailable — fund breadth filter will not be used."
        )
        FUNDS      = None
        FUND_PARAMS_GRID = None
    else:

 
        FUND_PARAMS_GRID = [    
            {
            "lookback_days":      30,   # medium asymmetric
            "entry_roll_thresh":  0.05,
            "entry_since_thresh": 0.08,
            "exit_roll_thresh":  -0.06,
            "exit_since_thresh": -0.10
            },
            {
            "lookback_days":      30, #strong asymmetric tight entry loose exit
            "entry_roll_thresh":  0.03,
            "entry_since_thresh": 0.05,
            "exit_roll_thresh":  -0.10,
            "exit_since_thresh": -0.15
            },
            {
            "lookback_days":      30, #original idea
            "entry_roll_thresh":  0.10,
            "entry_since_thresh": 0.15,
            "exit_roll_thresh":  -0.10,
            "exit_since_thresh": -0.15
            }
        ]
        #============================
        # Fund correlation check

        funds_df=FUNDS

        corr_matrix = funds_df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if r > 0.98:
                    high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    round(r, 4)
            ))

        if high_corr_pairs:
            logging.info("High correlation fund pairs (r>0.98):")
            for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
                logging.info("  %s / %s  r=%.4f", f1, f2, r)
else:
    FUNDS      = None 
    FUND_PARAMS_GRID = None
    
    
#============================

#===========GRIDS============
    # Define search grids once — passed to both the loop and neighbour_mean
    # so they are guaranteed to stay in sync
    X_grid = [0.08, 0.10, 0.12, 0.15, 0.20]
    Y_grid = [0.02, 0.03, 0.05, 0.07, 0.10]
    fast_grid   = [50, 75, 100]
    slow_grid = [150, 200, 250 ]
    tv_grid = [0.08, 0.10, 0.12, 0.15, 0.20]
    sl_grid     = [0.05, 0.08, 0.10, 0.15]
    mom_lookback_grid = [252]           # [126, 252]    # ADD
    
#============================

#----------------------------------------------
# Robustness check — shorter sample
#----------------------------------------------


if USE_SHORTER_SAMPLE:
    
    df_short   = df.loc[
        (df.index >= short_sample_start) &
        (df.index <= short_sample_end)
    ]
    CASH_short = CASH.loc[
        (CASH.index >= short_sample_start) &
        (CASH.index <= short_sample_end)
    ]
        
    df   = df_short      # was df.short — typo
    CASH = CASH_short    # was missing
    if FUNDS is not None:
        FUNDS = FUNDS.loc[
            (FUNDS.index >= short_sample_start) &
            (FUNDS.index <= short_sample_end)
    ]
        
    logging.info("=" * 80)
    logging.info("Diagnostic sub-sample run - does the result hold in chosen sub-sample?")
    logging.info(
        "Data start forced to %s, data end forced to %s",
        short_sample_start, short_sample_end
    )
    logging.info(
        "Sub-sample: %d rows from %s to %s",
        len(df),
        df.index.min().date(),
        df.index.max().date()
    )
    logging.info("=" * 80)
#----------------------------------------------

# ============================
# REPORTING
# ============================



# -------- Walk Forward --------

logging.info("=" * 80)
logging.info("WALK-FORWARD OPTIMIZATION")
logging.info("=" * 80)

# -------------------------------------------------------
# Walk-Forward
# -------------------------------------------------------

# Fix — initialise before the if/else block
wf_trades = pd.DataFrame()
wf_metrics = None
trade_stats = None


_cpu_count = os.cpu_count() or 1
N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

logging.info(
        "Parallel grid search: %d logical cores detected, using %d jobs.",
        _cpu_count, N_JOBS
)

wf_equity, wf_results, wf_trades = walk_forward(
    df,
    cash_df=CASH,
    train_years=8,
    test_years=2,
    selected_mode=chosen_mode,
    vol_window=VOL_WINDOW,
    funds_df=FUNDS, #DIAG RUN - None             if not using fund filter  FUNDS if using fund filter
    fund_params_grid=FUND_PARAMS_GRID,   # NEW
    filter_modes_override=FORCE_FILTER_MODE,                                        # diagnostic
    X_grid = X_grid,
    Y_grid = Y_grid,
    fast_grid   = fast_grid,
    slow_grid = slow_grid,
    tv_grid = tv_grid,
    sl_grid     = sl_grid,
    mom_lookback_grid = mom_lookback_grid,    # ADD
    objective=OBJECTIVE,
    n_jobs=N_JOBS
    
)


BOUNDARY_EXITS = {"CARRY", "SAMPLE_END"}

# --- FIX 1: was checking wf.empty (old variable name, no longer exists) ---
if wf_equity.empty or wf_results.empty:
    logging.warning("Walk-forward produced no results.")
    wf_metrics = None
    trade_stats = None

else:
    logging.info("Walk-forward Results per Window:\n%s", wf_results.to_string(index=False))

    avg = wf_results.mean(numeric_only=True)
    logging.info("Walk-forward Per-Window Averages:\n%s", avg.to_string())

    # --- FIX 2: compute metrics on the stitched OOS equity curve ---
    # Previously metrics came from a cherry-picked full-sample re-run.
    # Now we report on the actual OOS track record.
    wf_metrics = compute_metrics(wf_equity)
    wf_metrics = {k: float(v) for k, v in wf_metrics.items()}
    logging.info("="*80)
    logging.info("Stitched OOS Metrics:")
    logging.info(
        "CAGR:  %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f ",
        wf_metrics["CAGR"]*100,
        wf_metrics["Vol"]*100,
        wf_metrics["Sharpe"],
        wf_metrics["MaxDD"]*100,
        wf_metrics["CalMAR"]
    )
    logging.info("="*80)
    # --- FIX 3: trade stats from OOS trades, not full-sample trades ---
    # Exclude boundary records from trade statistics
    
    # Safe filter regardless of whether trades exist or have the column
    if not wf_trades.empty and "Exit Reason" in wf_trades.columns:
        wf_trades_closed = wf_trades[
            ~wf_trades["Exit Reason"].isin(BOUNDARY_EXITS)
            ].copy()
    else:
            wf_trades_closed = pd.DataFrame()

    trade_stats = analyze_trades(wf_trades_closed, boundary_exits=BOUNDARY_EXITS) if not wf_trades_closed.empty else None



# Buy and hold for comparison

oos_start = wf_results["TestStart"].min()
oos_end   = wf_results["TestEnd"].max()

bh_equity, bh_metrics = compute_buy_and_hold(
    df,
    price_col="Zamkniecie",
    start=oos_start,
    end=oos_end
)
logging.info("="*80)
logging.info(
    "BUY & HOLD (OOS period %s to %s): "
    "CAGR: %.2f%% | Vol: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | CalMAR: %.2f | Sortino: %.2f",
    oos_start.date(), oos_end.date(),
    bh_metrics["CAGR"]*100,
    bh_metrics["Vol"]*100,
    bh_metrics["Sharpe"],
    bh_metrics["MaxDD"]*100,
    bh_metrics["CalMAR"],
    bh_metrics["Sortino"],
    )
logging.info("="*80)

# -------------------------------------------------------
# Report
# -------------------------------------------------------

# Fix — guard the call
if wf_metrics is not None:
    print_backtest_report(
        metrics=wf_metrics,
        trades=wf_trades,
        trade_stats=trade_stats,
        wf_results=wf_results,
        position_mode=chosen_mode,
        filter_modes_override=FORCE_FILTER_MODE 
    )
else:
    logging.warning("Skipping report — no WF metrics available.")



#---------------------------------------
# Regime analysis
#-------------------------------------









# Assuming you have these from your existing strategy runner:
#   close_oos, high_oos, low_oos   — price series over the OOS window
#   strat_daily                    — daily returns series (strategy)
#   bh_daily                       — daily returns series (buy & hold)
#   equity_strat, equity_bh        — cumulative equity series

inputs = prepare_regime_inputs(df, wf_results, wf_equity, bh_equity)

results = run_regime_decomposition(**{k: inputs[k] for k in
    ["close", "high", "low",
     "daily_returns_strat", "daily_returns_bh",
     "equity_strat", "equity_bh"]})







#-----------------
# monte Carlo Robustness Check
#=============================
if RUN_MONTE_CARLO:
    logging.info("=" * 80)
    logging.info("Monte Carlo robustness check calculation")
    logging.info("=" * 80)
    
    _cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    logging.info(
        "Parallel computing for Monte Carlo: %d logical cores detected, using %d jobs.",
        _cpu_count, N_JOBS
        )

    # Extract from existing wf_results DataFrame
    windows     = extract_windows_from_wf_results(wf_results, train_years=8)
    best_params = extract_best_params_from_wf_results(wf_results)

    # Run
    mc_results = run_monte_carlo_robustness(
        best_params   = best_params,
        windows       = windows,
        df            = df,
        cash_df       = CASH,
        vol_window    = VOL_WINDOW,
        selected_mode = chosen_mode,
        funds_df      = None,        # or FUNDS if fund filter enabled
        n_samples     = 1000,
        n_jobs        = N_JOBS,
        perturb_pct   = 0.20,
        seed          = 42,
        price_col     = "Zamkniecie"
    )

    # Report
    baseline = compute_metrics(wf_equity)
    analyze_robustness(mc_results, baseline)
else:
    logging.info("=" * 80)
    logging.info("Monte Carlo robustness check skipped by user choice")
    logging.info("=" * 80)



# -------------------------------------------------------
# Block Bootstrap Robustness Check
# -------------------------------------------------------
if RUN_BLOCK_BOOTSTRAP:
    logging.info("=" * 80)
    logging.info("Block bootstrap robustness check")
    logging.info("=" * 80)

    _cpu_count = os.cpu_count() or 1
    N_JOBS = max(1, _cpu_count - 1) if _cpu_count > 3 and sys.platform == "win32" else _cpu_count

    bb_results = run_block_bootstrap_robustness(
        df               = df,
        cash_df          = CASH,
        price_col        = "Zamkniecie",
        cash_price_col   = "Zamkniecie",
        n_samples        = 500,
        block_size       = 250,
        # --- wf_kwargs: mirrors the walk_forward call above ---
        train_years              = 8,
        test_years               = 2,
        vol_window               = VOL_WINDOW,
        funds_df                 = None,
        fund_params_grid         = None,
        selected_mode            = chosen_mode,
        filter_modes_override    = FORCE_FILTER_MODE,
        X_grid                   = X_grid,
        Y_grid                   = Y_grid,
        fast_grid                = fast_grid,
        slow_grid                = slow_grid,
        tv_grid                  = tv_grid,
        sl_grid                  = sl_grid,
        mom_lookback_grid        = mom_lookback_grid   # fix — currently missing from call
        
    )

    baseline = compute_metrics(wf_equity)
    

    analyze_bootstrap(bb_results, baseline)

else:
    logging.info("=" * 80)
    logging.info("Block bootstrap robustness check skipped by user choice")
    logging.info("=" * 80)



# -------------------------------------------------------
# Plotting — switch to OOS equity and OOS trades
# -------------------------------------------------------

# --- FIX 5: was plotting bt_full/trades_full (full-sample, in-sample) ---
# Plot the stitched OOS equity curve — this is the honest track record.

if wf_equity is not None and not wf_equity.empty:

    wf_equity.index = pd.to_datetime(wf_equity.index)

    if not wf_trades.empty:
        wf_trades['EntryDate'] = pd.to_datetime(wf_trades['EntryDate'])
        wf_trades['ExitDate']  = pd.to_datetime(wf_trades['ExitDate'])

    fig, ax = plt.subplots(figsize=(16, 8))

    # OOS equity curve
    ax.plot(wf_equity.index, wf_equity.values,
            label='OOS Equity Curve', color='blue', linewidth=2)

    # Buy-and-hold benchmark — aligned to OOS period
    bh_equity_aligned = bh_equity.loc[
        (bh_equity.index >= wf_equity.index.min()) &
        (bh_equity.index <= wf_equity.index.max())
    ]
    ax.plot(bh_equity_aligned.index, bh_equity_aligned.values,
            label='Buy & Hold WIG20TR', color='grey',
            linewidth=1.5, linestyle='--')

    # Shade OOS windows for orientation
    for _, row in wf_results.iterrows():
        ax.axvspan(row["TestStart"], row["TestEnd"], color='grey', alpha=0.07)

    # Annotate trades
    if not wf_trades.empty:
        for _, trade in wf_trades.iterrows():
            color = 'green' if trade['Return'] > 0 else 'red'
            ax.axvspan(trade['EntryDate'], trade['ExitDate'], color=color, alpha=0.15)

            trade_eq = wf_equity.loc[
                (wf_equity.index >= trade['EntryDate']) &
                (wf_equity.index <= trade['ExitDate'])
            ]

            if trade_eq.empty:
                continue

            y_pos = (trade_eq.max() * 1.02 if trade['Return'] > 0
                     else trade_eq.min() * 0.98)

            ax.text(
                trade['ExitDate'], y_pos,
                f"{trade['Return']*100:.1f}%",
                color=color, fontsize=8, ha='left', va='bottom', rotation=45
            )

    # Summary comparison box
    summary_text = (
        f"Strategy:   CAGR {wf_metrics['CAGR']*100:.1f}% | "
        f"Vol {wf_metrics['Vol']*100:.1f}% | "
        f"MaxDD {wf_metrics['MaxDD']*100:.1f}%\n"
        f"Buy & Hold: CAGR {bh_metrics['CAGR']*100:.1f}% | "
        f"Vol {bh_metrics['Vol']*100:.1f}% | "
        f"MaxDD {bh_metrics['MaxDD']*100:.1f}%"
    )
    ax.text(
        0.01, 0.97, summary_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_title("Walk-Forward OOS Equity Curve vs Buy & Hold", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Equity", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    import os
    os.makedirs("outputs", exist_ok=True)
    
    plt.savefig("outputs/StrategyPlot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Plot saved to outputs/StrategyPlot.png")
    logging.info("RUN END")
    



from daily_output import build_daily_outputs

outputs = build_daily_outputs(
    wf_equity   = wf_equity,
    wf_trades   = wf_trades,
    wf_metrics  = wf_metrics,
    wf_results  = wf_results,
    bh_equity   = bh_equity,
    bh_metrics  = bh_metrics,
    df          = df,
    output_dir  = "outputs",
    price_col   = "Zamkniecie",
    gdrive_folder_id   = os.getenv("GDRIVE_FOLDER_ID"),
    gdrive_credentials = "/tmp/credentials.json"
)
# outputs["action"] → "ENTER" / "EXIT" / "HOLD"
# Write to GH Actions env for conditional email step:
# echo "ACTION=$action" >> $GITHUB_ENV