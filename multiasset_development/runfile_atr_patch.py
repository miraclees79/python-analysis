"""
runfile_atr_patch.py
====================
This file documents the exact changes required in the two runfiles.
It is NOT meant to be run directly — it is a reference for the diff.

CHANGES REQUIRED IN:
  - multiasset_runfile.py
  - multiasset_daily_runfile.py

Both files require identical changes in two places:
  1. USER SETTINGS block — add ATR parameters
  2. walk_forward() call for equity (Phase 2) — pass ATR parameters
  3. walk_forward() call for bond (Phase 3) — pass ATR parameters
     (ATR stop can be configured independently per asset if desired)
"""

# ============================================================
# CHANGE 1: USER SETTINGS block
# ============================================================
#
# ADD this block immediately after the existing X_GRID_EQ definition.
# The new block sits between the equity grids and the bond signal source
# section in both runfiles.
#
# BEFORE (existing lines, do not delete):
#     X_GRID_EQ   = [0.08, 0.10, 0.12, 0.15, 0.20]
#     Y_GRID_EQ   = [0.02, 0.03, 0.05, 0.07, 0.10]
#     FAST_EQ     = [50, 75, 100]
#     SLOW_EQ     = [150, 200, 250]
#     TV_EQ       = [0.08, 0.10, 0.12, 0.15, 0.20]
#     SL_EQ       = [0.05, 0.08, 0.10, 0.15]
#     MOM_LB_EQ   = [126, 252]
#
# AFTER (add the new block immediately below MOM_LB_EQ):

ATR_SETTINGS_BLOCK = """
# --- Trailing stop mode ---
# USE_ATR_STOP = False : fixed percentage trailing stop (current default)
#     X_GRID_EQ is used; stop fires when price < (1 - X) * peak
# USE_ATR_STOP = True  : ATR-scaled Chandelier exit
#     N_ATR_GRID is used; stop fires when price < peak - N * ATR
#     ATR = rolling mean of |daily price change| over ATR_WINDOW bars
#     Recommended N_ATR range for WIG: 3–6 (wider = more room to breathe)
#     ATR_WINDOW: 20 matches VOL_WINDOW; increase to 40–60 for a slower ATR
#
# The bond walk-forward (Phase 3) uses a separate flag USE_ATR_STOP_BD
# so equity and bond stops can be tuned independently.
# Set USE_ATR_STOP_BD = USE_ATR_STOP to keep them in sync.
#
USE_ATR_STOP    = False          # Equity trailing stop mode
ATR_WINDOW      = 20             # Rolling window for ATR estimate (days)
N_ATR_GRID      = [2.0, 3.0, 4.0, 5.0, 6.0]   # Multiplier grid for IS search

USE_ATR_STOP_BD = False          # Bond trailing stop mode (can differ from equity)
ATR_WINDOW_BD   = 20
N_ATR_GRID_BD   = [2.0, 3.0, 4.0, 5.0, 6.0]
"""

# ============================================================
# CHANGE 2: walk_forward call for equity (Phase 2)
# ============================================================
#
# In both runfiles, find the equity walk_forward call and add three kwargs.
#
# BEFORE (end of existing equity walk_forward call):
#     n_jobs                = N_JOBS,
#     fast_mode=FAST_MODE    # (multiasset_runfile.py only; daily runfile uses fast_mode=True)
# )
#
# AFTER:
#     n_jobs                = N_JOBS,
#     fast_mode=FAST_MODE,         # (or fast_mode=True in daily runfile)
#     use_atr_stop          = USE_ATR_STOP,
#     N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
#     atr_window            = ATR_WINDOW,
# )

# ============================================================
# CHANGE 3: walk_forward call for bond (Phase 3)
# ============================================================
#
# BEFORE (end of existing bond walk_forward call):
#     n_jobs                = N_JOBS,
#     entry_gate_series     = bond_entry_gate,
#     fast_mode=FAST_MODE    # (multiasset_runfile.py only)
# )
#
# AFTER:
#     n_jobs                = N_JOBS,
#     entry_gate_series     = bond_entry_gate,
#     fast_mode=FAST_MODE,         # (or fast_mode=True in daily runfile)
#     use_atr_stop          = USE_ATR_STOP_BD,
#     N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
#     atr_window            = ATR_WINDOW_BD,
# )

# ============================================================
# CHANGE 4: MC robustness call for equity (Phase 7, multiasset_runfile.py only)
# ============================================================
#
# run_monte_carlo_robustness already picks up use_atr_stop and N_atr
# from extract_best_params_from_wf_results(wf_results_eq) — no change
# needed in the MC call itself.
#
# run_block_bootstrap_robustness forwards all wf_kwargs to walk_forward
# via **wf_kwargs, so add the ATR kwargs to the bootstrap call:
#
# BEFORE (end of equity bootstrap call):
#         mom_lookback_grid     = MOM_LB_EQ,
#     )
#
# AFTER:
#         mom_lookback_grid     = MOM_LB_EQ,
#         use_atr_stop          = USE_ATR_STOP,
#         N_atr_grid            = N_ATR_GRID if USE_ATR_STOP else None,
#         atr_window            = ATR_WINDOW,
#     )
#
# And similarly for the bond bootstrap call:
#
# BEFORE (end of bond bootstrap call):
#         mom_lookback_grid     = [252],
#     )
#
# AFTER:
#         mom_lookback_grid     = [252],
#         use_atr_stop          = USE_ATR_STOP_BD,
#         N_atr_grid            = N_ATR_GRID_BD if USE_ATR_STOP_BD else None,
#         atr_window            = ATR_WINDOW_BD,
#     )
