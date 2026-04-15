# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:41:45 2026

@author: adamg
"""

import pandas as pd
import numpy as np
import logging
from joblib import Parallel, delayed
from moj_system.core.backtest import run_strategy_with_trades
from moj_system.core.indicators import compute_metrics

def neighbour_mean(key, scores, stop_grid, Y_grid):
    """Oblicza średnią z sąsiednich parametrów na siatce (wygładzanie optymalizacji)."""
    filter_mode, stop_param, Y, fast, slow, tv, sl, mom_lookback = key
    si = min(range(len(stop_grid)), key=lambda i: abs(stop_grid[i] - stop_param))
    yi = min(range(len(Y_grid)),    key=lambda i: abs(Y_grid[i] - Y))

    neighbours = []
    for ds in[-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nsi, nyi = si + ds, yi + dy
            if 0 <= nsi < len(stop_grid) and 0 <= nyi < len(Y_grid):
                nkey = (filter_mode, stop_grid[nsi], Y_grid[nyi], fast, slow, tv, sl, mom_lookback)
                if nkey in scores:
                    neighbours.append(scores[nkey])
    return np.mean(neighbours) if neighbours else scores[key]

# W pliku moj_system/core/optimizer.py
# Podmień również funkcję evaluate_params

def evaluate_params(
    filter_mode, stop_val, Y, fast, slow, tv, stop_loss, train, cash_train, vol_window, 
    selected_mode, train_start, train_end, objective, mom_lookback,
    fast_mode, use_atr_stop, atr_window
):
    # Reszta funkcji bez zmian...
    bt, metrics, _, _ = run_strategy_with_trades(
        train, cash_df=cash_train, price_col="Zamkniecie",
        X=stop_val if not use_atr_stop else 0.1, Y=Y, stop_loss=stop_loss,
        fast=fast, slow=slow, target_vol=tv, vol_window=vol_window,
        position_mode=selected_mode, filter_mode=filter_mode,
        mom_lookback=mom_lookback, fast_mode=fast_mode,
        use_atr_stop=use_atr_stop, N_atr=stop_val if use_atr_stop else 3.0, atr_window=atr_window
    )
    if not metrics: return None
    
    max_dd = metrics.get("MaxDD", 0)
    calmar = metrics["CAGR"] / abs(max_dd) if max_dd != 0 else None
    
    if objective == "calmar" and calmar is not None: obj_value = calmar
    elif objective == "sharpe": obj_value = metrics.get("Sharpe", 0)
    else: return None
    
    key = (filter_mode, stop_val, Y, fast, slow, tv, stop_loss, mom_lookback)
    return key, obj_value


# W pliku moj_system/core/optimizer.py
# Podmień całą funkcję walk_forward na tę wersję:

def walk_forward(
    df: pd.DataFrame, cash_df: pd.DataFrame, train_years=8, test_years=2,
    vol_window=20, selected_mode="full", filter_modes_override=None,
    X_grid=[0.08, 0.10, 0.12], Y_grid=[0.02, 0.05], fast_grid=[50, 100], slow_grid=[150, 200],
    tv_grid=[0.10], sl_grid=[0.05, 0.10], mom_lookback_grid=[252], objective="calmar",
    n_jobs=1, fast_mode=True, use_atr_stop=False, N_atr_grid=None, atr_window=20
):
    N_atr_grid = N_atr_grid or [2.0, 3.0, 4.0, 5.0]
    stop_grid = N_atr_grid if use_atr_stop else X_grid
    
    data_end = df.index.max()
    oos_equity_slices, results, all_oos_trades = [], [],[]
    start = df.index.min()
    carry_state = None

    while True:
        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        train = df.loc[(df.index >= train_start) & (df.index < train_end)]
        
        next_window_has_enough = len(df.loc[df.index >= test_end]) >= (test_years * 252)
        if not next_window_has_enough: test_end = data_end + pd.DateOffset(days=1)
        test = df.loc[(df.index >= train_end) & (df.index < test_end)]

        if train.empty or len(test) < 30: break
        cash_train = cash_df.loc[(cash_df.index >= train_start) & (cash_df.index < train_end)]

        filter_modes = filter_modes_override if filter_modes_override else ["ma", "mom", "mom_blend"]
        
        param_combinations = []
        for fm in filter_modes:
            for stop_val in stop_grid:
                for y in Y_grid:
                    for f in (fast_grid if fm == "ma" else [50]):
                        for s in (slow_grid if fm == "ma" else [200]):
                            for tv in tv_grid:
                                for sl in sl_grid:
                                    for ml in (mom_lookback_grid if fm == "mom" else [252]):
                                        if not (fm == "ma" and s - f < 75) and (use_atr_stop or sl < stop_val):
                                            param_combinations.append(
                                                (fm, stop_val, y, f, s, tv, sl, ml)
                                            )

        # --- POPRAWIONE WYWOŁANIE PARALLEL ---
        # Przekazujemy argumenty pozycyjnie, co jest bezpieczne w tym kontekście
        results_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_params)(
                fm, stop_val, y, f, s, tv, sl, train, cash_train, vol_window,
                selected_mode, train_start, train_end, objective, ml,
                fast_mode, use_atr_stop, atr_window
            ) for (fm, stop_val, y, f, s, tv, sl, ml) in param_combinations
        )
        # --- KONIEC POPRAWEK ---
        
        param_scores = {k: score for res in results_list if res for k, score in [res]}
        if not param_scores:
            start += pd.DateOffset(years=test_years)
            continue

        # Logika wyboru najlepszych parametrów (bez zmian)
        best_score, best_params, best_raw_score = -np.inf, None, -np.inf
        top_candidates = []

        for key, raw_score in param_scores.items():
            filter_mode, stop_param, Y, fast, slow, tv, stop_loss, mom_lookback = key
            
            simplified_key = (filter_mode, stop_param, Y, fast, slow, tv, stop_loss, mom_lookback)
            relevant_scores = {k: v for k, v in param_scores.items() if k[0] == filter_mode and k[3] == fast and k[4] == slow and k[5] == tv and k[6] == stop_loss and k[7] == mom_lookback}
            stability = neighbour_mean(simplified_key, relevant_scores, stop_grid, Y_grid)
            combined = 0.5 * raw_score + 0.5 * stability

            if len(top_candidates) < 5 or combined > top_candidates[-1][0]:
                top_candidates.append((combined, raw_score, key))
                top_candidates.sort(key=lambda x: x[0], reverse=True)
                if len(top_candidates) > 5: top_candidates.pop()
            
            if combined > best_score:
                best_score = combined
                best_raw_score = raw_score
                best_params = {
                    "filter_mode": key[0], ("N_atr" if use_atr_stop else "X"): key[1],
                    "Y": key[2], "fast": key[3], "slow": key[4],
                    "target_vol": key[5], "stop_loss": key[6], "mom_lookback": key[7]
                }
        """
        logging.info("TOP 5 KANDYDATÓW W OKNIE TRENINGOWYM:")
        for combined_score, raw_score, key in top_candidates:
            (fm, stop_val, y, f, s, tv, sl, ml) = key
            stop_label = "N_atr" if use_atr_stop else "X"
            logging.info(f"  Combined: {combined_score:.6f}, Raw: {raw_score:.6f} | Params: mode={fm}, {stop_label}={stop_val}, Y={y}, fast={f}, slow={s}, sl={sl}, mom_lb={ml}")
        """
        if best_params is None: break
        
        warmup = train.iloc[-(best_params["slow"] + vol_window + 10):]
        cash_wt = cash_df.loc[(cash_df.index >= warmup.index.min()) & (cash_df.index < test_end)]
        bt_oos, test_metrics, oos_trades, end_state = run_strategy_with_trades(
            test, cash_df=cash_wt, price_col="Zamkniecie",
            X=best_params.get("X", 0.1), Y=best_params["Y"], stop_loss=best_params["stop_loss"],
            fast=best_params["fast"], slow=best_params["slow"], vol_window=vol_window,
            position_mode=selected_mode, filter_mode=best_params["filter_mode"],
            mom_lookback=best_params["mom_lookback"], fast_mode=fast_mode,
            use_atr_stop=use_atr_stop, N_atr=best_params.get("N_atr", 3.0), atr_window=atr_window,
            initial_state=carry_state, warmup_df=warmup
        )
        carry_state = end_state
        if bt_oos is not None:
            slice_eq = bt_oos["equity"] * (oos_equity_slices[-1].iloc[-1] if oos_equity_slices else 1.0)
            oos_equity_slices.append(slice_eq)
            if not oos_trades.empty:
                oos_trades["WF_Window"] = train_start
                all_oos_trades.append(oos_trades)
            results.append({
                "TrainStart": train_start, "TestStart": train_end, "TestEnd": test_end,
                **best_params, **(test_metrics or {})
            })
        start += pd.DateOffset(years=test_years)
        if not next_window_has_enough: break

    return (pd.concat(oos_equity_slices).sort_index() if oos_equity_slices else pd.Series(dtype=float)), \
           pd.DataFrame(results), (pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame())