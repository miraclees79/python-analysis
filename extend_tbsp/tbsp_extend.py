# -*- coding: utf-8 -*-
"""
TBSP Full Back-extension with Duration, Curvature, Convexity, Intercept

Created on Thu Mar 19 2026
@author: U120137
"""

import sys
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from strategy_test_library import load_csv
import matplotlib.pyplot as plt

from datetime import datetime
USE_SHORT_SAMPLE=False

short_sample_end = pd.to_datetime("2016-12-31", format="%Y-%m-%d")
# =========================
# 1. LOAD TBSP
# =========================
df_TBSP = load_csv("^tbsp_d.csv")
if df_TBSP is None:
    logging.error("FAIL: load_csv returned None for TBSP — exiting.")
    sys.exit(1)

df_TBSP = df_TBSP.drop(columns=['Otwarcie', 'Najwyzszy','Najnizszy','Wolumen'], errors='ignore')
df_t = df_TBSP.copy()
df_t.columns = df_t.columns.str.strip()

if not isinstance(df_t.index, pd.RangeIndex):
    df_t = df_t.reset_index()

# Detect date column
date_col = [c for c in df_t.columns if c.lower() in ["data", "date"]][0]
df_t = df_t.rename(columns={date_col: "Date", "Zamkniecie": "TBSP"})
df_t["Date"] = pd.to_datetime(df_t["Date"])
df_t = df_t.sort_values("Date")
df_t["ret"] = df_t["TBSP"].pct_change()

# =========================
# 2. LOAD YIELDS
# =========================
df_y = pd.read_csv("TBYield.csv", sep=";", decimal=",")
df_y.rename(columns={"PL10YT_RR_YL": "PL10YT_RR_YLD"}, inplace=True)
df_y["Date"] = pd.to_datetime(df_y["Date"], format="%d.%m.%Y")
df_y = df_y.sort_values("Date")
for col in df_y.columns:
    if col != "Date":
        df_y[col] = pd.to_numeric(df_y[col], errors="coerce")

yield_cols = ["PL2YT_RR_YLD", "PL5YT_RR_YLD", "PL10YT_RR_YLD"]
df_y[yield_cols] = df_y[yield_cols].ffill()

# =========================
# 3. MERGE TBSP + YIELDS
# =========================
df = pd.merge(df_t, df_y, on="Date", how="inner")
for col in yield_cols:
    df[col] = df[col] / 100.0
if USE_SHORT_SAMPLE:
    
    # Filter by the Date column
    df_short = df.loc[df["Date"] <= short_sample_end].copy()
    df = df_short
    
# =========================
# 4. BUILD FACTORS
# =========================
df["level"] = df["PL5YT_RR_YLD"]
df["slope"] = df["PL10YT_RR_YLD"].fillna(df["PL5YT_RR_YLD"]) - df["PL2YT_RR_YLD"]
df["dslope"] = df["slope"].diff().fillna(0)
df["curvature"] = 2*df["PL5YT_RR_YLD"] - df["PL2YT_RR_YLD"] - df["PL10YT_RR_YLD"].fillna(df["PL5YT_RR_YLD"])
df["dcurv"] = df["curvature"].diff().fillna(0)
df["d5"] = df["PL5YT_RR_YLD"].diff().fillna(0)
df["carry"] = (0.2*df["PL2YT_RR_YLD"] + 0.5*df["PL5YT_RR_YLD"] + 0.3*df["PL10YT_RR_YLD"].fillna(df["PL5YT_RR_YLD"]))/252.0
df["d5_sq"] = df["d5"]**2
df["dslope_sq"] = df["dslope"]**2

# =========================
# 5. ESTIMATE AVERAGE DURATION
# =========================
dur_features = ["d5","dslope","dcurv"]
df_dur = df.dropna(subset=["ret"] + dur_features).copy()
X_dur = df_dur[dur_features].values
y_dur = df_dur["ret"].values

dur_model = LinearRegression(fit_intercept=False)
dur_model.fit(X_dur, y_dur)

# crude mapping to years: d5=5Y, dslope=8Y, dcurv=5Y
avg_duration = np.sum(dur_model.coef_ * [5,8,5])
print(f"Estimated avg duration (years, unscaled): {avg_duration:.2f}")

# =========================
# 6. REGRESSION TO RECONSTRUCT TBSP
# =========================
features = ["d5","dslope","dcurv","carry","d5_sq","dslope_sq"]
df_model = df.dropna(subset=["ret"] + features).copy()
X = df_model[features].values
y = df_model["ret"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression(fit_intercept=True)
model.fit(X_scaled, y)

print("\nCoefficients (scaled space):")
for f, c in zip(features, model.coef_):
    print(f"{f}: {c:.4f}")
print("Intercept:", model.intercept_)

# Diagnostics
y_pred = model.predict(X_scaled)
corr = np.corrcoef(y, y_pred)[0,1]
rmse = np.sqrt(np.mean((y - y_pred)**2))
print("\nDiagnostics:")
print(f"Correlation: {corr:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"Vol real: {np.std(y):.6f}, Vol model: {np.std(y_pred):.6f}")
print(f"Mean real: {np.mean(y):.6f}, Mean model: {np.mean(y_pred):.6f}")

# =========================
# 7. FULL SERIES RECONSTRUCTION
# =========================
# Full date range for back-extension
full_dates = pd.date_range(start=df_y["Date"].min(), end=df_y["Date"].max(), freq='B')
df_full = pd.DataFrame({"Date": full_dates})
df_full = pd.merge(df_full, df_y, on="Date", how="left")
df_full[yield_cols] = df_full[yield_cols].ffill()
for col in yield_cols:
    df_full[col] = df_full[col] / 100.0

# Recompute factors
df_full["level"] = df_full["PL5YT_RR_YLD"]
df_full["slope"] = df_full["PL10YT_RR_YLD"].fillna(df_full["PL5YT_RR_YLD"]) - df_full["PL2YT_RR_YLD"]
df_full["dslope"] = df_full["slope"].diff().fillna(0)
df_full["curvature"] = 2*df_full["PL5YT_RR_YLD"] - df_full["PL2YT_RR_YLD"] - df_full["PL10YT_RR_YLD"].fillna(df_full["PL5YT_RR_YLD"])
df_full["dcurv"] = df_full["curvature"].diff().fillna(0)
df_full["d5"] = df_full["PL5YT_RR_YLD"].diff().fillna(0)
df_full["carry"] = (0.2*df_full["PL2YT_RR_YLD"] + 0.5*df_full["PL5YT_RR_YLD"] + 0.3*df_full["PL10YT_RR_YLD"].fillna(df_full["PL5YT_RR_YLD"]))/252.0
df_full["d5_sq"] = df_full["d5"]**2
df_full["dslope_sq"] = df_full["dslope"]**2

X_full = df_full[features].fillna(0).values
X_full_scaled = scaler.transform(X_full)
df_full["ret_model"] = model.predict(X_full_scaled)

# Initialize synthetic TBSP
df_full["TBSP_synth"] = np.nan
first_real_idx = df["TBSP"].first_valid_index()
anchor_date = df.loc[first_real_idx,"Date"]
anchor_value = df.loc[first_real_idx,"TBSP"]

# Forward synthetic from anchor
df_full.loc[df_full["Date"] >= anchor_date,"TBSP_synth"] = (1 + df_full.loc[df_full["Date"] >= anchor_date,"ret_model"]).cumprod()
scale = anchor_value / df_full.loc[df_full["Date"] == anchor_date,"TBSP_synth"].values[0]
df_full.loc[df_full["Date"] >= anchor_date,"TBSP_synth"] *= scale

# Backward extension iteratively
back_idx = df_full[df_full["Date"] < anchor_date].index[::-1]
for idx in back_idx:
    next_idx = idx + 1
    df_full.loc[idx,"TBSP_synth"] = df_full.loc[next_idx,"TBSP_synth"] / (1 + df_full.loc[idx,"ret_model"])

# =========================
# 8. EXPORT
# =========================
df_full_out = df_full[["Date","TBSP_synth"]].copy()
df_full_out.to_csv("tbsp_extended_full.csv", index=False)
print("\nDone. Saved full back-extended TBSP to tbsp_extended_full.csv")

# =========================
# 9. PLOT
# =========================
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["TBSP"], label="TBSP Real", color="blue", linewidth=1.5)
plt.plot(df_full["Date"], df_full["TBSP_synth"], label="TBSP Synthetic", color="orange", linewidth=1.5, linestyle="--")
plt.title("TBSP Index: Real vs Synthetic (Back-extended to earliest yields)")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()