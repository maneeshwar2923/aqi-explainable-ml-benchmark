"""
01_preprocess.py
-----------------
Implements the preprocessing described in Section 3.2 of the paper:
  - Load city_day.csv
  - Median-impute missing pollutant values (column-wise)
  - Drop rows missing >40% of pollutant sub-index values
  - Cap outliers beyond 3 std devs at the 99th percentile
  - Time-based split: 2015-2018 train, 2019-2020 test
  - Z-score standardize features (fit on train only)

Run this first. It writes train.csv / test.csv (standardized features + raw AQI target)
into the same folder as the source CSV, plus a preprocessing_summary.txt log.
"""

import pandas as pd
import numpy as np
import os

# ---- EDIT THIS if your path changes ----
RAW_PATH = r"C:\Users\manee\OneDrive\Documents\CHRIST\Research\final_analysis\city_day.csv"
OUT_DIR = os.path.dirname(RAW_PATH)
# -----------------------------------------

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]
TARGET = "AQI"

def log(msg, logfile):
    print(msg)
    logfile.write(msg + "\n")

def main():
    with open(os.path.join(OUT_DIR, "preprocessing_summary.txt"), "w") as logfile:
        df = pd.read_csv(RAW_PATH, parse_dates=["Date"])
        log(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns", logfile)
        log(f"Cities found: {df['City'].nunique()} -> {sorted(df['City'].unique())}", logfile)

        # Keep only columns we need
        cols_needed = ["City", "Date"] + POLLUTANTS + [TARGET]
        missing_cols = [c for c in cols_needed if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Expected columns missing from CSV: {missing_cols}")
        df = df[cols_needed].copy()

        # Drop rows with no AQI target at all (can't train/test without it)
        before = len(df)
        df = df.dropna(subset=[TARGET])
        log(f"Dropped {before - len(df)} rows with missing AQI target", logfile)

        # Drop rows missing more than 40% of the 8 pollutant sub-indices (i.e. >3 of 8 missing)
        missing_frac = df[POLLUTANTS].isna().mean(axis=1)
        before = len(df)
        df = df[missing_frac <= 0.40].reset_index(drop=True)
        log(f"Dropped {before - len(df)} rows missing >40% of pollutant values", logfile)

        # Median imputation (column-wise, computed on the full retained set)
        for col in POLLUTANTS:
            med = df[col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            log(f"Imputed {n_filled} missing values in {col} with median={med:.2f}", logfile)

        # Outlier capping: beyond 3 std devs -> capped at the 99th percentile
        for col in POLLUTANTS + [TARGET]:
            mean, std = df[col].mean(), df[col].std()
            p99 = df[col].quantile(0.99)
            lower, upper = mean - 3 * std, mean + 3 * std
            n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = np.where(df[col] > upper, p99, df[col])
            df[col] = np.where(df[col] < lower, df[col], df[col])  # lower tail left as-is (pollution has a natural floor at 0)
            log(f"Capped {n_capped} outliers in {col} (3-sigma bounds: {lower:.2f} to {upper:.2f}) at 99th pct = {p99:.2f}", logfile)

        # Time-based split
        df["Year"] = df["Date"].dt.year
        train_df = df[(df["Year"] >= 2015) & (df["Year"] <= 2018)].copy()
        test_df = df[(df["Year"] >= 2019) & (df["Year"] <= 2020)].copy()
        log(f"Train set (2015-2018): {len(train_df)} rows", logfile)
        log(f"Test set (2019-2020): {len(test_df)} rows", logfile)

        # Z-score standardization, fit on train only, applied to both
        means = train_df[POLLUTANTS].mean()
        stds = train_df[POLLUTANTS].std()
        for col in POLLUTANTS:
            train_df[col + "_z"] = (train_df[col] - means[col]) / stds[col]
            test_df[col + "_z"] = (test_df[col] - means[col]) / stds[col]

        z_cols = [c + "_z" for c in POLLUTANTS]
        final_cols = ["City", "Date"] + z_cols + [TARGET]
        train_out = train_df[final_cols]
        test_out = test_df[final_cols]

        train_out.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
        test_out.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

        log(f"\nSaved train.csv ({train_out.shape}) and test.csv ({test_out.shape}) to {OUT_DIR}", logfile)
        log("Preprocessing complete.", logfile)

if __name__ == "__main__":
    main()
