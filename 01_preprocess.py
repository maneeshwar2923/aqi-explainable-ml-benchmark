"""
01_preprocess.py
-----------------
Leakage-Free, Scientifically Defensible Preprocessing Pipeline for Urban AQI Modeling:
  1. Parse CPCB daily observations (city_day.csv) with strict date parsing.
  2. Drop rows lacking the ground-truth target AQI (no target imputation).
  3. Filter uninformative rows missing >40% of the 8 criteria pollutants (>3 missing).
  4. Chronological partitioning:
     - Training partition:   2015-01-01 to 2017-12-31 (7,556 rows)
     - Validation partition: 2018-01-01 to 2018-12-31 (5,658 rows)
     - Combined Train (dev): 2015-01-01 to 2018-12-31 (13,214 rows)
     - Final Test partition: 2019-01-01 to 2020-07-01 (11,410 rows)
  5. Imputation: Compute column-wise medians exclusively on the training partition(s)
     and apply to validation and test features without leakage.
  6. Outlier handling:
     - Ground-truth target AQI is strictly preserved (never clipped).
     - Pollutant features maintain physical concentrations (no-clipping baseline).
     - Audited clipping parameters (fit on training only) are recorded for sensitivity analysis.
  7. Feature scaling: Continuous pollutants retain raw physical units (ug/m3, mg/m3)
     for tree-based models and direct physical interpretability in SHAP.
  8. Emits: train.csv, val.csv, test.csv, train_sub.csv,
     preprocessing_summary.txt, and preprocessing_report.json.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]
TARGET = "AQI"
METADATA_COLS = ["City", "Date"]

def parse_args():
    parser = argparse.ArgumentParser(description="Leakage-free AQI preprocessing pipeline.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing city_day.csv. Defaults to script directory.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def log(msg, logfile=None):
    print(msg)
    if logfile:
        logfile.write(msg + "\n")

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir if args.data_dir else script_dir
    raw_path = os.path.join(data_dir, "city_day.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Source file not found at: {raw_path}")

    summary_txt_path = os.path.join(data_dir, "preprocessing_summary.txt")
    report_json_path = os.path.join(data_dir, "preprocessing_report.json")

    with open(summary_txt_path, "w", encoding="utf-8") as logfile:
        log("=" * 70, logfile)
        log("REVISED PREPROCESSING PIPELINE: LEAKAGE-FREE URBAN AQI DATASET", logfile)
        log("=" * 70, logfile)

        df = pd.read_csv(raw_path, parse_dates=["Date"])
        raw_rows, raw_cols = df.shape
        log(f"Loaded raw data: {raw_rows} rows, {raw_cols} columns from {raw_path}", logfile)

        all_cities = sorted(df["City"].dropna().unique().tolist())
        log(f"Total unique cities in raw dataset: {len(all_cities)}", logfile)

        # Check required columns
        required_cols = METADATA_COLS + POLLUTANTS + [TARGET]
        missing_from_csv = [c for c in required_cols if c not in df.columns]
        if missing_from_csv:
            raise ValueError(f"Missing expected columns in CSV: {missing_from_csv}")

        df = df[required_cols].copy()

        # Step 1: Drop missing target AQI
        missing_target_count = int(df[TARGET].isna().sum())
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        log(f"Dropped {missing_target_count} rows missing target '{TARGET}' (retained: {len(df)})", logfile)

        # Step 2: Drop rows missing >40% of the 8 criteria pollutants (>3 missing)
        missing_frac = df[POLLUTANTS].isna().mean(axis=1)
        sparse_count = int((missing_frac > 0.40).sum())
        df = df[missing_frac <= 0.40].reset_index(drop=True)
        log(f"Dropped {sparse_count} rows missing >40% of pollutant features (retained: {len(df)})", logfile)

        # Chronological partitioning
        df["Year"] = df["Date"].dt.year
        date_min, date_max = df["Date"].min().strftime("%Y-%m-%d"), df["Date"].max().strftime("%Y-%m-%d")
        log(f"Overall date range of retained observations: {date_min} to {date_max}", logfile)

        # Mask definitions
        train_sub_mask = (df["Year"] >= 2015) & (df["Year"] <= 2017)
        val_mask = (df["Year"] == 2018)
        train_full_mask = (df["Year"] >= 2015) & (df["Year"] <= 2018)
        test_mask = (df["Year"] >= 2019) & (df["Year"] <= 2020)

        train_sub_df = df[train_sub_mask].copy()
        val_df = df[val_mask].copy()
        train_full_df = df[train_full_mask].copy()
        test_df = df[test_mask].copy()

        log(f"\nChronological Split Breakdown:", logfile)
        log(f"  - Training Subset (2015-2017): {len(train_sub_df)} rows ({train_sub_df['Date'].min().strftime('%Y-%m-%d')} to {train_sub_df['Date'].max().strftime('%Y-%m-%d')})", logfile)
        log(f"  - Validation Set  (2018):      {len(val_df)} rows ({val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')})", logfile)
        log(f"  - Full Train Set  (2015-2018): {len(train_full_df)} rows ({train_full_df['Date'].min().strftime('%Y-%m-%d')} to {train_full_df['Date'].max().strftime('%Y-%m-%d')})", logfile)
        log(f"  - Final Test Set  (2019-2020): {len(test_df)} rows ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})", logfile)

        # City coverage audit across splits
        train_cities = sorted(train_full_df["City"].unique().tolist())
        test_cities = sorted(test_df["City"].unique().tolist())
        unseen_test_cities = sorted(list(set(test_cities) - set(train_cities)))
        log(f"\nCity Coverage Audit:", logfile)
        log(f"  - Cities in Training set: {len(train_cities)} -> {train_cities}", logfile)
        log(f"  - Cities in Test set:     {len(test_cities)} -> {test_cities}", logfile)
        log(f"  - Unseen cities in Test:  {len(unseen_test_cities)} -> {unseen_test_cities}", logfile)
        unseen_rows = int(test_df["City"].isin(unseen_test_cities).sum())
        log(f"  - Unseen city observations in Test: {unseen_rows} / {len(test_df)} ({unseen_rows/len(test_df)*100:.2f}%)", logfile)

        # Step 3: Imputation (Fitted strictly on training partitions)
        log(f"\nMissing Value Imputation (Train-only fitted statistics):", logfile)
        
        # 1. Medians for model selection (fitted on 2015-2017)
        sub_medians = train_sub_df[POLLUTANTS].median().to_dict()
        # 2. Medians for final training & test evaluation (fitted on full 2015-2018 train)
        full_medians = train_full_df[POLLUTANTS].median().to_dict()

        for col in POLLUTANTS:
            train_sub_miss = int(train_sub_df[col].isna().sum())
            val_miss = int(val_df[col].isna().sum())
            train_full_miss = int(train_full_df[col].isna().sum())
            test_miss = int(test_df[col].isna().sum())

            log(f"  {col:6s} | TrainFull Miss: {train_full_miss:4d} | Test Miss: {test_miss:4d} | "
                f"TrainFull Median: {full_medians[col]:8.3f} | TrainSub Median: {sub_medians[col]:8.3f}", logfile)

        # Apply imputation
        # For val_df: imputed with sub_medians (strictly prior to 2018)
        val_imputed = val_df.copy()
        train_sub_imputed = train_sub_df.copy()
        for col in POLLUTANTS:
            train_sub_imputed[col] = train_sub_imputed[col].fillna(sub_medians[col])
            val_imputed[col] = val_imputed[col].fillna(sub_medians[col])

        # For train_full_df and test_df: imputed with full_medians (2015-2018)
        train_full_imputed = train_full_df.copy()
        test_imputed = test_df.copy()
        for col in POLLUTANTS:
            train_full_imputed[col] = train_full_imputed[col].fillna(full_medians[col])
            test_imputed[col] = test_imputed[col].fillna(full_medians[col])

        # Step 4: Outlier Audit & Parameter Recording
        # As per Task 2.C:
        # - Target AQI is never clipped or altered.
        # - Main experiment uses the unclipped physical baseline.
        # - Audited upper clipping parameters fit on train_full_imputed are recorded for reference.
        log(f"\nOutlier Threshold Audit (Training-only statistics, unclipped baseline retained):", logfile)
        outlier_audit = {}
        for col in POLLUTANTS:
            mean = float(train_full_imputed[col].mean())
            std = float(train_full_imputed[col].std())
            p99 = float(train_full_imputed[col].quantile(0.99))
            upper_3sigma = mean + 3 * std
            n_train_over_3sigma = int((train_full_imputed[col] > upper_3sigma).sum())
            n_test_over_3sigma = int((test_imputed[col] > upper_3sigma).sum())
            outlier_audit[col] = {
                "mean": mean,
                "std": std,
                "upper_3sigma": upper_3sigma,
                "p99": p99,
                "train_exceedances_3sigma": n_train_over_3sigma,
                "test_exceedances_3sigma": n_test_over_3sigma
            }
            log(f"  {col:6s} | Mean={mean:7.2f}, Std={std:7.2f} | 3-Sigma Upper={upper_3sigma:7.2f} | "
                f"P99={p99:7.2f} | Train >3s: {n_train_over_3sigma:3d} | Test >3s: {n_test_over_3sigma:3d}", logfile)

        # Audit target AQI
        aqi_mean = float(train_full_imputed[TARGET].mean())
        aqi_std = float(train_full_imputed[TARGET].std())
        aqi_max = float(train_full_imputed[TARGET].max())
        test_aqi_max = float(test_imputed[TARGET].max())
        log(f"  Target AQI summary: Train Mean={aqi_mean:.2f}, Std={aqi_std:.2f}, Max={aqi_max:.1f} | Test Max={test_aqi_max:.1f}", logfile)
        log("  NOTE: Target AQI is NEVER clipped. Predictions will be evaluated against true ground-truth AQI.", logfile)

        # Step 5: Save Datasets
        # Include 'Split' column in train_full to denote 'train' (2015-2017) vs 'val' (2018)
        train_full_imputed["Split"] = np.where(train_full_imputed["Year"] == 2018, "val", "train")
        test_imputed["Split"] = "test"
        train_sub_imputed["Split"] = "train"
        val_imputed["Split"] = "val"

        export_cols = METADATA_COLS + POLLUTANTS + [TARGET, "Split"]

        # Export CSVs
        train_full_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")
        val_path = os.path.join(data_dir, "val.csv")
        train_sub_path = os.path.join(data_dir, "train_sub.csv")

        train_full_imputed[export_cols].to_csv(train_full_path, index=False)
        test_imputed[export_cols].to_csv(test_path, index=False)
        val_imputed[export_cols].to_csv(val_path, index=False)
        train_sub_imputed[export_cols].to_csv(train_sub_path, index=False)

        log(f"\nSerialized Datasets:", logfile)
        log(f"  - train.csv     (Combined 2015-2018 dev set): {train_full_imputed[export_cols].shape}", logfile)
        log(f"  - val.csv       (Chronological 2018 val set): {val_imputed[export_cols].shape}", logfile)
        log(f"  - train_sub.csv (Chronological 2015-2017 set):{train_sub_imputed[export_cols].shape}", logfile)
        log(f"  - test.csv      (Final 2019-2020 test set):   {test_imputed[export_cols].shape}", logfile)

        # Build machine-readable report
        report = {
            "metadata": {
                "script": "01_preprocess.py",
                "raw_file": "city_day.csv",
                "raw_rows": raw_rows,
                "raw_columns": raw_cols,
                "target_variable": TARGET,
                "features": POLLUTANTS,
                "metadata_columns": METADATA_COLS
            },
            "cleaning_summary": {
                "missing_target_dropped": missing_target_count,
                "sparse_rows_dropped_gt_40pct": sparse_count,
                "retained_valid_rows": len(df)
            },
            "split_boundaries": {
                "train_sub": {"years": [2015, 2017], "rows": len(train_sub_df), "date_start": train_sub_df["Date"].min().strftime("%Y-%m-%d"), "date_end": train_sub_df["Date"].max().strftime("%Y-%m-%d")},
                "val":       {"years": [2018, 2018], "rows": len(val_df), "date_start": val_df["Date"].min().strftime("%Y-%m-%d"), "date_end": val_df["Date"].max().strftime("%Y-%m-%d")},
                "train_full":{"years": [2015, 2018], "rows": len(train_full_df), "date_start": train_full_df["Date"].min().strftime("%Y-%m-%d"), "date_end": train_full_df["Date"].max().strftime("%Y-%m-%d")},
                "test":      {"years": [2019, 2020], "rows": len(test_df), "date_start": test_df["Date"].min().strftime("%Y-%m-%d"), "date_end": test_df["Date"].max().strftime("%Y-%m-%d")}
            },
            "spatial_coverage": {
                "train_cities_count": len(train_cities),
                "train_cities": train_cities,
                "test_cities_count": len(test_cities),
                "test_cities": test_cities,
                "unseen_test_cities_count": len(unseen_test_cities),
                "unseen_test_cities": unseen_test_cities,
                "unseen_test_rows": unseen_rows,
                "unseen_test_percentage": round(unseen_rows / len(test_df) * 100, 2)
            },
            "learned_parameters": {
                "train_full_medians": full_medians,
                "train_sub_medians": sub_medians,
                "outlier_audit_train_full": outlier_audit
            },
            "policy": {
                "target_capping": "None (unaltered observed AQI)",
                "feature_scaling": "Raw physical concentration units (no scaling for tree invariance)",
                "leakage_safeguard": "All preprocessing statistics fit strictly on training partitions"
            }
        }

        with open(report_json_path, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        log(f"\nMachine-readable report saved to {report_json_path}", logfile)
        log("Preprocessing complete.", logfile)

if __name__ == "__main__":
    main()
