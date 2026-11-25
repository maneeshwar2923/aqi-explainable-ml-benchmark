"""
NEW_ANALYSIS/scripts/01_preprocess.py
-------------------------------------
Leakage-Free Preprocessing Pipeline for Urban AQI Estimation:
  1. Ingests raw CPCB ambient air monitoring records (city_day.csv).
  2. Drops records lacking the observed target AQI (no target imputation).
  3. Drops uninformative rows missing >40% of the 8 criteria pollutants (>3 missing).
  4. Partitions data chronologically BEFORE any parameter estimation:
     - Training partition (sub):   2015-01-01 to 2017-12-31 (7,556 rows)
     - Validation partition:       2018-01-01 to 2018-12-31 (5,658 rows)
     - Full Development partition: 2015-01-01 to 2018-12-31 (13,214 rows)
     - Out-of-Time Test partition: 2019-01-01 to 2020-07-01 (11,410 rows)
  5. Imputation: Computes feature medians strictly on training partitions and
     applies them to validation and test partitions without lookahead leakage.
  6. Target Integrity: Observed AQI is 100% unaltered (zero artificial capping).
  7. Physical Units: Criteria pollutants retain raw physical units (ug/m3, mg/m3)
     for direct TreeSHAP interpretability without scale distortion.
  8. Emits processed CSV partitions to NEW_ANALYSIS/data_processed/ and comprehensive
     audit reports to NEW_ANALYSIS/reports/.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]
TARGET = "AQI"
METADATA_COLS = ["City", "Date"]

def parse_args():
    parser = argparse.ArgumentParser(description="Leakage-free AQI Preprocessing Pipeline")
    parser.add_argument("--raw-file", type=str, default=None,
                        help="Path to raw city_day.csv. Defaults to project root or workspace.")
    parser.add_argument("--output-data-dir", type=str, default=None,
                        help="Directory to save preprocessed CSVs. Defaults to NEW_ANALYSIS/data_processed.")
    parser.add_argument("--output-report-dir", type=str, default=None,
                        help="Directory to save preprocessing reports. Defaults to NEW_ANALYSIS/reports.")
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
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Resolve raw data path
    if args.raw_file:
        raw_path = os.path.abspath(args.raw_file)
    else:
        candidates = [
            os.path.join(project_root, "city_day.csv"),
            os.path.join(project_root, "data", "city_day.csv"),
            os.path.join(script_dir, "city_day.csv")
        ]
        raw_path = None
        for c in candidates:
            if os.path.exists(c):
                raw_path = c
                break
        if raw_path is None:
            raise FileNotFoundError(f"Could not locate city_day.csv in candidates: {candidates}")

    # Resolve output directories
    out_data_dir = args.output_data_dir if args.output_data_dir else os.path.abspath(os.path.join(script_dir, "..", "data_processed"))
    out_report_dir = args.output_report_dir if args.output_report_dir else os.path.abspath(os.path.join(script_dir, "..", "reports"))
    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_report_dir, exist_ok=True)

    summary_txt_path = os.path.join(out_report_dir, "preprocessing_summary.txt")
    report_json_path = os.path.join(out_report_dir, "preprocessing_report.json")

    with open(summary_txt_path, "w", encoding="utf-8") as logfile:
        log("=" * 75, logfile)
        log("REVISED LEAKAGE-FREE URBAN AQI PREPROCESSING PIPELINE", logfile)
        log("=" * 75, logfile)
        log(f"Raw source dataset: {raw_path}", logfile)
        log(f"Processed CSV target: {out_data_dir}", logfile)
        log(f"Report target: {out_report_dir}", logfile)

        df = pd.read_csv(raw_path, parse_dates=["Date"])
        raw_rows, raw_cols = df.shape
        log(f"\n1. Ingestion: Loaded {raw_rows} raw records, {raw_cols} columns.", logfile)

        all_cities = sorted(df["City"].dropna().unique().tolist())
        log(f"   Total monitoring cities present in raw data: {len(all_cities)}", logfile)

        # Check required columns
        required_cols = METADATA_COLS + POLLUTANTS + [TARGET]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing from raw data: {missing_cols}")

        df = df[required_cols].copy()

        # Step 1: Drop missing target AQI (no target imputation)
        n_missing_target = int(df[TARGET].isna().sum())
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        log(f"2. Target Cleaning: Dropped {n_missing_target} records with missing target AQI (Retained: {len(df)}).", logfile)

        # Step 2: Drop uninformative rows missing >40% of criteria pollutants (>3 missing)
        missing_frac = df[POLLUTANTS].isna().mean(axis=1)
        sparse_count = int((missing_frac > 0.40).sum())
        df = df[missing_frac <= 0.40].reset_index(drop=True)
        log(f"3. Feature Cleaning: Dropped {sparse_count} rows missing >40% of criteria pollutants (Retained: {len(df)}).", logfile)

        # Step 3: Chronological partitioning BEFORE any feature transformation
        df["Year"] = df["Date"].dt.year
        date_min, date_max = df["Date"].min().strftime("%Y-%m-%d"), df["Date"].max().strftime("%Y-%m-%d")
        log(f"\n4. Chronological Partitioning ({date_min} to {date_max}):", logfile)

        train_sub_mask = (df["Year"] >= 2015) & (df["Year"] <= 2017)
        val_mask = (df["Year"] == 2018)
        train_full_mask = (df["Year"] >= 2015) & (df["Year"] <= 2018)
        test_mask = (df["Year"] >= 2019) & (df["Year"] <= 2020)

        train_sub = df[train_sub_mask].copy().reset_index(drop=True)
        val = df[val_mask].copy().reset_index(drop=True)
        train_full = df[train_full_mask].copy().reset_index(drop=True)
        test = df[test_mask].copy().reset_index(drop=True)

        log(f"   - Training Sub-Partition (2015-2017): {len(train_sub):>6} rows (tuning / early stopping fold)", logfile)
        log(f"   - Validation Partition   (2018):      {len(val):>6} rows (validation stopping fold)", logfile)
        log(f"   - Full Dev Partition     (2015-2018): {len(train_full):>6} rows (final model refitting fold)", logfile)
        log(f"   - Out-of-Time Test       (2019-2020): {len(test):>6} rows (single-pass holdout assessment)", logfile)

        # Spatial distribution across splits
        train_cities = sorted(train_full["City"].unique().tolist())
        test_cities = sorted(test["City"].unique().tolist())
        unseen_test_cities = sorted(list(set(test_cities) - set(train_cities)))
        log(f"\n5. Spatial Coverage Analysis:", logfile)
        log(f"   - Cities active in development period (2015-2018): {len(train_cities)} cities", logfile)
        log(f"   - Cities active in test period (2019-2020):        {len(test_cities)} cities", logfile)
        log(f"   - Unseen cities newly commissioned in test period: {len(unseen_test_cities)} cities ({unseen_test_cities})", logfile)
        unseen_rows = int(test["City"].isin(unseen_test_cities).sum())
        log(f"   - Unseen city observations in test set: {unseen_rows} / {len(test)} ({unseen_rows/len(test)*100:.2f}%)", logfile)

        # Step 4: Leakage-Free Imputation
        # Compute medians strictly on training partitions
        sub_medians = train_sub[POLLUTANTS].median().to_dict()
        full_medians = train_full[POLLUTANTS].median().to_dict()

        log(f"\n6. Leakage-Free Feature Imputation (Fitted on Training Partitions Only):", logfile)
        log(f"   {'Pollutant':<8} | {'TrainFull Median':<18} | {'TrainSub Median':<18}", logfile)
        log(f"   {'-'*8}-+-{'-'*18}-+-{'-'*18}", logfile)
        for p in POLLUTANTS:
            log(f"   {p:<8} | {full_medians[p]:<18.4f} | {sub_medians[p]:<18.4f}", logfile)

        # Apply to subsets
        # For validation tuning fold: impute using train_sub medians
        train_sub_imputed = train_sub.copy()
        val_imputed = val.copy()
        for p in POLLUTANTS:
            train_sub_imputed[p] = train_sub_imputed[p].fillna(sub_medians[p])
            val_imputed[p] = val_imputed[p].fillna(sub_medians[p])

        # For final development & test evaluation: impute using full development medians
        train_full_imputed = train_full.copy()
        test_imputed = test.copy()
        for p in POLLUTANTS:
            train_full_imputed[p] = train_full_imputed[p].fillna(full_medians[p])
            test_imputed[p] = test_imputed[p].fillna(full_medians[p])

        # Step 5: Target Integrity Verification
        log(f"\n7. Target AQI Integrity Check:", logfile)
        raw_test_aqi = test[TARGET].values
        proc_test_aqi = test_imputed[TARGET].values
        max_target_deviation = float(np.max(np.abs(raw_test_aqi - proc_test_aqi)))
        log(f"   Max absolute deviation between observed and processed test AQI: {max_target_deviation:.8f}", logfile)
        assert max_target_deviation == 0.0, "Critical Failure: Target AQI was modified!"
        log("   [VERIFIED] Target AQI is 100% unaltered. Zero synthetic capping applied.", logfile)

        # Step 6: Post-Processing Completeness Check
        for split_name, s_df in [("TrainSub", train_sub_imputed), ("Val", val_imputed),
                                 ("TrainFull", train_full_imputed), ("Test", test_imputed)]:
            null_count = int(s_df[POLLUTANTS + [TARGET]].isna().sum().sum())
            inf_count = int(np.isinf(s_df[POLLUTANTS + [TARGET]].values).sum())
            if null_count > 0 or inf_count > 0:
                raise ValueError(f"Partition {split_name} contains {null_count} nulls or {inf_count} infs!")
        log("   [VERIFIED] All partitions contain zero missing (NaN) and zero infinite (Inf) values.", logfile)

        # Step 7: Export Clean Partitions
        cols_to_save = ["City", "Date"] + POLLUTANTS + [TARGET]
        train_sub_path = os.path.join(out_data_dir, "train_sub.csv")
        val_path = os.path.join(out_data_dir, "val.csv")
        train_full_path = os.path.join(out_data_dir, "train.csv")
        test_path = os.path.join(out_data_dir, "test.csv")

        train_sub_imputed[cols_to_save].to_csv(train_sub_path, index=False)
        val_imputed[cols_to_save].to_csv(val_path, index=False)
        train_full_imputed[cols_to_save].to_csv(train_full_path, index=False)
        test_imputed[cols_to_save].to_csv(test_path, index=False)

        log(f"\n8. Exported Processed Partitions:", logfile)
        log(f"   - {train_sub_path} ({train_sub_imputed.shape})", logfile)
        log(f"   - {val_path} ({val_imputed.shape})", logfile)
        log(f"   - {train_full_path} ({train_full_imputed.shape})", logfile)
        log(f"   - {test_path} ({test_imputed.shape})", logfile)

        # Step 8: Build and Serialize JSON Preprocessing Audit Report
        report_data = {
            "metadata": {
                "script": "NEW_ANALYSIS/scripts/01_preprocess.py",
                "raw_file": os.path.relpath(raw_path, project_root),
                "raw_rows": raw_rows,
                "raw_columns": raw_cols,
                "target_variable": TARGET,
                "features": POLLUTANTS,
                "metadata_columns": METADATA_COLS
            },
            "cleaning_summary": {
                "missing_target_dropped": n_missing_target,
                "sparse_rows_dropped_gt_40pct": sparse_count,
                "retained_valid_rows": len(df)
            },
            "split_boundaries": {
                "train_sub": {
                    "years": [2015, 2017],
                    "rows": len(train_sub),
                    "date_start": train_sub["Date"].min().strftime("%Y-%m-%d"),
                    "date_end": train_sub["Date"].max().strftime("%Y-%m-%d")
                },
                "val": {
                    "years": [2018, 2018],
                    "rows": len(val),
                    "date_start": val["Date"].min().strftime("%Y-%m-%d"),
                    "date_end": val["Date"].max().strftime("%Y-%m-%d")
                },
                "train_full": {
                    "years": [2015, 2018],
                    "rows": len(train_full),
                    "date_start": train_full["Date"].min().strftime("%Y-%m-%d"),
                    "date_end": train_full["Date"].max().strftime("%Y-%m-%d")
                },
                "test": {
                    "years": [2019, 2020],
                    "rows": len(test),
                    "date_start": test["Date"].min().strftime("%Y-%m-%d"),
                    "date_end": test["Date"].max().strftime("%Y-%m-%d")
                }
            },
            "spatial_coverage": {
                "train_cities_count": len(train_cities),
                "train_cities": train_cities,
                "test_cities_count": len(test_cities),
                "test_cities": test_cities,
                "unseen_test_cities_count": len(unseen_test_cities),
                "unseen_test_cities": unseen_test_cities,
                "unseen_test_observations": unseen_rows,
                "unseen_test_percentage": round(unseen_rows / len(test) * 100, 2)
            },
            "learned_parameters": {
                "train_full_medians": {p: round(float(full_medians[p]), 4) for p in POLLUTANTS},
                "train_sub_medians": {p: round(float(sub_medians[p]), 4) for p in POLLUTANTS}
            },
            "target_integrity": {
                "synthetic_capping_applied": False,
                "max_target_deviation": max_target_deviation,
                "verified_unaltered": True
            }
        }

        with open(report_json_path, "w", encoding="utf-8") as jf:
            json.dump(report_data, jf, indent=2)

        log(f"   - Preprocessing report serialized to: {report_json_path}", logfile)
        log("Preprocessing pipeline finished successfully.", logfile)

if __name__ == "__main__":
    main()
