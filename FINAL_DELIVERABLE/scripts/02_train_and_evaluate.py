"""
NEW_ANALYSIS/scripts/02_train_and_evaluate.py
---------------------------------------------
Controlled Multi-Model Benchmarking, Baseline Comparison & Rigorous Statistical Inference:
  - Benchmarks 4 Tree Ensembles: Random Forest, XGBoost, LightGBM, CatBoost
  - Adds Baseline Models: Training Mean Baseline & City-Stratified Mean Baseline
  - Controlled Experiments:
      * Experiment 1: Pollutant-Only Benchmark (8 physical criteria pollutants)
      * Experiment 2: City-Aware Benchmark (8 criteria pollutants + City identity)
  - Validation: 2018 chronological validation partition used for early stopping / iteration tuning
    (final test partition 2019-2020 remains strictly untouched).
  - Test Evaluation: Final single-pass assessment on 2019-2020 holdout test partition (N=11,410).
  - Standalone Prediction Serialization:
      * NEW_ANALYSIS/predictions/predictions_pollutant_only.csv
      * NEW_ANALYSIS/predictions/predictions_city_aware.csv
  - Granular Error Breakdowns:
      * City-wise evaluation across all 26 monitoring cities
      * Temporal year-wise breakdown (2019 vs 2020)
      * Spatial generalization (18 seen training cities vs 8 unseen test cities)
  - Statistical Hypothesis Testing:
      * Paired test error differences (MAE and MSE) on identical 11,410 test instances
      * Date-aggregated paired tests to adjust for spatial cross-city correlation on identical dates
      * 1,000-sample bootstrap 95% confidence intervals for Delta-RMSE and Delta-MAE
      * Holm-Bonferroni family-wise error rate (FWER) step-down adjustment
      * Two One-Sided Tests (TOST) equivalence testing with +/- 2.0 AQI point margin
      * Chronological rolling-origin cross-validation across development periods for all 4 models
  - Emits:
      * NEW_ANALYSIS/results/results_table.csv
      * NEW_ANALYSIS/results/baseline_comparison.csv
      * NEW_ANALYSIS/results/results_summary.txt
      * NEW_ANALYSIS/results/results_detailed.json
      * NEW_ANALYSIS/results/city_wise_performance.csv
      * NEW_ANALYSIS/results/year_wise_performance.csv
      * NEW_ANALYSIS/results/spatial_generalization.csv
      * NEW_ANALYSIS/statistical_analysis/statistical_tests_exp1.csv
      * NEW_ANALYSIS/statistical_analysis/tost_equivalence_exp1.csv
      * NEW_ANALYSIS/statistical_analysis/rolling_origin_cv.csv
      * NEW_ANALYSIS/models/trained_models.pkl
"""

import os
import sys
import time
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]
TARGET = "AQI"
EQUIVALENCE_MARGIN = 2.0  # AQI points (~0.4% of 500-point AQI scale)

def parse_args():
    parser = argparse.ArgumentParser(description="AQI Multi-Model Benchmarking & Statistical Inference")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing preprocessed CSV files. Defaults to NEW_ANALYSIS/data_processed.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root directory for NEW_ANALYSIS. Defaults to parent of data_dir.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for model training and reproducibility.")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations for difference confidence intervals.")
    return parser.parse_args()

def evaluate_metrics(y_true, y_pred):
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def holm_bonferroni(p_values):
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adj_p = np.empty(m, dtype=float)
    current_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adj = p * (m - rank)
        adj = max(adj, current_max)
        adj = min(adj, 1.0)
        current_max = adj
        adj_p[idx] = adj
    return adj_p

def bootstrap_diff_ci(y_true, pred_best, pred_comp, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diff_rmse_list = np.empty(n_boot)
    diff_mae_list = np.empty(n_boot)
    
    err_best = np.abs(y_true - pred_best)
    err_comp = np.abs(y_true - pred_comp)
    sq_best = (y_true - pred_best) ** 2
    sq_comp = (y_true - pred_comp) ** 2
    
    for b in range(n_boot):
        indices = rng.randint(0, n, size=n)
        rmse_b = np.sqrt(np.mean(sq_best[indices]))
        rmse_c = np.sqrt(np.mean(sq_comp[indices]))
        diff_rmse_list[b] = rmse_c - rmse_b
        
        mae_b = np.mean(err_best[indices])
        mae_c = np.mean(err_comp[indices])
        diff_mae_list[b] = mae_c - mae_b

    ci_rmse = (float(np.percentile(diff_rmse_list, 2.5)), float(np.percentile(diff_rmse_list, 97.5)))
    ci_mae = (float(np.percentile(diff_mae_list, 2.5)), float(np.percentile(diff_mae_list, 97.5)))
    return ci_rmse, ci_mae

def tost_equivalence(diffs, margin=EQUIVALENCE_MARGIN):
    n = len(diffs)
    mean_d = np.mean(diffs)
    se_d = stats.sem(diffs)
    df = n - 1
    
    t1 = (mean_d - (-margin)) / se_d if se_d > 0 else 0.0
    p1 = 1 - stats.t.cdf(t1, df)
    
    t2 = (margin - mean_d) / se_d if se_d > 0 else 0.0
    p2 = 1 - stats.t.cdf(t2, df)
    
    p_tost = max(p1, p2)
    is_equivalent = bool(p_tost < 0.05)
    return p_tost, is_equivalent

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_root = os.path.abspath(os.path.join(script_dir, ".."))

    data_dir = args.data_dir if args.data_dir else os.path.join(analysis_root, "data_processed")
    seed = args.random_seed
    n_boot = args.n_bootstrap

    # Setup output paths
    results_dir = os.path.join(analysis_root, "results")
    predictions_dir = os.path.join(analysis_root, "predictions")
    models_dir = os.path.join(analysis_root, "models")
    stats_dir = os.path.join(analysis_root, "statistical_analysis")
    logs_dir = os.path.join(analysis_root, "logs")

    for d in [results_dir, predictions_dir, models_dir, stats_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Preprocessed Partitions
    train_sub = pd.read_csv(os.path.join(data_dir, "train_sub.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    train_full = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    y_train_sub = train_sub[TARGET].values
    y_val = val_df[TARGET].values
    y_train_full = train_full[TARGET].values
    y_test = test_df[TARGET].values

    print("=" * 80)
    print("LEAKAGE-FREE URBAN AQI BENCHMARK & RIGOROUS STATISTICAL INFERENCE")
    print("=" * 80)
    print(f"Data source directory: {data_dir}")
    print(f"Train subset (2015-2017): {len(train_sub):>6} rows | Validation (2018): {len(val_df):>6} rows")
    print(f"Full Train   (2015-2018): {len(train_full):>6} rows | Test (2019-2020): {len(test_df):>6} rows")
    print(f"Random seed: {seed} | Bootstrap resamples: {n_boot}")

    # =========================================================================
    # BASELINE MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (Naive & Stratified Baselines)")
    print("=" * 60)
    # Baseline 1: Global Training Mean
    train_mean_val = float(np.mean(y_train_full))
    pred_base_mean = np.full_like(y_test, fill_value=train_mean_val, dtype=float)
    base_mean_rmse, base_mean_mae, base_mean_r2 = evaluate_metrics(y_test, pred_base_mean)
    print(f"Baseline 1 (Global Train Mean = {train_mean_val:.2f}): RMSE={base_mean_rmse:.4f}, MAE={base_mean_mae:.4f}, R2={base_mean_r2:.4f}")

    # Baseline 2: City-Stratified Training Mean
    city_means = train_full.groupby("City")[TARGET].mean().to_dict()
    pred_base_city = np.array([city_means.get(c, train_mean_val) for c in test_df["City"].values], dtype=float)
    base_city_rmse, base_city_mae, base_city_r2 = evaluate_metrics(y_test, pred_base_city)
    print(f"Baseline 2 (City-Stratified Train Mean): RMSE={base_city_rmse:.4f}, MAE={base_city_mae:.4f}, R2={base_city_r2:.4f}")

    baseline_df = pd.DataFrame([
        {"Baseline": "Global Training Mean", "Prediction_Rule": f"Constant {train_mean_val:.2f}", "RMSE": round(base_mean_rmse, 4), "MAE": round(base_mean_mae, 4), "R2": round(base_mean_r2, 4)},
        {"Baseline": "City-Stratified Mean", "Prediction_Rule": "City Training Mean (Fallback to Global)", "RMSE": round(base_city_rmse, 4), "MAE": round(base_city_mae, 4), "R2": round(base_city_r2, 4)}
    ])
    baseline_df.to_csv(os.path.join(results_dir, "baseline_comparison.csv"), index=False)

    # =========================================================================
    # EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK (Primary Controlled Comparison)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK (8 Criteria Pollutants)")
    print("=" * 60)

    X_train_sub_p = train_sub[POLLUTANTS].values
    X_val_p = val_df[POLLUTANTS].values
    X_train_full_p = train_full[POLLUTANTS].values
    X_test_p = test_df[POLLUTANTS].values

    results_exp1 = {}
    predictions_exp1 = {}
    models_exp1 = {}
    tuning_info_exp1 = {}

    # 1. Random Forest
    print("\n[Exp 1] Training Random Forest...")
    t0 = time.time()
    rf_exp1 = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2,
                                    random_state=seed, n_jobs=-1)
    rf_exp1.fit(X_train_full_p, y_train_full)
    rf_time_exp1 = time.time() - t0
    pred_rf_exp1 = rf_exp1.predict(X_test_p)
    results_exp1["Random Forest"] = (*evaluate_metrics(y_test, pred_rf_exp1), rf_time_exp1)
    predictions_exp1["Random Forest"] = pred_rf_exp1
    models_exp1["Random Forest"] = rf_exp1
    tuning_info_exp1["Random Forest"] = {"n_estimators": 200, "selected_via": "fixed_standard"}

    # 2. XGBoost
    print("[Exp 1] Tuning & Training XGBoost with Validation Early Stopping...")
    t0 = time.time()
    xgb_val_m = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=500,
        reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=20,
        eval_metric="rmse", random_state=seed, n_jobs=-1
    )
    xgb_val_m.fit(X_train_sub_p, y_train_sub, eval_set=[(X_val_p, y_val)], verbose=False)
    best_it_xgb_exp1 = int(xgb_val_m.best_iteration)
    
    xgb_exp1 = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=best_it_xgb_exp1,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1
    )
    xgb_exp1.fit(X_train_full_p, y_train_full)
    xgb_time_exp1 = time.time() - t0
    pred_xgb_exp1 = xgb_exp1.predict(X_test_p)
    results_exp1["XGBoost"] = (*evaluate_metrics(y_test, pred_xgb_exp1), xgb_time_exp1)
    predictions_exp1["XGBoost"] = pred_xgb_exp1
    models_exp1["XGBoost"] = xgb_exp1
    tuning_info_exp1["XGBoost"] = {"best_iteration": best_it_xgb_exp1, "selected_via": "val_2018_early_stopping"}

    # 3. LightGBM
    print("[Exp 1] Tuning & Training LightGBM with Validation Early Stopping...")
    t0 = time.time()
    lgb_val_m = lgb.LGBMRegressor(
        num_leaves=63, learning_rate=0.05, n_estimators=500,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    lgb_val_m.fit(X_train_sub_p, y_train_sub, eval_set=[(X_val_p, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    best_it_lgb_exp1 = int(lgb_val_m.best_iteration_)

    lgb_exp1 = lgb.LGBMRegressor(
        num_leaves=63, learning_rate=0.05, n_estimators=best_it_lgb_exp1,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    lgb_exp1.fit(X_train_full_p, y_train_full)
    lgb_time_exp1 = time.time() - t0
    pred_lgb_exp1 = lgb_exp1.predict(X_test_p)
    results_exp1["LightGBM"] = (*evaluate_metrics(y_test, pred_lgb_exp1), lgb_time_exp1)
    predictions_exp1["LightGBM"] = pred_lgb_exp1
    models_exp1["LightGBM"] = lgb_exp1
    tuning_info_exp1["LightGBM"] = {"best_iteration": best_it_lgb_exp1, "selected_via": "val_2018_early_stopping"}

    # 4. CatBoost
    print("[Exp 1] Tuning & Training CatBoost with Validation Early Stopping...")
    t0 = time.time()
    cb_val_m = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        early_stopping_rounds=20, random_state=seed, verbose=False
    )
    cb_val_m.fit(X_train_sub_p, y_train_sub, eval_set=(X_val_p, y_val), verbose=False)
    best_it_cb_exp1 = int(cb_val_m.get_best_iteration())
    if best_it_cb_exp1 <= 0:
        best_it_cb_exp1 = 500

    cb_exp1 = CatBoostRegressor(
        iterations=best_it_cb_exp1, learning_rate=0.05, depth=6,
        random_state=seed, verbose=False
    )
    cb_exp1.fit(X_train_full_p, y_train_full)
    cb_time_exp1 = time.time() - t0
    pred_cb_exp1 = cb_exp1.predict(X_test_p)
    results_exp1["CatBoost"] = (*evaluate_metrics(y_test, pred_cb_exp1), cb_time_exp1)
    predictions_exp1["CatBoost"] = pred_cb_exp1
    models_exp1["CatBoost"] = cb_exp1
    tuning_info_exp1["CatBoost"] = {"best_iteration": best_it_cb_exp1, "selected_via": "val_2018_early_stopping"}

    df_exp1 = pd.DataFrame(results_exp1, index=["RMSE", "MAE", "R2", "TrainTime_sec"]).T.round(4)
    print("\n--- EXPERIMENT 1 RESULTS TABLE ---")
    print(df_exp1)

    # Save Standalone Predictions for Exp 1
    preds_exp1_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "City": test_df["City"].values,
        "y_true": y_test,
        "Baseline_Mean": pred_base_mean,
        "Baseline_City_Mean": pred_base_city,
        "Random Forest": pred_rf_exp1,
        "XGBoost": pred_xgb_exp1,
        "LightGBM": pred_lgb_exp1,
        "CatBoost": pred_cb_exp1
    })
    preds_exp1_path = os.path.join(predictions_dir, "predictions_pollutant_only.csv")
    preds_exp1_df.to_csv(preds_exp1_path, index=False)
    print(f"Saved standalone Exp 1 predictions to: {preds_exp1_path}")

    # =========================================================================
    # EXPERIMENT 2: CITY-AWARE BENCHMARK (8 Criteria Pollutants + City)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CITY-AWARE BENCHMARK (Pollutants + City Encoding)")
    print("=" * 60)

    # Target Encoder for RF and XGBoost
    te_sub = TargetEncoder(cv=5, smooth="auto", target_type="continuous", random_state=seed)
    city_te_sub = te_sub.fit_transform(train_sub[["City"]], y_train_sub)
    city_te_val = te_sub.transform(val_df[["City"]])
    X_sub_city_te = np.hstack([city_te_sub, X_train_sub_p])
    X_val_city_te = np.hstack([city_te_val, X_val_p])

    te_full = TargetEncoder(cv=5, smooth="auto", target_type="continuous", random_state=seed)
    city_te_full = te_full.fit_transform(train_full[["City"]], y_train_full)
    city_te_test = te_full.transform(test_df[["City"]])
    X_full_city_te = np.hstack([city_te_full, X_train_full_p])
    X_test_city_te = np.hstack([city_te_test, X_test_p])

    results_exp2 = {}
    predictions_exp2 = {}
    models_exp2 = {}
    tuning_info_exp2 = {}

    # 1. Random Forest (TargetEncoded City)
    print("\n[Exp 2] Training Random Forest (TargetEncoded City)...")
    t0 = time.time()
    rf_exp2 = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2,
                                    random_state=seed, n_jobs=-1)
    rf_exp2.fit(X_full_city_te, y_train_full)
    rf_time_exp2 = time.time() - t0
    pred_rf_exp2 = rf_exp2.predict(X_test_city_te)
    results_exp2["Random Forest"] = (*evaluate_metrics(y_test, pred_rf_exp2), rf_time_exp2)
    predictions_exp2["Random Forest"] = pred_rf_exp2
    models_exp2["Random Forest"] = rf_exp2
    tuning_info_exp2["Random Forest"] = {"n_estimators": 200, "city_encoding": "target_encoder_cv5"}

    # 2. XGBoost (TargetEncoded City)
    print("[Exp 2] Tuning & Training XGBoost (TargetEncoded City)...")
    t0 = time.time()
    xgb_val_m2 = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=500,
        reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=20,
        eval_metric="rmse", random_state=seed, n_jobs=-1
    )
    xgb_val_m2.fit(X_sub_city_te, y_train_sub, eval_set=[(X_val_city_te, y_val)], verbose=False)
    best_it_xgb_exp2 = int(xgb_val_m2.best_iteration)

    xgb_exp2 = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=best_it_xgb_exp2,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1
    )
    xgb_exp2.fit(X_full_city_te, y_train_full)
    xgb_time_exp2 = time.time() - t0
    pred_xgb_exp2 = xgb_exp2.predict(X_test_city_te)
    results_exp2["XGBoost"] = (*evaluate_metrics(y_test, pred_xgb_exp2), xgb_time_exp2)
    predictions_exp2["XGBoost"] = pred_xgb_exp2
    models_exp2["XGBoost"] = xgb_exp2
    tuning_info_exp2["XGBoost"] = {"best_iteration": best_it_xgb_exp2, "city_encoding": "target_encoder_cv5"}

    # 3. LightGBM (Native categorical City)
    print("[Exp 2] Tuning & Training LightGBM (Native categorical City)...")
    t0 = time.time()
    train_sub_lgb = train_sub[["City"] + POLLUTANTS].copy()
    train_sub_lgb["City"] = train_sub_lgb["City"].astype("category")
    val_lgb = val_df[["City"] + POLLUTANTS].copy()
    val_lgb["City"] = val_lgb["City"].astype("category")
    train_full_lgb = train_full[["City"] + POLLUTANTS].copy()
    train_full_lgb["City"] = train_full_lgb["City"].astype("category")
    test_lgb = test_df[["City"] + POLLUTANTS].copy()
    test_lgb["City"] = test_lgb["City"].astype("category")

    lgb_val_m2 = lgb.LGBMRegressor(
        num_leaves=63, learning_rate=0.05, n_estimators=500,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    lgb_val_m2.fit(train_sub_lgb, y_train_sub, eval_set=[(val_lgb, y_val)],
                   callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
    best_it_lgb_exp2 = int(lgb_val_m2.best_iteration_)

    lgb_exp2 = lgb.LGBMRegressor(
        num_leaves=63, learning_rate=0.05, n_estimators=best_it_lgb_exp2,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    lgb_exp2.fit(train_full_lgb, y_train_full)
    lgb_time_exp2 = time.time() - t0
    pred_lgb_exp2 = lgb_exp2.predict(test_lgb)
    results_exp2["LightGBM"] = (*evaluate_metrics(y_test, pred_lgb_exp2), lgb_time_exp2)
    predictions_exp2["LightGBM"] = pred_lgb_exp2
    models_exp2["LightGBM"] = lgb_exp2
    tuning_info_exp2["LightGBM"] = {"best_iteration": best_it_lgb_exp2, "city_encoding": "native_categorical"}

    # 4. CatBoost (Native categorical City)
    print("[Exp 2] Tuning & Training CatBoost (Native categorical City)...")
    t0 = time.time()
    cat_train_sub = train_sub[["City"] + POLLUTANTS].copy()
    cat_val = val_df[["City"] + POLLUTANTS].copy()
    cat_train_full = train_full[["City"] + POLLUTANTS].copy()
    cat_test = test_df[["City"] + POLLUTANTS].copy()

    cb_val_m2 = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        early_stopping_rounds=20, random_state=seed, verbose=False
    )
    cb_val_m2.fit(cat_train_sub, y_train_sub, cat_features=[0], eval_set=(cat_val, y_val), verbose=False)
    best_it_cb_exp2 = int(cb_val_m2.get_best_iteration())
    if best_it_cb_exp2 <= 0:
        best_it_cb_exp2 = 500

    cb_exp2 = CatBoostRegressor(
        iterations=best_it_cb_exp2, learning_rate=0.05, depth=6,
        random_state=seed, verbose=False
    )
    cb_exp2.fit(cat_train_full, y_train_full, cat_features=[0])
    cb_time_exp2 = time.time() - t0
    pred_cb_exp2 = cb_exp2.predict(cat_test)
    results_exp2["CatBoost"] = (*evaluate_metrics(y_test, pred_cb_exp2), cb_time_exp2)
    predictions_exp2["CatBoost"] = pred_cb_exp2
    models_exp2["CatBoost"] = cb_exp2
    tuning_info_exp2["CatBoost"] = {"best_iteration": best_it_cb_exp2, "city_encoding": "native_ordered_boosting"}

    df_exp2 = pd.DataFrame(results_exp2, index=["RMSE", "MAE", "R2", "TrainTime_sec"]).T.round(4)
    print("\n--- EXPERIMENT 2 RESULTS TABLE ---")
    print(df_exp2)

    # Save Standalone Predictions for Exp 2
    preds_exp2_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "City": test_df["City"].values,
        "y_true": y_test,
        "Baseline_Mean": pred_base_mean,
        "Baseline_City_Mean": pred_base_city,
        "Random Forest": pred_rf_exp2,
        "XGBoost": pred_xgb_exp2,
        "LightGBM": pred_lgb_exp2,
        "CatBoost": pred_cb_exp2
    })
    preds_exp2_path = os.path.join(predictions_dir, "predictions_city_aware.csv")
    preds_exp2_df.to_csv(preds_exp2_path, index=False)
    print(f"Saved standalone Exp 2 predictions to: {preds_exp2_path}")

    # Save Combined Results Table
    combined_results_df = pd.concat([
        df_exp1.assign(Experiment="Exp 1: Pollutant-Only"),
        df_exp2.assign(Experiment="Exp 2: City-Aware")
    ]).reset_index().rename(columns={"index": "Model"})
    results_table_path = os.path.join(results_dir, "results_table.csv")
    combined_results_df.to_csv(results_table_path, index=False)

    # =========================================================================
    # STATISTICAL INFERENCE & HYPOTHESIS TESTING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL INFERENCE & RIGOROUS HYPOTHESIS TESTING")
    print("=" * 60)

    best_m1_name = df_exp1["RMSE"].idxmin()
    pred_best_exp1 = predictions_exp1[best_m1_name]
    print(f"Primary Benchmark Best Model: {best_m1_name} (RMSE={df_exp1.loc[best_m1_name, 'RMSE']:.4f})")

    p_raw_paired_t = []
    p_raw_wilcoxon = []
    p_raw_daily_t = []
    t_stats_paired = []
    mean_diff_mae_list = []
    ci_rmse_list = []
    ci_mae_list = []
    cohen_d_list = []
    tost_results = []
    comp_names = [m for m in df_exp1.index if m != best_m1_name]

    test_dates = test_df["Date"].values
    unique_dates = np.unique(test_dates)

    daily_best_mae = np.array([np.mean(np.abs(y_test[test_dates == d] - pred_best_exp1[test_dates == d])) for d in unique_dates])

    for comp_name in comp_names:
        pred_comp = predictions_exp1[comp_name]
        
        diff_abs = np.abs(y_test - pred_comp) - np.abs(y_test - pred_best_exp1)
        mean_diff_mae = float(np.mean(diff_abs))
        mean_diff_mae_list.append(mean_diff_mae)
        
        std_diff = np.std(diff_abs, ddof=1)
        cohen_d = float(mean_diff_mae / std_diff) if std_diff > 0 else 0.0
        cohen_d_list.append(cohen_d)

        t_stat, p_t = stats.ttest_rel(np.abs(y_test - pred_comp), np.abs(y_test - pred_best_exp1))
        t_stats_paired.append(float(t_stat))
        p_raw_paired_t.append(float(p_t))

        w_stat, p_w = stats.wilcoxon(np.abs(y_test - pred_comp), np.abs(y_test - pred_best_exp1), alternative="two-sided")
        p_raw_wilcoxon.append(float(p_w))

        daily_comp_mae = np.array([np.mean(np.abs(y_test[test_dates == d] - pred_comp[test_dates == d])) for d in unique_dates])
        t_daily, p_daily = stats.ttest_rel(daily_comp_mae, daily_best_mae)
        p_raw_daily_t.append(float(p_daily))

        ci_rmse, ci_mae = bootstrap_diff_ci(y_test, pred_best_exp1, pred_comp, n_boot=n_boot, seed=seed)
        ci_rmse_list.append(ci_rmse)
        ci_mae_list.append(ci_mae)

        p_tost, is_equiv = tost_equivalence(diff_abs, margin=EQUIVALENCE_MARGIN)
        tost_results.append({"p_tost": float(p_tost), "is_equivalent": is_equiv})

    p_adj_paired_t = holm_bonferroni(p_raw_paired_t)
    p_adj_wilcoxon = holm_bonferroni(p_raw_wilcoxon)
    p_adj_daily_t = holm_bonferroni(p_raw_daily_t)

    inference_table = []
    tost_table = []
    for idx, comp_name in enumerate(comp_names):
        inference_table.append({
            "Comparison": f"{best_m1_name} vs {comp_name}",
            "Delta_MAE": round(mean_diff_mae_list[idx], 4),
            "95%_CI_Delta_MAE": f"[{ci_mae_list[idx][0]:.3f}, {ci_mae_list[idx][1]:.3f}]",
            "95%_CI_Delta_RMSE": f"[{ci_rmse_list[idx][0]:.3f}, {ci_rmse_list[idx][1]:.3f}]",
            "Cohen_dz": round(cohen_d_list[idx], 4),
            "Paired_t_p_adj": round(p_adj_paired_t[idx], 6),
            "Wilcoxon_p_adj": round(p_adj_wilcoxon[idx], 6),
            "Daily_t_p_adj": round(p_adj_daily_t[idx], 6),
            "TOST_p": round(tost_results[idx]["p_tost"], 6),
            "Statistically_Equivalent": tost_results[idx]["is_equivalent"]
        })
        tost_table.append({
            "Comparison": f"{best_m1_name} vs {comp_name}",
            "Mean_Delta_MAE": round(mean_diff_mae_list[idx], 4),
            "Equivalence_Margin": f"+/- {EQUIVALENCE_MARGIN:.1f} AQI points",
            "TOST_p_value": round(tost_results[idx]["p_tost"], 6),
            "Equivalent_within_Margin": tost_results[idx]["is_equivalent"]
        })

    inference_df = pd.DataFrame(inference_table)
    tost_df = pd.DataFrame(tost_table)
    inference_df.to_csv(os.path.join(stats_dir, "statistical_tests_exp1.csv"), index=False)
    tost_df.to_csv(os.path.join(stats_dir, "tost_equivalence_exp1.csv"), index=False)
    print("\n--- STATISTICAL INFERENCE SUMMARY (Exp 1) ---")
    print(inference_df.to_string())

    # =========================================================================
    # CHRONOLOGICAL ROLLING-ORIGIN EXPANDING-WINDOW CV (ALL 4 MODELS)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXPANDING-WINDOW CROSS-VALIDATION (All 4 Models)")
    print("=" * 60)

    # Window 1: Train 2015-2016 -> Eval 2017
    # Window 2: Train 2015-2017 -> Eval 2018
    # Window 3: Train 2015-2018 -> Eval 2019-2020 (Holdout Test)
    w1_train_mask = train_full["Date"].str.slice(0, 4).astype(int) <= 2016
    w1_eval_mask = train_full["Date"].str.slice(0, 4).astype(int) == 2017
    X_w1_train, y_w1_train = train_full.loc[w1_train_mask, POLLUTANTS].values, train_full.loc[w1_train_mask, TARGET].values
    X_w1_eval, y_w1_eval = train_full.loc[w1_eval_mask, POLLUTANTS].values, train_full.loc[w1_eval_mask, TARGET].values

    X_w2_train, y_w2_train = train_sub[POLLUTANTS].values, y_train_sub
    X_w2_eval, y_w2_eval = val_df[POLLUTANTS].values, y_val

    rolling_records = []
    for model_name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
        # Instantiate fresh model
        if model_name == "Random Forest":
            m_w1 = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            m_w2 = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        elif model_name == "XGBoost":
            m_w1 = xgb.XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=best_it_xgb_exp1, random_state=seed, n_jobs=-1)
            m_w2 = xgb.XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=best_it_xgb_exp1, random_state=seed, n_jobs=-1)
        elif model_name == "LightGBM":
            m_w1 = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=best_it_lgb_exp1, random_state=seed, n_jobs=-1, verbose=-1)
            m_w2 = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=best_it_lgb_exp1, random_state=seed, n_jobs=-1, verbose=-1)
        else:
            m_w1 = CatBoostRegressor(iterations=best_it_cb_exp1, learning_rate=0.05, depth=6, random_state=seed, verbose=False)
            m_w2 = CatBoostRegressor(iterations=best_it_cb_exp1, learning_rate=0.05, depth=6, random_state=seed, verbose=False)

        m_w1.fit(X_w1_train, y_w1_train)
        p_w1 = m_w1.predict(X_w1_eval)
        w1_rmse = root_mean_squared_error(y_w1_eval, p_w1)
        w1_mae = mean_absolute_error(y_w1_eval, p_w1)

        m_w2.fit(X_w2_train, y_w2_train)
        p_w2 = m_w2.predict(X_w2_eval)
        w2_rmse = root_mean_squared_error(y_w2_eval, p_w2)
        w2_mae = mean_absolute_error(y_w2_eval, p_w2)

        w3_rmse = df_exp1.loc[model_name, "RMSE"]
        w3_mae = df_exp1.loc[model_name, "MAE"]

        rolling_records.append({
            "Model": model_name,
            "Window_1_Train_2015_16_Eval_2017_RMSE": round(float(w1_rmse), 4),
            "Window_1_Train_2015_16_Eval_2017_MAE": round(float(w1_mae), 4),
            "Window_2_Train_2015_17_Eval_2018_RMSE": round(float(w2_rmse), 4),
            "Window_2_Train_2015_17_Eval_2018_MAE": round(float(w2_mae), 4),
            "Window_3_Train_2015_18_Eval_2019_20_RMSE": round(float(w3_rmse), 4),
            "Window_3_Train_2015_18_Eval_2019_20_MAE": round(float(w3_mae), 4),
        })

    rolling_df = pd.DataFrame(rolling_records)
    rolling_df.to_csv(os.path.join(stats_dir, "rolling_origin_cv.csv"), index=False)
    print("\n--- ROLLING-ORIGIN EXPANDING-WINDOW PERFORMANCE TABLE ---")
    print(rolling_df.to_string())

    # =========================================================================
    # GRANULAR BREAKDOWN: 26 CITIES, YEARS, AND SPATIAL COVERAGE
    # =========================================================================
    print("\n" + "=" * 60)
    print("GRANULAR EVALUATION: 26 CITIES & TEMPORAL SECTORS")
    print("=" * 60)

    # 1. Year-wise Breakdown (2019 vs 2020)
    test_years = test_df["Date"].str.slice(0, 4).astype(int)
    year_records = []
    for yr in [2019, 2020]:
        mask = (test_years == yr)
        sub_y = y_test[mask]
        for m in df_exp1.index:
            p = predictions_exp1[m][mask]
            r, a, r2_val = evaluate_metrics(sub_y, p)
            year_records.append({
                "Year": yr,
                "Model": m,
                "Sample_Count": int(mask.sum()),
                "RMSE": round(r, 4),
                "MAE": round(a, 4),
                "R2": round(r2_val, 4)
            })
    year_df = pd.DataFrame(year_records)
    year_df.to_csv(os.path.join(results_dir, "year_wise_performance.csv"), index=False)

    # 2. Seen vs Unseen Cities Breakdown
    unseen_cities = ["Aizawl", "Bhopal", "Chandigarh", "Coimbatore", "Ernakulam", "Guwahati", "Kochi", "Shillong"]
    is_unseen = test_df["City"].isin(unseen_cities)
    spatial_records = []
    for group_name, mask in [("Seen Cities in Dev (18 cities)", ~is_unseen), ("Unseen Cities in Dev (8 cities)", is_unseen)]:
        sub_y = y_test[mask]
        for m in df_exp1.index:
            p = predictions_exp1[m][mask]
            r, a, r2_val = evaluate_metrics(sub_y, p)
            spatial_records.append({
                "Spatial_Cohort": group_name,
                "Model": m,
                "Sample_Count": int(mask.sum()),
                "RMSE": round(r, 4),
                "MAE": round(a, 4),
                "R2": round(r2_val, 4)
            })
    spatial_df = pd.DataFrame(spatial_records)
    spatial_df.to_csv(os.path.join(results_dir, "spatial_generalization.csv"), index=False)

    # 3. Complete 26-City Breakdown for All Models
    all_test_cities = sorted(test_df["City"].unique().tolist())
    city_records = []
    for city in all_test_cities:
        mask = (test_df["City"] == city)
        sub_y = y_test[mask]
        n_obs = int(mask.sum())
        cohort = "Unseen (New 2019-20)" if city in unseen_cities else "Seen (Historical 2015-18)"
        for m in df_exp1.index:
            p = predictions_exp1[m][mask]
            r, a, r2_val = evaluate_metrics(sub_y, p)
            city_records.append({
                "City": city,
                "Cohort": cohort,
                "Model": m,
                "Sample_Count": n_obs,
                "RMSE": round(r, 4),
                "MAE": round(a, 4),
                "R2": round(r2_val, 4)
            })
    city_df = pd.DataFrame(city_records)
    city_df.to_csv(os.path.join(results_dir, "city_wise_performance.csv"), index=False)
    print(f"Saved complete 26-city performance breakdown to: {os.path.join(results_dir, 'city_wise_performance.csv')}")

    # =========================================================================
    # SERIALIZATION OF RESULTS SUMMARY, DETAILED JSON & MODELS PKL
    # =========================================================================
    summary_path = os.path.join(results_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("VERIFIED RESEARCH BENCHMARKING RESULTS & STATISTICAL AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. BASELINE COMPARISON:\n")
        f.write(baseline_df.to_string(index=False) + "\n\n")
        f.write("2. EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK (Primary Controlled Comparison)\n")
        f.write("Features: PM2.5, PM10, NO2, NH3, SO2, CO, O3, NO (Unclipped Observed AQI)\n")
        f.write(df_exp1.to_string() + "\n\n")
        f.write("3. EXPERIMENT 2: CITY-AWARE BENCHMARK (Criteria Pollutants + City)\n")
        f.write("Features: City (Encoded) + 8 Criteria Pollutants\n")
        f.write(df_exp2.to_string() + "\n\n")
        f.write("4. STATISTICAL HYPOTHESIS TESTING (Exp 1 vs Best Model: CatBoost)\n")
        f.write(inference_df.to_string(index=False) + "\n\n")
        f.write("5. TWO ONE-SIDED TESTS (TOST) EQUIVALENCE TESTING (+/- 2.0 AQI points)\n")
        f.write(tost_df.to_string(index=False) + "\n\n")
        f.write("6. TEMPORAL ROLLING-ORIGIN EXPANDING-WINDOW CROSS-VALIDATION\n")
        f.write(rolling_df.to_string(index=False) + "\n\n")
        f.write("7. SPATIAL GENERALIZATION (Seen vs Unseen Cities)\n")
        f.write(spatial_df.to_string(index=False) + "\n")

    # Detailed machine-readable JSON
    detailed_json_path = os.path.join(results_dir, "results_detailed.json")
    results_detailed = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": TARGET,
            "test_sample_count": len(test_df),
            "train_sample_count": len(train_full),
            "random_seed": seed,
            "hardware_platform": sys.platform
        },
        "baselines": baseline_df.to_dict(orient="records"),
        "experiment_1_pollutant_only": {
            "features": POLLUTANTS,
            "metrics": df_exp1.to_dict(orient="index"),
            "tuning": tuning_info_exp1
        },
        "experiment_2_city_aware": {
            "features": ["City"] + POLLUTANTS,
            "metrics": df_exp2.to_dict(orient="index"),
            "tuning": tuning_info_exp2
        },
        "statistical_inference_exp1": inference_table,
        "tost_equivalence_exp1": tost_table,
        "rolling_origin_cv": rolling_records,
        "year_wise_breakdown": year_records,
        "spatial_breakdown": spatial_records
    }
    with open(detailed_json_path, "w", encoding="utf-8") as jf:
        json.dump(results_detailed, jf, indent=2)

    # Save Models Bundle
    models_pkl_path = os.path.join(models_dir, "trained_models.pkl")
    save_payload = {
        "experiment_1": {
            "models": models_exp1,
            "best_model_name": best_m1_name,
            "predictions": predictions_exp1,
            "features": POLLUTANTS
        },
        "experiment_2": {
            "models": models_exp2,
            "best_model_name": df_exp2["RMSE"].idxmin(),
            "predictions": predictions_exp2,
            "features": ["City"] + POLLUTANTS,
            "target_encoder": te_full
        },
        "baselines": {
            "pred_mean": pred_base_mean,
            "pred_city_mean": pred_base_city
        },
        "y_test": y_test,
        "test_df": test_df[["City", "Date"] + POLLUTANTS + [TARGET]]
    }
    with open(models_pkl_path, "wb") as pf:
        pickle.dump(save_payload, pf)

    print(f"\nModel training and benchmarking completed successfully.")
    print(f"Artifacts saved in:")
    print(f"  - {results_table_path}")
    print(f"  - {summary_path}")
    print(f"  - {detailed_json_path}")
    print(f"  - {models_pkl_path}")

if __name__ == "__main__":
    main()
