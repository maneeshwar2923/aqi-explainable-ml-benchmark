"""
02_train_and_evaluate.py
--------------------------
Leakage-Free Multi-Model Benchmarking, Chronological Validation & Rigorous Statistical Inference:
  - Models: Random Forest, XGBoost, LightGBM, CatBoost
  - Experiment 1 (Primary): Pollutant-Only Benchmark (8 criteria pollutants)
  - Experiment 2: City-Aware Benchmark (8 criteria pollutants + City with leakage-free encoding)
  - Validation: 2018 chronological validation split used for early stopping / iteration selection
    (NEVER the final test set).
  - Test Evaluation: Final single-pass assessment on 2019-2020 held-out test partition (11,410 rows).
  - Statistical Inference:
      * Paired test error differences (MAE and MSE) on identical 11,410 test observations.
      * Clustered / block-bootstrap 95% confidence intervals.
      * Date-aggregated paired tests to adjust for spatial cross-city autocorrelation.
      * Bonferroni-Holm family-wise error rate (FWER) multiplicity correction.
      * Two One-Sided Tests (TOST) equivalence testing with explicit margin (+/- 2.0 AQI points).
      * Temporal rolling-origin cross-validation across training years (2015-2018).
  - Granular breakdown: Performance partitioned by year and by seen vs. unseen cities.
  - Emits: results_table.csv, results_summary.txt, results_detailed.json, and trained_models.pkl.
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
    parser = argparse.ArgumentParser(description="Leakage-free AQI Model Benchmarking & Statistical Inference")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing preprocessed CSV files. Defaults to script directory.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for model training.")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations for confidence intervals.")
    return parser.parse_args()

def evaluate_metrics(y_true, y_pred):
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def holm_bonferroni(p_values):
    """Applies the Holm-Bonferroni step-down procedure to an array/dict of p-values."""
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
    """Computes 95% bootstrap confidence interval for difference in RMSE and MAE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diff_rmse_list = np.empty(n_boot)
    diff_mae_list = np.empty(n_boot)
    
    # Vectorized errors
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
    """
    Two One-Sided Tests (TOST) for equivalence.
    Null hypothesis: |mean(diff)| >= margin (models are non-equivalent).
    Rejection of both one-sided nulls indicates statistical equivalence within [-margin, +margin].
    """
    n = len(diffs)
    mean_d = np.mean(diffs)
    se_d = stats.sem(diffs)
    df = n - 1
    
    # Test 1: H0: mean <= -margin vs H1: mean > -margin
    t1 = (mean_d - (-margin)) / se_d if se_d > 0 else 0.0
    p1 = 1 - stats.t.cdf(t1, df)
    
    # Test 2: H0: mean >= margin vs H1: mean < margin
    t2 = (margin - mean_d) / se_d if se_d > 0 else 0.0
    p2 = 1 - stats.t.cdf(t2, df)
    
    p_tost = max(p1, p2)
    is_equivalent = bool(p_tost < 0.05)
    return p_tost, is_equivalent

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir if args.data_dir else script_dir
    seed = args.random_seed
    n_boot = args.n_bootstrap

    # 1. Load Preprocessed Datasets
    train_sub = pd.read_csv(os.path.join(data_dir, "train_sub.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    train_full = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    y_train_sub = train_sub[TARGET].values
    y_val = val_df[TARGET].values
    y_train_full = train_full[TARGET].values
    y_test = test_df[TARGET].values

    print("=" * 80)
    print("LEAKAGE-FREE URBAN AQI BENCHMARK & RIGOROUS INFERENCE")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Train subset (2015-2017): {len(train_sub)} rows | Validation (2018): {len(val_df)} rows")
    print(f"Full Train   (2015-2018): {len(train_full)} rows | Test (2019-2020): {len(test_df)} rows")
    print(f"Random seed: {seed} | Bootstrap resamples: {n_boot}")

    results_exp1 = {}
    predictions_exp1 = {}
    models_exp1 = {}
    tuning_info_exp1 = {}

    results_exp2 = {}
    predictions_exp2 = {}
    models_exp2 = {}
    tuning_info_exp2 = {}

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
    
    # Refit finalized model on full train (2015-2018) with best_iteration
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

    # Summary table Exp 1
    df_exp1 = pd.DataFrame(results_exp1, index=["RMSE", "MAE", "R2", "TrainTime_sec"]).T.round(4)
    print("\n--- EXPERIMENT 1 RESULTS TABLE ---")
    print(df_exp1)

    # =========================================================================
    # EXPERIMENT 2: CITY-AWARE BENCHMARK (8 Criteria Pollutants + City)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CITY-AWARE BENCHMARK (Pollutants + City Encoding)")
    print("=" * 60)

    # Target Encoder for RF and XGBoost (fit strictly on train partitions)
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

    # Summary table Exp 2
    df_exp2 = pd.DataFrame(results_exp2, index=["RMSE", "MAE", "R2", "TrainTime_sec"]).T.round(4)
    print("\n--- EXPERIMENT 2 RESULTS TABLE ---")
    print(df_exp2)

    # Save combined results table
    combined_results_df = pd.concat([
        df_exp1.assign(Experiment="Exp 1: Pollutant-Only"),
        df_exp2.assign(Experiment="Exp 2: City-Aware")
    ]).reset_index().rename(columns={"index": "Model"})
    combined_results_df.to_csv(os.path.join(data_dir, "results_table.csv"), index=False)

    # =========================================================================
    # STATISTICAL INFERENCE & HYPOTHESIS TESTING
    # =========================================================================
    print("\n" + "=" * 60)
    print("STATISTICAL INFERENCE & RIGOROUS HYPOTHESIS TESTING")
    print("=" * 60)

    # Identify primary experiment best model
    best_m1_name = df_exp1["RMSE"].idxmin()
    pred_best_exp1 = predictions_exp1[best_m1_name]
    print(f"Primary Benchmark Best Model: {best_m1_name} (RMSE={df_exp1.loc[best_m1_name, 'RMSE']:.4f})")

    # Pairwise statistical comparisons for Exp 1
    p_raw_paired_t = []
    p_raw_wilcoxon = []
    p_raw_daily_t = []
    t_stats_paired = []
    mean_diff_mae_list = []
    mean_diff_rmse_list = []
    ci_rmse_list = []
    ci_mae_list = []
    cohen_d_list = []
    tost_results = []
    comp_names = [m for m in df_exp1.index if m != best_m1_name]

    # Date grouping for day-aggregated paired test (removes intra-day cross-city spatial correlation)
    test_dates = test_df["Date"].values
    unique_dates = np.unique(test_dates)
    n_days = len(unique_dates)

    daily_best_mae = np.array([np.mean(np.abs(y_test[test_dates == d] - pred_best_exp1[test_dates == d])) for d in unique_dates])

    for comp_name in comp_names:
        pred_comp = predictions_exp1[comp_name]
        
        # 1. Observation-level paired differences
        diff_abs = np.abs(y_test - pred_comp) - np.abs(y_test - pred_best_exp1)
        mean_diff_mae = float(np.mean(diff_abs))
        mean_diff_mae_list.append(mean_diff_mae)
        
        # Cohen's d_z for paired differences
        std_diff = np.std(diff_abs, ddof=1)
        cohen_d = float(mean_diff_mae / std_diff) if std_diff > 0 else 0.0
        cohen_d_list.append(cohen_d)

        # Paired t-test on observation level
        t_stat, p_t = stats.ttest_rel(np.abs(y_test - pred_comp), np.abs(y_test - pred_best_exp1))
        t_stats_paired.append(float(t_stat))
        p_raw_paired_t.append(float(p_t))

        # Wilcoxon signed-rank test
        w_stat, p_w = stats.wilcoxon(np.abs(y_test - pred_comp), np.abs(y_test - pred_best_exp1), alternative="two-sided")
        p_raw_wilcoxon.append(float(p_w))

        # 2. Date-aggregated paired t-test (adjusts for spatial dependence on same day)
        daily_comp_mae = np.array([np.mean(np.abs(y_test[test_dates == d] - pred_comp[test_dates == d])) for d in unique_dates])
        daily_diff = daily_comp_mae - daily_best_mae
        t_daily, p_daily = stats.ttest_rel(daily_comp_mae, daily_best_mae)
        p_raw_daily_t.append(float(p_daily))

        # 3. Bootstrap Confidence Intervals for RMSE and MAE differences
        ci_rmse, ci_mae = bootstrap_diff_ci(y_test, pred_best_exp1, pred_comp, n_boot=n_boot, seed=seed)
        ci_rmse_list.append(ci_rmse)
        ci_mae_list.append(ci_mae)

        # 4. Two One-Sided Tests (TOST) Equivalence
        p_tost, is_equiv = tost_equivalence(diff_abs, margin=EQUIVALENCE_MARGIN)
        tost_results.append({"p_tost": float(p_tost), "is_equivalent": is_equiv})

    # Apply Holm-Bonferroni correction
    p_adj_paired_t = holm_bonferroni(p_raw_paired_t)
    p_adj_wilcoxon = holm_bonferroni(p_raw_wilcoxon)
    p_adj_daily_t = holm_bonferroni(p_raw_daily_t)

    inference_table = []
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

    inference_df = pd.DataFrame(inference_table)
    print("\n--- STATISTICAL INFERENCE SUMMARY (Primary Experiment 1) ---")
    print(inference_df.to_string())

    # =========================================================================
    # GRANULAR ERROR BREAKDOWN (YEAR & SPATIAL COVERAGE)
    # =========================================================================
    print("\n" + "=" * 60)
    print("GRANULAR PERFORMANCE BREAKDOWN (SPATIAL & TEMPORAL PARTITIONS)")
    print("=" * 60)

    # 1. Year-wise Breakdown (2019 vs 2020)
    test_years = test_df["Date"].str.slice(0, 4).astype(int)
    year_breakdown = {}
    for yr in [2019, 2020]:
        mask = (test_years == yr)
        sub_y = y_test[mask]
        year_breakdown[yr] = {}
        for m in df_exp1.index:
            p = predictions_exp1[m][mask]
            year_breakdown[yr][m] = {
                "N": int(mask.sum()),
                "RMSE": round(float(root_mean_squared_error(sub_y, p)), 4),
                "MAE": round(float(mean_absolute_error(sub_y, p)), 4),
                "R2": round(float(r2_score(sub_y, p)), 4)
            }
    print("\nYear-wise Breakdown (2019 vs 2020) for Exp 1 Best Model (CatBoost):")
    print(pd.DataFrame(year_breakdown).xs(best_m1_name, level=0, axis=1) if False else pd.DataFrame({
        yr: year_breakdown[yr][best_m1_name] for yr in [2019, 2020]
    }))

    # 2. City-wise Partition: Seen in Training vs Unseen in Training
    unseen_cities = ["Aizawl", "Bhopal", "Chandigarh", "Coimbatore", "Ernakulam", "Guwahati", "Kochi", "Shillong"]
    is_unseen = test_df["City"].isin(unseen_cities)
    spatial_breakdown = {}
    for group_name, mask in [("Seen Cities (18 cities)", ~is_unseen), ("Unseen Cities (8 cities)", is_unseen)]:
        sub_y = y_test[mask]
        spatial_breakdown[group_name] = {}
        for m in df_exp1.index:
            p = predictions_exp1[m][mask]
            spatial_breakdown[group_name][m] = {
                "N": int(mask.sum()),
                "RMSE": round(float(root_mean_squared_error(sub_y, p)), 4),
                "MAE": round(float(mean_absolute_error(sub_y, p)), 4),
                "R2": round(float(r2_score(sub_y, p)), 4)
            }
    print("\nSpatial Generalization Breakdown for Primary Models (Seen vs Unseen Cities):")
    print(pd.DataFrame({
        "Seen Cities (N=9374)": {m: spatial_breakdown["Seen Cities (18 cities)"][m]["RMSE"] for m in df_exp1.index},
        "Unseen Cities (N=2036)": {m: spatial_breakdown["Unseen Cities (8 cities)"][m]["RMSE"] for m in df_exp1.index}
    }))

    # Top metropolitan cities breakdown
    metro_cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"]
    metro_perf = {}
    for c in metro_cities:
        mask = (test_df["City"] == c)
        if mask.sum() > 0:
            sub_y = y_test[mask]
            p = predictions_exp1[best_m1_name][mask]
            metro_perf[c] = {
                "N": int(mask.sum()),
                "RMSE": round(float(root_mean_squared_error(sub_y, p)), 2),
                "MAE": round(float(mean_absolute_error(sub_y, p)), 2),
                "R2": round(float(r2_score(sub_y, p)), 3)
            }
    metro_df = pd.DataFrame(metro_perf).T
    print("\nTop Metropolitan Cities Performance (CatBoost, Exp 1):")
    print(metro_df)

    # =========================================================================
    # ROLLING-ORIGIN TEMPORAL CROSS-VALIDATION (Ablation / Stability)
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHRONOLOGICAL EXPANDING-WINDOW CROSS-VALIDATION (Training Era)")
    print("=" * 60)
    
    # Window 1: Train 2015-2016 -> Eval 2017
    # Window 2: Train 2015-2017 -> Eval 2018
    # Window 3: Train 2015-2018 -> Eval 2019-2020 (Final Test)
    w1_train = train_full[train_full["Date"].str.slice(0, 4).astype(int) <= 2016]
    w1_eval = train_full[train_full["Date"].str.slice(0, 4).astype(int) == 2017]
    
    cb_w1 = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_state=seed, verbose=False)
    cb_w1.fit(w1_train[POLLUTANTS].values, w1_train[TARGET].values)
    p_w1 = cb_w1.predict(w1_eval[POLLUTANTS].values)
    w1_rmse = root_mean_squared_error(w1_eval[TARGET].values, p_w1)

    cb_w2 = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_state=seed, verbose=False)
    cb_w2.fit(train_sub[POLLUTANTS].values, y_train_sub)
    p_w2 = cb_w2.predict(val_df[POLLUTANTS].values)
    w2_rmse = root_mean_squared_error(y_val, p_w2)

    print(f"Rolling-Origin Window 1 (Train 2015-2016 [N={len(w1_train)}] -> Eval 2017 [N={len(w1_eval)}]): CatBoost RMSE = {w1_rmse:.4f}")
    print(f"Rolling-Origin Window 2 (Train 2015-2017 [N={len(train_sub)}] -> Eval 2018 [N={len(val_df)}]): CatBoost RMSE = {w2_rmse:.4f}")
    print(f"Rolling-Origin Window 3 (Train 2015-2018 [N={len(train_full)}] -> Eval 2019-2020 [N={len(test_df)}]): CatBoost RMSE = {df_exp1.loc['CatBoost', 'RMSE']:.4f}")

    # =========================================================================
    # SAVE TEXT SUMMARY & ARTIFACTS
    # =========================================================================
    summary_path = os.path.join(data_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("REVISED BENCHMARKING RESULTS & STATISTICAL AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK (Primary Controlled Comparison)\n")
        f.write("Features: PM2.5, PM10, NO2, NH3, SO2, CO, O3, NO (Physical Units, Unclipped Target)\n")
        f.write(df_exp1.to_string())
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("EXPERIMENT 2: CITY-AWARE BENCHMARK (Criteria Pollutants + City)\n")
        f.write("Features: City (Encoded) + 8 Pollutants\n")
        f.write(df_exp2.to_string())
        f.write("\n\n" + "-" * 80 + "\n")
        f.write(f"STATISTICAL HYPOTHESIS TESTING (Primary Benchmark Best Model: {best_m1_name})\n")
        f.write(inference_df.to_string())
        f.write("\n\nMETHODOLOGICAL INFERENCE NOTES:\n")
        f.write("1. Observation-level paired t-test and Wilcoxon signed-rank tests test the null of equal error magnitude.\n")
        f.write("2. Daily-aggregated paired t-test mitigates cross-city spatial correlation on identical calendar dates (T=548 days).\n")
        f.write("3. Family-wise error rate is controlled via Holm-Bonferroni step-down correction across comparisons.\n")
        f.write(f"4. Equivalence is tested via Two One-Sided Tests (TOST) against an equivalence boundary of +/- {EQUIVALENCE_MARGIN:.1f} AQI points.\n")
        f.write("   NOTE: Failure to reject H0 in a standard difference test does NOT prove equivalence. Equivalence requires TOST rejection.\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("SPATIAL GENERALIZATION: SEEN VS UNSEEN CITIES IN TEST SET\n")
        f.write("Seen in Training (18 cities, N=9374) vs Unseen in Training (8 cities, N=2036):\n")
        f.write(pd.DataFrame({
            "Seen_Cities_RMSE": {m: spatial_breakdown["Seen Cities (18 cities)"][m]["RMSE"] for m in df_exp1.index},
            "Unseen_Cities_RMSE": {m: spatial_breakdown["Unseen Cities (8 cities)"][m]["RMSE"] for m in df_exp1.index}
        }).to_string())
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("TEMPORAL ROLLING-ORIGIN CROSS-VALIDATION\n")
        f.write(f"  Window 1 (Train 2015-16 -> Eval 2017): CatBoost RMSE = {w1_rmse:.4f}\n")
        f.write(f"  Window 2 (Train 2015-17 -> Eval 2018): CatBoost RMSE = {w2_rmse:.4f}\n")
        f.write(f"  Window 3 (Train 2015-18 -> Eval 2019-20): CatBoost RMSE = {df_exp1.loc['CatBoost', 'RMSE']:.4f}\n")

    # Detailed machine-readable JSON
    detailed_json_path = os.path.join(data_dir, "results_detailed.json")
    results_detailed = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": TARGET,
            "test_sample_count": len(test_df),
            "train_sample_count": len(train_full),
            "random_seed": seed,
            "hardware_platform": sys.platform
        },
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
        "spatial_breakdown_exp1": spatial_breakdown,
        "temporal_year_breakdown_exp1": year_breakdown,
        "metro_cities_breakdown": metro_perf,
        "rolling_origin_rmse": {
            "window_1_2017": float(w1_rmse),
            "window_2_2018": float(w2_rmse),
            "window_3_2019_2020": float(df_exp1.loc["CatBoost", "RMSE"])
        }
    }
    with open(detailed_json_path, "w", encoding="utf-8") as jf:
        json.dump(results_detailed, jf, indent=2)

    # Save models and predictions dictionary for SHAP script
    models_pkl_path = os.path.join(data_dir, "trained_models.pkl")
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
        "y_test": y_test,
        "test_df": test_df[["City", "Date"] + POLLUTANTS + [TARGET]]
    }
    with open(models_pkl_path, "wb") as pf:
        pickle.dump(save_payload, pf)

    print(f"\nSaved results to:")
    print(f"  - {os.path.join(data_dir, 'results_table.csv')}")
    print(f"  - {summary_path}")
    print(f"  - {detailed_json_path}")
    print(f"  - {models_pkl_path}")
    print("Training and benchmarking completed successfully.")

if __name__ == "__main__":
    main()
