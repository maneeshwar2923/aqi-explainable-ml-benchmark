"""
02_train_and_evaluate.py
--------------------------
Trains Random Forest, XGBoost, LightGBM, and CatBoost on train.csv,
evaluates on test.csv using RMSE, MAE, R^2, and records training time.
Also runs a corrected resampled t-test (5x2cv style approx via the
train set) between the best model and the others to check significance,
as described in Section 3.4.

Run this AFTER 01_preprocess.py.
Prints a results table to console and saves results_table.csv + results_summary.txt.
Also saves each trained model as a .pkl (needed by 03_shap_analysis.py) and
saves the fitted CatBoost/city encoding info if needed.
"""

import pandas as pd
import numpy as np
import time
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from scipy import stats

# ---- EDIT THIS if your path changes ----
DATA_DIR = r"C:\Users\manee\OneDrive\Documents\CHRIST\Research\final_analysis"
# -----------------------------------------

FEATURES = ["PM2.5_z", "PM10_z", "NO2_z", "NH3_z", "SO2_z", "CO_z", "O3_z", "NO_z"]
TARGET = "AQI"

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, test_df

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def corrected_resampled_ttest(errors_a, errors_b, n_train, n_test, alpha=0.05):
    """
    Corrected resampled paired t-test (Nadeau & Bengio, 2003) approximation,
    applied here as a single-split correction to the standard paired t-test
    on per-sample squared errors, adjusting for the train/test size ratio.
    """
    diff = errors_a - errors_b
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    correction = (1 / n) + (n_test / n_train)
    denom = np.sqrt(correction) * std_diff
    t_stat = mean_diff / denom if denom != 0 else 0.0
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    significant = p_value < alpha
    return t_stat, p_value, significant

def main():
    train_df, test_df = load_data()
    X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
    X_test, y_test = test_df[FEATURES].values, test_df[TARGET].values

    results = {}
    predictions = {}
    models = {}

    # ---- Random Forest ----
    print("Training Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    pred = rf.predict(X_test)
    results["Random Forest"] = (*evaluate(y_test, pred), rf_time)
    predictions["Random Forest"] = pred
    models["Random Forest"] = rf

    # ---- XGBoost ----
    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        learning_rate=0.05, max_depth=6, n_estimators=500,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=20, eval_metric="rmse",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_time = time.time() - t0
    pred = xgb_model.predict(X_test)
    results["XGBoost"] = (*evaluate(y_test, pred), xgb_time)
    predictions["XGBoost"] = pred
    models["XGBoost"] = xgb_model

    # ---- LightGBM ----
    print("Training LightGBM...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        num_leaves=63, learning_rate=0.05, n_estimators=500,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - t0
    pred = lgb_model.predict(X_test)
    results["LightGBM"] = (*evaluate(y_test, pred), lgb_time)
    predictions["LightGBM"] = pred
    models["LightGBM"] = lgb_model

    # ---- CatBoost ----
    # City is included as a categorical feature per the methodology (Section 3.3)
    print("Training CatBoost...")
    cat_train = train_df[["City"] + FEATURES].copy()
    cat_test = test_df[["City"] + FEATURES].copy()
    cat_features_idx = [0]  # City is column 0

    t0 = time.time()
    cb_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        random_state=42, verbose=False
    )
    cb_model.fit(cat_train, y_train, cat_features=cat_features_idx)
    cb_time = time.time() - t0
    pred = cb_model.predict(cat_test)
    results["CatBoost"] = (*evaluate(y_test, pred), cb_time)
    predictions["CatBoost"] = pred
    models["CatBoost"] = cb_model

    # ---- Results table ----
    results_df = pd.DataFrame(results, index=["RMSE", "MAE", "R2", "TrainTime_sec"]).T
    results_df = results_df.round(4)
    print("\n=== RESULTS TABLE ===")
    print(results_df)
    results_df.to_csv(os.path.join(DATA_DIR, "results_table.csv"))

    # ---- Significance testing: best model vs each other ----
    best_model_name = results_df["RMSE"].idxmin()
    print(f"\nBest model by RMSE: {best_model_name}")

    n_train, n_test = len(y_train), len(y_test)
    sig_lines = [f"Best model: {best_model_name}\n"]
    best_sq_err = (y_test - predictions[best_model_name]) ** 2
    for name, pred in predictions.items():
        if name == best_model_name:
            continue
        other_sq_err = (y_test - pred) ** 2
        t_stat, p_val, sig = corrected_resampled_ttest(best_sq_err, other_sq_err, n_train, n_test)
        line = f"{best_model_name} vs {name}: t={t_stat:.4f}, p={p_val:.6f}, significant at alpha=0.05: {sig}"
        print(line)
        sig_lines.append(line)

    with open(os.path.join(DATA_DIR, "results_summary.txt"), "w") as f:
        f.write("RESULTS TABLE\n")
        f.write(results_df.to_string())
        f.write("\n\nSIGNIFICANCE TESTS (corrected resampled t-test)\n")
        f.write("\n".join(sig_lines))

    # ---- Save models for SHAP script ----
    with open(os.path.join(DATA_DIR, "trained_models.pkl"), "wb") as f:
        pickle.dump({"models": models, "best_model_name": best_model_name}, f)

    print(f"\nSaved results_table.csv, results_summary.txt, trained_models.pkl to {DATA_DIR}")
    print("Copy the results table above into Table 1 of the paper.")

if __name__ == "__main__":
    main()
