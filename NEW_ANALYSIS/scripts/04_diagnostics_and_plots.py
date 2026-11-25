"""
NEW_ANALYSIS/scripts/04_diagnostics_and_plots.py
------------------------------------------------
Comprehensive Predictive Diagnostics & Residual Analysis for Urban AQI Modeling:
  - Ingests out-of-sample predictions from NEW_ANALYSIS/predictions/predictions_pollutant_only.csv
  - Generates:
      1. Observed vs. Predicted Scatter with 1:1 ideal line and R2 fit (observed_vs_predicted_scatter.png)
      2. Residual Distribution Histogram & KDE vs. Gaussian fit (residual_distribution_histogram.png)
      3. Residuals vs. Predicted Values Scatter for heteroscedasticity inspection (residuals_vs_predicted.png)
      4. Error Performance Decomposition across CPCB AQI Severity Bands (error_by_aqi_category.png)
      5. Year-over-Year Performance Comparison: 2019 baseline vs. 2020 lockdown (error_by_year.png)
      6. Diagnostic Error Summary CSV (error_by_aqi_category.csv)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Diagnostics and Error Analysis")
    parser.add_argument("--predictions-file", type=str, default=None,
                        help="Path to predictions_pollutant_only.csv.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save diagnostic plots. Defaults to NEW_ANALYSIS/diagnostics.")
    return parser.parse_args()

def assign_aqi_category(aqi):
    if aqi <= 50:
        return "Good (0-50)"
    elif aqi <= 100:
        return "Satisfactory (51-100)"
    elif aqi <= 200:
        return "Moderate (101-200)"
    elif aqi <= 300:
        return "Poor (201-300)"
    elif aqi <= 400:
        return "Very Poor (301-400)"
    elif aqi <= 500:
        return "Severe (401-500)"
    else:
        return "Extreme Hazardous (>500)"

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_root = os.path.abspath(os.path.join(script_dir, ".."))

    preds_file = args.predictions_file if args.predictions_file else os.path.join(analysis_root, "predictions", "predictions_pollutant_only.csv")
    out_dir = args.output_dir if args.output_dir else os.path.join(analysis_root, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(preds_file):
        raise FileNotFoundError(f"Missing predictions file: {preds_file}")

    df = pd.read_csv(preds_file, parse_dates=["Date"])
    y_true = df["y_true"].values
    y_pred = df["CatBoost"].values  # Primary model
    residuals = y_true - y_pred

    print("=" * 75)
    print("PREDICTIVE DIAGNOSTICS & RESIDUAL FORENSICS")
    print("=" * 75)
    print(f"Loaded predictions: {len(df)} test instances from {preds_file}")

    # Residual summary statistics
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    res_skew = stats.skew(residuals)
    res_kurt = stats.kurtosis(residuals)

    print(f"Residual Mean:     {res_mean:.4f}")
    print(f"Residual Std Dev:  {res_std:.4f}")
    print(f"Residual Skewness: {res_skew:.4f}")
    print(f"Residual Kurtosis: {res_kurt:.4f}")

    # =========================================================================
    # 1. OBSERVED VS PREDICTED SCATTER PLOT
    # =========================================================================
    plt.figure(figsize=(7.5, 7.5), dpi=300)
    plt.scatter(y_pred, y_true, alpha=0.25, color="#1f77b4", edgecolors="none", s=18, label="Test Instances (N=11,410)")
    max_val = max(np.max(y_true), np.max(y_pred)) + 20
    plt.plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal 1:1 Identity Line")
    
    # Linear fit line
    slope, intercept, r_val, p_val, std_err = stats.linregress(y_pred, y_true)
    x_vals = np.linspace(0, max_val, 100)
    plt.plot(x_vals, slope * x_vals + intercept, color="#2ca02c", linewidth=1.8,
             label=f"Fitted Trend ($R^2={r_val**2:.3f}$, Slope={slope:.2f})")

    plt.title("Observed vs. Predicted AQI (CatBoost Primary Model, Out-of-Time Test)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted AQI (Score)", fontsize=11)
    plt.ylabel("Observed True AQI (Score)", fontsize=11)
    plt.xlim(0, min(max_val, 900))
    plt.ylim(0, min(max_val, 900))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, loc="upper left", fontsize=10)
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "observed_vs_predicted_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved observed vs. predicted scatter to: {scatter_path}")

    # =========================================================================
    # 2. RESIDUAL DISTRIBUTION HISTOGRAM & NORMALITY CHECK
    # =========================================================================
    plt.figure(figsize=(8, 5.5), dpi=300)
    n_bins = 70
    counts, bins, patches = plt.hist(residuals, bins=n_bins, density=True, alpha=0.6, color="#4a7bb0", edgecolor="white", label="Empirical Residuals ($y - \hat{y}$)")
    
    # Gaussian overlay
    x_range = np.linspace(np.percentile(residuals, 0.5), np.percentile(residuals, 99.5), 200)
    gaussian_pdf = stats.norm.pdf(x_range, loc=res_mean, scale=res_std)
    plt.plot(x_range, gaussian_pdf, color="#c0392b", linewidth=2.2, label=f"Gaussian Fit ($\mu={res_mean:.2f}, \sigma={res_std:.2f}$)")

    plt.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.title(f"Prediction Error Distribution (Skewness = {res_skew:.2f}, Kurtosis = {res_kurt:.2f})", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Prediction Error / Residual (AQI Points)", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.xlim(np.percentile(residuals, 0.2), np.percentile(residuals, 99.8))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, loc="upper right", fontsize=10)
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "residual_distribution_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual distribution histogram to: {hist_path}")

    # =========================================================================
    # 3. RESIDUALS VS PREDICTED VALUES (HETEROSCEDASTICITY INSPECTION)
    # =========================================================================
    plt.figure(figsize=(8.5, 5.5), dpi=300)
    plt.scatter(y_pred, residuals, alpha=0.25, color="#2e7d32", edgecolors="none", s=18)
    plt.axhline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Zero Error Line")
    
    # Moving percentiles (25th, 50th, 75th)
    pred_bins = pd.qcut(y_pred, q=10, duplicates="drop")
    bin_medians = df.groupby(pred_bins, observed=False)["CatBoost"].median()
    bin_res_p25 = df.groupby(pred_bins, observed=False).apply(lambda g: np.percentile(g["y_true"] - g["CatBoost"], 25))
    bin_res_p75 = df.groupby(pred_bins, observed=False).apply(lambda g: np.percentile(g["y_true"] - g["CatBoost"], 75))
    
    plt.plot(bin_medians, bin_res_p25, color="#e67e22", linestyle=":", linewidth=2, label="25th & 75th Percentile Error Envelope")
    plt.plot(bin_medians, bin_res_p75, color="#e67e22", linestyle=":", linewidth=2)

    plt.title("Residuals vs. Predicted AQI (Heteroscedasticity Analysis)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Predicted AQI (Score)", fontsize=11)
    plt.ylabel("Residual (Observed - Predicted, Points)", fontsize=11)
    plt.ylim(-200, 350)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, loc="upper left", fontsize=10)
    plt.tight_layout()
    res_pred_path = os.path.join(out_dir, "residuals_vs_predicted.png")
    plt.savefig(res_pred_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residuals vs. predicted plot to: {res_pred_path}")

    # =========================================================================
    # 4. ERROR PERFORMANCE ACROSS CPCB AQI CATEGORIES
    # =========================================================================
    df["AQI_Category"] = df["y_true"].apply(assign_aqi_category)
    category_order = [
        "Good (0-50)", "Satisfactory (51-100)", "Moderate (101-200)",
        "Poor (201-300)", "Very Poor (301-400)", "Severe (401-500)",
        "Extreme Hazardous (>500)"
    ]

    cat_records = []
    for cat in category_order:
        sub = df[df["AQI_Category"] == cat]
        if len(sub) == 0:
            continue
        err_cat = sub["y_true"] - sub["CatBoost"]
        rmse_cat = np.sqrt(np.mean(err_cat ** 2))
        mae_cat = np.mean(np.abs(err_cat))
        cat_records.append({
            "AQI_Category": cat,
            "Sample_Count": len(sub),
            "Percentage": round(len(sub) / len(df) * 100, 2),
            "Mean_Observed": round(float(sub["y_true"].mean()), 1),
            "Mean_Predicted": round(float(sub["CatBoost"].mean()), 1),
            "RMSE": round(float(rmse_cat), 2),
            "MAE": round(float(mae_cat), 2)
        })

    cat_df = pd.DataFrame(cat_records)
    cat_csv_path = os.path.join(out_dir, "error_by_aqi_category.csv")
    cat_df.to_csv(cat_csv_path, index=False)
    print(f"Saved category breakdown CSV to: {cat_csv_path}")
    print("\n--- ERROR BREAKDOWN BY CPCB AQI CATEGORY ---")
    print(cat_df.to_string(index=False))

    # Bar chart of MAE and RMSE across categories
    plt.figure(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(cat_df))
    width = 0.35
    plt.bar(x - width/2, cat_df["MAE"], width, label="MAE", color="#3498db")
    plt.bar(x + width/2, cat_df["RMSE"], width, label="RMSE", color="#e74c3c")
    plt.xticks(x, [c.split(" ")[0] for c in cat_df["AQI_Category"]], fontsize=10)
    plt.title("CatBoost Error Metrics Across Official CPCB AQI Severity Bands", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("CPCB Health Advisory Category", fontsize=11)
    plt.ylabel("Error Magnitude (AQI Points)", fontsize=11)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    cat_bar_path = os.path.join(out_dir, "error_by_aqi_category.png")
    plt.savefig(cat_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved category error bar plot to: {cat_bar_path}")

    # =========================================================================
    # 5. YEAR-OVER-YEAR COMPARATIVE ERROR (2019 VS 2020)
    # =========================================================================
    df["Year"] = df["Date"].dt.year
    models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    
    y2019 = df[df["Year"] == 2019]
    y2020 = df[df["Year"] == 2020]

    rmse_2019 = [np.sqrt(np.mean((y2019["y_true"] - y2019[m])**2)) for m in models]
    rmse_2020 = [np.sqrt(np.mean((y2020["y_true"] - y2020[m])**2)) for m in models]

    plt.figure(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, rmse_2019, width, label="2019 Baseline (N=7,031)", color="#5c6bc0")
    plt.bar(x + width/2, rmse_2020, width, label="2020 Lockdown Era (N=4,379)", color="#26a69a")
    plt.xticks(x, models, fontsize=10)
    plt.title("Model Generalization Across Atmospheric Regimes (2019 vs. 2020)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Machine Learning Algorithm", fontsize=11)
    plt.ylabel("Holdout RMSE (AQI Points)", fontsize=11)
    plt.ylim(0, 58)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    year_bar_path = os.path.join(out_dir, "error_by_year.png")
    plt.savefig(year_bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved year-wise comparison plot to: {year_bar_path}")

    print("\nDiagnostic plots and analyses completed successfully.")

if __name__ == "__main__":
    main()
