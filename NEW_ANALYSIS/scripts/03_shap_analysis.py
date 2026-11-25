"""
NEW_ANALYSIS/scripts/03_shap_analysis.py
----------------------------------------
TreeSHAP Interpretability Suite for Urban AQI Estimation:
  - Ingests finalized models from NEW_ANALYSIS/models/trained_models.pkl
  - Computes TreeSHAP for primary model (CatBoost) on held-out test partition (N=11,410)
  - Experiment 1 (Pollutant-Only): 8 physical criteria pollutants with real units (ug/m3, mg/m3)
  - Experiment 2 (City-Aware): 8 criteria pollutants + City categorical encoding
  - Outputs:
      * NEW_ANALYSIS/shap_outputs/shap_feature_ranking_pollutant_only.csv
      * NEW_ANALYSIS/shap_outputs/shap_feature_ranking_city_aware.csv
      * NEW_ANALYSIS/shap_outputs/shap_beeswarm_pollutant_only.png
      * NEW_ANALYSIS/shap_outputs/shap_beeswarm_city_aware.png
      * NEW_ANALYSIS/shap_outputs/shap_dependence_pm25.png
  - Scientific Framing: SHAP quantifies statistical model feature attribution,
    NOT physical causality or intervention efficacy.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]

FEATURE_UNITS = {
    "PM2.5": "PM2.5 (ug/m3)",
    "PM10":  "PM10 (ug/m3)",
    "NO2":   "NO2 (ug/m3)",
    "NH3":   "NH3 (ug/m3)",
    "SO2":   "SO2 (ug/m3)",
    "CO":    "CO (mg/m3)",
    "O3":    "O3 (ug/m3)",
    "NO":    "NO (ug/m3)",
    "City":  "City Identity"
}

def parse_args():
    parser = argparse.ArgumentParser(description="TreeSHAP Explainability Suite for Urban AQI")
    parser.add_argument("--models-file", type=str, default=None,
                        help="Path to trained_models.pkl. Defaults to NEW_ANALYSIS/models/trained_models.pkl.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save SHAP outputs. Defaults to NEW_ANALYSIS/shap_outputs.")
    return parser.parse_args()

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_root = os.path.abspath(os.path.join(script_dir, ".."))

    models_file = args.models_file if args.models_file else os.path.join(analysis_root, "models", "trained_models.pkl")
    out_dir = args.output_dir if args.output_dir else os.path.join(analysis_root, "shap_outputs")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(models_file):
        raise FileNotFoundError(f"Missing models bundle file at: {models_file}")

    with open(models_file, "rb") as f:
        bundle = pickle.load(f)

    test_df = bundle["test_df"]
    y_test = bundle["y_test"]
    exp1_bundle = bundle["experiment_1"]
    exp2_bundle = bundle["experiment_2"]

    print("=" * 75)
    print("TREESHAP EXPLAINABLE AI (XAI) SUITE FOR URBAN AQI ESTIMATION")
    print("=" * 75)
    print(f"Loaded model bundle: {models_file}")
    print(f"Held-out test instances evaluated: {len(test_df)}")

    # =========================================================================
    # EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK SHAP
    # =========================================================================
    best_m1_name = exp1_bundle["best_model_name"]
    best_m1_model = exp1_bundle["models"][best_m1_name]
    features_exp1 = exp1_bundle["features"]

    print(f"\n[Experiment 1] Computing TreeSHAP for primary model: {best_m1_name}")
    X_test_exp1 = test_df[features_exp1].copy()

    explainer_exp1 = shap.TreeExplainer(best_m1_model)
    shap_values_exp1 = explainer_exp1.shap_values(X_test_exp1)

    assert shap_values_exp1.shape == X_test_exp1.shape, (
        f"SHAP shape mismatch: {shap_values_exp1.shape} vs {X_test_exp1.shape}"
    )

    # Compute Feature Importance Ranking
    mean_abs_shap1 = np.abs(shap_values_exp1).mean(axis=0)
    total_attribution1 = np.sum(mean_abs_shap1)
    rel_attribution1 = (mean_abs_shap1 / total_attribution1) * 100.0

    ranking_exp1 = pd.DataFrame({
        "Feature": features_exp1,
        "Feature_Label": [FEATURE_UNITS.get(f, f) for f in features_exp1],
        "MeanAbsSHAP": np.round(mean_abs_shap1, 4),
        "RelativeAttributionPct": np.round(rel_attribution1, 2)
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    csv_exp1_path = os.path.join(out_dir, "shap_feature_ranking_pollutant_only.csv")
    ranking_exp1.to_csv(csv_exp1_path, index=False)
    print(f"Saved Exp 1 feature ranking to: {csv_exp1_path}")
    print("\n--- EXPERIMENT 1 SHAP FEATURE RANKING ---")
    print(ranking_exp1.to_string())

    # Publication-Grade Beeswarm Summary Plot (Exp 1)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.title("TreeSHAP Global Summary (Experiment 1: Pollutant-Only CatBoost)", fontsize=13, fontweight="bold", pad=12)
    # Prepare display labels with units
    X_display_exp1 = X_test_exp1.rename(columns=FEATURE_UNITS)
    shap.summary_plot(shap_values_exp1, X_display_exp1, show=False, plot_size=(10, 6))
    plt.xlabel("SHAP Value (Impact on AQI Prediction, Points)", fontsize=11, labelpad=8)
    plt.tight_layout()
    beeswarm_exp1_path = os.path.join(out_dir, "shap_beeswarm_pollutant_only.png")
    plt.savefig(beeswarm_exp1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Exp 1 Beeswarm plot to: {beeswarm_exp1_path}")

    # Top Feature Dependence Plot (PM2.5)
    pm25_idx = features_exp1.index("PM2.5")
    plt.figure(figsize=(8, 5.5), dpi=300)
    shap.dependence_plot(
        "PM2.5", shap_values_exp1, X_test_exp1,
        interaction_index="CO",
        show=False,
        x_jitter=0.1
    )
    plt.title("TreeSHAP Partial Dependence: PM2.5 vs. AQI Impact (Colored by CO)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("PM2.5 Concentration (ug/m3)", fontsize=10)
    plt.ylabel("SHAP Value for PM2.5 (AQI Points)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    dependence_path = os.path.join(out_dir, "shap_dependence_pm25.png")
    plt.savefig(dependence_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PM2.5 Dependence plot to: {dependence_path}")

    # =========================================================================
    # EXPERIMENT 2: CITY-AWARE BENCHMARK SHAP
    # =========================================================================
    best_m2_name = exp2_bundle["best_model_name"]
    best_m2_model = exp2_bundle["models"][best_m2_name]
    features_exp2 = exp2_bundle["features"]

    print(f"\n[Experiment 2] Computing TreeSHAP for City-Aware model: {best_m2_name}")
    X_test_exp2 = test_df[features_exp2].copy()

    explainer_exp2 = shap.TreeExplainer(best_m2_model)
    shap_values_exp2 = explainer_exp2.shap_values(X_test_exp2)

    mean_abs_shap2 = np.abs(shap_values_exp2).mean(axis=0)
    total_attribution2 = np.sum(mean_abs_shap2)
    rel_attribution2 = (mean_abs_shap2 / total_attribution2) * 100.0

    ranking_exp2 = pd.DataFrame({
        "Feature": features_exp2,
        "Feature_Label": [FEATURE_UNITS.get(f, f) for f in features_exp2],
        "MeanAbsSHAP": np.round(mean_abs_shap2, 4),
        "RelativeAttributionPct": np.round(rel_attribution2, 2)
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    csv_exp2_path = os.path.join(out_dir, "shap_feature_ranking_city_aware.csv")
    ranking_exp2.to_csv(csv_exp2_path, index=False)
    print(f"Saved Exp 2 feature ranking to: {csv_exp2_path}")
    print("\n--- EXPERIMENT 2 SHAP FEATURE RANKING ---")
    print(ranking_exp2.to_string())

    # Beeswarm Plot for Exp 2 (City-Aware)
    plt.figure(figsize=(10, 6.5), dpi=300)
    plt.title("TreeSHAP Global Summary (Experiment 2: City-Aware CatBoost)", fontsize=13, fontweight="bold", pad=12)
    X_display_exp2 = X_test_exp2.rename(columns=FEATURE_UNITS)
    # Convert category to numeric codes for plotting display if needed
    if "City Identity" in X_display_exp2.columns:
        X_display_exp2["City Identity"] = pd.factorize(X_display_exp2["City Identity"])[0]
    shap.summary_plot(shap_values_exp2, X_display_exp2, show=False, plot_size=(10, 6.5))
    plt.xlabel("SHAP Value (Impact on AQI Prediction, Points)", fontsize=11, labelpad=8)
    plt.tight_layout()
    beeswarm_exp2_path = os.path.join(out_dir, "shap_beeswarm_city_aware.png")
    plt.savefig(beeswarm_exp2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Exp 2 Beeswarm plot to: {beeswarm_exp2_path}")

    print("\nSHAP interpretability analysis completed successfully.")

if __name__ == "__main__":
    main()
