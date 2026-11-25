"""
03_shap_analysis.py
---------------------
Publication-Ready Explainable AI (XAI) Suite via TreeSHAP:
  - Loads finalized models and untouched test observations from trained_models.pkl.
  - Experiment 1 (Primary): Evaluates top model (CatBoost) on physical pollutant features
    with real units (ug/m3, mg/m3).
  - Experiment 2: Evaluates City-aware model with native categorical handling.
  - Computes exact TreeSHAP attributions across all 11,410 held-out test observations.
  - Generates:
      * Global Mean |SHAP| feature ranking table (shap_feature_ranking.csv)
      * Publication-grade Beeswarm summary plot (shap_beeswarm.png)
      * Non-linear Partial Dependence plot with physical units (shap_dependence_top_feature.png)
      * Experiment 2 rankings and beeswarm (shap_feature_ranking_exp2.csv, shap_beeswarm_exp2.png)
  - Scientific disclaimer: SHAP values quantify algorithmic model attribution,
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
    "City":  "City (Metropolitan Hub)"
}

def parse_args():
    parser = argparse.ArgumentParser(description="TreeSHAP Explainability Suite for Urban AQI")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing trained_models.pkl. Defaults to script directory.")
    return parser.parse_args()

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir if args.data_dir else script_dir

    models_pkl = os.path.join(data_dir, "trained_models.pkl")
    if not os.path.exists(models_pkl):
        raise FileNotFoundError(f"Missing trained models file at: {models_pkl}")

    with open(models_pkl, "rb") as f:
        data_bundle = pickle.load(f)

    exp1_bundle = data_bundle["experiment_1"]
    exp2_bundle = data_bundle["experiment_2"]
    test_df = data_bundle["test_df"]
    y_test = data_bundle["y_test"]

    print("=" * 70)
    print("SHAP INTERPRETABILITY SUITE (TreeSHAP on Final Test Partition)")
    print("=" * 70)
    print(f"Total test instances evaluated: {len(test_df)}")

    # =========================================================================
    # EXPERIMENT 1: POLLUTANT-ONLY BENCHMARK SHAP ANALYSIS
    # =========================================================================
    best_m1_name = exp1_bundle["best_model_name"]
    best_m1_model = exp1_bundle["models"][best_m1_name]
    features_exp1 = exp1_bundle["features"]

    print(f"\n[Experiment 1] Computing TreeSHAP for primary model: {best_m1_name}")
    X_test_exp1 = test_df[features_exp1].copy()

    explainer_exp1 = shap.TreeExplainer(best_m1_model)
    shap_values_exp1 = explainer_exp1.shap_values(X_test_exp1)

    # Validate array dimensions
    assert shap_values_exp1.shape == X_test_exp1.shape, (
        f"SHAP shape mismatch: {shap_values_exp1.shape} vs {X_test_exp1.shape}"
    )

    # Compute Global Mean Absolute SHAP Ranking
    mean_abs_shap1 = np.abs(shap_values_exp1).mean(axis=0)
    total_attribution1 = np.sum(mean_abs_shap1)
    rel_attribution1 = (mean_abs_shap1 / total_attribution1) * 100.0

    ranking_exp1 = pd.DataFrame({
        "Feature": features_exp1,
        "Feature_Label": [FEATURE_UNITS.get(f, f) for f in features_exp1],
        "MeanAbsSHAP": mean_abs_shap1,
        "RelativeAttributionPct": rel_attribution1
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    print("\n--- EXPERIMENT 1 FEATURE ATTRIBUTION RANKING ---")
    print(ranking_exp1[["Feature", "MeanAbsSHAP", "RelativeAttributionPct"]].to_string())

    # Save ranking CSVs
    ranking_csv_main = os.path.join(data_dir, "shap_feature_ranking.csv")
    ranking_csv_exp1 = os.path.join(data_dir, "shap_feature_ranking_exp1.csv")
    ranking_exp1.to_csv(ranking_csv_main, index=False)
    ranking_exp1.to_csv(ranking_csv_exp1, index=False)
    print(f"Saved: {ranking_csv_main} and {ranking_csv_exp1}")

    # Publication-Quality Beeswarm Plot for Exp 1
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "figure.titlesize": 14
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    display_names_exp1 = [FEATURE_UNITS.get(f, f) for f in features_exp1]
    shap.summary_plot(
        shap_values_exp1,
        X_test_exp1,
        feature_names=display_names_exp1,
        show=False,
        plot_size=None,
        color_bar=True
    )
    plt.title(f"TreeSHAP Attribution Distribution — {best_m1_name} (Pollutant-Only Benchmark)\n"
              f"Evaluated on N=11,410 Chronological Test Observations (2019–2020)",
              fontsize=12, pad=15, weight="bold")
    plt.xlabel("SHAP value (Impact on AQI Output, AQI points)", fontsize=11, labelpad=8)
    plt.tight_layout()
    beeswarm_path_main = os.path.join(data_dir, "shap_beeswarm.png")
    beeswarm_path_exp1 = os.path.join(data_dir, "shap_beeswarm_exp1.png")
    plt.savefig(beeswarm_path_main, dpi=300, bbox_inches="tight")
    plt.savefig(beeswarm_path_exp1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication beeswarm: {beeswarm_path_main}")

    # Dependence Plot for Top Feature (e.g. PM2.5)
    top_feature = ranking_exp1.iloc[0]["Feature"]
    top_feature_idx = features_exp1.index(top_feature)
    second_feature = ranking_exp1.iloc[1]["Feature"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    shap.dependence_plot(
        top_feature_idx,
        shap_values_exp1,
        X_test_exp1.values,
        feature_names=[FEATURE_UNITS.get(f, f) for f in features_exp1],
        interaction_index=features_exp1.index(second_feature),
        show=False,
        ax=ax
    )
    ax.set_title(f"SHAP Dependence Plot: {FEATURE_UNITS.get(top_feature, top_feature)}\n"
                 f"Colored by {FEATURE_UNITS.get(second_feature, second_feature)} Interaction",
                 fontsize=12, weight="bold", pad=12)
    ax.set_ylabel(f"SHAP value for {FEATURE_UNITS.get(top_feature, top_feature)} (AQI Points)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    dep_path = os.path.join(data_dir, "shap_dependence_top_feature.png")
    plt.savefig(dep_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dependence plot: {dep_path}")

    # =========================================================================
    # EXPERIMENT 2: CITY-AWARE BENCHMARK SHAP ANALYSIS
    # =========================================================================
    best_m2_name = exp2_bundle["best_model_name"]
    best_m2_model = exp2_bundle["models"][best_m2_name]
    features_exp2 = exp2_bundle["features"]

    print(f"\n[Experiment 2] Computing TreeSHAP for City-aware model: {best_m2_name}")
    cat_test = test_df[["City"] + POLLUTANTS].copy()

    explainer_exp2 = shap.TreeExplainer(best_m2_model)
    shap_values_exp2 = explainer_exp2.shap_values(cat_test)

    mean_abs_shap2 = np.abs(shap_values_exp2).mean(axis=0)
    total_attribution2 = np.sum(mean_abs_shap2)
    rel_attribution2 = (mean_abs_shap2 / total_attribution2) * 100.0

    ranking_exp2 = pd.DataFrame({
        "Feature": features_exp2,
        "Feature_Label": [FEATURE_UNITS.get(f, f) for f in features_exp2],
        "MeanAbsSHAP": mean_abs_shap2,
        "RelativeAttributionPct": rel_attribution2
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    print("\n--- EXPERIMENT 2 FEATURE ATTRIBUTION RANKING ---")
    print(ranking_exp2[["Feature", "MeanAbsSHAP", "RelativeAttributionPct"]].to_string())

    ranking_csv_exp2 = os.path.join(data_dir, "shap_feature_ranking_exp2.csv")
    ranking_exp2.to_csv(ranking_csv_exp2, index=False)
    print(f"Saved: {ranking_csv_exp2}")

    # Beeswarm for Exp 2
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    display_names_exp2 = [FEATURE_UNITS.get(f, f) for f in features_exp2]
    shap.summary_plot(
        shap_values_exp2,
        cat_test,
        feature_names=display_names_exp2,
        show=False,
        plot_size=None,
        color_bar=True
    )
    plt.title(f"TreeSHAP Attribution Distribution — {best_m2_name} (City-Aware Benchmark)\n"
              f"Evaluated on N=11,410 Chronological Test Observations (2019–2020)",
              fontsize=12, pad=15, weight="bold")
    plt.xlabel("SHAP value (Impact on AQI Output, AQI points)", fontsize=11, labelpad=8)
    plt.tight_layout()
    beeswarm_path_exp2 = os.path.join(data_dir, "shap_beeswarm_exp2.png")
    plt.savefig(beeswarm_path_exp2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {beeswarm_path_exp2}")

    print("\n" + "=" * 70)
    print("SCIENTIFIC INTERPRETABILITY DISCLAIMER:")
    print("SHAP values quantify statistical feature attribution within the fitted model;")
    print("they represent algorithmic marginal contributions to predicted AQI scores,")
    print("NOT physical causal mechanisms or verified public-policy intervention impacts.")
    print("=" * 70)

if __name__ == "__main__":
    main()
