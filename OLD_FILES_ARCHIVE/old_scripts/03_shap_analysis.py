"""
03_shap_analysis.py
---------------------
Loads the best model from trained_models.pkl, runs SHAP TreeExplainer
on the test set, and produces:
  - a beeswarm summary plot (saved as shap_beeswarm.png)
  - a dependence plot for the top-ranked feature (saved as shap_dependence_top_feature.png)
  - a printed ranking of mean |SHAP value| per feature (saved as shap_feature_ranking.csv)

Run this AFTER 02_train_and_evaluate.py.

Note: if CatBoost turns out to be the best model, this script uses its
Pool-based test features so the "City" categorical column is handled
correctly by TreeExplainer.
"""

import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt

# ---- EDIT THIS if your path changes ----
DATA_DIR = r"C:\Users\manee\OneDrive\Documents\CHRIST\Research\final_analysis"
# -----------------------------------------

FEATURES = ["PM2.5_z", "PM10_z", "NO2_z", "NH3_z", "SO2_z", "CO_z", "O3_z", "NO_z"]

def main():
    with open(os.path.join(DATA_DIR, "trained_models.pkl"), "rb") as f:
        saved = pickle.load(f)
    models = saved["models"]
    best_name = saved["best_model_name"]
    best_model = models[best_name]
    print(f"Running SHAP on best model: {best_name}")

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    if best_name == "CatBoost":
        X_test = test_df[["City"] + FEATURES]
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        feature_names = ["City"] + FEATURES
    else:
        X_test = test_df[FEATURES]
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        feature_names = FEATURES

    # Feature ranking by mean |SHAP value|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    print("\nFeature ranking (mean |SHAP value|):")
    print(ranking)
    ranking.to_csv(os.path.join(DATA_DIR, "shap_feature_ranking.csv"), index=False)

    # Beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "shap_beeswarm.png"), dpi=200)
    plt.close()
    print(f"Saved shap_beeswarm.png")

    # Dependence plot for top feature
    top_feature = ranking.iloc[0]["Feature"]
    if top_feature != "City":  # dependence plots need numeric features
        plt.figure()
        shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "shap_dependence_top_feature.png"), dpi=200)
        plt.close()
        print(f"Saved shap_dependence_top_feature.png for feature: {top_feature}")
    else:
        # fall back to the top numeric feature if City ranks first
        top_numeric = ranking[ranking["Feature"] != "City"].iloc[0]["Feature"]
        plt.figure()
        shap.dependence_plot(top_numeric, shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "shap_dependence_top_feature.png"), dpi=200)
        plt.close()
        print(f"City ranked highest; used top numeric feature instead: {top_numeric}")

    print(f"\nAll SHAP outputs saved to {DATA_DIR}")

if __name__ == "__main__":
    main()

