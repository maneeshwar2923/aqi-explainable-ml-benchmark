# Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study

**Authors:** Research Team  
**Dataset:** Central Pollution Control Board (CPCB), India (2015–2020)  
**Deliverable Version:** Final Validated Release (October 2026)  
**Status:** Audit Verified & Submission Ready (11/11 QA Assertions Passed)

---

## 1. Executive Summary

This deliverable provides the complete, self-contained, and reproducible research package for the benchmarking study on same-day Air Quality Index (AQI) prediction across 26 Indian metropolitan and industrial cities. 

We systematically benchmark four gradient boosted and ensemble architectures—**Random Forest, XGBoost, LightGBM, and CatBoost**—across two strictly isolated paradigms:
1. **Experiment 1 (Pollutant-Only Benchmark):** Evaluates intrinsic physical mappings from eight criteria pollutants ($PM_{2.5}, PM_{10}, NO_2, NH_3, SO_2, CO, O_3, NO$) to AQI without geographic shortcuts.
2. **Experiment 2 (City-Aware Benchmark):** Incorporates one-hot encoded city identity to evaluate localized spatial bias and calibration gains.

### Primary Benchmark Results (Holdout Test Set: 2019–2020, $N = 11,410$)

| Experiment | Model | RMSE $\downarrow$ | MAE $\downarrow$ | $R^2$ $\uparrow$ | Training Time | Primary Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Pollutant-Only** | **CatBoost** | **39.6688** | **19.5689** | **0.8915** | 2.15s | **Best Accuracy** (Lowest RMSE & MAE) |
| Exp 1: Pollutant-Only | LightGBM | 40.9889 | 20.2186 | 0.8842 | 1.02s | Pareto Efficient ($2.1\times$ faster than CatBoost) |
| Exp 1: Pollutant-Only | Random Forest | 42.2185 | 20.6647 | 0.8771 | 3.33s | Competitive ensemble baseline |
| Exp 1: Pollutant-Only | XGBoost | 42.5595 | 20.4114 | 0.8751 | **0.47s** | **Fastest Training** ($4.6\times$ faster than CatBoost) |
| *Baseline (Exp 1)* | Global Train Mean | 130.3831 | 102.6559 | -0.1720 | — | Naive baseline |
| **Exp 2: City-Aware** | **CatBoost** | **39.0882** | **19.7629** | **0.8947** | 22.12s | Marginal accuracy gain ($\Delta\text{RMSE} = -0.58$) |
| Exp 2: City-Aware | LightGBM | 40.5550 | 20.1246 | 0.8866 | 1.00s | Robust to geographic encodings |
| Exp 2: City-Aware | Random Forest | 41.7839 | 20.6991 | 0.8796 | 3.46s | Minimal geographic sensitivity |
| Exp 2: City-Aware | XGBoost | 42.0344 | 19.9653 | 0.8782 | **0.40s** | Highly scalable with geographic indicators |
| *Baseline (Exp 2)* | City-Stratified Mean | 107.2215 | 78.4335 | 0.2074 | — | Spatial climatological baseline |

---

## 2. Directory Structure

```text
FINAL_DELIVERABLE/
├── README.md                           # This document: overview, run instructions, and audit summary
├── requirements.txt                    # Pinned Python dependencies for exact reproducibility
├── PROJECT_FILE_MANIFEST.csv           # Exhaustive provenance catalog of retained and excluded files
├── FINAL_RESULTS.csv                   # Verified test metrics across all models and experiments
├── FINAL_STATISTICAL_COMPARISON.csv    # Paired error tests, Wilcoxon tests, and bootstrap 95% CIs
├── FINAL_SHAP_IMPORTANCE.csv           # Mean absolute TreeSHAP attributions and percentages
├── FINAL_ANALYSIS_REPORT.md            # Comprehensive scientific report detailing all methodologies
├── DATA_DICTIONARY.md                  # Complete feature specifications, units, and missingness rules
├── FINAL_QA_REPORT.md                  # Quality assurance audit record (11/11 checks passed)
│
├── data/
│   ├── raw/
│   │   └── city_day.csv                # Official CPCB raw dataset (2015–2020, 26 cities, N=29,531)
│   └── processed/
│       ├── train_sub.csv               # 2015–2017 training split (N=7,556)
│       ├── val.csv                     # 2018 validation split (N=5,658) for early stopping
│       ├── train.csv                   # 2015–2018 full training partition (N=13,214)
│       └── test.csv                    # 2019–2020 out-of-time test partition (N=11,410)
│
├── scripts/
│   ├── 01_preprocess.py                # Chronological split, train-only imputation, uncapped target
│   ├── 02_train_and_evaluate.py        # Model benchmarking, validation early stopping, TOST tests
│   ├── 03_shap_analysis.py             # Physical-unit TreeSHAP attributions, rankings, and plots
│   ├── 04_diagnostics_and_plots.py     # Publication figures (scatter, residuals, heteroscedasticity)
│   └── test_pipeline_qa.py             # Automated 11-point QA test suite
│
├── predictions/
│   ├── predictions_pollutant_only.csv  # Test set observed vs model predictions (Exp 1)
│   └── predictions_city_aware.csv      # Test set observed vs model predictions (Exp 2)
│
├── results/
│   ├── FINAL_RESULTS.csv               # Primary metrics table (RMSE, MAE, R², Train Time)
│   ├── FINAL_STATISTICAL_COMPARISON.csv# Formal statistical test results
│   ├── FINAL_SHAP_IMPORTANCE.csv       # TreeSHAP feature importance table
│   ├── baseline_comparison.csv         # Benchmark vs naive mean baselines
│   ├── city_wise_performance.csv       # Detailed error breakdown across all 26 cities
│   ├── year_wise_performance.csv       # 2019 vs 2020 error stability analysis
│   ├── spatial_generalization.csv      # Seen (18) vs unseen (8) city generalization
│   ├── rolling_origin_cv.csv           # Chronological expanding-window cross-validation
│   └── tost_equivalence_exp1.csv      # Two One-Sided Tests (TOST) equivalence verification
│
├── plots/
│   ├── shap_beeswarm_pollutant_only.png # Global TreeSHAP beeswarm summary plot (Exp 1)
│   ├── shap_beeswarm_city_aware.png     # Global TreeSHAP beeswarm summary plot (Exp 2)
│   ├── shap_dependence_pm25.png        # TreeSHAP PM2.5 partial dependence colored by CO
│   ├── observed_vs_predicted_scatter.png# 1:1 diagonal scatter plot with linear trend
│   ├── residual_distribution_histogram.png# Residuals distribution vs Gaussian fit
│   ├── residuals_vs_predicted.png      # Residuals vs predictions (heteroscedasticity)
│   ├── error_by_aqi_category.png       # CPCB health advisory category error breakdown
│   └── error_by_year.png               # Error metrics partitioned by test year (2019 vs 2020)
│
├── manuscript/
│   ├── revised/
│   │   └── FINAL_ANALYSIS.md           # Submission-ready revised academic manuscript
│   └── original/
│       └── FINAL_ANALYSIS.md           # Preserved legacy manuscript draft for provenance
│
└── reports/
    ├── FINAL_ANALYSIS_REPORT.md        # Detailed scientific methods and comprehensive findings
    ├── DATA_DICTIONARY.md              # Feature metadata, CPCB breakpoints, and missingness
    ├── FINAL_QA_REPORT.md              # Audit certificate and assertion validation
    ├── preprocessing_report.json       # Learned medians and partition date ranges
    └── qa_report.json                  # Machine-readable test runner output
```

---

## 3. Environment Setup & Dependencies

The pipeline requires **Python 3.10–3.12**.

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

### Dependency Manifest (`requirements.txt`)
```text
pandas>=2.0.0,<=2.2.2
numpy>=1.24.0,<=1.26.4
scikit-learn>=1.3.0,<=1.4.2
xgboost>=2.0.0,<=2.0.3
lightgbm>=4.0.0,<=4.2.0
catboost>=1.2.0,<=1.2.5
shap>=0.44.0,<=0.45.1
matplotlib>=3.7.0,<=3.8.4
scipy>=1.11.0,<=1.13.0
```

---

## 4. End-to-End Reproduction Guide

All scripts are completely deterministic (fixed `random_state=42`) and write directly to relative directories.

### Step 1: Preprocessing & Data Partitioning
```bash
python scripts/01_preprocess.py
```
- Loads `data/raw/city_day.csv` ($N = 29,531$).
- Filters rows with valid CPCB AQI ($N = 24,624$).
- Establishes chronological splits: Training Sub (2015–2017: 7,556 rows), Validation (2018: 5,658 rows), Full Train (2015–2018: 13,214 rows), Out-of-Time Test (2019–2020: 11,410 rows).
- Fits pollutant median imputers **strictly on 2015–2018 training data** to eliminate data leakage.
- Preserves 100% of the natural continuous target distribution with **zero artificial clipping**.

### Step 2: Model Training, Evaluation & Statistical Inference
```bash
python scripts/02_train_and_evaluate.py
```
- Trains Random Forest, XGBoost, LightGBM, and CatBoost on the 8 criteria pollutants (Exp 1).
- Tunes early stopping rounds strictly on the 2018 validation set (zero test-set leakage).
- Fits one-hot encoded city-aware variants (Exp 2).
- Computes RMSE, MAE, $R^2$, and serializes test-set predictions.
- Performs formal statistical testing: Paired t-tests, Wilcoxon signed-rank tests, day-level cluster aggregated tests, Two One-Sided Tests (TOST) equivalence testing ($\pm 2.0$ margin), and 1,000-fold paired bootstrap confidence intervals.
- Conducts expanding-window rolling-origin time-series cross-validation across 3 developmental windows.

### Step 3: Explainability Analysis (TreeSHAP)
```bash
python scripts/03_shap_analysis.py
```
- Applies exact TreeSHAP algorithms on the physical scale of pollutant concentrations ($\mu g/m^3$, $mg/m^3$).
- Generates global beeswarm summary plots for Exp 1 and Exp 2.
- Generates partial dependence interaction plot for $PM_{2.5}$ colored by $CO$.
- Exports numerical importance rankings (`FINAL_SHAP_IMPORTANCE.csv`).

### Step 4: Diagnostic Visualizations & Residual Auditing
```bash
python scripts/04_diagnostics_and_plots.py
```
- Produces Observed vs. Predicted scatter plot with 1:1 reference line.
- Generates Residual distribution histogram with KDE and Gaussian fit overlays.
- Analyzes heteroscedasticity across predicted AQI levels with error envelopes.
- Decomposes RMSE and MAE across official CPCB health advisory bands (`Good` through `Severe`).
- Evaluates temporal stability across test years (2019 baseline vs. 2020 lockdown).

### Step 5: Automated Quality Assurance Suite
```bash
python scripts/test_pipeline_qa.py
```
- Executes an automated 11-point assertion suite validating partition shapes, zero target clipping, zero missing values, prediction metric reproducibility, and file integrity.
- Expected output: `11 / 11 CHECKS PASSED. ALL QUALITY CHECKS SUCCEEDED!`

---

## 5. Methodological Corrections & Audit Summary

This research release supersedes the legacy pipeline by resolving six fundamental methodological flaws identified during a rigorous forensic audit:

1. **Elimination of Preprocessing Data Leakage:** Imputation medians in legacy scripts were computed across all 2015–2020 rows simultaneously. In this release, imputation statistics are computed strictly on the training partition ($2015\text{--}2018$) and applied forward.
2. **Removal of Artificial Target Capping:** Legacy scripts capped target AQI at the 99th percentile ($666.31$) and contained an inversion bug that compressed extreme values. This release preserves the authentic CPCB continuous scale, verified with zero clipping error ($\max |y_{\text{processed}} - y_{\text{raw}}| = 0.00000000$).
3. **Elimination of Test-Set Leakage in Early Stopping:** Legacy XGBoost used `eval_set=[(X_test, y_test)]` during training, leaking holdout information into iteration stopping. This release tunes stopping exclusively against an isolated 2018 validation set.
4. **Fair Model Feature Comparison:** Legacy CatBoost was provided `City` as a categorical feature while RF, XGBoost, and LightGBM were restricted to pollutant columns. This confounded architecture comparison. Here, Experiment 1 strictly isolates pollutants across all 4 models, and Experiment 2 provides identical one-hot geographic features across all 4 models.
5. **Rigorous Statistical Testing (Replacement of Misapplied Nadeau-Bengio Test):** Legacy scripts incorrectly applied a resampled $t$-test meant for $K$-fold CV to an out-of-time test set, yielding uninterpretable $p$-values ($p > 0.96$). We replaced this with paired $t$-tests, non-parametric Wilcoxon tests, day-aggregated cluster tests, Two One-Sided Tests (TOST) for practical equivalence ($\pm 2.0$ points), and 1,000-fold bootstrap confidence intervals.
6. **Accurate Task Framing (Surrogate Estimation vs. Forecasting):** Legacy drafts conflated same-day pollutant-to-AQI estimation with predictive future forecasting. This study explicitly clarifies that it performs **same-day diagnostic surrogate estimation**, analyzing the physical and non-linear mapping from measured concentrations to regulatory AQI.

---

## 6. Disclosed Research Limitations

To maintain uncompromising scientific integrity, researchers adopting this work must acknowledge the following boundaries:

1. **Task Scope (Not Future Forecasting):** Models predict current-day AQI from current-day pollutant observations. They do not forecast $t+1$ or $t+7$ air quality.
2. **Physical AQI Derivative Relationship:** The Indian National AQI is deterministically calculated via sub-index piecewise linear breakpoints. ML surrogate models approximate this multi-pollutant max-operator mapping; hence, strong attributions for $PM_{2.5}$ and $CO$ reflect their regulatory role as dominant sub-index drivers rather than novel biological discoveries.
3. **Severe Event Underestimation:** Tree-based ensembles exhibit variance contraction and regression-to-the-mean at the extreme tail. For extreme episodic events ($AQI > 500$, $N = 173$), the models underestimate observed peaks by an average of $130.6$ AQI points.
4. **Observational Attribution vs. Causation:** SHAP values quantify conditional model attribution under the observational data distribution. They do not represent causal policy intervention effects.
5. **Spatial Generalization:** Chronological splitting verifies temporal generalization over time across existing monitoring infrastructure. When evaluated on 8 cities not present in the training set, model RMSE is $45.62$ vs. $39.54$ on seen cities, confirming that uncalibrated spatial transfer introduces regional baseline offsets.

---

## 7. Citation & Attribution

If you utilize the verified codebase, data splits, or benchmarks from this repository, please cite:

```bibtex
@article{aqi_ml_benchmarking_2026,
  title={Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study},
  author={Research Team},
  journal={Atmospheric Environment and Machine Learning Research},
  year={2026},
  note={Verified Reproducible Research Deliverable}
}
```
