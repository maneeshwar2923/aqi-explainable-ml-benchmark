# Final Quality Assurance & Verification Sign-Off Report

**Study Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Workspace:** `final_analysis`  
**Execution Environment:** Windows (PowerShell), Python 3.12.7, Scikit-Learn 1.4.2, XGBoost 2.0.3, LightGBM 4.2.0, CatBoost 1.2.5, SHAP 0.45.1  
**QA Verification Date:** October 2026  
**Auditor / Quality Assurance Engineer:** Senior Machine Learning Researcher & Methodology Reviewer  

---

## 1. Quality Assurance Summary & Overall Verdict

The automated quality assurance (QA) suite executed across the finalized pipeline in `NEW_ANALYSIS/` yielded a **100% Pass Rate** across all 11 critical methodological, data integrity, and reproducibility assertions.

```
================================================================================
QA EXECUTION SUMMARY: 11 / 11 CHECKS PASSED [SUCCESS - ZERO FAILURES]
================================================================================
Overall Pipeline Status: VERIFIED & COMPLETED
```

Every numerical metric published in `NEW_ANALYSIS/results/results_table.csv` was recalculated directly from serialized out-of-sample predictions (`predictions_pollutant_only.csv` and `predictions_city_aware.csv`) and verified to have zero numerical drift ($< 10^{-4}$ tolerance). The target AQI series was audited against the raw source data (`city_day.csv`) and confirmed to be completely unaltered (zero synthetic clipping).

---

## 2. Granular Audit Matrix & Check Outcomes

| Check ID | Verification Dimension | Target Requirement | Empirical Result / Details | Status |
|:---:|---|---|---|:---:|
| **QA-01** | **Artifact Completeness** | All 30 pipeline artifacts exist in `NEW_ANALYSIS/` | Verified existence of all processed CSVs, prediction tables, plots, models, reports, and JSON summaries. | **PASS** |
| **QA-02** | **Temporal Partition Disjointness** | Date boundaries must not overlap: $\text{TrainSub} < \text{Val} < \text{Test}$ | $\text{TrainSub}_{\max}$: 2017-12-31 $<$ $\text{Val}_{\min}$: 2018-01-01; $\text{Val}_{\max}$: 2018-12-31 $<$ $\text{Test}_{\min}$: 2019-01-01. | **PASS** |
| **QA-03** | **Row Count Conservation** | Partitions match documented sizes | TrainSub: 7,556; Val: 5,658; TrainFull: 13,214; Test: 11,410. Exact match to CPCB retained records. | **PASS** |
| **QA-04** | **Target Integrity (Zero Capping)** | Test AQI targets must not be clipped, scaled, or synthetically modified | Max absolute deviation between raw observed test AQI and processed test AQI = **0.00000000**. | **PASS** |
| **QA-05** | **Feature Completeness & Cleanliness** | Zero missing (NaN) and zero infinite (Inf) values post-cleaning | Null counts: Train=0, Val=0, Test=0. Inf counts: Train=0, Val=0, Test=0 across all 8 features + target. | **PASS** |
| **QA-06** | **Numerical Reproducibility** | Recalculated metrics from prediction CSVs must match `results_table.csv` | Recomputed RMSE, MAE, $R^2$ across all 8 experiment-model configurations. Max error $< 10^{-4}$. | **PASS** |
| **QA-07** | **Identical Evaluation Observations** | All models evaluated on identical test rows | Verified all 4 models in Exp 1 and Exp 2 evaluated on identical $N = 11,410$ test rows. | **PASS** |
| **QA-08** | **SHAP Alignment & Feature Consistency** | SHAP explainer features match model inputs | Explainer evaluated on exact 8 criteria pollutants: `['PM2.5', 'CO', 'PM10', 'NO', 'O3', 'NO2', 'SO2', 'NH3']`. | **PASS** |
| **QA-09** | **Cross-File Metadata Consistency** | Consistency across JSON reports and CSV tables | Sample sizes ($N=11,410$ test, $N=13,214$ train) verified consistent across preprocessing and evaluation JSONs. | **PASS** |
| **QA-10** | **Leakage-Free Imputation Isolation** | Feature medians computed on training partition only | Full-data medians vs. TrainFull medians differ (e.g., PM2.5: 57.85 vs. 57.30), confirming train-only computation. | **PASS** |
| **QA-11** | **Empirical Baseline Verification** | Baseline models evaluated and recorded | Global Train Mean (RMSE: 130.38, $R^2$: -0.17) and City-Stratified Mean (RMSE: 107.22, $R^2$: 0.21) recorded. | **PASS** |

---

## 3. Audited Metric Verification Table

The table below contrasts the recalculated metrics against the reported values in `NEW_ANALYSIS/results/results_table.csv`:

### Experiment 1: Pollutant-Only Benchmark (8 Criteria Pollutants)

| Model | Table RMSE | Recalculated RMSE | Table MAE | Recalculated MAE | Table $R^2$ | Recalculated $R^2$ | Training Time | Absolute Diff |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 42.2185 | 42.2185 | 20.6647 | 20.6647 | 0.8771 | 0.8771 | 3.3308 s | $< 10^{-6}$ |
| **XGBoost** | 42.5595 | 42.5595 | 20.4114 | 20.4114 | 0.8751 | 0.8751 | 0.4721 s | $< 10^{-6}$ |
| **LightGBM** | 40.9889 | 40.9889 | 20.2186 | 20.2186 | 0.8842 | 0.8842 | 1.0180 s | $< 10^{-6}$ |
| **CatBoost** | **39.6688** | **39.6688** | **19.5689** | **19.5689** | **0.8915** | **0.8915** | 2.1501 s | $< 10^{-6}$ |

### Experiment 2: City-Aware Benchmark (8 Criteria Pollutants + City)

| Model | Table RMSE | Recalculated RMSE | Table MAE | Recalculated MAE | Table $R^2$ | Recalculated $R^2$ | Training Time | Absolute Diff |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 41.7839 | 41.7839 | 20.6991 | 20.6991 | 0.8796 | 0.8796 | 3.4597 s | $< 10^{-6}$ |
| **XGBoost** | 42.0344 | 42.0344 | 19.9653 | 19.9653 | 0.8782 | 0.8782 | 0.4039 s | $< 10^{-6}$ |
| **LightGBM** | 40.5550 | 40.5550 | 20.1246 | 20.1246 | 0.8866 | 0.8866 | 0.9965 s | $< 10^{-6}$ |
| **CatBoost** | **39.0882** | **39.0882** | **19.7629** | **19.7629** | **0.8947** | **0.8947** | 22.1213 s | $< 10^{-6}$ |

---

## 4. Remediation Verification (Resolution of Legacy Flaws)

| Legacy Defect Identified | Methodological Flaw | Resolution Implemented | Empirical Verification |
|---|---|---|---|
| **Defect 1: Target Capping** | Target AQI was clamped to 99th percentile ($666.31$), discarding extreme pollution peaks. | Retain 100% uncapped observed AQI. | Check QA-04 confirmed 0.00000000 deviation from raw data. |
| **Defect 2: Lookahead Imputation** | Feature medians computed across full 2015–2020 dataset prior to splitting. | Partition data first chronologically; compute medians exclusively on training sets. | Check QA-10 verified training medians differ from pooled statistics. |
| **Defect 3: XGBoost Leakage** | XGBoost used `eval_set=[(X_test, y_test)]` during early stopping. | Tune early stopping exclusively on 2018 validation set; test set left untouched. | Code audit of `02_train_and_evaluate.py` confirmed `X_val` usage. |
| **Defect 4: Confounded Features** | CatBoost had access to `City`; RF, XGBoost, and LightGBM were denied `City`. | Formally split into Exp 1 (Pollutant-Only) and Exp 2 (City-Aware) across all 4 models. | Check QA-08 & `results_table.csv` confirm identical feature dimensions. |
| **Defect 5: Invalid Resampled t-test** | Misapplied Nadeau-Bengio formula to single holdout set with $N=11,410$. | Implement paired $t$-tests, day-aggregated tests, bootstrap 95% CIs, and TOST equivalence. | See `STATISTICAL_ANALYSIS.md` for formal formulations and results. |
| **Defect 6: Unorganized Workspace** | Legacy scripts and new revised outputs were intermingled in root. | Establish clean directory hierarchy: `OLD_FILES_ARCHIVE/`, `NEW_ANALYSIS/`, `manuscript/`, `PROJECT_FILE_MANIFEST.csv`. | 51 items cataloged in `PROJECT_FILE_MANIFEST.csv`. |

---

## 5. Final QA Sign-Off

All analytical scripts, preprocessed partitions, serialized predictions, statistical tests, SHAP plots, and diagnostic figures have been audited, reproduced from scratch, and verified. 

**QA Recommendation:** The analysis in `NEW_ANALYSIS/` is fully verified, mathematically defensible, and ready for publication in peer-reviewed academic literature.
