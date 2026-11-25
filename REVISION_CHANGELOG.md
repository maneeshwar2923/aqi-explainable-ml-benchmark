# Revision Changelog: Leakage-Free AQI Prediction & Explainable ML

**Project:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Institution:** CHRIST (Deemed to be University) — Research Project  
**Date of Audit & Revision:** October 2026  
**Auditor / Reviewer:** Senior ML Engineer & Academic Methodology Reviewer  

---

## 1. Executive Summary of Audit Findings

A thorough forensic audit of the data, code (`01_preprocess.py`, `02_train_and_evaluate.py`, `03_shap_analysis.py`), stored outputs, and documentation (`FINAL_ANALYSIS.md`, `README.md`) revealed severe methodological flaws that invalidated the previously reported results (e.g., CatBoost RMSE 30.58, MAE 18.37, $R^2$ 0.916). 

The headline findings include:
1. **Pervasive Data Leakage:** Imputation statistics, outlier thresholds, and scaling parameters were computed across the entire dataset (including 2019–2020 test observations) *before* splitting.
2. **Target Modification:** Ground-truth AQI targets up to 2049 were capped at 666.31 across the full dataset, meaning models were evaluated on synthetic clipped targets rather than actual observed pollution.
3. **Outlier Calculation Bug:** A hybrid $3\sigma$ / 99th-percentile rule inadvertently *increased* more than 2,000 pollutant values due to severe right-skewness where $\mu + 3\sigma < P_{99}$.
4. **Test Set Contamination:** XGBoost early stopping was fitted directly against the final test set `[(X_test, y_test)]`.
5. **Confounded Benchmarking:** CatBoost was granted access to the `City` categorical feature, whereas Random Forest, XGBoost, and LightGBM were restricted to 8 pollutant features.
6. **Unseen Spatial Distributions:** 8 of the 26 cities (17.84% of test observations) exist solely in the 2019–2020 test set, which was unaddressed in the modeling methodology.
7. **Invalid Statistical Testing:** A single holdout set of 11,410 observations was incorrectly passed into a formula mislabeled as Nadeau-Bengio (2003) corrected resampled $t$-test; failure to reject the null hypothesis was erroneously claimed as "proof of statistical equivalence."
8. **Task Mischaracterization:** Same-day concurrent AQI estimation from same-day pollutant sub-indices was inaccurately described as "early-warning future forecasting."

All original files were backed up to `_backup_original/` prior to any code modifications.

---

## 2. Granular Issue Log & Planned Corrections

| Issue ID | Module / Component | Severity | Nature of Flaw | Planned Correction | Outputs Requiring Regeneration |
|:---:|---|:---:|---|---|---|
| **ISSUE-01** | `01_preprocess.py` (L56–61) | **CRITICAL** | **Data Leakage in Imputation:** Column medians for 8 pollutants were computed on the pooled 2015–2020 dataset prior to splitting, leaking future clean-air data (e.g. 2020 lockdowns) into the training period. | Fit imputation statistics strictly on the training partition (2015–2018 or 2015–2017 training fold). Apply frozen parameters to validation and test partitions without refitting. Save parameters to report. | `train.csv`, `val.csv`, `test.csv`, `preprocessing_summary.txt` |
| **ISSUE-02** | `01_preprocess.py` (L63–71) | **CRITICAL** | **Ground Truth Target Capping:** The regression target `AQI` was capped at $P_{99} = 666.31$. Observed AQI peaks up to 2,049 were clipped, artificially depressing holdout RMSE from ~50 to ~32. | Never clip or modify the ground-truth target variable `AQI`. Evaluate all models strictly against the unaltered real-world observed AQI. | `train.csv`, `test.csv`, all model evaluations |
| **ISSUE-03** | `01_preprocess.py` (L63–71) | **HIGH** | **Mathematical Capping Flaw:** Code replaced values $> \mu + 3\sigma$ with $P_{99}$. In right-skewed data, $\mu + 3\sigma < P_{99}$ (e.g. PM2.5: $254.36 < 306.55$), causing $>2,000$ values to be artificially inflated. | Tree-based models are invariant to monotonic transformations and robust to input outliers. Use an unclipped physical baseline; if capping is tested, enforce $x \leftarrow \min(x, \text{threshold})$ fitted strictly on training data. | `train.csv`, `test.csv`, `preprocessing_summary.txt` |
| **ISSUE-04** | `01_preprocess.py` (L73–78) | **HIGH** | **Absence of Validation Partition:** Pipeline performed only a single train/test split. No chronological validation split was provided for model selection or early stopping. | Partition data chronologically into: Training (2015–2017: 7,556 rows), Validation (2018: 5,658 rows), and Final Test (2019–2020: 11,410 rows). Also retain full Training (2015–2018: 13,214 rows). | `train.csv`, `val.csv`, `test.csv`, `train_full.csv` |
| **ISSUE-05** | `01_preprocess.py` (L80–90) | **MEDIUM** | **Unnecessary Scaling & Unit Loss:** Continuous pollutants were z-score standardized, altering column names to `_z` and stripping physical units ($\mu\text{g/m}^3$) needed for intuitive SHAP interpretation. | Remove standardization for tree models to preserve physical units and interpretability, or document training-only scaling parameters. Standardize feature naming across all scripts. | `train.csv`, `test.csv`, SHAP plots |
| **ISSUE-06** | `01_preprocess.py`, `02_train_and_evaluate.py`, `03_shap_analysis.py` | **MEDIUM** | **Hardcoded Path Failures:** Hardcoded path `C:\Users\manee\OneDrive\Documents\...` did not match the actual repository location `...\OneDrive\Pictures\Documents\...`. | Use dynamic path resolution relative to script location (`__file__`) with configurable override arguments. | All scripts |
| **ISSUE-07** | `02_train_and_evaluate.py` (L95) | **CRITICAL** | **Test Set Early Stopping Leakage:** XGBoost was trained with `eval_set=[(X_test, y_test)]`, directly contaminating iteration selection with the final test set. | Configure early stopping strictly on the chronological validation set (2018). The final test set remains untouched until final inference. | `trained_models.pkl`, `results_table.csv`, `results_summary.txt` |
| **ISSUE-08** | `02_train_and_evaluate.py` (L67, 119) | **HIGH** | **Unfair Feature Comparison:** CatBoost was trained with `City`, while RF, XGBoost, and LightGBM were restricted to 8 pollutants. | Split benchmark into two explicit experiments: **Experiment 1 (Pollutant-only benchmark)** across all 4 models; **Experiment 2 (City-aware benchmark)** across all 4 models using algorithm-appropriate encoding (target/one-hot/native). | `results_table.csv`, `results_summary.txt`, `trained_models.pkl` |
| **ISSUE-09** | `city_day.csv` distribution | **MEDIUM** | **Unseen Test Categories:** 8 cities (Guwahati, Coimbatore, Chandigarh, Bhopal, Shillong, Kochi, Ernakulam, Aizawl; 2,036 test rows / 17.84%) appear only in 2019–2020. | Handle unseen categories robustly (e.g. global target mean imputation for target encoders; native missing handling in CatBoost/LightGBM). Report metrics partitioned by seen vs unseen cities. | Results reporting & paper |
| **ISSUE-10** | `02_train_and_evaluate.py` (L47–63) | **HIGH** | **Misapplied Resampled $t$-Test:** Nadeau-Bengio formula was misapplied to $N=11,410$ observations from a single holdout split, then claimed as a "5x2cv style test." $p > 0.96$ was wrongly claimed to prove equivalence. | Replace with defensible statistical inference: paired error analysis on holdout set, cluster-robust standard errors, Bonferroni-Holm adjustment, bootstrap 95% CIs for metric differences, and rolling-origin cross-validation during tuning. Avoid claiming equivalence without TOST. | `results_summary.txt`, `FINAL_ANALYSIS.md` |
| **ISSUE-11** | `03_shap_analysis.py` (L40–50) | **MEDIUM** | **SHAP Alignment & Causal Conflation:** SHAP values were computed on z-scaled features with `City` index shifting. Documentation interpreted attribution percentages as physical causal impacts. | Compute SHAP on finalized models using exact features. Re-label with physical pollutant units. Frame SHAP strictly as model attribution, explicitly disclaiming causal policy interventions. | `shap_feature_ranking.csv`, `shap_beeswarm.png`, `shap_dependence_top_feature.png` |
| **ISSUE-12** | `FINAL_ANALYSIS.md`, `README.md` | **HIGH** | **Task Mischaracterization:** Same-day AQI prediction was repeatedly described as "future forecasting" and "early-warning alerts." | Correct research terminology throughout the paper and documentation: same-day concurrent estimation/reconstruction (virtual sensing / surrogate modeling) vs. genuine prognostic forecasting. | `FINAL_ANALYSIS.md`, `README.md`, `data/content.ts` |

---

## 3. Execution Plan

1. **Phase 1 — Preprocessing Overhaul (`01_preprocess.py`):**
   - Clean missing AQI targets (drop 4,681 rows).
   - Drop rows with $>40\%$ missing pollutants (drop 226 rows).
   - Chronological split: Training (2015–2017: 7,556), Validation (2018: 5,658), Test (2019–2020: 11,410); full training (2015–2018: 13,214).
   - Fit pollutant medians strictly on Training data. Apply without refitting to Validation and Test.
   - Retain raw uncapped ground-truth AQI target. Retain physical pollutant concentrations (no z-score scaling, with optional standardized columns if needed).
   - Emit machine-readable report `preprocessing_report.json` and human-readable `preprocessing_summary.txt`.

2. **Phase 2 — Fair Modeling & Evaluation Engine (`02_train_and_evaluate.py`):**
   - Implement **Experiment 1 (Pollutant-only benchmark)**: 8 criteria pollutants across RF, XGBoost, LightGBM, CatBoost.
   - Implement **Experiment 2 (City-aware benchmark)**: 8 pollutants + City across all 4 models with robust unseen-category handling.
   - Validation & Early Stopping: Use 2018 validation set to tune and stop boosting iterations without touching the test set.
   - Final evaluation: Evaluate on 2019–2020 test set once. Report RMSE, MAE, $R^2$, and wall-clock training time.
   - Granular breakdown: Report metrics partitioned by year and by city (highlighting seen vs unseen cities).

3. **Phase 3 — Rigorous Statistical Inference:**
   - Paired absolute and squared error analysis on the common holdout test set.
   - Clustered / block-bootstrap 95% confidence intervals for error differences.
   - Pairwise tests with Bonferroni-Holm multiple comparison adjustments.
   - Temporal rolling-origin cross-validation across training years (2015–2018).
   - Precise language distinguishing lack of statistical significance ($p \ge \alpha$) from equivalence claims.

4. **Phase 4 — SHAP Interpretability Overhaul (`03_shap_analysis.py`):**
   - Compute TreeSHAP on test set for finalized models.
   - Export publication-ready beeswarm plot and partial dependence plot with physical units.
   - Save feature attribution rankings to CSV for both experiments.
   - Ground interpretations in model attribution rather than physical causality.

5. **Phase 5 — Manuscript & Documentation Update:**
   - Update `FINAL_ANALYSIS.md`, `README.md`, and Next.js data stores (`data/models.ts`, `data/content.ts`) with honest, verified numbers.
   - Update methodology, results tables, figures, statistical discussions, and limitations.

6. **Phase 6 — Automated Quality Assurance:**
   - Execute QA script verifying: zero leakage, date separation, identical targets, column alignments, no NaNs/inf, and metric reproducibility.
