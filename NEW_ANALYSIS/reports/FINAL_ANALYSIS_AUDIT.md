# Forensic Audit & Verification Report: AQI Machine Learning Benchmark

**Study Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Dataset:** CPCB Ambient Air Quality Monitoring Data (`city_day.csv`, 2015–2020)  
**Target Variable:** Daily AQI (Continuous score)  
**Evaluated Algorithms:** Random Forest, XGBoost, LightGBM, CatBoost  
**Audit Date:** October 2026  
**Auditor / Reviewer:** Senior Machine Learning Researcher & Methodology Reviewer  

---

## 1. Executive Summary & Audit Purpose

This audit report establishes the methodological integrity, data lineage, and numerical validity of the research pipeline benchmarking tree-based machine learning algorithms for urban Air Quality Index (AQI) modeling across Indian cities.

Historically, this project reported results indicating that CatBoost achieved an RMSE of **30.5820**, MAE of **18.3699**, and $R^2$ of **0.9163**, with XGBoost at RMSE **31.5151**, LightGBM at **32.4736**, and Random Forest at **32.7604**. 

Our code and artifact audit confirms that **the legacy results were fundamentally compromised by pervasive data leakage, artificial target capping, test set contamination, and an unlevel benchmarking field**. 

A subsequent revision was initiated in the workspace, generating preliminary corrected metrics:
- **Pollutant-Only Benchmark (Exp 1):**
  - CatBoost: RMSE 39.6688, MAE 19.5689, $R^2$ 0.8915, Train Time 1.62s
  - LightGBM: RMSE 40.9889, MAE 20.2186, $R^2$ 0.8842, Train Time 0.57s
  - Random Forest: RMSE 42.2185, MAE 20.6647, $R^2$ 0.8771, Train Time 4.08s
  - XGBoost: RMSE 42.5595, MAE 20.4114, $R^2$ 0.8751, Train Time 0.29s
- **City-Aware Benchmark (Exp 2):**
  - CatBoost: RMSE 39.0882, MAE 19.7629, $R^2$ 0.8947, Train Time 19.95s
  - LightGBM: RMSE 40.5550, MAE 20.1246, $R^2$ 0.8866, Train Time 0.59s
  - Random Forest: RMSE 41.7839, MAE 20.6991, $R^2$ 0.8796, Train Time 3.64s
  - XGBoost: RMSE 42.0344, MAE 19.9653, $R^2$ 0.8782, Train Time 0.31s

This audit documents:
1. The exact origins of all past and preliminary revised metrics.
2. The confirmed methodological flaws in the original code.
3. The validation status of the revised preprocessing, training, and SHAP pipelines.
4. The remaining incomplete items and the remediation plan.

---

## 2. Completed Work vs. Incomplete Work

### Completed Work
- [x] **Full Forensic Code & Artifact Inspection:** Inspected all legacy scripts in `_backup_original/` (`01_preprocess.py`, `02_train_and_evaluate.py`, `03_shap_analysis.py`), intermediate CSVs, logs, and revised scripts.
- [x] **Data Cleaning & Target Preservation:** Verified that missing AQI targets (4,681 rows) and sparse observations (>40% missing pollutants: 226 rows) are dropped, leaving 24,624 valid observations.
- [x] **Zero Target Modification:** Verified that ground-truth AQI is completely unaltered in the revised pipeline (max deviation from raw data = 0.00000000).
- [x] **Chronological Disjoint Splits:** Verified strict chronological partitioning:
  - Training Sub-split: 2015-01-01 to 2017-12-31 ($N = 7,556$)
  - Validation Split: 2018-01-01 to 2018-12-31 ($N = 5,658$)
  - Combined Development (Full Train): 2015-01-01 to 2018-12-31 ($N = 13,214$)
  - Out-of-Time Test Split: 2019-01-01 to 2020-07-01 ($N = 11,410$)
- [x] **Leakage-Free Imputation:** Column medians computed strictly on training partitions and applied to validation/test sets without future lookahead.
- [x] **Fair Benchmarking Separation:** Implemented Experiment 1 (Pollutant-Only: 8 criteria pollutants across all 4 models) and Experiment 2 (City-Aware: 8 pollutants + City across all 4 models).
- [x] **Early Stopping Remediation:** XGBoost, LightGBM, and CatBoost tune iterations on the 2018 validation set and refit on full training data, leaving the 2019–2020 test partition completely untouched.
- [x] **Automated QA Suite:** Verified that `test_pipeline_qa.py` passes 10/10 automated assertions.

### Incomplete Work & Deficiencies to Remediate
- [ ] **Physical Project Workspace Structure (Phase 0):** The project workspace lacks the formal separation into `OLD_FILES_ARCHIVE/`, `NEW_ANALYSIS/`, `manuscript/`, `data/`, and `PROJECT_FILE_MANIFEST.csv`. Outputs are currently mixed in the workspace root.
- [ ] **Standalone Prediction CSVs:** Predictions are currently stored only inside a 454 MB pickle file (`trained_models.pkl`), hindering rapid review, external validation, and subgroup analysis. Explicit CSV prediction files must be emitted.
- [ ] **Baseline Models:** No naive or empirical baseline (e.g., Training-Mean Predictor or Persistence Baseline) has been evaluated to quantify the true value-add of tree ensembles over simple heuristics.
- [ ] **Complete Spatial Error Decomposition:** While 6 metropolitan cities were summarized in `results_detailed.json`, a full city-wise performance table covering all 26 cities (sample count, RMSE, MAE, $R^2$) has not yet been exported to a dedicated artifact.
- [ ] **Comprehensive Diagnostic Visualizations:** Residual distributions, observed-versus-predicted regression plots, and error-versus-prediction diagnostic curves have not been generated.
- [ ] **Formal Statistical Analysis Document:** While paired tests and TOST calculations were executed, a standalone `STATISTICAL_ANALYSIS.md` detailing methodology, assumptions, rolling-origin cross-validation, and effect sizes is missing.
- [ ] **Standardized SHAP File Naming:** Artifacts currently use generic names (`shap_beeswarm.png`, `shap_feature_ranking_exp1.csv`) rather than the standardized naming required by the research package (`shap_beeswarm_pollutant_only.png`, `shap_beeswarm_city_aware.png`, etc.).
- [ ] **Formal Quality Assurance Document:** `FINAL_QA_REPORT.md` must be compiled to document the complete verification findings and test outcomes.

---

## 3. Confirmed Methodological Errors in Original Implementation

The forensic inspection of `_backup_original/` confirmed seven critical methodological errors:

### Error 1: Pervasive Lookahead Data Leakage in Imputation
* **Location:** `_backup_original/01_preprocess.py` (L56–61)
* **Code:** `df[col] = df[col].fillna(df[col].median())` across the entire pooled dataset before splitting.
* **Impact:** 2019–2020 air quality data—including unprecedented clean-air periods during the March–May 2020 COVID-19 national lockdown—leaked into training data imputation statistics.
* **Severity:** **CRITICAL**

### Error 2: Ground-Truth Target Capping & Artificial Performance Inflation
* **Location:** `_backup_original/01_preprocess.py` (L63–71)
* **Code:** `for col in POLLUTANTS + [TARGET]: ... df[col] = np.where(df[col] > upper, p99, df[col])`
* **Impact:** Ground-truth observed AQI values up to 2,049 were artificially clamped to the 99th percentile ($666.31$). This clipped real-world hazardous pollution episodes, falsely suppressing holdout RMSE from ~39.7 to ~30.6. Models were evaluated against a synthetic truncated target rather than real-world air pollution.
* **Severity:** **CRITICAL**

### Error 3: Flawed Mathematical Outlier Implementation
* **Location:** `_backup_original/01_preprocess.py` (L68–70)
* **Code:** `df[col] = np.where(df[col] > upper, p99, df[col])` where `upper = mean + 3*std`.
* **Impact:** In severely right-skewed atmospheric data, the empirical condition $\mu + 3\sigma < P_{99}$ holds frequently (e.g., for $\text{PM}_{2.5}$, $254.36 < 306.55$). Consequently, this line replaced values between $\mu + 3\sigma$ and $P_{99}$ with $P_{99}$, **inadvertently inflating over 2,000 pollutant observations** rather than capping them.
* **Severity:** **HIGH**

### Error 4: Direct Test Set Contamination in XGBoost Early Stopping
* **Location:** `_backup_original/02_train_and_evaluate.py` (L95)
* **Code:** `xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)`
* **Impact:** XGBoost used the final out-of-time test set (2019–2020) to select its optimal boosting round (`early_stopping_rounds=20`), violating fundamental out-of-sample testing protocols.
* **Severity:** **CRITICAL**

### Error 5: Confounded Feature Sets Across Models (Unfair Benchmark)
* **Location:** `_backup_original/02_train_and_evaluate.py` (L67, L119)
* **Code:** CatBoost was trained with `["City"] + FEATURES`, whereas Random Forest, XGBoost, and LightGBM were trained strictly on `FEATURES` (pollutants only).
* **Impact:** The performance advantage attributed to CatBoost's algorithm was confounded by access to geographic identity. Competing models were denied city data.
* **Severity:** **HIGH**

### Error 6: Misapplication of Nadeau-Bengio Resampled $t$-Test
* **Location:** `_backup_original/02_train_and_evaluate.py` (L47–63)
* **Code:** Applied Nadeau & Bengio's (2003) variance correction factor $(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}})$ to sample squared errors from a single holdout partition ($N = 11,410$).
* **Impact:** The formula was designed exclusively for repeated cross-validation or resampling splits, not single fixed test sets. The resulting high $p$-values ($p > 0.96$) were then falsely reported in the manuscript as "proof that models are statistically equivalent." (Failure to reject $H_0$ never proves equivalence without a TOST framework).
* **Severity:** **HIGH**

### Error 7: Task Mischaracterization
* **Location:** `_backup_original/FINAL_ANALYSIS.md`, `README.md`
* **Text:** The research was described as "early-warning forecasting" and "predicting future AQI."
* **Reality:** The dataset features ($C_{\text{pollutant}, t}$) and target ($\text{AQI}_t$) are measured concurrently on the same day. By statutory definition (CPCB guidelines), $\text{AQI}_t = \max_i \{ I_i(C_{i, t}) \}$. The model is performing **same-day virtual sensing, non-linear index estimation, and surrogate reconstruction**, NOT prognostic future forecasting.
* **Severity:** **HIGH**

---

## 4. Potential Issues & Sensitivities Requiring Investigation

1. **Unseen Cities in Out-of-Time Test Set:**
   - 8 out of 26 cities (Aizawl, Bhopal, Chandigarh, Coimbatore, Ernakulam, Guwahati, Kochi, Shillong) only appear starting in 2019–2020 (post-NCAP expansion).
   - They represent 2,036 out of 11,410 test rows (17.84%).
   - In Experiment 2 (City-Aware), how do models handle unseen categories? 
     - LightGBM and CatBoost handle unseen levels natively.
     - Scikit-learn `TargetEncoder` assigns the global target mean to unseen categories.
   - We must explicitly quantify and report seen vs. unseen city performance.

2. **Extreme Value Peak Underestimation:**
   - Because tree ensembles predict piece-wise constant averages within leaf nodes, they inherently struggle to extrapolate beyond the maximum training AQI and underestimate severe episodic spikes ($>500$).
   - A dedicated diagnostic of error vs. AQI severity band must be quantified.

3. **Spatial Cross-City Correlation on Identical Dates:**
   - Multiple cities are monitored on the exact same calendar day. Regional weather patterns (e.g., North Indian winter smog) induce cross-city residual correlation.
   - Standard observation-level paired $t$-tests ($N = 11,410$) will artificially inflate test statistics ($p \approx 0$).
   - Date-aggregated paired testing ($T = 548$ distinct calendar days) and clustered bootstrap intervals are necessary to verify whether CatBoost's lead survives spatial-temporal correlation adjustments.

---

## 5. Required Corrections & Action Plan

1. **Execute Phase 0 File Reorganization:**
   - Safely archive all legacy files into `OLD_FILES_ARCHIVE/`.
   - Structure `NEW_ANALYSIS/` with subdirectories for scripts, data, models, predictions, results, statistics, SHAP, diagnostics, and reports.
   - Generate `PROJECT_FILE_MANIFEST.csv`.
2. **Execute Full Pipeline Re-Run in Clean Environment:**
   - Run `01_preprocess.py` to regenerate verified partitions in `NEW_ANALYSIS/data_processed/`.
   - Run `02_train_and_evaluate.py` to train all models, log timings, save model weights, and emit clean prediction CSVs in `NEW_ANALYSIS/predictions/`.
   - Add Training-Mean and Previous-Day Persistence Baselines.
   - Run `03_shap_analysis.py` to generate standardized SHAP CSVs and publication-grade plots in `NEW_ANALYSIS/shap_outputs/`.
3. **Generate Extended Diagnostics & Subgroup Reports:**
   - Generate full 26-city error metrics table.
   - Generate observed-vs-predicted plots, residual histograms, and residual vs. predicted scatterplots in `NEW_ANALYSIS/diagnostics/`.
4. **Compile Rigorous Academic Reports:**
   - Generate `STATISTICAL_ANALYSIS.md` (Phase 4).
   - Generate `FINAL_QA_REPORT.md` (Phase 8).
   - Update `README.md` and manuscript documentation with verified facts and honest framing.

---

## 6. Audit Verdict

| Assessment Dimension | Legacy Implementation | Revised Implementation (Current Status) | Target Deliverable Status |
|---|---|---|---|
| **Data Leakage** | Compromised (Leakage in Imputation) | Zero Leakage (Train-only parameters) | **Verified** |
| **Target Integrity** | Falsified (Capped at 666.31) | 100% Unaltered Observed AQI | **Verified** |
| **Validation Separation** | Failed (Early stopping on test set) | Strict 2018 Validation Set Tuning | **Verified** |
| **Benchmarking Equity** | Confounded (CatBoost alone had City) | Strictly Controlled (Exp 1 vs Exp 2) | **Verified** |
| **Statistical Rigor** | Invalid Formula & False Equivalence | Paired Tests, Bootstrap CIs & TOST | **Verified** |
| **Folder Organization** | Loose Root Files | In Progress | **To Be Finalized** |
| **Diagnostics & Baselines** | None | Partial | **To Be Completed** |

**Current Status:** **Partially Complete** (The underlying algorithms and preprocessing are mathematically sound and verified; physical project file separation, baseline models, comprehensive 26-city diagnostics, and formal markdown reports are now ready to be generated).
