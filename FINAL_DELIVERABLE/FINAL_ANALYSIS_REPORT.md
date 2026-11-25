# Final Analysis & Research Verification Report

**Project Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Institution:** CHRIST (Deemed to be University) — Research Project  
**Lead Reviewer & Methodology Auditor:** Senior Machine Learning Researcher & Methodology Reviewer  
**Audit & Delivery Date:** October 2026  
**Final Status:** **VERIFIED & COMPLETE**  

---

## 1. Executive Summary

This report delivers the verified findings, methodology, and empirical evidence for the comparative benchmarking study of machine learning algorithms predicting the daily Air Quality Index (AQI) across 26 Indian cities.

Historically, this project reported results indicating that CatBoost achieved an RMSE of **30.5820** and an $R^2$ of **0.9163**. Our forensic audit confirmed that those legacy figures were artifacts of:
1. Artificial target clipping at $666.31$ (truncating extreme hazardous pollution events up to $2,049$).
2. Lookahead imputation leakage across the 2015–2020 dataset prior to splitting.
3. Test set contamination during XGBoost early stopping.
4. Confounded feature sets (CatBoost was granted access to `City`, while competing models were denied it).

### Remediated Research Findings
Following full remediation, models were evaluated on $11,410$ strictly untouched out-of-time test instances (2019–2020) against unaltered observed AQI:
- **Experiment 1 (Pollutant-Only Benchmark: 8 Criteria Pollutants):**
  - **CatBoost** achieved the lowest error: **$\text{RMSE} = 39.6688$**, **$\text{MAE} = 19.5689$**, **$R^2 = 0.8915$**.
  - **LightGBM** ranked second: **$\text{RMSE} = 40.9889$**, **$\text{MAE} = 20.2186$**, **$R^2 = 0.8842$**, training $2.11\times$ faster than CatBoost.
  - **Random Forest** ranked third: **$\text{RMSE} = 42.2185$**, **$\text{MAE} = 20.6647$**, **$R^2 = 0.8771$**.
  - **XGBoost** demonstrated superior computational throughput: **$\text{RMSE} = 42.5595$**, **$\text{MAE} = 20.4114$**, **$R^2 = 0.8751$**, training in **$0.4721$ seconds** ($4.55\times$ faster than CatBoost).
- **Experiment 2 (City-Aware Benchmark: 8 Pollutants + Encoded City):**
  - **CatBoost** achieved **$\text{RMSE} = 39.0882$**, **$\text{MAE} = 19.7629$**, **$R^2 = 0.8947$**.
  - **LightGBM** achieved **$\text{RMSE} = 40.5550$**, **$\text{MAE} = 20.1246$**, **$R^2 = 0.8866$**.
  - **Random Forest** achieved **$\text{RMSE} = 41.7839$**, **$\text{MAE} = 20.6991$**, **$R^2 = 0.8796$**.
  - **XGBoost** achieved **$\text{RMSE} = 42.0344$**, **$\text{MAE} = 19.9653$**, **$R^2 = 0.8782$**, training in **$0.4039$ seconds** ($55\times$ faster than CatBoost).
- **Statistical Equivalence:**
  Under Two One-Sided Tests (TOST) with an equivalence boundary of $\pm 2.0$ AQI points, all pairwise MAE differences between CatBoost, LightGBM, and XGBoost are **statistically equivalent** ($p_{\text{TOST}} < 0.0001$), with negligible Cohen's effect sizes ($d_z \le 0.063$).
- **Interpretability:**
  TreeSHAP identified fine particulate matter ($\text{PM}_{2.5}$, 40.98% relative share) and carbon monoxide ($\text{CO}$, 22.39% relative share) as the dominant statutory index drivers.

---

## 2. Chronological Data Partitioning & Data Leakage Prevention

```
========================================================================================
CHRONOLOGICAL DATA PARTITIONING SCHEMA (2015-01-01 to 2020-07-01, N=24,624 Valid Rows)
========================================================================================
Partition          Date Span                  Observations  Operational Role in Pipeline
----------------------------------------------------------------------------------------
train_sub.csv      2015-01-01 to 2017-12-31     7,556 rows  Tuning fold for validation early stopping
val.csv            2018-01-01 to 2018-12-31     5,658 rows  Early stopping stopping horizon (20 rounds)
train.csv (Full)   2015-01-01 to 2018-12-31    13,214 rows  Combined development set for final refitting
test.csv           2019-01-01 to 2020-07-01    11,410 rows  Untouched single-pass out-of-time evaluation
========================================================================================
```

### Methodological Safeguards:
1. **Target Integrity:** Target AQI was not clamped or modified (max deviation from observed data = $0.00000000$).
2. **Train-Only Imputation:** Medians were computed strictly on training partitions:
   - $\text{PM}_{2.5}: 57.85\,\mu\text{g/m}^3$, $\text{PM}_{10}: 108.41\,\mu\text{g/m}^3$, $\text{NO}_2: 25.36\,\mu\text{g/m}^3$, $\text{NH}_3: 21.20\,\mu\text{g/m}^3$, $\text{SO}_2: 8.44\,\mu\text{g/m}^3$, $\text{CO}: 1.02\,\text{mg/m}^3$, $\text{O}_3: 31.30\,\mu\text{g/m}^3$, $\text{NO}: 10.21\,\mu\text{g/m}^3$.
3. **No Early Stopping Contamination:** The 2019–2020 test set was isolated until final inference.

---

## 3. Verified Benchmark Results

### Table 1: Model Performance vs. Naive Baselines (Evaluated on Identical N=11,410 Observations)

| Experiment Category | Model Architecture | Encoding / Rule | RMSE ($\downarrow$) | MAE ($\downarrow$) | $R^2$ ($\uparrow$) | Train Time (s) | Relative Speed vs CatBoost |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | **Global Training Mean** | Constant (190.01) | 130.3831 | 102.6559 | -0.1720 | — | — |
| **Baseline** | **City-Stratified Mean** | City Mean (Fallback Global) | 107.2215 | 78.4335 | 0.2074 | — | — |
| **Experiment 1 (Pollutants)** | **CatBoost Regressor** | Unscaled Physical Features | **39.6688** | **19.5689** | **0.8915** | 2.1501 s | $1.00\times$ (Baseline) |
| **Experiment 1 (Pollutants)** | **LightGBM Regressor** | Unscaled Physical Features | **40.9889** | **20.2186** | **0.8842** | 1.0180 s | **$2.11\times$ faster** |
| **Experiment 1 (Pollutants)** | **Random Forest Regressor** | Unscaled Physical Features | **42.2185** | **20.6647** | **0.8771** | 3.3308 s | $0.65\times$ |
| **Experiment 1 (Pollutants)** | **XGBoost Regressor** | Unscaled Physical Features | **42.5595** | **20.4114** | **0.8751** | **0.4721 s** | **$4.55\times$ faster** |
| **Experiment 2 (City-Aware)** | **CatBoost Regressor** | Native Ordered Boosting | **39.0882** | **19.7629** | **0.8947** | 22.1213 s | $1.00\times$ |
| **Experiment 2 (City-Aware)** | **LightGBM Regressor** | Native Histogram Categorical | **40.5550** | **20.1246** | **0.8866** | 0.9965 s | **$22.2\times$ faster** |
| **Experiment 2 (City-Aware)** | **Random Forest Regressor** | TargetEncoder (5-fold CV) | **41.7839** | **20.6991** | **0.8796** | 3.4597 s | $6.40\times$ faster |
| **Experiment 2 (City-Aware)** | **XGBoost Regressor** | TargetEncoder (5-fold CV) | **42.0344** | **19.9653** | **0.8782** | **0.4039 s** | **$54.8\times$ faster** |

---

## 4. Rigorous Statistical Hypothesis Testing & Equivalence Analysis

### Table 2: Paired Inferential Comparisons (Primary CatBoost vs. Competitors on N=11,410 Test Rows)

| Comparison Pair | $\Delta\text{MAE}$ (pts) | 95% Bootstrap CI $\Delta\text{MAE}$ | 95% Bootstrap CI $\Delta\text{RMSE}$ | Cohen's $d_z$ | Paired $t$ $p_{\text{adj}}$ | Wilcoxon $p_{\text{adj}}$ | Daily $t$ $p_{\text{adj}}$ ($T=548$) | TOST $p$ ($\pm 2.0$ pts) | Practical Equivalence |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost vs Random Forest** | **+1.0958** | $[0.775, 1.427]$ | $[0.465, 4.849]$ | 0.0633 | $< 10^{-6}$ | $5.0 \times 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | **Statistically Equivalent** |
| **CatBoost vs XGBoost** | **+0.8425** | $[0.562, 1.131]$ | $[0.798, 5.005]$ | 0.0554 | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | $< 10^{-6}$ | **Statistically Equivalent** |
| **CatBoost vs LightGBM** | **+0.6497** | $[0.347, 0.946]$ | $[-0.787, 3.557]$ | 0.0400 | $2.0 \times 10^{-5}$ | $3.7 \times 10^{-5}$ | $1.6 \times 10^{-5}$ | $< 10^{-6}$ | **Statistically Equivalent** |

### Key Statistical Takeaways:
1. **Negligible Effect Sizes:** While large sample sizes ($N = 11,410$) produce small $p$-values ($p < 0.0001$), standardized effect sizes are universally negligible ($d_z \le 0.063$).
2. **Rejection of Practical Disparity (TOST):** Under Two One-Sided Tests against an equivalence boundary of $\pm 2.0$ AQI points ($<0.4\%$ of the scale), all pairwise differences are **statistically equivalent**.
3. **RMSE Confidence Interval Overlap:** The 95% bootstrap confidence interval for $\Delta\text{RMSE}$ between CatBoost and LightGBM spans $[-0.787, +3.557]$, confirming that **their RMSE difference is not statistically distinguishable at the 95% level**.

### Table 3: Multi-Window Expanding Chronological Cross-Validation

| Model Name | Window 1 (2017 Eval) RMSE | Window 1 (2017 Eval) MAE | Window 2 (2018 Eval) RMSE | Window 2 (2018 Eval) MAE | Window 3 (2019–20 Test) RMSE | Window 3 (2019–20 Test) MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 56.1169 | 31.3202 | 63.7841 | 31.6678 | 42.2185 | 20.6647 |
| **XGBoost** | 57.9338 | 31.1692 | 63.8972 | 31.7247 | 42.5595 | 20.4114 |
| **LightGBM** | 50.7447 | 29.8111 | 63.7473 | 31.0198 | 40.9889 | 20.2186 |
| **CatBoost** | **48.9591** | **28.6355** | **58.8201** | **29.0780** | **39.6688** | **19.5689** |

---

## 5. TreeSHAP Attribution Rankings & Visualizations

### Table 4: TreeSHAP Feature Attribution on Test Observations

| Feature Rank | Chemical Species | Units | Mean Abs SHAP (Exp 1) | Relative Share (Exp 1) | Mean Abs SHAP (Exp 2) | Relative Share (Exp 2) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **PM2.5** | $\mu\text{g/m}^3$ | **50.8072** | **40.98%** | **51.8816** | **41.63%** |
| 🥈 | **CO** | $\text{mg/m}^3$ | **27.7623** | **22.39%** | **25.9241** | **20.80%** |
| 🥉 | **PM10** | $\mu\text{g/m}^3$ | **20.9591** | **16.91%** | **17.5962** | **14.12%** |
| 4 | **City** | Categorical | — | — | **7.6091** | **6.11%** |
| 5 | **NO** | $\mu\text{g/m}^3$ | **7.7998** | **6.29%** | **7.5252** | **6.04%** |
| 6 | **O3** | $\mu\text{g/m}^3$ | **6.2459** | **5.04%** | **5.3123** | **4.26%** |
| 7 | **NO2** | $\mu\text{g/m}^3$ | **4.2701** | **3.44%** | **4.6306** | **3.72%** |
| 8 | **SO2** | $\mu\text{g/m}^3$ | **3.7345** | **3.01%** | **3.5954** | **2.88%** |
| 9 | **NH3** | $\mu\text{g/m}^3$ | **2.3935** | **1.93%** | **0.5572** | **0.45%** |

---

## 6. Diagnostic Insights & Error Breakdown

1. **CPCB Category Accuracy:**
   - Good (0–50): $\text{MAE} = 19.03$, $\text{RMSE} = 21.78$
   - Satisfactory (51–100): $\text{MAE} = 9.93$, $\text{RMSE} = 13.69$
   - Moderate (101–200): $\text{MAE} = 18.27$, $\text{RMSE} = 30.73$
   - Poor (201–300): $\text{MAE} = 30.22$, $\text{RMSE} = 40.71$
   - Very Poor (301–400): $\text{MAE} = 31.18$, $\text{RMSE} = 42.16$
   - Severe (401–500): $\text{MAE} = 45.83$, $\text{RMSE} = 61.73$
   - Extreme Hazardous ($>500$): $\text{MAE} = 165.75$, $\text{RMSE} = 232.32$
   *Finding:* Over $96.6\%$ of observations fall in standard bands ($0$ to $400$) where models operate with high accuracy. Overall holdout RMSE is driven by the extreme right tail ($1.5\%$ of observations $>500$).
2. **Spatial Transferability:**
   On the 8 newly commissioned test cities with zero historical training records ($N=2,036$), CatBoost achieves **$\text{RMSE} = 29.23$**, confirming strong spatial generalization.

---

## 7. Mandatory Disclosures & Limitations

1. **Same-Day Estimation vs. Forecasting:** This study evaluates concurrent virtual sensing / surrogate index estimation from same-day pollutant measurements. It is not an early-warning prognostic forecasting model.
2. **Extreme Peak Underestimation:** Because decision tree ensembles perform piece-wise constant averaging within leaves, they underestimate severe episodic spikes exceeding $500$ AQI.
3. **Attribution vs. Causation:** SHAP values represent statistical model attribution within the tree structure and do not prove causal chemical mechanics.
