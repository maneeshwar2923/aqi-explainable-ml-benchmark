# Explainable Machine Learning for Air Quality Index Estimation in Indian Cities

**Academic Research Repository, Methodological Framework & Interactive Data Story**  
**Institution:** CHRIST (Deemed to be University) — Research Project  
**Study Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  

---

## 1. Research Overview & Methodological Safeguards

This repository hosts the audited, leakage-free machine learning benchmarking suite for estimating the daily Air Quality Index (AQI) across **26 Indian metropolitan and industrial centers** using official Central Pollution Control Board (CPCB) criteria pollutant records from **2015 to 2020** ($N = 24,624$ valid daily observations).

### Methodological Safeguards:
1. **Strict Chronological Partitioning:**
   - **Training Sub-Partition (2015–2017):** 7,556 observations (used for hyperparameter tuning).
   - **Validation Partition (2018):** 5,658 observations (used strictly for early stopping and iteration selection).
   - **Combined Development Partition (2015–2018):** 13,214 observations (used for final model refitting).
   - **Out-of-Time Held-out Test Partition (2019–2020):** 11,410 observations (evaluated strictly once).
2. **Zero Lookahead Imputation:** Feature medians for the 8 criteria pollutants are fit strictly on training partitions and applied to validation and test partitions without future lookahead bias.
3. **Unaltered Target Integrity:** The ground-truth target variable (`AQI`) is **never clipped, scaled, or synthetically truncated**; models are benchmarked against actual real-world atmospheric pollution up to peak episodic values ($>2,000$).
4. **Controlled Dual-Experiment Architecture:**
   - **Experiment 1 (Primary):** Pollutant-only benchmark across Random Forest, XGBoost, LightGBM, and CatBoost on identical physical features (`PM2.5`, `PM10`, `NO2`, `NH3`, `SO2`, `CO`, `O3`, `NO`).
   - **Experiment 2 (City-Aware):** Geographic benchmark assessing spatial transferability across 18 development cities and 8 newly commissioned test cities using algorithm-appropriate encodings.
5. **Defensible Statistical Inference:**
   - Paired observation-level and date-aggregated paired $t$-tests ($T = 548$ days) to adjust for spatial cross-city correlation on identical dates.
   - Holm-Bonferroni family-wise error rate (FWER) step-down multiplicity corrections.
   - 1,000-sample non-parametric bootstrap 95% confidence intervals for $\Delta\text{RMSE}$ and $\Delta\text{MAE}$.
   - Two One-Sided Tests (TOST) against an explicit equivalence margin ($\pm 2.0$ AQI points).
   - Multi-window rolling-origin expanding temporal cross-validation across all development periods.
6. **Game-Theoretic Interpretability (TreeSHAP):**
   - Exact TreeSHAP attributions computed on unscaled physical pollutant units ($\mu\text{g/m}^3$, $\text{mg/m}^3$).
   - Explicit scientific framing of SHAP as statistical model attribution rather than physical causal interventions.

---

## 2. Verified Research Results

All models evaluated on the untouched out-of-time test partition ($N = 11,410$ observations, January 2019 – July 2020) against unaltered observed AQI:

### Naive & Empirical Baselines

| Baseline Architecture | Prediction Rule | RMSE ($\downarrow$) | MAE ($\downarrow$) | $R^2$ Score ($\uparrow$) |
|---|---|:---:|:---:|:---:|
| **Global Training Mean** | Constant historical mean ($190.01$) | 130.3831 | 102.6559 | -0.1720 |
| **City-Stratified Training Mean** | City-specific historical mean (fallback to global) | 107.2215 | 78.4335 | 0.2074 |

### Primary Experiment 1: Pollutant-Only Benchmark (8 Physical Pollutants)

| Model Architecture | RMSE ($\downarrow$) | MAE ($\downarrow$) | $R^2$ Score ($\uparrow$) | Training Latency (s) | Relative Speed vs CatBoost |
|---|:---:|:---:|:---:|:---:|:---:|
| **CatBoost Regressor** (Accuracy Best) | **39.6688** | **19.5689** | **0.8915** | 2.1501 s | $1.0\times$ (Baseline) |
| **LightGBM Regressor** | **40.9889** | **20.2186** | **0.8842** | 1.0180 s | **$2.1\times$ faster** |
| **Random Forest Regressor** | **42.2185** | **20.6647** | **0.8771** | 3.3308 s | $0.6\times$ |
| **XGBoost Regressor** (Speed Best) | **42.5595** | **20.4114** | **0.8751** | **0.4721 s** | **$4.6\times$ faster** |

### Experiment 2: City-Aware Benchmark (8 Pollutants + Encoded City)

| Model Architecture | Categorical Encoding Strategy | RMSE ($\downarrow$) | MAE ($\downarrow$) | $R^2$ Score ($\uparrow$) | Training Latency (s) |
|---|---|:---:|:---:|:---:|:---:|
| **CatBoost** | Native Categorical (Ordered Boosting) | **39.0882** | **19.7629** | **0.8947** | 22.1213 s |
| **LightGBM** | Native Histogram Categorical | **40.5550** | **20.1246** | **0.8866** | 0.9965 s |
| **Random Forest** | TargetEncoder (5-fold cross-validated) | **41.7839** | **20.6991** | **0.8796** | 3.4597 s |
| **XGBoost** | TargetEncoder (5-fold cross-validated) | **42.0344** | **19.9653** | **0.8782** | 0.4039 s |

### Statistical Inference & Equivalence Summary (Exp 1 vs. CatBoost)

| Pairwise Comparison | $\Delta \text{MAE}$ (AQI pts) | 95% Bootstrap CI ($\Delta \text{MAE}$) | 95% Bootstrap CI ($\Delta \text{RMSE}$) | Cohen's $d_z$ | Holm-Adjusted $p$-Value | TOST Equivalence ($\pm 2.0$ pts) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost vs Random Forest** | $+1.0958$ | $[+0.775, +1.427]$ | $[+0.465, +4.849]$ | $0.0633$ | $p < 0.00001$ | **Statistically Equivalent** ($p_{\text{TOST}} < 0.0001$) |
| **CatBoost vs XGBoost** | $+0.8425$ | $[+0.562, +1.131]$ | $[+0.798, +5.005]$ | $0.0554$ | $p < 0.00001$ | **Statistically Equivalent** ($p_{\text{TOST}} < 0.0001$) |
| **CatBoost vs LightGBM** | $+0.6497$ | $[+0.347, +0.946]$ | $[-0.787, +3.557]$ | $0.0400$ | $p = 0.00002$ | **Statistically Equivalent** ($p_{\text{TOST}} < 0.0001$) |

*Operational Note: While differences are statistically discernible due to high test-set power ($N = 11,410$), all standardized effect sizes are negligible ($d_z < 0.07$) and error margins fall strictly within a 2-point equivalence band on a 500-point scale.*

### Global TreeSHAP Attribution Ranking (Experiment 1)

| Rank | Criteria Pollutant | Physical Units | Mean Absolute SHAP (AQI pts) | Relative Attribution Share (%) |
|:---:|---|---|:---:|:---:|
| 🥇 | **PM2.5** | $\mu\text{g/m}^3$ | **50.8072** | **40.98%** |
| 🥈 | **CO** | $\text{mg/m}^3$ | **27.7623** | **22.39%** |
| 🥉 | **PM10** | $\mu\text{g/m}^3$ | **20.9591** | **16.91%** |
| 4 | **NO** | $\mu\text{g/m}^3$ | **7.7998** | **6.29%** |
| 5 | **O3** | $\mu\text{g/m}^3$ | **6.2459** | **5.04%** |
| 6 | **NO2** | $\mu\text{g/m}^3$ | **4.2701** | **3.44%** |
| 7 | **SO2** | $\mu\text{g/m}^3$ | **3.7345** | **3.01%** |
| 8 | **NH3** | $\mu\text{g/m}^3$ | **2.3935** | **1.93%** |

---

## 3. Directory Structure & Repository Architecture

```text
final_analysis/
│
├── OLD_FILES_ARCHIVE/                  # Preserved unrevised legacy files
│   ├── old_scripts/                    # Legacy 01_preprocess.py, 02_train_and_evaluate.py, 03_shap_analysis.py
│   ├── old_results/                    # Legacy results_table.csv, results_summary.txt
│   ├── old_shap_outputs/               # Legacy shap_feature_ranking.csv
│   ├── old_paper_and_docs/             # Legacy FINAL_ANALYSIS.md, PROJECT_SPEC.md, README.md
│   └── old_logs_and_misc/              # Legacy preprocessing_summary.txt
│
├── NEW_ANALYSIS/                       # Audited, self-contained reproducible research suite
│   ├── scripts/                        # Modular analysis scripts (01 to 04 + test_pipeline_qa.py)
│   ├── data_processed/                 # Clean, leakage-free CSV partitions (train_sub, val, train, test)
│   ├── models/                         # Serialized trained model bundles (trained_models.pkl)
│   ├── predictions/                    # Standalone out-of-sample prediction CSVs
│   ├── results/                        # Verified metric tables (results_table.csv, city_wise, year_wise)
│   ├── statistical_analysis/           # Paired tests, TOST equivalence, and rolling-origin CV tables
│   ├── shap_outputs/                   # Publication-grade Beeswarm and Partial Dependence plots & CSVs
│   ├── diagnostics/                    # Observed vs predicted, residual distributions, and AQI band errors
│   ├── reports/                        # JSON and TXT preprocessing and QA audit reports
│   └── logs/                           # Automated execution logs
│
├── manuscript/
│   ├── original/                       # Preserved original paper draft
│   └── revised/                        # Revised, audited academic manuscript
│
├── data/                               # Next.js interactive app data & raw dataset documentation
│   └── README.md
│
├── city_day.csv                        # Raw official CPCB daily monitoring observations (2015-2020)
├── PROJECT_FILE_MANIFEST.csv           # 51-entry complete file inventory & audit classification
├── FINAL_ANALYSIS_AUDIT.md             # Forensic audit report (Phase 1)
├── STATISTICAL_ANALYSIS.md             # Rigorous statistical methodology report (Phase 4)
├── FINAL_QA_REPORT.md                  # Complete quality assurance sign-off report (Phase 8)
├── FINAL_ANALYSIS.md                   # Full research paper and technical documentation
└── README.md                           # This document
```

---

## 4. Complete Reproduction Instructions

### Prerequisites
Python 3.10+ with standard scientific libraries:
```bash
pip install -r requirements.txt
```

### End-to-End Pipeline Reproduction Commands
Execute the revised pipeline sequentially from the workspace root:

```bash
# Step 1: Preprocessing, leakage-free chronological splitting & parameter estimation
python NEW_ANALYSIS/scripts/01_preprocess.py

# Step 2: Multi-model training, early stopping, baselines & statistical inference
python NEW_ANALYSIS/scripts/02_train_and_evaluate.py

# Step 3: TreeSHAP computation, beeswarm and dependence plots
python NEW_ANALYSIS/scripts/03_shap_analysis.py

# Step 4: Predictive diagnostics, residual distributions & category breakdowns
python NEW_ANALYSIS/scripts/04_diagnostics_and_plots.py

# Step 5: Automated 11-assertion quality assurance suite
python NEW_ANALYSIS/scripts/test_pipeline_qa.py
```

All new outputs will be generated in `NEW_ANALYSIS/` with zero overwriting of archived legacy files.

---

## 5. Web Application Quick Start (Interactive Data Story)

To launch the accompanying Next.js 15 interactive presentation layer:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.
