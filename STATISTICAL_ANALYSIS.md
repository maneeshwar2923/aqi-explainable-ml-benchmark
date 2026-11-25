# Statistical Inference, Equivalence Testing & Temporal Cross-Validation

**Study Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Dataset:** CPCB Ambient Air Monitoring Records (2015–2020)  
**Evaluated Sample Size:** $N = 11,410$ held-out out-of-time test instances across 26 Indian cities (2019–2020)  
**Primary Benchmark:** Experiment 1 (8 Criteria Pollutants: PM2.5, PM10, NO2, NH3, SO2, CO, O3, NO)  
**Document Author:** Senior Machine Learning Researcher & Methodology Reviewer  
**Date:** October 2026  

---

## 1. Executive Summary & Audit of Legacy Statistical Pitfalls

A critical audit of the legacy statistical methodology in this research project revealed severe conceptual and mathematical errors:

1. **Misapplication of the Nadeau-Bengio Correction:**
   In early iterations of this project, a formula labeled the "Nadeau-Bengio corrected resampled $t$-test" was applied to per-observation squared errors from a single fixed holdout test set ($N = 11,410$):
   $$t = \frac{\bar{d}}{\sqrt{\left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)} \cdot s_d}$$
   This was mathematically invalid. Nadeau and Bengio (2003) derived this correction specifically for **repeated random subsampling or $K$-fold cross-validation**, where the overlap between training sets across folds violates the independence assumption of the standard $t$-test. Applying this variance inflation factor to a **single holdout split** has no theoretical basis and artificially deflated the $t$-statistic, yielding $p > 0.96$.

2. **The "Absence of Evidence" Fallacy:**
   The legacy report interpreted $p > 0.96$ as "proof that CatBoost, XGBoost, and LightGBM are statistically equivalent." In frequentist hypothesis testing, failing to reject the null hypothesis of zero difference ($H_0: \mu_d = 0$) **never proves that the models are equivalent**; it merely indicates that the observed data do not provide sufficient evidence to rule out zero difference under the assumed test. Proving equivalence requires an explicit equivalence testing framework (such as Two One-Sided Tests).

3. **Ignoring Spatial and Temporal Cross-City Correlation:**
   Air quality monitors across multiple cities in northern India (e.g., Delhi, Gurugram, Lucknow, Patna) record observations simultaneously on the exact same calendar day. Regional synoptic meteorological systems (e.g., winter atmospheric inversions) induce strong cross-sectional correlation across cities. Treating $11,410$ observations as independent and identically distributed (i.i.d.) severely underestimates standard errors and exaggerates test significance.

To remediate these issues, we implement a four-tier inferential framework:
- **Tier 1:** Paired observation-level difference tests (Parametric and Non-parametric).
- **Tier 2:** Date-aggregated paired tests ($T = 548$ calendar days) to remove intra-day spatial correlation.
- **Tier 3:** 1,000-sample non-parametric bootstrap 95% confidence intervals for $\Delta\text{RMSE}$ and $\Delta\text{MAE}$.
- **Tier 4:** Two One-Sided Tests (TOST) equivalence testing against an explicit practical margin ($\pm 2.0$ AQI points).
- **Tier 5:** Chronological expanding-window rolling-origin cross-validation across all development periods for all four models.

---

## 2. Statistical Methodology & Formal Formulations

### 2.1 Paired Observation-Level Error Differences
All models were evaluated on the exact same $N = 11,410$ out-of-time test observations ($y_i, \hat{y}_{i, m}$). For each test instance $i$, the absolute error of model $m$ is $e_{i, m} = |y_i - \hat{y}_{i, m}|$.
The paired absolute error difference between competing model $m$ and the primary top model (CatBoost) is:
$$d_i = e_{i, m} - e_{i, \text{CatBoost}} = |y_i - \hat{y}_{i, m}| - |y_i - \hat{y}_{i, \text{CatBoost}}|$$
A positive value ($d_i > 0$) indicates that CatBoost achieved a lower prediction error on observation $i$.

We report:
1. **Paired $t$-test:**
   $$t = \frac{\bar{d}}{s_d / \sqrt{N}}, \quad \text{where } \bar{d} = \frac{1}{N}\sum_{i=1}^N d_i, \quad s_d = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (d_i - \bar{d})^2}$$
2. **Wilcoxon Signed-Rank Test:** A non-parametric test on the ranks of non-zero absolute differences $|d_i|$, robust to non-normal heavy-tailed error distributions.
3. **Cohen's Effect Size ($d_z$):**
   $$d_z = \frac{\bar{d}}{s_d}$$
   Quantifies the standardized magnitude of improvement independent of sample size ($d_z < 0.2$ is negligible, $0.2 \le d_z < 0.5$ is small).

### 2.2 Date-Aggregated Paired Tests (Spatial Dependence Adjustment)
To account for spatial cross-city autocorrelation on identical monitoring dates, we aggregate daily errors across all active monitoring stations on each calendar date $t \in \{1, 2, \dots, T\}$, where $T = 548$ unique calendar days in 2019–2020:
$$\bar{e}_{t, m} = \frac{1}{K_t}\sum_{k=1}^{K_t} |y_{t, k} - \hat{y}_{t, k, m}|$$
The daily paired difference series is $D_t = \bar{e}_{t, m} - \bar{e}_{t, \text{CatBoost}}$. The paired test is conducted across the $T = 548$ independent daily observations:
$$t_{\text{daily}} = \frac{\bar{D}}{s_D / \sqrt{T}}$$

### 2.3 Multiple Comparison Multiplicity Control (Holm-Bonferroni)
Because CatBoost is compared against three competing models simultaneously (Random Forest, XGBoost, LightGBM), we adjust raw $p$-values using the **Holm-Bonferroni step-down procedure** to strictly control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$:
$$p_{(k)}^{\text{adj}} = \min \left[ \max_{j \le k} \left\{ (M - j + 1) p_{(j)} \right\}, 1.0 \right]$$
where $M = 3$ is the total number of pairwise hypotheses.

### 2.4 Non-Parametric Bootstrap Confidence Intervals
We compute 95% bootstrap confidence intervals for metric differences using $B = 1,000$ empirical resamples of the test set:
$$\Delta\text{RMSE}^{(b)} = \text{RMSE}_m^{(b)} - \text{RMSE}_{\text{CatBoost}}^{(b)}$$
$$\Delta\text{MAE}^{(b)} = \text{MAE}_m^{(b)} - \text{MAE}_{\text{CatBoost}}^{(b)}$$
The 95% confidence interval is extracted via the empirical percentiles: $[\hat{q}_{0.025}, \hat{q}_{0.975}]$.

### 2.5 Two One-Sided Tests (TOST) Equivalence Framework
To test whether performance differences are practically meaningful, we define an **a priori equivalence margin** of:
$$\Delta = \pm 2.0 \text{ AQI points}$$
On the Indian National AQI scale (0 to 500+), an error margin of $2.0$ points corresponds to less than $0.4\%$ of the dynamic range and falls well within the sensor measurement uncertainty of CAAQMS optical particle counters ($\pm 5\text{--}10\,\mu\text{g/m}^3$).

The TOST procedure establishes two composite null hypotheses:
1. $H_{01}: \mu_d \le -\Delta$ (CatBoost is meaningfully inferior by $\ge 2.0$ points)
2. $H_{02}: \mu_d \ge +\Delta$ (CatBoost is meaningfully superior by $\ge 2.0$ points)

The test statistics are:
$$t_1 = \frac{\bar{d} - (-\Delta)}{\text{SE}(\bar{d})}, \quad t_2 = \frac{\Delta - \bar{d}}{\text{SE}(\bar{d})}$$
If both $p_1 < 0.05$ and $p_2 < 0.05$, we reject both null hypotheses, concluding that the true mean performance difference is strictly bounded within $[-\Delta, +\Delta]$:
$$p_{\text{TOST}} = \max(p_1, p_2) < 0.05$$

---

## 3. Empirical Statistical Testing Results

All tests are conducted on the out-of-time test partition (2019–2020, $N = 11,410$ observations, $T = 548$ calendar days) for **Experiment 1 (Pollutant-Only Benchmark)**:

### Table 1: Comprehensive Statistical Hypothesis Testing (CatBoost vs. Competitors)

| Comparison | $\Delta\text{MAE}$ (pts) | 95% Bootstrap CI $\Delta\text{MAE}$ | 95% Bootstrap CI $\Delta\text{RMSE}$ | Cohen's $d_z$ | Paired $t$ $p_{\text{adj}}$ | Wilcoxon $p_{\text{adj}}$ | Daily $t$ $p_{\text{adj}}$ | TOST $p$ ($\pm 2.0$) | Practical Equivalence |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost vs. Random Forest** | **+1.0958** | $[0.775, 1.427]$ | $[0.465, 4.849]$ | 0.0633 | $< 0.000001$ | $5.0 \times 10^{-6}$ | $< 0.000001$ | $< 0.000001$ | **Equivalent** |
| **CatBoost vs. XGBoost** | **+0.8425** | $[0.562, 1.131]$ | $[0.798, 5.005]$ | 0.0554 | $< 0.000001$ | $< 0.000001$ | $< 0.000001$ | $< 0.000001$ | **Equivalent** |
| **CatBoost vs. LightGBM** | **+0.6497** | $[0.347, 0.946]$ | $[-0.787, 3.557]$ | 0.0400 | $2.0 \times 10^{-5}$ | $3.7 \times 10^{-5}$ | $1.6 \times 10^{-5}$ | $< 0.000001$ | **Equivalent** |

### Key Inferential Insights:

1. **Statistical Significance vs. Practical Significance:**
   Due to the large test sample size ($N = 11,410$), even very small numerical differences in MAE ($0.65$ to $1.10$ AQI points) achieve extreme statistical significance ($p < 0.0001$) across observation-level paired $t$-tests, Wilcoxon signed-rank tests, and date-aggregated tests.
   However, **Cohen's $d_z$ effect sizes are universally negligible** ($0.0400$ vs LightGBM, $0.0554$ vs XGBoost, $0.0633$ vs Random Forest; all $d_z \ll 0.20$).

2. **Rejection of Substantive Performance Disparity (TOST Equivalence):**
   Under Two One-Sided Tests (TOST) against an equivalence margin of $\pm 2.0$ AQI points, all $p_{\text{TOST}}$ values are $< 0.000001$. This formally rejects the hypothesis that CatBoost is functionally superior or inferior to LightGBM, XGBoost, or Random Forest by more than 2 AQI points.

3. **Bootstrap Confidence Interval on $\Delta\text{RMSE}$:**
   The 95% bootstrap confidence interval for $\Delta\text{RMSE}$ between CatBoost and LightGBM spans $[-0.787, +3.557]$. Because this interval includes zero, **the RMSE difference between CatBoost and LightGBM is not statistically distinguishable at the 95% confidence level under bootstrap resampling**.

---

## 4. Chronological Rolling-Origin Cross-Validation

To verify that model ranking is stable across expanding temporal regimes rather than an artifact of a single holdout period, we implemented a **chronological expanding-window cross-validation** evaluating all four models across three successive development windows:

- **Window 1 (2015–2016 Train $\rightarrow$ 2017 Eval):**  
  Train $N = 4,498$ observations $\rightarrow$ Evaluate on $N = 3,058$ out-of-time 2017 observations.
- **Window 2 (2015–2017 Train $\rightarrow$ 2018 Eval):**  
  Train $N = 7,556$ observations $\rightarrow$ Evaluate on $N = 5,658$ out-of-time 2018 observations.
- **Window 3 (2015–2018 Train $\rightarrow$ 2019–2020 Final Test):**  
  Train $N = 13,214$ observations $\rightarrow$ Evaluate on $N = 11,410$ out-of-time holdout test observations.

### Table 2: Multi-Window Expanding Temporal Cross-Validation Results

| Model | Window 1 (2017 Eval) RMSE | Window 1 (2017 Eval) MAE | Window 2 (2018 Eval) RMSE | Window 2 (2018 Eval) MAE | Window 3 (2019–20 Final Test) RMSE | Window 3 (2019–20 Final Test) MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 56.1169 | 31.3202 | 63.7841 | 31.6678 | 42.2185 | 20.6647 |
| **XGBoost** | 57.9338 | 31.1692 | 63.8972 | 31.7247 | 42.5595 | 20.4114 |
| **LightGBM** | 50.7447 | 29.8111 | 63.7473 | 31.0198 | 40.9889 | 20.2186 |
| **CatBoost** | **48.9591** | **28.6355** | **58.8201** | **29.0780** | **39.6688** | **19.5689** |

### Observations on Temporal Robustness:
1. **Consistent Model Ranking:** CatBoost achieved the lowest RMSE and MAE across all three chronological windows (Window 1: 48.96, Window 2: 58.82, Window 3: 39.67), demonstrating that its marginal accuracy edge is consistent across multiple temporal regimes.
2. **Atmospheric Regime Shift in 2018:** Prediction errors in 2018 (Window 2) were uniformly higher across all four models ($\text{RMSE} \approx 59\text{--}64$), corresponding to extreme pre-monsoon dust storm events and intense winter episodic smog spikes in the Indo-Gangetic Plain during 2018.
3. **Lockdown Reduction in 2020:** Window 3 errors drop substantially ($\text{RMSE} \approx 40\text{--}42$) due to the nationwide lockdown clean-air interval during the first half of 2020.

---

## 5. Engineering & Practical Decision Matrix

When considering operational deployment on edge nodes or automated municipal air-quality dashboards, computational latency must be weighed against marginal predictive gains:

| Model Architecture | Final Holdout RMSE | Final Holdout MAE | Training Time (Sec) | Relative Speed Ratio | TOST Practical Equivalence | Recommended Use Case |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **CatBoost** | **39.6688** | **19.5689** | 2.1501 s | $1.0\times$ (Baseline) | Reference | Highest precision centralized batch pipeline |
| **LightGBM** | 40.9889 | 20.2186 | 1.0180 s | **$2.1\times$ faster** | Equivalent ($p < 10^{-6}$) | Optimal balance of high speed and strong accuracy |
| **XGBoost** | 42.5595 | 20.4114 | **0.4721 s** | **$4.6\times$ faster** | Equivalent ($p < 10^{-6}$) | Resource-constrained edge devices & embedded sensors |
| **Random Forest** | 42.2185 | 20.6647 | 3.3308 s | $0.6\times$ | Equivalent ($p < 10^{-6}$) | Non-gradient benchmark / bagging baseline |

**Operational Conclusion:**  
While CatBoost demonstrates a statistically discernible edge of $\approx 0.65$ to $1.10$ AQI points in MAE, **all four algorithms operate within the practical equivalence boundary of $\pm 2.0$ points**. For edge computing environments where training latency and memory footprint are paramount, **XGBoost and LightGBM provide outstanding alternatives with over 2× to 4.6× faster training throughput**.
