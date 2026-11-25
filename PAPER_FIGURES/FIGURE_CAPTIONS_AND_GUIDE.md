# Research Paper Figures & Captions Guide
**Project:** Explainable Machine Learning for Air Quality Index Estimation in Indian Cities  
**Institution:** CHRIST (Deemed to be University)  
**Figure Resolution:** High-Resolution Publication Grade (300 DPI)

---

## Tier 1: Essential Main Text Figures

### Figure 1: Model Calibration & Agreement (1:1 Scatter)
- **File:** observed_vs_predicted_scatter.png
- **Recommended Section:** Section 5.1 (Empirical Results — Primary Benchmark)
- **Caption:**  
  *Figure 1: Out-of-time test set calibration (2019–2020, N = 11,410). Observed true AQI versus CatBoost predicted AQI. The red dashed line denotes the ideal 1:1 identity agreement, and the green line indicates the linear regression fit (R² = 0.892, slope = 0.89).*
- **LaTeX Template:**
`latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\linewidth]{observed_vs_predicted_scatter.png}
  \caption{Out-of-time test set calibration (2019--2020,  = 11,410$). Observed true AQI versus CatBoost predicted AQI. The red dashed line denotes ideal 1:1 agreement; the green line indicates empirical linear fit (^2 = 0.892$, slope = .89$).}
  \label{fig:calib_scatter}
\end{figure}
`

---

### Figure 2: Error Performance Across CPCB Health Advisory Bands
- **File:** error_by_aqi_category.png
- **Recommended Section:** Section 5.3 / Section 6 (Comparative Discussion & Policy Relevance)
- **Caption:**  
  *Figure 2: Model error metrics (MAE and RMSE in AQI points) decomposed across official CPCB National Air Quality Index health advisory bands on the holdout test set.*
- **LaTeX Template:**
`latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\linewidth]{error_by_aqi_category.png}
  \caption{Model error metrics (MAE and RMSE in AQI points) decomposed across official CPCB National Air Quality Index health advisory bands on the holdout test set.}
  \label{fig:error_by_category}
\end{figure}
`

---

### Figure 3: TreeSHAP Attribution Summary Beeswarm (Primary Pollutant-Only Model)
- **File:** shap_beeswarm_pollutant_only.png
- **Recommended Section:** Section 8.1 & 8.2 (Explainable AI — Feature Attribution)
- **Caption:**  
  *Figure 3: TreeSHAP summary beeswarm plot for the primary pollutant-only benchmark (N = 11,410 test instances). Features are sorted by mean absolute SHAP value. Points represent individual daily observations colored by normalized concentration (blue = low, red = high). Feature concentrations retain original physical units (µg/m³, mg/m³).*
- **LaTeX Template:**
`latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.85\linewidth]{shap_beeswarm_pollutant_only.png}
  \caption{TreeSHAP summary beeswarm plot for the primary pollutant-only benchmark ( = 11,410$ test instances). Points represent daily observations colored by relative concentration (blue = low, red = high). Physical concentrations are in $\mu\text{g/m}^3$ and $\text{mg/m}^3$.}
  \label{fig:shap_beeswarm_pollutants}
\end{figure}
`

---

### Figure 4: Non-linear Marginal Response Dynamics for PM2.5
- **File:** shap_dependence_pm25.png
- **Recommended Section:** Section 8.3 (Non-linear Response Dynamics)
- **Caption:**  
  *Figure 4: TreeSHAP partial dependence curve for PM2.5 concentration (µg/m³). Vertical spread illustrates interaction effects with co-pollutants. Distinct linear interpolation, steep growth, and hazardous plateau regimes are visible.*
- **LaTeX Template:**
`latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{shap_dependence_pm25.png}
  \caption{TreeSHAP partial dependence curve for $\text{PM}_{2.5}$ concentration ($\mu\text{g/m}^3$). Vertical spread indicates interaction effects with co-pollutants.}
  \label{fig:shap_dependence_pm25}
\end{figure}
`

---

## Tier 2: Diagnostic & Robustness Figures (Main Text or Appendix)

### Figure 5: Prediction Error Distribution & Kurtosis Analysis
- **File:** 
esidual_distribution_histogram.png
- **Recommended Section:** Section 6.2 (Inferential Analysis / Residual Normality)
- **Caption:**  
  *Figure 5: Empirical prediction error distribution (y - y_hat) with fitted Gaussian distribution overlay (µ = 0.52, σ = 39.67), showing zero-centered residuals and heavy-tail kurtosis.*

### Figure 6: Model Robustness Across Macro-Environmental Regimes (2019 vs 2020 Lockdown)
- **File:** error_by_year.png
- **Recommended Section:** Section 7.1 (Temporal Generalization)
- **Caption:**  
  *Figure 6: Model generalization across atmospheric regimes: holdout RMSE across 2019 baseline (N = 7,031) versus 2020 COVID lockdown era (N = 4,379) for all four benchmarked architectures.*

### Figure 7: Heteroscedasticity Analysis (Residuals vs Predicted)
- **File:** 
esiduals_vs_predicted.png
- **Recommended Section:** Methodology Appendix / Diagnostic Diagnostics
- **Caption:**  
  *Figure 7: Residuals versus predicted AQI scores, displaying moving 25th and 75th percentile error envelopes.*

### Figure 8: Spatial City-Aware Attribution Beeswarm (Experiment 2)
- **File:** shap_beeswarm_city_aware.png
- **Recommended Section:** Section 7.2 / Spatial Modeling Appendix
- **Caption:**  
  *Figure 8: TreeSHAP beeswarm plot for Experiment 2 (criteria pollutants with City categorical encoding).*
