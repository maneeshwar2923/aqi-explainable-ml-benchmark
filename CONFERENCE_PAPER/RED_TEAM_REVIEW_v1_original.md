# IEEE Reviewer Report: Red-Team Evaluation

**Manuscript:** "Explainable Machine Learning for Air Quality Index Estimation in Indian Cities"
**File reviewed:** `AQI IEEE Research Paper - Final.pdf` (14 pp., single column)
**Review date:** 25 Nov 2026
**Reviewer stance:** Adversarial (red team). I tried to break every claim, then wrote it up as an IEEE desk editor (Phase 1) and as a technical referee (Phase 2).

---

## 0. Bottom line

| Stage | Verdict | Main reason |
|---|---|---|
| **Phase 1: Desk screening** | **DESK REJECT (return without review)** | Not in IEEE format, no author or affiliation block, figures appear to be missing (only bracketed captions), broken Table V, figures cited out of order, no related-work section, citation numbers point to the wrong papers. |
| **Phase 2: Technical review** (if it were fixed and resubmitted) | **REJECT** (journal); **Major Revision at best** (regional conference) | The task is to recover a published, deterministic formula (CPCB AQI = max of sub-indices), but the paper never compares against that formula. The stated use case (missing sub-index or station outage) is never tested. The headline ranking is then shown to be practically meaningless by the paper's own TOST. Novelty on this widely used Kaggle dataset is low. |

The authors deserve credit for how careful the writing is: the chronological split, imputation fitted on training data only, effect sizes, TOST, and an unusually honest limitations section. These are strengths and I list them in §4. But being careful about evaluation does not make a contribution. The red-team findings below would sink it with any competent referee.

---

## 1. Phase 1: Desk-rejection screen

An editor-in-chief or associate editor spends 5 to 15 minutes on this check. Each item below is a standard reason for an IEEE desk reject.

### D1. Not in the IEEE template (fatal)
- Single-column Letter page with "Page X of 14" footers and a running header reading **"AQI IEEE Research Paper - Final"**. The PDF metadata shows it was printed from a browser (Creator: Chromium, Producer: Skia/PDF). This is a web document exported to PDF, not an IEEEtran LaTeX or IEEE Word manuscript.
- The byline reads "Oct 16, 2026 . @Nitin sarvesh". That is a document-editor mention tag. It is not an author list, and there are no affiliations, emails, ORCID, or corresponding author.
- Most IEEE venues check the format automatically (PDF eXpress) and would bounce this before an editor sees it.

### D2. Figures missing (fatal)
- All eight figures appear only as bracketed captions, e.g. *"[Figure 1: Out-of-time test set calibration …]"*. I found no embedded image objects in the PDF. If images are present, they are not rendering. In either case the reviewer cannot check Figs. 1–8, yet the Results and SHAP sections depend on them.

### D3. Broken table (serious)
- Table V is raw Markdown pipe syntax run into one paragraph (`| Rank | Feature | Mean |SHAP| … |---|---|`). It cannot be read as a table.

### D4. Figure numbering out of order (serious)
- Figures are first cited in the order 1, 5, 7, 2, 3, 8, 4, 6. IEEE requires figures to be numbered in order of first citation.

### D5. Citations point to the wrong works (serious, and an integrity flag)
| Text claim | Cited | What the reference list actually contains |
|---|---|---|
| "Random Forest builds its ensemble from independent trees **[2]**" | [2] | Chen & Guestrin, *XGBoost* (Breiman is [5]) |
| "XGBoost, LightGBM, and CatBoost … **[3-5]**" | [3]–[5] | LightGBM, CatBoost, **Random Forest** (XGBoost is [2]) |
| "TOST **[8]**" | [8] | Westfall & Young, *Resampling-Based Multiple Testing*. That book is not the TOST source (Schuirmann, 1987). |
| "TreeSHAP **[6]**" (§III-F) vs "TreeSHAP … **[7]**" (§I) | inconsistent | [6] is KernelSHAP/SHAP (2017); [7] is TreeSHAP (2020) |
| Ref. [4] author "A. **Veronika**" | n/a | The co-author is A. **Vorobev** |

Five citation errors in an eight-item list suggests the references were renumbered or generated without being checked. Editors treat this as a quality and integrity warning.

### D6. No related-work section; references too thin (serious)
- Eight references, all of them methods papers (the algorithms, SHAP, the CPCB report). **There is no citation to any prior AQI or air-quality ML study**, even though dozens of published papers use this same `city_day.csv` with RF/XGBoost/LightGBM/CatBoost and SHAP. A desk editor will read this as either unawareness of the field or avoidance of it.
- The dataset itself is not cited (source, URL, version, licence).

### D7. Scope and venue fit
- No venue is named. For IEEE Access, IEEE Sensors J., IEEE TGRS, or IEEE JSTARS, a benchmark of four off-the-shelf models on a Kaggle dataset is below the novelty bar and often desk-rejected for scope alone. A conference (e.g., an IEEE regional or student conference) is a realistic target.

**Phase 1 decision: Return without review.** Fix D1–D6 before resubmitting anywhere.

---

## 2. Phase 2: Red-team technical findings

Severity: **[F] Fatal**, a referee would reject on this alone; **[M] Major**, must be fixed; **[m] Minor**.

### 2.1 Problem formulation and contribution

**[F1] No comparison against the formula the model is trying to learn.**
The CPCB AQI is defined as the **maximum of piecewise-linear sub-indices**. It needs at least three pollutants, one of them PM2.5 or PM10, and uses 24-h means for most pollutants and the 8-h maximum for CO and O3. The paper says this outright ("the model is largely recovering a formula", §VI). The obvious baseline is therefore to **compute the CPCB AQI directly from the available daily concentrations**. Without that baseline:
- R² = 0.89 carries no information. A formula-based baseline may well score higher. If it does, the ML models add nothing. If it doesn't, that gap is the real finding, and it would point to data-processing differences rather than to learned "nonlinear interactions".
- The only baselines given (global mean and city mean) are strawmen. Add the formula baseline, linear regression on the sub-indices, and a max-of-available-sub-indices baseline.

**[F2] The stated use case is never tested.**
The abstract and introduction motivate the work by *"when a monitoring station goes down or a pollutant sub-index is missing"*. But:
- If a **station goes down**, its pollutant readings are gone too, and this model has no inputs. The motivation does not fit the method. That use case would need spatial interpolation from neighbouring stations, satellite AOD, or meteorology.
- For a **missing sub-index**, the paper fills the gap with a training-set median and never measures performance under controlled masking. The experiment the use case calls for is: take test rows with complete pollutant data, mask PM2.5 (or CO, etc.), and compare the model against the formula applied to the remaining pollutants. It is missing.
- "Virtual sensing" in the Index Terms is never demonstrated.

**[F3] The headline result is undercut by the paper's own analysis.**
Contribution 1 is a benchmark ranking (CatBoost > LightGBM > RF > XGBoost). Table III then shows every pairwise MAE difference is equivalent within ±2 AQI points, with Cohen's dz between 0.04 and 0.06. The authors handle this honestly, but the result is "four standard tree ensembles perform the same on a near-deterministic target". That is expected and not publishable as a primary contribution.

**[M1] Low novelty.**
`city_day.csv` (2015–Jul 2020, 26 cities) is one of the most heavily used Kaggle air-quality datasets. Comparing RF, XGB, LGBM, and CatBoost with SHAP on it has been done many times. The claimed novelty (strict chronological split, TOST) is good practice, not new science. A referee will ask what the reader learns that wasn't known.

### 2.2 Data and preprocessing

**[M2] Data provenance is misstated.**
`city_day.csv` with these 16 columns is the community-compiled Kaggle dataset ("Air Quality Data in India 2015–2020"), built from CPCB CCR hourly data. As far as I know, its AQI column was **computed by the compiler** from rolling averages; it is not an official CPCB bulletin value. The QA claim that targets *"match raw CPCB figures"* therefore needs explaining: which raw figures, from where? This matters because any mismatch between the daily means used as inputs and the hourly rolling averages behind the target produces exactly the irreducible residual the paper attributes to "peak events being hard".

**[M3] Unexplained feature selection.**
The file has NOx, Benzene, Toluene and Xylene, all dropped without comment. NO is not a CPCB AQI pollutant, yet it is used as a predictor. Pb and the 8-h CO/O3 aggregates required by the AQI definition are not available. Each choice needs a stated reason.

**[M4] Median imputation is inconsistent with the target.**
The reported AQI was computed from whichever pollutants were present. Filling a missing pollutant with a global training median (for example PM2.5 ≈ 50 µg/m³, the same in Delhi and Shillong) creates an input that does not match the target. Use missingness indicators, per-city or seasonal imputation, or the models' native NaN handling (XGBoost, LightGBM and CatBoost all support it), and report the effect.

**[M5] Outlier cities are not addressed.**
The error distribution has **kurtosis 133.7**, and the "Extreme" band has MAE 165 and RMSE 230. CO is ranked as the 2nd most important feature (22% of SHAP). That is surprising, because under the CPCB formula CO is rarely the dominant sub-index. My suspicion is that a few cities with anomalous CO and AQI records (Ahmedabad is known for this in this dataset) drive both the heavy tail and the CO importance. Report per-city errors for all 26 cities, the frequency of each dominant pollutant according to the formula, and results with and without the anomalous city.

**[M6] The "unseen cities" test is not designed, and its result is not credible.**
The eight unseen cities simply have no data before 2019. They are mostly cleaner (Aizawl, Shillong, Kochi, Ernakulam, Coimbatore), and Kochi and Ernakulam overlap geographically. Lower RMSE on them (29.2 vs 41.6) reflects their lower pollution level, as the authors concede. Use leave-one-city-out or grouped spatial CV, and report scale-free errors (MAPE, or RMSE divided by mean AQI).

### 2.3 Experimental protocol

**[M7] No real hyperparameter search, despite the claim.**
§III-C says "the test partition was not touched at any stage of hyperparameter search", but the only thing tuned is the boosting iteration count (early stopping). RF is explicitly untuned. Learning rate, depth, leaves, regularisation and `n_estimators` are not reported. Ranking models that differ by 0.6–1.1 MAE points without tuning or reported hyperparameters is not a fair benchmark.

**[M8] Only one run; seed variance not measured.**
Results are reported to four decimals from a single seed. Differences between models of about 0.6 MAE may fall within seed-to-seed variation for RF, LightGBM (bagging and feature fraction) and CatBoost. Report mean ± SD over at least 5–10 seeds.

**[M9] Methods are claimed but their results are missing.**
§III-E promises Wilcoxon tests, 95% block-bootstrap CIs and date-aggregated paired comparisons. None of these appear in Table III, which gives only paired-t and TOST. The block length and the bootstrap scheme are not stated. The TOST p-values (< 1e-5) appear to assume independent observations, which the limitations section itself says is wrong.

**[M10] Contradiction over whether the TOST margin was prespecified.**
The Conclusion calls the ±2 margin **"prespecified"**. §VI says it was chosen as *"what was convenient for this study"* and *"has no basis in CPCB guidelines"*. Both cannot be true. If the margin was chosen after seeing the results, the equivalence claim does not hold.

**[M11] The Experiment 2 encoding is described two different ways.**
§III-B says RF **and** XGBoost used 5-fold target encoding. §VI says "target encoding, and standard **one-hot** (others)". The handling of unseen cities in LightGBM's categorical mode is also not described.

**[m1]** Training-time comparisons (e.g., "5.58× faster") are given without hardware, thread count or library versions, so they have no value. CatBoost's time jumps from 1.6 s to 19.9 s in Experiment 2 without explanation.

### 2.4 Results and interpretation

**[M12] Calling Jan–Jul 2020 the "COVID-19 lockdown period" is wrong, and the comparison is confounded.**
India's lockdown began on 25 Mar 2020, so Jan to mid-Mar 2020 is pre-lockdown. The 2020 slice also leaves out Oct–Dec, the high-pollution post-monsoon and winter season. The 2019 vs 2020 comparison therefore mixes three things: season, lockdown, and scale. The paper says it "cannot separate those contributions", but it can: **compare Jan–Jul 2019 with Jan–Jul 2020**, and split 2020 at 25 March.

**[M13] Internal contradictions in the error analysis.**
- Fig. 7 text: residual spread is *widest* at AQI 0–300 and *narrows* at high predicted values. Fig. 2 text: errors *rise monotonically* to MAE 165 in the top band. These can be reconciled (binning by predicted vs observed AQI, with high observed values regressing toward the mean), but the paper never does so. As written, they read as contradictory.
- A fitted slope of 1.05 (Fig. 1) is described alongside "predictions pull short at extremes". Whether that is consistent depends on which variable is on which axis, and that is not stated.
- **"Extreme band"** does not exist. The CPCB bands are Good, Satisfactory, Moderate, Poor, Very Poor and Severe. The text also starts the band analysis at "Satisfactory" and leaves out "Good".

**[M14] Errors in the SHAP interpretation.**
- The plot is called a *"TreeSHAP partial dependence curve"*. It is a SHAP dependence plot, which is a different thing from partial dependence.
- Low regime: *"SHAP values are negative … decreasing sharply as concentrations rise"*. A monotone positive relationship would make them *increase*. The Conclusion then calls this same regime "flat". These are three incompatible descriptions.
- The plateau is attributed to *"AQI capped by the hazardous classification ceiling"*, but the paper says twice that targets were **left uncapped**. Values above 500 exist in this dataset, so the explanation contradicts the method.
- PM2.5, PM10 and CO are strongly correlated. TreeSHAP credit split among correlated features is unstable and depends on whether path-dependent or interventional SHAP is used, and the paper does not say which. The 41/22/17% split should be checked against (a) the formula's dominant-pollutant frequencies and (b) SHAP stability across seeds.
- City Identity has a SHAP range of −20 to +50. That shows the model uses the feature; it does not show the feature adds *meaningful signal*. The MAE got worse.

**[m2]** "CatBoost … followed by LightGBM, then Random Forest and XGBoost": on MAE, XGBoost (20.41) beats RF (20.66). Say which metric the ranking uses.
**[m3]** Four-decimal precision (RMSE 39.6688) is false precision given single-seed variance. Two decimals, with CIs, is enough.
**[m4]** "Global-mean baseline R² = −0.172": explain that this comes from the lower test-set mean (distribution shift). The city-mean baseline's handling of the eight unseen cities is not stated.
**[m5]** The 2020 test set ends on 1 July 2020, but the abstract says "July 2020". Be precise.
**[m6]** No uncertainty quantification. For an advisory use case, prediction intervals (conformal or quantile) are needed, and the paper lists them only as future work.

### 2.5 Reproducibility
- No code, no hyperparameters, no library versions, no seeds, no hardware details. The statement that *"replication … is straightforward"* is not supported. IEEE Access and most IEEE journals now expect a code and data availability statement.

---

## 3. Arithmetic and consistency audit (red-team cross-checks)

I recomputed every number that the paper gives enough information to derive. **All of them are internally consistent**, which is a point in the authors' favour:

| Check | Result |
|---|---|
| 29,531 − 4,681 − 226 = 24,624 | ✔ |
| 7,556 + 5,658 = 13,214; 13,214 + 11,410 = 24,624 | ✔ |
| 2,036 / 11,410 = 17.84% unseen-city share | ✔ |
| Pooled RMSE from 2019 (45.48, n=7,031) and 2020 (27.90, n=4,379) gives 39.67 | ✔ matches Table I |
| Pooled MAE from the yearly MAEs gives 19.569 | ✔ |
| Pooled RMSE from seen (41.59) and unseen (29.23) cities gives 39.67 | ✔ |
| MAE differences in Table III = Table I differences (1.0958, 0.8425, 0.6497) | ✔ |
| Paired t ≈ dz·√n: 0.040·√11,410 ≈ 4.27, so p ≈ 2e-5 (LightGBM) | ✔ matches 0.00002 |
| 1.6232 / 0.2910 = 5.58×; 42.5595 − 39.6688 = 2.89 RMSE | ✔ |
| SHAP shares add up to 100% (40.98+22.39+16.91+6.29+5.04+3.44+3.01+1.93) | ✔ |

There is no sign of fabricated numbers. The problems are in **design and interpretation**, not arithmetic.

---

## 4. Strengths (to include in the report so the authors know what to keep)
1. A strict chronological split, with imputation fitted on training data only. Many published papers on this dataset leak through random K-fold splits; this one does not.
2. Statistical significance is reported separately from practical equivalence (effect sizes plus TOST). This is unusually mature for an applied ML paper.
3. The limitations section is candid: it names the circularity, the lack of forecasting, and the missing meteorology.
4. Results are broken down by year, by city, and by AQI band instead of reporting one aggregate number.
5. The numbers are internally consistent (§3).

---

## 5. What would turn this into an acceptable paper

Roughly in order of impact:
1. **Add the CPCB-formula baseline** (computed from the available daily means) and report how much the ML models gain over it, if anything.
2. **Run the use-case experiment.** Mask one or more pollutants (including the dominant one) on complete test rows. Compare ML, the formula on the remaining pollutants, and simple imputers. This becomes the paper's main contribution: "ML vs formula under missing sub-indices".
3. **Handle missing data properly.** Use native NaN handling or missingness indicators, and report the effect against median imputation.
4. **Audit outliers and data provenance.** Report per-city metrics, check the dominant pollutant (especially CO in anomalous cities), and give a dataset citation with a description of how its AQI was computed.
5. **Make the benchmark fair.** Tune all four models with the same budget (e.g., Optuna on 2018 validation), run multiple seeds, and report every hyperparameter.
6. **Fix the inference.** Use block-bootstrap CIs and TOST on day-level aggregated errors, choose the margin before looking at results and justify it (e.g., half the width of the narrowest CPCB band), and report the promised Wilcoxon and date-aggregated results.
7. **Fix the temporal analysis.** Compare Jan–Jul 2019 with Jan–Jul 2020 and split at 25 Mar 2020.
8. **Fix the SHAP section.** Correct the regime descriptions, state whether SHAP is interventional or path-dependent, compare with the formula's dominant-pollutant frequencies, and drop the "capped ceiling" explanation.
9. **Write a related-work section** with 20–30 references on ML for AQI in India and elsewhere, virtual sensing, and imputation of air-quality data.
10. **Put it in IEEEtran format**: real author block, embedded vector figures numbered in order of citation, a real Table V, fixed citations, and a code and data availability statement.

---

## 6. Summary scorecard (typical IEEE review form)

| Criterion | Score (1–5) | Comment |
|---|---|---|
| Relevance to venue | 3 | Applied ML for environmental monitoring fits IEEE Access or a conference |
| Technical novelty | 1.5 | Standard models on a widely used dataset; no new method or insight |
| Technical soundness | 2.5 | Good leakage control, but no formula baseline, use case untested, untuned models, single seed |
| Experimental validation | 2 | Promised analyses missing; confounded temporal and spatial comparisons |
| Clarity and presentation | 2 | Clear prose, but not an IEEE manuscript: figures missing, broken table, wrong citations |
| References and related work | 1 | 8 references, no domain literature, dataset not cited |
| Reproducibility | 1.5 | No code or hyperparameters |
| **Overall recommendation** | **Reject** (desk reject as submitted) | Resubmission encouraged after §5 items 1–3 and 10 |

**Confidential comments to the editor:** The manuscript reads as a carefully written technical report rather than a research contribution. Five of its eight references are mis-mapped, and the PDF appears to be a browser export of a notes document. I recommend returning it without review. I see no evidence of data fabrication; the reported numbers cross-check consistently.
