# Red-Team Review: Revised Conference Version

**Manuscript:** "When Does Machine Learning Beat the Formula? Reconstructing India's Air Quality Index Under Analyser Outages" (`paper.tex`)
**Reviewed against:** the first review of the original manuscript (`RED_TEAM_REVIEW_v1_original.md`)
**Stance:** Adversarial IEEE reviewer, desk stage (Phase 1) and technical stage (Phase 2).

## Bottom line

| Stage | Original paper | Revised paper |
|---|---|---|
| Phase 1, desk | Desk reject | **Pass**, once compiled and the items in §3 are filled in |
| Phase 2, technical | Reject | **Weak accept / accept with minor revisions** at a mid-tier IEEE conference (regional conference, IEEE Access-adjacent workshop, an environmental-informatics track) |

The paper now makes one claim it can defend: ML adds little over the CPCB formula when every analyser reports, and adds a lot when the analyser that sets the index fails. That claim rests on a data audit, a strong deterministic baseline, a tuned multi-seed benchmark, bootstrap intervals, and a sensitivity analysis.

---

## 1. How each original fatal or major issue was resolved

| Original finding | Status | Where |
|---|---|---|
| D1 not IEEE format, D2 figures missing, D3 broken table, D4 figure order | **Fixed**: IEEEtran conference class, 3 vector figures numbered in citation order, booktabs tables | `paper.tex` |
| D5 mis-mapped citations, D6 8 references and no related work | **Fixed**: 23 BibTeX references, cited via keys so the numbering cannot drift; a related-work section | `references.bib`, §II |
| F1 no formula baseline | **Fixed**: B1/B2/B3; the two-day formula beats every same-day model | Table I |
| F2 stated use case never tested | **Fixed**: outage experiment with 1/3/7-day outages, analyser-group masking, stale-aware model | §V-C, Table III, Fig. 2 |
| F3 headline ranking erased by TOST | **Removed**: TOST dropped; reports bootstrap CIs; the ranking is no longer the contribution | §IV-D |
| M1 low novelty | **Improved**: the data audit and the formula-vs-ML-under-outage question are new for this dataset | §I |
| M2 data provenance | **Partly**: dataset cited as a Kaggle compilation; the claim "AQI derived from hourly data" should be checked against the Kaggle page (see §3) | §III-A |
| M3 feature choice | **Fixed**: NOx added (the formula prefers it), BTX excluded because it is not an AQI pollutant | §IV-B |
| M4 median imputation | **Fixed**: native NaN handling | §IV-B |
| M5 outlier cities | **Fixed, and now the strongest finding**: Ahmedabad CO = NO in 94% of rows | §III-B, Fig. 1 |
| M6 unseen cities | **Reported honestly**: ML worse than the formula on unseen cities | Table II |
| M7 no tuning, M8 single seed | **Fixed**: 20-trial random search per model with an equal budget; 5 seeds | §IV-B, Table I |
| M9 analyses promised but missing | **Fixed**: every method mentioned has a result | n/a |
| M10 margin contradiction, M11 encoding contradiction | **Removed**: TOST and the city-aware experiment are gone | n/a |
| M12 lockdown confound | **Fixed**: comparisons over the same calendar window, split at 25 Mar | Table II |
| M13 error-analysis contradictions, M14 SHAP errors | **Fixed**: SHAP checked against the formula's dominant pollutant (the ground truth); wrong regime claims removed | §V-D, Fig. 3 |

## 2. Red-team checks run on the revised paper

Every number in `paper.tex` was traced to an output file in `NEW_ANALYSIS/`. Issues found and fixed during this pass:

1. **Overclaim:** Table I originally bolded CatBoost for RMSE. The two-day formula actually has the best RMSE (18.31). Fixed, and the text now says ML does not beat the formula on RMSE.
2. **Overclaim:** "ML is worse on unseen cities" was written about all models; it was tested only for the best one. Fixed.
3. **Inference stated as fact:** "the AQI was computed from these columns" was rewritten as "under the formula, CO would set …; AQI reaches 2,049 vs ≤956 elsewhere".
4. **Weak reference:** The first outage comparison used "formula + last reading". For outages of 3 or more days, dropping the analyser is better, so ML is now compared with the better of the two formula strategies, chosen on the test data. That choice favours the formula.
5. **Unreported loss:** Random Forest-stale *loses* by 1.37 points when PM2.5 is out for 7 days. This is now in the text.
6. **The headline depends on excluding Ahmedabad:** with Ahmedabad, same-day ML beats the formula. A sensitivity paragraph now shows that the entire reversal comes from Ahmedabad's 528 rows; on the other 25 cities the formula still wins.
7. **SHAP caveat:** PM10 sets the index on 48.7% of days but gets 20.7% of the SHAP attribution. This is reported as a finding about SHAP, not hidden.
8. **Two numbers drafted from memory were wrong and corrected:** the development + test row count (23,297) and the count of test rows with complete inputs (8,003).
9. **Reproducibility:** rerunning the outage experiment reproduced every MAE to within 5e-15.

## 3. Remaining risks a reviewer could raise (none fatal)

| Risk | Severity | Mitigation already in the paper / what you can add |
|---|---|---|
| Excluding Ahmedabad was decided after looking at the data | Medium | Objective criterion (CO = NO equality) plus a full sensitivity rerun (§V-D). |
| Outages are simulated, not real station failures | Medium | Stated as limitation (ii). Best upgrade: validate on real gaps in `station_day.csv`. |
| The stale model is trained on the same outage lengths it is tested on | Low–Medium | Stated as a limitation. An optional extra run would train on {1,3,7} and test at 5 and 14 days. |
| The dominant-analyser scenario is an oracle stress test | Low | Stated as limitation (iii); single-analyser scenarios provide the average-case view. |
| 18 outage cells reported without multiplicity correction | Low | The headline cells have CIs far from zero (for example [−6.88, −5.77]). |
| Single test window (2019 to mid-2020) | Low | Stated as limitation (iv). |
| Daily means rather than hourly data | Low | Stated as limitation (i). |
| Only 20 tuning trials per model | Low | Equal budget per model; seed SD ≤ 0.16 is far below the gaps the claims rest on. |

## 4. Before you submit: required actions

1. **Compile on Overleaf** (no LaTeX is installed on this machine, so the PDF has not been built). Upload `paper.tex`, `references.bib` and `figures/`. Check that the paper fits the venue's page limit; it is about 6 pages in IEEE two-column. If it runs over, the easiest cuts are Table II's band rows and limitation (vii).
2. **Verify two references** marked `VERIFY BEFORE SUBMISSION` in `references.bib` (Kumar & Pande 2023; Natarajan et al. 2024). They were written from memory. Replace them with papers you have read if they don't match.
3. **Check the Kaggle dataset page** to confirm its AQI is computed from hourly data with 24-h and 8-h windows (§III-A). If the page says otherwise, adjust that sentence.
4. **Author block:** keep it anonymous for double-blind review; fill in names and affiliations for the camera-ready version.
5. **Code link:** replace the red `[anonymised repository URL]` (for example an anonymous.4open.science link to the repository with `NEW_ANALYSIS/scripts/05–10`).
6. **IEEE PDF eXpress** check if the venue requires it.

## 5. Verdict

> **Recommendation: Accept with minor revisions.** A careful, honest study whose main contributions (a dataset defect affecting a widely used benchmark, a strong formula baseline that same-day ML fails to beat, and evidence of where ML does help) are useful to the applied community. The claims are scoped, the negative results are reported, and the analysis is reproducible.
