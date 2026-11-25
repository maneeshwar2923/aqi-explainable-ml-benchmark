# AQI Prediction in India — An Interactive Data Story

A production-quality, cinematic Next.js 15 web application dramatizing an end-to-end air quality prediction data science pipeline for 26 Indian cities.

---

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Production Build
```bash
npm run build
npm run start
```

---

## Key Features & Architecture

* **Framework:** Next.js 15 (App Router, TypeScript)
* **Smooth Inertia Scroll:** `lenis` + `GSAP ScrollTrigger` ticker synchronization
* **Micro-Animations:** `framer-motion` spring dynamics & 3D tilt cards
* **Graphics & Visualizations:**
  * **Canvas2D Smog Particle Engine:** Dynamic particle count based on AQI severity and scroll progress.
  * **Spatial 26-City India Map:** Interactive glowing node distribution across monitoring hubs.
  * **5-Step Preprocessing Pipeline:** Interactive node flow with chronological train/test timeline split (13,214 vs 11,410 observations).
  * **Metric Bar Race:** Benchmark comparisons across Root Mean Squared Error, MAE, and R² scores.
  * **Statistical Significance Dial:** Resampled paired t-test confidence meter.
  * **SHAP Interpretability Suite:** Feature importance bars (PM2.5, CO, PM10, City), Canvas beeswarm plot, and interactive PM2.5 dependence slider.
* **Explain Mode:** Persistent top-right toggle (`?`) that opens plain-language definition sheets for complex statistical terms.
* **Accessibility:** Automatic `prefers-reduced-motion` detection that replaces particle motion with ambient static lighting.

---

## Data Fidelity & Reported vs. Illustrative Status

Per the project specification, this application strictly adheres to honest data encoding rules:

| Component / Metric | Status | Real Value in App | Source |
|---|---|---|---|
| Dataset Scope | **Reported** | 26 Cities, 2015-2020 | `city_day.csv` |
| Train / Test Split | **Reported** | Train: 13,214 (2015-2018) · Test: 11,410 (2019-2020) | `preprocessing_summary.txt` |
| XGBoost Performance | **Reported** | RMSE: 31.52 · MAE: 18.78 · R²: 0.911 | `results_summary.txt` |
| CatBoost Performance | **Reported** | RMSE: 30.58 · MAE: 18.37 · R²: 0.916 (Winner) | `results_summary.txt` |
| Mean \|SHAP\| Importance | **Reported** | PM2.5: 48.63 · CO: 22.81 · PM10: 18.70 · City: 9.06 | `shap_feature_ranking.csv` |
| Random Forest / LightGBM Metrics | **Pending** | Rendered as shimmer skeleton placeholders with badges | Spec §5 / `data/models.ts` |
| SHAP Beeswarm Point Cloud | **Illustrative** | Rendered via Canvas force simulation | Clearly labeled as conceptual |
| SHAP Dependence Slider | **Illustrative** | Reactive non-linear trend simulation | Clearly labeled as simulation |

---

## How to Plug in Real Data Later

All data metrics are strictly typed and centralized in `data/models.ts`. To plug in new numbers from your Python notebooks:

1. Open `data/models.ts`.
2. Change a model's `status` from `'pending'` to `'reported'`.
3. Update `rmse`, `mae`, `r2`, and `trainTimeSec` fields with your exact numbers:
   ```typescript
   {
     id: "lightgbm",
     name: "LightGBM",
     status: "reported",
     rmse: 32.47,
     mae: 19.82,
     r2: 0.906,
     trainTimeSec: 3.83,
     // ...
   }
   ```
4. The UI components (`MetricBarRace`, `ModelCard`) will automatically replace the shimmer skeletons with live animated bars and numbers without needing any code modifications.
