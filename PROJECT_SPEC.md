# AQI Prediction in India — Interactive Data Story
## React Native App Specification (for Antigravity build)

---

## 0. High-Concept

A single-scroll, cinematic, chapter-based **data story app** (like an Apple News / Bloomberg interactive feature) that walks the user through an AQI machine-learning project: problem → data → pipeline → models → results → significance → explainability → insights.

Built as **Expo + React Native**, using **Reanimated 3**, **React Native Skia** (for charts/particles), **Moti** (declarative animation), and **React Navigation** for a horizontal chapter-swipe layer on top of vertical scroll-per-chapter.

Tone: dark, moody, data-viz aesthetic — think "environmental dashboard meets editorial longform." Pollutant orange/red gradients against near-black backgrounds, glowing particle smog effects that literally get denser/lighter as AQI numbers change.

---

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | Expo (React Native, TypeScript) | fast iteration, EAS build, easy device testing |
| Animation | `react-native-reanimated` v3 | 60fps UI-thread animations, shared values |
| Micro-interactions | `moti` | declarative wrapper over Reanimated, fast to write |
| Charts / custom viz | `@shopify/react-native-skia` | GPU-accelerated custom drawing (beeswarm, particles, bar race) |
| Gestures | `react-native-gesture-handler` | swipe between chapters, drag on charts |
| Navigation | `react-navigation` (or custom `PagerView`) | chapter pager + deep scroll per chapter |
| Data viz (standard charts) | `victory-native` or custom Skia | bar/line charts for RMSE/MAE/R² comparisons |
| State | `zustand` | lightweight global state (current chapter, theme, "explain mode" toggle) |
| Fonts | `expo-font` + a geometric sans (e.g. Space Grotesk) + a monospace for numbers (e.g. JetBrains Mono) | editorial + data feel |
| Haptics | `expo-haptics` | tap feedback on chapter transitions & reveal moments |

---

## 2. App Structure

```
app/
  _layout.tsx                 // root stack + font loading + splash
  index.tsx                   // Chapter Pager (horizontal swipe entry point)
components/
  chapters/
    Chapter1Problem.tsx
    Chapter2Dataset.tsx
    Chapter3Preprocessing.tsx
    Chapter4Models.tsx
    Chapter5Comparison.tsx
    Chapter6Significance.tsx
    Chapter7SHAP.tsx
    Chapter8Insights.tsx
    FinalThoughts.tsx
  viz/
    SmogParticles.tsx          // Skia particle system, density = f(AQI)
    AQIGauge.tsx                // animated radial gauge (0-500 AQI scale + color bands)
    CityMap.tsx                 // stylized India map, 26 city dots pulse by pollution
    PipelineFlow.tsx            // animated step-by-step data pipeline (steps 1-5 morph in sequence)
    ModelCard.tsx                // flip/tilt card per model (RF/XGB/LightGBM/CatBoost)
    MetricBarRace.tsx            // animated horizontal bar chart RMSE/MAE/R² race across models
    SignificanceMeter.tsx        // p-value / t-test confidence dial
    SHAPBeeswarm.tsx             // Skia-drawn beeswarm plot, points animate in by feature
    SHAPDependencePlot.tsx       // scatter with animated trend line, draggable PM2.5 slider
    FeatureImportanceBars.tsx    // horizontal bars for PM2.5/CO/PM10/City, staggered grow-in
  ui/
    ChapterHeader.tsx
    ScrollProgressRail.tsx       // side rail showing chapter/section progress, tappable
    RevealText.tsx               // word/line-by-word fade+slide-up on scroll into view
    StatCounter.tsx              // animated count-up numbers (e.g. 13,214 / 11,410 / 26 cities)
    Chip.tsx / Tag.tsx
store/
  useStoryStore.ts              // zustand: currentChapter, scrollProgress, explainMode
data/
  content.ts                    // all copy + numbers pulled from the source doc, typed
  models.ts                     // model metric data incl. placeholders for RF/LightGBM (flag as "pending" in UI)
theme/
  colors.ts                     // AQI color bands (Good→Hazardous), dark palette
  typography.ts
```

---

## 3. Navigation & Interaction Model

- **Primary gesture:** vertical scroll *within* a chapter; horizontal swipe (or tap on `ScrollProgressRail`) to jump *between* chapters — like Instagram Stories crossed with a scrollytelling site.
- **ScrollProgressRail**: persistent thin vertical rail on the right edge, 9 segments (8 chapters + final thoughts). Active segment glows; tapping jumps with a smooth `withTiming` cross-fade + particle transition.
- **Explain Mode toggle** (top-right, a small pulsing "?"): when on, tapping any stat/chart shows a bottom-sheet with plain-language explanation (e.g., tapping "R²" explains what it means). Built with `@gorhom/bottom-sheet`.
- **Pull-to-reveal**: at the very top of Chapter 1, a smog-particle background is idle; as the user scrolls down for the first time, particles "clear" revealing the title — sets the emotional tone (bad air → clarity through data).

---

## 4. Chapter-by-Chapter Animation Direction

### Chapter 1 — The Problem
- Full-bleed `SmogParticles` background, dense and slow-drifting, dark red/orange tint.
- Title "AQI Prediction in India" fades/slides in letter-by-letter (`RevealText`, stagger 30ms/char).
- Bullet list ("Monitor pollution trends," "Plan mitigation strategies," etc.) fades in as icons that draw themselves (Skia stroke-animation) one at a time on scroll.
- As user scrolls to bottom of chapter, particles thin out ~40%, foreshadowing "we can predict/control this."

### Chapter 2 — Meet the Dataset
- `CityMap`: stylized outline of India, 26 glowing dots placed roughly by real city coordinates, each pulsing at a different rate (representing different pollution intensity — can be randomized/stylized since city-level breakdown isn't in source data, label as illustrative).
- Tapping a dot shows a small callout: city name + "part of the 26-city dataset."
- Below the map: 4 `StatCounter` tiles that count up on scroll-into-view: **26 cities**, **2015–2020**, **8 pollutants**, **1 target (AQI)**.
- Pollutant feature chips (PM2.5, PM10, NO, NO2, NH3, SO2, CO, O3) fly in from edges and settle into a grid, staggered spring physics.

### Chapter 3 — Preparing the Data
- `PipelineFlow`: a vertical animated pipeline, 5 connected nodes (Remove Incomplete → Median Imputation → Outlier Capping → Chronological Split → Standardization). As user scrolls, a glowing "signal" travels down the connecting line node-to-node (`Reanimated` path animation), each node expanding into a detail card as the signal reaches it.
- Step 4 (Chronological Split) gets special treatment: an animated timeline bar 2015→2020 splits with a wipe animation into Train (2015–2018, 13,214 obs) and Test (2019–2020, 11,410 obs) blocks, with `StatCounter` counting up the observation numbers.
- Step 5 (Standardization): two overlapping bell-curve Skia paths morph — a lumpy "raw" distribution smoothing into a standardized normal curve.

### Chapter 4 — Building Prediction Models
- Four `ModelCard`s (Random Forest, XGBoost, LightGBM, CatBoost) in a horizontal swipeable carousel (or 2x2 grid on tablet).
- Each card: 3D tilt-on-drag (Reanimated `useAnimatedGestureHandler` + perspective transform), front shows model name + icon, back (flip on tap) shows strengths as animated checklist.
- Subtle tree-branching line-art animates in the card background per model (decision-tree motif), looping slowly.

### Chapter 5 — Comparing the Models
- `MetricBarRace`: horizontal bars for RMSE, MAE, R² per model animate growing from 0 to value on scroll-into-view, staggered by model.
- **Data note the app should encode honestly:** source doc has full numbers only for XGBoost (RMSE 31.52, MAE 18.78, R² 0.911) and CatBoost (RMSE 30.58, MAE 18.37, R² 0.916) — Random Forest and LightGBM rows are blank in the source. UI should render RF/LightGBM bars as a "data pending" skeleton/shimmer state rather than inventing numbers, with a small label "not reported in source data."
- CatBoost bar glows gold and gets a small trophy/checkmark micro-animation with haptic tick when it settles as best performer.

### Chapter 6 — Is the Improvement Significant?
- `SignificanceMeter`: an animated dial/needle sweeping to illustrate "statistically significant" vs "could be chance," framed around the corrected resampled paired t-test comparing CatBoost against each competitor.
- Small animated diagram: CatBoost node at center, thin animated connector lines to each other model, each line "pulses" with a checkmark once compared.
- Keep this chapter conceptual/illustrative since the source doc doesn't give actual p-values — label clearly as a conceptual explainer, not a literal result readout.

### Chapter 7 — Opening the Black Box (SHAP)
- `FeatureImportanceBars`: horizontal bars for PM2.5 (48.63), CO (22.81), PM10 (18.70), City (9.06) — grow in with staggered spring, longest (PM2.5) first, in a warm gradient scaled to magnitude.
- `SHAPBeeswarm`: Skia canvas — since this is a placeholder in the source doc, build a *stylized illustrative* beeswarm (clearly labeled "illustrative — plug in real SHAP values from your notebook") with points animating in via a swarm/force-simulation settle animation.
- `SHAPDependencePlot`: interactive — draggable horizontal slider representing a PM2.5 value; scatter points and a trend line redraw/react live as the slider moves, again clearly labeled as illustrative pending real data.

### Chapter 8 — Key Insights
- Six insight cards stack and fan out like cards being dealt, each with an icon, staggered entrance, subtle parallax on scroll.

### Final Thoughts
- Four pillars (Data preprocessing, Machine learning, Statistical validation, Explainable AI) animate in as glowing nodes that converge into a single "pipeline" icon/badge — a visual "so this is the complete system" payoff.
- Closing line fades in slow, particles fully cleared, warm light palette (contrast to Chapter 1's dense red smog) — narrative full-circle.

---

## 5. Data Fidelity Rules (important)

The app must **not fabricate numbers**. Where the source document has real values, use them exactly:

- Cities: 26 · Period: 2015–2020 · Train: 13,214 obs (2015–2018) · Test: 11,410 obs (2019–2020)
- Pollutant features: PM2.5, PM10, NO, NO2, NH3, SO2, CO, O3
- XGBoost: RMSE 31.52, MAE 18.78, R² 0.911
- CatBoost: RMSE 30.58, MAE 18.37, R² 0.916 (best)
- SHAP mean |value|: PM2.5 48.63, CO 22.81, PM10 18.70, City 9.06

Where the source has **no value** (Random Forest / LightGBM metrics, SHAP beeswarm/dependence plots, significance test p-values), the app should:
- Render an explicit "data pending / illustrative" state (shimmer skeleton or a small badge), and
- Keep a clearly typed `data/models.ts` field like `status: 'reported' | 'pending'` so it's trivial to plug in real numbers later without touching animation code.

---

## 6. Suggested Build Order (for the agent)

1. Scaffold Expo TS app, install deps (reanimated, skia, moti, gesture-handler, zustand, bottom-sheet, expo-font).
2. Theme + typography + `data/content.ts` with all copy/numbers above.
3. `ScrollProgressRail` + chapter pager shell with dummy chapters (get navigation feel right first).
4. Build `SmogParticles` (Skia) — reusable, density prop-driven.
5. Build Chapters 1–3 (highest content density, sets visual language).
6. Build `ModelCard`, `MetricBarRace` for Chapters 4–5.
7. Build `SignificanceMeter` for Chapter 6.
8. Build SHAP components for Chapter 7 (mark illustrative).
9. Chapter 8 + Final Thoughts.
10. Polish pass: haptics, transition easing curves, dark/light toggle (optional), accessibility (reduce-motion fallback that swaps animations for simple fades).

---

## 7. Accessibility / Performance Notes

- Respect `AccessibilityInfo.isReduceMotionEnabled()` — provide a low-motion mode (simple fades, no particle systems).
- Keep Skia particle counts adaptive to device (lower count on older/Android low-end via `Platform`/`Device` checks).
- All `StatCounter` and chart animations should run once per scroll-into-view (use `useSharedValue` + intersection-style trigger via `onLayout`/`InteractionManager`, not on every re-render).
