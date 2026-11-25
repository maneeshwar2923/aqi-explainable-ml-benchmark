export interface ChapterMeta {
  id: string;
  index: number;
  title: string;
  subtitle: string;
  category: string;
}

export const CHAPTERS: ChapterMeta[] = [
  { id: "problem", index: 1, title: "The Problem", subtitle: "Air Pollution in India", category: "Introduction" },
  { id: "dataset", index: 2, title: "Meet the Dataset", subtitle: "26 Cities, 8 Pollutants, 5.5 Years", category: "Data Science" },
  { id: "preprocessing", index: 3, title: "Preparing the Data", subtitle: "Leakage-Free Cleaning & Splitting", category: "Pipeline" },
  { id: "models", index: 4, title: "Building Prediction Models", subtitle: "Controlled Multi-Model Benchmark", category: "Machine Learning" },
  { id: "comparison", index: 5, title: "Comparing Models", subtitle: "RMSE, MAE & R² Benchmarks", category: "Evaluation" },
  { id: "significance", index: 6, title: "Is Improvement Significant?", subtitle: "Paired Inference & Equivalence", category: "Statistics" },
  { id: "shap", index: 7, title: "Opening the Black Box", subtitle: "SHAP Feature Interpretability", category: "Explainable AI" },
  { id: "insights", index: 8, title: "Key Insights", subtitle: "Atmospheric & Machine Learning Takeaways", category: "Synthesis" },
  { id: "final", index: 9, title: "Final Thoughts", subtitle: "The Complete AQI System", category: "Conclusion" },
];

export const CHAPTER_CONTENT = {
  problem: {
    badge: "Chapter 01",
    title: "AQI Estimation in India",
    subtitle: "A leakage-free, explainable machine learning benchmark for urban air quality index estimation.",
    bullets: [
      { id: "1", text: "Map criteria pollutant concentrations to official CPCB AQI indices.", icon: "Activity" },
      { id: "2", text: "Benchmarking 4 tree ensembles across identical physical inputs.", icon: "ShieldAlert" },
      { id: "3", text: "Rigorously test performance differences with paired statistical inference.", icon: "BellRing" },
      { id: "4", text: "Quantify feature attribution via game-theoretic TreeSHAP interpretability.", icon: "Cpu" },
    ],
  },
  dataset: {
    badge: "Chapter 02",
    title: "The Multi-City Dataset",
    subtitle: "26 Indian cities spanning 2015 to 2020 with daily criteria pollutant observations.",
    stats: [
      { value: 26, label: "Indian Cities", suffix: "" },
      { value: 5.5, label: "Years (2015-2020)", suffix: " yrs" },
      { value: 8, label: "Criteria Pollutants", suffix: " species" },
      { value: 1, label: "Target (AQI)", suffix: " score" },
    ],
    pollutants: [
      { name: "PM2.5", desc: "Fine particulate matter <2.5µm (ug/m3)" },
      { name: "PM10", desc: "Coarse particulate matter <10µm (ug/m3)" },
      { name: "NO2", desc: "Nitrogen Dioxide (ug/m3)" },
      { name: "NH3", desc: "Ammonia (ug/m3)" },
      { name: "SO2", desc: "Sulfur Dioxide (ug/m3)" },
      { name: "CO", desc: "Carbon Monoxide (mg/m3)" },
      { name: "O3", desc: "Ground-level Ozone (ug/m3)" },
      { name: "NO", desc: "Nitric Oxide (ug/m3)" },
    ],
  },
  preprocessing: {
    badge: "Chapter 03",
    title: "Leakage-Free Preprocessing Pipeline",
    subtitle: "Strict chronological partitioning, train-only parameter estimation, and unclipped ground-truth target integrity.",
    steps: [
      { step: 1, title: "Target Cleaning", desc: "Removed 4,681 records missing target AQI and 226 records with >40% missing pollutants." },
      { step: 2, title: "Train-Only Imputation", desc: "Computed column medians strictly on training partitions; applied to validation and test without lookahead leakage." },
      { step: 3, title: "Target Integrity", desc: "Ground-truth AQI is preserved without artificial capping, evaluating models on true observed atmospheric extremes." },
      { step: 4, title: "Chronological Partitioning", desc: "Training (2015–2017: 7,556), Validation (2018: 5,658), Full Dev (13,214), and Final Test (2019–2020: 11,410)." },
      { step: 5, title: "Physical Units Preservation", desc: "Retained raw concentration units (ug/m3, mg/m3) for tree invariance and transparent SHAP explanations." },
    ],
  },
  models: {
    badge: "Chapter 04",
    title: "Model Architecture",
    subtitle: "Comparing ensemble decision trees and advanced gradient boosting algorithms.",
  },
  comparison: {
    badge: "Chapter 05",
    title: "Model Evaluation",
    subtitle: "Rigorously evaluating performance across Root Mean Squared Error, MAE, and R².",
  },
  significance: {
    badge: "Chapter 06",
    title: "Statistical Validation",
    subtitle: "Evaluating paired prediction differences with Holm-Bonferroni correction and TOST equivalence testing.",
  },
  shap: {
    badge: "Chapter 07",
    title: "Model Interpretability (SHAP)",
    subtitle: "Explaining global feature contributions and local pollutant dependencies.",
  },
  insights: {
    badge: "Chapter 08",
    title: "Key Scientific Insights",
    subtitle: "Empirical findings derived from leakage-free benchmarking and TreeSHAP attribution.",
    cards: [
      { title: "PM2.5 Dominance", text: "PM2.5 accounts for ~41% of model attribution (Mean |SHAP| = 50.81 AQI points), serving as the primary index driver.", icon: "Zap" },
      { title: "Combustion Tracer", text: "CO represents the second largest attribution share (~22.4%), acting as a tracer for vehicular and incomplete combustion.", icon: "Flame" },
      { title: "Controlled Benchmarking", text: "In the pollutant-only benchmark, CatBoost achieves RMSE 39.67, LightGBM 40.99, RF 42.22, and XGBoost 42.56.", icon: "Award" },
      { title: "Computational Tradeoff", text: "XGBoost trains in 0.29 seconds—5.6× faster than CatBoost—with an R² difference of only 1.6% (0.875 vs 0.892).", icon: "CheckCircle2" },
      { title: "Spatial Generalization", text: "Models generalize effectively to 8 newly commissioned cities in 2019-2020 (RMSE 29.23 vs 41.59 in historical cities).", icon: "MapPin" },
      { title: "Statistical Parity", text: "Under TOST equivalence testing (+/- 2.0 AQI points), error differences between gradient boosting models fall within practical equivalence margins.", icon: "Calendar" },
    ],
  },
  final: {
    badge: "Chapter 09",
    title: "The Complete Pipeline",
    subtitle: "Synthesizing data science, statistical rigor, and explainable AI for cleaner air.",
    pillars: [
      { title: "Data Preprocessing", desc: "Train-only median imputation & chronological validation" },
      { title: "Machine Learning", desc: "Multi-model benchmarking on identical physical features" },
      { title: "Statistical Validation", desc: "Paired tests with Holm-Bonferroni correction & TOST" },
      { title: "Explainable AI", desc: "TreeSHAP attribution with physical pollutant units" },
    ],
  },
};

export const EXPLANATIONS: Record<string, { title: string; body: string }> = {
  "AQI": {
    title: "Air Quality Index (AQI)",
    body: "A standardized numerical scale (0-500+) formulated by the CPCB as the maximum sub-index of monitored criteria pollutants.",
  },
  "RMSE": {
    title: "Root Mean Squared Error (RMSE)",
    body: "Measures average magnitude of prediction error in AQI units. Heavily penalizes large estimation misses on severe pollution days.",
  },
  "MAE": {
    title: "Mean Absolute Error (MAE)",
    body: "Average absolute difference between actual observed AQI and model predictions. Robust against heavy-tailed atmospheric distributions.",
  },
  "R2": {
    title: "R-squared (R² Score)",
    body: "Proportion of variance in observed AQI explained by model features. Scores around 0.89 indicate that ~89% of index variation is captured.",
  },
  "SHAP": {
    title: "SHAP (SHapley Additive exPlanations)",
    body: "A game-theoretic framework calculating exact Shapley marginal feature attributions for tree ensembles on test instances.",
  },
  "t-Test": {
    title: "Paired Error Difference Testing",
    body: "Statistical hypothesis test evaluating pairwise model error differences on identical out-of-time test instances with Holm-Bonferroni multiplicity correction.",
  },
  "PM2.5": {
    title: "PM2.5 Fine Particles",
    body: "Inhalable particulate matter under 2.5 micrometers in aerodynamic diameter, carrying severe cardiopulmonary health risks.",
  },
};
