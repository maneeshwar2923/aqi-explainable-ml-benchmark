export interface ModelMetric {
  id: string;
  name: string;
  category: string;
  status: 'reported' | 'pending' | 'illustrative';
  rmse: number | null;
  mae: number | null;
  r2: number | null;
  trainTimeSec?: number | null;
  isBest?: boolean;
  description: string;
  strengths: string[];
}

export const MODELS_DATA: ModelMetric[] = [
  {
    id: "catboost",
    name: "CatBoost",
    category: "Symmetric Oblivious Trees",
    status: "reported",
    rmse: 39.67,
    mae: 19.57,
    r2: 0.892,
    trainTimeSec: 1.62,
    isBest: true,
    description: "Ordered boosting with symmetric oblivious decision trees evaluated on uncapped true AQI.",
    strengths: [
      "Lowest holdout RMSE (39.67) and MAE (19.57) in primary pollutant benchmark",
      "Ordered boosting minimizes target shift across chronological splits",
      "Tuned via 2018 validation early stopping (275 iterations)",
    ],
  },
  {
    id: "lightgbm",
    name: "LightGBM",
    category: "Leaf-wise Gradient Boosting",
    status: "reported",
    rmse: 40.99,
    mae: 20.22,
    r2: 0.884,
    trainTimeSec: 0.57,
    isBest: false,
    description: "Highly efficient histogram-based gradient boosting using leaf-wise growth.",
    strengths: [
      "Second-ranked overall accuracy (RMSE: 40.99, R²: 0.884)",
      "Exceptional computational efficiency (0.57s training latency)",
      "Native categorical handling in Experiment 2 (RMSE: 40.55)",
    ],
  },
  {
    id: "rf",
    name: "Random Forest",
    category: "Bagged Decision Trees",
    status: "reported",
    rmse: 42.22,
    mae: 20.66,
    r2: 0.877,
    trainTimeSec: 4.08,
    isBest: false,
    description: "Traditional bagging ensemble of 200 variance-reducing decision trees.",
    strengths: [
      "Robust non-linear mapping without gradient descent step-tuning",
      "Competitive R² score of 0.877 on uncapped holdout observations",
      "Serves as the non-boosting baseline architecture",
    ],
  },
  {
    id: "xgboost",
    name: "XGBoost",
    category: "Extreme Gradient Boosting",
    status: "reported",
    rmse: 42.56,
    mae: 20.41,
    r2: 0.875,
    trainTimeSec: 0.29,
    isBest: false,
    description: "Second-order gradient tree boosting with L1/L2 regularization and validation early stopping.",
    strengths: [
      "Fastest training latency (0.29 seconds, 5.6× faster than CatBoost)",
      "Strict early stopping on 2018 validation set (best iteration: 79)",
      "Optimal engineering compromise for high-throughput edge nodes",
    ],
  },
];

export interface FeatureImportance {
  feature: string;
  label: string;
  meanAbsShap: number;
  status: 'reported' | 'pending';
  desc: string;
}

export const SHAP_FEATURE_IMPORTANCE: FeatureImportance[] = [
  { feature: "PM2.5", label: "PM2.5 (ug/m3)", meanAbsShap: 50.81, status: "reported", desc: "Dominant primary driver explaining 40.98% of model attribution." },
  { feature: "CO", label: "CO (mg/m3)", meanAbsShap: 27.76, status: "reported", desc: "Combustion tracer from vehicular and industrial sources (22.39%)." },
  { feature: "PM10", label: "PM10 (ug/m3)", meanAbsShap: 20.96, status: "reported", desc: "Coarse inhalable particulate matter from road dust and construction (16.91%)." },
  { feature: "NO", label: "NO (ug/m3)", meanAbsShap: 7.80, status: "reported", desc: "Primary vehicular tailpipe combustion byproduct (6.29%)." },
  { feature: "O3", label: "O3 (ug/m3)", meanAbsShap: 6.25, status: "reported", desc: "Secondary photochemical ground-level ozone oxidizer (5.04%)." },
  { feature: "City", label: "City Identity (Exp 2)", meanAbsShap: 7.61, status: "reported", desc: "Metropolitan geographical baseline and local airshed variation." },
];
