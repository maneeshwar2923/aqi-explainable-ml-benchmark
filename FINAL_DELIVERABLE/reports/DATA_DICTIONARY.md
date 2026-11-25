# Research Data Dictionary & Schema Specification

**Study Title:** Explainable Machine Learning for Air Quality Index Prediction in Indian Cities: A Comparative Benchmarking Study  
**Dataset Origin:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India  
**Continuous Ambient Air Quality Monitoring Stations (CAAQMS):** 26 Urban Centers  
**Temporal Span:** January 1, 2015 – July 1, 2020  
**Raw Dimensions:** 29,531 daily records × 16 attributes  
**Retained Valid Sample:** 24,624 daily records across 26 cities  

---

## 1. Target Variable Specification

| Variable Name | Data Type | Physical Units | Valid Dynamic Range | Regulatory Description | Missingness Handling | Outlier Treatment |
|---|---|:---:|:---:|---|---|---|
| **`AQI`** | Continuous Float / Int | Index Points (Unitless) | 0 to 500+ (Max observed: 2,049) | Composite daily Air Quality Index defined by Indian NAQI guidelines: $\text{AQI} = \max_i \{ I_i(C_i) \}$. | All 4,681 rows lacking target AQI dropped. Target is **never imputed**. | **Strictly Unaltered.** Zero synthetic clipping or capping. Models are benchmarked on real observed concentrations. |

### Statutory CPCB AQI Health Advisory Categories:
- **Good (0–50):** Minimal impact on respiratory health.
- **Satisfactory (51–100):** Minor breathing discomfort to sensitive individuals.
- **Moderate (101–200):** Breathing discomfort to people with lung, asthma, and heart diseases.
- **Poor (201–300):** Breathing discomfort to most people on prolonged exposure.
- **Very Poor (301–400):** Respiratory illness on prolonged exposure.
- **Severe (401–500+):** Affects healthy people and seriously impacts those with existing diseases.

---

## 2. Criteria Pollutant Explanatory Features

The 8 criteria pollutants monitored under CPCB guidelines serve as input features:

| Feature Name | Chemical Species | Data Type | Standard Metric Units | Physical Emission Sources | Physical Role in Atmosphere | Imputation Strategy (Leakage-Free) |
|---|---|---|:---:|---|---|---|
| **`PM2.5`** | Fine Particulate Matter ($< 2.5\,\mu\text{m}$) | Continuous Float | $\mu\text{g/m}^3$ | Vehicle exhaust, stubble biomass burning, coal power generation | Penetrates deep alveolar lung capillaries; dominant optical scatterer | Train-only median ($57.85\,\mu\text{g/m}^3$) |
| **`PM10`** | Coarse Inhalable Particles ($< 10\,\mu\text{m}$) | Continuous Float | $\mu\text{g/m}^3$ | Road dust resuspension, construction debris, mechanical wear | Upper respiratory tract irritant; thoracic fraction | Train-only median ($108.41\,\mu\text{g/m}^3$) |
| **`NO2`** | Nitrogen Dioxide | Continuous Float | $\mu\text{g/m}^3$ | High-temperature fossil fuel combustion, heavy diesel engines | Precursor to photochemical ground-level ozone and aerosol nitrates | Train-only median ($25.36\,\mu\text{g/m}^3$) |
| **`NH3`** | Ammonia | Continuous Float | $\mu\text{g/m}^3$ | Fertilizer application, agricultural livestock waste, untreated sewage | Neutralizes ambient sulfuric and nitric acid to form secondary aerosol salts | Train-only median ($21.20\,\mu\text{g/m}^3$) |
| **`SO2`** | Sulfur Dioxide | Continuous Float | $\mu\text{g/m}^3$ | Coal-fired thermal power plants, industrial oil refining | Potent bronchoconstrictor; precursor to acid precipitation | Train-only median ($8.44\,\mu\text{g/m}^3$) |
| **`CO`** | Carbon Monoxide | Continuous Float | $\text{mg/m}^3$ | Incomplete hydrocarbon combustion, slow-moving vehicular traffic | Systemic asphyxiant; robust primary tracer of vehicle tailpipes | Train-only median ($1.02\,\text{mg/m}^3$) |
| **`O3`** | Ground-level Ozone | Continuous Float | $\mu\text{g/m}^3$ | Secondary photochemistry: $\text{NO}_x + \text{VOC} + h\nu$ | Strong pulmonary oxidant peaking during sunny afternoon hours | Train-only median ($31.30\,\mu\text{g/m}^3$) |
| **`NO`** | Nitric Oxide | Continuous Float | $\mu\text{g/m}^3$ | Fresh tailpipe emissions from internal combustion engines | Rapidly titrates with ambient ozone to produce secondary $\text{NO}_2$ | Train-only median ($10.21\,\mu\text{g/m}^3$) |

---

## 3. Metadata & Contextual Identifiers

| Variable Name | Data Type | Permissible Values | Role in Experiment 1 | Role in Experiment 2 |
|---|---|---|---|---|
| **`City`** | Categorical String | 26 distinct Indian cities | **Excluded** (ensures strict physical criteria pollutant benchmark) | **Included** (encoded via CatBoost/LightGBM native categorical or TargetEncoder cv=5) |
| **`Date`** | Temporal (YYYY-MM-DD) | 2015-01-01 to 2020-07-01 | Used strictly for chronological partitioning and temporal day aggregation | Used strictly for chronological partitioning and temporal day aggregation |

---

## 4. Chronological Partitioning & Sample Distribution

Data partitioning was executed chronologically before estimating any summary statistics:

| Partition Name | Calendar Date Range | Total Rows | Target Role in Pipeline | Active City Count |
|---|---|:---:|---|:---:|
| **`train_sub.csv`** | 2015-01-01 to 2017-12-31 | **7,556** | Hyperparameter exploration & early stopping training fold | 18 |
| **`val.csv`** | 2018-01-01 to 2018-12-31 | **5,658** | Validation early stopping fold (stops boosting before test set) | 18 |
| **`train.csv` (Full Dev)** | 2015-01-01 to 2018-12-31 | **13,214** | Final refitting partition using validated iteration counts | 18 |
| **`test.csv` (Holdout)** | 2019-01-01 to 2020-07-01 | **11,410** | Single-pass out-of-time evaluation of finalized models | **26** |

### Spatial Distribution Across Splits:
- **Historical Development Cities (18 Cities):** Ahmedabad, Amaravati, Amritsar, Bengaluru, Brajrajnagar, Chennai, Delhi, Gurugram, Hyderabad, Jaipur, Jorapokhar, Kolkata, Lucknow, Mumbai, Patna, Talcher, Thiruvananthapuram, Visakhapatnam.
- **Newly Commissioned Test Cities (8 Cities, 2,036 rows / 17.84% of test set):** Aizawl, Bhopal, Chandigarh, Coimbatore, Ernakulam, Guwahati, Kochi, Shillong.
