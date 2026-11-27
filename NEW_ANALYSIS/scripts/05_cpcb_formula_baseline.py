"""
NEW_ANALYSIS/scripts/05_cpcb_formula_baseline.py
------------------------------------------------
Deterministic CPCB-formula baselines (reviewer item F1).

The CPCB National AQI is the maximum of piecewise-linear pollutant sub-indices,
valid only when >= 3 pollutants are available and at least one of them is
PM2.5 or PM10. This script computes that formula directly from the daily mean
concentrations in city_day.csv and benchmarks it against the ML models on the
exact same out-of-time test rows (2019-01-01 .. 2020-07-01).

Baselines produced
  F1  CPCB formula    : max sub-index over the available {PM2.5, PM10, NO2, SO2, NH3, CO, O3}
                         on raw (non-imputed) inputs. Metrics are reported on all rows and on the
                         CPCB-valid subset (>=3 pollutants incl. PM2.5 or PM10).
  F2  CPCB-NOx         : as F1 but NOx (not NO2) feeds the nitrogen sub-index, matching the
                         public Kaggle AQI-calculation tutorial for this dataset.
  F3  CPCB-imputed     : as F1 but on the median-imputed inputs the ML models were given.
  F4  Linear-on-SI     : OLS on the 7 sub-indices + their max + missingness flags,
                         fit on the 2015-2018 development partition only.
  F5  Recalibrated     : AQI ~ a * F1 + b (two parameters), fit on 2015-2018 only. Absorbs the
                         systematic low bias of daily means vs the 8-h maxima / rolling means.
  F6  Two-day formula  : CPCB formula on (1-w)*today + w*yesterday for every pollutant (yesterday =
                         same city, previous calendar day; today's value if absent). w is chosen on
                         2015-2018 by grid search. Tests whether the target's averaging window spans
                         the previous day, which same-day inputs cannot see.

Limitations (stated in the output): city_day.csv only has daily means, whereas CPCB uses
8-h maxima for CO and O3 and 24-h rolling means for the rest, so F1 cannot reproduce the
target exactly. The size of that gap is itself the result.

Outputs (NEW_ANALYSIS/formula_baseline/)
  formula_baseline_results.csv     metrics for F1-F4 next to the four ML models
  formula_vs_ml_paired.csv         paired MAE/RMSE differences vs CatBoost with date-block bootstrap CIs
  formula_city_wise.csv            per-city error of F1 and CatBoost (all test cities)
  dominant_pollutant_frequency.csv share of test rows where each pollutant gives the max sub-index
  formula_predictions.csv          row-level F1-F4 predictions aligned with the ML predictions
  formula_baseline_summary.txt     human-readable summary
"""

import os
import argparse
import numpy as np
import pandas as pd

# Concentration breakpoints (CPCB National AQI, 2014). Sub-index breakpoints are
# 0, 50, 100, 200, 300, 400 for every pollutant. Above the last breakpoint the
# sub-index is extended linearly with the slope of the "Very Poor" segment, so
# values > 500 are possible (the dataset's AQI column is likewise uncapped).
BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250],
    "PM10":  [0, 50, 100, 250, 350, 430],
    "NO2":   [0, 40, 80, 180, 280, 400],
    "SO2":   [0, 40, 80, 380, 800, 1600],
    "NH3":   [0, 200, 400, 800, 1200, 1800],
    "CO":    [0, 1.0, 2.0, 10, 17, 34],      # mg/m3
    "O3":    [0, 50, 100, 168, 208, 748],
}
SI_LEVELS = [0, 50, 100, 200, 300, 400]
AQI_POLLUTANTS = list(BREAKPOINTS.keys())
ML_POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "SO2", "CO", "O3", "NO"]
ML_MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
TEST_START, TEST_END = "2019-01-01", "2020-07-01"


def sub_index(conc, pollutant):
    """Vectorised CPCB sub-index; NaN in -> NaN out."""
    bp = np.asarray(BREAKPOINTS[pollutant], dtype=float)
    si = np.asarray(SI_LEVELS, dtype=float)
    # Extend one virtual segment past the top so np.interp extrapolates with the Very Poor slope.
    top_c = bp[-1] + (bp[-1] - bp[-2])
    bp_ext = np.append(bp, top_c)
    si_ext = np.append(si, 500.0)
    x = np.asarray(conc, dtype=float)
    out = np.interp(x, bp_ext, si_ext)
    slope = 100.0 / (bp[-1] - bp[-2])
    over = x > top_c
    out[over] = 500.0 + (x[over] - top_c) * slope
    out[np.isnan(x)] = np.nan
    return out


def formula_aqi(df, nitrogen_col="NO2"):
    """Returns (aqi, dominant_pollutant, valid_mask)."""
    cols = {}
    for p in AQI_POLLUTANTS:
        src = nitrogen_col if p == "NO2" else p
        cols[p] = sub_index(df[src].values, p)
    si = pd.DataFrame(cols, index=df.index)
    n_avail = si.notna().sum(axis=1)
    has_pm = si["PM2.5"].notna() | si["PM10"].notna()
    valid = (n_avail >= 3) & has_pm
    aqi = si.max(axis=1, skipna=True)
    dom = si.fillna(-1.0).idxmax(axis=1).where(n_avail > 0)
    return aqi.values, dom.values, valid.values, si


def locf_fill(rows, raw, pollutant, days=7):
    """Most recent reading of `pollutant` in the same city strictly before the row's date,
    within `days` days. Uses only past data. Returns array with NaN where none exists."""
    left = rows[["City", "Date"]].reset_index().sort_values("Date")
    right = raw.loc[raw[pollutant].notna(), ["City", "Date", pollutant]].sort_values("Date")
    m = pd.merge_asof(left, right, on="Date", by="City", direction="backward",
                      allow_exact_matches=False, tolerance=pd.Timedelta(days=days))
    return m.set_index("index").loc[rows.index, pollutant].values


def metrics(y, p):
    e = p - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    return rmse, mae, r2


def block_bootstrap_diff(y, p_ref, p_cmp, dates, n_boot=1000, seed=42):
    """Date-block bootstrap: resample whole days (all cities on a day move together)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(dates)
    n_days = len(uniq)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_days + 1))
    ae_r, ae_c = np.abs(p_ref - y)[order], np.abs(p_cmp - y)[order]
    se_r, se_c = ((p_ref - y) ** 2)[order], ((p_cmp - y) ** 2)[order]
    # Per-day sums and counts
    cnt = np.diff(bounds)
    sum_ae_r = np.add.reduceat(ae_r, bounds[:-1]); sum_ae_c = np.add.reduceat(ae_c, bounds[:-1])
    sum_se_r = np.add.reduceat(se_r, bounds[:-1]); sum_se_c = np.add.reduceat(se_c, bounds[:-1])
    d_mae, d_rmse = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_days, n_days)
        n = cnt[idx].sum()
        d_mae[b] = (sum_ae_c[idx].sum() - sum_ae_r[idx].sum()) / n
        d_rmse[b] = np.sqrt(sum_se_c[idx].sum() / n) - np.sqrt(sum_se_r[idx].sum() / n)
    return np.percentile(d_mae, [2.5, 97.5]), np.percentile(d_rmse, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap.add_argument("--raw-file", default=os.path.join(root, "city_day.csv"))
    ap.add_argument("--analysis-dir", default=os.path.abspath(os.path.join(here, "..")))
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.analysis_dir, "formula_baseline")
    os.makedirs(out_dir, exist_ok=True)

    # --- Reproduce the exact row filter of 01_preprocess.py, keeping raw NaNs ---
    raw = pd.read_csv(args.raw_file, parse_dates=["Date"])
    df = raw.dropna(subset=["AQI"]).copy()
    df = df[df[ML_POLLUTANTS].isna().mean(axis=1) <= 0.40].reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    dev = df[(df["Year"] >= 2015) & (df["Year"] <= 2018)].reset_index(drop=True)
    test = df[(df["Year"] >= 2019) & (df["Year"] <= 2020)].reset_index(drop=True)

    # --- Align with saved ML predictions ---
    pred_path = os.path.join(args.analysis_dir, "predictions", "predictions_pollutant_only.csv")
    ml = pd.read_csv(pred_path, parse_dates=["Date"])
    test = test.merge(ml, on=["Date", "City"], how="inner", validate="one_to_one")
    assert len(test) == len(ml) == 11410, f"Row alignment failed: {len(test)} vs {len(ml)}"
    assert np.allclose(test["AQI"].values, test["y_true"].values), "Target mismatch with ML predictions"
    y = test["AQI"].values.astype(float)

    # --- F1 / F2 on raw inputs ---
    f1, dom, valid, _ = formula_aqi(test, "NO2")
    f3, _, valid_nox, _ = formula_aqi(test, "NOx")
    # Rows with zero AQI pollutants available cannot be computed; fall back to dev global mean.
    gmean = float(dev["AQI"].mean())
    for arr in (f1, f3):
        arr[np.isnan(arr)] = gmean

    # --- F3: formula on the median-imputed inputs the ML models saw ---
    med = dev[ML_POLLUTANTS].median()
    test_imp = test.copy()
    test_imp[ML_POLLUTANTS] = test_imp[ML_POLLUTANTS].fillna(med)
    f4, _, _, _ = formula_aqi(test_imp, "NO2")

    # --- F4: linear calibration on sub-indices (fit on dev only) ---
    def design(frame):
        _, _, _, si = formula_aqi(frame, "NO2")
        X = si.fillna(0.0).copy()
        X["SI_max"] = si.max(axis=1).fillna(0.0)
        for p in AQI_POLLUTANTS:
            X[f"miss_{p}"] = si[p].isna().astype(float)
        X["const"] = 1.0
        return X.values
    X_dev, X_test = design(dev), design(test)
    beta, *_ = np.linalg.lstsq(X_dev, dev["AQI"].values.astype(float), rcond=None)
    f5 = X_test @ beta

    # --- F5: affine recalibration of F1 (AQI ~ a*F1 + b), fit on dev only ---
    f1_dev, _, _, _ = formula_aqi(dev, "NO2")
    ok = ~np.isnan(f1_dev)
    calib = np.polyfit(f1_dev[ok], dev["AQI"].values[ok].astype(float), 1)
    f6 = np.polyval(calib, f1)

    # --- F6: formula on a blend of today's and yesterday's readings, w chosen on dev ---
    def blended(frame, w):
        X = frame.copy()
        for p in AQI_POLLUTANTS:
            lag = locf_fill(frame, raw, p, days=1)
            X[p] = np.where(np.isnan(lag), X[p], (1 - w) * X[p] + w * lag)
        return formula_aqi(X, "NO2")[0]
    dev_lag = {p: locf_fill(dev, raw, p, days=1) for p in AQI_POLLUTANTS}
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    dev_mae = {}
    for wgt in grid:
        X = dev.copy()
        for p in AQI_POLLUTANTS:
            X[p] = np.where(np.isnan(dev_lag[p]), X[p], (1 - wgt) * X[p] + wgt * dev_lag[p])
        fd = formula_aqi(X, "NO2")[0]
        okd = ~np.isnan(fd) & (dev["City"] != "Ahmedabad").values
        dev_mae[wgt] = float(np.mean(np.abs(fd[okd] - dev["AQI"].values[okd])))
    w_best = min(dev_mae, key=dev_mae.get)
    f7 = blended(test, w_best)
    f7 = np.where(np.isnan(f7), gmean, f7)

    preds = {
        "F1 CPCB formula": f1,
        "F2 CPCB formula, NOx for nitrogen": f3,
        "F3 CPCB formula on median-imputed inputs": f4,
        "F4 Linear regression on sub-indices": f5,
        "F5 Affine-recalibrated CPCB formula": f6,
        "F6 CPCB formula on today/yesterday blend": f7,
    }
    for m in ML_MODELS:
        preds[m] = test[m].values.astype(float)

    # --- Metrics: all rows, and excluding Ahmedabad (anomalous CO record, see city table) ---
    # (Every test row passes the CPCB validity rule, so no separate valid-subset table is needed.)
    assert valid.all(), "Some test rows fail the CPCB validity rule; add a valid-subset table."
    keep = (test["City"] != "Ahmedabad").values
    rows = []
    for name, p in preds.items():
        rmse, mae, r2 = metrics(y, p)
        rmse_x, mae_x, r2_x = metrics(y[keep], p[keep])
        rows.append({"Method": name, "RMSE_all": rmse, "MAE_all": mae, "R2_all": r2,
                     "Bias_all": float(np.mean(p - y)),
                     "RMSE_exAhmedabad": rmse_x, "MAE_exAhmedabad": mae_x, "R2_exAhmedabad": r2_x,
                     "Bias_exAhmedabad": float(np.mean(p[keep] - y[keep])),
                     "Within_1pt": float(np.mean(np.abs(p - y) <= 1.0)),
                     "Within_5pct": float(np.mean(np.abs(p - y) <= 0.05 * y))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "formula_baseline_results.csv"), index=False)

    # --- Paired comparison vs CatBoost (date-block bootstrap) ---
    ref = preds["CatBoost"]
    dates = test["Date"].values
    pair_rows = []
    for name, p in preds.items():
        if name == "CatBoost":
            continue
        ci_mae, ci_rmse = block_bootstrap_diff(y, ref, p, dates, args.n_bootstrap, args.seed)
        pair_rows.append({"Comparison": f"{name} minus CatBoost",
                          "dMAE": metrics(y, p)[1] - metrics(y, ref)[1],
                          "dMAE_CI_low": ci_mae[0], "dMAE_CI_high": ci_mae[1],
                          "dRMSE": metrics(y, p)[0] - metrics(y, ref)[0],
                          "dRMSE_CI_low": ci_rmse[0], "dRMSE_CI_high": ci_rmse[1]})
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "formula_vs_ml_paired.csv"), index=False)

    # --- Dominant pollutant (formula-valid rows) ---
    dom_s = pd.Series(dom[valid]).value_counts(normalize=True).rename("share").reset_index()
    dom_s.columns = ["Dominant_pollutant", "Share_of_valid_test_rows"]
    dom_s.to_csv(os.path.join(out_dir, "dominant_pollutant_frequency.csv"), index=False)

    # --- Per-city errors ---
    test["F1"] = f1
    test["dominant"] = dom
    city = test.groupby("City").apply(lambda g: pd.Series({
        "n": len(g),
        "mean_AQI": g["AQI"].mean(),
        "F1_MAE": np.mean(np.abs(g["F1"] - g["AQI"])),
        "F1_bias": np.mean(g["F1"] - g["AQI"]),
        "CatBoost_MAE": np.mean(np.abs(g["CatBoost"] - g["AQI"])),
        "share_CO_dominant": np.mean(g["dominant"] == "CO"),
        "max_AQI": g["AQI"].max(),
        "share_CO_equals_NO_raw": float(np.mean((g["CO"] - g["NO"]).abs() < 1e-9)),
    }), include_groups=False).reset_index().sort_values("F1_MAE", ascending=False)
    city.to_csv(os.path.join(out_dir, "formula_city_wise.csv"), index=False)

    out_pred = test[["Date", "City", "AQI"]].copy()
    for k, v in preds.items():
        out_pred[k] = v
    out_pred["formula_valid"] = valid
    out_pred["dominant_pollutant"] = dom
    out_pred.to_csv(os.path.join(out_dir, "formula_predictions.csv"), index=False)

    # --- Summary ---
    with open(os.path.join(out_dir, "formula_baseline_summary.txt"), "w", encoding="utf-8") as f:
        def w(s=""):
            print(s); f.write(s + "\n")
        w("=" * 78)
        w("CPCB FORMULA BASELINE vs ML (out-of-time test, 2019-01-01 .. 2020-07-01)")
        w("=" * 78)
        w(f"Test rows: {len(test)} | formula-valid (>=3 pollutants incl. PM): {valid.sum()} "
          f"({valid.mean()*100:.1f}%) | NOx-variant valid: {valid_nox.sum()}")
        w("Inputs are DAILY MEANS. CPCB uses 8-h max for CO/O3 and 24-h rolling means; exact match is not expected.\n")
        w(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nPaired differences vs CatBoost (positive = worse than CatBoost), date-block bootstrap 95% CI:")
        w(pd.DataFrame(pair_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nDominant pollutant frequency (formula-valid test rows):")
        w(dom_s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nPer-city formula error (worst 10):")
        w(city.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        w("\nF4 coefficients (fit on 2015-2018):")
        names = AQI_POLLUTANTS + ["SI_max"] + [f"miss_{p}" for p in AQI_POLLUTANTS] + ["const"]
        w(", ".join(f"{n}={b:.3f}" for n, b in zip(names, beta)))
        w(f"F5 calibration (fit on 2015-2018): AQI = {calib[0]:.4f} * F1 + {calib[1]:.3f}")
        dev_x = dev[dev["City"] != "Ahmedabad"]
        f1_dx, _, _, _ = formula_aqi(dev_x, "NO2"); okx = ~np.isnan(f1_dx)
        calib_x = np.polyfit(f1_dx[okx], dev_x["AQI"].values[okx].astype(float), 1)
        px = np.polyval(calib_x, f1[keep])
        rx, mx, _ = metrics(y[keep], px)
        w(f"F5 sensitivity: refit excluding Ahmedabad from dev -> AQI = {calib_x[0]:.4f} * F1 + {calib_x[1]:.3f}; "
          f"ex-Ahmedabad test RMSE={rx:.3f} MAE={mx:.3f} bias={np.mean(px - y[keep]):.3f}")
        w(f"F6 blend weight on yesterday chosen on 2015-2018 (ex-Ahmedabad dev MAE): w = {w_best:.2f} "
          f"(dev MAE {dev_mae[w_best]:.3f}; w=0 gives {dev_mae[0.0]:.3f})")
        co_eq = (raw["CO"] - raw["NO"]).abs() < 1e-9
        w(f"Data-integrity check: raw rows with CO == NO exactly -> Ahmedabad "
          f"{co_eq[raw['City'] == 'Ahmedabad'].mean()*100:.1f}%, all other cities "
          f"{co_eq[raw['City'] != 'Ahmedabad'].mean()*100:.2f}% (CO column appears duplicated from NO).")
        sse = lambda m: np.sum((preds[m] - y) ** 2)
        w(f"Ahmedabad share of test rows: {(~keep).mean()*100:.1f}% | share of CatBoost squared error: "
          f"{np.sum(((preds['CatBoost'] - y) ** 2)[~keep]) / sse('CatBoost') * 100:.1f}%")


if __name__ == "__main__":
    main()
