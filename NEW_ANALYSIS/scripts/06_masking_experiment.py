"""
NEW_ANALYSIS/scripts/06_masking_experiment.py
---------------------------------------------
Controlled missing-sub-index experiment (reviewer item F2).

Question: when one pollutant reading is missing, does ML estimate the AQI better than
the CPCB formula applied to the pollutants that remain (or to a simple imputation)?

Protocol
  * Evaluation rows: out-of-time test rows (2019-01-01 .. 2020-07-01) on which all eight
    pollutants are observed, so the masked value is known and the target is unchanged.
  * Scenarios: no mask; mask each AQI pollutant on its own; mask the row's dominant
    pollutant (the one setting the formula AQI before masking); mask both PM2.5 and PM10.
  * Training uses 2015-2018 only. Iteration counts come from early stopping on 2018,
    as in 02_train_and_evaluate.py. Fallback statistics (medians) come from the
    development data only.
  * Primary run leaves out Ahmedabad (its CO column duplicates NO, see
    05_cpcb_formula_baseline.py). Run with --include-ahmedabad for the sensitivity
    analysis; that run also checks that the retrained published models reproduce
    predictions/predictions_pollutant_only.csv.

Methods
  Formula family (no learning, apart from a 2-parameter affine recalibration fit on 2015-2018)
    FORM-raw        CPCB formula on the remaining pollutants
    FORM-recal      the same, recalibrated (AQI ~ a*F + b)
    FORM-recal+LOCF masked value filled with the city's last reading in the previous 7 days
    FORM-recal+med  masked value filled with the city's 2015-2018 median
  ML family (same hyperparameters as the published benchmark)
    <model>-median  published setup: masked value filled with the 2015-2018 global median
    <model>-NaN     trained on the raw inputs with native missing-value handling; masked value = NaN
    <model>-aug     as -NaN, plus training copies with one random AQI pollutant masked
    CatBoost-aug+LOCF  CatBoost-aug fed the LOCF-filled value instead of NaN

Outputs (NEW_ANALYSIS/masking_experiment[_with_ahmedabad]/)
  masking_results.csv    one row per (scenario, method): n, MAE, RMSE, bias, dMAE vs reference + CI
  masking_mae_table.csv  MAE pivot table (scenario x method)
  masking_summary.txt    readable summary
"""

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "05_cpcb_formula_baseline.py"))
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)

POLLUTANTS = fb.ML_POLLUTANTS             # model features (8, includes NO)
AQI_POLL = fb.AQI_POLLUTANTS              # formula pollutants (7)
REFERENCE = "FORM-recal+LOCF"
LOCF_DAYS = 7


# ----------------------------------------------------------------------------- models
def fit_xgb(Xs, ys, Xv, yv, Xf, yf, seed):
    m = xgb.XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=500, reg_alpha=0.1,
                         reg_lambda=1.0, early_stopping_rounds=20, eval_metric="rmse",
                         random_state=seed, n_jobs=-1)
    m.fit(Xs, ys, eval_set=[(Xv, yv)], verbose=False)
    it = int(m.best_iteration)
    f = xgb.XGBRegressor(learning_rate=0.05, max_depth=6, n_estimators=it, reg_alpha=0.1,
                         reg_lambda=1.0, random_state=seed, n_jobs=-1)
    return f.fit(Xf, yf)


def fit_lgb(Xs, ys, Xv, yv, Xf, yf, seed):
    m = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=500,
                          random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(Xs, ys, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(20, verbose=False)])
    it = int(m.best_iteration_)
    f = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=it,
                          random_state=seed, n_jobs=-1, verbose=-1)
    return f.fit(Xf, yf)


def fit_cb(Xs, ys, Xv, yv, Xf, yf, seed):
    m = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                          early_stopping_rounds=20, random_state=seed, verbose=False)
    m.fit(Xs, ys, eval_set=(Xv, yv), verbose=False)
    it = int(m.get_best_iteration()) or 500
    f = CatBoostRegressor(iterations=it, learning_rate=0.05, depth=6, random_state=seed, verbose=False)
    return f.fit(Xf, yf)


def fit_rf(Xs, ys, Xv, yv, Xf, yf, seed):
    return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xf, yf)


FITTERS = {"Random Forest": fit_rf, "XGBoost": fit_xgb, "LightGBM": fit_lgb, "CatBoost": fit_cb}


def augment(frame, rng):
    """Original rows plus one copy per row with a random observed AQI pollutant set to NaN."""
    cp = frame.copy()
    obs = cp[AQI_POLL].notna().values
    for i in range(len(cp)):
        cand = np.flatnonzero(obs[i])
        if len(cand):
            cp.iat[i, cp.columns.get_loc(AQI_POLL[rng.choice(cand)])] = np.nan
    return pd.concat([frame, cp], ignore_index=True)


# ----------------------------------------------------------------------------- helpers
def main():
    ap = argparse.ArgumentParser()
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    ap.add_argument("--raw-file", default=os.path.join(root, "city_day.csv"))
    ap.add_argument("--analysis-dir", default=os.path.abspath(os.path.join(HERE, "..")))
    ap.add_argument("--include-ahmedabad", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = os.path.join(args.analysis_dir, "masking_experiment" + ("_with_ahmedabad" if args.include_ahmedabad else ""))
    os.makedirs(out_dir, exist_ok=True)

    # ---- data (same row filter as 01_preprocess.py, raw NaNs kept) ----
    raw = pd.read_csv(args.raw_file, parse_dates=["Date"])
    df = raw.dropna(subset=["AQI"]).copy()
    df = df[df[POLLUTANTS].isna().mean(axis=1) <= 0.40].reset_index(drop=True)
    if not args.include_ahmedabad:
        df = df[df["City"] != "Ahmedabad"].reset_index(drop=True)
    yr = df["Date"].dt.year
    sub, val, dev, test = (df[(yr >= 2015) & (yr <= 2017)].reset_index(drop=True), df[yr == 2018].reset_index(drop=True),
                           df[yr <= 2018].reset_index(drop=True), df[yr >= 2019].reset_index(drop=True))
    complete = test[test[POLLUTANTS].notna().all(axis=1)].copy()
    y = complete["AQI"].values.astype(float)

    med_sub, med_dev = sub[POLLUTANTS].median(), dev[POLLUTANTS].median()
    city_med = dev.groupby("City")[POLLUTANTS].median()

    def imp(frame, med):
        return frame[POLLUTANTS].fillna(med).values

    # ---- formula recalibration (fit on dev) ----
    f_dev, _, _, _ = fb.formula_aqi(dev)
    ok = ~np.isnan(f_dev)
    calib = np.polyfit(f_dev[ok], dev["AQI"].values[ok].astype(float), 1)

    # ---- train models ----
    print(f"Training models ({'with' if args.include_ahmedabad else 'without'} Ahmedabad)...")
    models = {}
    for name, fit in FITTERS.items():
        models[f"{name}-median"] = fit(imp(sub, med_sub), sub["AQI"].values, imp(val, med_sub), val["AQI"].values,
                                       imp(dev, med_dev), dev["AQI"].values, args.seed)
    for name in ["XGBoost", "LightGBM", "CatBoost"]:
        models[f"{name}-NaN"] = FITTERS[name](sub[POLLUTANTS].values, sub["AQI"].values, val[POLLUTANTS].values,
                                              val["AQI"].values, dev[POLLUTANTS].values, dev["AQI"].values, args.seed)
    sub_a, val_a, dev_a = augment(sub, rng), augment(val, rng), augment(dev, rng)
    for name in ["LightGBM", "CatBoost"]:
        models[f"{name}-aug"] = FITTERS[name](sub_a[POLLUTANTS].values, sub_a["AQI"].values, val_a[POLLUTANTS].values,
                                              val_a["AQI"].values, dev_a[POLLUTANTS].values, dev_a["AQI"].values, args.seed)

    # ---- reproduction check against the published predictions ----
    verify_msg = "Reproduction check skipped (Ahmedabad excluded; published models include it)."
    if args.include_ahmedabad:
        pub = pd.read_csv(os.path.join(args.analysis_dir, "predictions", "predictions_pollutant_only.csv"), parse_dates=["Date"])
        t = test.merge(pub, on=["Date", "City"], validate="one_to_one")
        diffs = {n: float(np.max(np.abs(models[f"{n}-median"].predict(t[POLLUTANTS].fillna(med_dev).values) - t[n].values)))
                 for n in FITTERS}
        verify_msg = "Reproduction check, max |retrained - published| prediction: " + \
                     ", ".join(f"{k}={v:.4g}" for k, v in diffs.items())
    print(verify_msg)

    # ---- scenarios ----
    f_full, dom_full, _, _ = fb.formula_aqi(complete)
    scenarios = {"none": [[] for _ in range(len(complete))]}
    for p in AQI_POLL:
        scenarios[p] = [[p]] * len(complete)
    scenarios["dominant"] = [[d] for d in dom_full]
    scenarios["PM2.5+PM10"] = [["PM2.5", "PM10"]] * len(complete)

    locf = {p: fb.locf_fill(complete, raw, p) for p in AQI_POLL}
    locf_cov = {p: float(np.mean(~np.isnan(locf[p]))) for p in AQI_POLL}
    cmed = {p: complete["City"].map(city_med[p]).fillna(med_dev[p]).values for p in AQI_POLL}
    for p in AQI_POLL:  # LOCF fallback: city median, then global median
        locf[p] = np.where(np.isnan(locf[p]), cmed[p], locf[p])

    rows = []
    for scen, masks in scenarios.items():
        X_nan = complete[POLLUTANTS].copy()
        X_med, X_city, X_locf = X_nan.copy(), X_nan.copy(), X_nan.copy()
        for p in AQI_POLL:
            hit = np.array([p in m for m in masks])
            if not hit.any():
                continue
            X_nan.loc[hit, p] = np.nan
            X_med.loc[hit, p] = med_dev[p]
            X_city.loc[hit, p] = cmed[p][hit]
            X_locf.loc[hit, p] = locf[p][hit]

        f_rem, _, _, _ = fb.formula_aqi(X_nan)
        f_rem = np.where(np.isnan(f_rem), float(dev["AQI"].mean()), f_rem)  # no pollutant left at all
        preds = {
            "FORM-raw": f_rem,
            "FORM-recal": np.polyval(calib, f_rem),
            "FORM-recal+LOCF": np.polyval(calib, fb.formula_aqi(X_locf)[0]),
            "FORM-recal+med": np.polyval(calib, fb.formula_aqi(X_city)[0]),
        }
        for name in FITTERS:
            preds[f"{name}-median"] = models[f"{name}-median"].predict(X_med.values)
        for k in ["XGBoost-NaN", "LightGBM-NaN", "CatBoost-NaN", "LightGBM-aug", "CatBoost-aug"]:
            preds[k] = models[k].predict(X_nan.values)
        preds["CatBoost-aug+LOCF"] = models["CatBoost-aug"].predict(X_locf.values)

        ref = preds[REFERENCE]
        for meth, p in preds.items():
            rmse, mae, r2 = fb.metrics(y, p)
            if meth == REFERENCE:
                ci = (np.nan, np.nan)
            else:
                ci, _ = fb.block_bootstrap_diff(y, ref, p, complete["Date"].values, args.n_bootstrap, args.seed)
            rows.append({"Scenario": scen, "Method": meth, "n": len(y), "MAE": mae, "RMSE": rmse, "R2": r2,
                         "Bias": float(np.mean(p - y)), f"dMAE_vs_{REFERENCE}": mae - fb.metrics(y, ref)[1],
                         "dMAE_CI_low": ci[0], "dMAE_CI_high": ci[1]})
        print(f"  scenario {scen:<11} done")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "masking_results.csv"), index=False)
    order = list(scenarios)
    mae_tab = res.pivot(index="Scenario", columns="Method", values="MAE").loc[order, list(preds)]
    mae_tab.to_csv(os.path.join(out_dir, "masking_mae_table.csv"))

    with open(os.path.join(out_dir, "masking_summary.txt"), "w", encoding="utf-8") as f:
        def w(s=""):
            print(s); f.write(s + "\n")
        w("=" * 90)
        w(f"MISSING-SUB-INDEX MASKING EXPERIMENT ({'Ahmedabad INCLUDED' if args.include_ahmedabad else 'Ahmedabad excluded'})")
        w("=" * 90)
        w(f"Complete-case test rows (all 8 pollutants observed): {len(complete)} of {len(test)} test rows, "
          f"{complete['City'].nunique()} cities")
        w(f"Formula recalibration (fit on 2015-2018): AQI = {calib[0]:.4f} * F + {calib[1]:.3f}")
        w(f"LOCF coverage (reading in previous {LOCF_DAYS} days): " + ", ".join(f"{p}={v*100:.1f}%" for p, v in locf_cov.items()))
        w("Dominant pollutant on complete rows: " + ", ".join(
            f"{k}={v*100:.1f}%" for k, v in pd.Series(dom_full).value_counts(normalize=True).items()))
        w(verify_msg)
        w("\nMAE by scenario (rows) and method (columns):")
        w(mae_tab.T.to_string(float_format=lambda v: f"{v:.2f}"))
        w(f"\nBest method per scenario, with dMAE vs {REFERENCE} (date-block bootstrap 95% CI):")
        for scen in order:
            s = res[res.Scenario == scen].sort_values("MAE")
            b = s.iloc[0]
            ci = "" if b.Method == REFERENCE else f" [{b.dMAE_CI_low:+.2f}, {b.dMAE_CI_high:+.2f}]"
            bf = s[s.Method.str.startswith("FORM")].iloc[0]
            w(f"  {scen:<11} best={b.Method:<18} MAE={b.MAE:6.2f}  dMAE={b[f'dMAE_vs_{REFERENCE}']:+.2f}{ci}"
              f" | best formula={bf.Method} ({bf.MAE:.2f})")


if __name__ == "__main__":
    main()
