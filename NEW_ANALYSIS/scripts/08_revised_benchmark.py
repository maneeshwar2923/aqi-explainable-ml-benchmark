"""
NEW_ANALYSIS/scripts/08_revised_benchmark.py
--------------------------------------------
Revised benchmark for the conference version of the paper.

Changes from 02_train_and_evaluate.py (each answers a red-team item):
  * Ahmedabad is excluded from training and test (its CO column duplicates NO, see
    07_data_audit.py). --include-ahmedabad runs the sensitivity analysis.
  * Feature sets:  S  = same-day readings of 9 gases/particulates (PM2.5, PM10, NO, NO2, NOx,
                        NH3, CO, SO2, O3)
                   SL = S + the same 9 readings from the previous calendar day (same city).
    The target is built from trailing averaging windows, so the previous day is legitimately
    available information and is needed for a fair comparison with the two-day formula.
  * Missing inputs: native NaN handling for all models (no median imputation).
  * Equal-budget random search (N_TRIALS configurations per model) on 2015-2017 -> 2018,
    selected by validation MAE; boosting rounds by early stopping. Refit on 2015-2018 with
    five seeds; mean and SD over seeds are reported.
  * Strongest deterministic baseline: CPCB formula with nitrogen source (NO2 vs NOx), weight on
    yesterday's readings, and an affine recalibration all chosen on 2015-2018.
  * Extra metrics: bias, CPCB-category accuracy, period-matched temporal comparison
    (Jan-Mar 24 vs Mar 25-Jul 1, 2019 vs 2020), seen vs unseen cities, per-band error.
Outputs: NEW_ANALYSIS/revised[_with_ahmedabad]/
"""
import os, json, time, argparse, importlib.util, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "05_cpcb_formula_baseline.py"))
fb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fb)

FILTER_COLS = fb.ML_POLLUTANTS                     # row filter identical to the original paper
GASES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
LAGS = [f"{g}_lag1" for g in GASES]
FEATS = {"S": GASES, "SL": GASES + LAGS}
MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
SEEDS = [0, 1, 2, 3, 4]
N_TRIALS = 20
BANDS = [0, 50, 100, 200, 300, 400, np.inf]
BAND_NAMES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
LOCKDOWN = pd.Timestamp("2020-03-25")


# ----------------------------------------------------------------------------- data
def load(include_ahmedabad):
    raw = pd.read_csv(os.path.join(ROOT, "city_day.csv"), parse_dates=["Date"])
    lag = raw[["City", "Date"] + GASES].copy()
    lag["Date"] = lag["Date"] + pd.Timedelta(days=1)
    lag.columns = ["City", "Date"] + LAGS
    df = raw.dropna(subset=["AQI"]).copy()
    df = df[df[FILTER_COLS].isna().mean(axis=1) <= 0.40]
    df = df.merge(lag, on=["City", "Date"], how="left", validate="one_to_one")
    if not include_ahmedabad:
        df = df[df.City != "Ahmedabad"]
    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    yr = df.Date.dt.year
    parts = {"sub": df[yr <= 2017], "val": df[yr == 2018], "dev": df[yr <= 2018], "test": df[yr >= 2019]}
    return raw, {k: v.reset_index(drop=True) for k, v in parts.items()}


# ----------------------------------------------------------------------------- models
def sample_params(name, rng):
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    if name == "Random Forest":
        return {"max_features": float(rng.choice([0.33, 0.5, 0.7, 1.0])),
                "min_samples_leaf": int(rng.choice([1, 2, 3, 5, 8])),
                "max_depth": [None, 12, 20, 30][rng.integers(4)]}
    if name == "XGBoost":
        return {"max_depth": int(rng.integers(3, 11)), "learning_rate": lu(0.01, 0.2),
                "subsample": float(rng.uniform(0.6, 1)), "colsample_bytree": float(rng.uniform(0.6, 1)),
                "min_child_weight": lu(1, 20), "reg_lambda": lu(0.1, 10)}
    if name == "LightGBM":
        return {"num_leaves": int(rng.integers(15, 256)), "learning_rate": lu(0.01, 0.2),
                "min_child_samples": int(rng.integers(5, 101)), "subsample": float(rng.uniform(0.6, 1)),
                "subsample_freq": 1, "colsample_bytree": float(rng.uniform(0.6, 1)), "reg_lambda": lu(0.1, 10)}
    if name == "CatBoost":
        return {"depth": int(rng.integers(4, 11)), "learning_rate": lu(0.01, 0.2),
                "l2_leaf_reg": lu(1, 10), "random_strength": float(rng.uniform(0, 2))}


def make(name, p, seed, n_iter=None):
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1, **p)
    if name == "XGBoost":
        kw = dict(n_estimators=n_iter or 5000, random_state=seed, n_jobs=-1, tree_method="hist", **p)
        if n_iter is None:
            kw.update(early_stopping_rounds=100, eval_metric="rmse")
        return xgb.XGBRegressor(**kw)
    if name == "LightGBM":
        return lgb.LGBMRegressor(n_estimators=n_iter or 5000, random_state=seed, n_jobs=-1, verbose=-1, **p)
    if name == "CatBoost":
        kw = dict(iterations=n_iter or 5000, random_seed=seed, verbose=False, thread_count=-1,
                  allow_writing_files=False, **p)
        if n_iter is None:
            kw.update(early_stopping_rounds=100)
        return CatBoostRegressor(**kw)


def fit_val(name, p, Xs, ys, Xv, yv, seed):
    """Fit on sub with early stopping on val. Returns (model, best_iteration)."""
    m = make(name, p, seed)
    if name == "Random Forest":
        m.fit(Xs, ys); return m, None
    if name == "XGBoost":
        m.fit(Xs, ys, eval_set=[(Xv, yv)], verbose=False); return m, int(m.best_iteration) + 1
    if name == "LightGBM":
        m.fit(Xs, ys, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, int(m.best_iteration_)
    m.fit(Xs, ys, eval_set=(Xv, yv)); return m, int(m.get_best_iteration()) + 1


def tune(name, P, fs, seed=0):
    rng = np.random.default_rng(1000 + MODELS.index(name))
    X = {k: P[k][FEATS[fs]].values for k in ("sub", "val")}
    best = None
    log = []
    for t in range(N_TRIALS):
        p = sample_params(name, rng)
        m, it = fit_val(name, p, X["sub"], P["sub"].AQI.values, X["val"], P["val"].AQI.values, seed)
        mae = float(np.mean(np.abs(m.predict(X["val"]) - P["val"].AQI.values)))
        log.append({"trial": t, "val_MAE": mae, "iters": it, **{k: str(v) for k, v in p.items()}})
        print(f"    [{fs}] {name} trial {t:2d}: val MAE {mae:6.2f} iters {it} {time.strftime('%H:%M:%S')}", flush=True)
        if best is None or mae < best[0]:
            best = (mae, p, it)
    return best, log


# ----------------------------------------------------------------------------- formula
def strongest_formula(raw, P):
    """Nitrogen source, yesterday weight and affine recalibration chosen on dev (2015-2018)."""
    dev = P["dev"]
    lagd = {p: fb.locf_fill(dev, raw, p, days=1) for p in fb.AQI_POLLUTANTS + ["NOx"]}

    def blend(frame, lagv, w):
        X = frame.copy()
        for p in fb.AQI_POLLUTANTS + ["NOx"]:
            X[p] = np.where(np.isnan(lagv[p]), X[p], (1 - w) * X[p] + w * lagv[p])
        return X

    best = None
    for nsrc in ("NO2", "NOx"):
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            f = fb.formula_aqi(blend(dev, lagd, w), nsrc)[0]
            ok = ~np.isnan(f)
            mae = float(np.mean(np.abs(f[ok] - dev.AQI.values[ok])))
            if best is None or mae < best[0]:
                best = (mae, nsrc, float(w))
    _, nsrc, w = best
    f_dev = fb.formula_aqi(blend(dev, lagd, w), nsrc)[0]
    ok = ~np.isnan(f_dev)
    a, b = np.polyfit(f_dev[ok], dev.AQI.values[ok], 1)

    def predict(frame, lag_override=None):
        lagv = lag_override if lag_override is not None else {p: fb.locf_fill(frame, raw, p, days=1) for p in fb.AQI_POLLUTANTS + ["NOx"]}
        f = fb.formula_aqi(blend(frame, lagv, w), nsrc)[0]
        f = np.where(np.isnan(f), float(dev.AQI.mean()), f)
        return a * f + b
    return predict, {"nitrogen": nsrc, "w_yesterday": w, "a": float(a), "b": float(b), "dev_MAE_unrecal": best[0]}


# ----------------------------------------------------------------------------- metrics
def band(v):
    return np.digitize(v, BANDS[1:-1], right=True)


def metr(y, p):
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), "Bias": float(np.mean(e)),
            "BandAcc": float(np.mean(band(p) == band(y)))}


def boot_diff(ae_ref, ae_cmp, dates, n_boot=2000, seed=42):
    """Date-block bootstrap CI of mean(ae_cmp - ae_ref); whole days are resampled."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(dates))
    d = pd.Series(ae_cmp - ae_ref).groupby(codes).agg(["sum", "size"])
    s, c = d["sum"].values, d["size"].values
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    stats = s[idx].sum(1) / c[idx].sum(1)
    return float(np.mean(ae_cmp - ae_ref)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-ahmedabad", action="store_true")
    args = ap.parse_args()
    out = os.path.join(HERE, "..", "revised" + ("_with_ahmedabad" if args.include_ahmedabad else ""))
    os.makedirs(out, exist_ok=True)
    raw, P = load(args.include_ahmedabad)
    test, y = P["test"], P["test"].AQI.values.astype(float)
    print({k: len(v) for k, v in P.items()}, "cities in test:", test.City.nunique())

    preds, info, tune_logs, times = {}, {}, [], {}
    # ---- ML: tune once (seed 0), refit on dev with 5 seeds ----
    for fs in ("S", "SL"):
        for name in MODELS:
            t0 = time.time()
            (vmae, p, it), log = tune(name, P, fs)
            for r in log:
                r.update(model=name, features=fs)
            tune_logs += log
            runs = []
            for s in SEEDS:
                m = make(name, p, s, n_iter=it)
                t1 = time.time()
                m.fit(P["dev"][FEATS[fs]].values, P["dev"].AQI.values)
                times.setdefault(f"{name} [{fs}]", []).append(time.time() - t1)
                runs.append(m.predict(test[FEATS[fs]].values))
                if name == "CatBoost" and s == 0:
                    m.save_model(os.path.join(out, f"catboost_{fs}_seed0.cbm"))
            key = f"{name} [{fs}]"
            preds[key] = np.vstack(runs)                     # seeds x rows
            info[key] = {"params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in p.items()},
                         "iterations": it, "val_MAE": vmae, "search_seconds": time.time() - t0}
            print(f"{key:<24} val MAE {vmae:6.2f}  iters {it}  test MAE "
                  f"{np.mean([np.mean(np.abs(r - y)) for r in runs]):6.2f}")

    # ---- deterministic baselines ----
    f_strong, f_info = strongest_formula(raw, P)
    base = {}
    f1 = fb.formula_aqi(test, "NO2")[0]
    base["CPCB formula (same day)"] = np.where(np.isnan(f1), P["dev"].AQI.mean(), f1)
    fd = fb.formula_aqi(P["dev"], "NO2")[0]; ok = ~np.isnan(fd)
    a, b = np.polyfit(fd[ok], P["dev"].AQI.values[ok], 1)
    base["Recalibrated formula (same day)"] = a * base["CPCB formula (same day)"] + b
    base["Two-day recalibrated formula"] = f_strong(test)
    cm = P["dev"].groupby("City").AQI.mean()
    base["City-mean"] = test.City.map(cm).fillna(P["dev"].AQI.mean()).values

    # ---- results table ----
    rows = []
    for k, v in base.items():
        rows.append({"Method": k, "Type": "baseline", **metr(y, v), **{f"{m}_sd": 0.0 for m in ("MAE", "RMSE", "R2", "BandAcc")}})
    for k, R in preds.items():
        ms = [metr(y, r) for r in R]
        row = {"Method": k, "Type": "ML", **{m: float(np.mean([x[m] for x in ms])) for m in ms[0]}}
        row.update({f"{m}_sd": float(np.std([x[m] for x in ms], ddof=1)) for m in ("MAE", "RMSE", "R2", "BandAcc")})
        row["fit_seconds"] = float(np.median(times[k]))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "main_results.csv"), index=False)

    # ---- paired comparisons (seed-averaged absolute error per row) ----
    ae = {k: np.abs(v - y) for k, v in base.items()}
    ae.update({k: np.mean(np.abs(R - y), axis=0) for k, R in preds.items()})
    best_ml = min(preds, key=lambda k: ae[k].mean())
    pairs = []
    for ref, cmp in ([("Two-day recalibrated formula", k) for k in preds] +
                     [(best_ml, k) for k in preds if k != best_ml] +
                     [(f"{m} [S]", f"{m} [SL]") for m in MODELS]):
        d, lo, hi = boot_diff(ae[ref], ae[cmp], test.Date.values)
        pairs.append({"reference": ref, "comparison": cmp, "dMAE(cmp-ref)": d, "CI_low": lo, "CI_high": hi})
    pd.DataFrame(pairs).to_csv(os.path.join(out, "paired_comparisons.csv"), index=False)

    # ---- breakdowns for the best ML model and the strongest formula ----
    pb = preds[best_ml].mean(0)
    pf = base["Two-day recalibrated formula"]
    seen = set(P["dev"].City)
    test = test.assign(pred=pb, form=pf, y=y, band=pd.Categorical.from_codes(band(y), BAND_NAMES))
    md = test.Date.dt.month * 100 + test.Date.dt.day
    test["window"] = np.where(md < 325, "Jan 1-Mar 24", np.where(md <= 701, "Mar 25-Jul 1", "Jul 2-Dec 31"))
    test["year"] = test.Date.dt.year
    def g(fr):
        return pd.Series({"n": len(fr), "mean_AQI": fr.y.mean(), "MAE_ML": np.mean(np.abs(fr.pred - fr.y)),
                          "MAE_formula": np.mean(np.abs(fr.form - fr.y)),
                          "nMAE_ML": np.mean(np.abs(fr.pred - fr.y)) / fr.y.mean()})
    temporal = test.groupby(["window", "year"]).apply(g, include_groups=False).reset_index()
    spatial = test.assign(group=np.where(test.City.isin(seen), "seen", "unseen")).groupby("group").apply(g, include_groups=False).reset_index()
    bands = test.groupby("band", observed=True).apply(g, include_groups=False).reset_index()
    city = test.groupby("City").apply(g, include_groups=False).reset_index().sort_values("MAE_ML", ascending=False)
    temporal.to_csv(os.path.join(out, "temporal_matched.csv"), index=False)
    spatial.to_csv(os.path.join(out, "spatial.csv"), index=False)
    bands.to_csv(os.path.join(out, "per_band.csv"), index=False)
    city.to_csv(os.path.join(out, "per_city.csv"), index=False)
    pd.DataFrame(tune_logs).to_csv(os.path.join(out, "tuning_log.csv"), index=False)

    pred_out = test[["Date", "City", "AQI"]].copy()
    for k, v in base.items():
        pred_out[k] = v
    for k, R in preds.items():
        pred_out[k] = R.mean(0)
    pred_out.to_csv(os.path.join(out, "predictions_seedmean.csv"), index=False)
    json.dump({"rows": {k: len(v) for k, v in P.items()}, "test_cities": int(test.City.nunique()),
               "unseen_cities": sorted(set(test.City) - seen), "best_ml": best_ml,
               "formula": f_info, "same_day_recal": [float(a), float(b)], "models": info,
               "n_trials": N_TRIALS, "seeds": SEEDS},
              open(os.path.join(out, "run_info.json"), "w"), indent=2, default=str)

    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as fh:
        def w(s=""):
            print(s); fh.write(s + "\n")
        w(f"Rows: {({k: len(v) for k, v in P.items()})}; best ML: {best_ml}; formula: {f_info}")
        w(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nPaired dMAE, date-block bootstrap 95% CI:")
        w(pd.DataFrame(pairs).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        w("\nTemporal (period-matched):"); w(temporal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nSpatial:"); w(spatial.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nPer band:"); w(bands.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        w("\nPer city:"); w(city.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
