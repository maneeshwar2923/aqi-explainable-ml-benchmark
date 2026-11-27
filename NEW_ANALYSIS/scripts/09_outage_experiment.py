"""
NEW_ANALYSIS/scripts/09_outage_experiment.py
--------------------------------------------
Simulated analyser outages (revision of 06_masking_experiment.py).

An outage of length L removes an analyser's readings for days D-L+1 .. D, so on the target
day D the last available reading is from day D-L or earlier. Analysers are masked as groups:
the chemiluminescence NOx analyser reports NO, NO2 and NOx together, so all three go missing.

Evaluation rows: test rows (2019-01-01 .. 2020-07-01, Ahmedabad excluded) on which every input is
observed today, so the masked values are known and the target is unchanged.

Methods (all fitted on 2015-2018 only; hyperparameters from 08_revised_benchmark.py)
  Formula-drop      two-day recalibrated CPCB formula on the analysers that remain
  Formula+LOCF      the same, masked analyser filled with its last available reading
  <M>               <M> [SL] model, masked inputs = NaN (native missing handling)
  <M>-aug           trained with extra copies that have one random analyser masked
                    (today only, or today and yesterday), masked inputs = NaN
  <M>-stale         trained with extra copies whose masked analyser is filled with its last
                    reading L in {1,3,7} days old, plus an "age in days" input per analyser;
                    at test time it is given the stale reading and its true age
Outputs: NEW_ANALYSIS/outage_experiment/
"""
import os, json, importlib.util, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, file):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rb = _load("rb", "08_revised_benchmark.py")
fb = rb.fb

GROUPS = {"PM2.5": ["PM2.5"], "PM10": ["PM10"], "NO2": ["NO", "NO2", "NOx"], "SO2": ["SO2"],
          "NH3": ["NH3"], "CO": ["CO"], "O3": ["O3"]}
AGE = [f"age_{g}" for g in GROUPS]
LENGTHS = [1, 3, 7]
SCEN = ["none", "PM2.5", "PM10", "CO", "NO2", "dominant", "PM2.5+PM10"]
SEEDS = [0, 1, 2]
ML = ["LightGBM", "Random Forest"]
REF = "Formula+LOCF"
OUT = os.path.join(HERE, "..", "outage_experiment")


def stale_value(frame, raw, col, L, tol_days=30):
    """Last reading of `col` in the same city on or before day D-L (within tol_days)."""
    left = frame[["City", "Date"]].reset_index()
    left["key"] = left["Date"] - pd.Timedelta(days=L - 1)
    left = left.sort_values("key")
    right = raw.loc[raw[col].notna(), ["City", "Date", col]].rename(columns={"Date": "key"}).sort_values("key")
    m = pd.merge_asof(left, right, on="key", by="City", direction="backward",
                      allow_exact_matches=False, tolerance=pd.Timedelta(days=tol_days))
    return m.set_index("index").loc[frame.index, col].values


def apply_outage(frame, raw, groups_per_row, L, fill, fallback):
    """Return a copy of frame with the listed analyser groups out for L days.
    fill='nan' -> NaN; fill='stale' -> last available reading (fallback: city median)."""
    X = frame.copy()
    for g, cols in GROUPS.items():
        hit = np.array([g in gs for gs in groups_per_row])
        if not hit.any():
            continue
        X.loc[hit, f"age_{g}"] = L
        for c in cols:
            if fill == "nan":
                X.loc[hit, c] = np.nan
                if L >= 2:
                    X.loc[hit, f"{c}_lag1"] = np.nan
            else:
                v = stale_value(frame.loc[hit], raw, c, L)
                v = np.where(np.isnan(v), frame.loc[hit, "City"].map(fallback[c]).values, v)
                X.loc[hit, c] = v
                if L >= 2:
                    X.loc[hit, f"{c}_lag1"] = v
    return X


def augment(frame, raw, rng, fill, fallback):
    """Original rows + one copy per row with one random observed analyser out."""
    cp = frame.copy()
    groups = list(GROUPS)
    pick = [[groups[rng.integers(len(groups))]] for _ in range(len(cp))]
    Ls = rng.choice(LENGTHS, len(cp)) if fill == "stale" else rng.choice([1, 2], len(cp))
    parts = []
    for L in np.unique(Ls):
        idx = np.flatnonzero(Ls == L)
        sub = cp.iloc[idx]
        parts.append(apply_outage(sub, raw, [pick[i] for i in idx], int(L), fill, fallback))
    return pd.concat([frame] + parts, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    raw, P = rb.load(include_ahmedabad=False)
    info = json.load(open(os.path.join(HERE, "..", "revised", "run_info.json")))
    for k in ("sub", "val", "dev", "test"):
        P[k][AGE] = 0
    fallback = {c: P["dev"].groupby("City")[c].median().reindex(P["test"].City.unique())
                .fillna(P["dev"][c].median()) for cols in GROUPS.values() for c in cols}
    test = P["test"]
    complete = test[test[rb.GASES].notna().all(axis=1)].copy()
    y = complete.AQI.values.astype(float)
    f_pred, f_info = rb.strongest_formula(raw, P)
    print(f"complete-case rows {len(complete)} / {len(test)}, cities {complete.City.nunique()}; formula {f_info}")

    # dominant analyser per row (formula on full information)
    dom = fb.formula_aqi(complete, f_info["nitrogen"])[1]
    dom = np.where(dom == "NO2", "NO2", dom)

    # ---- train ML variants ----
    feats = rb.FEATS["SL"]
    models = {}
    for name in ML:
        key = f"{name} [SL]"
        p, it = info["models"][key]["params"], info["models"][key]["iterations"]
        p = {k: (None if v == "None" else v) for k, v in p.items()}
        for s in SEEDS:
            models[(name, s)] = rb.make(name, p, s, n_iter=it).fit(P["dev"][feats].values, P["dev"].AQI.values)
            for fill, tag in (("nan", "aug"), ("stale", "stale")):
                da = augment(P["dev"], raw, np.random.default_rng(100 + s), fill, {c: P["dev"].groupby("City")[c].median().reindex(P["dev"].City.unique()).fillna(P["dev"][c].median()) for c in fallback})
                cols = feats + (AGE if tag == "stale" else [])
                # Twice the rows: keep the tuned learning rate, allow proportionally more rounds.
                n_it = None if it is None else int(it * 1.5)
                models[(f"{name}-{tag}", s)] = (rb.make(name, p, s, n_iter=n_it).fit(da[cols].values, da.AQI.values), cols)
            print(f"  trained {name} seed {s}")

    # ---- evaluate ----
    rows = []
    for scen in SCEN:
        if scen == "none":
            gs = [[] for _ in range(len(complete))]
        elif scen == "dominant":
            gs = [[d] for d in dom]
        else:
            gs = [scen.split("+")] * len(complete)
        for L in (LENGTHS if scen != "none" else [0]):
            Xn = apply_outage(complete, raw, gs, max(L, 1), "nan", fallback) if scen != "none" else complete.copy()
            Xs = apply_outage(complete, raw, gs, max(L, 1), "stale", fallback) if scen != "none" else complete.copy()
            lag_n = {p: Xn[f"{p}_lag1"].values for p in fb.AQI_POLLUTANTS + ["NOx"]}
            lag_s = {p: Xs[f"{p}_lag1"].values for p in fb.AQI_POLLUTANTS + ["NOx"]}
            ae = {"Formula-drop": [np.abs(f_pred(Xn, lag_n) - y)], REF: [np.abs(f_pred(Xs, lag_s) - y)]}
            for name in ML:
                for s in SEEDS:
                    ae.setdefault(name, []).append(np.abs(models[(name, s)].predict(Xn[feats].values) - y))
                    m, cols = models[(f"{name}-aug", s)]
                    ae.setdefault(f"{name}-aug", []).append(np.abs(m.predict(Xn[cols].values) - y))
                    m, cols = models[(f"{name}-stale", s)]
                    ae.setdefault(f"{name}-stale", []).append(np.abs(m.predict(Xs[cols].values) - y))
            ref = ae[REF][0]
            # Conservative reference: whichever formula strategy has the lower MAE in this cell
            # (an oracle choice made on the test data, which favours the formula).
            best_f = min(("Formula-drop", REF), key=lambda k: ae[k][0].mean())
            ref_best = ae[best_f][0]
            for meth, lst in ae.items():
                A = np.vstack(lst)
                per_seed = A.mean(1)
                if meth == REF:
                    d, lo, hi = 0.0, np.nan, np.nan
                else:
                    d, lo, hi = rb.boot_diff(ref, A.mean(0), complete.Date.values)
                db, lob, hib = rb.boot_diff(ref_best, A.mean(0), complete.Date.values)
                rows.append({"scenario": scen, "outage_days": L, "method": meth, "n": len(y),
                             "MAE": float(per_seed.mean()), "MAE_sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                             "dMAE_vs_ref": d, "CI_low": lo, "CI_high": hi,
                             "best_formula": best_f, "dMAE_vs_best_formula": db,
                             "CIb_low": lob, "CIb_high": hib})
            print(f"  {scen:<11} L={L}: " + ", ".join(f"{k}={np.vstack(v).mean():.2f}" for k, v in ae.items()))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "outage_results.csv"), index=False)
    piv = res.pivot_table(index=["scenario", "outage_days"], columns="method", values="MAE").reindex(
        [(s, L) for s in SCEN for L in (LENGTHS if s != "none" else [0])])
    piv.to_csv(os.path.join(OUT, "outage_mae_table.csv"))
    with open(os.path.join(OUT, "outage_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"complete-case rows {len(complete)} of {len(test)}; cities {complete.City.nunique()}\n")
        fh.write(f"dominant analyser shares: {pd.Series(dom).value_counts(normalize=True).round(3).to_dict()}\n")
        fh.write(piv.to_string(float_format=lambda v: f"{v:.2f}") + "\n\n")
        fh.write(res[res.method != REF].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(piv.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
