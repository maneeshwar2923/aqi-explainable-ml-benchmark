"""
NEW_ANALYSIS/scripts/10_shap_and_figures.py
-------------------------------------------
TreeSHAP for the revised same-day CatBoost model (Ahmedabad excluded and included) and the
figures for the conference paper (vector PDF, IEEE single-column width 3.5 in).

Figures (NEW_ANALYSIS/paper_figures/)
  fig1_ahmedabad_co_no.pdf   CO vs NO scatter: Ahmedabad against every other city
  fig2_shap_vs_dominant.pdf  share of mean |SHAP| per analyser group, with and without Ahmedabad,
                             next to how often each analyser sets the formula AQI
  fig3_outage.pdf            MAE against outage length when the dominant analyser fails
  fig4_per_band.pdf          MAE by CPCB category, best ML model vs strongest formula
"""
import os, json, importlib.util, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("rb", os.path.join(HERE, "08_revised_benchmark.py"))
rb = importlib.util.module_from_spec(s); s.loader.exec_module(rb)
fb = rb.fb
NA = os.path.join(HERE, "..")
OUT = os.path.join(NA, "paper_figures"); os.makedirs(OUT, exist_ok=True)
GROUP_OF = {"PM2.5": "PM2.5", "PM10": "PM10", "NO": "NOx grp", "NO2": "NOx grp", "NOx": "NOx grp",
            "NH3": "NH3", "CO": "CO", "SO2": "SO2", "O3": "O3"}
ORDER = ["PM2.5", "PM10", "CO", "NOx grp", "O3", "SO2", "NH3"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"ml": "#1f4e9c", "ml2": "#6a9bd8", "form": "#c0392b", "form2": "#e8a09a", "grey": "#9e9e9e"}
W = 3.5


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def shap_shares(run_dir, include):
    _, P = rb.load(include)
    m = CatBoostRegressor(); m.load_model(os.path.join(run_dir, "catboost_S_seed0.cbm"))
    X = P["test"][rb.FEATS["S"]]
    sv = shap.TreeExplainer(m).shap_values(X.values)
    imp = pd.Series(np.abs(sv).mean(0), index=rb.FEATS["S"])
    grp = imp.groupby(GROUP_OF).sum()
    share = grp / grp.sum()
    dom = fb.formula_aqi(P["test"], "NO2")[1]
    dshare = pd.Series(dom).map(lambda d: "NOx grp" if d == "NO2" else d).value_counts(normalize=True)
    return imp, share.reindex(ORDER).fillna(0), dshare.reindex(ORDER).fillna(0), sv, X


def main():
    res = {}
    imp_x, sh_x, dom_x, sv, X = shap_shares(os.path.join(NA, "revised"), False)
    imp_a, sh_a, dom_a, _, _ = shap_shares(os.path.join(NA, "revised_with_ahmedabad"), True)
    tab = pd.DataFrame({"SHAP_share_exAhmedabad": sh_x, "SHAP_share_withAhmedabad": sh_a,
                        "dominant_share_exAhmedabad": dom_x, "dominant_share_withAhmedabad": dom_a})
    tab.to_csv(os.path.join(NA, "revised", "shap_vs_dominant.csv"))
    pd.DataFrame({"meanabs_exAhmedabad": imp_x, "meanabs_withAhmedabad": imp_a}).to_csv(
        os.path.join(NA, "revised", "shap_feature_meanabs.csv"))
    print(tab.round(3))

    # ---- Fig 1: CO vs NO ----
    raw = pd.read_csv(os.path.join(rb.ROOT, "city_day.csv"))
    fig, ax = plt.subplots(figsize=(W, 2.3))
    o = raw[raw.City != "Ahmedabad"]; a = raw[raw.City == "Ahmedabad"]
    ax.scatter(o.NO, o.CO, s=2, c=C["grey"], alpha=0.35, lw=0, label="25 other cities", rasterized=True)
    ax.scatter(a.NO, a.CO, s=3, c=C["form"], alpha=0.6, lw=0, label="Ahmedabad", rasterized=True)
    lim = [0, 160]
    ax.plot(lim, lim, "k--", lw=0.6, label="CO = NO")
    ax.set_xlim(lim); ax.set_ylim(0, 160)
    ax.set_xlabel(r"NO (µg/m$^3$)"); ax.set_ylabel(r"CO (mg/m$^3$)")
    ax.legend(loc="upper left", frameon=False, markerscale=3)
    save(fig, "fig1_ahmedabad_co_no")

    # ---- Fig 2: SHAP share vs dominant share ----
    fig, ax = plt.subplots(figsize=(W, 2.1))
    xi = np.arange(len(ORDER)); bw = 0.27
    ax.bar(xi - bw, sh_a.values * 100, bw, color=C["ml2"], label="SHAP share, with Ahmedabad")
    ax.bar(xi, sh_x.values * 100, bw, color=C["ml"], label="SHAP share, Ahmedabad excluded")
    ax.bar(xi + bw, dom_x.values * 100, bw, color=C["form"], label="Sets formula AQI (excl.)")
    ax.set_xticks(xi); ax.set_xticklabels(ORDER); ax.set_ylabel("Share (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=6)
    save(fig, "fig2_shap_vs_dominant")

    # ---- Fig 3: outage ----
    o = pd.read_csv(os.path.join(NA, "outage_experiment", "outage_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W + 0.3, 2.0), sharey=True)
    styles = {"Formula-drop": (C["form2"], "s", "--"), "Formula+LOCF": (C["form"], "s", "-"),
              "LightGBM-aug": (C["ml2"], "o", "--"), "LightGBM-stale": (C["ml"], "o", "-")}
    for ax, scen, title in zip(axes, ["dominant", "PM2.5+PM10"], ["Dominant analyser out", "PM2.5 and PM10 out"]):
        d = o[o.scenario == scen]
        for meth, (col, mk, ls) in styles.items():
            r = d[d.method == meth].sort_values("outage_days")
            ax.plot(r.outage_days, r.MAE, color=col, marker=mk, ls=ls, ms=3, lw=1, label=meth)
        base = o[(o.scenario == "none") & (o.method == "LightGBM")].MAE.iloc[0]
        ax.axhline(base, color="k", lw=0.5, ls=":")
        ax.set_title(title); ax.set_xticks([1, 3, 7]); ax.set_xlabel("Outage length (days)")
    axes[0].set_ylabel("MAE (AQI points)")
    axes[0].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6)
    save(fig, "fig3_outage")

    # ---- Fig 4: per band ----
    b = pd.read_csv(os.path.join(NA, "revised", "per_band.csv"))
    fig, ax = plt.subplots(figsize=(W, 1.9))
    xi = np.arange(len(b))
    ax.bar(xi - 0.2, b.MAE_ML, 0.4, color=C["ml"], label="Best ML model")
    ax.bar(xi + 0.2, b.MAE_formula, 0.4, color=C["form"], label="Two-day formula")
    ax.set_xticks(xi); ax.set_xticklabels([f"{n}\n(n={int(k)})" for n, k in zip(b.band, b.n)], fontsize=6)
    ax.set_ylabel("MAE (AQI points)"); ax.legend(frameon=False)
    save(fig, "fig4_per_band")

    print("figures written to", OUT)


if __name__ == "__main__":
    main()
