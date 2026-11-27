"""
NEW_ANALYSIS/scripts/07_data_audit.py
-------------------------------------
Data-integrity audit of city_day.csv (revision, red-team item M5).

Per city it reports:
  * share of rows where two pollutant columns are exactly equal (duplicated column)
  * share of rows in "stuck sensor" runs (same value >= 5 consecutive days)
  * negative or zero readings
  * how far the dataset AQI is from the CPCB formula on the daily means
  * which pollutant sets the formula AQI (dominant)
Outputs to NEW_ANALYSIS/data_audit/.
"""
import os
import itertools
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "data_audit")
os.makedirs(OUT, exist_ok=True)
_spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "05_cpcb_formula_baseline.py"))
fb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(fb)

POLL = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

raw = pd.read_csv(os.path.join(ROOT, "city_day.csv"), parse_dates=["Date"]).sort_values(["City", "Date"])

# --- duplicated columns ---------------------------------------------------------
dup_rows = []
for a, b in itertools.combinations(POLL, 2):
    both = raw[a].notna() & raw[b].notna() & (raw[a] > 0)
    eq = both & ((raw[a] - raw[b]).abs() < 1e-9)
    if eq.sum() == 0:
        continue
    g = pd.DataFrame({"City": raw["City"], "eq": eq, "both": both}).groupby("City").sum()
    for city, r in g.iterrows():
        if r["both"] > 0 and r["eq"] / r["both"] > 0.05:
            dup_rows.append({"City": city, "col_a": a, "col_b": b, "rows_both_observed": int(r["both"]),
                             "share_identical": r["eq"] / r["both"]})
dup = pd.DataFrame(dup_rows).sort_values("share_identical", ascending=False) if dup_rows else pd.DataFrame()
dup.to_csv(os.path.join(OUT, "duplicated_columns.csv"), index=False)

# --- stuck sensors (>= 5 consecutive identical daily values) ----------------------
def stuck_share(s):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return float(np.mean(run_len >= 5))

stuck = raw.groupby("City")[["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]].agg(stuck_share)
stuck.to_csv(os.path.join(OUT, "stuck_sensor_share.csv"))

# --- non-positive readings --------------------------------------------------------
nonpos = raw.groupby("City")[["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3"]].agg(lambda s: float(np.mean(s.dropna() <= 0)) if s.notna().any() else np.nan)
nonpos.to_csv(os.path.join(OUT, "nonpositive_share.csv"))

# --- AQI vs formula, per city -----------------------------------------------------
d = raw.dropna(subset=["AQI"]).copy()
f, dom, valid, _ = fb.formula_aqi(d, "NO2")
d["F"] = f; d["dom"] = dom
d["ratio"] = d["AQI"] / d["F"]
city = d.groupby("City").apply(lambda g: pd.Series({
    "rows_with_AQI": len(g),
    "median_AQI_over_formula": float(np.nanmedian(g["ratio"])),
    "MAE_formula": float(np.nanmean(np.abs(g["F"] - g["AQI"]))),
    "max_AQI": float(g["AQI"].max()),
    "CO_dominant_share": float(np.mean(g["dom"] == "CO")),
    "median_CO_mg": float(g["CO"].median()),
    "CO_equals_NO_share": float(np.mean((g["CO"] - g["NO"]).abs() < 1e-9)),
}), include_groups=False).sort_values("MAE_formula", ascending=False)
city.to_csv(os.path.join(OUT, "city_formula_consistency.csv"))

# --- Ahmedabad CO-equals-NO, by year ----------------------------------------------
ah = raw[raw.City == "Ahmedabad"].copy()
ah["eq"] = (ah["CO"] - ah["NO"]).abs() < 1e-9
by_year = ah.groupby(ah.Date.dt.year).agg(rows=("eq", "size"), co_eq_no=("eq", "mean"),
                                          median_CO=("CO", "median"), median_AQI=("AQI", "median"))
by_year.to_csv(os.path.join(OUT, "ahmedabad_by_year.csv"))

with open(os.path.join(OUT, "audit_summary.txt"), "w", encoding="utf-8") as fh:
    def w(s=""):
        print(s); fh.write(s + "\n")
    w("DUPLICATED COLUMNS (>5% identical rows within a city):")
    w(dup.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(dup) else "  none")
    w("\nSTUCK-SENSOR SHARE (rows inside >=5-day identical runs), cities with any value > 5%:")
    w(stuck[(stuck > 0.05).any(axis=1)].to_string(float_format=lambda v: f"{v:.3f}"))
    w("\nNON-POSITIVE READINGS, cities with any value > 1%:")
    w(nonpos[(nonpos > 0.01).any(axis=1)].to_string(float_format=lambda v: f"{v:.3f}"))
    w("\nCITY FORMULA CONSISTENCY (all years, rows with AQI):")
    w(city.to_string(float_format=lambda v: f"{v:.3f}"))
    w("\nAHMEDABAD BY YEAR:")
    w(by_year.to_string(float_format=lambda v: f"{v:.3f}"))
