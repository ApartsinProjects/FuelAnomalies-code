"""P62: data-flow table for the trip panels (external review brief item 10) and the retained-vs-excluded comparison
for the 136-vehicle attribution panel, both from trip_features.parquet with the filters used by attrib_setup.py.
Output: data/processed/p62_panel_flow.json"""
import os, json, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
R = {"quality_filtered": {"trips": len(v), "vehicles": int(v.VehId.nunique())}}
lo, hi = v.fuel_per_100km.quantile([.01, .99]); t = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)]
R["fuel_tails_removed"] = {"trips": len(t), "vehicles": int(t.VehId.nunique())}
a = t[t.ltft1_cov >= 0.2].dropna(subset=["ltft1_mean"] + VF); R["trim_covered"] = {"trips": len(a), "vehicles": int(a.VehId.nunique())}
p = a[a.groupby("VehId").ltft1_mean.transform("size") >= 15]; R["attribution_panel"] = {"trips": len(p), "vehicles": int(p.VehId.nunique())}
ex = t.drop(p.index)
dist = "distance_km" if "distance_km" in t else [c for c in t.columns if "dist" in c][0]
for name, d in (("retained", p), ("excluded_from_fuel_trimmed", ex)):
    R[name] = {"trips": len(d), "median_fuel_l_100km": round(float(d.fuel_per_100km.median()), 2),
               "median_distance_km": round(float(d[dist].median()), 2), "median_duration_min": round(float(d.dur_min.median()), 1)}
assert R["attribution_panel"] == {"trips": 11304, "vehicles": 136}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p62_panel_flow.json"), "w"), indent=1); print(json.dumps(R, indent=1))
