"""P64: held-out variance of fleet excess explained by each axis, on the 11,304-trip attribution panel (same panel as
the signature contrast of Table 6; replaces the p10 values that were computed on 11,609 trim-covered trips).
Excess: conditions-only model, out-of-fold by vehicle. Axis regressions: gradient-boosted, out-of-fold by vehicle.
Output: data/processed/p64_axis_r2_panel.json"""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v.fuel_per_100km.quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)]
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
v = v[v.ltft1_cov >= 0.2].dropna(subset=["ltft1_mean", "stft1_mean"] + VF); v = v[v.groupby("VehId").ltft1_mean.transform("size") >= 15].reset_index(drop=True)
assert (len(v), v.VehId.nunique()) == (11304, 136), (len(v), v.VehId.nunique())
g = v.VehId.to_numpy(); GBM = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
CTX = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb", "speed_mean", "pct_hwy"]
KIN = ["speed_std", "accel_pos_mean", "accel_p95", "decel_p05", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km", "stops_per_km", "idle_frac", "vsp_mean"]
def excess(y): return y - cross_val_predict(GBM(), v[CTX], y, cv=GroupKFold(5), groups=g)
def r2x(cols, z):
    p = cross_val_predict(GBM(), v[cols].to_numpy(), z, cv=GroupKFold(5), groups=g); return round(float(1 - np.sum((z - p) ** 2) / np.sum((z - z.mean()) ** 2)), 3)
eb = excess(v.fuel_per_100km.to_numpy()); ec = excess((v.fuel_per_100km * (1 + (v.stft1_mean + v.ltft1_mean) / 100)).to_numpy())
R = {"n_trips": len(v), "n_vehicles": 136, "kinematic_block": r2x(KIN, eb), "ltft_only": r2x(["ltft1_mean"], eb), "stft_only": r2x(["stft1_mean"], eb),
     "both_trims": r2x(["stft1_mean", "ltft1_mean"], eb), "both_trims_trim_corrected_target": r2x(["stft1_mean", "ltft1_mean"], ec),
     "ignored_fuel_fraction_pct_median": round(float((v.stft1_mean + v.ltft1_mean).abs().median()), 2), "ignored_fuel_fraction_pct_p90": round(float((v.stft1_mean + v.ltft1_mean).abs().quantile(.9)), 2)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p64_axis_r2_panel.json"), "w"), indent=1); print(R)
