"""Shared lean setup: frozen UAH kinematic signature + VED trim-covered injection panel (lines 1-75 of p33, no baselines)."""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import norm as N, binomtest
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
rng = np.random.default_rng(0)

def kin(spd):
    v = np.asarray(spd, float) / 3.6; a = np.diff(v) if len(v) > 1 else np.array([0.]); j = np.diff(a) if len(a) > 1 else np.array([0.])
    d = np.nansum(v) / 1000.0 if len(v) else 1e-9
    return [np.nanstd(v) * 3.6, np.nanpercentile(a, 95) if len(a) else 0.0, np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2)) if len(j) else 0.0]
ur, uy = [], []
for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
    d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); seg = np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]); d = d.assign(_s=seg)
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]; wid = ((rec["Timestamp (seconds)"] - t0) // 30).astype(int)
        for _, w in rec.groupby(wid):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            ur.append(kin(w["Speed (km/h)"])); uy.append(1 if c == 2 else 0)
Uu = np.array(ur); uy = np.array(uy); su = StandardScaler().fit(Uu); Uz = su.transform(Uu)
Wk = LogisticRegression(max_iter=1000).fit(Uz, uy).coef_[0]; dirn = Uz[uy == 1].mean(0) - Uz[uy == 0].mean(0)

v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
ENV = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb", "speed_mean", "pct_hwy"]
v["excess"] = v.fuel_per_100km - cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0),
                                                   v[ENV], v.fuel_per_100km, cv=GroupKFold(5), groups=v.VehId)
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean"] + VF).copy()
cnt = v.groupby("VehId")["ltft1_mean"].transform("size"); v = v[cnt >= 15].copy()
sv = StandardScaler().fit(v[VF]); Zv = sv.transform(v[VF]); kraw = Zv @ Wk; n = len(v); grp = v.VehId.to_numpy()
cz0 = ((v["ltft1_mean"] - v.groupby("VehId")["ltft1_mean"].transform("mean")) / (v.groupby("VehId")["ltft1_mean"].transform("std") + 1e-9)).to_numpy()
S_fault = 3.49 / v.groupby("VehId")["ltft1_mean"].std().median(); ksd = kraw.std()  # LiU-ICE smallest measured fueling deviation, pp (p37)
z = lambda x, ref: (x - ref.mean()) / ref.std()
kzn = z(kraw, kraw)

