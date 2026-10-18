"""P61: prequential (chronological, causal) evaluation of the rolling trim reference by history length, and per-driver
results of the frozen UAH signature (external review brief items 7, 9).
(1) Each vehicle's trips in order; trip t is scored only against its previous K trips (K = 5, 10, 20, 50; burn-in K).
    Two-sided threshold calibrated to 10% false alarms on November-April trips, applied to May-October trips (as in
    Appendix E). Reported per K: trips and vehicles scored, false alarms per 1,000 trips and per 1,000 driving hours on
    the held-out season, and first-trip detection of +3.49 / +6 / +10 pp steps.
(2) UAH-DriveSet, 30 s windows, frozen four-feature speed signature: leave-one-driver-out AUC per held-out driver and
    the driver-level spread (6 drivers), against the pooled 0.694.
Output: data/processed/p61_history_and_drivers.json"""
import os, glob, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean"] + VF).copy(); v = v[v.groupby("VehId")["ltft1_mean"].transform("size") >= 15]
v["m"] = pd.to_datetime(v["week"].str.extract(r"VED_(\d{6})_")[0], format="%y%m%d").dt.month; v = v.sort_values(["VehId", "Trip"]).reset_index(drop=True)
R = {}
for K in (5, 10, 20, 50):
    gL = v.groupby("VehId")["ltft1_mean"]
    mu = gL.transform(lambda s: s.shift(1).rolling(K, min_periods=K).mean()); sd = gL.transform(lambda s: s.shift(1).rolling(K, min_periods=K).std())
    z = (v.ltft1_mean - mu) / (sd + 1e-9); ok = z.notna() & (sd > 0)
    cal = ok & v.m.isin([11, 12, 1, 2, 3, 4]); te = ok & v.m.isin([5, 6, 7, 8, 9, 10])
    t = float(np.quantile(z[cal].abs(), 0.90)); fa = (z[te].abs() > t)
    hours = v.dur_min[te].sum() / 60
    R[f"K{K}"] = {"trips_scored": int(ok.sum()), "vehicles_scored": int(v.VehId[ok].nunique()), "threshold": round(t, 3),
                  "false_alarms_per_1000_trips": round(float(fa.mean() * 1000), 1), "false_alarms_per_1000_hours": round(float(fa.sum() / hours * 1000), 1),
                  "first_trip_detection": {str(s): round(float(((z[ok] + s / (sd[ok] + 1e-9)).abs() > t).mean()), 3) for s in (3.49, 6.0, 10.0)}}
# per-driver UAH
X, y, gdr = [], [], []
def kin(spd):
    vv = np.asarray(spd, float) / 3.6; a = np.diff(vv); j = np.diff(a); d = np.nansum(vv) / 1000.0
    return [np.nanstd(vv) * 3.6, np.nanpercentile(a, 95), np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2))]
for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
    drv = os.path.basename(f).split("_")[0]; d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); d = d.assign(_s=np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]))
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]
        for _, w in rec.groupby(((rec["Timestamp (seconds)"] - t0) // 30).astype(int)):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            X.append(kin(w["Speed (km/h)"])); y.append(int(c == 2)); gdr.append(drv)
X, y, gdr = np.array(X), np.array(y), np.array(gdr); pooled = np.zeros(len(y)); per = {}
for k in np.unique(gdr):
    tr = gdr != k; s = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=1000).fit(s.transform(X[tr]), y[tr])
    pooled[~tr] = m.decision_function(s.transform(X[~tr]))
    per[k] = {"windows": int((~tr).sum()), "aggressive": int(y[~tr].sum()), "auc": round(float(roc_auc_score(y[~tr], pooled[~tr])), 3) if 0 < y[~tr].sum() < (~tr).sum() else None}
R["uah_lodo"] = {"pooled_auc": round(float(roc_auc_score(y, pooled)), 3), "per_driver": per,
                 "per_driver_auc_range": [min(p["auc"] for p in per.values() if p["auc"]), max(p["auc"] for p in per.values() if p["auc"])]}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p61_history_and_drivers.json"), "w"), indent=1); print(json.dumps(R, indent=1))
# robust variant (brief item 8): rolling median and MAD instead of mean and SD, K = 20, same calibration and test seasons
gL = v.groupby("VehId")["ltft1_mean"]; K = 20
med = gL.transform(lambda s: s.shift(1).rolling(K, min_periods=K).median())
mad = gL.transform(lambda s: s.shift(1).rolling(K, min_periods=K).apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)) * 1.4826
z = (v.ltft1_mean - med) / (mad + 1e-9); ok = z.notna() & (mad > 0)
cal = ok & v.m.isin([11, 12, 1, 2, 3, 4]); te = ok & v.m.isin([5, 6, 7, 8, 9, 10]); t = float(np.quantile(z[cal].abs(), 0.90)); fa = z[te].abs() > t
R["K20_median_mad"] = {"trips_scored": int(ok.sum()), "vehicles_scored": int(v.VehId[ok].nunique()), "false_alarms_per_1000_trips": round(float(fa.mean() * 1000), 1),
                       "false_alarms_per_1000_hours": round(float(fa.sum() / (v.dur_min[te].sum() / 60) * 1000), 1),
                       "first_trip_detection": {str(s): round(float(((z[ok] + s / (mad[ok] + 1e-9)).abs() > t).mean()), 3) for s in (3.49, 6.0, 10.0)}}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p61_history_and_drivers.json"), "w"), indent=1); print("median_mad", R["K20_median_mad"])
