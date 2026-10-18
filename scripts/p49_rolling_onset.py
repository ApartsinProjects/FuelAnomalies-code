"""P49: rolling per-vehicle trim reference (Section 8.7), reconstructed as a standalone script.
Each trip's long-term trim is referenced to the same vehicle's previous K trips: z = (L_i - mean_prev) / sd_prev.
The threshold is calibrated to 10% false alarms on November-April trips and applied to May-October.
Onset: a persistent step of s pp starting at trip i; first-trip detection = P(z_i + s / sd_prev > t).
Invariant: must reproduce the published values (false alarms 0.091/0.093/0.087 at K 10/20/40; first-trip detection
0.48 at 2.98 pp, 0.90 at 6 pp); then reports the LiU-ICE anchor, 3.49 pp."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean"] + VF).copy()
v = v[v.groupby("VehId")["ltft1_mean"].transform("size") >= 15].reset_index(drop=True)
v["m"] = pd.to_datetime(v["week"].str.extract(r"VED_(\d{6})_")[0], format="%y%m%d").dt.month
v = v.sort_values(["VehId", "Trip"]).reset_index(drop=True)
R = {}
for side in ("one", "two"):
    for K in (10, 20, 40):
        g = v.groupby("VehId")["ltft1_mean"]
        mu = g.transform(lambda s: s.shift(1).rolling(K, min_periods=K).mean()); sd = g.transform(lambda s: s.shift(1).rolling(K, min_periods=K).std())
        z = (v.ltft1_mean - mu) / (sd + 1e-9); ok = z.notna()
        cal = ok & v.m.isin([11, 12, 1, 2, 3, 4]); te = ok & v.m.isin([5, 6, 7, 8, 9, 10])
        stat = z if side == "one" else z.abs()
        t = float(np.quantile(stat[cal], 0.90)); fa = float((stat[te] > t).mean())
        out = {"tau": round(t, 3), "false_alarm_test": round(fa, 3), "n_test": int(te.sum())}
        if K == 20:
            det = {}
            for s in (1.5, 2.98, 3.49, 6, 8.0, 10, 15, 19.5):
                zz = z[ok] + s / (sd[ok] + 1e-9); det[str(s)] = round(float(((zz if side == "one" else zz.abs()) > t).mean()), 3)
            out["first_trip_detection"] = det
        R[f"{side}_sided_K{K}"] = out
R["checks"] = {"reproduces_published": any(abs(R[f"{s}_sided_K20"]["false_alarm_test"] - 0.093) < 0.005 and abs(R[f"{s}_sided_K20"]["first_trip_detection"]["2.98"] - 0.48) < 0.02 for s in ("one", "two"))}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p49_rolling_onset.json"), "w"), indent=1); print(json.dumps(R, indent=1))
