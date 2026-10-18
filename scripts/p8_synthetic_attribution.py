"""P8 (must-fix #3): semi-synthetic ATTRIBUTION ACCURACY.

Inject two kinds of excess into REAL VED trips, with the SAME fuel increase for both
(so fuel magnitude alone cannot distinguish them):
  - DRIVER excess: add signal S (in units of natural within-vehicle variation) to the
    KINEMATIC axis; combustion axis untouched.
  - FAULT excess: add signal S to the COMBUSTION axis (fuel trims); kinematics untouched.
Real-trip deviations supply the confounding noise. Test whether a signature attributor
(kinematic + combustion deviations vs vehicle baseline) recovers the injected cause,
GroupKFold by vehicle, vs a fuel-only baseline that must sit at chance.

Addresses reviewer W6 (attribution never measured) and sidesteps W1/W2 (trim-blind target,
range restriction) by controlling the injected effect size.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = os.path.join(os.path.dirname(__file__), "..")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean", "stft1_mean"]).copy()
# vehicles with enough trimmed trips for a stable baseline
cnt = v.groupby("VehId")["ltft1_mean"].transform("size")
v = v[cnt >= 15].copy()
print(f"base trips: {len(v)}  vehicles: {v['VehId'].nunique()}")

KIN = ["accel_p95", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km", "speed_std", "vsp_mean"]
COMB = "ltft1_mean"

def z_within_vehicle(df, cols):
    """z-score each column relative to that trip's vehicle baseline (mean/std over the vehicle)."""
    g = df.groupby("VehId")
    z = (df[cols] - g[cols].transform("mean")) / (g[cols].transform("std") + 1e-9)
    return z

zk = z_within_vehicle(v, KIN).mean(axis=1).to_numpy()      # kinematic deviation score
zc = z_within_vehicle(v, [COMB])[COMB].to_numpy()          # combustion deviation score
groups = v["VehId"].to_numpy()
rng = np.random.default_rng(0)

def make_dataset(S):
    """Build driver+fault injected trips at signature strength S (in within-vehicle SD units)."""
    n = len(v)
    # driver: +S on kinematic axis, combustion untouched
    dk = zk + S;                 dc = zc.copy()
    # fault: +S on combustion axis, kinematics untouched
    fk = zk.copy();              fc = zc + S
    X = np.vstack([np.c_[dk, dc], np.c_[fk, fc]])
    y = np.r_[np.zeros(n), np.ones(n)]        # 0=driver, 1=fault
    g = np.r_[groups, groups]
    fuel_excess = np.r_[rng.normal(0.06, 0.01, n), rng.normal(0.06, 0.01, n)]  # SAME for both causes
    return X, y, g, fuel_excess

print("\n=== attribution accuracy vs signature strength S (within-vehicle SD units) ===")
print(f"{'S':>5} {'signature_acc':>14} {'signature_F1':>13} {'fuel_only_acc':>14}")
curve = []
for S in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    X, y, g, fe = make_dataset(S)
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                             X, y, cv=GroupKFold(5), groups=g)
    acc = accuracy_score(y, pred); f1 = f1_score(y, pred)
    # fuel-only baseline: can only use fuel excess magnitude (identical distribution for both)
    predf = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                              fe.reshape(-1, 1), y, cv=GroupKFold(5), groups=g)
    accf = accuracy_score(y, predf)
    curve.append((S, acc, f1, accf))
    print(f"{S:5.1f} {acc:14.3f} {f1:13.3f} {accf:14.3f}")

# ---- bench-calibrated operating point ----
# bench rich fault combustion shift in % fuel-per-air: (14.7/13.68-1)-(14.7/14.07-1)
bench_shift_pct = (14.7/13.68 - 1)*100 - (14.7/14.07 - 1)*100
ltft_sd = v.groupby("VehId")["ltft1_mean"].std().median()   # natural within-vehicle LTFT SD (%)
S_bench = bench_shift_pct / ltft_sd
print(f"\nbench rich-fault combustion shift = {bench_shift_pct:.1f}% fuel/air; "
      f"VED within-vehicle LTFT SD (median) = {ltft_sd:.2f}%  =>  S_bench ~= {S_bench:.2f}")
# accuracy at S_bench
Xb, yb, gb, feb = make_dataset(S_bench)
predb = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                          Xb, yb, cv=GroupKFold(5), groups=gb)
print(f"attribution accuracy at bench-calibrated S={S_bench:.2f}: {accuracy_score(yb, predb):.3f}  "
      f"F1={f1_score(yb, predb):.3f}")
print("confusion (rows=true [driver,fault], cols=pred):")
print(confusion_matrix(yb, predb))

pd.DataFrame(curve, columns=["S","signature_acc","signature_f1","fuel_only_acc"]).to_csv(
    os.path.join(ROOT, "data", "processed", "attribution_curve.csv"), index=False)
print("\nwrote data/processed/attribution_curve.csv")
