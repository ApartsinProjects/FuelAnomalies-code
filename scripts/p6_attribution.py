"""P6: formalize signature-based attribution as a DOUBLE DISSOCIATION.

Claim: excess fuel is attributable by which axis it loads on.
 - KINEMATIC axis (transient aggression): explains DRIVER-excess (VED), absent on bench.
 - COMBUSTION axis (AFR/fuel-trim deviation from stoich): explains FAULT-excess (bench),
   silent for driver-excess (VED healthy fleet).
The combustion axis is measured in BOTH worlds and put on a common scale (% fuel-air vs stoich):
 VED: (STFT+LTFT)% ; Bench: (14.7/AFR - 1)*100.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from scipy.stats import pearsonr

ROOT = os.path.join(os.path.dirname(__file__), "..")

def r2_of(X, y, groups=None):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
    if groups is not None:
        pred = cross_val_predict(m, X, y, cv=GroupKFold(5), groups=groups)
    else:
        pred = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)

# ================= VED (driver-excess) =================
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99])
v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
# excess = fuel not explained by trip CONTEXT (route/temp/aux/weight) -> leaves behaviour+vehicle
ENV = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb", "speed_mean", "pct_hwy"]
m_env = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
v["excess"] = v["fuel_per_100km"].to_numpy() - cross_val_predict(
    m_env, v[ENV], v["fuel_per_100km"], cv=GroupKFold(5), groups=v["VehId"])
KIN = ["speed_std", "accel_pos_mean", "accel_p95", "decel_p05", "jerk_rms",
       "harsh_accel_per_km", "harsh_brake_per_km", "stops_per_km", "idle_frac", "vsp_mean"]
vt = v[v["ltft1_cov"] >= 0.2].dropna(subset=["stft1_mean", "ltft1_mean"]).copy()
COMB_V = ["stft1_mean", "ltft1_mean"]
print("=== VED (driver-excess), n=%d trips (trim subset n=%d) ===" % (len(v), len(vt)))
r2_kin = r2_of(v[KIN].to_numpy(), v["excess"].to_numpy(), v["VehId"].to_numpy())
r2_comb_v = r2_of(vt[COMB_V].to_numpy(), vt["excess"].to_numpy(), vt["VehId"].to_numpy())
print(f"  excess explained by KINEMATIC axis : R2 = {r2_kin:+.3f}")
print(f"  excess explained by COMBUSTION axis: R2 = {r2_comb_v:+.3f}  (fuel trims)")
# common combustion axis (% fuel-air vs stoich) correlation
vt["comb_pct"] = vt["stft1_mean"] + vt["ltft1_mean"]
rc, pc = pearsonr(vt["comb_pct"], vt["excess"])
print(f"  corr(excess, combustion% [STFT+LTFT]) = {rc:+.3f} (p={pc:.1e})")

# ================= EngineFaultDB (fault-excess) =================
e = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv"))
OP = ["RPM", "MAP", "TPS", "Force", "Power", "Speed"]
norm = e[e.Fault == 0]
M = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
M.fit(norm[OP].to_numpy(), norm["Consumption L/100KM"].to_numpy())
e["excess"] = e["Consumption L/100KM"].to_numpy() - M.predict(e[OP].to_numpy())
rich = e[e.Fault == 1].copy()          # rich mixture = the fuel-relevant fault
COMB_E = ["AFR", "Lambda", "CO", "HC"]
print("\n=== EngineFaultDB (fault-excess, RICH), n=%d ===" % len(rich))
r2_comb_e = r2_of(rich[COMB_E].to_numpy(), rich["excess"].to_numpy())
print(f"  excess explained by COMBUSTION axis: R2 = {r2_comb_e:+.3f}  (AFR/lambda/CO/HC)")
print(f"  (KINEMATIC axis absent: bench = steady-state operating points, no driver transients)")
rich["comb_pct"] = (14.7 / rich["AFR"] - 1) * 100
rc2, pc2 = pearsonr(rich["comb_pct"], rich["excess"])
print(f"  corr(excess, combustion% [14.7/AFR-1]) = {rc2:+.3f} (p={pc2:.1e})")

# ================= DOUBLE DISSOCIATION TABLE =================
print("\n" + "="*60)
print("DOUBLE DISSOCIATION (variance of excess explained, R2)")
print("="*60)
print(f"{'axis':<20}{'DRIVER-excess (VED)':>22}{'FAULT-excess (bench)':>22}")
print(f"{'kinematic':<20}{r2_kin:>22.3f}{'n/a (steady-state)':>22}")
print(f"{'combustion':<20}{r2_comb_v:>22.3f}{r2_comb_e:>22.3f}")
print("\nInterpretation: kinematic axis explains driver-excess; combustion axis explains")
print("fault-excess but is SILENT for driver-excess -> the two causes are separable by signature.")
