"""P3: variance decomposition of trip fuel into DRIVING-BEHAVIOUR vs VEHICLE-BASELINE
vs ENVIRONMENT vs UNEXPLAINED, plus a behaviour counterfactual and RQ4 preview
(how much of the vehicle effect is static engine size/weight).

All models use OOF predictions with GroupKFold by vehicle (no vehicle-ID leakage).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
stat = pd.read_excel(os.path.join(RAW, "VED_Static_ICE_HEV.xlsx"))
df = df.merge(stat[["VehId", "Vehicle Class", "Engine Configuration & Displacement"]],
              on="VehId", how="left")

BEHAV = ["speed_mean", "speed_std", "speed_p85", "accel_pos_mean", "accel_p95",
         "decel_p05", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km",
         "idle_frac", "stops_per_km", "pct_hwy", "vsp_mean"]
ENV   = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb"]
TARGET = "fuel_per_100km"

d = df.dropna(subset=[TARGET]).copy()
lo, hi = d[TARGET].quantile([.01, .99]); d = d[(d[TARGET] >= lo) & (d[TARGET] <= hi)].copy()
y = d[TARGET].to_numpy(); g = d["VehId"].to_numpy()
SS_tot = np.sum((y - y.mean())**2)

def oof_r2(feats, seed=0):
    X = d[feats].to_numpy(); pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          max_depth=6, random_state=seed)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    r2 = 1 - np.sum((y - pred)**2) / SS_tot
    return r2, pred

r2_full, pred_full = oof_r2(BEHAV + ENV)
r2_env, _ = oof_r2(ENV)                       # env+weight only
r2_behav, _ = oof_r2(BEHAV)                   # behaviour only

# vehicle fixed effect on full-model residual
d["resid"] = y - pred_full
fe = d.groupby("VehId")["resid"].mean()
d["veh_fe"] = d["VehId"].map(fe)
SS_between = np.sum((d["veh_fe"] - d["resid"].mean())**2)   # trip-weighted between-veh SS
veh_share = SS_between / SS_tot

# ---- variance decomposition (sequential, order noted) ----
behav_marginal = r2_full - r2_env            # behaviour's marginal R2 over env
env_marginal   = r2_full - r2_behav          # env's marginal over behaviour
unexplained = 1 - r2_full - veh_share
print("=== VARIANCE DECOMPOSITION of trip fuel_per_100km (n=%d, %d veh) ===" %
      (len(d), d["VehId"].nunique()))
print(f"  R2 full (behaviour+env, no vehicle ID) : {r2_full:.3f}")
print(f"  R2 env-only (route/temp/aux/weight)    : {r2_env:.3f}")
print(f"  R2 behaviour-only                      : {r2_behav:.3f}")
print(f"  -> behaviour MARGINAL over env         : {behav_marginal:.3f}")
print(f"  -> env MARGINAL over behaviour         : {env_marginal:.3f}")
print(f"  vehicle-baseline (fixed effect) share  : {veh_share:.3f}")
print(f"  unexplained (residual noise)           : {unexplained:.3f}")

# ---- RQ4 preview: how much of the vehicle effect is static engine size/weight? ----
disp = d["Engine Configuration & Displacement"].astype(str).str.extract(r"(\d\.\d)")[0]
d["disp_L"] = pd.to_numeric(disp, errors="coerce")
veh_tbl = d.groupby("VehId").agg(fe=("veh_fe","first"), n=("resid","size"),
                                 disp=("disp_L","first"), wt=("weight_lb","first")).dropna(subset=["fe"])
veh_tbl = veh_tbl[veh_tbl["n"] >= 15]
vv = veh_tbl.dropna(subset=["disp","wt"])
from numpy.polynomial import polynomial as _  # noqa
import numpy.linalg as la
Xs = np.column_stack([np.ones(len(vv)), vv["disp"], vv["wt"]])
beta, *_ = la.lstsq(Xs, vv["fe"].to_numpy(), rcond=None)
pred_fe = Xs @ beta
r2_static = 1 - np.sum((vv["fe"]-pred_fe)**2)/np.sum((vv["fe"]-vv["fe"].mean())**2)
print(f"\n=== RQ4: vehicle fixed effect explained by static params ===")
print(f"  FE ~ displacement + weight : R2 = {r2_static:.3f}  (n={len(vv)} vehicles)")
print(f"  => ~{r2_static*100:.0f}% of vehicle-baseline is engine size/weight; "
      f"~{(1-r2_static)*100:.0f}% is persistent unexplained per-vehicle effect.")

# ---- behaviour counterfactual: gentle-driving reference ----
AGG = ["accel_pos_mean", "accel_p95", "jerk_rms", "harsh_accel_per_km",
       "harsh_brake_per_km", "speed_std"]
m_full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
m_full.fit(d[BEHAV + ENV].to_numpy(), y)
cf = d[BEHAV + ENV].copy()
for c in AGG:                              # set aggression to gentle (20th pct) reference
    cf[c] = d[c].quantile(0.20)
d["pred_actual"] = m_full.predict(d[BEHAV + ENV].to_numpy())
d["pred_gentle"] = m_full.predict(cf.to_numpy())
d["behav_comp"] = (d["pred_actual"] - d["pred_gentle"]).clip(lower=0)
print(f"\n=== behaviour counterfactual (fuel above gentle-driving reference) ===")
print(f"  behaviour component L/100km: median={d['behav_comp'].median():.2f}  "
      f"mean={d['behav_comp'].mean():.2f}  p90={d['behav_comp'].quantile(.9):.2f}")
# monotonicity vs an independent aggression index (harsh events + jerk)
d["agg_idx"] = (d["harsh_accel_per_km"] + d["harsh_brake_per_km"])
dec = pd.qcut(d["agg_idx"].rank(method="first"), 10, labels=False)
mono = d.groupby(dec)["behav_comp"].mean()
print("  behaviour component by aggression decile (0=gentle..9=harsh):")
print("   ", np.round(mono.to_numpy(), 2).tolist())

d[["VehId","Trip","fuel_per_100km","pred_actual","behav_comp","veh_fe","resid"]].to_csv(
    os.path.join(ROOT, "data", "processed", "trip_decomposition.csv"), index=False)
print("\nwrote data/processed/trip_decomposition.csv")
