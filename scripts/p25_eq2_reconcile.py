"""P25: item 3 of EVAL_STRENGTHENING_PLAN.md - primary-target reconciliation.

The paper's primary target (Eq. 1) is fuel = MAF/14.7 (no trims); VED's own Algorithm 1 includes the trims
(Eq. 2: fuel x (1 + (STFT+LTFT)/100)). Every fuel-dependent real-data result is recomputed for BOTH targets
in ONE pass on ONE subset (trips with trims; STFT missing -> 0, as in p18), same model, same folds, same seed.
Results that do not use fuel at all (injected attribution accuracy 0.785/0.856, falsification controls) are
unaffected by construction and are not recomputed.

Recomputed per target: fuel-model OOF R2 (behaviour+conditions), env-only R2, behaviour-only R2, behaviour
marginal, vehicle-baseline share (p3 method); real-trip dissociation (p18 method: kinematic-dominant vs
trim-dominant excess, corr of excess with each axis); driver-injection fuel cost (p20 method).
PRE-REGISTERED: every recomputed headline changes by less than its reporting precision band
(R2 and shares +/- 0.02; dissociation +/- 0.15 L/100km and same sign; driver fuel cost +/- 0.5 pp).
A larger change means the target choice is load-bearing and Eq. 2 must become the primary target.
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw"); OUT = os.path.join(ROOT, "data", "processed")
BEHAV = ["speed_mean", "speed_std", "speed_p85", "accel_pos_mean", "accel_p95", "decel_p05", "jerk_rms",
         "harsh_accel_per_km", "harsh_brake_per_km", "idle_frac", "stops_per_km", "pct_hwy", "vsp_mean"]
ENV = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb"]
CTX = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb", "speed_mean", "pct_hwy"]
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]

v = pd.read_parquet(os.path.join(OUT, "trip_features.parquet"))

v = v.merge(pd.read_parquet(os.path.join(OUT, "signature_features.parquet")).rename(columns=lambda c: c if c in ("VehId", "Trip") else "sig_" + c), on=["VehId", "Trip"], how="left")   # source-definition signature features (p63), used only for the driver shift
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean"] + VF).copy()
v = v[v.groupby("VehId")["ltft1_mean"].transform("size") >= 15].reset_index(drop=True)
v["fuel_eq1"] = v["fuel_per_100km"]
v["fuel_eq2"] = v["fuel_per_100km"] * (1 + (v["stft1_mean"].fillna(0) + v["ltft1_mean"]) / 100.0)
g = v["VehId"].to_numpy()

def hgb(seed=0): return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=seed)
def oof(feats, y):
    X = v[feats].to_numpy(); pred = np.full(len(v), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = hgb().fit(X[tr], y[tr]).predict(X[te])
    return pred

# frozen UAH driver signature for the p20 driver-cost recomputation
def kin(spd):
    s = np.asarray(spd, float) / 3.6; a = np.diff(s) if len(s) > 1 else np.array([0.]); j = np.diff(a) if len(a) > 1 else np.array([0.])
    d = np.nansum(s) / 1000.0 if len(s) else 1e-9
    return [np.nanstd(s) * 3.6, np.nanpercentile(a, 95) if len(a) else 0.0, np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2)) if len(j) else 0.0]
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
sc = Uz @ Wk; S_driver = abs(sc[uy == 1].mean() - sc[uy == 0].mean()) / np.sqrt((sc[uy == 1].var() + sc[uy == 0].var()) / 2)

R = {"n_trips": int(len(v)), "n_vehicles": int(v.VehId.nunique())}
for tgt in ["fuel_eq1", "fuel_eq2"]:
    y = v[tgt].to_numpy(); SS = np.sum((y - y.mean()) ** 2); r2 = lambda p: 1 - np.sum((y - p) ** 2) / SS
    p_full = oof(BEHAV + ENV, y); p_env = oof(ENV, y); p_beh = oof(BEHAV, y)
    resid = y - p_full; fe = pd.Series(resid).groupby(g).transform("mean").to_numpy()
    res = {"r2_full": r2(p_full), "r2_env": r2(p_env), "r2_behav": r2(p_beh)}
    res["behav_marginal"] = res["r2_full"] - res["r2_env"]
    res["veh_share"] = float(np.sum((fe - resid.mean()) ** 2) / SS)
    # p18 dissociation
    ex = y - cross_val_predict(hgb(), v[CTX], y, cv=GroupKFold(5), groups=g)
    KIN = ["speed_std", "accel_pos_mean", "accel_p95", "decel_p05", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km", "vsp_mean"]
    gg = v.groupby("VehId")
    zk = ((v[KIN] - gg[KIN].transform("mean")) / (gg[KIN].transform("std") + 1e-9)).mean(axis=1).to_numpy()
    zc = ((v["ltft1_mean"] - gg["ltft1_mean"].transform("mean")) / (gg["ltft1_mean"].transform("std") + 1e-9)).abs().to_numpy()
    kp75, cp75, kp50, cp50 = np.quantile(zk, .75), np.quantile(zc, .75), np.quantile(zk, .5), np.quantile(zc, .5)
    res["kin_dominant_excess"] = float(ex[(zk > kp75) & (zc < cp50)].mean())
    res["trim_dominant_excess"] = float(ex[(zc > cp75) & (zk < kp50)].mean())
    res["corr_excess_kin"] = float(np.corrcoef(zk, ex)[0, 1]); res["corr_excess_trim"] = float(np.corrcoef(zc, ex)[0, 1])
    # p20 driver-injection fuel cost
    vs = v.copy()
    for c in VF: vs[c] = v["sig_" + c]                # the driver shift lives in the signature's feature space
    M = hgb().fit(vs[BEHAV + ENV].to_numpy(), y)
    sv = StandardScaler().fit(vs[VF]); kraw = sv.transform(vs[VF]) @ Wk
    raw_delta = ((S_driver * kraw.std() / (Wk @ dirn)) * dirn) * sv.scale_
    base = M.predict(vs[BEHAV + ENV].to_numpy()); vp = vs[BEHAV + ENV].copy()
    for i, c in enumerate(VF): vp[c] = np.clip(vs[c].to_numpy() + raw_delta[i], vs[c].quantile(.001), vs[c].quantile(.999))
    res["driver_fuel_cost_pct_median"] = float(np.median((M.predict(vp.to_numpy()) - base) / base * 100))
    R[tgt] = {k: round(float(x), 4) for k, x in res.items()}
    print(tgt, R[tgt], flush=True)

a, b = R["fuel_eq1"], R["fuel_eq2"]
band = {"r2_full": .02, "r2_env": .02, "r2_behav": .02, "behav_marginal": .02, "veh_share": .02,
        "kin_dominant_excess": .15, "trim_dominant_excess": .15, "driver_fuel_cost_pct_median": .5}
R["deltas"] = {k: round(b[k] - a[k], 4) for k in band}
R["within_band"] = {k: bool(abs(b[k] - a[k]) <= band[k]) for k in band}
R["same_sign_dissociation"] = bool(np.sign(a["kin_dominant_excess"]) == np.sign(b["kin_dominant_excess"]) and
                                   np.sign(a["trim_dominant_excess"]) == np.sign(b["trim_dominant_excess"]))
R["OUTCOME"] = "TARGET NOT LOAD-BEARING" if all(R["within_band"].values()) and R["same_sign_dissociation"] else "TARGET LOAD-BEARING (switch primary to Eq.2)"
json.dump(R, open(os.path.join(OUT, "p25_eq2_reconcile.json"), "w"), indent=2)
print(json.dumps({k: R[k] for k in ["deltas", "within_band", "same_sign_dissociation", "OUTCOME"]}, indent=2))
