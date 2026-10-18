"""P33: attribution baselines on the SAME p24 feature-level injection (one pass, one config, one artifact).

Every method sees the same 2n variants of the same real VED trips (driver variant: kinematic features shifted
along the UAH aggressive direction to S_driver; fault variant: within-vehicle LTFT z shifted by S_fault = 3.49 pp /
within-vehicle SD; fuel held equal). Two operating points: window-level S_driver 0.65, trip-level 1.3.

Methods
 OURS-margin  frozen UAH kinematic score vs within-vehicle LTFT z, argmax (paper headline, p24).
 OURS-LR      source-calibrated likelihood-ratio fusion (Algorithm 1, p19), forced decision (no abstention).
 B1 SHAP-style fuel-anomaly explanation (Barbado & Corcho 2022 family): gradient-boosted model of the trip's
    context-residual fuel excess on the kinematic features + within-vehicle LTFT z, trained on REAL VED trips
    (GroupKFold by vehicle, out-of-fold); a variant is attributed to the feature group (kinematic vs trim) with the
    larger exact two-player Shapley contribution to predicted excess, reference = fleet-median feature vector.
 B2 Weak supervision: proxy labels on REAL trips from axis thresholds (driver: kin z > 1 and LTFT z < 0.5;
    fault: LTFT z > 1 and kin z < 0.5), logistic regression on the 4 kinematic features + LTFT z, trained on
    proxy-labelled trips of training vehicles, applied to held-out vehicles' variants (GroupKFold by vehicle).
 B3 Pooled-source supervised transfer: one logistic classifier trained on the two SOURCE datasets pooled
    (UAH windows: aggressive -> driver, normal -> none; EngineFaultDB: rich -> fault, normal -> none), each
    domain standardized on its own data, the missing axis imputed at its normal mean (0); applied to the target
    variants; prediction = argmax(P(driver), P(fault)).
 B3b as B3 with class_weight="balanced" (added after the first run: the sources differ ~30x in size and
    unweighted B3 called nearly everything "fault"; the stronger of B3/B3b is the one to cite).
 FUEL-ONLY    matched fuel increase only (chance by construction).
 ORACLE       logistic on 4 kinematic features + LTFT z trained on the target INJECTION labels (GroupKFold by
    vehicle); an upper reference no deployable method has.

PRE-REGISTERED (fixed before the first run):
 P1 At trip level (1.3), OURS-LR accuracy exceeds B1 by >= 0.10, vehicle-cluster bootstrap 95% CI of the
    difference excludes 0.
 P2 At trip level, OURS-LR is non-inferior to B2 and B3: difference >= -0.01 (CI lower bound reported).
 I1 invariant: FUEL-ONLY accuracy within 0.50 +/- 0.01.  I2 invariant: ORACLE >= OURS-LR - 0.005.
 I3 invariant: OURS-margin reproduces p24 exactly.
Output: data/processed/p33_baselines.json.
"""
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

# ---- B1 fuel model on REAL trips (out-of-fold by vehicle); per-fold models kept for scoring variants ----
Xreal = np.c_[Zv, cz0]; gkf = list(GroupKFold(5).split(Xreal, groups=grp)); fold_of = np.empty(n, int); b1_models = []
for k, (tr, te) in enumerate(gkf):
    b1_models.append(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0).fit(Xreal[tr], v.excess.to_numpy()[tr])); fold_of[te] = k
ref = np.median(Xreal, axis=0)
def b1_attr(X):
    out = np.empty(len(X), int)
    for k, m in enumerate(b1_models):
        idx = np.flatnonzero(np.r_[fold_of, fold_of] == k) if len(X) == 2 * n else None
        Xi = X[idx]; Rk = np.tile(ref, (len(Xi), 1)); Rc = np.tile(ref, (len(Xi), 1))
        Rk[:, :4] = Xi[:, :4]; Rc[:, 4] = Xi[:, 4]                     # coalitions: {kin}, {trim}
        f_all, f_none, f_k, f_c = m.predict(Xi), m.predict(np.tile(ref, (len(Xi), 1))), m.predict(Rk), m.predict(Rc)
        phi_k = 0.5 * ((f_k - f_none) + (f_all - f_c)); phi_c = 0.5 * ((f_c - f_none) + (f_all - f_k))
        out[idx] = (phi_c > phi_k).astype(int)
    return out

# ---- B2 weak supervision on REAL trips ----
kz_real = kzn
prox = np.full(n, -1); prox[(kz_real > 1) & (cz0 < 0.5)] = 0; prox[(cz0 > 1) & (kz_real < 0.5)] = 1
def b2_attr(X):
    out = np.empty(len(X), int); g2 = np.r_[grp, grp]
    for tr, te in gkf:
        trm = np.zeros(n, bool); trm[tr] = True; lab = trm & (prox >= 0)
        m = LogisticRegression(max_iter=1000).fit(Xreal[lab], prox[lab])
        tev = np.isin(g2, np.unique(grp[te])); out[tev] = m.predict(X[tev])
    return out

# ---- B3 pooled-source supervised transfer ----
e = pd.read_csv(os.path.join(RAW, "EngineFaultDB_Final.csv")); e = e[e.Fault.isin([0, 1])]
ec = (14.7 / e.AFR - 1) * 100; ecz = (ec - ec[e.Fault == 0].mean()) / ec[e.Fault == 0].std()
Xs = np.r_[np.c_[Uz, np.zeros(len(Uz))], np.c_[np.zeros((len(e), 4)), ecz.to_numpy()]]
ys = np.r_[np.where(uy == 1, 1, 0), np.where(e.Fault.to_numpy() == 1, 2, 0)]   # 0 none, 1 driver, 2 fault
b3 = LogisticRegression(max_iter=2000).fit(Xs, ys)
b3b = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, ys)   # post-run fix: sources differ ~30x in size
ea = pd.read_csv(os.path.join(RAW, "EngineFaultDB_Final.csv")); ea = ea[ea.Fault.isin([0, 1, 2])].copy()
ea["blk"] = ea.groupby("Fault").cumcount() // 1000; bafr = ea.groupby(["Fault", "blk"]).AFR.transform("mean")
rich_c = (bafr < 12.5).to_numpy(); norm_c = ((ea.Fault == 0) & (bafr >= 12.5)).to_numpy(); keep = rich_c | norm_c
ec3 = (14.7 / ea.AFR - 1) * 100; ec3z = ((ec3 - ec3[norm_c].mean()) / ec3[norm_c].std()).to_numpy()[keep]
Xs3 = np.r_[np.c_[Uz, np.zeros(len(Uz))], np.c_[np.zeros((keep.sum(), 4)), ec3z]]
ys3 = np.r_[np.where(uy == 1, 1, 0), np.where(rich_c[keep], 2, 0)]
b3c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs3, ys3)   # measured-mixture bench labels (p47)
def b3_attr(X, m=None):
    P = (m or b3).predict_proba(X); return (P[:, 2] > P[:, 1]).astype(int)

def boot_diff(c1, c2, g2, B=1000):
    vehs = np.unique(g2); idx = {u: np.flatnonzero(g2 == u) for u in vehs}; d = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(vehs, len(vehs))]); d.append(c1[s].mean() - c2[s].mean())
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]

R = {"n_trips": int(n), "n_vehicles": int(len(np.unique(grp))), "S_fault": round(float(S_fault), 2)}
for name, S_d in [("window_d0.65", 0.65), ("trip_d1.3", 1.3)]:
    PERT = (S_d * ksd / (Wk @ dirn)) * dirn
    Xd = np.c_[Zv + PERT, cz0]; Xf = np.c_[Zv, cz0 + S_fault]; X = np.r_[Xd, Xf]; y = np.r_[np.zeros(n), np.ones(n)].astype(int); g2 = np.r_[grp, grp]
    kd = z((Zv + PERT) @ Wk, kraw); KZ = np.r_[kd, kzn]; CZ = np.r_[z(cz0, cz0), z(cz0 + S_fault, cz0)]
    pred = {"OURS-margin": (CZ - KZ > 0).astype(int),
            "OURS-LR": ((N.logpdf(CZ, S_fault, 1) - N.logpdf(CZ, 0, 1)) > (N.logpdf(KZ, S_d, 1) - N.logpdf(KZ, 0, 1))).astype(int),
            "B1-SHAP-fuel": b1_attr(X), "B2-weak-sup": b2_attr(X), "B3-pooled-source": b3_attr(X), "B3b-pooled-balanced": b3_attr(X, b3b), "B3c-pooled-mixture": b3_attr(X, b3c),
            "FUEL-ONLY": rng.integers(0, 2, 2 * n)}
    orc = np.empty(2 * n, int)
    for tr, te in GroupKFold(5).split(X, y, groups=g2):
        orc[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict(X[te])
    pred["ORACLE"] = orc
    corr = {k: (p == y).astype(float) for k, p in pred.items()}
    blk = {k: {"accuracy": round(float(c.mean()), 3), "driver_recall": round(float(c[:n].mean()), 3), "fault_recall": round(float(c[n:].mean()), 3)} for k, c in corr.items()}
    for b in ["B1-SHAP-fuel", "B2-weak-sup", "B3-pooled-source", "B3b-pooled-balanced", "B3c-pooled-mixture", "OURS-margin"]:
        a, bb = corr["OURS-LR"], corr[b]; n01 = int(((a == 1) & (bb == 0)).sum()); n10 = int(((a == 0) & (bb == 1)).sum())
        blk[b]["diff_vs_OURS-LR"] = round(float(a.mean() - bb.mean()), 3); blk[b]["diff_CI95"] = boot_diff(a, bb, g2)
        blk[b]["mcnemar_p"] = float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    R[name] = blk; print(name, json.dumps(blk, indent=1), flush=True)
t = R["trip_d1.3"]
R["checks"] = {"P1": t["B1-SHAP-fuel"]["diff_vs_OURS-LR"] >= 0.10 and t["B1-SHAP-fuel"]["diff_CI95"][0] > 0,
               "P2_B2": t["B2-weak-sup"]["diff_vs_OURS-LR"] >= -0.01, "P2_B3": t["B3-pooled-source"]["diff_vs_OURS-LR"] >= -0.01, "P2_B3b": t["B3b-pooled-balanced"]["diff_vs_OURS-LR"] >= -0.01,
               "I1": abs(t["FUEL-ONLY"]["accuracy"] - 0.5) <= 0.01 and abs(R["window_d0.65"]["FUEL-ONLY"]["accuracy"] - 0.5) <= 0.01,
               "I2": t["ORACLE"]["accuracy"] >= t["OURS-LR"]["accuracy"] - 0.005,
               "I3": R["window_d0.65"]["OURS-margin"]["accuracy"] == json.load(open(os.path.join(ROOT, "data", "processed", "p24_operating_points.json")))["window_level_d0.65"]["accuracy"] and t["OURS-margin"]["accuracy"] == json.load(open(os.path.join(ROOT, "data", "processed", "p24_operating_points.json")))["trip_level_d1.3"]["accuracy"]}
print(json.dumps(R["checks"]))
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p33_baselines.json"), "w"), indent=2)
