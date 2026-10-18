"""P26: evaluation package items 1 (seasonal placebo + de-seasonalised combustion axis) and
2 (zero-effect control + vehicle-level bootstrap CIs). See EVAL_STRENGTHENING_PLAN.md.

Shared setup is identical to p17/p19 (frozen UAH driver signature; VED trip filter; within-vehicle
standardized LTFT as the combustion axis; S_fault = 3.49 pp / median within-vehicle LTFT SD).

ITEM 1 operational definitions (fixed before running):
  Vehicles with >= 10 winter (Dec-Feb) and >= 10 summer (Jun-Aug) trips. Reference period R and test
  period T, both directions (R=winter,T=summer and R=summer,T=winter), pooled.
  RAW axis      : cz = (LTFT - mean_R) / sd_R per vehicle (baseline learned in R, as at commissioning).
  DESEASON axis : LTFT minus a fleet month effect estimated LEAVE-VEHICLE-OUT (mean of other vehicles'
                  annually-centred LTFT in that month), then standardized on R per vehicle.
  Detector      : two-sided |cz| > t, t = pooled 90th percentile of |cz| on R trips (nominal FPR 0.10 in R);
                  also one-sided cz > t1 (fault direction used by the injection), t1 = 90th pct of cz on R.
  Test FPR      : fraction of T trips flagged.
  PRE-REGISTERED: raw two-sided test FPR >= 0.25 (the threat is real) ; deseason two-sided test FPR <= 0.15
                  (fix works) ; injected-fault frozen accuracy with the deseasonalised axis drops < 0.03 vs
                  the raw axis (fix is not costly). KILL: deseason costs > 0.05 accuracy or deseason FPR > 0.20.
ITEM 2 operational definitions (fixed before running):
  Algorithm 1 rule on UNPERTURBED trips: ld, lf = source-calibrated log likelihood ratios; "none" if both
  < 0; else the larger cause, issued only if |ld - lf| >= conformal threshold (p19, Hoeffding, delta 0.1),
  otherwise "abstain". Issued cause on an unperturbed trip = false attribution UNDER THE ASSUMPTION that
  unperturbed trips carry no cause (real trips do contain natural aggressive driving, so this is an
  upper bound on the false-alarm rate).
  PRE-REGISTERED: fleet false-attribution rate <= alpha for alpha in {0.05, 0.10}; per-vehicle rate <= 2 alpha
  for >= 90% of vehicles. KILL: fleet rate > 2 alpha.
  Bootstrap: 1000 vehicle-level resamples of the frozen accuracy at d=0.65 (headline 0.785) and d=1.3
  (0.856). PRE-REGISTERED: 95% CI half-width < 0.02.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw"); OUT = os.path.join(ROOT, "data", "processed")
rng = np.random.default_rng(0)

# ---------- frozen UAH driver signature (identical to p17/p19) ----------
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
sc = Uz @ Wk; S_driver = abs(sc[uy == 1].mean() - sc[uy == 0].mean()) / np.sqrt((sc[uy == 1].var() + sc[uy == 0].var()) / 2)

# ---------- VED trips (identical filter) ----------
v = pd.read_parquet(os.path.join(OUT, "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
VF = ["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]
v = v[v["ltft1_cov"] >= 0.2].dropna(subset=["ltft1_mean"] + VF).copy()
v = v[v.groupby("VehId")["ltft1_mean"].transform("size") >= 15].reset_index(drop=True)
v["m"] = pd.to_datetime(v["week"].str.extract(r"VED_(\d{6})_")[0], format="%y%m%d").dt.month
grp = v["VehId"].to_numpy(); n = len(v)
Zv = StandardScaler().fit_transform(v[VF]); kraw = Zv @ Wk
L = v["ltft1_mean"].to_numpy()

def within_std(x, g):
    s = pd.Series(x); gg = s.groupby(g)
    return ((s - gg.transform("mean")) / (gg.transform("std") + 1e-9)).to_numpy(), float(gg.std().median())

# leave-vehicle-out fleet month effect on annually-centred LTFT
cen = L - pd.Series(L).groupby(grp).transform("mean").to_numpy()
mm = v["m"].to_numpy()
tab = pd.DataFrame({"veh": grp, "m": mm, "c": cen})
sums = tab.groupby("m")["c"].agg(["sum", "count"])
own = tab.groupby(["veh", "m"])["c"].agg(["sum", "count"])
eff = np.zeros(n)
for i, (vv, m_) in enumerate(zip(grp, mm)):
    s_all, c_all = sums.loc[m_, "sum"], sums.loc[m_, "count"]; s_o, c_o = own.loc[(vv, m_), "sum"], own.loc[(vv, m_), "count"]
    eff[i] = (s_all - s_o) / max(c_all - c_o, 1)
L_ds = L - eff
R = {"S_driver": round(float(S_driver), 3)}

# ================= ITEM 1A: seasonal false alarms =================
win = np.isin(mm, [12, 1, 2]); sum_ = np.isin(mm, [6, 7, 8])
cw = pd.Series(win).groupby(grp).transform("sum").to_numpy(); cs = pd.Series(sum_).groupby(grp).transform("sum").to_numpy()
elig = (cw >= 10) & (cs >= 10)
R["item1_vehicles"] = int(len(np.unique(grp[elig])))

def fpr(series):
    out = {}
    for name, (Rm, Tm) in {"ref_winter_test_summer": (win, sum_), "ref_summer_test_winter": (sum_, win)}.items():
        zR, zT = [], []
        for vv in np.unique(grp[elig]):
            mv = grp == vv; xr = series[mv & Rm]; xt = series[mv & Tm]
            mu, sd = xr.mean(), xr.std() + 1e-9
            zR.append((xr - mu) / sd); zT.append((xt - mu) / sd)
        zR = np.concatenate(zR); zT = np.concatenate(zT)
        t2 = np.quantile(np.abs(zR), 0.90); t1 = np.quantile(zR, 0.90)
        out[name] = {"two_sided_test_fpr": round(float(np.mean(np.abs(zT) > t2)), 3),
                     "one_sided_test_fpr": round(float(np.mean(zT > t1)), 3), "n_test": int(len(zT))}
    out["two_sided_pooled"] = round(float(np.mean([out[k]["two_sided_test_fpr"] for k in out if k.startswith("ref")])), 3)
    out["one_sided_pooled"] = round(float(np.mean([out[k]["one_sided_test_fpr"] for k in out if k.startswith("ref")])), 3)
    return out
R["item1_raw"] = fpr(L); R["item1_deseason"] = fpr(L_ds)

# ================= ITEM 1B: injected-fault accuracy, raw vs deseasonalised axis =================
def z(x, ref): return (x - ref.mean()) / ref.std()
def frozen_acc(cz0, S_fault, S_d):
    PERT = (S_d * kraw.std() / (Wk @ dirn)) * dirn
    kd = z((Zv + PERT) @ Wk, kraw); kn = z(kraw, kraw); cz = z(cz0, cz0); cf = z(cz0 + S_fault, cz0)
    X1 = np.c_[kd, cz]; X2 = np.c_[kn, cf]
    pred = np.r_[(X1[:, 1] > X1[:, 0]).astype(int), (X2[:, 1] > X2[:, 0]).astype(int)]
    y = np.r_[np.zeros(n), np.ones(n)]
    return pred, y, kd, kn, cz, cf
cz_raw, sd_raw = within_std(L, grp); cz_ds, sd_ds = within_std(L_ds, grp)
Sf_raw = 3.49 / sd_raw; Sf_ds = 3.49 / sd_ds  # LiU-ICE smallest measured fueling deviation, pp (p37)
p_r, y_r, *_ = frozen_acc(cz_raw, Sf_raw, S_driver); p_d, y_d, *_ = frozen_acc(cz_ds, Sf_ds, S_driver)
R["item1B"] = {"within_sd_raw_pp": round(sd_raw, 3), "within_sd_deseason_pp": round(sd_ds, 3),
               "S_fault_raw_SD": round(Sf_raw, 3), "S_fault_deseason_SD": round(Sf_ds, 3),
               "acc_raw": round(float((p_r == y_r).mean()), 3), "acc_deseason": round(float((p_d == y_d).mean()), 3)}
R["item1B"]["acc_change"] = round(R["item1B"]["acc_deseason"] - R["item1B"]["acc_raw"], 3)
a1 = R["item1_raw"]["two_sided_pooled"]; a2 = R["item1_deseason"]["two_sided_pooled"]; dacc = R["item1B"]["acc_change"]
R["item1_verdict"] = {"threat_real_raw_ge_0.25": bool(a1 >= 0.25), "fix_works_deseason_le_0.15": bool(a2 <= 0.15),
                      "fix_cheap_acc_drop_lt_0.03": bool(dacc > -0.03),
                      "KILL": bool(dacc < -0.05 or a2 > 0.20)}

# ================= ITEM 2: zero-effect control + bootstrap =================
def lrs(KZ, CZ, Sf):
    ld = N.logpdf(KZ, S_driver, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, Sf, 1) - N.logpdf(CZ, 0, 1)
    return ld, lf
# conformal threshold calibrated on held-out vehicles' injected variants (as p19)
_, _, kd, kn, cz, cf = frozen_acc(cz_raw, Sf_raw, S_driver)
KZ = np.r_[kd, kn]; CZ = np.r_[cz, cf]; y = np.r_[np.zeros(n), np.ones(n)]; g2 = np.r_[grp, grp]
ld, lf = lrs(KZ, CZ, Sf_raw); pred = (lf > ld).astype(int); conf = np.abs(ld - lf)
vehs = np.unique(grp); rng.shuffle(vehs); cal = np.isin(g2, vehs[:len(vehs) // 2]); delta = 0.1
def tau_for(alpha):
    c = conf[cal]; e = (pred[cal] != y[cal]).astype(int); o = np.argsort(-c)
    ns = np.arange(1, len(c) + 1); K = 100; pos = np.unique(np.clip((np.arange(1, K + 1) / K * len(c)).astype(int), 1, len(c))) - 1
    ucb = __import__('scipy.stats', fromlist=['beta']).beta.ppf(1 - delta / K, np.cumsum(e[o])[pos] + 1, np.maximum(ns[pos] - np.cumsum(e[o])[pos], 1e-9))   # exact binomial (Clopper-Pearson) UCB, Bonferroni over K candidate thresholds
    ok = np.where(ucb <= alpha)[0]; return c[o][pos[ok[-1]]] if len(ok) else np.inf
ld0, lf0 = lrs(z(kraw, kraw), cz_raw, Sf_raw)      # UNPERTURBED trips
none0 = (ld0 < 0) & (lf0 < 0)
R["item2"] = {"none_rate_rule_only": round(float(none0.mean()), 3)}
for alpha in [0.05, 0.10]:
    t = tau_for(alpha); issued = (~none0) & (np.abs(ld0 - lf0) >= t)
    pv = pd.Series(issued).groupby(grp).mean()
    R["item2"][f"alpha_{alpha}"] = {"tau": round(float(t), 3), "fleet_false_attribution": round(float(issued.mean()), 3),
                                    "abstain_rate": round(float(((~none0) & ~issued).mean()), 3),
                                    "frac_vehicles_le_2alpha": round(float((pv <= 2 * alpha).mean()), 3),
                                    "issued_driver_share": round(float((issued & (ld0 > lf0)).mean()), 3),
                                    "issued_fault_share": round(float((issued & (lf0 > ld0)).mean()), 3)}
R["item2_verdict"] = {f"alpha_{a}": {"pass_fleet_le_alpha": bool(R["item2"][f"alpha_{a}"]["fleet_false_attribution"] <= a),
                                     "pass_vehicles": bool(R["item2"][f"alpha_{a}"]["frac_vehicles_le_2alpha"] >= 0.90),
                                     "KILL_fleet_gt_2alpha": bool(R["item2"][f"alpha_{a}"]["fleet_false_attribution"] > 2 * a)} for a in [0.05, 0.10]}

# vehicle-level bootstrap of frozen accuracy at d=0.65 and d=1.3
for S_d, tag in [(S_driver, "d065"), (1.3, "d13")]:
    pr, yy, *_ = frozen_acc(cz_raw, Sf_raw, S_d); correct = (pr == yy).astype(float)
    corr2 = correct.reshape(2, n).T                               # per base trip: [driver variant, fault variant]
    byveh = pd.DataFrame({"g": grp, "c": corr2.sum(1), "k": 2}).groupby("g").sum()
    cv, kv = byveh["c"].to_numpy(), byveh["k"].to_numpy(); B = []
    for _ in range(1000):
        idx = rng.integers(0, len(cv), len(cv)); B.append(cv[idx].sum() / kv[idx].sum())
    lo_, hi_ = np.percentile(B, [2.5, 97.5])
    R[f"item2_bootstrap_{tag}"] = {"acc": round(float(correct.mean()), 3), "ci95": [round(float(lo_), 3), round(float(hi_), 3)],
                                   "half_width": round(float((hi_ - lo_) / 2), 4), "pass_half_width_lt_0.02": bool((hi_ - lo_) / 2 < 0.02)}
json.dump(R, open(os.path.join(OUT, "p26_seasonal_zero.json"), "w"), indent=2)
print(json.dumps(R, indent=2))
