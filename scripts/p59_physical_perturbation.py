"""P59: physically consistent driver perturbations on real VED speed traces (external review brief item 3).
Instead of shifting the four signature features directly, each trip's 1 Hz speed trace is made more aggressive and the
VED feature pipeline (p1_features.trip_features) is rerun, so speed, acceleration, jerk and harsh-event counts stay
mutually consistent. Transform: v' = trend + (1 + gamma) (v - trend), trend = 15 s centred rolling mean, clipped at 0.
A perturbed trip is INVALID if any second has |acceleration| > 4 m/s^2 that the original trip did not have (beyond
plausible passenger-car limits). A second variant re-integrates the amplified trace so every 1 s step obeys |a| <= 4 m/s^2
(valid by construction) and is calibrated separately. gamma is chosen on the first week so the mean frozen-score shift equals the window
effect 0.65; the attributor (S_d 0.65, S_f 1.94) is unchanged. Weeks: the first three VED weeks.
Reports: invalid share, score-shift distribution, accuracy with physically perturbed driver variants against the
feature-level injection on the same trips. Output: data/processed/p59_physical_perturbation.json"""
import os, io, json, glob, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
    p1 = runpy.run_path(os.path.join(ROOT, "scripts", "p1_features.py"), run_name="p1")
v, Wk, dirn, kraw, cz0, VF, sv = (g[k] for k in ["v", "Wk", "dirn", "kraw", "cz0", "VF", "sv"])
SF = g["S_fault"]; resample_trip, trip_features = p1["resample_trip"], p1["trip_features"]
key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(v.VehId, v.Trip))}
cols = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]",
        "Air Conditioning Power[Watts]", "Heater Power[Watts]", "Short Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 1[%]"]
trips = []
for w in sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv")))[:3]:
    df = pd.read_csv(w, usecols=lambda c: c in cols)
    for (vid, tr), gdf in df.groupby(["VehId", "Trip"], sort=False):
        i = key.get((int(vid), int(tr)))
        if i is None: continue
        r = resample_trip(gdf)
        if r is not None: trips.append((i, r))
score = lambda F: float((sv.transform(np.array([[F[c] for c in VF]])) @ Wk)[0])
AMAX = 4.0
def limit_acc(s):                                     # re-integrate so every 1 s speed step respects |a| <= AMAX
    v = s / 3.6; o = v.copy()
    for t in range(1, len(v)): o[t] = min(max(v[t], o[t - 1] - AMAX), o[t - 1] + AMAX)
    return np.clip(o, 0, None) * 3.6
LIMITED = False
def perturb(r, gamma):
    s = r["speed"].to_numpy(); tr_ = pd.Series(s).rolling(15, center=True, min_periods=1).mean().to_numpy()
    s2 = np.clip(tr_ + (1 + gamma) * (s - tr_), 0, None)
    if LIMITED: s2 = limit_acc(s2)
    r2 = r.copy(); r2["speed"] = s2
    a0 = np.abs(np.gradient(s / 3.6)); a1 = np.abs(np.gradient(s2 / 3.6))
    invalid = bool(np.nanmax(a1) > 4.0 and not np.nanmax(a0) > 4.0)
    return r2, invalid
base = {}
for i, r in trips:
    F = trip_features(r)
    if F is not None: base[i] = score(F)
ids = sorted(base); ksd = kraw.std()
def shifts(gamma, sub):
    out, inv = [], []
    for i, r in sub:
        if i not in base: continue
        r2, bad = perturb(r, gamma); F = trip_features(r2)
        if F is None: continue
        out.append((i, (score(F) - base[i]) / ksd)); inv.append(bad)
    return out, inv
first = trips[: max(50, len(trips) // 3)]
PERT = (0.65 * ksd / (Wk @ dirn)) * dirn
def acc(kd, k0, idx):
    KZ = np.r_[kd, k0]; CZ = np.r_[cz0[idx], cz0[idx] + SF]; y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
    la = N.logpdf(KZ, 0.65, 1) - N.logpdf(KZ, 0, 1); lb = N.logpdf(CZ, SF, 1) - N.logpdf(CZ, 0, 1)
    return {"trips": int(len(idx)), "fusion": round(float(((lb > la) == y).mean()), 3), "argmax": round(float(((CZ > KZ) == y).mean()), 3)}
R = {"weeks": 3, "trips_in_panel": len(base), "max_abs_accel_m_s2": AMAX}
for variant, lim in (("amplified", False), ("amplified_accel_limited", True)):
    LIMITED = lim; lo, hi = 0.0, 3.0
    for _ in range(14):                               # bisection on gamma for a mean score shift of 0.65
        mid = (lo + hi) / 2; s, _ = shifts(mid, first); m = np.mean([x for _, x in s])
        lo, hi = (mid, hi) if m < 0.65 else (lo, mid)
    gamma = (lo + hi) / 2; sh, inv = shifts(gamma, trips); inv = np.array(inv)
    idx = np.array([i for i, _ in sh]); dk = np.array([x for _, x in sh])
    k0 = (kraw[idx] - kraw.mean()) / ksd; kphys = k0 + dk; kfeat = ((g["Zv"][idx] + PERT) @ Wk - kraw.mean()) / ksd
    ok = ~inv
    R[variant] = {"trips_perturbed": int(len(idx)), "gamma": round(float(gamma), 3), "invalid_share": round(float(inv.mean()), 4),
                  "score_shift_mean": round(float(dk.mean()), 3), "score_shift_sd": round(float(dk.std()), 3),
                  "score_shift_q10_q90": [round(float(np.quantile(dk, .1)), 3), round(float(np.quantile(dk, .9)), 3)],
                  "physical_perturbation": acc(kphys, k0, idx), "feature_level_same_trips": acc(kfeat, k0, idx),
                  "physical_perturbation_valid_only": acc(kphys[ok], k0[ok], idx[ok]), "feature_level_valid_only": acc(kfeat[ok], k0[ok], idx[ok])}
    print(variant, R[variant], flush=True)
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p59_physical_perturbation.json"), "w"), indent=1); print(json.dumps(R, indent=1))
