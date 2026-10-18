"""P52: attribution accuracy when the injected effects do NOT follow the attributor's assumptions (IJPHM review R1-R3).
The attributor is unchanged: frozen UAH score, within-vehicle trim z, likelihoods with S_d = 0.65 (window) or 1.3 (trip)
and S_f = 1.94, forced two-way decision (fusion and argmax). Only the injected truth changes:
  M0 headline (driver shift along the UAH class-mean direction scaled to S_d score units; fault +S_f on trim z)
  M1 raw transport: driver shift = the UAH aggressive-minus-normal mean feature difference in standardized units, NOT
     rescaled to the score (the score shift is whatever it turns out to be on VED)
  M2 rotated direction: driver shift direction rotated 45 degrees from the class-mean direction (same feature-space norm as M0)
  M3 heterogeneous sizes: per-variant driver size S_d * LogNormal(0, 0.5); fault 3.49 pp / sigma_veh * LogNormal(0, 0.5)
  M4 heavy tails: driver and fault sizes scaled by |t_3| / E|t_3| (same mean size)
  M5 cross-axis leakage: driver variant also moves trim z by -0.25 * S_d, fault variant also moves the score by +0.25 * S_f
  M6 ramp fault, causal rolling reference: fault grows linearly over 5 trips; scored on the 5th trip against the previous
     20 trips, which already contain the partial fault (3 of 4 earlier ramp trips inside the window)
  M7 all of M1, M3, M4, M5 together
Invariant: M0 reproduces p33 (window fusion 0.847, argmax 0.801).
Output: data/processed/p52_misspecified_injection.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N, chi2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, Zv, Wk, dirn, kraw, cz0, grp, n, Uz, uy = (g[k] for k in ["v", "Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n", "Uz", "uy"])
FAULT, SF = 3.49, 3.49 / float(np.median(v.groupby("VehId")["ltft1_mean"].std()))
sdv = v.groupby("VehId")["ltft1_mean"].transform("std").to_numpy()
z = lambda x: (x - kraw.mean()) / kraw.std()
def decide(KZ, CZ, y, Sd):
    ld = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, SF, 1) - N.logpdf(CZ, 0, 1)
    return round(float(((lf > ld) == y).mean()), 3), round(float(((CZ > KZ) == y).mean()), 3)
def run(Sd, kind):
    k0 = z(kraw); y = np.r_[np.zeros(n), np.ones(n)]
    unit = dirn / np.linalg.norm(dirn); a = Sd * kraw.std() / (Wk @ unit)          # M0 feature-space size along dirn
    sd_i = np.full(n, Sd); sf_i = np.full(n, SF); dvec = np.tile(unit * a, (n, 1)); leak_d = np.zeros(n); leak_f = np.zeros(n)
    if kind in ("M1", "M7"):
        dvec = np.tile(Uz[uy == 1].mean(0) - Uz[uy == 0].mean(0), (n, 1))
    if kind == "M2":
        perp = np.linalg.svd(np.c_[unit, np.eye(len(unit))])[0][:, 1]; perp -= (perp @ unit) * unit; perp /= np.linalg.norm(perp)
        dvec = np.tile((np.cos(np.pi / 4) * unit + np.sin(np.pi / 4) * perp) * a, (n, 1))
    if kind in ("M3", "M7"):
        m = rng.lognormal(0, 0.5, n); dvec = dvec * m[:, None]; sf_i = FAULT / sdv * rng.lognormal(0, 0.5, n)
    if kind in ("M4", "M7"):
        t = np.abs(rng.standard_t(3, n)); t /= np.mean(t); dvec = dvec * t[:, None]
        t2 = np.abs(rng.standard_t(3, n)); t2 /= np.mean(t2); sf_i = sf_i * t2
    kd = z((Zv + dvec) @ Wk)
    if kind in ("M5", "M7"):
        leak_d = -0.25 * (kd - k0); leak_f = 0.25 * sf_i
    KZ = np.r_[kd, k0 + leak_f]; CZ = np.r_[cz0 + leak_d, cz0 + sf_i]
    Xk = np.r_[Zv + dvec, Zv]; mu = Zv.mean(0); Si = np.linalg.inv(np.cov(Zv, rowvar=False))
    p_k = chi2.sf(np.einsum("ij,jk,ik->i", Xk - mu, Si, Xk - mu), 4); p_c = 2 * N.sf(np.abs((CZ - cz0.mean()) / cz0.std()))
    out = {"fusion_argmax": decide(KZ, CZ, y, Sd), "target_only": round(float(((p_c < p_k) == y).mean()), 3),
           "mean_injected_score_shift": round(float(np.mean(kd - k0)), 3)}
    return out
R = {}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    for kind in ("M0", "M1", "M2", "M3", "M4", "M5", "M7"):
        R[f"{lvl}_{kind}"] = run(Sd, kind)
# M6 ramp fault with the causal rolling reference
o = v.reset_index(drop=True).copy(); o["_i"] = np.arange(len(o)); o = o.sort_values(["VehId", "Trip"]).reset_index(drop=True)
res = {"window": [], "trip": []}
for veh, gdf in o.groupby("VehId"):
    L = gdf["ltft1_mean"].to_numpy(); idx = gdf["_i"].to_numpy()
    for t in range(24, len(L)):                       # need 20 reference trips before the ramp start (t-4)
        ramp = np.zeros(len(L)); ramp[t - 4:t + 1] = FAULT * np.arange(1, 6) / 5
        ref = (L + ramp)[t - 20:t]; zf = (L[t] + FAULT - ref.mean()) / ref.std()
        refh = L[t - 20:t]; zh = (L[t] - refh.mean()) / refh.std()
        res["window"].append((idx[t], zh, zf))
R["ramp_trips"] = len(res["window"])
arr = np.array(res["window"]); ii = arr[:, 0].astype(int); zh, zf = arr[:, 1], arr[:, 2]
S_roll = 2.32
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn; kd = z((Zv + PERT) @ Wk)[ii]; kn = z(kraw)[ii]
    KZ = np.r_[kd, kn]; CZ = np.r_[zh, zf]; y = np.r_[np.zeros(len(ii)), np.ones(len(ii))]
    ld = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, S_roll, 1) - N.logpdf(CZ, 0, 1)
    R[f"{lvl}_M6_ramp_rolling"] = {"fusion_argmax": (round(float(((lf > ld) == y).mean()), 3), round(float(((CZ > KZ) == y).mean()), 3)),
                                   "median_fault_z_on_5th_trip": round(float(np.median(zf - zh)), 3)}
R["checks"] = {"M0_reproduces_p33": R["window_M0"]["fusion_argmax"] == (0.847, 0.801)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p52_misspecified_injection.json"), "w"), indent=1); print(json.dumps(R, indent=1))
