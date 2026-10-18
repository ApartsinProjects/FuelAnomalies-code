"""P50: deployment-faithful checks of the injected-attribution accuracy (fresh-eye review, 2026-10-18).
(1) PER-VEHICLE SCALE: the fault is 3.49 pp of raw trim on every car, so in each car's own standardized units it is
    3.49 / sigma_veh (headline: a constant 3.49 / median sigma = 1.94).
(2) CAUSAL ROLLING REFERENCE: trim referenced to the same car's previous 20 trips only (as deployed), fault = +3.49 pp
    on the scored trip; z = (LTFT_i + fault - mean_prev20) / sd_prev20; the likelihood uses S_f = 3.49 / median sd_prev20.
Both use the frozen UAH kinematic score and the same driver injection as p33 (window d 0.65, trip d 1.3), argmax and
LR-fusion decisions. Invariant: with the headline construction (whole-history z, constant shift) the script must
reproduce p33 (window fusion 0.847, argmax 0.801).
Output: data/processed/p50_deployment_faithful.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, Zv, Wk, dirn, kraw, cz0, grp, n = g["v"], g["Zv"], g["Wk"], g["dirn"], g["kraw"], g["cz0"], g["grp"], g["n"]
FAULT = 3.49
z = lambda x, ref: (x - ref.mean()) / ref.std()
def acc(kd, kn, czd, czf, Sd, Sf):
    KZ = np.r_[kd, kn]; CZ = np.r_[czd, czf]; y = np.r_[np.zeros(len(kd)), np.ones(len(kn))]
    ld = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, Sf, 1) - N.logpdf(CZ, 0, 1)
    return round(float(((lf > ld) == y).mean()), 3), round(float(((CZ > KZ) == y).mean()), 3)
def kin(Sd, mask=None):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn
    kd = z((Zv + PERT) @ Wk, kraw); kn = z(kraw, kraw)
    return (kd, kn) if mask is None else (kd[mask], kn[mask])
sd_veh = v.groupby("VehId")["ltft1_mean"].transform("std").to_numpy(); S_head = FAULT / np.median(v.groupby("VehId")["ltft1_mean"].std())
R = {"S_fault_headline": round(float(S_head), 3)}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    kd, kn = kin(Sd)
    R[f"{lvl}_headline_fusion_argmax"] = acc(kd, kn, cz0, cz0 + S_head, Sd, S_head)
    R[f"{lvl}_per_vehicle_scale_fusion_argmax"] = acc(kd, kn, cz0, cz0 + FAULT / sd_veh, Sd, S_head)
# causal rolling reference (trips ordered by trip number within vehicle)
o = v.reset_index(drop=True).copy(); o["_i"] = np.arange(len(o)); o = o.sort_values(["VehId", "Trip"])
gL = o.groupby("VehId")["ltft1_mean"]
mu = gL.transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean()); sd = gL.transform(lambda s: s.shift(1).rolling(20, min_periods=20).std())
ok = mu.notna() & (sd > 0); idx = o["_i"].to_numpy()[ok.to_numpy()]
zr = ((o["ltft1_mean"] - mu) / sd).to_numpy()[ok.to_numpy()]; zf = ((o["ltft1_mean"] + FAULT - mu) / sd).to_numpy()[ok.to_numpy()]
S_roll = FAULT / float(np.median(sd[ok])); m = np.zeros(n, bool); m[idx] = True
order = np.argsort(idx); zr, zf = zr[order], zf[order]     # align to v order of the masked rows
R["rolling_trips"] = int(m.sum()); R["rolling_vehicles"] = int(len(np.unique(grp[m]))); R["S_fault_rolling"] = round(S_roll, 3)
R["rolling_z_sd_healthy"] = round(float(np.std(zr)), 3)
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    kd, kn = kin(Sd, m)
    R[f"{lvl}_rolling_fusion_argmax"] = acc(kd, kn, zr, zf, Sd, S_roll)
    R[f"{lvl}_headline_same_trips_fusion_argmax"] = acc(kd, kn, cz0[m], cz0[m] + S_head, Sd, S_head)
R["checks"] = {"reproduces_p33_window": R["window_headline_fusion_argmax"] == (0.847, 0.801)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p50_deployment_faithful.json"), "w"), indent=1); print(json.dumps(R, indent=1))
