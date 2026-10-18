"""P35: component ablation of Algorithm 1 (one pass, same p24/p33 injected variants) + runtime per trip.

Ablations (each removes ONE component; accuracy at window S_d 0.65 and trip S_d 1.3, plus conformal coverage):
 FULL           frozen UAH weights, within-vehicle LTFT z, source-calibrated LLR fusion, conformal abstention.
 -fusion        argmax of the two unit-variance scores instead of calibrated LLRs (= paper's frozen argmax).
 -within-veh    combustion axis standardized across the FLEET (raw LTFT / fleet SD) instead of within vehicle;
                fault shift stays 2.98 pp, so its calibrated size becomes 2.98 / fleet LTFT SD.
 -frozen-w      equal weights over the 4 standardized kinematic features instead of the frozen UAH logistic.
 -multi-feat    a single kinematic feature (95th-percentile acceleration) instead of the 4-feature signature.
 -abstention    forced decisions (coverage 1.0) vs conformal alpha = 0.05 / 0.10 (selective error reported).
PRE-REGISTERED invariants: FULL reproduces p33 OURS-LR (0.811 / 0.858); -fusion reproduces p33 OURS-margin
(0.785 / 0.856); conformal selective error <= alpha on the held-out half at both levels.
Runtime: wall-clock per trip for kinematic features + both scores + fusion on real 1 Hz trips (Sonata, ~1,900 s each),
single CPU thread, median of 5 repeats.
Output: data/processed/p35_ablation_runtime.json
"""
import os, io, glob, json, time, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "p33_attribution_baselines.py"))
Zv, cz0, Wk, dirn, kraw, ksd, n, grp, v, S_fault = (g[k] for k in ["Zv", "cz0", "Wk", "dirn", "kraw", "ksd", "n", "grp", "v", "S_fault"])
z = lambda x, ref: (x - ref.mean()) / ref.std()
rng = np.random.default_rng(0)
ltft = v["ltft1_mean"].to_numpy(); fleet_sd = ltft.std()

def run(S_d, W=Wk, D=dirn, Zk=Zv, comb="within", fusion=True):
    PERT = (S_d * ksd / (Wk @ dirn)) * dirn        # the injected driver effect is FIXED (UAH direction, frozen-score size);
    k0 = Zk @ W; KZ = np.r_[z((Zk + PERT) @ W, k0), z(k0, k0)]   # only the attributor's weights W vary
    if comb == "within":
        CZ = np.r_[z(cz0, cz0), z(cz0 + S_fault, cz0)]; Sf = S_fault
    else:                                          # fleet-standardized raw LTFT, same 2.98 pp physical shift
        CZ = np.r_[z(ltft, ltft), z(ltft + 3.49, ltft)]; Sf = 3.49 / fleet_sd  # LiU-ICE smallest measured fueling deviation, pp (p37)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    if fusion:
        ld = N.logpdf(KZ, S_d, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, Sf, 1) - N.logpdf(CZ, 0, 1)
        pred = (lf > ld).astype(int); conf = np.abs(ld - lf)
    else:
        pred = (CZ - KZ > 0).astype(int); conf = np.abs(CZ - KZ)
    return pred, conf, y

def conformal(pred, conf, y, alpha, delta=0.1):
    g2 = np.r_[grp, grp]; vehs = np.unique(g2); r = np.random.default_rng(0); r.shuffle(vehs)
    cal = np.isin(g2, vehs[: len(vehs) // 2]); te = ~cal
    c = conf[cal]; e = (pred[cal] != y[cal]).astype(int); o = np.argsort(-c)
    ns = np.arange(1, len(c) + 1); K = 100; pos = np.unique(np.clip((np.arange(1, K + 1) / K * len(c)).astype(int), 1, len(c))) - 1
    ucb = __import__('scipy.stats', fromlist=['beta']).beta.ppf(1 - delta / K, np.cumsum(e[o])[pos] + 1, np.maximum(ns[pos] - np.cumsum(e[o])[pos], 1e-9))   # exact binomial (Clopper-Pearson) UCB, Bonferroni over K candidate thresholds
    ok = np.where(ucb <= alpha)[0]; tau = c[o][pos[ok[-1]]] if len(ok) else np.inf
    m = conf[te] >= tau
    return {"coverage": round(float(m.mean()), 3), "selective_error": round(float((pred[te][m] != y[te][m]).mean()), 3) if m.any() else None}

eq = np.ones(4) / 2.0                              # equal weights, unit norm
one = np.zeros(4); one[1] = 1.0                    # accel_p95 only
CFG = {"FULL": {}, "-fusion": {"fusion": False}, "-within-veh": {"comb": "fleet"}, "-frozen-w": {"W": eq}, "-multi-feat": {"W": one}}
R = {"n_trips": int(n)}
for name, kw in CFG.items():
    row = {}
    for lvl, S_d in [("window", 0.65), ("trip", 1.3)]:
        p, c, y = run(S_d, **kw); row[f"{lvl}_acc"] = round(float((p == y).mean()), 3)
        if name == "FULL":
            row[f"{lvl}_conformal_a05"] = conformal(p, c, y, 0.05); row[f"{lvl}_conformal_a10"] = conformal(p, c, y, 0.10)
    R[name] = row; print(name, row, flush=True)
_b = json.load(open(os.path.join(ROOT, "data", "processed", "p33_baselines.json")))
R["checks"] = {"FULL_reproduces_p33": R["FULL"]["window_acc"] == _b["window_d0.65"]["OURS-LR"]["accuracy"] and R["FULL"]["trip_acc"] == _b["trip_d1.3"]["OURS-LR"]["accuracy"],
               "minus_fusion_reproduces_margin": R["-fusion"]["window_acc"] == _b["window_d0.65"]["OURS-margin"]["accuracy"] and R["-fusion"]["trip_acc"] == _b["trip_d1.3"]["OURS-margin"]["accuracy"],
               "conformal_within_alpha": all((R["FULL"][f"{l}_conformal_{a}"]["selective_error"] or 0) <= float("0." + a[1:])
                                             for l in ("window", "trip") for a in ("a05", "a10"))}

# ---- runtime per trip on real 1 Hz trips ----
kin, su = g["kin"], g["su"]
trips = [pd.read_csv(f, usecols=["car_speed", "long_fuel_bank"]) for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "sonata", "*", "All_*.csv")))]
trips = [t for t in trips if not (t.long_fuel_bank <= -99).all()]
mu_k, sd_k, mu_c, sd_c = 0.0, 1.0, 0.0, 1.79
def attribute(t):
    kz = (float((su.transform(np.array([kin(t.car_speed.to_numpy())])) @ Wk)[0]) - mu_k) / sd_k
    cz = (t.long_fuel_bank.mean() - mu_c) / sd_c
    ld = N.logpdf(kz, 1.3, 1) - N.logpdf(kz, 0, 1); lf = N.logpdf(cz, 1.66, 1) - N.logpdf(cz, 0, 1)
    return lf > ld, abs(ld - lf)
reps = []
for _ in range(5):
    t0 = time.perf_counter()
    for t in trips: attribute(t)
    reps.append((time.perf_counter() - t0) / len(trips))
R["runtime"] = {"n_trips": len(trips), "mean_trip_seconds_of_data": round(float(np.mean([len(t) for t in trips])), 0),
                "ms_per_trip_median": round(1000 * float(np.median(reps)), 3), "parameters": "4 kinematic weights + 4 scaler means + 4 scaler SDs + 2 effect sizes + 1 threshold"}
print(json.dumps(R["checks"]), R["runtime"])
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p35_ablation_runtime.json"), "w"), indent=2)
