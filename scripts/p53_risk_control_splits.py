"""P53: empirical validity of the abstention guarantee under clustering by vehicle (IJPHM review R1, R3).
Same injected variants and fusion as p19 (window d 0.65 and trip d 1.3). For 200 random half/half vehicle splits,
the threshold is calibrated on one half (Clopper-Pearson UCB, delta = 0.1, Bonferroni over 100 candidate margins) and
the realized selective error measured on the other half. The guarantee requires P(error > alpha) <= delta = 0.10.
A second, cluster-robust calibration uses one randomly drawn variant per vehicle and cause (independent units).
Also: Clopper-Pearson 95% intervals for the healthy-car false-fault counts (Kia 5/704, Sonata 0/665).
Output: data/processed/p53_risk_control_splits.json"""
import os, io, json, contextlib, runpy, numpy as np
from scipy.stats import norm as N, beta
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(1)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
Zv, Wk, dirn, kraw, cz0, grp, n, S_fault = (g[k] for k in ["Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n", "S_fault"])
z = lambda x: (x - kraw.mean()) / kraw.std()
def tau_for(conf, err, alpha, delta=0.1, K=100):
    o = np.argsort(-conf); c = conf[o]; e = err[o]; ns = np.arange(1, len(c) + 1)
    pos = np.unique(np.clip((np.arange(1, K + 1) / K * len(c)).astype(int), 1, len(c))) - 1
    k = np.cumsum(e)[pos]; ucb = beta.ppf(1 - delta / K, k + 1, np.maximum(ns[pos] - k, 1e-9))
    ok = np.where(ucb <= alpha)[0]; return c[pos[ok[-1]]] if len(ok) else np.inf
R = {}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn
    KZ = np.r_[z((Zv + PERT) @ Wk), z(kraw)]; CZ = np.r_[cz0, cz0 + S_fault]; y = np.r_[np.zeros(n), np.ones(n)]; G = np.r_[grp, grp]
    ld = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, S_fault, 1) - N.logpdf(CZ, 0, 1)
    err = ((lf > ld) != y).astype(int); conf = np.abs(ld - lf); vehs = np.unique(grp)
    for alpha in (0.05, 0.10):
        ex, cov, ex1, cov1 = [], [], [], []
        for _ in range(200):
            p = rng.permutation(vehs); cal = np.isin(G, p[:len(p) // 2]); te = ~cal
            t = tau_for(conf[cal], err[cal], alpha); m = te & (conf >= t)
            if m.sum(): ex.append(err[m].mean() > alpha); cov.append(m.mean() / te.mean())
            # cluster-robust: one variant per (vehicle, cause) in the calibration half
            ci = np.flatnonzero(cal); pick = []
            for cls in (0, 1):
                cc = ci[y[ci] == cls]; gg = G[cc]
                for vv in np.unique(gg): pick.append(rng.choice(cc[gg == vv]))
            pick = np.array(pick); t1 = tau_for(conf[pick], err[pick], alpha); m1 = te & (conf >= t1)
            if m1.sum(): ex1.append(err[m1].mean() > alpha); cov1.append(m1.mean() / te.mean())
        R[f"{lvl}_alpha{alpha}"] = {"trip_level_calibration": {"exceed_rate": round(float(np.mean(ex)), 3), "coverage_median": round(float(np.median(cov)), 3)},
                                    "one_per_vehicle_calibration": {"exceed_rate": round(float(np.mean(ex1)), 3) if ex1 else None,
                                                                    "coverage_median": round(float(np.median(cov1)), 3) if cov1 else 0.0, "splits_with_decisions": len(ex1)}}
cp = lambda k, n_: [round(float(beta.ppf(0.025, k, n_ - k + 1)) if k else 0.0, 4), round(float(beta.ppf(0.975, k + 1, n_ - k)), 4)]
R["healthy_car_CP95"] = {"Kia_5_of_704": cp(5, 704), "Sonata_0_of_665": cp(0, 665)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p53_risk_control_splits.json"), "w"), indent=1); print(json.dumps(R, indent=1))
