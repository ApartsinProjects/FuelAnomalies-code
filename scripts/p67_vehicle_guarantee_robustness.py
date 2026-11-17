"""P67: robustness of the vehicle-level guarantee of p65 part C (alpha 0.10, 102 calibration / 34 test vehicles,
kinematic score referenced to the vehicle, 200 vehicle splits). Varied one at a time: the pre-set start of the fixed
sequence (40%, 50%, 60%, 25% coverage), the level delta (0.1, 0.05, 0.025; 0.025 is a Bonferroni correction for the
four configurations examined in p65), and a threshold grid fixed in advance in margin units (2.0 down to 0 in steps of
0.05) instead of quantiles of the calibration margins. Reported per configuration: splits with decisions, median
coverage, median held-out selective error pooled over variants and averaged over vehicles (the test controls the
vehicle-level ratio), and the share of splits whose pooled held-out error exceeds alpha.
Output: data/processed/p67_vehicle_guarantee_robustness.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
kraw, cz0, grp, n, SF = (g[k] for k in ["kraw", "cz0", "grp", "n", "S_fault"])
gs = pd.Series(kraw).groupby(grp); kw0 = ((kraw - gs.transform("mean").to_numpy()) / (gs.transform("std").to_numpy() + 1e-9)); sc = kw0.std(); kw0 = kw0 / sc
def wsr_p(x, alpha, delta):
    m = len(x); i = np.arange(1, m + 1); mu = (0.5 + np.cumsum(x)) / (1 + i); s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (1 + i)
    s2 = np.r_[0.25, s2[:-1]]; nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / (m * s2)), 1.0)
    return float(min(1.0, np.exp(-np.cumsum(np.log(1 - nu * (x - alpha))).max())))
def calibrate(conf, err, G, cal, alpha, delta, rng, start=None, numeric=None):
    vid = {u: i for i, u in enumerate(cal)}; m = np.isin(G, cal); idx = np.array([vid[u] for u in G[m]]); c, e = conf[m], err[m]
    nv = np.bincount(idx, minlength=len(cal)).astype(float); order = rng.permutation(len(cal)); keep = np.inf
    grid = numeric if numeric is not None else np.quantile(c, np.linspace(1 - start, 0.0, int(round((1 - start) * 100)) + 1))
    for tau in grid:
        d = c >= tau; y = (np.bincount(idx[d], weights=e[d], minlength=len(cal)) - alpha * np.bincount(idx[d], minlength=len(cal))) / nv + alpha
        if wsr_p(y[order], alpha, delta) > delta: break
        keep = tau
    return keep
vehs = np.unique(grp); ALPHA = 0.10; R = {}
CONF = [("start40", dict(start=0.4), 0.1), ("start50", dict(start=0.5), 0.1), ("start60", dict(start=0.6), 0.1), ("start25", dict(start=0.25), 0.1),
        ("start50_delta0.05", dict(start=0.5), 0.05), ("start50_delta0.025", dict(start=0.5), 0.025),
        ("numeric_grid_2.0", dict(numeric=np.arange(2.0, -1e-9, -0.05)), 0.1), ("numeric_grid_2.0_delta0.025", dict(numeric=np.arange(2.0, -1e-9, -0.05)), 0.025)]
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    K2 = np.r_[kw0 + Sd / sc, kw0]; C2 = np.r_[cz0, cz0 + SF]; y2 = np.r_[np.zeros(n), np.ones(n)].astype(int); G2 = np.r_[grp, grp]
    la = N.logpdf(K2, Sd / sc, 1) - N.logpdf(K2, 0, 1); lb = N.logpdf(C2, SF, 1) - N.logpdf(C2, 0, 1); conf = np.abs(la - lb); err = ((lb > la) != y2).astype(float)
    for name, kw, delta in CONF:
        cov, pe, ve, ex = [], [], [], []
        for s in range(200):
            r = np.random.default_rng(500 + s); vs = vehs.copy(); r.shuffle(vs); cal, te = vs[:102], vs[102:]; assert not set(cal) & set(te)
            tau = calibrate(conf, err, G2, cal, ALPHA, delta, r, **kw); m = np.isin(G2, te); d = m & (conf >= tau); cov.append(d.sum() / m.sum())
            if d.sum():
                pe.append(err[d].mean()); ex.append(err[d].mean() > ALPHA)
                ve.append(float(pd.Series(err[d]).groupby(G2[d]).mean().mean()))
        R[f"{lvl}_{name}"] = {"splits_with_decisions": len(pe), "coverage_median": round(float(np.median(cov)), 3),
                              "pooled_error_median": round(float(np.median(pe)), 4) if pe else None, "vehicle_mean_error_median": round(float(np.median(ve)), 4) if ve else None,
                              "exceed_rate": round(float(np.mean(ex)), 3) if ex else None}
        print(lvl, name, R[f"{lvl}_{name}"], flush=True)
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p67_vehicle_guarantee_robustness.json"), "w"), indent=1)
