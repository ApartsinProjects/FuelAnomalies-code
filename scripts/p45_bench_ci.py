"""P45: 95% intervals for Table 5 (bench excess fuel by fault class, BSFC gap for rich).
Bench readings are strongly autocorrelated (lag-1 0.76-0.93, p44), so intervals use a moving-block bootstrap over
contiguous blocks of each class (block 200 readings; 50 and 500 reported as a sensitivity check).
Invariant: the point estimates must reproduce Table 5 (+0.50 L/100km / +6.2% rich; +0.15 / +1.8% ignition)."""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
d = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv"))
OP = ["RPM", "MAP", "TPS", "Force", "Power", "Speed"]; T = "Consumption L/100KM"; med = d[T].median()
norm = d[d.Fault == 0].copy(); oof = np.full(len(norm), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=0).split(norm):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0).fit(norm[OP].to_numpy()[tr], norm[T].to_numpy()[tr])
    oof[te] = m.predict(norm[OP].to_numpy()[te])
nres = norm[T].to_numpy() - oof
M = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0).fit(norm[OP].to_numpy(), norm[T].to_numpy())
d["bsfc"] = d["Consumption L/H"] / d["Power"].clip(lower=0.1)
def blocks(x, L):
    nb = int(np.ceil(len(x) / L)); st = rng.integers(0, len(x) - L + 1, nb)
    return np.concatenate([x[s:s + L] for s in st])[: len(x)]
R = {}
for L in (50, 200, 500):
    for k in (1, 2, 3):
        exc = d[d.Fault == k][T].to_numpy() - M.predict(d[d.Fault == k][OP].to_numpy())
        pen = exc.mean() - nres.mean(); bs = [blocks(exc, L).mean() - blocks(nres, L).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        R[f"L{L}_fault{k}"] = {"penalty": round(pen, 3), "ci": [round(lo, 3), round(hi, 3)], "pct": round(pen / med * 100, 2), "pct_ci": [round(lo / med * 100, 2), round(hi / med * 100, 2)]}
    b0, b1 = d[d.Fault == 0].bsfc.to_numpy(), d[d.Fault == 1].bsfc.to_numpy(); pt = np.median(b1) / np.median(b0) - 1
    bs = [np.median(blocks(b1, L)) / np.median(blocks(b0, L)) - 1 for _ in range(2000)]
    R[f"L{L}_bsfc_rich"] = {"pct": round(pt * 100, 2), "ci": [round(float(np.percentile(bs, 2.5)) * 100, 2), round(float(np.percentile(bs, 97.5)) * 100, 2)]}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p45_bench_ci.json"), "w"), indent=1); print(json.dumps(R, indent=1))

# ---- primary interval: cluster bootstrap over operating segments (a segment ends at an RPM jump > 300) ----
def segs(sub):
    r = sub.RPM.to_numpy(); return np.concatenate([[0], np.cumsum(np.abs(np.diff(r)) > 300)])
def seg_means(x, g): return np.array([x[g == s].mean() for s in np.unique(g)]), np.array([np.sum(g == s) for s in np.unique(g)])
n0 = d[d.Fault == 0]; g0 = segs(n0); nr_m, nr_w = seg_means(nres, g0)
C = {"normal_segments": int(len(nr_m))}
for k in (1, 2, 3):
    sub = d[d.Fault == k]; g = segs(sub); exc = sub[T].to_numpy() - M.predict(sub[OP].to_numpy()); em, ew = seg_means(exc, g)
    pen = exc.mean() - nres.mean(); bs = []
    for _ in range(4000):
        a = rng.integers(0, len(em), len(em)); b = rng.integers(0, len(nr_m), len(nr_m))
        bs.append(np.average(em[a], weights=ew[a]) - np.average(nr_m[b], weights=nr_w[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    C[f"fault{k}"] = {"segments": int(len(em)), "penalty": round(pen, 3), "ci": [round(lo, 3), round(hi, 3)], "pct": round(pen / med * 100, 2),
                      "pct_ci": [round(lo / med * 100, 2), round(hi / med * 100, 2)], "share_segments_positive": round(float(np.mean(em > np.average(nr_m, weights=nr_w))), 3)}
R["segment_cluster_bootstrap"] = C
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p45_bench_ci.json"), "w"), indent=1); print(json.dumps(C, indent=1))

# ---- robust segment-level summary: L/100km explodes at near-zero speed, so a few slow segments dominate the mean ----
from scipy.stats import binomtest
ref = np.median(nr_m); Rb = {"normal_share_above_own_median": round(float(np.mean(nr_m > ref)), 3)}   # invariant ~0.5
for k in (1, 2, 3):
    sub = d[d.Fault == k]; g = segs(sub); exc = sub[T].to_numpy() - M.predict(sub[OP].to_numpy()); em, _ = seg_means(exc, g)
    pos = int(np.sum(em > ref)); md = np.median(em) - ref
    bs = [np.median(em[rng.integers(0, len(em), len(em))]) - np.median(nr_m[rng.integers(0, len(nr_m), len(nr_m))]) for _ in range(4000)]
    Rb[f"fault{k}"] = {"segments": int(len(em)), "above_normal_median": pos, "sign_test_p": float(binomtest(pos, len(em), 0.5).pvalue),
                       "median_seg_penalty": round(float(md), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                       "pct": round(float(md / med * 100), 2), "pct_ci": [round(float(np.percentile(bs, 2.5) / med * 100), 2), round(float(np.percentile(bs, 97.5) / med * 100), 2)]}
print("speed of segments, median km/h by class:", {k: round(float(d[d.Fault == k].Speed.median()), 1) for k in range(4)})
R["segment_robust"] = Rb; json.dump(R, open(os.path.join(ROOT, "data", "processed", "p45_bench_ci.json"), "w"), indent=1); print(json.dumps(Rb))

# ---- PRIMARY (final): the experimental unit is the operating condition. EngineFaultDB holds each class as consecutive
# 1,000-reading blocks recorded at one steady condition (normal 16, rich 11, lean 15, ignition 14). Cluster bootstrap over blocks.
# Data defect found: 14 of 14 ignition blocks reproduce lean blocks (identical MAP/TPS traces, other channels within ~2%).
def blk(sub): return np.arange(len(sub)) // 1000
nb0 = blk(n0); nrb = np.array([nres[nb0 == i].mean() for i in np.unique(nb0)])
afr0 = np.array([n0.AFR.to_numpy()[nb0 == i].mean() for i in np.unique(nb0)]); C2 = {"normal_conditions": int(len(nrb))}
for k in (1, 2, 3):
    sub = d[d.Fault == k].reset_index(drop=True); b = blk(sub); exc = sub[T].to_numpy() - M.predict(sub[OP].to_numpy())
    eb = np.array([exc[b == i].mean() for i in np.unique(b)]); ab = np.array([sub.AFR.to_numpy()[b == i].mean() for i in np.unique(b)])
    pen = exc.mean() - nres.mean(); bs, ba = [], []
    for _ in range(4000):
        i = rng.integers(0, len(eb), len(eb)); j = rng.integers(0, len(nrb), len(nrb))
        bs.append(eb[i].mean() - nrb[j].mean()); ba.append((14.7 / ab[i].mean() - 14.7 / afr0[j].mean()) * 100)
    C2[f"fault{k}"] = {"conditions": int(len(eb)), "penalty": round(pen, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                       "pct": round(pen / med * 100, 2), "pct_ci": [round(float(np.percentile(bs, 2.5) / med * 100), 1), round(float(np.percentile(bs, 97.5) / med * 100), 1)],
                       "fuel_per_air_shift_pp": round((14.7 / sub.AFR.mean() - 14.7 / n0.AFR.mean()) * 100, 2),
                       "shift_ci": [round(float(np.percentile(ba, 2.5)), 2), round(float(np.percentile(ba, 97.5)), 2)],
                       "conditions_afr_below_12.5": int(np.sum(ab < 12.5)), "normal_conditions_afr_below_12.5": int(np.sum(afr0 < 12.5))}
R["condition_cluster_bootstrap"] = C2; json.dump(R, open(os.path.join(ROOT, "data", "processed", "p45_bench_ci.json"), "w"), indent=1)
print("CONDITION-LEVEL", json.dumps(C2))
