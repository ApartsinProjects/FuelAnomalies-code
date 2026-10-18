"""P43 check: is the UAH driver effect size d = 0.65 inflated by scoring the logistic on its own training windows?
Invariant: leave-one-driver-out d must be within the in-sample d's 95% interval [0.41, 0.84]."""
import os, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
def kin(spd):
    v = np.asarray(spd, float) / 3.6; a = np.diff(v); j = np.diff(a); d = np.nansum(v) / 1000.0
    return [np.nanstd(v) * 3.6, np.nanpercentile(a, 95), np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2))]
X, y, g = [], [], []
for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
    drv = os.path.basename(f).split("_")[0]
    d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); d = d.assign(_s=np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]))
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]
        for _, w in rec.groupby(((rec["Timestamp (seconds)"] - t0) // 30).astype(int)):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            X.append(kin(w["Speed (km/h)"])); y.append(int(c == 2)); g.append(drv)
X, y, g = np.array(X), np.array(y), np.array(g)
def cd(s, y): return (s[y == 1].mean() - s[y == 0].mean()) / np.sqrt((s[y == 1].var() + s[y == 0].var()) / 2)
su = StandardScaler().fit(X); s_in = su.transform(X) @ LogisticRegression(max_iter=1000).fit(su.transform(X), y).coef_[0]
s_lo = np.zeros(len(y))
for k in np.unique(g):
    tr = g != k; s2 = StandardScaler().fit(X[tr]); W = LogisticRegression(max_iter=1000).fit(s2.transform(X[tr]), y[tr]).coef_[0]
    sc = s2.transform(X) @ W; s_lo[~tr] = (sc[~tr] - sc[tr].mean()) / sc[tr].std()   # held-out driver scored on the training scale
print("windows", len(y), "aggr", y.sum(), "drivers", np.unique(g).tolist())
print("in-sample d", round(cd(s_in, y), 3), " leave-one-driver-out d", round(cd(s_lo, y), 3))
rng = np.random.default_rng(0); u = np.unique(g); bs = []
for _ in range(2000):
    idx = np.concatenate([np.flatnonzero(g == k) for k in rng.choice(u, len(u))]); bs.append(cd(s_lo[idx], y[idx]))
print("LODO d driver-bootstrap 95%", np.round(np.nanpercentile(bs, [2.5, 97.5]), 3))
