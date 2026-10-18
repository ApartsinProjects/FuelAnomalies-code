"""P44 check of the Section 7 aggregation AUC: contiguous blocks of n consecutive readings (file order) instead of
i.i.d. draws, which ignore within-run correlation. Invariant: n=1 must reproduce the per-sample AUC (0.54)."""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
e = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv")); e["comb"] = (14.7 / e["AFR"] - 1) * 100
nm = e[e.Fault == 0]["comb"].to_numpy(); mu, sd = nm.mean(), nm.std()
s = {k: (e[e.Fault == k]["comb"].to_numpy() - mu) / sd for k in (0, 1)}
print("rows normal/rich", len(s[0]), len(s[1]))
lag1 = {k: round(float(np.corrcoef(v[:-1], v[1:])[0, 1]), 3) for k, v in s.items()}; print("lag-1 autocorrelation", lag1)
for n in (1, 5, 10, 20, 50):
    b = {k: v[: len(v) // n * n].reshape(-1, n).mean(1) for k, v in s.items()}
    print(n, "contiguous-block AUC", round(roc_auc_score(np.r_[np.ones(len(b[1])), np.zeros(len(b[0]))], np.r_[b[1], b[0]]), 3), "blocks", len(b[1]), len(b[0]))
