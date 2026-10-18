"""P4: external validation of the KINEMATIC/behaviour axis on an independent open
dataset (Zenodo 6570972, Driving Events). Do kinematic features (accel magnitude,
jerk, harsh events) separate AGGRESSIVE vs NON-AGGRESSIVE driving events, with the
driver held out? Validates that the axis used for VED driver-excess is real & generalizes.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "zenodo_driving")

def feats(win):
    """kinematic features for one event window (400 Hz linear accel, gravity removed)."""
    x, y, z = win["x"].to_numpy(), win["y"].to_numpy(), win["z"].to_numpy()
    t = win["t"].to_numpy()
    mag = np.sqrt(x**2 + y**2 + z**2)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0025
    jerk = np.diff(mag) / dt if len(mag) > 1 else np.array([0.0])
    dur = t[-1] - t[0] if len(t) > 1 else 1.0
    return {
        "mag_mean": mag.mean(), "mag_std": mag.std(),
        "mag_p95": np.percentile(mag, 95), "mag_max": mag.max(),
        "jerk_rms": np.sqrt(np.mean(jerk**2)),
        "jerk_p95": np.percentile(np.abs(jerk), 95),
        "harsh_per_s": np.sum(mag > 3.0) / dur,      # >3 m/s^2 events
        "x_std": x.std(), "y_std": y.std(), "z_std": z.std(),
        "energy": np.mean(mag**2),
    }

rows, labels, drivers = [], [], []
for i in [1, 2, 3]:
    a = pd.read_csv(os.path.join(RAW, f"Linear_Acceleration_{i}.csv"))
    a.columns = ["t", "x", "y", "z"]
    e = pd.read_csv(os.path.join(RAW, f"Labeled_events_{i}.csv"))
    tvals = a["t"].to_numpy()
    for _, r in e.iterrows():
        m = (tvals >= r["start"]) & (tvals <= r["end"])
        if m.sum() < 20:
            continue
        rows.append(feats(a.loc[m]))
        labels.append(1 if int(r["target"]) >= 1 else 0)   # aggressive vs non
        drivers.append(i)

X = pd.DataFrame(rows); yb = np.array(labels); grp = np.array(drivers)
print(f"events used: {len(X)}  aggressive={yb.sum()}  non-aggressive={(yb==0).sum()}  drivers={set(grp)}")

# --- univariate separation (maps to VED kinematic features) ---
print("\n=== univariate aggressive-vs-normal separation ===")
for c in ["jerk_rms", "mag_std", "harsh_per_s", "mag_p95", "energy"]:
    agg, non = X.loc[yb == 1, c], X.loc[yb == 0, c]
    try:
        auc = roc_auc_score(yb, X[c])
    except Exception:
        auc = np.nan
    u, p = mannwhitneyu(agg, non)
    print(f"  {c:12s}: aggr_med={agg.median():.3f}  non_med={non.median():.3f}  "
          f"AUC={auc:.3f}  MWU p={p:.1e}")

# --- classifier, DRIVER HELD OUT (leave-one-driver-out) ---
print("\n=== classifier (leave-one-driver-out) ===")
for name, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                  ("rf", RandomForestClassifier(n_estimators=300, random_state=0))]:
    logo = LeaveOneGroupOut(); probs = np.full(len(X), np.nan)
    for tr, te in logo.split(X, yb, grp):
        clf.fit(X.iloc[tr], yb[tr]); probs[te] = clf.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(yb, probs)
    # per-driver AUC
    per = []
    for d in sorted(set(grp)):
        mask = grp == d
        if len(set(yb[mask])) == 2:
            per.append(roc_auc_score(yb[mask], probs[mask]))
    print(f"  {name:6s}: pooled held-out AUC={auc:.3f}  per-driver AUC={[round(v,3) for v in per]}")
