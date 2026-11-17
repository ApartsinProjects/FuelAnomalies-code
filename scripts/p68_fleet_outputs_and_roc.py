"""P68: (1) outputs of Algorithm 1 on the UNCHANGED trips of the VED fleet (no injection) at both operating points,
with the abstention thresholds of the risk audit (p58: calibrated on 68 vehicles, alpha 0.05 and 0.10) applied to
the other 68 vehicles' real trips: shares of none / driver / fault / abstain. The window operating point uses the
30-second driver effect (0.65), the trip operating point the trip effect (1.3); in both the scored unit is a trip.
(2) Figure 2: ROC of the deployed four-feature 1 Hz speed signature on UAH-DriveSet, leave-one-driver-out, pooled
and per held-out driver (replaces the 400 Hz Driving Events ROC).
Output: data/processed/p68_fleet_outputs.json, figures/fig4_behaviour_roc.{svg,png}"""
import os, io, glob, json, contextlib, runpy, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm as N, beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
Zv, Wk, dirn, kraw, cz0, grp, n, SF, kin = (g[k] for k in ["Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n", "S_fault", "kin"])
def tau_for(conf, err, alpha, delta=0.1, K=100):      # identical to p58
    o = np.argsort(-conf); c = conf[o]; e = err[o]; ns = np.arange(1, len(c) + 1)
    pos = np.unique(np.clip((np.arange(1, K + 1) / K * len(c)).astype(int), 1, len(c))) - 1
    k = np.cumsum(e)[pos]; ucb = beta.ppf(1 - delta / K, k + 1, np.maximum(ns[pos] - k, 1e-9)); ok = np.where(ucb <= alpha)[0]
    return c[pos[ok[-1]]] if len(ok) else np.inf
R = {}; k0 = (kraw - kraw.mean()) / kraw.std(); rng = np.random.default_rng(0)
for lvl, Sd in (("window_operating_point", 0.65), ("trip_operating_point", 1.3)):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn; kA = ((Zv + PERT) @ Wk - kraw.mean()) / kraw.std()
    K2 = np.r_[kA, k0]; C2 = np.r_[cz0, cz0 + SF]; y2 = np.r_[np.zeros(n), np.ones(n)]; G2 = np.r_[grp, grp]
    la = N.logpdf(K2, Sd, 1) - N.logpdf(K2, 0, 1); lb = N.logpdf(C2, SF, 1) - N.logpdf(C2, 0, 1)
    vehs = np.unique(grp); rng.shuffle(vehs); cal = vehs[: len(vehs) // 2]; calm = np.isin(G2, cal); te = ~np.isin(grp, cal)
    ld = N.logpdf(k0, Sd, 1) - N.logpdf(k0, 0, 1); lf = N.logpdf(cz0, SF, 1) - N.logpdf(cz0, 0, 1)
    base = np.where((ld < 0) & (lf < 0), "none", np.where(lf > ld, "fault", "driver")).astype(object)
    out = {"no_abstention_all_trips": {k: round(float((base == k).mean()), 3) for k in ("none", "driver", "fault")}}
    for alpha in (0.05, 0.10):
        tau = tau_for(np.abs(la - lb)[calm], ((lb > la) != y2)[calm].astype(int), alpha)
        o = base.copy(); o[(o != "none") & (np.abs(ld - lf) < tau)] = "abstain"; o = o[te]
        out[f"alpha_{alpha}"] = {"threshold_margin": round(float(tau), 3), "held_out_trips": int(te.sum()), **{k: round(float((o == k).mean()), 3) for k in ("none", "driver", "fault", "abstain")}}
    R[lvl] = out; print(lvl, out, flush=True)
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p68_fleet_outputs.json"), "w"), indent=1)
# ---- Figure 2: deployed signature ROC, leave-one-driver-out ----
X, y, dr = [], [], []
for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "uah", "D*_merged.csv"))):
    drv = os.path.basename(f).split("_")[0]; d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); d = d.assign(_s=np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]))
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]
        for _, w in rec.groupby(((rec["Timestamp (seconds)"] - t0) // 30).astype(int)):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            X.append(kin(w["Speed (km/h)"])); y.append(int(c == 2)); dr.append(drv)
X, y, dr = np.array(X), np.array(y), np.array(dr); sc = np.zeros(len(y))
for k in np.unique(dr):
    tr = dr != k; s = StandardScaler().fit(X[tr]); sc[~tr] = LogisticRegression(max_iter=1000).fit(s.transform(X[tr]), y[tr]).decision_function(s.transform(X[~tr]))
auc = roc_auc_score(y, sc); assert abs(auc - 0.694) < 0.002, auc
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "font.size": 10.5, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for k in np.unique(dr):
    m = dr == k; fpr, tpr, _ = roc_curve(y[m], sc[m]); ax.plot(fpr, tpr, color="#9a9a9a", lw=1, label=f"{k} ({roc_auc_score(y[m], sc[m]):.2f})")
fpr, tpr, _ = roc_curve(y, sc); ax.plot(fpr, tpr, color="#7a2020", lw=2.4, label=f"pooled ({auc:.3f})"); ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--")
ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.set_title("Deployed speed signature, leave-one-driver-out", fontsize=10.5, fontweight="bold")
ax.legend(title="held-out driver (AUC)", fontsize=8, title_fontsize=8, frameon=False, loc="lower right")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
fig.tight_layout()
for e in ("svg", "png"): fig.savefig(os.path.join(ROOT, "figures", f"fig4_behaviour_roc.{e}"), bbox_inches="tight", dpi=150)
print("figure written; pooled AUC", round(float(auc), 3))
