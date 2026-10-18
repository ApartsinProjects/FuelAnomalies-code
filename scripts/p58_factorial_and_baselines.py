"""P58: four-condition factorial injection, matched-coverage baselines, risk-coverage, and a risk-control audit table
(external review brief items 3, 4, 8). One pass on the 11,304-trip panel, window operating point (d 0.65) and trip
point (d 1.3), fault S_f = 1.94, frozen UAH score, within-vehicle trim z.
CONDITIONS (equal priors): none (real trip unchanged), driver only, fault only, both (driver and fault shifts added).
OUTPUTS of the published rule: none if l_A, l_B < 0; otherwise the cause with the larger log-likelihood ratio. An
exploratory "both" output is added when l_A > 0 and l_B > 0 and |l_A - l_B| < 1 (not part of the published method).
BASELINES on the same two scores (all two-way, for the forced-choice comparison on single-cause variants):
  fusion (published; with equal variances it is exactly the linear rule S_f*z_B - S_d*z_A > (S_f^2 - S_d^2)/2, asserted
  below), argmax of the two standardized scores, single-axis thresholds (trim only: fault if z_B > S_f/2; kinematic
  only: driver if z_A > S_d/2), logistic regression on (z_A, z_B) trained
  on injection labels of held-out vehicles (an upper reference), target-only anomaly scoring.
METRICS: accuracy, balanced accuracy, macro-F1, per-class recall; risk-coverage curves (selective error vs the share of
decisions issued, ranking by each method's own margin); risk-control audit at alpha 0.05 / 0.10 with vehicle-disjoint
calibration and test halves, counts of decisions, errors, vehicles and the Clopper-Pearson interval of the realized
error; leakage assertions (no vehicle and no base trip on both sides of a split).
Output: data/processed/p58_factorial_and_baselines.json"""
import os, io, json, contextlib, runpy, numpy as np
from scipy.stats import norm as N, chi2, beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
Zv, Wk, dirn, kraw, cz0, grp, n, SF = (g[k] for k in ["Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n", "S_fault"])
z = lambda x: (x - kraw.mean()) / kraw.std()
def cp(k, m): return [round(float(beta.ppf(0.025, k, m - k + 1)) if k else 0.0, 4), round(float(beta.ppf(0.975, k + 1, m - k)) if k < m else 1.0, 4)]
def tau_for(conf, err, alpha, delta=0.1, K=100):
    o = np.argsort(-conf); c = conf[o]; e = err[o]; ns = np.arange(1, len(c) + 1)
    pos = np.unique(np.clip((np.arange(1, K + 1) / K * len(c)).astype(int), 1, len(c))) - 1
    k = np.cumsum(e)[pos]; ucb = beta.ppf(1 - delta / K, k + 1, np.maximum(ns[pos] - k, 1e-9))
    ok = np.where(ucb <= alpha)[0]; return c[pos[ok[-1]]] if len(ok) else np.inf
R = {}
mu = Zv.mean(0); Si = np.linalg.inv(np.cov(Zv, rowvar=False))
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn
    kA = z((Zv + PERT) @ Wk); k0 = z(kraw)
    # ---- factorial: 0 none, 1 driver, 2 fault, 3 both ----
    KZ = np.r_[k0, kA, k0, kA]; CZ = np.r_[cz0, cz0, cz0 + SF, cz0 + SF]; y = np.repeat([0, 1, 2, 3], n)
    lA = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lB = N.logpdf(CZ, SF, 1) - N.logpdf(CZ, 0, 1)
    pub = np.where((lA < 0) & (lB < 0), 0, np.where(lB > lA, 2, 1))
    expl = np.where((lA > 0) & (lB > 0) & (np.abs(lA - lB) < 1), 3, pub)
    def cm(pred, labels):
        M = np.zeros((4, len(labels)), int)
        for t in range(4):
            for j, l in enumerate(labels): M[t, j] = int(np.sum((y == t) & (pred == l)))
        return M.tolist()
    R[f"{lvl}_factorial"] = {"rows_true": ["none", "driver", "fault", "both"], "cols_published": ["none", "driver", "fault"],
        "confusion_published": cm(pub, [0, 1, 2]), "cols_exploratory": ["none", "driver", "fault", "both"], "confusion_exploratory": cm(expl, [0, 1, 2, 3]),
        "false_driver_on_fault_only": round(float(np.mean(pub[y == 2] == 1)), 3), "false_fault_on_driver_only": round(float(np.mean(pub[y == 1] == 2)), 3),
        "none_on_none": round(float(np.mean(pub[y == 0] == 0)), 3), "exploratory_both_on_both": round(float(np.mean(expl[y == 3] == 3)), 3),
        "exploratory_both_on_single": round(float(np.mean(expl[(y == 1) | (y == 2)] == 3)), 3)}
    # ---- two-way methods on single-cause variants ----
    K2 = np.r_[kA, k0]; C2 = np.r_[cz0, cz0 + SF]; y2 = np.r_[np.zeros(n), np.ones(n)].astype(int); G2 = np.r_[grp, grp]
    la = N.logpdf(K2, Sd, 1) - N.logpdf(K2, 0, 1); lb = N.logpdf(C2, SF, 1) - N.logpdf(C2, 0, 1)
    vehs = np.unique(grp); rng.shuffle(vehs); calv = vehs[: len(vehs) // 2]; cal = np.isin(G2, calv); te = ~cal
    assert not set(G2[cal]) & set(G2[te]), "vehicle leakage"
    base_trip = np.r_[np.arange(n), np.arange(n)]; assert not set(base_trip[cal]) & set(base_trip[te]), "paired base-trip leakage"
    lr = LogisticRegression().fit(np.c_[K2, C2][cal], y2[cal])
    Xk = np.r_[Zv + PERT, Zv]; p_k = chi2.sf(np.einsum("ij,jk,ik->i", Xk - mu, Si, Xk - mu), 4); p_c = 2 * N.sf(np.abs((C2 - cz0.mean()) / cz0.std()))
    # fusion IS the transparent linear rule: lb - la = SF*zB - Sd*zA - (SF^2 - Sd^2)/2 (equal-variance Gaussians)
    assert np.allclose(lb - la, SF * C2 - Sd * K2 - (SF ** 2 - Sd ** 2) / 2), "LLR is not linear in the scores"
    methods = {"fusion": (lb - la, None), "argmax": (C2 - K2, None),
               "trim_only_threshold": (C2 - SF / 2, None), "kinematic_only_threshold": (Sd / 2 - K2, None),
               "logistic_upper_reference": (lr.decision_function(np.c_[K2, C2]), None), "target_only": (np.log(p_k + 1e-300) - np.log(p_c + 1e-300), None)}
    out = {}
    for name, (s, _) in methods.items():
        pred = (s > 0).astype(int); mte = te
        out[name] = {"accuracy": round(float((pred[mte] == y2[mte]).mean()), 3), "balanced_accuracy": round(float(balanced_accuracy_score(y2[mte], pred[mte])), 3),
                     "macro_f1": round(float(f1_score(y2[mte], pred[mte], average="macro")), 3),
                     "recall_driver": round(float((pred[mte & (y2 == 0)] == 0).mean()), 3), "recall_fault": round(float((pred[mte & (y2 == 1)] == 1).mean()), 3)}
        o = np.argsort(-np.abs(s[mte])); e = (pred[mte] != y2[mte])[o]; cov = np.arange(1, len(e) + 1) / len(e); risk = np.cumsum(e) / np.arange(1, len(e) + 1)
        out[name]["risk_at_coverage"] = {str(c): round(float(risk[int(c * len(e)) - 1]), 4) for c in (0.25, 0.5, 0.75, 1.0)}
        out[name]["risk_coverage_curve"] = [[round(float(cov[i]), 3), round(float(risk[i]), 4)] for i in np.linspace(len(e) // 50, len(e) - 1, 40).astype(int)]
    R[f"{lvl}_two_way"] = out
    # ---- risk-control audit (published fusion margin) ----
    err = ((lb > la) != y2).astype(int); conf = np.abs(la - lb); aud = {}
    for alpha in (0.05, 0.10):
        t = tau_for(conf[cal], err[cal], alpha); m = te & (conf >= t); k = int(err[m].sum()); d = int(m.sum())
        aud[str(alpha)] = {"threshold_margin": round(float(t), 3), "calibration_vehicles": int(len(calv)), "test_vehicles": int(len(set(G2[te]))),
                           "test_variants": int(te.sum()), "decisions": d, "errors": k, "coverage": round(d / te.sum(), 3),
                           "selective_error": round(k / d, 4) if d else None, "selective_error_CP95": cp(k, d) if d else None,
                           "test_vehicles_with_decisions": int(len(set(G2[m])))}
    R[f"{lvl}_risk_control_audit"] = aud
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p58_factorial_and_baselines.json"), "w"), indent=1)
print(json.dumps({k: (v if "two_way" not in k else {m: {kk: vv for kk, vv in d.items() if kk != "risk_coverage_curve"} for m, d in v.items()}) for k, v in R.items()}, indent=1))
