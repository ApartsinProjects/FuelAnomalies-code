"""Why does the SHAP-style fuel-anomaly baseline miss faults? Measure how much its fuel model responds to the
injected trim shift vs the injected kinematic shift (same models as p33)."""
import io, contextlib, runpy, os, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "p33_attribution_baselines.py"))
Zv, cz0, Wk, dirn, ksd, S_fault, models, fold_of, ex = g["Zv"], g["cz0"], g["Wk"], g["dirn"], g["ksd"], g["S_fault"], g["b1_models"], g["fold_of"], g["v"].excess.to_numpy()
X0 = np.c_[Zv, cz0]; PERT = (1.3 * ksd / (Wk @ dirn)) * dirn
Xd = np.c_[Zv + PERT, cz0]; Xf = np.c_[Zv, cz0 + S_fault]
dd, df = np.empty(len(X0)), np.empty(len(X0))
for k, m in enumerate(models):
    i = fold_of == k; b = m.predict(X0[i]); dd[i] = m.predict(Xd[i]) - b; df[i] = m.predict(Xf[i]) - b
print("mean predicted excess change, driver variant: %.3f L/100km, fault variant: %.3f L/100km" % (dd.mean(), df.mean()))
print("fleet excess SD %.2f; corr(real excess, LTFT z) %.3f" % (ex.std(), np.corrcoef(ex, cz0)[0, 1]))
