"""P48: construct-matched real-data signature contrast, replacing the bench column of Table 6 (the bench "fault excess"
is a model artifact, p47). One pass, one definition on three real datasets: correlation of fuel excess with the frozen
kinematic score and with the trim axis, cluster bootstrap (vehicles for VED, sessions for HCRL and Sonata), TOST at +/-0.1.
Invariant: VED trim cell must reproduce p36 C (-0.041) and HCRL/Sonata cells must reproduce p36 C (-0.023 / 0.100)."""
import os, io, json, contextlib, runpy, numpy as np
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
def cr(x, y, cl, B=2000):
    x, y, cl = map(np.asarray, (x, y, cl)); u = np.unique(cl); idx = {k: np.flatnonzero(cl == k) for k in u}
    r0 = np.corrcoef(x, y)[0, 1]; bs = np.array([np.corrcoef(x[s], y[s])[0, 1] for s in (np.concatenate([idx[k] for k in rng.choice(u, len(u))]) for _ in range(B))])
    se = bs.std(); p = max(N.sf((r0 + 0.1) / se), N.cdf((r0 - 0.1) / se))
    return {"r": round(float(r0), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "ci90": [round(float(np.percentile(bs, 5)), 3), round(float(np.percentile(bs, 95)), 3)], "tost_p": round(float(p), 4), "n": int(len(x)), "clusters": int(len(u))}
R = {}
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, kraw, cz0, grp = g["v"], g["kraw"], g["cz0"], g["grp"]
R["VED"] = {"kin": cr(v.excess, kraw, grp), "trim": cr(v.excess, cz0, grp)}
for name, script in (("HCRL", "p29_hcrl.py"), ("Sonata", "p31_sonata.py")):
    with contextlib.redirect_stdout(io.StringIO()):
        W = runpy.run_path(os.path.join(ROOT, "scripts", script))["W"]
    R[name] = {"kin": cr(W.excess, W.kin, W.sess), "trim": cr(W.excess, W.ltft_dev, W.sess)}
R["checks"] = {"VED_trim_reproduces_p36": abs(R["VED"]["trim"]["r"] + 0.041) < 0.005,
               "HCRL_trim_reproduces_p36": abs(R["HCRL"]["trim"]["r"] + 0.023) < 0.005, "Sonata_trim_reproduces_p36": abs(R["Sonata"]["trim"]["r"] - 0.100) < 0.005,
               "kin_gt_trim_all": all(R[k]["kin"]["r"] > R[k]["trim"]["r"] for k in ("VED", "HCRL", "Sonata"))}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p48_signature_contrast_real.json"), "w"), indent=1); print(json.dumps(R, indent=1))
