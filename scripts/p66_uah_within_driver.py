"""P66: driver effect on the frozen UAH score measured WITHIN driver: each driver's 30 s windows are referenced to that
driver's normal-driving mean and SD. This is the construct that matches a kinematic score referenced to the vehicle's
own trips. Reported: pooled Cohen's d (reproduces 0.65), within-driver d, mean aggressive shift in normal-driving SDs
with a driver bootstrap, per-driver shifts. Output: data/processed/p66_uah_within_driver.json"""
import os, io, glob, json, contextlib, runpy, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
kin, su, Wk = g["kin"], g["su"], g["Wk"]
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "uah", "D*_merged.csv"))):
    drv = os.path.basename(f).split("_")[0]; d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); d = d.assign(_s=np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]))
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]
        for _, w in rec.groupby(((rec["Timestamp (seconds)"] - t0) // 30).astype(int)):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            rows.append((drv, int(c == 2), float((su.transform(np.array([kin(w["Speed (km/h)"])])) @ Wk)[0])))
U = pd.DataFrame(rows, columns=["drv", "y", "s"])
def cohen(s, y): return float(abs(s[y == 1].mean() - s[y == 0].mean()) / np.sqrt((s[y == 1].var() + s[y == 0].var()) / 2))
U["z"] = np.nan
for k, gk in U.groupby("drv"):
    ref = gk.s[gk.y == 0]; U.loc[gk.index, "z"] = (gk.s - ref.mean()) / ref.std()
rng = np.random.default_rng(0); D = U.drv.unique(); bs = []
for _ in range(1000):
    q = pd.concat([U[U.drv == p] for p in rng.choice(D, len(D))]); bs.append(q.z[q.y == 1].mean())
R = {"windows": int(len(U)), "pooled_d": round(cohen(U.s.to_numpy(), U.y.to_numpy()), 3), "within_driver_d": round(cohen(U.z.to_numpy(), U.y.to_numpy()), 3),
     "mean_shift_in_normal_sd": round(float(U.z[U.y == 1].mean()), 3), "mean_shift_ci95_driver_bootstrap": [round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])],
     "per_driver_shift": {k: round(float(q.z[q.y == 1].mean()), 2) for k, q in U.groupby("drv")}}
assert abs(R["pooled_d"] - 0.65) < 0.01, R["pooled_d"]   # invariant: reproduces the published pooled effect
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p66_uah_within_driver.json"), "w"), indent=1); print(R)
