"""Clean malfunction test: does the RICH-mixture fault (1) cause excess fuel vs NORMAL (0)
at matched operating point? Train fuel model on NORMAL only, score on RICH.
Also quantify the AFR-implied fuel penalty and BSFC gap. Faults 2 (lean) / 3 (ignition)
reported for contrast (2/3 label mapping uncertain per paper Table 4)."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = os.path.join(os.path.dirname(__file__), "..")
d = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv"))
OP = ["RPM", "MAP", "TPS", "Force", "Power", "Speed"]
T = "Consumption L/100KM"
NAMES = {0: "normal", 1: "rich", 2: "lean?", 3: "ignition?"}

norm = d[d.Fault == 0].copy()
# OOF within normal for an honest normal baseline error
oof = np.full(len(norm), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=0).split(norm):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
    m.fit(norm[OP].to_numpy()[tr], norm[T].to_numpy()[tr]); oof[te] = m.predict(norm[OP].to_numpy()[te])
norm_resid = norm[T].to_numpy() - oof
# full normal model to score faults
M = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
M.fit(norm[OP].to_numpy(), norm[T].to_numpy())

print("=== excess fuel vs NORMAL-trained expectation (L/100km), matched operating point ===")
print(f"  normal (OOF): mean {norm_resid.mean():+.3f}  sd {norm_resid.std():.3f}")
for k in [1, 2, 3]:
    sub = d[d.Fault == k]
    exc = sub[T].to_numpy() - M.predict(sub[OP].to_numpy())
    # standardized effect vs normal residual sd
    dpen = exc.mean() - norm_resid.mean()
    print(f"  fault {k} ({NAMES[k]:9s}): excess mean {exc.mean():+.3f}  median {np.median(exc):+.3f}  "
          f"=> penalty vs normal {dpen:+.3f} L/100km ({dpen/d[T].median()*100:+.1f}% of median fuel)")

print("\n=== AFR / lambda (rich = below stoich => extra fuel per unit air) ===")
for k in [0, 1, 2, 3]:
    sub = d[d.Fault == k]
    afr, lam = sub["AFR"].mean(), sub["Lambda"].mean()
    # extra fuel vs stoich 14.7 for same air: (14.7/AFR - 1)
    extra = (14.7 / afr - 1) * 100
    print(f"  fault {k} ({NAMES[k]:9s}): AFR {afr:.2f}  lambda {lam:.3f}  "
          f"=> {extra:+.1f}% fuel/air vs stoich")

print("\n=== BSFC proxy (L/H per kW) ===")
d["bsfc"] = d["Consumption L/H"] / d["Power"].clip(lower=0.1)
b = d.groupby("Fault")["bsfc"].median()
for k in [0,1,2,3]:
    print(f"  fault {k} ({NAMES[k]:9s}): BSFC median {b[k]:.3f}  "
          f"({(b[k]/b[0]-1)*100:+.1f}% vs normal)")

print("\n=== SIGNATURE separability: rich-fault excess is EMISSIONS-linked, not kinematic ===")
# correlate excess fuel with AFR/CO (fault signature) vs with Speed/RPM (operating) within rich
rich = d[d.Fault == 1].copy()
rich["exc"] = rich[T].to_numpy() - M.predict(rich[OP].to_numpy())
for c in ["AFR", "CO", "Lambda", "HC"]:
    print(f"  corr(rich excess, {c}) = {rich['exc'].corr(rich[c]):+.3f}")
