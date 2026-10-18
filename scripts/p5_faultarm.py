"""Malfunction arm (EngineFaultDB): does a labelled fault cause EXCESS fuel at a
matched operating point? Mirrors the VED arm: predict fuel from operating-point/demand
features (no fault label), then test whether the residual differs by fault class.
Also brake-specific fuel consumption (BSFC proxy = L/H per kW) by fault.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import f_oneway, kruskal

ROOT = os.path.join(os.path.dirname(__file__), "..")
d = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv"))
print("shape", d.shape, "| faults", sorted(d["Fault"].unique()))

# operating-point / demand features only (exclude fuel, fault, and combustion-outcome emissions)
OP = ["RPM", "MAP", "TPS", "Force", "Power", "Speed"]
TARGET = "Consumption L/100KM"
y = d[TARGET].to_numpy()

# --- expected fuel from operating point (5-fold OOF), no fault label ---
X = d[OP].to_numpy()
m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
pred = cross_val_predict(m, X, y, cv=5)
d["resid"] = y - pred
r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
print(f"\noperating-point fuel model: R2={r2:.3f}  (features={OP})")

print("\n=== EXCESS FUEL (residual, L/100km) by fault class ===")
print(d.groupby("Fault")["resid"].agg(["mean", "median", "std", "size"]).round(3).to_string())
groups = [d.loc[d.Fault == k, "resid"].to_numpy() for k in sorted(d.Fault.unique())]
F, p = f_oneway(*groups); H, ph = kruskal(*groups)
print(f"ANOVA F={F:.1f} p={p:.2e} | Kruskal H={H:.1f} p={ph:.2e}")
# effect size vs normal(0)
base = d.loc[d.Fault == 0, "resid"].mean()
for k in sorted(d.Fault.unique()):
    dv = d.loc[d.Fault == k, "resid"].mean() - base
    print(f"  fault {k}: residual excess vs normal = {dv:+.3f} L/100km")

# --- BSFC proxy: L/H per kW, matched by construction (per-work) ---
d["bsfc"] = d["Consumption L/H"] / d["Power"].clip(lower=0.1)
print("\n=== BSFC proxy (L per kWh) by fault class ===")
print(d.groupby("Fault")["bsfc"].agg(["mean", "median", "std"]).round(3).to_string())

# --- can we classify fault from signals incl. fuel+emissions? (dataset's task, for context) ---
FEAT = ["MAP","TPS","Force","Power","RPM","Consumption L/H","Consumption L/100KM",
        "Speed","CO","HC","CO2","O2","Lambda","AFR"]
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
yp = cross_val_predict(clf, d[FEAT], d["Fault"], cv=StratifiedKFold(5))
print(f"\n=== fault classification (all signals) macro-F1 = {f1_score(d.Fault, yp, average='macro'):.3f} ===")
# fuel-only classification: how diagnostic is fuel alone?
yp2 = cross_val_predict(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
                        d[["Consumption L/H","Consumption L/100KM"]], d["Fault"], cv=StratifiedKFold(5))
print(f"    fuel-only macro-F1 = {f1_score(d.Fault, yp2, average='macro'):.3f}")
print("confusion (all signals):"); print(confusion_matrix(d.Fault, yp))
