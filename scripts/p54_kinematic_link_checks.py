"""P54: is the excess-kinematic correlation mechanical? (IJPHM review R1-R3)
The contrast's excess is fuel minus an out-of-fold model of CONDITIONS only (attrib_setup ENV: temperature, AC and
heater power, distance, duration, weight, mean speed, highway share); the four signature features are not inputs.
Checks on the 11,304-trip panel: (1) within-vehicle (vehicle-demeaned) and between-vehicle correlations; (2) placebo:
score permuted within vehicle; (3) calibration slope of fuel on the out-of-fold prediction (1 = no shrinkage);
(4) placebo predictors that are already conditions (weight, temperature) must correlate ~0 with excess.
Output: data/processed/p54_kinematic_link_checks.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, kraw, grp, ENV, VF = g["v"], g["kraw"], g["grp"], g["ENV"], g["VF"]
d = pd.DataFrame({"veh": grp, "ex": v.excess.to_numpy(), "k": kraw, "fuel": v.fuel_per_100km.to_numpy(),
                  "w": v.weight_lb.to_numpy(), "t": v.oat_mean.to_numpy()})
d["pred"] = d.fuel - d.ex
def boot(f, B=1000):
    u = d.veh.unique(); idx = {k: np.flatnonzero(d.veh.to_numpy() == k) for k in u}; out = []
    for _ in range(B):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); out.append(f(d.iloc[s]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]
def within(x):
    a = x.ex - x.groupby("veh").ex.transform("mean"); b = x.k - x.groupby("veh").k.transform("mean"); return np.corrcoef(a, b)[0, 1]
def between(x):
    m = x.groupby("veh")[["ex", "k"]].mean(); return np.corrcoef(m.ex, m.k)[0, 1]
R = {"signature_features_in_excess_model": sorted(set(VF) & set(ENV)), "excess_model_inputs": ENV,
     "pooled_r": round(float(np.corrcoef(d.ex, d.k)[0, 1]), 3)}
R["within_vehicle_r"] = {"r": round(float(within(d)), 3), "ci95": boot(within)}
R["between_vehicle_r"] = {"r": round(float(between(d)), 3), "vehicles": int(d.veh.nunique())}
perm = d.copy(); perm["k"] = perm.groupby("veh").k.transform(lambda s: rng.permutation(s.to_numpy()))
R["placebo_within_permuted_r"] = round(float(within(perm)), 3)
R["calibration_slope_fuel_on_pred"] = round(float(np.polyfit(d.pred, d.fuel, 1)[0]), 3)
R["placebo_condition_r"] = {"weight": round(float(d[["ex", "w"]].dropna().corr().iloc[0, 1]), 3), "temperature": round(float(d[["ex", "t"]].dropna().corr().iloc[0, 1]), 3)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p54_kinematic_link_checks.json"), "w"), indent=1); print(json.dumps(R, indent=1))
