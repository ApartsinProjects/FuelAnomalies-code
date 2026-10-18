"""P60: LiU-ICE matching sensitivity and condition-clustered uncertainty, and fuel-target bias under an air-flow fault
(external review brief items 5, 6).
(1) For the leak (4, 6 mm) and air-flow sensor gain runs (WLTP), the fuel-per-measured-air shift is re-estimated with
    operating-point bins of engine speed x manifold pressure at 8x8, 12x12 (published) and 16x16, and with throttle
    position added as a third matching variable (8x8x4). Uncertainty: bootstrap over matched bins (the operating
    condition is the resampling unit, not the 20 Hz sample). Nominal run 2 vs run 1 gives the floor.
(2) Fuel-target bias: Eq. 1 fuel is computed from the MEASURED air flow, so a sensor reading g times the true flow
    biases Eq. 1 by about g - 1 while injected fuel barely moves. Reported: matched change in measured air flow and in
    injected fuel for each gain run, i.e. the Eq. 1 error the fault creates.
(3) LiU-ICE channel list (does the benchmark log STFT/LTFT?).
Output: data/processed/p60_bench_and_target_checks.json"""
import os, glob, json, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "data", "raw", "liuice"); rng = np.random.default_rng(0)
nf = sorted(glob.glob(os.path.join(D, "wltp_NF*.csv"))); ref = pd.read_csv(nf[0])
R = {"channels": list(ref.columns), "logs_fuel_trims": any("trim" in c.lower() for c in ref.columns)}
thr = ref.air_mass_flow.quantile(0.25)
runs = {"NF2": nf[1], "leak_4mm": os.path.join(D, "wltp_f_iml_4mm.csv"), "leak_6mm": os.path.join(D, "wltp_f_iml_6mm.csv")}
for gname in ("085", "090", "095", "105", "110", "115"):
    p = os.path.join(D, f"wltp_f_waf_{gname}.csv")
    if os.path.exists(p): runs[f"maf_gain_{gname}"] = p
def edges(col, k): return np.quantile(ref[col][ref.time > 130], np.linspace(0, 1, k + 1))
def binned(x, spec):
    x = x[(x.time > 130) & (x.air_mass_flow > thr)].copy(); b = np.zeros(len(x), int); mult = 1
    for col, e in spec:
        k = len(e) - 1; i = np.clip(np.searchsorted(e, x[col], side="right") - 1, 0, k - 1); b = b + i * mult; mult *= k
    x["bin"] = b; x["r"] = x.injected_fuel_mass / x.air_mass_flow
    return x.groupby("bin").agg(r=("r", "median"), n=("r", "size"), fuel=("injected_fuel_mass", "median"), air=("air_mass_flow", "median"))
specs = {"8x8": [("engine_speed", edges("engine_speed", 8)), ("intake_manifold_pressure", edges("intake_manifold_pressure", 8))],
         "12x12": [("engine_speed", edges("engine_speed", 12)), ("intake_manifold_pressure", edges("intake_manifold_pressure", 12))],
         "16x16": [("engine_speed", edges("engine_speed", 16)), ("intake_manifold_pressure", edges("intake_manifold_pressure", 16))],
         "8x8x4_throttle": [("engine_speed", edges("engine_speed", 8)), ("intake_manifold_pressure", edges("intake_manifold_pressure", 8)), ("throttle_position", edges("throttle_position", 4))]}
def wmed(vals, w):
    o = np.argsort(vals); c = np.cumsum(w[o]); return float(vals[o][np.searchsorted(c, c[-1] / 2)])
dat = {k: pd.read_csv(p) for k, p in runs.items()}
for sname, spec in specs.items():
    A = binned(ref, spec); out = {}
    for k, b in dat.items():
        J = A.join(binned(b, spec), lsuffix="_a", rsuffix="_b", how="inner"); J = J[(J.n_a >= 30) & (J.n_b >= 30)]
        w = np.minimum(J.n_a, J.n_b).to_numpy(float); dr = ((J.r_b - J.r_a) / J.r_a).to_numpy(); df_ = ((J.fuel_b - J.fuel_a) / J.fuel_a).to_numpy(); da = ((J.air_b - J.air_a) / J.air_a).to_numpy()
        bs = [wmed(dr[s], w[s]) for s in (rng.integers(0, len(dr), len(dr)) for _ in range(1000))]
        out[k] = {"bins": int(len(J)), "fuel_per_air_pct": round(wmed(dr, w) * 100, 2), "ci95_bins": [round(float(np.percentile(bs, 2.5)) * 100, 2), round(float(np.percentile(bs, 97.5)) * 100, 2)],
                  "fuel_pct": round(wmed(df_, w) * 100, 2), "measured_air_pct": round(wmed(da, w) * 100, 2)}
    R[sname] = out
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p60_bench_and_target_checks.json"), "w"), indent=1)
for s in specs: print(s, {k: (v["fuel_per_air_pct"], v["ci95_bins"], v["bins"], v["measured_air_pct"], v["fuel_pct"]) for k, v in R[s].items()})
print("trims logged:", R["logs_fuel_trims"])
