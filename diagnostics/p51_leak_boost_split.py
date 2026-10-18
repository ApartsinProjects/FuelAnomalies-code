"""P51: LiU-ICE intake-manifold leak shift split by manifold pressure regime (fresh-eye review, 2026-10-18).
Under vacuum (manifold pressure below ambient) a leak draws unmetered air in (fuel per metered air rises under closed
loop); under boost it expels metered air (the shift should fall or reverse). Same operating-point matching as p30
(engine speed x manifold pressure bins, per-bin medians, weighted median of relative differences), computed
separately for bins whose median manifold pressure is below vs above the ambient pressure.
Invariant: nominal run 2 vs run 1 must stay below 1% in both regimes."""
import os, glob, json, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "data", "raw", "liuice")
nf = sorted(glob.glob(os.path.join(D, "wltp_NF*.csv")))
dat = {"NF2": pd.read_csv(nf[1]), "leak_4mm": pd.read_csv(os.path.join(D, "wltp_f_iml_4mm.csv")), "leak_6mm": pd.read_csv(os.path.join(D, "wltp_f_iml_6mm.csv"))}
ref = pd.read_csv(nf[0]); thr = ref.air_mass_flow.quantile(0.25); amb = float(ref.ambient_pressure.median())
se = np.quantile(ref.engine_speed[ref.time > 130], np.linspace(0, 1, 13)); pe = np.quantile(ref.intake_manifold_pressure[ref.time > 130], np.linspace(0, 1, 13))
def binned(x):
    x = x[(x.time > 130) & (x.air_mass_flow > thr)]
    i = np.clip(np.searchsorted(se, x.engine_speed, side="right") - 1, 0, 11); j = np.clip(np.searchsorted(pe, x.intake_manifold_pressure, side="right") - 1, 0, 11)
    return x.assign(bin=i * 12 + j, r=x.injected_fuel_mass / x.air_mass_flow).groupby("bin").agg(r=("r", "median"), n=("r", "size"), p=("intake_manifold_pressure", "median"), fuel=("injected_fuel_mass", "median"))
A = binned(ref); R = {"ambient_pa": round(amb)}
for k, b in dat.items():
    J = A.join(binned(b), lsuffix="_a", rsuffix="_b", how="inner"); J = J[(J.n_a >= 30) & (J.n_b >= 30)]
    out = {}
    for reg, mask in (("vacuum", J.p_a < amb), ("boost", J.p_a >= amb)):
        S = J[mask]; w = np.minimum(S.n_a, S.n_b).to_numpy(float)
        if len(S) == 0: out[reg] = None; continue
        v = ((S.r_b - S.r_a) / S.r_a).to_numpy(); f = ((S.fuel_b - S.fuel_a) / S.fuel_a).to_numpy(); o = np.argsort(v); c = np.cumsum(w[o])
        of = np.argsort(f); cf = np.cumsum(w[of])
        out[reg] = {"bins": int(len(S)), "samples": int(w.sum()), "fuel_per_air_pct": round(float(v[o][np.searchsorted(c, c[-1] / 2)]) * 100, 2),
                    "fuel_pct": round(float(f[of][np.searchsorted(cf, cf[-1] / 2)]) * 100, 2)}
    R[k] = out
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p51_leak_boost_split.json"), "w"), indent=1); print(json.dumps(R, indent=1))
