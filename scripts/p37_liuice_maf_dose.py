"""P37: dosed, two-sided fueling fault on a real engine: LiU-ICE air-mass-flow sensor gain faults.

An air-flow sensor that reads g times the true flow makes a closed-loop controller restore lambda by changing fuel
relative to the MEASURED air, so fuel per measured air should shift by (1/g - 1) while the fuel actually injected is
unchanged. Faults: wltp_f_waf_{085,090,095,105,110,115} (g = 0.85 ... 1.15) and the same on the FTP-75 city cycle;
pressure-sensor faults (pim, pic) serve as non-fueling controls. Method as p30: operating-point matching against the
nominal run (bins of engine speed x intake manifold pressure, weighted median of per-bin relative differences of
r = injected_fuel_mass / air_mass_flow, and of injected fuel), samples after fault onset only.

PRE-REGISTERED (fixed before running):
 INVARIANT  WLTP nominal run 2 vs run 1: |fuel-per-air shift| < 1%.
 H1  On each cycle, the fuel-per-measured-air shift has the sign of (1/g - 1) at all six gains, and Spearman(g, shift) = -1.
 H2  For g in {0.90, 0.95, 1.05, 1.10}, |shift - (1/g - 1)| < 3 percentage points (the extremes 0.85 / 1.15 are reported).
 H3  |injected-fuel change| < 3% at every gain on both cycles (closed loop compensates; no fuel excess).
 Descriptive: pressure-sensor faults (pim, pic) shifts, for contrast.
Output: data/processed/p37_liuice_maf_dose.json
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data", "raw", "liuice"); OUTF = os.path.join(ROOT, "data", "processed", "p37_liuice_maf_dose.json")
GAINS = ["085", "090", "095", "105", "110", "115"]

def matched(ref, b, t0):
    thr = ref.air_mass_flow.quantile(0.25)
    a = ref[(ref.time > t0) & (ref.air_mass_flow > thr)]; b = b[(b.time > t0) & (b.air_mass_flow > thr)]
    se = np.quantile(ref.engine_speed[ref.time > t0], np.linspace(0, 1, 13))
    pe = np.quantile(ref.intake_manifold_pressure[ref.time > t0], np.linspace(0, 1, 13))
    def binned(x):
        i = np.clip(np.searchsorted(se, x.engine_speed, side="right") - 1, 0, 11)
        j = np.clip(np.searchsorted(pe, x.intake_manifold_pressure, side="right") - 1, 0, 11)
        return x.assign(bin=i * 12 + j, r=x.injected_fuel_mass / x.air_mass_flow)
    agg = dict(r=("r", "median"), n=("r", "size"), fuel=("injected_fuel_mass", "median"))
    J = binned(a).groupby("bin").agg(**agg).join(binned(b).groupby("bin").agg(**agg), lsuffix="_a", rsuffix="_b", how="inner")
    J = J[(J.n_a >= 30) & (J.n_b >= 30)]; w = np.minimum(J.n_a, J.n_b).to_numpy(float)
    def wmed(v):
        o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2)])
    return {"n_bins": int(len(J)), "samples": int(w.sum()),
            "fuel_per_air_pct": round(wmed(((J.r_b - J.r_a) / J.r_a).to_numpy()) * 100, 2),
            "fuel_pct": round(wmed(((J.fuel_b - J.fuel_a) / J.fuel_a).to_numpy()) * 100, 2)}

R = {}
ld = lambda f: pd.read_csv(os.path.join(D, f))
nf_w = ld("wltp_NF.csv")
R["invariant_NF2_vs_NF1"] = matched(nf_w, ld("wltp_NF_2.csv"), 135)
for cyc, nf_file, t0 in (("wltp", "wltp_NF.csv", 135), ("ftp75", "ftp75city2_NF.csv", 135)):
    ref = ld(nf_file); out = {}
    for kind in ("waf", "pim", "pic"):
        for g in GAINS:
            f = f"{cyc}{'city2' if cyc == 'ftp75' else ''}_f_{kind}_{g}.csv"
            if not os.path.exists(os.path.join(D, f)): continue
            res = matched(ref, ld(f), t0); gv = int(g) / 100
            if kind == "waf": res["expected_fuel_per_air_pct"] = round((1 / gv - 1) * 100, 2)
            out[f"{kind}_{g}"] = res
    R[cyc] = out; json.dump(R, open(OUTF, "w"), indent=2); print(cyc, json.dumps(out), flush=True)

chk = {"invariant_floor_lt_1pct": abs(R["invariant_NF2_vs_NF1"]["fuel_per_air_pct"]) < 1.0}
for cyc in ("wltp", "ftp75"):
    w = {g: R[cyc].get(f"waf_{g}") for g in GAINS}
    if any(v is None for v in w.values()):
        chk[f"{cyc}_complete"] = False; continue
    gs = np.array([int(g) / 100 for g in GAINS]); sh = np.array([w[g]["fuel_per_air_pct"] for g in GAINS])
    chk[f"{cyc}_H1_signs"] = bool(all(np.sign(s) == np.sign(1 / x - 1) for s, x in zip(sh, gs)))
    chk[f"{cyc}_H1_spearman"] = round(float(spearmanr(gs, sh).statistic), 3)
    chk[f"{cyc}_H2_core_within_3pp"] = bool(all(abs(w[g]["fuel_per_air_pct"] - w[g]["expected_fuel_per_air_pct"]) < 3 for g in ("090", "095", "105", "110")))
    chk[f"{cyc}_H3_fuel_within_3pct"] = bool(all(abs(w[g]["fuel_pct"]) < 3 for g in GAINS))
    chk[f"{cyc}_slope_obs_vs_expected"] = round(float(np.polyfit([w[g]["expected_fuel_per_air_pct"] for g in GAINS], sh, 1)[0]), 3)
R["checks"] = chk; json.dump(R, open(OUTF, "w"), indent=2); print(json.dumps(chk, indent=1))
