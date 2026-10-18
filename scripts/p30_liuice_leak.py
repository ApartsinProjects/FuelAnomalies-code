"""P30: measured fuel-trim-like response to a real intake-manifold leak (LiU-ICE engine bench).

Data: LiU-ICE diagnosis benchmark (Jung, Frisk, Krysander; arXiv 2408.13269; Jung et al., Control Eng. Practice
164:106427, 2025). Turbocharged gasoline engine on a test bench, 20 Hz, WLTP cycle; nominal runs wltp_NF*.csv and
intake-manifold leak runs wltp_f_iml_4mm.csv / wltp_f_iml_6mm.csv (fault from ~120 s). Signals include
air_mass_flow (measured at the air filter) and injected_fuel_mass (requested by the ECU).

Quantity: fuel per measured air, r(t) = injected_fuel_mass / air_mass_flow (a trim-like ratio). A leak in the
boosted intake manifold lets metered air escape before the cylinders. With closed-loop lambda control the ECU
cuts fuel relative to METERED air (r drops); with open-loop MAF-based fueling r stays unchanged.

Comparison: runs follow the same cycle, so samples are paired by time; only t > 130 s and loaded samples
(air_mass_flow above its 25th percentile in the nominal run) are used. Statistic = median paired relative
difference of r, and of fuel, vs the first nominal run.
PRE-REGISTERED:
  INVARIANT: nominal run 2 vs nominal run 1 gives the noise floor (|median diff| expected < 1%).
  H1: both leak runs shift r by more than 3x the nominal-vs-nominal floor, with |6 mm| > |4 mm| (monotone).
  H2 (descriptive): sign of the shift (negative = closed-loop trims fuel down against metered air) and the
      cycle-level fuel change.
  KILL: leak shifts indistinguishable from the floor (then the bench gives no measurable trim response).
"""
import os, glob, json
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "data", "raw", "liuice"); OUT = os.path.join(ROOT, "data", "processed")
nf = sorted(glob.glob(os.path.join(D, "wltp_NF*.csv")))
runs = {"NF1": nf[0], "NF2": nf[1] if len(nf) > 1 else None,
        "leak_4mm": os.path.join(D, "wltp_f_iml_4mm.csv"), "leak_6mm": os.path.join(D, "wltp_f_iml_6mm.csv")}
dat = {k: pd.read_csv(v) for k, v in runs.items() if v}
ref = dat["NF1"]
R = {"files": {k: os.path.basename(v) for k, v in runs.items() if v}, "n_samples": {k: int(len(v)) for k, v in dat.items()}}
thr = ref.air_mass_flow.quantile(0.25)

def paired(name):
    a, b = ref, dat[name]
    n = min(len(a), len(b)); a = a.iloc[:n].reset_index(drop=True); b = b.iloc[:n].reset_index(drop=True)
    time_gap = float(np.max(np.abs(a.time.to_numpy() - b.time.to_numpy())))
    m = (a.time > 130) & (a.air_mass_flow > thr) & (b.air_mass_flow > 0)
    ra = a.injected_fuel_mass / a.air_mass_flow; rb = b.injected_fuel_mass / b.air_mass_flow
    rel_r = (rb[m] - ra[m]) / ra[m]
    rel_fuel = (b.injected_fuel_mass[m] - a.injected_fuel_mass[m]) / a.injected_fuel_mass[m]
    rel_air = (b.air_mass_flow[m] - a.air_mass_flow[m]) / a.air_mass_flow[m]
    spd_corr = float(np.corrcoef(a.engine_speed[m], b.engine_speed[m])[0, 1])
    return {"n_used": int(m.sum()), "max_time_misalignment_s": round(time_gap, 3), "engine_speed_corr_with_NF1": round(spd_corr, 4),
            "median_rel_diff_fuel_per_air_pct": round(float(rel_r.median() * 100), 3),
            "iqr_rel_diff_fuel_per_air_pct": [round(float(rel_r.quantile(.25) * 100), 3), round(float(rel_r.quantile(.75) * 100), 3)],
            "median_rel_diff_fuel_pct": round(float(rel_fuel.median() * 100), 3),
            "median_rel_diff_air_pct": round(float(rel_air.median() * 100), 3),
            "cycle_fuel_change_pct": round(float((b.injected_fuel_mass[m].sum() / a.injected_fuel_mass[m].sum() - 1) * 100), 3)}
def matched(name):
    """Operating-point matching (fix after the time-pairing invariant failed: runs are not time-synchronized,
    NF2 vs NF1 engine-speed corr 0.64). Bins of engine speed x intake manifold pressure; per-bin median of
    r = fuel/air; weighted median of per-bin relative differences over bins with >= 30 samples in both runs."""
    a = ref[(ref.time > 130) & (ref.air_mass_flow > thr)]; b = dat[name]; b = b[(b.time > 130) & (b.air_mass_flow > thr)]
    se = np.quantile(ref.engine_speed[ref.time > 130], np.linspace(0, 1, 13))
    pe = np.quantile(ref.intake_manifold_pressure[ref.time > 130], np.linspace(0, 1, 13))
    def binned(x):
        i = np.clip(np.searchsorted(se, x.engine_speed, side="right") - 1, 0, 11)
        j = np.clip(np.searchsorted(pe, x.intake_manifold_pressure, side="right") - 1, 0, 11)
        return x.assign(bin=i * 12 + j, r=x.injected_fuel_mass / x.air_mass_flow)
    A = binned(a).groupby("bin").agg(r=("r", "median"), n=("r", "size"), fuel=("injected_fuel_mass", "median"), air=("air_mass_flow", "median"))
    B = binned(b).groupby("bin").agg(r=("r", "median"), n=("r", "size"), fuel=("injected_fuel_mass", "median"), air=("air_mass_flow", "median"))
    J = A.join(B, lsuffix="_a", rsuffix="_b", how="inner"); J = J[(J.n_a >= 30) & (J.n_b >= 30)]
    w = np.minimum(J.n_a, J.n_b).to_numpy(float)
    def wmed(v):
        o = np.argsort(v); c = np.cumsum(w[o]); return float(v[o][np.searchsorted(c, c[-1] / 2)])
    return {"n_bins": int(len(J)), "samples_matched": int(w.sum()),
            "matched_rel_diff_fuel_per_air_pct": round(wmed(((J.r_b - J.r_a) / J.r_a).to_numpy()) * 100, 3),
            "matched_rel_diff_fuel_pct": round(wmed(((J.fuel_b - J.fuel_a) / J.fuel_a).to_numpy()) * 100, 3),
            "matched_rel_diff_air_pct": round(wmed(((J.air_b - J.air_a) / J.air_a).to_numpy()) * 100, 3)}
for k in dat:
    if k != "NF1":
        R[k] = {"time_paired_INVALID": paired(k), **matched(k)}
floor = abs(R["NF2"]["matched_rel_diff_fuel_per_air_pct"]) if "NF2" in R else float("nan")
s4, s6 = R["leak_4mm"]["matched_rel_diff_fuel_per_air_pct"], R["leak_6mm"]["matched_rel_diff_fuel_per_air_pct"]
R["VERDICT"] = {"invariant_floor_lt_1pct": bool(floor < 1.0), "floor_pct": floor,
                "H1_both_gt_3x_floor": bool(abs(s4) > 3 * floor and abs(s6) > 3 * floor),
                "H1_monotone_6mm_gt_4mm": bool(abs(s6) > abs(s4)),
                "H2_sign": "negative (fuel trimmed down vs metered air: closed loop)" if (s4 < 0 and s6 < 0) else
                           ("positive" if (s4 > 0 and s6 > 0) else "mixed")}
R["KILL"] = bool(not R["VERDICT"]["H1_both_gt_3x_floor"])
json.dump(R, open(os.path.join(OUT, "p30_liuice_leak.json"), "w"), indent=2)
print(json.dumps(R, indent=2))
