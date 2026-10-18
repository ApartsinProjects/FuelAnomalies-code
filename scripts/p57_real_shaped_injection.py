"""P57: injections shaped by real data, high-load coupling, and the two-reference trim rule (IJPHM editor items).
(1) REAL DRIVER EFFECTS: on the Kia (10 drivers) and Sonata (4 drivers) the frozen kinematic score of each driver's
    windows gives a between-driver effect size d_ij = (mean_i - mean_j) / pooled SD for every ordered pair with
    d_ij > 0. Each VED driver variant draws one d_ij from this empirical set (the attributor still assumes d = 0.65).
(2) OPEL-SHAPED FAULT: the Opel trim is largest at idle (+29.7%) and smallest at high air flow (+3.9%), so the fault
    shift of trip i is 3.49 pp x w_i / mean(w), with w_i = 0.13 + idle share of trip i (0.13 = 3.9 / 29.7).
(3) both together. Fusion, argmax and target-only accuracy, window and trip operating points of the attributor.
(4) HIGH-LOAD COUPLING: within-vehicle correlation between the kinematic score and the trim z, and between excess and
    trim z, in load-mean terciles (power enrichment happens at high load).
(5) TWO-REFERENCE RULE: share of VED vehicles whose mean trim exceeds a fleet threshold (vehicle-mean z against the
    fleet), and whether the Opel (mean trim +16.0%) is flagged at thresholds that flag 1%, 2% and 5% of VED vehicles.
Output: data/processed/p57_real_shaped_injection.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N, chi2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, Zv, Wk, dirn, kraw, cz0, grp, n = (g[k] for k in ["v", "Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n"])
SF = 3.49 / float(np.median(v.groupby("VehId")["ltft1_mean"].std())); MED = float(np.median(v.groupby("VehId")["ltft1_mean"].std()))
dpairs = {}
for name, script in (("Kia", "p29_hcrl.py"), ("Sonata", "p31_sonata.py")):
    with contextlib.redirect_stdout(io.StringIO()):
        W = runpy.run_path(os.path.join(ROOT, "scripts", script))["W"]
    st = W.groupby("driver").kin.agg(["mean", "var", "size"]); ds = []
    for i in st.index:
        for j in st.index:
            if i != j and st.loc[i, "mean"] > st.loc[j, "mean"]:
                ds.append((st.loc[i, "mean"] - st.loc[j, "mean"]) / np.sqrt((st.loc[i, "var"] + st.loc[j, "var"]) / 2))
    dpairs[name] = np.array(ds)
D_REAL = np.r_[dpairs["Kia"], dpairs["Sonata"]]
R = {"real_driver_d": {"pairs": int(len(D_REAL)), "median": round(float(np.median(D_REAL)), 3), "q10_q90": [round(float(np.quantile(D_REAL, .1)), 3), round(float(np.quantile(D_REAL, .9)), 3)],
                       "Kia_median": round(float(np.median(dpairs["Kia"])), 3), "Sonata_median": round(float(np.median(dpairs["Sonata"])), 3)}}
z = lambda x: (x - kraw.mean()) / kraw.std(); unit = dirn / np.linalg.norm(dirn)
idle = v.idle_frac.to_numpy(); w = 0.13 + idle; fshape = 3.49 * w / w.mean() / MED
mu = Zv.mean(0); Si = np.linalg.inv(np.cov(Zv, rowvar=False))
def evaluate(drv, flt, Sd):
    dsz = rng.choice(D_REAL, n) if drv == "real" else np.full(n, Sd)
    a = dsz * kraw.std() / (Wk @ unit); Xd = Zv + a[:, None] * unit[None, :]
    sf = fshape if flt == "opel" else np.full(n, SF)
    KZ = np.r_[z(Xd @ Wk), z(kraw)]; CZ = np.r_[cz0, cz0 + sf]; y = np.r_[np.zeros(n), np.ones(n)]
    ld = N.logpdf(KZ, Sd, 1) - N.logpdf(KZ, 0, 1); lf = N.logpdf(CZ, SF, 1) - N.logpdf(CZ, 0, 1)
    Xk = np.r_[Xd, Zv]; p_k = chi2.sf(np.einsum("ij,jk,ik->i", Xk - mu, Si, Xk - mu), 4); p_c = 2 * N.sf(np.abs((CZ - cz0.mean()) / cz0.std()))
    return {"fusion": round(float(((lf > ld) == y).mean()), 3), "argmax": round(float(((CZ > KZ) == y).mean()), 3), "target_only": round(float(((p_c < p_k) == y).mean()), 3)}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    R[f"{lvl}_real_driver"] = evaluate("real", "fixed", Sd)
    R[f"{lvl}_opel_fault"] = evaluate("fixed", "opel", Sd)
    R[f"{lvl}_real_driver_opel_fault"] = evaluate("real", "opel", Sd)
R["opel_fault_shift_pp_q10_q90"] = [round(float(np.quantile(fshape * MED, .1)), 2), round(float(np.quantile(fshape * MED, .9)), 2)]
# (4) high-load coupling
d = pd.DataFrame({"veh": grp, "k": kraw, "c": cz0, "ex": v.excess.to_numpy(), "load": v.load_mean.to_numpy()}).dropna()
d["t"] = pd.qcut(d.load, 3, labels=["low", "mid", "high"])
def wc(x, a, b):
    xa = x[a] - x.groupby("veh")[a].transform("mean"); xb = x[b] - x.groupby("veh")[b].transform("mean"); return round(float(np.corrcoef(xa, xb)[0, 1]), 3)
R["coupling_by_load_tercile"] = {t: {"kin_vs_trim": wc(d[d.t == t], "k", "c"), "excess_vs_trim": wc(d[d.t == t], "ex", "c"), "trips": int((d.t == t).sum())} for t in ["low", "mid", "high"]}
R["coupling_all"] = {"kin_vs_trim": wc(d, "k", "c"), "excess_vs_trim": wc(d, "ex", "c")}
# (5) two-reference rule
vm = g["v"].groupby("VehId").ltft1_mean.mean(); fz = (vm - vm.mean()) / vm.std(); opel_z = (16.0 - vm.mean()) / vm.std()
R["fleet_reference"] = {"vehicles": int(len(vm)), "fleet_mean_trim": round(float(vm.mean()), 2), "fleet_sd_vehicle_means": round(float(vm.std()), 2), "opel_fleet_z": round(float(opel_z), 2),
                        "opel_percentile": round(float((vm < 16.0).mean() * 100), 1)}
for q in (0.99, 0.98, 0.95):
    thr = float(fz.quantile(q)); R["fleet_reference"][f"flag_top_{round((1 - q) * 100)}pct"] = {"z_threshold": round(thr, 2), "opel_flagged": bool(opel_z > thr)}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p57_real_shaped_injection.json"), "w"), indent=1); print(json.dumps(R, indent=1))
