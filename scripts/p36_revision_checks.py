"""P36: checks requested by the simulated EAAI review (REVIEW_EAAI_SIMULATED.md). One script, sections saved as they finish.

A  Bench fault size, one derivation: c = (14.7/AFR - 1)*100 per point; median and mean shift of each fault vs normal.
B  Driver effect size with uncertainty: Cohen's d of the frozen UAH score, aggressive vs normal, at window level (30 s),
   on 7-minute blocks (VED median trip length) and on whole recordings; 95% CI by bootstrapping drivers.
C  Equivalence tests for the "silent" trim results: vehicle-cluster bootstrap CI of r and TOST against |r| < 0.1
   (VED trip level, VED highest-trim quartile, HCRL and Sonata window level with session clusters).
D  Analytic false-fault rate of Algorithm 1 under the null (healthy car: kz, cz ~ N(0,1)), alpha 0.05 and 0.10,
   for a car whose trim spread equals the fleet prior (sd ratio 1) and the two fixed cars' measured ratios.
E  Selection: retained (injection panel) vs excluded VED trips/vehicles on fuel, trip length, trim SD.
F  Label-free comparator NOT built from the two source signatures: group-wise anomaly attribution on the target
   (Mahalanobis tail probability of the 4 kinematic features vs the |trim z| tail; smaller p-value wins), same variants.
G  Two-sided fault rule: fault variants shift the trim by +S or -S (random sign); LLR uses the symmetric mixture.
PRE-REGISTERED: (B) the 7-min-block d lies between the window d and the whole-recording d; (C) every |r| CI upper bound
reported as-is (pass = TOST p < 0.05 at margin 0.1); (D) analytic null rate <= alpha; (F) none; (G) two-sided accuracy
<= one-sided accuracy (less information), within 0.05.
Output: data/processed/p36_revision_checks.json
"""
import os, io, glob, json, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N, chi2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
OUTF = os.path.join(ROOT, "data", "processed", "p36_revision_checks.json")
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
save = lambda: json.dump(R, open(OUTF, "w"), indent=2)
rng = np.random.default_rng(0)

# ---------------- A bench fault size ----------------
e = pd.read_csv(os.path.join(RAW, "EngineFaultDB_Final.csv")); e["c"] = (14.7 / e.AFR - 1) * 100
A = {"AFR_median": e.groupby("Fault").AFR.median().round(3).to_dict(), "c_median": e.groupby("Fault").c.median().round(3).to_dict()}
for k in (1, 2, 3):
    A[f"shift_median_fault{k}"] = round(float(e.c[e.Fault == k].median() - e.c[e.Fault == 0].median()), 3)
    A[f"shift_mean_fault{k}"] = round(float(e.c[e.Fault == k].mean() - e.c[e.Fault == 0].mean()), 3)
A["rich_vs_stoich_pct"] = round(float(14.7 / e.AFR[e.Fault == 1].median() * 100 - 100), 2)
A["rich_vs_normal_fuel_per_air_pct"] = round(float((e.AFR[e.Fault == 0].median() / e.AFR[e.Fault == 1].median() - 1) * 100), 2)
R["A_bench"] = A; save(); print("A", A, flush=True)

# shared setup (frozen signature, VED injection panel)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
kin, su, Wk, Zv, cz0, n, grp, v, S_fault, kraw, ksd, dirn = (g[k] for k in ["kin", "su", "Wk", "Zv", "cz0", "n", "grp", "v", "S_fault", "kraw", "ksd", "dirn"])

# ---------------- B driver effect size with CI ----------------
if "B_driver_d" not in R:
    rows = []   # (driver, recording, block_id, label, features)
    for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
        drv = os.path.basename(f)[:2]; d = pd.read_csv(f)
        ts = d["Timestamp (seconds)"].to_numpy(); d = d.assign(_s=np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]))
        for rid, rec in d.groupby("_s"):
            rec = rec.reset_index(drop=True); t = rec["Timestamp (seconds)"] - rec["Timestamp (seconds)"].iloc[0]
            for level, width in (("window", 30), ("block7min", 420), ("recording", 10 ** 9)):
                for b, w in rec.groupby((t // width).astype(int)):
                    if len(w) < (15 if level == "window" else 120): continue
                    r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
                    if c == 1: continue
                    rows.append((level, drv, f"{drv}_{rid}", 1 if c == 2 else 0, float((su.transform(np.array([kin(w["Speed (km/h)"])])) @ Wk)[0])))
    D = pd.DataFrame(rows, columns=["level", "driver", "rec", "aggr", "score"])
    def cd_arr(sc, ag):
        a, b = sc[ag == 1], sc[ag == 0]
        return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
    B = {}
    for level, x in D.groupby("level"):
        sc = x.score.to_numpy(); ag = x.aggr.to_numpy(); drv = x.driver.to_numpy(); u = np.unique(drv)
        idx = {k: np.flatnonzero(drv == k) for k in u}; boots = []
        for _ in range(2000):
            s_ = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); boots.append(cd_arr(sc[s_], ag[s_]))
        B[level] = {"d": round(float(cd_arr(sc, ag)), 3), "ci95": [round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)],
                    "n_aggr": int((ag == 1).sum()), "n_normal": int((ag == 0).sum())}
    B["check_block_between"] = bool(min(B["window"]["d"], B["recording"]["d"]) <= B["block7min"]["d"] <= max(B["window"]["d"], B["recording"]["d"]))
    R["B_driver_d"] = B; save(); print("B", B, flush=True)

# ---------------- C equivalence tests ----------------
def clustered_r(x, y, cl, B=2000):
    x, y, cl = np.asarray(x), np.asarray(y), np.asarray(cl); u = np.unique(cl); idx = {k: np.flatnonzero(cl == k) for k in u}
    r0 = np.corrcoef(x, y)[0, 1]; bs = []
    for _ in range(B):
        s = np.concatenate([idx[k] for k in rng.choice(u, len(u))]); bs.append(np.corrcoef(x[s], y[s])[0, 1])
    bs = np.array(bs); se = bs.std()
    p_tost = max(N.sf((r0 + 0.1) / se), N.cdf((r0 - 0.1) / se))      # H0: |r| >= 0.1
    return {"r": round(float(r0), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "tost_p_margin0.1": float(p_tost), "equivalent": bool(p_tost < 0.05), "n": int(len(x)), "clusters": int(len(u))}
if "C_equivalence" not in R:
    C = {}
    ex = v.excess.to_numpy()
    C["VED_excess_vs_LTFTz"] = clustered_r(ex, cz0, grp)
    hi_veh = v.groupby("VehId").ltft1_mean.mean(); hi_veh = hi_veh[hi_veh >= hi_veh.quantile(0.75)].index
    m = v.VehId.isin(hi_veh).to_numpy()
    C["VED_highest_trim_quartile_raw_LTFT"] = clustered_r(ex[m], v.ltft1_mean.to_numpy()[m], grp[m])
    C["VED_highest_trim_quartile_LTFTz"] = clustered_r(ex[m], cz0[m], grp[m])
    for name, script, key in (("HCRL", "p29_hcrl.py", "W"), ("Sonata", "p31_sonata.py", "W")):
        with contextlib.redirect_stdout(io.StringIO()):
            gw = runpy.run_path(os.path.join(ROOT, "scripts", script))
        W = gw[key]
        C[f"{name}_excess_vs_LTFTdev"] = clustered_r(W.excess, W.ltft_dev, W.sess)
        C[f"{name}_excess_vs_kin"] = clustered_r(W.excess, W.kin, W.sess)
    R["C_equivalence"] = C; save(); print("C", json.dumps(C, indent=1), flush=True)

# ---------------- D analytic null false-fault rate ----------------
if "D_null_rate" not in R:
    M = 2_000_000; kz = rng.standard_normal(M); D = {}
    for ratio_name, ratio in (("prior_scale_1.0", 1.0), ("HCRL_0.64", 1.15 / 1.79), ("Sonata_0.26", 0.47 / 1.79)):
        cz = rng.standard_normal(M) * ratio
        for lvl, S_d in (("trip", 1.3), ("window", 0.65)):
            ld = N.logpdf(kz, S_d, 1) - N.logpdf(kz, 0, 1); lf = N.logpdf(cz, S_fault, 1) - N.logpdf(cz, 0, 1)
            _t = json.load(open(os.path.join(ROOT, "data", "processed", "p26_seasonal_zero.json")))["item2"]
            for a, tau in ((0.05, _t["alpha_0.05"]["tau"]), (0.10, _t["alpha_0.1"]["tau"])):
                fault = (lf > ld) & (lf >= 0) & (np.abs(ld - lf) >= tau)
                D[f"{ratio_name}_{lvl}_alpha{a}"] = round(float(fault.mean()), 4)
    D["all_le_alpha"] = all(val <= float(k.split("alpha")[1]) for k, val in D.items() if "alpha" in k)
    R["D_null_rate"] = D; save(); print("D", D, flush=True)

# ---------------- E selection ----------------
if "E_selection" not in R:
    allv = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
    keep = allv.set_index(["VehId", "Trip"]).index.isin(v.set_index(["VehId", "Trip"]).index)
    def summ(x):
        return {"trips": int(len(x)), "vehicles": int(x.VehId.nunique()), "fuel_L100_median": round(float(x.fuel_per_100km.median()), 2),
                "dist_km_median": round(float(x.dist_km.median()), 2), "dur_min_median": round(float(x.dur_min.median()), 1),
                "speed_mean_median": round(float(x.speed_mean.median()), 1)}
    R["E_selection"] = {"retained": summ(allv[keep]), "excluded": summ(allv[~keep])}; save(); print("E", R["E_selection"], flush=True)

# ---------------- F + G on the same injected variants ----------------
if "FG_methods" not in R:
    out = {}
    Sig = np.cov(Zv, rowvar=False); Si = np.linalg.inv(Sig); mu = Zv.mean(0)
    for lvl, S_d in (("window", 0.65), ("trip", 1.3)):
        PERT = (S_d * ksd / (Wk @ dirn)) * dirn
        Xk = np.r_[Zv + PERT, Zv]; CZ = np.r_[cz0, cz0 + S_fault]; y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        m2 = np.einsum("ij,jk,ik->i", Xk - mu, Si, Xk - mu); p_k = chi2.sf(m2, 4)
        czz = (CZ - cz0.mean()) / cz0.std(); p_c = 2 * N.sf(np.abs(czz))
        acc_F = float(((p_c < p_k).astype(int) == y).mean())
        # G two-sided: fault sign random
        sign = rng.choice([-1, 1], n); CZ2 = np.r_[cz0, cz0 + sign * S_fault]; cz2 = (CZ2 - cz0.mean()) / cz0.std()
        kd = ((Zv + PERT) @ Wk - kraw.mean()) / kraw.std(); KZ = np.r_[kd, (kraw - kraw.mean()) / kraw.std()]
        ld = N.logpdf(KZ, S_d, 1) - N.logpdf(KZ, 0, 1)
        lf2 = np.log(0.5 * N.pdf(cz2, S_fault, 1) + 0.5 * N.pdf(cz2, -S_fault, 1)) - N.logpdf(cz2, 0, 1)
        acc_G = float(((lf2 > ld).astype(int) == y).mean())
        lf1 = N.logpdf(czz, S_fault, 1) - N.logpdf(czz, 0, 1); acc_1 = float(((lf1 > ld).astype(int) == y).mean())
        out[lvl] = {"F_groupwise_anomaly_acc": round(acc_F, 3), "F_driver_recall": round(float((p_c >= p_k)[:n].mean()), 3),
                    "F_fault_recall": round(float((p_c < p_k)[n:].mean()), 3),
                    "G_two_sided_acc": round(acc_G, 3), "one_sided_acc_same_code": round(acc_1, 3)}
    out["G_check"] = all(out[l]["G_two_sided_acc"] <= out[l]["one_sided_acc_same_code"] + 1e-9 and out[l]["one_sided_acc_same_code"] - out[l]["G_two_sided_acc"] <= 0.05 for l in ("window", "trip"))
    R["FG_methods"] = out; save(); print("FG", out, flush=True)
print("done")
