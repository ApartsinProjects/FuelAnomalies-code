"""P65: (A) risk control that is valid at the vehicle level, and (B) a kinematic score standardized within vehicle.
(A) The published threshold uses a binomial bound over variants with a Bonferroni correction over 100 margins; variants
    cluster by vehicle. Here the unit is the vehicle. For a margin threshold tau, each calibration vehicle v gives
    Z_v(tau) = (errors_v - alpha * decisions_v) / variants_v, bounded in [-alpha, 1 - alpha]; selective error <= alpha
    (ratio of vehicle-level expectations) is E[Z] <= 0. Thresholds are tested in a FIXED SEQUENCE from a pre-set start (25% coverage)
    down in margin (no multiplicity correction needed); the last threshold before the first non-rejection is kept. Two tests of
    H0: E[Z] > 0 at level delta = 0.1 on the calibration vehicles: a finite-sample betting test (Waudby-Smith and Ramdas,
    as used in Learn then Test) and a one-sided t-test on vehicle means (asymptotic, cluster-robust). Reported on one
    68/68 vehicle split (the split of p58) and over 200 random vehicle splits: coverage, selective error on held-out
    vehicles, share of splits whose held-out selective error exceeds alpha.
    Invariant stated in advance: exceedance share <= delta for the betting test.
(B) Kinematic score standardized within vehicle (as the trim is), the evaluator's driver effect being the injected raw
    shift over the median within-vehicle score SD (the analogue of S_fault). Same pass: fleet-standardized (published)
    against within-vehicle, two-way accuracy, the four-condition confusion, correlation with excess, and trip-level
    outputs on the two fixed cars with the score referenced to each car's other trips.
Output: data/processed/p65_vehicle_level_control.json"""
import os, io, glob, json, contextlib, runpy, numpy as np, pandas as pd
from scipy.stats import norm as N, t as T
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
Zv, Wk, dirn, kraw, cz0, grp, n, SF, v, su, kin, sv, VF = (g[k] for k in ["Zv", "Wk", "dirn", "kraw", "cz0", "grp", "n", "S_fault", "v", "su", "kin", "sv", "VF"])
DELTA = 0.1; R = {}
def wsr_p(x, alpha, delta=DELTA):
    """p-value for H0: E[x] >= alpha, x in [0, 1] (betting / hedged capital, as in Learn then Test)."""
    m = len(x); i = np.arange(1, m + 1); mu = (0.5 + np.cumsum(x)) / (1 + i); s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (1 + i)
    s2 = np.r_[0.25, s2[:-1]]; nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / (m * s2)), 1.0)
    K = np.cumsum(np.log(1 - nu * (x - alpha))); return float(min(1.0, np.exp(-K.max())))
def t_p(x, alpha):
    m = len(x); s = x.std(ddof=1); return 1.0 if s == 0 and x.mean() >= alpha else float(T.cdf((x.mean() - alpha) / (s / np.sqrt(m) + 1e-12), m - 1))
def calibrate(conf, err, G, cal_veh, alpha, test, rng, start=0.25):
    vid = {u: i for i, u in enumerate(cal_veh)}; m = np.isin(G, cal_veh); idx = np.array([vid[u] for u in G[m]]); c, e = conf[m], err[m]
    nv = np.bincount(idx, minlength=len(cal_veh)).astype(float); order = rng.permutation(len(cal_veh))
    grid = np.quantile(c, np.linspace(1 - start, 0.0, int(round((1 - start) * 100)) + 1)); keep = np.inf   # pre-set start at 25% coverage: at stricter margins almost nothing is decided and no test has power
    for tau in grid:                                  # fixed sequence: largest margin first
        d = c >= tau; ev = np.bincount(idx[d], weights=e[d], minlength=len(cal_veh)); dv = np.bincount(idx[d], minlength=len(cal_veh))
        y = (ev - alpha * dv) / nv + alpha            # in [0, 1]; H0: mean >= alpha
        if (wsr_p(y[order], alpha) if test == "betting" else t_p(y, alpha)) > DELTA: break
        keep = tau
    return keep
def evaluate(conf, err, G, te_veh, tau):
    m = np.isin(G, te_veh) & (conf >= tau); tot = np.isin(G, te_veh).sum(); d = int(m.sum())
    return {"decisions": d, "errors": int(err[m].sum()), "coverage": round(d / tot, 3), "selective_error": round(float(err[m].mean()), 4) if d else None,
            "test_vehicles_with_decisions": int(len(set(G[m])))}
vehs0 = np.unique(grp)
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    PERT = (Sd * kraw.std() / (Wk @ dirn)) * dirn; z = lambda x: (x - kraw.mean()) / kraw.std()
    kA, k0 = z((Zv + PERT) @ Wk), z(kraw)
    K2 = np.r_[kA, k0]; C2 = np.r_[cz0, cz0 + SF]; y2 = np.r_[np.zeros(n), np.ones(n)].astype(int); G2 = np.r_[grp, grp]
    la = N.logpdf(K2, Sd, 1) - N.logpdf(K2, 0, 1); lb = N.logpdf(C2, SF, 1) - N.logpdf(C2, 0, 1); conf = np.abs(la - lb); err = ((lb > la) != y2).astype(float)
    for alpha in (0.05, 0.10):
        out = {}
        for test in ("betting", "t"):
            rng = np.random.default_rng(0); vs = vehs0.copy(); rng.shuffle(vs); cal, te = vs[: len(vs) // 2], vs[len(vs) // 2:]
            assert not set(cal) & set(te), "vehicle leakage"
            tau = calibrate(conf, err, G2, cal, alpha, test, rng); one = {"threshold_margin": None if np.isinf(tau) else round(float(tau), 3), **evaluate(conf, err, G2, te, tau)}
            ex, cov, se = [], [], []
            for s in range(200):
                r = np.random.default_rng(100 + s); vs = vehs0.copy(); r.shuffle(vs); cal, te = vs[: len(vs) // 2], vs[len(vs) // 2:]
                e_ = evaluate(conf, err, G2, te, calibrate(conf, err, G2, cal, alpha, test, r)); cov.append(e_["coverage"])
                if e_["decisions"]: ex.append(e_["selective_error"] > alpha); se.append(e_["selective_error"])
            out[test] = {"one_split": one, "splits_200": {"exceed_rate": round(float(np.mean(ex)), 3) if ex else None, "splits_with_decisions": len(ex),
                         "coverage_median": round(float(np.median(cov)), 3), "selective_error_median": round(float(np.median(se)), 4) if se else None}}
        R[f"A_{lvl}_alpha{alpha}"] = out; print(lvl, alpha, json.dumps(out), flush=True)

# ---------------- (B) within-vehicle kinematic score ----------------
gs = pd.Series(kraw).groupby(grp); vm = gs.transform("mean").to_numpy(); vsd = gs.transform("std").to_numpy()
kw0 = (kraw - vm) / (vsd + 1e-9); sc = kw0.std(); kw0 = kw0 / sc; med_sd = float(pd.Series(kraw).groupby(grp).std().median())
ex = v["excess"].to_numpy(); kf0 = (kraw - kraw.mean()) / kraw.std()
B = {"corr_excess_fleet_standardized": round(float(np.corrcoef(ex, kf0)[0, 1]), 3), "corr_excess_within_vehicle": round(float(np.corrcoef(ex, kw0)[0, 1]), 3),
     "median_within_vehicle_score_sd_over_pooled": round(med_sd / float(kraw.std()), 3)}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    shift_raw = Sd * kraw.std()                        # the injected raw score shift (unchanged from the published injection)
    for name, k0_, kA_, Sd_eval in (("fleet_standardized", kf0, kf0 + Sd, Sd),
                                   ("within_vehicle", kw0, kw0 + shift_raw / (vsd + 1e-9) / sc, shift_raw / med_sd / sc),
                                   # driver effect of Sd within-vehicle SDs, the construct measured within driver on UAH
                                   ("within_vehicle_source_effect", kw0, kw0 + Sd / sc, Sd / sc)):
        KZ = np.r_[k0_, kA_, k0_, kA_]; CZ = np.r_[cz0, cz0, cz0 + SF, cz0 + SF]; y = np.repeat([0, 1, 2, 3], n)
        lA = N.logpdf(KZ, Sd_eval, 1) - N.logpdf(KZ, 0, 1); lB = N.logpdf(CZ, SF, 1) - N.logpdf(CZ, 0, 1)
        pub = np.where((lA < 0) & (lB < 0), 0, np.where(lB > lA, 2, 1)); two = (y == 1) | (y == 2)
        conf_ = [[round(float(np.mean(pub[y == t] == o)) * 100, 1) for o in (0, 1, 2)] for t in range(4)]
        B[f"{lvl}_{name}"] = {"driver_effect_assumed": round(float(Sd_eval), 3), "two_way_fusion": round(float(np.mean((lB > lA)[two] == (y[two] == 2))), 3),
                             "two_way_argmax": round(float(np.mean((CZ > KZ)[two] == (y[two] == 2))), 3), "confusion_pct_rows_none_driver_fault_both": conf_,
                             "three_way_accuracy_none_driver_fault": round(float(np.mean(pub[y < 3] == y[y < 3])), 3)}
        print(lvl, name, B[f"{lvl}_{name}"], flush=True)
# fixed cars, trip level, score referenced to the car's other trips (windows already are, p34)
ved_scores = sv.transform(v[VF]) @ Wk; score = lambda spd: float((su.transform(np.array([kin(spd)])) @ Wk)[0])
tau05 = json.load(open(os.path.join(ROOT, "data", "processed", "p26_seasonal_zero.json")))["item2"]["alpha_0.05"]["tau"]
def cars():
    h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0], usecols=["Vehicle_speed", "Long_Term_Fuel_Trim_Bank1", "Time(s)", "Class", "PathOrder"])
    h["sess"] = h.Class + h.PathOrder.astype(str); h = h.groupby(["sess", "Time(s)"], as_index=False)[["Vehicle_speed", "Long_Term_Fuel_Trim_Bank1"]].mean()
    yield "kia", h.rename(columns={"Time(s)": "t", "Vehicle_speed": "spd", "Long_Term_Fuel_Trim_Bank1": "ltft"})
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "sonata", "*", "All_*.csv"))):
        d = pd.read_csv(f, usecols=["car_speed", "long_fuel_bank"])
        if (d.long_fuel_bank <= -99).all(): continue
        fr.append(pd.DataFrame({"sess": f, "t": np.arange(len(d)), "spd": d.car_speed, "ltft": d.long_fuel_bank.where(d.long_fuel_bank > -99)}))
    yield "sonata", pd.concat(fr, ignore_index=True)
def decide(kz, cz, S_d, tau):
    ld = N.logpdf(kz, S_d, 1) - N.logpdf(kz, 0, 1); lf = N.logpdf(cz, 3.49 / 1.796, 1) - N.logpdf(cz, 0, 1)
    out = np.where((ld < 0) & (lf < 0), "none", np.where(lf > ld, "fault", "driver")).astype(object); out[(out != "none") & (np.abs(ld - lf) < tau)] = "abstain"; return out
for car, df in cars():
    tr = df.groupby("sess"); tl = tr.ltft.mean(); ks = pd.Series({s: score(gg.sort_values("t").spd.to_numpy()) for s, gg in tr})
    cz = np.array([(tl[s] - tl.drop(s).mean()) / 1.796 for s in tl.index])
    kz_fleet = ((ks - ved_scores.mean()) / ved_scores.std()).to_numpy(); kz_car = np.array([(ks[s] - ks.drop(s).mean()) / ks.drop(s).std() for s in ks.index])
    B[f"{car}_trips"] = {"n": int(len(ks)), "fleet_referenced_score_mean": round(float(kz_fleet.mean()), 2),
                         "fleet_referenced": {k: int((decide(kz_fleet, cz, 1.3, tau05) == k).sum()) for k in ("none", "driver", "fault", "abstain")},
                         "car_referenced": {k: int((decide(kz_car, cz, 1.3, tau05) == k).sum()) for k in ("none", "driver", "fault", "abstain")}}
    print(car, B[f"{car}_trips"], flush=True)
R["B_within_vehicle_kinematic"] = B
# ---------------- (C) vehicle-level betting test with a larger calibration share and the within-vehicle score ----------------
C = {}
for lvl, Sd in (("window", 0.65), ("trip", 1.3)):
    K2 = np.r_[kw0 + Sd / sc, kw0]; C2 = np.r_[cz0, cz0 + SF]; y2 = np.r_[np.zeros(n), np.ones(n)].astype(int); G2 = np.r_[grp, grp]
    la = N.logpdf(K2, Sd / sc, 1) - N.logpdf(K2, 0, 1); lb = N.logpdf(C2, SF, 1) - N.logpdf(C2, 0, 1); conf = np.abs(la - lb); err = ((lb > la) != y2).astype(float)
    for alpha in (0.05, 0.10):
        for frac, start in ((0.5, 0.25), (0.75, 0.25), (0.5, 0.5), (0.75, 0.5)):
            ex, cov, se = [], [], []
            for s_ in range(200):
                r = np.random.default_rng(500 + s_); vs = vehs0.copy(); r.shuffle(vs); k = int(len(vs) * frac); cal, te = vs[:k], vs[k:]
                e_ = evaluate(conf, err, G2, te, calibrate(conf, err, G2, cal, alpha, "betting", r, start)); cov.append(e_["coverage"])
                if e_["decisions"]: ex.append(e_["selective_error"] > alpha); se.append(e_["selective_error"])
            C[f"{lvl}_alpha{alpha}_cal{int(frac * 136)}_start{start}"] = {"splits_with_decisions": len(ex), "exceed_rate": round(float(np.mean(ex)), 3) if ex else None,
                "coverage_median": round(float(np.median(cov)), 3), "coverage_median_when_deciding": round(float(np.median([c for c in cov if c > 0])), 3) if ex else None,
                "selective_error_median": round(float(np.median(se)), 4) if se else None}
            print("C", lvl, alpha, frac, start, C[f"{lvl}_alpha{alpha}_cal{int(frac * 136)}_start{start}"], flush=True)
R["C_betting_within_vehicle"] = C
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p65_vehicle_level_control.json"), "w"), indent=1); print("done")
