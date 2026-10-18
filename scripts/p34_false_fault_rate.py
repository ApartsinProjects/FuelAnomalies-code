"""P34: real-data false-fault rate of Algorithm 1 on two healthy fixed cars (HCRL Kia Soul, Sonata).

Both cars are healthy and fixed across all trips, so every "fault" decision is a false alarm. Algorithm 1 as in
the paper: source-calibrated LLRs l_d = log N(kz; S_d,1) - log N(kz; 0,1), l_f = log N(cz; S_f,1) - log N(cz; 0,1);
"none" if both < 0, else argmax; decision issued only if |l_d - l_f| >= tau_alpha (tau 2.598 at alpha 0.05,
1.136 at alpha 0.10, from p19); S_f = 1.66.
Deployment-faithful inputs (no information from the trip being scored):
 cz = (trip or window LTFT mean - mean LTFT of the car's OTHER trips) / 1.79 pp (VED median within-vehicle LTFT SD,
      the fleet prior a new vehicle starts from).
 kz = frozen UAH kinematic score standardized against the VED fleet trip distribution (trip level, S_d = 1.3) or
      against the car's other trips' windows (window level, 60 s, S_d = 0.65).
PRE-REGISTERED: issued-fault rate <= alpha on both cars, at both levels, for alpha 0.05 and 0.10.
Output: data/processed/p34_false_fault.json
"""
import os, io, glob, json, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "p33_attribution_baselines.py"))
kin, su, Wk, kraw_ved = g["kin"], g["su"], g["Wk"], None
v = g["v"]; VF = g["VF"]; sv = g["sv"]
# VED fleet trip-score reference: same frozen score on VED trip features (as the deployed axis)
ved_scores = sv.transform(v[VF]) @ Wk
_t = json.load(open(os.path.join(ROOT, "data", "processed", "p26_seasonal_zero.json")))["item2"]
S_F, SD_PRIOR = 3.49 / 1.79, 1.79   # LiU-ICE anchor (p37) over the fleet median within-vehicle LTFT SD
TAU = {0.05: _t["alpha_0.05"]["tau"], 0.10: _t["alpha_0.1"]["tau"]}
score_src = lambda spd: float((su.transform(np.array([kin(spd)])) @ Wk)[0])

def load_hcrl():
    h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0], usecols=["Vehicle_speed", "Long_Term_Fuel_Trim_Bank1", "Time(s)", "Class", "PathOrder"])
    h["sess"] = h.Class + h.PathOrder.astype(str)
    h = h.groupby(["sess", "Time(s)"], as_index=False)[["Vehicle_speed", "Long_Term_Fuel_Trim_Bank1"]].mean()
    return h.rename(columns={"Time(s)": "t", "Vehicle_speed": "spd", "Long_Term_Fuel_Trim_Bank1": "ltft"})
def load_sonata():
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "sonata", "*", "All_*.csv"))):
        d = pd.read_csv(f, usecols=["car_speed", "long_fuel_bank"])
        if (d.long_fuel_bank <= -99).all(): continue
        fr.append(pd.DataFrame({"sess": f, "t": np.arange(len(d)), "spd": d.car_speed, "ltft": d.long_fuel_bank.where(d.long_fuel_bank > -99)}))
    return pd.concat(fr, ignore_index=True)

def decide(kz, cz, S_d, tau):
    ld = N.logpdf(kz, S_d, 1) - N.logpdf(kz, 0, 1); lf = N.logpdf(cz, S_F, 1) - N.logpdf(cz, 0, 1)
    out = np.where((ld < 0) & (lf < 0), "none", np.where(lf > ld, "fault", "driver")).astype(object)
    out[(out != "none") & (np.abs(ld - lf) < tau)] = "abstain"
    return out

R = {}
for car, df in [("hcrl_kia", load_hcrl()), ("sonata", load_sonata())]:
    trips = df.groupby("sess"); tl = trips.ltft.mean()
    # trip level
    kz_t = np.array([(score_src(gg.sort_values("t").spd.to_numpy()) - ved_scores.mean()) / ved_scores.std() for _, gg in trips])
    cz_t = np.array([(tl[s] - tl.drop(s).mean()) / SD_PRIOR for s in tl.index])
    # window level
    rows = []
    for s, gg in trips:
        gg = gg.sort_values("t")
        for _, w in gg.groupby((gg.t - gg.t.iloc[0]) // 60):
            if len(w) < 45 or w.spd.sum() / 3600 < 0.1 or w.ltft.isna().all(): continue
            rows.append((s, score_src(w.spd.to_numpy()), w.ltft.mean()))
    W = pd.DataFrame(rows, columns=["sess", "k", "ltft"])
    kz_w = np.empty(len(W)); cz_w = np.empty(len(W))
    for s in W.sess.unique():
        m = W.sess == s; o = W[~m]
        kz_w[m] = (W.k[m] - o.k.mean()) / o.k.std(); cz_w[m] = (W.ltft[m] - tl.drop(s).mean()) / SD_PRIOR
    blk = {"n_trips": int(len(tl)), "n_windows": int(len(W)), "cz_trip_range": [round(float(cz_t.min()), 2), round(float(cz_t.max()), 2)]}
    for a, tau in TAU.items():
        for lvl, kz, cz, S_d in [("trip", kz_t, cz_t, 1.3), ("window", kz_w, cz_w, 0.65)]:
            d = decide(kz, cz, S_d, tau); sh = {k: round(float((d == k).mean()), 3) for k in ["none", "driver", "fault", "abstain"]}
            blk[f"{lvl}_alpha{a}"] = {**sh, "n_fault": int((d == "fault").sum()), "pass": sh["fault"] <= a}
    R[car] = blk; print(car, json.dumps(blk), flush=True)
R["all_pass"] = all(R[c][k]["pass"] for c in ("hcrl_kia", "sonata") for k in R[c] if isinstance(R[c][k], dict))
print("all_pass", R["all_pass"])
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p34_false_fault.json"), "w"), indent=2)
