"""P39: derive the trip-level driver effect size from the well-measured window effect and the within-trip
correlation of the frozen kinematic score, instead of the 10-block UAH estimate.

If aggression shifts every 30-s window of a trip by the same amount while the window score has within-trip
intraclass correlation rho, the trip-mean effect over n windows is d_trip = d_window / sqrt(rho + (1 - rho) / n).
rho is label-free and is estimated on VED (sampled weeks), HCRL and Sonata; d_window = 0.654 [0.407, 0.836] (UAH, p36).
Both uncertainties are propagated by bootstrap (trips for rho; d_window drawn from a normal matched to its interval).

PRE-REGISTERED (fixed before running):
 INVARIANT  Applied to UAH itself (rho from UAH windows within recordings, n = windows per 7-min block and per
            recording), the formula's prediction lies inside the measured intervals (7-min block d 1.31 [-0.05, 2.30];
            recording d 1.15 [0.05, 2.30]).
 H1  The VED-derived d_trip for the median VED trip has a 95% interval whose lower bound exceeds the window value 0.65.
 Reported: d_trip with interval for VED, HCRL, Sonata at n = 14 (7 min), and the closed-form argmax/fusion accuracy.
Output: data/processed/p39_trip_effect_size.json
"""
import os, io, glob, json, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
OUTF = os.path.join(ROOT, "data", "processed", "p39_trip_effect_size.json")
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
save = lambda: json.dump(R, open(OUTF, "w"), indent=2)
rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
kin, su, Wk = g["kin"], g["su"], g["Wk"]
score = lambda spd: float(np.ravel(su.transform(np.array([kin(spd)], dtype=float)) @ Wk)[0])
D_W, D_W_SE = 0.654, (0.836 - 0.407) / 3.92

def windows(df, sess_col, t_col, spd_col, width=30, minlen=15):
    rows = []
    for s, gg in df.groupby(sess_col):
        gg = gg.sort_values(t_col); w = ((gg[t_col] - gg[t_col].iloc[0]) // width).astype(int)
        for _, x in gg.groupby(w):
            if len(x) >= minlen and x[spd_col].notna().sum() >= minlen: rows.append((s, score(x[spd_col].to_numpy())))
    return pd.DataFrame(rows, columns=["trip", "s"])

def trip_stats(W, min_w=6):
    W = W.groupby("trip").filter(lambda x: len(x) >= min_w)
    g_ = W.groupby("trip").s
    return g_.size().to_numpy(float), g_.mean().to_numpy(), g_.apply(lambda x: ((x - x.mean()) ** 2).sum()).to_numpy()

def icc_from(k, m, ssw):
    T, Ntot = len(k), k.sum(); n0 = (Ntot - (k ** 2).sum() / Ntot) / (T - 1)
    gm = (k * m).sum() / Ntot; msb = (k * (m - gm) ** 2).sum() / (T - 1); msw = ssw.sum() / (Ntot - T)
    return float((msb - msw) / (msb + (n0 - 1) * msw))

def derive(name, W, n_list=(14,)):
    k, m, ssw = trip_stats(W); rho = icc_from(k, m, ssw); T = len(k)
    boots = []
    for _ in range(1000):
        b = rng.integers(0, T, T); boots.append(icc_from(k[b], m[b], ssw[b]))
    boots = np.array(boots)
    out = {"rho": round(rho, 3), "rho_ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
           "n_trips": int(T), "n_windows": int(k.sum())}
    acc = lambda d: 0.5 * (N.cdf(d / np.sqrt(2)) + N.cdf(1.95 / np.sqrt(2)))
    for n in n_list:
        dws = rng.normal(D_W, D_W_SE, len(boots)); dt = dws / np.sqrt(boots + (1 - boots) / n)
        d0 = D_W / np.sqrt(rho + (1 - rho) / n)
        out[f"n{n}"] = {"d_trip": round(float(d0), 3), "ci95": [round(float(np.percentile(dt, 2.5)), 3), round(float(np.percentile(dt, 97.5)), 3)],
                        "argmax_acc_closed_form": round(float(acc(d0)), 3),
                        "fusion_acc_closed_form": round(float(N.cdf(np.sqrt(d0 ** 2 + 1.95 ** 2) / 2)), 3)}
    R[name] = out; save(); print(name, out, flush=True)

# UAH (invariant): windows within recordings
if "UAH" not in R:
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
        d = pd.read_csv(f, usecols=["Timestamp (seconds)", "Speed (km/h)"]); ts = d["Timestamp (seconds)"].to_numpy()
        d["rec"] = os.path.basename(f) + "_" + pd.Series(np.concatenate([[0], np.cumsum(np.diff(ts) < 0)])).astype(str); fr.append(d)
    U = pd.concat(fr); WU = windows(U, "rec", "Timestamp (seconds)", "Speed (km/h)")
    n_rec = int(WU.groupby("trip").size().median())
    derive("UAH", WU, n_list=(14, n_rec)); R["UAH"]["n_rec_median_windows"] = n_rec; save()
# HCRL, Sonata
if "HCRL" not in R:
    h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0], usecols=["Vehicle_speed", "Time(s)", "Class", "PathOrder"])
    h["sess"] = h.Class + h.PathOrder.astype(str); h = h.groupby(["sess", "Time(s)"], as_index=False).Vehicle_speed.mean()
    derive("HCRL", windows(h, "sess", "Time(s)", "Vehicle_speed"))
if "Sonata" not in R:
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "sonata", "*", "All_*.csv"))):
        d = pd.read_csv(f, usecols=["car_speed", "long_fuel_bank"])
        if (d.long_fuel_bank <= -99).all(): continue
        fr.append(pd.DataFrame({"trip": f, "t": np.arange(len(d)), "spd": d.car_speed}))
    derive("Sonata", windows(pd.concat(fr), "trip", "t", "spd"))
# VED: four weekly files spread over the year
if "VED" not in R:
    import sys; sys.path.insert(0, os.path.join(ROOT, "scripts")); from p1_features import resample_trip
    fr = []
    for f in [sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv")))[i] for i in (0, 13, 26, 39)]:
        cols = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]"]
        d = pd.read_csv(f, usecols=cols, dtype={"Vehicle Speed[km/h]": "float32"})
        for (veh, trip), gg in d.groupby(["VehId", "Trip"]):
            r = resample_trip(gg.assign(**{"MAF[g/sec]": np.nan}))
            if r is None or len(r) < 180: continue
            fr.append(pd.DataFrame({"trip": f"{veh}_{trip}_{os.path.basename(f)}", "t": r.t, "spd": r.speed}))
        print("read", os.path.basename(f), len(fr), flush=True)
    WV = windows(pd.concat(fr, ignore_index=True), "trip", "t", "spd")
    derive("VED", WV); R["VED"]["median_windows_per_trip"] = int(WV.groupby("trip").size().median()); save()

u = R["UAH"]; nr = u["n_rec_median_windows"]
R["checks"] = {"invariant_block_inside": -0.05 <= u["n14"]["d_trip"] <= 2.30, "invariant_recording_inside": 0.05 <= u[f"n{nr}"]["d_trip"] <= 2.30,
               "H1_VED_lower_gt_0.65": R["VED"]["n14"]["ci95"][0] > 0.65}
save(); print(json.dumps(R["checks"]))
