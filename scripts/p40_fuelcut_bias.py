"""P40: bias of the stoichiometric fuel target (Eq. 1: fuel = MAF / 14.7) from deceleration fuel cut-off.

During fuel cut-off air still flows but no fuel is injected, so Eq. 1 counts phantom fuel. If gentle drivers coast
more, the phantom fuel inflates gentle trips' excess and could distort the driver result.
HCRL (Kia Soul) logs the ECU fuel-cut flag and injected fuel. We build a phantom target that adds, in cut-off seconds,
the fuel a non-cut model (gradient boosting on engine load and speed, fitted on non-cut seconds) predicts there,
and recompute the window-level results under the true and the phantom target.
VED has no cut flag: closed-throttle decelerations (speed falling, absolute load below the car's 10th percentile) are
the proxy; their share of MAF fuel is removed per trip on six sampled weeks, and the fleet correlations recomputed.

PRE-REGISTERED (fixed before running):
 H1 (HCRL)  corr(excess, kinematic score) changes by less than 0.05 between the true and phantom targets, and the
            long-term-trim correlation stays equivalent to zero within 0.1 (TOST p < 0.05, session clusters) under both.
 H2 (VED)   removing the proxy phantom fuel changes corr(excess, kinematic z) and corr(excess, trim z) by less than 0.05.
 INVARIANT  The phantom share correlates negatively with the kinematic score (gentle driving coasts more), and
            removing it does not lower corr(excess, kinematic) by more than 0.01.
Output: data/processed/p40_fuelcut_bias.json
"""
import os, io, glob, json, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
OUTF = os.path.join(ROOT, "data", "processed", "p40_fuelcut_bias.json")
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
save = lambda: json.dump(R, open(OUTF, "w"), indent=2)
rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
kin, su, Wk, v, Zv, cz0 = g["kin"], g["su"], g["Wk"], g["v"], g["Zv"], g["cz0"]
score = lambda spd: float(np.ravel(su.transform(np.array([kin(spd)], dtype=float)) @ Wk)[0])

def clustered_r(x, y, cl, B=1500):
    x, y, cl = map(np.asarray, (x, y, cl)); u = np.unique(cl); idx = {k: np.flatnonzero(cl == k) for k in u}
    r0 = np.corrcoef(x, y)[0, 1]; bs = np.array([np.corrcoef(x[s], y[s])[0, 1] for s in
                                                  (np.concatenate([idx[k] for k in rng.choice(u, len(u))]) for _ in range(B))])
    se = bs.std(); p = max(N.sf((r0 + 0.1) / se), N.cdf((r0 - 0.1) / se))
    return {"r": round(float(r0), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "tost_p": float(p)}

# ---------------- HCRL ----------------
if "HCRL" not in R:
    cols = ["Fuel_consumption", "Engine_in_fuel_cut_off", "Calculated_LOAD_value", "Engine_speed", "Vehicle_speed",
            "Long_Term_Fuel_Trim_Bank1", "Calculated_road_gradient", "Time(s)", "Class", "PathOrder"]
    h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0], usecols=cols)
    h["sess"] = h.Class + h.PathOrder.astype(str)
    h = h.groupby(["sess", "Time(s)"], as_index=False).agg({c: "mean" for c in cols if c not in ("Time(s)", "Class", "PathOrder")} | {"Class": "first"})
    cut = h.Engine_in_fuel_cut_off > 0.5
    m = HistGradientBoostingRegressor(max_depth=4, random_state=0).fit(h.loc[~cut, ["Calculated_LOAD_value", "Engine_speed"]], h.loc[~cut, "Fuel_consumption"])
    h["fuel_phantom"] = h.Fuel_consumption; h.loc[cut, "fuel_phantom"] = np.maximum(m.predict(h.loc[cut, ["Calculated_LOAD_value", "Engine_speed"]]), 0)
    rows = []
    for s, gg in h.groupby("sess"):
        gg = gg.sort_values("Time(s)"); lt = gg.Long_Term_Fuel_Trim_Bank1.mean()
        for _, w in gg.groupby((gg["Time(s)"] - gg["Time(s)"].iloc[0]) // 60):
            dist = w.Vehicle_speed.sum() / 3600
            if len(w) < 45 or dist < 0.1: continue
            rows.append({"sess": s, "driver": w.Class.iloc[0], "kin": score(w.Vehicle_speed.to_numpy()), "spd": w.Vehicle_speed.mean(),
                         "grade": w.Calculated_road_gradient.mean(), "ltft_dev": w.Long_Term_Fuel_Trim_Bank1.mean() - lt,
                         "f_true": w.Fuel_consumption.sum() / dist, "f_ph": w.fuel_phantom.sum() / dist,
                         "phantom_share": (w.fuel_phantom.sum() - w.Fuel_consumption.sum()) / max(w.fuel_phantom.sum(), 1e-9)})
    W = pd.DataFrame(rows); out = {"n_windows": int(len(W)), "cut_share_seconds": round(float(cut.mean()), 4)}
    for tgt in ("f_true", "f_ph"):
        lo, hi = W[tgt].quantile([.01, .99]); X = W[(W[tgt] >= lo) & (W[tgt] <= hi)].copy()
        X["ex"] = X[tgt] - cross_val_predict(HistGradientBoostingRegressor(max_depth=4, random_state=0), X[["spd", "grade"]], X[tgt],
                                             cv=GroupKFold(X.driver.nunique()), groups=X.driver)
        out[tgt] = {"kin": clustered_r(X.ex, X.kin, X.sess), "ltft": clustered_r(X.ex, X.ltft_dev, X.sess)}
    out["phantom_share_vs_kin_r"] = round(float(np.corrcoef(W.phantom_share, W.kin)[0, 1]), 3)
    out["phantom_share_median"] = round(float(W.phantom_share.median()), 4)
    R["HCRL"] = out; save(); print("HCRL", json.dumps(out), flush=True)

# ---------------- VED (six sampled weeks) ----------------
if "VED" not in R:
    import sys; sys.path.insert(0, os.path.join(ROOT, "scripts")); from p1_features import resample_trip
    files = sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv"))); files = [files[i] for i in np.linspace(0, len(files) - 1, 6).astype(int)]
    share = []
    for f in files:
        cols = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]", "Absolute Load[%]"]
        d = pd.read_csv(f, usecols=cols, dtype={c: "float32" for c in cols[3:]})
        lowload = d.groupby("VehId")["Absolute Load[%]"].quantile(0.10)
        for (veh, trip), gg in d.groupby(["VehId", "Trip"]):
            r = resample_trip(gg)
            if r is None or r.maf.notna().mean() < 0.5: continue
            dec = (np.gradient(r.speed.to_numpy()) < -0.3) & (r.load.to_numpy() <= lowload.get(veh, np.nan))
            tot = np.nansum(r.maf.to_numpy()); share.append((veh, trip, float(np.nansum(r.maf.to_numpy()[dec]) / tot) if tot > 0 else np.nan))
        print("read", os.path.basename(f), len(share), flush=True)
    S = pd.DataFrame(share, columns=["VehId", "Trip", "phantom_share"])
    P = v.assign(kz=Zv @ Wk, czz=cz0)[["VehId", "Trip", "excess", "fuel_per_100km", "kz", "czz"]].merge(S, on=["VehId", "Trip"]).dropna()
    P["excess_adj"] = P.excess - P.phantom_share * P.fuel_per_100km
    out = {"n_trips": int(len(P)), "phantom_share_median": round(float(P.phantom_share.median()), 4),
           "phantom_share_vs_kz_r": round(float(np.corrcoef(P.phantom_share, P.kz)[0, 1]), 3)}
    for col in ("excess", "excess_adj"):
        out[col] = {"kin": clustered_r(P[col], P.kz, P.VehId), "trim": clustered_r(P[col], P.czz, P.VehId)}
    R["VED"] = out; save(); print("VED", json.dumps(out), flush=True)

h, ve = R["HCRL"], R["VED"]
R["checks"] = {"H1_kin_change_lt_0.05": abs(h["f_ph"]["kin"]["r"] - h["f_true"]["kin"]["r"]) < 0.05,
               "H1_ltft_equiv_both": h["f_true"]["ltft"]["tost_p"] < 0.05 and h["f_ph"]["ltft"]["tost_p"] < 0.05,
               "H2_VED_kin_change_lt_0.05": abs(ve["excess_adj"]["kin"]["r"] - ve["excess"]["kin"]["r"]) < 0.05,
               "H2_VED_trim_change_lt_0.05": abs(ve["excess_adj"]["trim"]["r"] - ve["excess"]["trim"]["r"]) < 0.05,
               "INV_share_neg_with_kin_HCRL": h["phantom_share_vs_kin_r"] < 0, "INV_share_neg_with_kin_VED": ve["phantom_share_vs_kz_r"] < 0,
               "INV_removal_not_lower_kin_VED": ve["excess_adj"]["kin"]["r"] >= ve["excess"]["kin"]["r"] - 0.01,
               "INV_removal_not_lower_kin_HCRL": h["f_true"]["kin"]["r"] >= h["f_ph"]["kin"]["r"] - 0.01}
save(); print(json.dumps(R["checks"], indent=1))
