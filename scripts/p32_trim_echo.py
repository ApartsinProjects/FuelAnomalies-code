"""P32: why the combustion axis is the LONG-term trim only (fast-trim transient echo), tested on three datasets.

Background: P31 (Sonata, discovery set) showed the short-term trim (STFT) dips a few seconds after each fuel burst,
so window-mean STFT carries driver signal. The attribution pipeline (p8/p10/p17/p19/p23/p24) already builds the
combustion axis from within-vehicle standardized LTFT only; this script tests the reason on two independent sets.

PRE-REGISTERED (fixed before running on HCRL and VED; Sonata is reported as the discovery set):
 E1 fast-trim echo: mean STFT over lags 2-6 s after a fuel burst (a moving second in the top 10% of that dataset's
    moving fuel/air flow) minus the dataset's mean moving STFT < -1.0 pp, on HCRL and on VED.
 E2 long-term trim does not echo: the same lag contrast for LTFT has |value| < 0.25 pp, on HCRL and on VED.
 E3 VED axis silence: trip-level |corr(context-residual excess, within-vehicle standardized LTFT)| < 0.1.
 I1 placebo invariant: the same contrast at an equal number of random moving seconds has |STFT value| < 0.3 pp.
Output: data/processed/p32_trim_echo.json (one block per dataset, appended as each finishes).
"""
import os, glob, json, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw"); OUTF = os.path.join(ROOT, "data", "processed", "p32_trim_echo.json")
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
def save(): json.dump(R, open(OUTF, "w"), indent=2)
rng = np.random.default_rng(0)

def echo(df, fuel, name):
    """df: 1 Hz rows with sess, spd, fuel, stft, ltft (time-ordered within sess)."""
    df = df.sort_values(["sess", "t"]).reset_index(drop=True)
    mv = df.spd > 3; thr = df.loc[mv, fuel].quantile(.9)
    burst = (mv & (df[fuel] > thr)).to_numpy()
    sess = df.sess.to_numpy(); n = len(df)
    def contrast(seed_idx, col):
        vals = []
        for lag in range(2, 7):
            j = seed_idx + lag; ok = j < n; j = j[ok]; ok2 = sess[j] == sess[seed_idx[ok]]
            vals.append(df[col].to_numpy()[j[ok2]])
        v = np.concatenate(vals); base = df.loc[mv, col].mean()
        return float(np.nanmean(v) - base)
    bi = np.flatnonzero(burst); mi = np.flatnonzero(mv.to_numpy()); pi = rng.choice(mi, size=len(bi), replace=False)
    out = {"n_seconds": int(n), "n_bursts": int(len(bi)),
           "E1_stft_echo_pp": round(contrast(bi, "stft"), 3), "E2_ltft_echo_pp": round(contrast(bi, "ltft"), 3),
           "I1_placebo_stft_pp": round(contrast(pi, "stft"), 3),
           "stft_lag_profile": {str(k): round(float(np.nanmean(df.stft.to_numpy()[[i + k for i in bi if i + k < n and sess[i + k] == sess[i]]]) - df.loc[mv, "stft"].mean()), 2) for k in (0, 1, 2, 3, 4, 6, 10, 20, 30)}}
    out["pass"] = {"E1": out["E1_stft_echo_pp"] < -1.0, "E2": abs(out["E2_ltft_echo_pp"]) < 0.25, "I1": abs(out["I1_placebo_stft_pp"]) < 0.3}
    R[name] = out; save(); print(name, json.dumps(out)); return out

# ---- Sonata (discovery set) ----
if "sonata" not in R:
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "sonata", "*", "All_*.csv"))):
        d = pd.read_csv(f, usecols=["fuel_usage", "car_speed", "short_fuel_bank", "long_fuel_bank"])
        if (d.short_fuel_bank <= -99).all(): continue
        fr.append(pd.DataFrame({"sess": f, "t": np.arange(len(d)), "spd": d.car_speed, "fuel": d.fuel_usage,
                                "stft": d.short_fuel_bank.where(d.short_fuel_bank > -99), "ltft": d.long_fuel_bank.where(d.long_fuel_bank > -99)}))
    echo(pd.concat(fr, ignore_index=True), "fuel", "sonata")

# ---- HCRL (confirmation) ----
if "hcrl" not in R:
    h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0],
                    usecols=["Fuel_consumption", "Vehicle_speed", "Short_Term_Fuel_Trim_Bank1", "Long_Term_Fuel_Trim_Bank1", "Time(s)", "Class", "PathOrder"])
    h = pd.DataFrame({"sess": h.Class + h.PathOrder.astype(str), "t": h["Time(s)"], "spd": h.Vehicle_speed, "fuel": h.Fuel_consumption,
                      "stft": h.Short_Term_Fuel_Trim_Bank1, "ltft": h.Long_Term_Fuel_Trim_Bank1})
    h = h.groupby(["sess", "t"], as_index=False).mean()   # HCRL logs ~2 rows per second: aggregate to 1 Hz as in p29
    full = h.groupby("sess").t.agg(["min", "max"])           # reindex to a gap-free 1 s grid so lags are true seconds
    h = pd.concat([g.set_index("t").reindex(range(int(full.loc[sn, "min"]), int(full.loc[sn, "max"]) + 1)).assign(sess=sn).rename_axis("t").reset_index()
                   for sn, g in h.groupby("sess")], ignore_index=True)
    echo(h, "fuel", "hcrl")

# ---- VED (confirmation): per-second MAF as fuel, trips with both trims logged ----
if "ved" not in R:
    import sys; sys.path.insert(0, os.path.join(ROOT, "scripts")); from p1_features import resample_trip
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv"))):
        cols = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]", "Short Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 1[%]"]
        d = pd.read_csv(f, usecols=cols, dtype={c: "float32" for c in cols[3:]})
        for (veh, trip), g in d.groupby(["VehId", "Trip"]):
            if g["Long Term Fuel Trim Bank 1[%]"].notna().mean() < 0.2: continue
            r = resample_trip(g)
            if r is None or r.maf.notna().mean() < 0.5: continue
            fr.append(pd.DataFrame({"sess": np.int32(len(fr)), "t": r.t.astype("int32"), "spd": r.speed.astype("float32"), "fuel": r.maf.astype("float32"),
                                    "stft": r.stft1.astype("float32"), "ltft": r.ltft1.astype("float32")}))
        del d; print("read", os.path.basename(f), "trips so far", len(fr), flush=True)
    echo(pd.concat(fr, ignore_index=True), "fuel", "ved")

# ---- E3: VED trip-level LTFT-axis silence (same panel/excess construction as p17/p10) ----
if "ved_E3" not in R:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
    lo, hi = v.fuel_per_100km.quantile([.01, .99]); v = v[(v.fuel_per_100km >= lo) & (v.fuel_per_100km <= hi)].copy()
    ENV = ["oat_mean", "ac_w_mean", "heat_w_mean", "dist_km", "dur_min", "weight_lb", "speed_mean", "pct_hwy"]
    v["excess"] = v.fuel_per_100km - cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0),
                                                       v[ENV], v.fuel_per_100km, cv=GroupKFold(5), groups=v.VehId)
    vt = v[v.ltft1_cov >= 0.2].dropna(subset=["ltft1_mean", "stft1_mean"]).copy()
    vt = vt[vt.groupby("VehId").ltft1_mean.transform("size") >= 15].copy(); gg = vt.groupby("VehId")
    z = lambda c: (vt[c] - gg[c].transform("mean")) / (gg[c].transform("std") + 1e-9)
    vt["zl"], vt["zs"] = z("ltft1_mean"), z("stft1_mean")
    R["ved_E3"] = {"n_trips": int(len(vt)), "n_vehicles": int(vt.VehId.nunique()),
                   "corr_excess_ltft_axis": round(float(vt.excess.corr(vt.zl)), 3),
                   "corr_excess_stft_z_registry": round(float(vt.excess.corr(vt.zs)), 3),
                   "corr_excess_sum_z_registry": round(float(vt.excess.corr(z("ltft1_mean") + z("stft1_mean"))), 3)}
    R["ved_E3"]["pass"] = abs(R["ved_E3"]["corr_excess_ltft_axis"]) < 0.1; save(); print("ved_E3", R["ved_E3"])
