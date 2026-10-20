"""P28: item 4 of EVAL_STRENGTHENING_PLAN.md - cold warm-up as a real non-driver excess (abstention test).

Warm-up enrichment is a real engine-side cause of extra fuel that the driver did not create. The fused
Algorithm-1 rule (frozen UAH kinematic signature + within-vehicle LTFT axis, source-calibrated likelihood
ratios, 'none' if both ratios < 1, conformal abstention thresholds from p26) must not blame the driver for it.

Segments (from the raw 1 Hz VED week files): for trips with median OAT < 0 C and duration >= 420 s,
COLD = first 120 s; WARM controls = non-overlapping 120 s segments starting at 300 s (up to 5 per trip).
Kinematic score standardized against the pooled WARM segments; LTFT axis = (segment LTFT - vehicle annual mean)
/ vehicle annual SD (the paper's axis, from trip_features). Enrichment check: idle (speed < 3 km/h) fuel rate in
COLD vs WARM segments.
PRE-REGISTERED (alpha = 0.05 primary): COLD segments get 'none' or 'abstain' in > 70% and 'driver' in < 15%.
KILL: 'driver' > 30% on COLD segments (idle/low-speed warm-up gets blamed on drivers).
Incremental + resumable: one row per segment appended to data/processed/p28_segments.csv, files already done skipped.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw"); OUT = os.path.join(ROOT, "data", "processed")
SEGCSV = os.path.join(OUT, "p28_segments.csv"); DONE = os.path.join(OUT, "p28_done_files.txt")

def kin(spd):
    s = np.asarray(spd, float) / 3.6; a = np.diff(s) if len(s) > 1 else np.array([0.]); j = np.diff(a) if len(a) > 1 else np.array([0.])
    d = np.nansum(s) / 1000.0 if len(s) else 1e-9
    return [np.nanstd(s) * 3.6, np.nanpercentile(a, 95) if len(a) else 0.0, np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2)) if len(j) else 0.0]
ur, uy = [], []
for f in sorted(glob.glob(os.path.join(RAW, "uah", "D*_merged.csv"))):
    d = pd.read_csv(f); ts = d["Timestamp (seconds)"].to_numpy(); seg = np.concatenate([[0], np.cumsum(np.diff(ts) < 0)]); d = d.assign(_s=seg)
    for _, rec in d.groupby("_s"):
        rec = rec.reset_index(drop=True); t0 = rec["Timestamp (seconds)"].iloc[0]; wid = ((rec["Timestamp (seconds)"] - t0) // 30).astype(int)
        for _, w in rec.groupby(wid):
            if len(w) < 15: continue
            r = [w["Ratio normal (base 1)"].sum(), w["Ratio drowsy (base 1)"].sum(), w["Ratio aggressive (base 1)"].sum()]; c = int(np.argmax(r))
            if c == 1: continue
            ur.append(kin(w["Speed (km/h)"])); uy.append(1 if c == 2 else 0)
Uu = np.array(ur); uy = np.array(uy); su = StandardScaler().fit(Uu); Uz = su.transform(Uu)
Wk = LogisticRegression(max_iter=1000).fit(Uz, uy).coef_[0]
sc = Uz @ Wk; S_driver = abs(sc[uy == 1].mean() - sc[uy == 0].mean()) / np.sqrt((sc[uy == 1].var() + sc[uy == 0].var()) / 2)

tf = pd.read_parquet(os.path.join(OUT, "trip_features.parquet"))

tf = tf.drop(columns=["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]).merge(pd.read_parquet(os.path.join(OUT, "signature_features.parquet")), on=["VehId", "Trip"], how="left")   # signature features with the source (UAH) definition, p63
tf = tf[(tf.ltft1_cov >= 0.2) & tf.ltft1_mean.notna()]
tf = tf[tf.groupby("VehId").ltft1_mean.transform("size") >= 15]
vstat = tf.groupby("VehId").ltft1_mean.agg(["mean", "std"])
S_fault = 3.49 / vstat["std"].median()  # LiU-ICE smallest measured fueling deviation, pp (p37)
taus = json.load(open(os.path.join(OUT, "p26_seasonal_zero.json")))["item2"]
TAU = {0.05: taus["alpha_0.05"]["tau"], 0.10: taus["alpha_0.1"]["tau"]}

COLS = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]", "OAT[DegC]",
        "Short Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 1[%]"]
done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
if not os.path.exists(SEGCSV):
    open(SEGCSV, "w").write("file,VehId,Trip,kind,oat,kin_raw,ltft,stft,idle_fuel_lph,n_idle\n")
for fpath in sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv"))):
    fn = os.path.basename(fpath)
    if fn in done: continue
    df = pd.read_csv(fpath, usecols=COLS, low_memory=False)
    df = df[df.VehId.isin(vstat.index)]
    rows = []
    for (vid, trip), g in df.groupby(["VehId", "Trip"]):
        oat = g["OAT[DegC]"].median()
        if not np.isfinite(oat) or oat >= 0: continue
        t = (g["Timestamp(ms)"].to_numpy() - g["Timestamp(ms)"].min()) / 1000.0
        if t.max() < 420: continue
        grid = np.arange(0, int(t.max()))
        def interp(col, ffill=False):
            x = g[col].to_numpy(dtype=float); m = np.isfinite(x)
            return np.interp(grid, t[m], x[m]) if m.sum() >= 2 else np.full(len(grid), np.nan)
        spd = interp("Vehicle Speed[km/h]"); maf = interp("MAF[g/sec]")
        lt = interp("Long Term Fuel Trim Bank 1[%]"); st = interp("Short Term Fuel Trim Bank 1[%]")
        fuel_lph = maf / (14.7 * 745) * 3600
        segs = [("cold", 0)] + [("warm", s0) for s0 in range(300, len(grid) - 120, 120)][:5]
        for kind, s0 in segs:
            sl = slice(s0, s0 + 120); sp = spd[sl]
            if len(sp) < 100 or not np.isfinite(sp).all(): continue
            idle = sp < 3
            rows.append(f"{fn},{vid},{trip},{kind},{oat:.2f},{float(su.transform(np.array([kin(sp)]))[0] @ Wk):.5f},"
                        f"{np.nanmean(lt[sl]):.4f},{np.nanmean(st[sl]):.4f},"
                        f"{(np.nanmean(fuel_lph[sl][idle]) if idle.any() else np.nan):.5f},{int(idle.sum())}")
    with open(SEGCSV, "a") as fh:
        fh.write("\n".join(rows) + ("\n" if rows else ""))
    with open(DONE, "a") as fh:
        fh.write(fn + "\n")
    print(f"[p28] {fn}: {len(rows)} segments", flush=True)

S = pd.read_csv(SEGCSV).dropna(subset=["kin_raw", "ltft"])
S = S[S.VehId.isin(vstat.index)].copy()
warm = S[S.kind == "warm"]
S["kz"] = (S.kin_raw - warm.kin_raw.mean()) / warm.kin_raw.std()
S["cz"] = (S.ltft - S.VehId.map(vstat["mean"])) / S.VehId.map(vstat["std"])
ld = N.logpdf(S.kz, S_driver, 1) - N.logpdf(S.kz, 0, 1); lf = N.logpdf(S.cz, S_fault, 1) - N.logpdf(S.cz, 0, 1)
none = (ld < 0) & (lf < 0)
R = {"n_cold_segments": int((S.kind == "cold").sum()), "n_warm_segments": int((S.kind == "warm").sum()),
     "n_vehicles": int(S.VehId.nunique()), "S_driver": round(float(S_driver), 3), "S_fault": round(float(S_fault), 3)}
for kind in ["cold", "warm"]:
    m = (S.kind == kind).to_numpy()
    R[f"{kind}_idle_fuel_lph_mean"] = round(float(S.loc[m, "idle_fuel_lph"].mean()), 4)
    R[f"{kind}_kz_mean"] = round(float(S.loc[m, "kz"].mean()), 3); R[f"{kind}_cz_mean"] = round(float(S.loc[m, "cz"].mean()), 3)
    for a, t in TAU.items():
        issued = (~none) & (np.abs(ld - lf) >= t)
        R[f"{kind}_alpha{a}"] = {"none": round(float(none[m].mean()), 3),
                                 "abstain": round(float(((~none) & ~issued)[m].mean()), 3),
                                 "driver": round(float((issued & (ld > lf))[m].mean()), 3),
                                 "fault": round(float((issued & (lf >= ld))[m].mean()), 3)}
c = R["cold_alpha0.05"]
R["VERDICT"] = {"pass_none_or_abstain_gt_0.70": bool(c["none"] + c["abstain"] > 0.70), "pass_driver_lt_0.15": bool(c["driver"] < 0.15),
                "KILL_driver_gt_0.30": bool(c["driver"] > 0.30)}
json.dump(R, open(os.path.join(OUT, "p28_coldstart.json"), "w"), indent=2)
print(json.dumps(R, indent=2))
