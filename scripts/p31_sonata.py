"""P31: second fixed-car replication with the same pre-registered tests as P29 (HCRL).
Data: "This Car is Mine!" (IEEE DataPort 10.21227/qar8-sd42): 4 drivers (A-D), 30 trips, ONE Hyundai YF Sonata, 1 Hz;
fuel_usage (instantaneous), car_speed (km/h, integer), short/long_fuel_bank (%), road_slope.

The car is fixed, so between-session and within-session fuel differences come from driving, traffic and the ECU's
state, never from a vehicle fault. This tests the paper's claim "the combustion (trim) axis is silent for
driver-caused excess" on real data outside VED, and the driver axis with the FROZEN UAH signature.

PRE-REGISTERED (fixed before computing):
 H1 driver signature tracks fuel: across the 20 sessions, Spearman(fuel per km, frozen kinematic score) > 0.3.
 H2 trims silent for driver-caused excess: (a) session level |Spearman(fuel/km, session LTFT)| < Spearman(fuel/km,
    kinematic); (b) window level (60 s): corr(excess, kinematic score) > 0.2 while |partial corr(excess, STFT |
    kinematic)| < 0.1 and |corr(excess, within-session LTFT deviation)| < 0.1. Excess = window fuel/km minus an
    out-of-fold (GroupKFold by driver) model of fuel/km on speed mean and road gradient only.
 H3 (descriptive, explains Kwak's driver-ID result): LTFT is a session-level state: intraclass correlation by
    session > 0.7 and within-session SD << between-session SD.
 KILL: H1 fails, or fuel/km tracks LTFT more strongly than the kinematic score.
"""
import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW = os.path.join(ROOT, "data", "raw"); OUT = os.path.join(ROOT, "data", "processed")

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
Uu = np.array(ur); uy = np.array(uy); su = StandardScaler().fit(Uu)
Wk = LogisticRegression(max_iter=1000).fit(su.transform(Uu), uy).coef_[0]

frames = []
for f in sorted(glob.glob(os.path.join(RAW, "sonata", "*", "All_*.csv"))):
    d = pd.read_csv(f); drv = os.path.basename(os.path.dirname(f)); trip = os.path.basename(f)[:-4]
    frames.append(pd.DataFrame({"sess": drv + "_" + trip, "Class": drv, "PathOrder": trip, "Time(s)": np.arange(len(d)),
        "fuel": d["fuel_usage"], "spd": d["car_speed"], "stft": d["short_fuel_bank"], "ltft": d["long_fuel_bank"],
        "grade": d["road_slope"]}))
s1 = pd.concat(frames, ignore_index=True)   # one row per second (dataset docs: 1 Hz)
# -100 is the "not available" code for both trims; it occurs only with fuel_usage == 0 (engine off, trip edges and two
# whole trips A_All_8 / B_All_5 with no fuel data). Treat as missing; drop trips with no valid fuel.
s1.loc[s1.stft <= -99, "stft"] = np.nan; s1.loc[s1.ltft <= -99, "ltft"] = np.nan
empty = s1.groupby("sess").stft.apply(lambda x: x.notna().mean() == 0); s1 = s1[~s1.sess.map(empty)].copy()
R_sentinel = {"dropped_empty_trips": sorted(empty[empty].index.tolist()), "stft_sentinel_frac": float(s1.stft.isna().mean()), "ltft_sentinel_frac": float(s1.ltft.isna().mean())}

R = {"n_sessions": int(s1.sess.nunique()), "n_seconds": int(len(s1)), **R_sentinel}

def score(spd_series):
    return float((su.transform(np.array([kin(spd_series)])) @ Wk)[0])

# ---- session level ----
rows = []
for sname, g in s1.groupby("sess"):
    g = g.sort_values("Time(s)"); dist = g.spd.sum() / 3600.0                       # km (1 Hz, km/h)
    rows.append({"sess": sname, "driver": g.Class.iloc[0], "fuel_per_km": g.fuel.sum() / max(dist, 1e-6),
                 "kin": score(g.spd.to_numpy()), "ltft": g.ltft.mean(), "stft": g.stft.mean(), "dist_km": dist})
S = pd.DataFrame(rows)
R["H1_spearman_fuel_kin"] = round(float(spearmanr(S.fuel_per_km, S.kin).statistic), 3)
Sl = S.dropna(subset=["ltft"]); R["H2a_spearman_fuel_ltft"] = round(float(spearmanr(Sl.fuel_per_km, Sl.ltft).statistic), 3)
Ss = S.dropna(subset=["stft"]); R["H2a_spearman_fuel_stft"] = round(float(spearmanr(Ss.fuel_per_km, Ss.stft).statistic), 3)
R["session_table"] = S.round(3).to_dict(orient="records")

# ---- window level (60 s) ----
W = []
for sname, g in s1.groupby("sess"):
    g = g.sort_values("Time(s)").reset_index(drop=True); wid = (g["Time(s)"] - g["Time(s)"].iloc[0]) // 60
    lt_sess = g.ltft.mean()
    for _, w in g.groupby(wid):
        if len(w) < 45: continue
        dist = w.spd.sum() / 3600.0
        if dist < 0.1: continue
        W.append({"driver": w.Class.iloc[0], "sess": sname, "fuel_per_km": w.fuel.sum() / dist, "kin": score(w.spd.to_numpy()),
                  "spd_mean": w.spd.mean(), "grade": w.grade.mean(), "stft": w.stft.mean(), "ltft_dev": w.ltft.mean() - lt_sess})
W = pd.DataFrame(W)
lo, hi = W.fuel_per_km.quantile([.01, .99]); W = W[(W.fuel_per_km >= lo) & (W.fuel_per_km <= hi)].reset_index(drop=True)
exp = cross_val_predict(HistGradientBoostingRegressor(max_iter=200, max_depth=4, random_state=0), W[["spd_mean", "grade"]],
                        W.fuel_per_km, cv=GroupKFold(W.driver.nunique()), groups=W.driver)
W["excess"] = W.fuel_per_km - exp
W = W.dropna(subset=["stft", "ltft_dev"]).reset_index(drop=True)
def pcorr(x, y, z):
    rx = x - LinearRegression().fit(z, x).predict(z); ry = y - LinearRegression().fit(z, y).predict(z)
    return float(np.corrcoef(rx, ry)[0, 1])
R["n_windows"] = int(len(W))
R["H2b_corr_excess_kin"] = round(float(np.corrcoef(W.excess, W.kin)[0, 1]), 3)
R["H2b_pcorr_excess_stft_given_kin"] = round(pcorr(W.excess.to_numpy(), W.stft.to_numpy(), W[["kin"]].to_numpy()), 3)
R["H2b_corr_excess_ltftdev"] = round(float(np.corrcoef(W.excess, W.ltft_dev)[0, 1]), 3)

# ---- H3: LTFT is a session-level state ----
sec = s1.copy()
between = sec.groupby("sess").ltft.mean().var(); within = sec.groupby("sess").ltft.var().mean()
R["H3_ltft_icc"] = round(float(between / (between + within)), 3)
R["H3_ltft_between_sd"] = round(float(np.sqrt(between)), 3); R["H3_ltft_within_sd"] = round(float(np.sqrt(within)), 3)

R["VERDICT"] = {"H1_pass": bool(R["H1_spearman_fuel_kin"] > 0.3),
                "H2a_pass": bool(abs(R["H2a_spearman_fuel_ltft"]) < R["H1_spearman_fuel_kin"]),
                "H2b_pass": bool(R["H2b_corr_excess_kin"] > 0.2 and abs(R["H2b_pcorr_excess_stft_given_kin"]) < 0.1 and abs(R["H2b_corr_excess_ltftdev"]) < 0.1),
                "H3_pass": bool(R["H3_ltft_icc"] > 0.7)}
R["KILL"] = bool((not R["VERDICT"]["H1_pass"]) or abs(R["H2a_spearman_fuel_ltft"]) > R["H1_spearman_fuel_kin"])
json.dump(R, open(os.path.join(OUT, "p31_sonata.json"), "w"), indent=2)
print(json.dumps({k: R[k] for k in R if k != "session_table"}, indent=2))
