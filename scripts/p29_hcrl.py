"""P29: item 5 of EVAL_STRENGTHENING_PLAN.md - real fixed-car, varying-driver check (HCRL / Kwak et al. 2016).
Data: 10 drivers (A-J) x 2 sessions (PathOrder 1/2) on ONE Kia Soul, same route; Fuel_consumption (instantaneous),
Vehicle_speed (km/h), STFT/LTFT Bank 1, Calculated_road_gradient. Cite: Kwak, Woo, Kim, PST 2016.

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

h = pd.read_csv(glob.glob(os.path.join(RAW, "hcrl", "*.csv"))[0])
h["sess"] = h["Class"] + h["PathOrder"].astype(str)
# collapse to 1 Hz (several rows per second in the raw file)
s1 = h.groupby(["sess", "Class", "PathOrder", "Time(s)"], as_index=False).agg(
    fuel=("Fuel_consumption", "mean"), spd=("Vehicle_speed", "mean"), stft=("Short_Term_Fuel_Trim_Bank1", "mean"),
    ltft=("Long_Term_Fuel_Trim_Bank1", "mean"), grade=("Calculated_road_gradient", "mean"))
R = {"n_sessions": int(s1.sess.nunique()), "n_seconds": int(len(s1))}

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
R["H2a_spearman_fuel_ltft"] = round(float(spearmanr(S.fuel_per_km, S.ltft).statistic), 3)
R["H2a_spearman_fuel_stft"] = round(float(spearmanr(S.fuel_per_km, S.stft).statistic), 3)
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
                        W.fuel_per_km, cv=GroupKFold(5), groups=W.driver)
W["excess"] = W.fuel_per_km - exp
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
json.dump(R, open(os.path.join(OUT, "p29_hcrl.json"), "w"), indent=2)
print(json.dumps({k: R[k] for k in R if k != "session_table"}, indent=2))
