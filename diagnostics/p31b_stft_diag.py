"""P31b diagnostic (not a test): why does window STFT lean negative against fuel excess on the Sonata?
Reuses p31 window construction exactly, adds candidate confounders, reports the partial corr under each."""
import os, sys, runpy, io, contextlib, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "p31_sonata.py"))
s1, score = g["s1"], g["score"]
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
W = []
for sname, gg in s1.groupby("sess"):
    gg = gg.sort_values("Time(s)").reset_index(drop=True); wid = (gg["Time(s)"] - gg["Time(s)"].iloc[0]) // 60
    lt = gg.ltft.mean()
    for k, w in gg.groupby(wid):
        if len(w) < 45: continue
        dist = w.spd.sum() / 3600.0
        if dist < 0.1: continue
        W.append({"driver": w.Class.iloc[0], "sess": sname, "wnum": int(k), "fuel_per_km": w.fuel.sum() / dist,
                  "kin": score(w.spd.to_numpy()), "spd_mean": w.spd.mean(), "grade": w.grade.mean(),
                  "stft": w.stft.mean(), "ltft_dev": w.ltft.mean() - lt, "cutfrac": w.stft.isna().mean(),
                  "idlefrac": (w.spd < 3).mean(), "accfrac": (w.spd.diff() > 0.5).mean(),
                  "stft_valid_fuel": w.fuel[w.stft.notna()].mean()})
W = pd.DataFrame(W)
lo, hi = W.fuel_per_km.quantile([.01, .99]); W = W[(W.fuel_per_km >= lo) & (W.fuel_per_km <= hi)].reset_index(drop=True)
W["excess"] = W.fuel_per_km - cross_val_predict(HistGradientBoostingRegressor(max_iter=200, max_depth=4, random_state=0),
    W[["spd_mean", "grade"]], W.fuel_per_km, cv=GroupKFold(W.driver.nunique()), groups=W.driver)
W = W.dropna(subset=["stft", "ltft_dev"]).reset_index(drop=True)
def pc(d, z):
    Z = d[z].to_numpy(); rx = d.excess - LinearRegression().fit(Z, d.excess).predict(Z); ry = d.stft - LinearRegression().fit(Z, d.stft).predict(Z)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)
print("n windows", len(W), " reproduce p31 (kin only):", pc(W, ["kin"]))
for z in (["kin", "cutfrac"], ["kin", "idlefrac"], ["kin", "accfrac"], ["kin", "spd_mean"], ["kin", "wnum"],
          ["kin", "cutfrac", "idlefrac", "accfrac", "spd_mean"]):
    print("control", z, pc(W, z))
# within-session: demean everything by session
D = W.copy()
for c in ["excess", "stft", "kin", "cutfrac", "idlefrac"]: D[c] = D[c] - D.groupby("sess")[c].transform("mean")
print("within-session, kin:", pc(D, ["kin"]), " +cut/idle:", pc(D, ["kin", "cutfrac", "idlefrac"]))
print("warm only (window>=5):", pc(W[W.wnum >= 5], ["kin"]), " first 5 min only:", pc(W[W.wnum < 5], ["kin"]))
print("per driver:", {d: pc(x, ["kin"]) for d, x in W.groupby("driver")})
print("raw corrs with stft:", W[["excess", "kin", "cutfrac", "idlefrac", "accfrac", "spd_mean", "wnum"]].corrwith(W.stft).round(3).to_dict())
print("raw corrs with excess:", W[["cutfrac", "idlefrac", "accfrac", "wnum"]].corrwith(W.excess).round(3).to_dict())
print("session-mean stft spread SD:", round(W.groupby("sess").stft.mean().std(), 3), " within-session SD:", round(W.groupby("sess").stft.std().mean(), 3))
# per-second: stft vs fuel conditional on speed bin (does STFT respond to transients?)
v = s1.dropna(subset=["stft"]).copy(); v["acc"] = v.groupby("sess").spd.diff()
v["sbin"] = pd.cut(v.spd, [-1, 3, 20, 50, 80, 200]); v["abin"] = pd.cut(v.acc, [-50, -1, -0.2, 0.2, 1, 50])
print("per-second mean STFT by accel bin:", v.groupby("abin", observed=True).stft.mean().round(2).to_dict())
print("per-second mean STFT by speed bin:", v.groupby("sbin", observed=True).stft.mean().round(2).to_dict())
v["after_cut"] = v.groupby("sess").stft.transform(lambda s: s.index.isin([]))  # placeholder
s = s1.sort_values(["sess", "Time(s)"]).copy(); s["cut"] = s.stft.isna(); s["since_cut"] = 0
for sn, gg in s.groupby("sess"):
    c = 999; out = []
    for x in gg.cut: c = 0 if x else c + 1; out.append(c)
    s.loc[gg.index, "since_cut"] = out
s = s[~s.cut]; s["scb"] = pd.cut(s.since_cut, [0, 3, 10, 30, 120, 10**6])
print("per-second mean STFT by seconds since fuel cut:", s.groupby("scb", observed=True).stft.mean().round(2).to_dict())
