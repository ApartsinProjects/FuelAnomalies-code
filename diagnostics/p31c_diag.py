"""P31c diagnostic: (1) is the STFT lean a driving-transient effect (nonlinear kinematic control)?
(2) why does session-level H1 fail (routes/distance/idle)? (3) data sanity (fuel at decel, slope, units)."""
import os, io, contextlib, runpy, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "p31_sonata.py"))
s1, score, S = g["s1"], g["score"], g["S"]
s1 = s1.sort_values(["sess", "Time(s)"]).copy(); s1["acc"] = s1.groupby("sess").spd.diff().fillna(0)
v = s1.dropna(subset=["stft"])
print("== data sanity ==")
print("speed integer frac", round((v.spd == v.spd.round()).mean(), 3), " fuel quantiles", v.fuel.quantile([0, .05, .5, .95, 1]).round(0).tolist())
print("fuel by accel bin:", v.groupby(pd.cut(v.acc, [-50, -3, -1, 0, 1, 3, 50]), observed=True).fuel.mean().round(0).to_dict())
print("fuel==0 frac when decel<=-3:", round((v.fuel[v.acc <= -3] == 0).mean(), 3))
print("slope quantiles", s1.grade.quantile([0, .05, .5, .95, 1]).round(2).tolist())
print("per-second corr(stft, fuel) within speed>20:", round(v[v.spd > 20][["stft", "fuel"]].corr().iloc[0, 1], 3))
print("lagged corr(stft_t, fuel_{t-k}) k=0,2,5,10:", [round(v.stft.corr(v.groupby("sess").fuel.shift(k)), 3) for k in (0, 2, 5, 10)])
print("== session level ==")
S2 = S.copy(); S2["idle_frac"] = s1.groupby("sess").apply(lambda x: (x.spd < 3).mean()).reindex(S2.sess).values
print(S2[["sess", "dist_km", "fuel_per_km", "kin", "idle_frac", "ltft"]].round(3).to_string(index=False))
print("spearman fuel/km vs dist", round(spearmanr(S2.fuel_per_km, S2.dist_km).statistic, 3), " vs idle", round(spearmanr(S2.fuel_per_km, S2.idle_frac).statistic, 3),
      " kin vs idle", round(spearmanr(S2.kin, S2.idle_frac).statistic, 3))
long = S2[S2.dist_km > S2.dist_km.median()]
print("H1 on longer half (n=%d):" % len(long), round(spearmanr(long.fuel_per_km, long.kin).statistic, 3))
print("== nonlinear kinematic control for STFT ==")
W = []
for sn, gg in s1.groupby("sess"):
    wid = (gg["Time(s)"] - gg["Time(s)"].iloc[0]) // 60
    for k, w in gg.groupby(wid):
        if len(w) < 45 or w.spd.sum() / 3600 < 0.1: continue
        a = w.acc.to_numpy()
        W.append({"driver": w.Class.iloc[0], "sess": sn, "wnum": k, "fuel_per_km": w.fuel.sum() / (w.spd.sum() / 3600), "kin": score(w.spd.to_numpy()),
                  "spd_mean": w.spd.mean(), "spd_sd": w.spd.std(), "grade": w.grade.mean(), "stft": w.stft.mean(),
                  **{f"a{i}": np.mean((a > lo) & (a <= hi)) for i, (lo, hi) in enumerate([(-50, -3), (-3, -1), (-1, 0), (0, 1), (1, 3), (3, 50)])},
                  "hi_spd": (w.spd > 80).mean()})
W = pd.DataFrame(W).dropna(subset=["stft"])
lo, hi = W.fuel_per_km.quantile([.01, .99]); W = W[(W.fuel_per_km >= lo) & (W.fuel_per_km <= hi)].reset_index(drop=True)
cv = GroupKFold(W.driver.nunique())
W["excess"] = W.fuel_per_km - cross_val_predict(HistGradientBoostingRegressor(max_depth=4, random_state=0), W[["spd_mean", "grade"]], W.fuel_per_km, cv=cv, groups=W.driver)
X = ["kin", "spd_mean", "spd_sd", "grade", "wnum", "hi_spd"] + [f"a{i}" for i in range(6)]
oof = lambda y: y - cross_val_predict(HistGradientBoostingRegressor(max_depth=3, max_iter=150, random_state=0), W[X], y, cv=cv, groups=W.driver)
re, rs = oof(W.excess), oof(W.stft)
print("partial corr(excess, STFT | rich driving context, out-of-fold):", round(np.corrcoef(re, rs)[0, 1], 3))
print("STFT explained by driving context, OOF R2:", round(1 - np.var(rs) / np.var(W.stft), 3))
print("raw corr STFT with accel bins:", W[[f"a{i}" for i in range(6)] + ["hi_spd"]].corrwith(W.stft).round(3).to_dict())
