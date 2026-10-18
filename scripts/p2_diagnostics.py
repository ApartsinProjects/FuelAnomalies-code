"""Root-cause diagnostics for the null go/no-go.

D1: How much of per-vehicle residual is just engine size/class heterogeneity?
D2: Trim-blindness of the target — recompute trim-corrected fuel, size the effect.
D3: WITHIN-vehicle (fixed-effects/demeaned) link between residual and trim over time
    (the correct axis for health *degradation*).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
stat = pd.read_excel(os.path.join(RAW, "VED_Static_ICE_HEV.xlsx"))
df = df.merge(stat[["VehId", "Vehicle Class", "Engine Configuration & Displacement"]],
              on="VehId", how="left")

FEATS = ["speed_mean", "speed_std", "speed_p85", "accel_pos_mean", "accel_p95",
         "decel_p05", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km",
         "idle_frac", "stops_per_km", "pct_hwy", "vsp_mean",
         "oat_mean", "ac_w_mean", "heat_w_mean", "weight_lb", "dist_km", "dur_min"]
TARGET = "fuel_per_100km"
d = df.dropna(subset=[TARGET]).copy()
lo, hi = d[TARGET].quantile([.01, .99]); d = d[(d[TARGET] >= lo) & (d[TARGET] <= hi)].copy()

oof = np.full(len(d), np.nan)
for tr, te in GroupKFold(5).split(d[FEATS], d[TARGET], d["VehId"]):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0)
    m.fit(d[FEATS].to_numpy()[tr], d[TARGET].to_numpy()[tr])
    oof[te] = m.predict(d[FEATS].to_numpy()[te])
d["pred"] = oof
d["resid"] = d[TARGET] - d["pred"]

# parse displacement (liters) from e.g. "V6 3.5L" / "I4 2.0L"
disp = d["Engine Configuration & Displacement"].astype(str).str.extract(r"(\d\.\d)")[0]
d["disp_L"] = pd.to_numeric(disp, errors="coerce")

# ---------- D1: residual vs engine size/class ----------
print("=== D1: is per-vehicle residual just engine-size heterogeneity? ===")
veh = d.groupby("VehId").agg(n_trips=("resid","size"), resid_mean=("resid","mean"),
                             disp_L=("disp_L","first"), weight=("weight_lb","first")).dropna(subset=["resid_mean"])
veh = veh[veh["n_trips"]>=15]
vv = veh.dropna(subset=["disp_L"])
pr,pp = pearsonr(vv["resid_mean"], vv["disp_L"])
print(f"resid_mean vs engine displacement: r={pr:+.3f} p={pp:.2e}  (n={len(vv)})")
print(f"resid_mean spread: sd={veh['resid_mean'].std():.2f} L/100km  range=[{veh['resid_mean'].min():.2f},{veh['resid_mean'].max():.2f}]")
print("mean |resid| by displacement bucket:")
print(vv.assign(bkt=pd.cut(vv.disp_L,[0,2,3,4,10])).groupby("bkt",observed=True)["resid_mean"].agg(["mean","count"]).round(2).to_string())

# ---------- D2: trim-blindness of the target ----------
print("\n=== D2: trim-blindness of MAF-stoich fuel ===")
# actual injected fuel ~ stoich * (1 + (STFT+LTFT)/100). Size the ignored fraction.
tot = (d["stft1_mean"].fillna(0) + d["ltft1_mean"].fillna(0))
have = d["ltft1_cov"]>=0.2
print(f"trips with trim: {have.sum()}")
print(f"total trim (STFT+LTFT) %, where present: median={tot[have].median():+.2f}  "
      f"mean={tot[have].mean():+.2f}  p10={tot[have].quantile(.1):+.2f}  p90={tot[have].quantile(.9):+.2f}")
print(f"=> MAF-stoich target ignores a median ~{abs(tot[have].median()):.1f}% (and up to "
      f"~{max(abs(tot[have].quantile(.1)),abs(tot[have].quantile(.9))):.0f}%) of true injected fuel.")

# ---------- D3: WITHIN-vehicle temporal link (fixed effects) ----------
print("\n=== D3: within-vehicle (demeaned) resid vs trim — the health-degradation axis ===")
dd = d[have].copy()
dd["ltft"] = dd["ltft1_mean"]; dd["ltft_abs"] = dd["ltft1_mean"].abs()
# demean within vehicle
for c in ["resid","ltft","ltft_abs"]:
    dd[c+"_dm"] = dd[c] - dd.groupby("VehId")[c].transform("mean")
# keep vehicles with >=15 trimmed trips
cnt = dd.groupby("VehId")["resid"].transform("size")
dd = dd[cnt>=15]
print(f"trimmed trips used: {len(dd)}  vehicles: {dd['VehId'].nunique()}")
for name,col in [("signed LTFT","ltft_dm"),("|LTFT|","ltft_abs_dm")]:
    pr,pp = pearsonr(dd[col], dd["resid_dm"]); sr,sp = spearmanr(dd[col], dd["resid_dm"])
    print(f"  within-veh resid vs {name}: Pearson r={pr:+.3f} p={pp:.2e} | Spearman r={sr:+.3f}")
# placebo: shuffle within-vehicle demeaned trim across all rows
rng=np.random.default_rng(0); sh=dd["ltft_abs_dm"].to_numpy().copy(); rng.shuffle(sh)
pr0,pp0=pearsonr(sh, dd["resid_dm"]); print(f"  placebo (shuffled): r={pr0:+.3f} p={pp0:.2e}")
