"""P2-lite + PRIMARY GO/NO-GO.

Hypothesis (RQ3): after predicting trip fuel from DRIVER-controllable + condition
features only (no engine-state, no trims, no vehicle ID), the per-vehicle mean
residual should correlate with that vehicle's fuel-trim drift |LTFT| — i.e. cars that
burn more than their driving explains are the cars whose ECU is compensating more.

Design: GroupKFold(by VehId) out-of-fold residuals so the model never sees the vehicle
it scores. Placebo: shuffle trim across vehicles -> correlation must vanish.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
print(f"trips={len(df)}  vehicles={df['VehId'].nunique()}")

# driver-controllable + conditions ONLY (exclude rpm/load = engine-state, and trims)
FEATS = ["speed_mean", "speed_std", "speed_p85", "accel_pos_mean", "accel_p95",
         "decel_p05", "jerk_rms", "harsh_accel_per_km", "harsh_brake_per_km",
         "idle_frac", "stops_per_km", "pct_hwy", "vsp_mean",
         "oat_mean", "ac_w_mean", "heat_w_mean", "weight_lb", "dist_km", "dur_min"]
TARGET = "fuel_per_100km"

d = df.dropna(subset=[TARGET]).copy()
# clip extreme target outliers (sensor glitches) at p1/p99
lo, hi = d[TARGET].quantile([.01, .99])
d = d[(d[TARGET] >= lo) & (d[TARGET] <= hi)].copy()
X = d[FEATS].to_numpy()
y = d[TARGET].to_numpy()
groups = d["VehId"].to_numpy()

# --- out-of-fold predictions, grouped by vehicle ---
oof = np.full(len(d), np.nan)
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, groups):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_depth=6, random_state=0)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict(X[te])
d["pred"] = oof
d["resid"] = d[TARGET] - d["pred"]
mape = (np.abs(d["resid"]) / d[TARGET]).mean() * 100
r2 = 1 - np.sum(d["resid"]**2) / np.sum((y - y.mean())**2)
print(f"\n=== FUEL MODEL (OOF, grouped by vehicle) ===")
print(f"MAPE={mape:.2f}%  R2={r2:.3f}  (driver+conditions features only, no vehicle ID)")

# --- per-vehicle aggregates ---
d["ltft_abs"] = d["ltft1_mean"].abs()
g = d.groupby("VehId")
veh = pd.DataFrame({
    "n_trips": g.size(),
    "resid_mean": g["resid"].mean(),
    "ltft_abs": g["ltft_abs"].mean(),        # trim-drift magnitude
    "ltft_signed": g["ltft1_mean"].mean(),
    "ltft_cov": g["ltft1_cov"].mean(),
})
# keep vehicles with stable estimates + real trim coverage
veh = veh[(veh["n_trips"] >= 15) & (veh["ltft_cov"] >= 0.2) & veh["ltft_abs"].notna()]
print(f"\nvehicles passing filter (>=15 trips, LTFT cov>=0.2): {len(veh)}")

# --- PRIMARY GO/NO-GO correlation ---
pr, pp = pearsonr(veh["resid_mean"], veh["ltft_abs"])
sr, sp = spearmanr(veh["resid_mean"], veh["ltft_abs"])
print(f"\n=== GO/NO-GO: per-vehicle resid_mean vs |LTFT| ===")
print(f"Pearson  r={pr:+.3f}  p={pp:.2e}")
print(f"Spearman r={sr:+.3f}  p={sp:.2e}")

# --- placebo: shuffle trim across vehicles ---
rng = np.random.default_rng(0)
shuf = veh["ltft_abs"].to_numpy().copy(); rng.shuffle(shuf)
pr0, pp0 = pearsonr(veh["resid_mean"], shuf)
print(f"placebo (shuffled |LTFT|): Pearson r={pr0:+.3f} p={pp0:.2e}  (should be ~0)")

# --- show the extremes (inspect, per verify-before-report) ---
v2 = veh.sort_values("resid_mean")
print("\nLowest-residual (efficient-for-driving) vehicles:")
print(v2.head(5)[["n_trips","resid_mean","ltft_abs","ltft_signed"]].round(2).to_string())
print("\nHighest-residual (burns-more-than-driving-explains) vehicles:")
print(v2.tail(5)[["n_trips","resid_mean","ltft_abs","ltft_signed"]].round(2).to_string())

veh.to_csv(os.path.join(ROOT, "data", "processed", "vehicle_gonogo.csv"))
print("\nwrote data/processed/vehicle_gonogo.csv")
