"""P56: do refuelling events step the long-term trim? (IJPHM review R2: ethanol blend changes at refuelling.)
Opel Corsa panel (17 months, 28 refuels). Trips = segment_file; refuel intervals = segment_id. Warm-engine trip-mean
LTFT. Compare the trip-to-trip change in LTFT across a refuel boundary (consecutive trips in different refuel
intervals) with changes between consecutive trips inside one interval, and express the refuel step in the rolling
detector's units (z against the previous 20 trips). Output: data/processed/p56_refuel_trim_steps.json"""
import os, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "data", "raw", "opel_corsa")
cols = ["timestamp", "LONG_FUEL_TRIM_1", "COOLANT_TEMP", "RPM", "segment_id", "segment_file"]
d = pd.read_csv(os.path.join(D, "dataset.csv"), usecols=cols, parse_dates=["timestamp"]).sort_values("timestamp")
w = d[(d.COOLANT_TEMP >= 70) & (d.RPM > 0)]
T = w.groupby("segment_file").agg(ltft=("LONG_FUEL_TRIM_1", "mean"), seg=("segment_id", "first"), t=("timestamp", "min"), n=("RPM", "size"))
T = T[T.n >= 120].sort_values("t").reset_index()
dl = T.ltft.diff().abs().to_numpy()[1:]; cross = (T.seg.diff() != 0).to_numpy()[1:]
mu = T.ltft.shift(1).rolling(20, min_periods=20).mean(); sd = T.ltft.shift(1).rolling(20, min_periods=20).std()
zz = ((T.ltft - mu) / sd).to_numpy()[1:]; okz = np.isfinite(zz)
R = {"trips": int(len(T)), "refuel_boundaries": int(cross.sum()),
     "abs_dLTFT_pp_median": {"across_refuel": round(float(np.median(dl[cross])), 3), "within_interval": round(float(np.median(dl[~cross])), 3)},
     "mannwhitney_p": float(mannwhitneyu(dl[cross], dl[~cross]).pvalue),
     "rolling_z_abs_median": {"first_trip_after_refuel": round(float(np.median(np.abs(zz[cross & okz]))), 3), "other_trips": round(float(np.median(np.abs(zz[~cross & okz]))), 3)},
     "share_abs_z_gt_1.9": {"first_trip_after_refuel": round(float(np.mean(np.abs(zz[cross & okz]) > 1.908)), 3), "other_trips": round(float(np.mean(np.abs(zz[~cross & okz]) > 1.908)), 3)}}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p56_refuel_trim_steps.json"), "w"), indent=1); print(json.dumps(R, indent=1))
