"""P63: the four signature features of every VED trip computed with the SAME definition used for the source
(UAH-DriveSet) and the two fixed cars (Kia, Sonata): one-second speed differences for acceleration and jerk,
harsh events = |a| > 2.5 m/s^2 per km (accelerations and brakings), speed SD in km/h. p1_features.py computes the
fleet's behaviour block with central differences and accelerations only; the frozen signature must see the
features it was trained on, so the signature scripts replace those four columns with these (bug found 2026-10-20).
Each trip is resampled to 1 Hz exactly as in p1 (p1_features.resample_trip).
Output: data/processed/signature_features.parquet (VehId, Trip, speed_std, accel_p95, harsh_accel_per_km, jerk_rms)."""
import os, io, glob, contextlib, runpy, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
COLS = ["VehId", "Trip", "Timestamp(ms)", "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]",
        "Air Conditioning Power[Watts]", "Heater Power[Watts]", "Short Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 1[%]"]
def kin(spd):                         # identical to attrib_setup.kin (source definition)
    v = np.asarray(spd, float) / 3.6; a = np.diff(v) if len(v) > 1 else np.array([0.]); j = np.diff(a) if len(a) > 1 else np.array([0.])
    d = np.nansum(v) / 1000.0 if len(v) else 1e-9
    return [np.nanstd(v) * 3.6, np.nanpercentile(a, 95) if len(a) else 0.0, np.sum(np.abs(a) > 2.5) / max(d, 1e-3), np.sqrt(np.nanmean(j ** 2)) if len(j) else 0.0]
def week(path):
    with contextlib.redirect_stdout(io.StringIO()):
        p1 = runpy.run_path(os.path.join(ROOT, "scripts", "p1_features.py"), run_name="p1")
    keys = set(map(tuple, pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"), columns=["VehId", "Trip"]).astype(int).to_numpy()))
    df = pd.read_csv(path, usecols=lambda c: c in COLS); out = []
    for (vid, tr), g in df.groupby(["VehId", "Trip"], sort=False):
        if (int(vid), int(tr)) not in keys: continue
        r = p1["resample_trip"](g)
        if r is None: continue
        out.append([int(vid), int(tr)] + kin(r["speed"].to_numpy()))
    return out
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv")))
    with Pool(int(os.environ.get("NPROC", "4"))) as P: rows = [r for part in P.imap_unordered(week, files) for r in part]
    S = pd.DataFrame(rows, columns=["VehId", "Trip", "speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]).drop_duplicates(["VehId", "Trip"])
    T = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"), columns=["VehId", "Trip", "speed_std"])
    M = T.merge(S, on=["VehId", "Trip"], how="left", suffixes=("_p1", ""))
    # invariants: every trip found; speed SD is definition-independent, so it must reproduce p1 exactly
    assert M.speed_std.notna().all(), f"{M.speed_std.isna().sum()} trips missing"
    assert np.allclose(M.speed_std, M.speed_std_p1, atol=1e-6), "speed SD differs from p1: resampling mismatch"
    S.to_parquet(os.path.join(ROOT, "data", "processed", "signature_features.parquet"), index=False)
    print(len(S), "trips; harsh zero share", round(float((S.harsh_accel_per_km == 0).mean()), 3))
