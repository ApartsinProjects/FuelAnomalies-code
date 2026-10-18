"""P1 feature pipeline for VED (ICE only).

Per (VehId, Trip): resample to 1 Hz, derive driving-behaviour features, MAF-derived
fuel target, trip conditions, and fuel-trim aggregates. Output one row per trip.

Fuel: L/hr = MAF[g/s]/14.7/745*3600  (stoichiometric gasoline, VED-standard).
Target: fuel_per_100km. Grade omitted in P1 (VSP uses grade=0; flagged as TODO).
"""
import glob, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

AFR, RHO = 14.7, 745.0          # gasoline stoich AFR, fuel density g/L
HARSH_ACCEL = 2.5               # m/s^2
HARSH_BRAKE = -2.5
IDLE_SPD = 3.0                  # km/h
# trip-quality filters
MIN_DIST_KM, MIN_DUR_S, MIN_MAF_COV = 0.5, 60, 0.5

def maf_to_lph(maf):
    return maf / AFR / RHO * 3600.0

def resample_trip(g):
    """Return a 1 Hz DataFrame for one trip, or None if too short/sparse."""
    g = g.dropna(subset=["Timestamp(ms)"]).sort_values("Timestamp(ms)")
    g = g.drop_duplicates(subset="Timestamp(ms)")
    if len(g) < 5:
        return None
    t = (g["Timestamp(ms)"].to_numpy() - g["Timestamp(ms)"].iloc[0]) / 1000.0
    dur = t[-1]
    if dur < MIN_DUR_S:
        return None
    grid = np.arange(0, np.floor(dur) + 1, 1.0)
    out = {"t": grid}
    cont = {"Vehicle Speed[km/h]": "speed", "MAF[g/sec]": "maf",
            "Engine RPM[RPM]": "rpm", "Absolute Load[%]": "load",
            "OAT[DegC]": "oat", "Air Conditioning Power[Watts]": "ac_w",
            "Heater Power[Watts]": "heat_w",
            "Short Term Fuel Trim Bank 1[%]": "stft1",
            "Long Term Fuel Trim Bank 1[%]": "ltft1"}
    for col, name in cont.items():
        if col not in g.columns:
            out[name] = np.full_like(grid, np.nan); continue
        s = g[col].to_numpy(dtype=float)
        m = ~np.isnan(s)
        if m.sum() >= 2:
            out[name] = np.interp(grid, t[m], s[m], left=np.nan, right=np.nan)
        else:
            out[name] = np.full_like(grid, np.nan)
    return pd.DataFrame(out)

def trip_features(r):
    """Compute trip-level features from a 1 Hz resampled trip r."""
    n = len(r)
    speed = r["speed"].to_numpy()                       # km/h
    v = speed / 3.6                                     # m/s
    maf = r["maf"].to_numpy()
    maf_cov = np.isfinite(maf).mean()
    if maf_cov < MIN_MAF_COV:
        return None
    # distance & fuel (1 Hz => each sample is 1 s)
    dist_km = np.nansum(speed) / 3600.0
    if dist_km < MIN_DIST_KM:
        return None
    lph = maf_to_lph(maf)
    fuel_L = np.nansum(lph) / 3600.0
    if not np.isfinite(fuel_L) or fuel_L <= 0:
        return None
    # kinematics
    a = np.gradient(v)                                  # m/s^2 (dt=1s)
    jerk = np.gradient(a)
    pos_a = a[a > 0]
    moving = speed > IDLE_SPD
    idle_frac = 1.0 - moving.mean()
    # stops: transitions moving -> idle
    stops = int(np.sum((moving[:-1]) & (~moving[1:])))
    f = {
        "n_sec": n,
        "dist_km": dist_km,
        "dur_min": (n - 1) / 60.0,
        "fuel_L": fuel_L,
        "fuel_per_100km": fuel_L / dist_km * 100.0,     # TARGET
        "maf_cov": maf_cov,
        # driving behaviour
        "speed_mean": np.nanmean(speed),
        "speed_std": np.nanstd(speed),
        "speed_p85": np.nanpercentile(speed, 85),
        "accel_pos_mean": pos_a.mean() if pos_a.size else 0.0,
        "accel_p95": np.nanpercentile(a, 95),
        "decel_p05": np.nanpercentile(a, 5),
        "jerk_rms": float(np.sqrt(np.nanmean(jerk**2))),
        "harsh_accel_per_km": int(np.sum(a > HARSH_ACCEL)) / dist_km,
        "harsh_brake_per_km": int(np.sum(a < HARSH_BRAKE)) / dist_km,
        "idle_frac": idle_frac,
        "stops_per_km": stops / dist_km,
        "pct_hwy": float((speed > 90).mean()),
        # VSP (grade=0, TODO add grade): W/kg proxy
        "vsp_mean": float(np.nanmean(v * (1.1 * a + 0.132) + 0.000302 * v**3)),
        # engine
        "rpm_mean": np.nanmean(r["rpm"].to_numpy()),
        "load_mean": np.nanmean(r["load"].to_numpy()),
        # conditions
        "oat_mean": np.nanmean(r["oat"].to_numpy()),
        "ac_w_mean": np.nanmean(r["ac_w"].to_numpy()),
        "heat_w_mean": np.nanmean(r["heat_w"].to_numpy()),
        # health signal (trip-level; per-vehicle drift computed downstream)
        "stft1_mean": np.nanmean(r["stft1"].to_numpy()),
        "ltft1_mean": np.nanmean(r["ltft1"].to_numpy()),
        "ltft1_cov": np.isfinite(r["ltft1"].to_numpy()).mean(),
        "trim_abs_mean": np.nanmean(np.abs(r["stft1"].to_numpy() + r["ltft1"].to_numpy())),
    }
    return f

def main():
    t0 = time.time()
    stat = pd.read_excel(os.path.join(RAW, "VED_Static_ICE_HEV.xlsx"))
    ice_ids = set(stat.loc[stat["Vehicle Type"] == "ICE", "VehId"].tolist())
    stat["Generalized_Weight"] = pd.to_numeric(stat["Generalized_Weight"], errors="coerce")
    wt = dict(zip(stat["VehId"], stat["Generalized_Weight"]))
    print(f"ICE vehicles: {len(ice_ids)}")
    weeks = sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv")))
    print(f"weeks: {len(weeks)}")
    rows = []
    for wi, w in enumerate(weeks):
        df = pd.read_csv(w)
        df = df[df["VehId"].isin(ice_ids)]
        kept = 0
        for (vid, trip), g in df.groupby(["VehId", "Trip"], sort=False):
            r = resample_trip(g)
            if r is None:
                continue
            f = trip_features(r)
            if f is None:
                continue
            f["VehId"] = int(vid); f["Trip"] = int(trip)
            f["weight_lb"] = wt.get(vid, np.nan)
            f["week"] = os.path.basename(w)
            rows.append(f); kept += 1
        print(f"[{wi+1:2d}/{len(weeks)}] {os.path.basename(w):24s} trips_kept={kept:4d} "
              f"total={len(rows):6d}  ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    p = os.path.join(OUT, "trip_features.parquet")
    out.to_parquet(p, index=False)
    print(f"\nWROTE {p}  shape={out.shape}")
    # quick sanity
    print("\n=== TARGET fuel_per_100km ===")
    q = out["fuel_per_100km"]
    print(f"n={len(q)}  median={q.median():.2f}  p10={q.quantile(.1):.2f}  "
          f"p90={q.quantile(.9):.2f}  p99={q.quantile(.99):.2f}")
    print("vehicles:", out["VehId"].nunique(),
          "| trips w/ LTFT cov>=0.2:", int((out["ltft1_cov"] >= 0.2).sum()))
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
