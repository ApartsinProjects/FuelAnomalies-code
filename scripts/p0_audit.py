"""P0 data-readiness audit for VED. Checks sampling rate, fuel-trim coverage,
key-column missingness, and ICE subset size."""
import glob, os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
weeks = sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv")))
print(f"weekly CSVs found: {len(weeks)}")

# --- Static data: ICE / HEV / EV split ---
stat_ice = pd.read_excel(os.path.join(RAW, "VED_Static_ICE_HEV.xlsx"))
stat_ev = pd.read_excel(os.path.join(RAW, "VED_Static_PHEV_EV.xlsx"))
print("\n=== STATIC ===")
print("ICE&HEV rows:", len(stat_ice), "| PHEV&EV rows:", len(stat_ev))
etype_col = [c for c in stat_ice.columns if "engine" in c.lower() and "type" in c.lower()]
print("ICE&HEV columns:", list(stat_ice.columns))
if etype_col:
    print("EngineType counts (ICE&HEV file):")
    print(stat_ice[etype_col[0]].value_counts())

# --- Load a few representative weeks (first, middle, last of part1) ---
sample_weeks = [weeks[0], weeks[len(weeks)//2], weeks[-1]]
frames = []
for w in sample_weeks:
    df = pd.read_csv(w)
    df["__week"] = os.path.basename(w)
    frames.append(df)
    print(f"\nloaded {os.path.basename(w)}: rows={len(df):,}")
d = pd.concat(frames, ignore_index=True)
print("\n=== COLUMNS ===")
for c in d.columns:
    print(" ", repr(c))

trim_cols = [c for c in d.columns if "Fuel Trim" in c]
key_cols = ["Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]",
            "Absolute Load[%]", "Fuel Rate[L/hr]", "Outside Air Temperature[DegC]"]
key_cols = [c for c in key_cols if c in d.columns]

def cov(col):
    s = d[col]
    nonnull = s.notna().mean()
    # VED uses sentinel-like blanks / zeros in some cols; report both
    return nonnull

print("\n=== MISSINGNESS (fraction non-null, 3-week sample) ===")
for c in key_cols + trim_cols:
    print(f"  {cov(c)*100:6.2f}%  {c}")

# --- Fuel-trim coverage per vehicle (how many vehicles have ANY trim reading) ---
print("\n=== FUEL-TRIM COVERAGE PER VEHICLE ===")
if trim_cols:
    veh_has_trim = d.groupby("VehId")[trim_cols].apply(lambda g: g.notna().any().any())
    n_veh = d["VehId"].nunique()
    print(f"vehicles in sample: {n_veh}")
    print(f"vehicles with >=1 non-null trim reading: {int(veh_has_trim.sum())} "
          f"({veh_has_trim.mean()*100:.1f}%)")
    # per-vehicle fraction of rows with a valid LTFT bank1
    if "Long Term Fuel Trim Bank 1[%]" in d.columns:
        ltft = "Long Term Fuel Trim Bank 1[%]"
        per_veh = d.groupby("VehId")[ltft].apply(lambda s: s.notna().mean())
        print(f"per-vehicle LTFT-bank1 non-null fraction: "
              f"median={per_veh.median()*100:.1f}%  mean={per_veh.mean()*100:.1f}%")
        print(f"vehicles with >=20% LTFT rows: {(per_veh>=0.2).sum()} / {len(per_veh)}")

# --- Sampling rate: dt within (VehId, Trip) ---
print("\n=== SAMPLING RATE (within VehId,Trip) ===")
d2 = d.sort_values(["VehId", "Trip", "Timestamp(ms)"])
dt = d2.groupby(["VehId", "Trip"])["Timestamp(ms)"].diff()
dt = dt[(dt > 0) & (dt < 60000)]  # drop trip boundaries / gaps
print(f"dt(ms) median={dt.median():.0f}  p10={dt.quantile(.1):.0f}  "
      f"p90={dt.quantile(.9):.0f}  mode~={dt.round(-2).mode().iloc[0]:.0f}")
print(f"implied rate: ~{1000/dt.median():.2f} Hz (median dt)")

# --- Trip inventory ---
print("\n=== TRIP / VEHICLE INVENTORY (3-week sample) ===")
print("unique vehicles:", d["VehId"].nunique())
print("unique (VehId,Trip):", d.groupby(["VehId","Trip"]).ngroups)
trip_len = d2.groupby(["VehId","Trip"]).size()
print(f"points/trip: median={trip_len.median():.0f}  "
      f"p10={trip_len.quantile(.1):.0f}  p90={trip_len.quantile(.9):.0f}")
trip_dur = d2.groupby(["VehId","Trip"])["Timestamp(ms)"].agg(lambda s: (s.max()-s.min())/60000)
print(f"trip duration(min): median={trip_dur.median():.1f}  "
      f"p90={trip_dur.quantile(.9):.1f}")

# --- Fuel rate sanity ---
if "Fuel Rate[L/hr]" in d.columns:
    fr = d["Fuel Rate[L/hr]"]
    print("\n=== FUEL RATE[L/hr] ===")
    print(f"non-null={fr.notna().mean()*100:.1f}%  "
          f"min={fr.min():.2f} median={fr.median():.2f} p99={fr.quantile(.99):.2f} max={fr.max():.2f}")
    print(f"zeros (idle/coast): {(fr==0).mean()*100:.1f}%")
print("\nAUDIT DONE.")
