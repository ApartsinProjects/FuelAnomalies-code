"""P41: one car, 17 months, with fill-up records (Kaggle OBD2_panel_opel_2012, CC BY 4.0; Opel Corsa 1.2, 2025-04..2026-09).
The car runs a persistent positive long-term trim (median about +14%), i.e. the controller adds fuel beyond the
air-flow stoichiometric amount throughout the panel.

A  Fuel target vs the pump. Per refuel interval (refueling_history.csv, Segment_id), integrate
   Eq. 1 fuel = MAF / 14.7 / 745 g/L (no trims) and Eq. 2 fuel = Eq. 1 x (1 + (STFT + LTFT)/100), over warm and cold
   running, and compare with the litres bought at the next fill-up.
B  Seasonal drift on one car: warm-engine trip-level LTFT (trip = segment_file) against month and intake temperature.
C  Unmetered-air signature: warm-engine LTFT across air-flow bins.

PRE-REGISTERED (fixed before running):
 A1  Eq. 2 matches the pump: median |Eq.2 - refuel| / refuel < 6% over refuel intervals.
 A2  Eq. 1 undercounts: median (Eq.1 - refuel) / refuel < -5%, and Eq. 2 is closer to the pump than Eq. 1 in at least
     75% of intervals.
 B1  Winter (Dec-Feb) trip-level LTFT exceeds summer (Jun-Aug) by > 0.5 within-car trip-level SD (the direction seen
     in VED: trims fall in summer); reported with a bootstrap interval over trips.
 C1  (descriptive) Spearman(air-flow bin median, LTFT bin median) < 0 indicates an unmetered-air profile.
 INVARIANT  Integration sanity: the summed Eq. 2 litres over all intervals are within 15% of the summed pump litres
     (a gross time-step or unit error would break this by a large factor).
Output: data/processed/p41_opel_panel.json
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); D = os.path.join(ROOT, "data", "raw", "opel_corsa")
OUTF = os.path.join(ROOT, "data", "processed", "p41_opel_panel.json"); rng = np.random.default_rng(0)
cols = ["timestamp", "SPEED", "MAF", "SHORT_FUEL_TRIM_1", "LONG_FUEL_TRIM_1", "COOLANT_TEMP", "INTAKE_TEMP", "RPM", "segment_id", "segment_file"]
d = pd.read_csv(os.path.join(D, "dataset.csv"), usecols=cols, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
rf = pd.read_csv(os.path.join(D, "refueling_history.csv"))
R = {"n_rows": int(len(d)), "span": [str(d.timestamp.min()), str(d.timestamp.max())], "n_refuels": int(len(rf))}

# ---- A: integrate fuel per segment ----
dt = d.groupby("segment_file").timestamp.diff().dt.total_seconds().fillna(0).clip(0, 5)   # cap logging gaps at 5 s
trim = (d.SHORT_FUEL_TRIM_1.fillna(0) + d.LONG_FUEL_TRIM_1.fillna(0)) / 100
eq1 = d.MAF.fillna(0) / 14.7 / 745.0 * dt; eq2 = eq1 * (1 + trim)
seg = pd.DataFrame({"segment_id": d.segment_id, "eq1": eq1, "eq2": eq2, "dist": d.SPEED.fillna(0) * dt / 3600}).groupby("segment_id").sum()
# The refuel table numbers its intervals differently from the log; pick the offset at which logged distance matches the
# recorded distance (fix after the first run compared each tank with the wrong driving: logged km of data segment k
# equalled the record's Total_km at Segment_id k + 2).
def joined(off):
    return rf.assign(seg_key=rf.Segment_id - off).merge(seg, left_on="seg_key", right_index=True, how="inner")
offs = {o: float((joined(o).dist / joined(o).Total_km - 1).abs().median()) for o in range(-4, 5) if len(joined(o)) >= 10}
OFF = min(offs, key=offs.get); A = joined(OFF)
R["segment_offset"] = {"chosen": OFF, "median_abs_km_mismatch_by_offset": {str(k): round(v, 4) for k, v in offs.items()}}
A["e1"] = (A.eq1 - A.Refueling) / A.Refueling; A["e2"] = (A.eq2 - A.Refueling) / A.Refueling; A["etr"] = (A.Trapez_l - A.Refueling) / A.Refueling
R["A"] = {"n_intervals": int(len(A)), "median_err_eq1_pct": round(float(A.e1.median() * 100), 2), "median_err_eq2_pct": round(float(A.e2.median() * 100), 2),
          "median_abs_err_eq2_pct": round(float(A.e2.abs().median() * 100), 2), "median_abs_err_author_trapez_pct": round(float(A.etr.abs().median() * 100), 2),
          "frac_eq2_closer": round(float((A.e2.abs() < A.e1.abs()).mean()), 3), "median_abs_km_mismatch": round(float((A.dist / A.Total_km - 1).abs().median()), 4),
          "sum_eq2_over_sum_refuel": round(float(A.eq2.sum() / A.Refueling.sum()), 3), "sum_eq1_over_sum_refuel": round(float(A.eq1.sum() / A.Refueling.sum()), 3),
          "km_ratio_logged_vs_record": round(float(A.dist.sum() / A.Total_km.sum()), 3),
          "per_interval": A[["Segment_id", "Refueling", "eq1", "eq2", "Trapez_l", "dist", "Total_km"]].round(2).to_dict(orient="records")}
json.dump(R, open(OUTF, "w"), indent=2, default=str); print("A", {k: v for k, v in R["A"].items() if k != "per_interval"}, flush=True)

# ---- B: seasonal drift, warm engine, trip level ----
w = d[(d.COOLANT_TEMP > 75) & (d.RPM > 600)].copy()
T = w.groupby("segment_file").agg(ltft=("LONG_FUEL_TRIM_1", "mean"), intake=("INTAKE_TEMP", "median"), t=("timestamp", "median"), n=("RPM", "size"))
T = T[T.n >= 120].copy(); T["m"] = T.t.dt.month
win, summ = T[T.m.isin([12, 1, 2])].ltft.to_numpy(), T[T.m.isin([6, 7, 8])].ltft.to_numpy(); sd = T.ltft.std()
bs = [(rng.choice(win, len(win)).mean() - rng.choice(summ, len(summ)).mean()) / sd for _ in range(2000)]
R["B"] = {"n_trips": int(len(T)), "within_car_trip_sd_pp": round(float(sd), 2), "winter_mean": round(float(win.mean()), 2), "summer_mean": round(float(summ.mean()), 2),
          "winter_minus_summer_sd": round(float((win.mean() - summ.mean()) / sd), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
          "spearman_ltft_vs_iat": round(float(spearmanr(T.ltft, T.intake).statistic), 3), "n_winter": int(len(win)), "n_summer": int(len(summ))}
json.dump(R, open(OUTF, "w"), indent=2, default=str); print("B", R["B"], flush=True)

# ---- C: LTFT across air-flow bins ----
w["mafbin"] = pd.qcut(w.MAF, 10, duplicates="drop")
C = w.groupby("mafbin", observed=True).agg(maf=("MAF", "median"), ltft=("LONG_FUEL_TRIM_1", "median"), stft=("SHORT_FUEL_TRIM_1", "median"))
R["C"] = {"bins": C.round(2).reset_index(drop=True).to_dict(orient="records"), "spearman_maf_ltft": round(float(spearmanr(C.maf, C.ltft).statistic), 3)}
a = R["A"]
R["checks"] = {"A1_eq2_median_abs_lt_6pct": a["median_abs_err_eq2_pct"] < 6, "A2_eq1_under_5pct": a["median_err_eq1_pct"] < -5,
               "A2_eq2_closer_75pct": a["frac_eq2_closer"] >= 0.75, "B1_winter_gt_summer_0.5sd": R["B"]["winter_minus_summer_sd"] > 0.5,
               "C1_unmetered_air_profile": R["C"]["spearman_maf_ltft"] < 0, "INV_sum_eq2_within_15pct": abs(a["sum_eq2_over_sum_refuel"] - 1) < 0.15, "INV_km_match_lt_3pct": a["median_abs_km_mismatch"] < 0.03}
json.dump(R, open(OUTF, "w"), indent=2, default=str); print("C", R["C"]["spearman_maf_ltft"], R["C"]["bins"]); print(json.dumps(R["checks"], indent=1))
