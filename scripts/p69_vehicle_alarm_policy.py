"""P69: a vehicle-level alarm policy on the trim axis, over 200 vehicle splits at matched false-alarm rates.
Real VED trim sequences (trips in order, vehicles with >= 45 trim-covered trips, as p55). Each trip's trim is
standardized against the same vehicle's previous 20 trips (causal). A trip is FLAGGED when z_t > h; the vehicle is
ALARMED when k of its last n trips are flagged, and the window is cleared after an alarm. Policies (k of n): 1 of 1
(the per-trip rule), 2 of 3, 2 of 5, 3 of 5.
False alarms are counted per scored trip on unchanged sequences and converted to a CALENDAR vehicle-year with the
panel trip rate (all trips of these vehicles divided by their observed calendar span), since a deployed monitor scores
every trip after its first 20.
For each of 200 random half splits of the vehicles and each policy:
  (a) deployable: h calibrated on the calibration half to ONE false alarm per vehicle-year; the held-out half gives
      the realized rate and the detection probabilities;
  (b) matched: h set so that the held-out realized rate is one per vehicle-year for every policy, so that detection
      is compared at the same realized false-alarm rate.
Detection: an upward step of 3.49, 6 or 10 pp from a random trip (50 onsets per vehicle, shared by all policies and
splits): probability of an alarm within 3, 5 and 10 trips, and the median delay.
Invariants (asserted): the vectorized simulation reproduces the scalar rule; the matched realized rate is within
0.15 of the target; detection does not decrease with step size or with the horizon.
Output: data/processed/p69_vehicle_alarm_policy.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v = g["v"].sort_values(["VehId", "Trip"]).copy()
v["date"] = pd.to_datetime(v["week"].str.extract(r"VED_(\d{6})_")[0], format="%y%m%d")
seqs, years = [], []
for _, s in v.groupby("VehId"):
    if len(s) >= 45:
        seqs.append(s["ltft1_mean"].to_numpy()); years.append(max((s.date.max() - s.date.min()).days + 7, 7) / 365.25)
years = np.array(years); V = len(seqs)
TRIPS_YEAR = float(sum(len(L) for L in seqs) / years.sum())            # calendar trip rate
SCORED_YEAR = float(sum(len(L) - 20 for L in seqs) / years.sum())      # scored trips per observed year (first 20 unscored)
GRID = np.round(np.arange(0.3, 5.0001, 0.025), 3); H = len(GRID)
POL = ((1, 1), (2, 3), (2, 5), (3, 5)); STEPS = (3.49, 6.0, 10.0); REPS = 50; NSPLIT = 200; TARGET = 1.0
def zseq(L):
    z = np.full(len(L), np.nan)
    for t in range(20, len(L)):
        r = L[t - 20:t]; z[t] = (L[t] - r.mean()) / (r.std() + 1e-9)
    return z
def alarms_scalar(z, h, k, n):
    al = np.zeros(len(z), bool); buf = []
    for t in range(len(z)):
        if np.isnan(z[t]): continue
        buf.append(z[t] > h); buf = buf[-n:]
        if sum(buf) >= k: al[t] = True; buf = []
    return al
def alarms_grid(z, k, n):
    """Alarm indicator for every threshold of GRID at once: array (T, H)."""
    al = np.zeros((len(z), H), bool); buf = np.zeros((H, n), bool)
    for t in range(len(z)):
        if np.isnan(z[t]): continue
        if n > 1: buf[:, :-1] = buf[:, 1:]
        buf[:, -1] = z[t] > GRID; a = buf.sum(1) >= k; al[t] = a; buf[a] = False
    return al
Z = [zseq(L) for L in seqs]; scored = np.array([(~np.isnan(z)).sum() for z in Z])
# invariant: vectorized simulation equals the scalar rule
for (k, n) in POL:
    for i in (0, V // 2, V - 1):
        A = alarms_grid(Z[i], k, n)
        for j in (20, 60, 100): assert (A[:, j] == alarms_scalar(Z[i], GRID[j], k, n)).all()
# false alarms on unchanged sequences: FA[p, vehicle, h]
FA = np.zeros((len(POL), V, H))
for p, (k, n) in enumerate(POL):
    for i in range(V): FA[p, i] = alarms_grid(Z[i], k, n).sum(0)
print("false-alarm counts done", flush=True)
# detection delays: D[p, step, vehicle, rep, h] (inf when no alarm before the sequence ends)
T0 = np.array([[int(rng.integers(25, len(seqs[i]) - 12)) for r in range(REPS)] for i in range(V)])
D = np.full((len(POL), len(STEPS), V, REPS, H), np.inf, dtype=np.float32)
for i in range(V):
    L = seqs[i]
    for r in range(REPS):
        t0 = T0[i, r]
        for si, s in enumerate(STEPS):
            Ls = L.copy(); Ls[t0:] += s; zs = zseq(Ls)
            for p, (k, n) in enumerate(POL):
                A = alarms_grid(zs, k, n)[t0:]; hit = A.any(0); D[p, si, i, r, hit] = A[:, hit].argmax(0) + 1
    if i % 10 == 0: print("delays vehicle", i, "of", V, flush=True)
def rate(p, ids): return FA[p][ids].sum(0) / scored[ids].sum() * TRIPS_YEAR          # per calendar vehicle-year, for each h
def summ(a):
    a = np.asarray(a, float); return {"median": round(float(np.median(a)), 3), "p05": round(float(np.quantile(a, 0.05)), 3), "p95": round(float(np.quantile(a, 0.95)), 3),
                                      "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}
rec = {m: {p: {"rate": [], "h": [], **{f"s{si}_w{w}": [] for si in range(len(STEPS)) for w in (3, 5, 10)}, **{f"s{si}_med": [] for si in range(len(STEPS))}} for p in range(len(POL))} for m in ("calibrated", "matched")}
for sp in range(NSPLIT):
    r = np.random.default_rng(1000 + sp); idx = r.permutation(V); cal, te = idx[: V // 2], idx[V // 2:]; assert not set(cal) & set(te)
    for p in range(len(POL)):
        for mode, ids in (("calibrated", cal), ("matched", te)):
            j = int(np.argmin(np.abs(rate(p, ids) - TARGET))); R = rec[mode][p]; R["rate"].append(rate(p, te)[j]); R["h"].append(GRID[j])
            for si in range(len(STEPS)):
                d = D[p, si][te][:, :, j].ravel(); R[f"s{si}_med"].append(float(np.median(d)))
                for w in (3, 5, 10): R[f"s{si}_w{w}"].append(float(np.mean(d <= w)))
out = {"vehicles": V, "held_out_vehicles": V - V // 2, "splits": NSPLIT, "onsets_per_vehicle": REPS, "target_false_alarms_per_vehicle_year": TARGET,
       "trips_per_calendar_vehicle_year": round(TRIPS_YEAR, 1), "scored_trips_per_observed_vehicle_year": round(SCORED_YEAR, 1)}
for mode in ("calibrated", "matched"):
    out[mode] = {}
    for p, (k, n) in enumerate(POL):
        R = rec[mode][p]; o = {"false_alarms_per_vehicle_year": summ(R["rate"]), "flag_threshold_z": summ(R["h"])}
        for si, s in enumerate(STEPS):
            o[f"step_{s}"] = {**{f"P_within_{w}": summ(R[f"s{si}_w{w}"]) for w in (3, 5, 10)}}
            med = np.array(R[f"s{si}_med"]); fin = np.isfinite(med)
            o[f"step_{s}"]["median_delay_trips_median_over_splits"] = float(np.median(med)) if fin.mean() > 0.5 else None
            o[f"step_{s}"]["median_delay_days"] = round(float(np.median(med)) / TRIPS_YEAR * 365.25, 1) if fin.mean() > 0.5 else None
            # invariants: detection does not decrease with the horizon
            assert all(a <= b + 1e-12 for a, b in zip(R[f"s{si}_w3"], R[f"s{si}_w5"])) and all(a <= b + 1e-12 for a, b in zip(R[f"s{si}_w5"], R[f"s{si}_w10"]))
        for w in (3, 5, 10):   # and with step size
            assert all(a <= b + 1e-12 and b <= c + 1e-12 for a, b, c in zip(R[f"s0_w{w}"], R[f"s1_w{w}"], R[f"s2_w{w}"]))
        out[mode][f"{k}_of_{n}"] = o
    print(mode, {f"{k}_of_{n}": (out[mode][f"{k}_of_{n}"]["false_alarms_per_vehicle_year"]["median"], out[mode][f"{k}_of_{n}"]["step_3.49"]["P_within_5"]["median"]) for k, n in POL}, flush=True)
for p in range(len(POL)): assert abs(np.median(rec["matched"][p]["rate"]) - TARGET) < 0.15, (p, np.median(rec["matched"][p]["rate"]))
# paired gain of each persistence rule over the per-trip rule, at the matched realized rate
out["matched_gain_over_per_trip"] = {}
for p, (k, n) in enumerate(POL[1:], 1):
    out["matched_gain_over_per_trip"][f"{k}_of_{n}"] = {}
    for si, s in enumerate(STEPS):
        for w in (5, 10):
            gdiff = np.array(rec["matched"][p][f"s{si}_w{w}"]) - np.array(rec["matched"][0][f"s{si}_w{w}"])
            out["matched_gain_over_per_trip"][f"{k}_of_{n}"][f"step_{s}_within_{w}"] = {**summ(gdiff), "share_of_splits_positive": round(float(np.mean(gdiff > 0)), 3)}
json.dump(out, open(os.path.join(ROOT, "data", "processed", "p69_vehicle_alarm_policy.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("vehicles", "splits", "trips_per_calendar_vehicle_year", "scored_trips_per_observed_vehicle_year")}))
print(json.dumps(out["matched_gain_over_per_trip"]["3_of_5"], indent=1))
