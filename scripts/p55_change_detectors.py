"""P55: conventional PHM change detectors on the long-term trim (IJPHM review R1, R2).
Real VED trim sequences (trips in order, vehicles with >= 45 trim-covered trips). Each trip's trim is standardized
against the same vehicle's previous 20 trips (causal), z_t. Three detectors of an upward step:
  ROLL  alarm when z_t > h (the paper's per-trip rule, one-sided)
  EWMA  e_t = lam z_t + (1 - lam) e_{t-1}, lam = 0.3, alarm when e_t > h, reset after an alarm
  CUSUM S_t = max(0, S_{t-1} + z_t - k), k = 0.5, alarm when S_t > h, reset after an alarm
Each h is set so that unperturbed sequences give 2 false alarms per 100 trips (vehicle-level alarm budget).
Detection: a +s pp step added from a random trip t0 (>= 25 trips in, >= 10 trips left); report the probability of an
alarm within 1, 3 and 5 trips and the median delay. Steps: 3.49 (anchor), 6 and 8.0 pp.
Invariant: with no step, the alarm rate equals the 2-per-100 budget by construction (checked on held-out vehicles).
Output: data/processed/p55_change_detectors.json"""
import os, io, json, contextlib, runpy, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v = g["v"].sort_values(["VehId", "Trip"])
seqs = [s["ltft1_mean"].to_numpy() for _, s in v.groupby("VehId") if len(s) >= 45]
def zseq(L):
    z = np.full(len(L), np.nan)
    for t in range(20, len(L)):
        r = L[t - 20:t]; z[t] = (L[t] - r.mean()) / (r.std() + 1e-9)
    return z
def run(z, kind, h):
    al = np.zeros(len(z), bool); s = 0.0
    for t in range(len(z)):
        if np.isnan(z[t]): continue
        if kind == "ROLL": al[t] = z[t] > h; continue
        s = 0.3 * z[t] + 0.7 * s if kind == "EWMA" else max(0.0, s + z[t] - 0.5)
        if s > h: al[t] = True; s = 0.0
    return al
idx = rng.permutation(len(seqs)); cal_i, te_i = idx[: len(idx) // 2], idx[len(idx) // 2:]
Z = [zseq(L) for L in seqs]
def rate(kind, h, ids): a = np.concatenate([run(Z[i], kind, h) for i in ids]); m = np.concatenate([~np.isnan(Z[i]) for i in ids]); return a[m].mean()
R = {"vehicles": len(seqs), "trips_scored": int(sum((~np.isnan(z)).sum() for z in Z))}
H = {}
for kind, grid in (("ROLL", np.linspace(1.0, 4.0, 121)), ("EWMA", np.linspace(0.3, 3.0, 109)), ("CUSUM", np.linspace(0.5, 12.0, 116))):
    rates = np.array([rate(kind, h, cal_i) for h in grid]); H[kind] = float(grid[np.argmin(np.abs(rates - 0.02))])
    R[kind] = {"h": round(H[kind], 3), "false_alarms_per_100_trips_heldout": round(100 * rate(kind, H[kind], te_i), 2)}
REPS = 50                                             # onsets per held-out vehicle, shared by all detectors (paired)
T0 = {(i, r): int(rng.integers(25, len(seqs[i]) - 10)) for i in te_i for r in range(REPS)}
R["onsets_per_vehicle"] = REPS
for s in (3.49, 6.0, 8.0):
    for kind in ("ROLL", "EWMA", "CUSUM"):
        delays = []
        for i in te_i:
            L = seqs[i]
            for r in range(REPS):
                t0 = T0[(i, r)]; Ls = L.copy(); Ls[t0:] += s
                al = run(zseq(Ls), kind, H[kind]); hit = np.flatnonzero(al[t0:])
                delays.append(hit[0] + 1 if len(hit) else np.inf)
        d = np.array(delays)
        R[kind][f"step_{s}"] = {"P_within_1": round(float(np.mean(d <= 1)), 3), "P_within_3": round(float(np.mean(d <= 3)), 3),
                                "P_within_5": round(float(np.mean(d <= 5)), 3), "median_delay_trips": float(np.median(d))}
json.dump(R, open(os.path.join(ROOT, "data", "processed", "p55_change_detectors.json"), "w"), indent=1); print(json.dumps(R, indent=1))
