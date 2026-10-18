"""P38: bank-1 vs bank-2 long-term-trim divergence in VED as natural, within-car fueling events.

Season, fuel blend, altitude and driving move both cylinder banks' trims together; a bank-specific split points to
a fueling-path event on one bank (a bank leak, an injector, an oxygen sensor). On vehicles that log both banks,
weeks with a large split are compared with the same vehicle's weeks with almost no split. The driver is the same
person in both, so the comparison asks whether driving changes (it should not) and what the fuel does.

Event week: weekly median |LTFT1 - LTFT2| >= 5 pp. Control week: same vehicle, weekly median < 1.5 pp.
Vehicles need >= 1 event week and >= 1 control week. Trip outcomes come from the shared injection panel
(attrib_setup: context-residual excess, frozen kinematic score), joined on (VehId, Trip).

PRE-REGISTERED (fixed before running):
 H1  Driving is unchanged: vehicle-mean (event - control) difference of the standardized frozen kinematic score is
     equivalent to zero within |d| < 0.2 (TOST over the vehicle-cluster bootstrap, p < 0.05).
 H2  Fuel: the (event - control) excess difference is reported with a vehicle-cluster 95% interval; prediction under
     closed-loop compensation: |difference| < 0.3 L/100km.
 INVARIANT (placebo)  Relabelling each vehicle's weeks at random (same counts) gives kinematic and excess
     differences whose 95% band contains 0.
Output: data/processed/p38_bank_divergence.json
"""
import os, io, glob, json, contextlib, runpy
import numpy as np, pandas as pd
from scipy.stats import norm as N
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); RAW = os.path.join(ROOT, "data", "raw")
OUTF = os.path.join(ROOT, "data", "processed", "p38_bank_divergence.json"); CACHE = os.path.join(ROOT, "data", "processed", "p38_trip_banks.parquet")
rng = np.random.default_rng(0); R = {}

if not os.path.exists(CACHE):
    parts = []
    for f in sorted(glob.glob(os.path.join(RAW, "VED_*_week.csv"))):
        cols = ["VehId", "Trip", "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]"]
        d = pd.read_csv(f, usecols=cols, dtype={c: "float32" for c in cols[2:]})
        d = d[d[cols[3]].notna() & d[cols[2]].notna()]
        if len(d): parts.append(d.groupby(["VehId", "Trip"]).agg(l1=(cols[2], "mean"), l2=(cols[3], "mean"), nb=(cols[3], "size")).reset_index())
        print("read", os.path.basename(f), flush=True)
    pd.concat(parts).groupby(["VehId", "Trip"]).agg(l1=("l1", "mean"), l2=("l2", "mean"), nb=("nb", "sum")).reset_index().to_parquet(CACHE)
B = pd.read_parquet(CACHE)

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(ROOT, "scripts", "attrib_setup.py"))
v, Zv, Wk = g["v"], g["Zv"], g["Wk"]
v = v.assign(kz=(Zv @ Wk - (Zv @ Wk).mean()) / (Zv @ Wk).std())
T = v[["VehId", "Trip", "week", "excess", "kz"]].merge(B, on=["VehId", "Trip"], how="inner")
T = T[T.nb >= 10].copy(); T["div"] = (T.l1 - T.l2).abs()
wk = T.groupby(["VehId", "week"]).agg(div=("div", "median"), n=("div", "size")).reset_index()
wk["lab"] = np.where(wk["div"] >= 5, "event", np.where(wk["div"] < 1.5, "control", "mid"))
T = T.merge(wk[["VehId", "week", "lab"]], on=["VehId", "week"])
keep = T.groupby("VehId").lab.agg(lambda s: ("event" in set(s)) and ("control" in set(s)))
T = T[T.VehId.isin(keep[keep].index) & T.lab.isin(["event", "control"])].copy()
R["n_vehicles"] = int(T.VehId.nunique()); R["n_trips"] = {k: int(x) for k, x in T.lab.value_counts().items()}
R["two_bank_trips_total"] = int(len(B)); R["median_div_event_control"] = T.groupby("lab")["div"].median().round(2).to_dict()

def veh_diffs(df, col):
    m = df.groupby(["VehId", "lab"])[col].mean().unstack(); return (m["event"] - m["control"]).dropna()
def boot(df, col, B_=2000):
    dv = veh_diffs(df, col).to_numpy(); bs = np.array([rng.choice(dv, len(dv)).mean() for _ in range(B_)])
    return float(dv.mean()), bs
out = {}
for col in ("kz", "excess"):
    mu, bs = boot(T, col); se = bs.std()
    out[col] = {"mean_diff": round(mu, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    if col == "kz": out[col]["tost_p_margin0.2"] = float(max(N.sf((mu + 0.2) / se), N.cdf((mu - 0.2) / se)))
R["diffs"] = out
# placebo: shuffle labels within vehicle
pl = {"kz": [], "excess": []}
for _ in range(500):
    S = T.copy(); S["lab"] = S.groupby("VehId").lab.transform(lambda s: rng.permutation(s.to_numpy()))
    for col in pl: pl[col].append(veh_diffs(S, col).mean())
R["placebo"] = {c: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)] for c, x in pl.items()}
R["checks"] = {"H1_kinematic_equivalent": out["kz"]["tost_p_margin0.2"] < 0.05,
               "H2_excess_abs_lt_0.3": abs(out["excess"]["mean_diff"]) < 0.3,
               "invariant_placebo_contains_0": all(lo <= 0 <= hi for lo, hi in R["placebo"].values())}
json.dump(R, open(OUTF, "w"), indent=2); print(json.dumps(R, indent=1))
