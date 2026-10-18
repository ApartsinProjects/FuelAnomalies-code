"""P47 root-cause inspection of EngineFaultDB at the operating-condition level (1,000-reading blocks).
Q1 do labels match the mixture? Q2 is fuel excess explained by mixture regardless of label (physics: at fixed air,
fuel scales with 14.7/AFR, so 1 pp richer -> ~1% more fuel)? Q3 how much of the rich excess comes from conditions
outside the normal model's operating range (extrapolation)? Invariant: pooled slope of excess% on fuel-per-air
deviation must be positive if the data are physically consistent."""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = pd.read_csv(os.path.join(ROOT, "data", "raw", "EngineFaultDB_Final.csv"))
OP = ["RPM", "MAP", "TPS", "Force", "Power", "Speed"]; T = "Consumption L/100KM"
n0 = d[d.Fault == 0]
M = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0).fit(n0[OP], n0[T])
lo, hi = n0[OP].quantile(0.01), n0[OP].quantile(0.99)
rows = []
for k in (0, 1, 2):
    s = d[d.Fault == k].reset_index(drop=True); b = np.arange(len(s)) // 1000
    for i in np.unique(b):
        x = s[b == i]; inside = float(((x[OP] >= lo) & (x[OP] <= hi)).all(1).mean())
        pred = M.predict(x[OP]) if k else None
        rows.append(dict(F=k, blk=i, fuel=x[T].mean(), afr=x.AFR.mean(), co=x.CO.mean(), power=x.Power.mean(), inside=inside,
                         exc_pct=((x[T].mean() / pred.mean() - 1) * 100) if k else np.nan))
C = pd.DataFrame(rows); C["fpa"] = (14.7 / C.afr - 1) * 100; C["fpa_dev"] = C.fpa - C[C.F == 0].fpa.mean()
print("Q1 conditions with lambda-rich mixture (AFR < 12.5): normal %d/16, rich %d/11, lean %d/15" % tuple((C[C.F == k].afr < 12.5).sum() for k in (0, 1, 2)))
print("   fuel-per-air dev by class, per condition:"); [print("  ", k, np.round(np.sort(C[C.F == k].fpa_dev.to_numpy()), 1)) for k in (0, 1, 2)]
F = C[C.F > 0]
print("Q2 fault conditions: Spearman(excess%%, fuel-per-air dev) = %.2f (n=%d)" % (spearmanr(F.exc_pct, F.fpa_dev)[0], len(F)))
In = F[F.inside > 0.9]
print("   inside normal operating range (%d conditions): Spearman %.2f, OLS slope %.2f" % (len(In), spearmanr(In.exc_pct, In.fpa_dev)[0], np.polyfit(In.fpa_dev, In.exc_pct, 1)[0]))
print("Q3 rich conditions:"); print(C[C.F == 1][["blk", "power", "inside", "afr", "fpa_dev", "exc_pct"]].round(2).to_string(index=False))
print("   rich mean excess%% all %.1f, inside-range only %.1f" % (C[C.F == 1].exc_pct.mean(), C[(C.F == 1) & (C.inside > 0.9)].exc_pct.mean()))
C.to_csv(os.path.join(ROOT, "data", "processed", "p47_bench_conditions.csv"), index=False)

# Q4 bug check: the paper's "out-of-fold residual SD 0.20 L/100km" used shuffled KFold, which puts readings of the SAME
# condition in train and test (near-duplicates). Leave-one-condition-out gives the true error on an unseen condition.
nb = np.arange(len(n0)) // 1000; X0, y0 = n0[OP].to_numpy(), n0[T].to_numpy(); errs = []
for i in np.unique(nb):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=0).fit(X0[nb != i], y0[nb != i])
    errs.append((y0[nb == i].mean() / m.predict(X0[nb == i]).mean() - 1) * 100)
errs = np.array(errs)
print("Q4 normal leave-one-condition-out error per condition (%):", np.round(np.sort(errs), 1))
print("   SD %.1f%%, mean |err| %.1f%%  (vs shuffled-KFold SD 0.20 L/100km = ~2.5%%)" % (errs.std(), np.abs(errs).mean()))
