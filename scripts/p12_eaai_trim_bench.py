"""P12 (EAAI revision, trim + bench):
E2  fuel-trim semantics: STFT vs LTFT separately; additive vs multiplicative correction.
E7  bench blocked CV (leave-operating-region-out) vs shuffled; deployable-subset fault R2.
E8  behaviour counterfactual sensitivity to reference percentile (10/20/30).
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
HGB=lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0)
R={}

# ================= E2: trim semantics (VED) =================
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
vt=v[v["ltft1_cov"]>=0.2].dropna(subset=["stft1_mean","ltft1_mean"]).copy()
gt=vt["VehId"].to_numpy()
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
def excess(col):
    yy=vt[col].to_numpy(); return yy-cross_val_predict(HGB(),vt[CTX],yy,cv=GroupKFold(5),groups=gt)
def r2x(X,z):
    p=cross_val_predict(HGB(),X,z,cv=GroupKFold(5),groups=gt); return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
ex=excess("fuel_per_100km")
R["E2_combR2_STFT_only"]=round(r2x(vt[["stft1_mean"]].to_numpy(),ex),3)
R["E2_combR2_LTFT_only"]=round(r2x(vt[["ltft1_mean"]].to_numpy(),ex),3)
R["E2_combR2_both"]=round(r2x(vt[["stft1_mean","ltft1_mean"]].to_numpy(),ex),3)
# additive vs multiplicative trim-corrected fuel -> combustion R2 on driver-excess
vt["fc_add"]=vt["fuel_per_100km"]*(1+(vt["stft1_mean"]+vt["ltft1_mean"])/100.0)
vt["fc_mul"]=vt["fuel_per_100km"]*(1+vt["stft1_mean"]/100.0)*(1+vt["ltft1_mean"]/100.0)
R["E2_combR2_driverexcess_additive"]=round(r2x(vt[["stft1_mean","ltft1_mean"]].to_numpy(),excess("fc_add")),3)
R["E2_combR2_driverexcess_multipl"]=round(r2x(vt[["stft1_mean","ltft1_mean"]].to_numpy(),excess("fc_mul")),3)
R["E2_median_abs_diff_add_mul_pct"]=round(float((vt["fc_add"]-vt["fc_mul"]).abs().median()),4)

# ================= E7: bench blocked CV + deployable subset =================
e=pd.read_csv(os.path.join(RAW,"EngineFaultDB_Final.csv"))
OP=["RPM","MAP","TPS","Force","Power","Speed"]; norm=e[e.Fault==0]
M=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=6,random_state=0).fit(
    norm[OP].to_numpy(),norm["Consumption L/100KM"].to_numpy())
e["excess"]=e["Consumption L/100KM"].to_numpy()-M.predict(e[OP].to_numpy())
rich=e[e.Fault==1].copy(); z=rich["excess"].to_numpy()
ALL=["AFR","Lambda","CO","HC"]; DEPLOY=["AFR","Lambda","O2"]   # O2/lambda ~ wideband sensor; CO/HC = bench analyzer
def r2_shuf(X):
    p=cross_val_predict(HGB(),X,z,cv=KFold(5,shuffle=True,random_state=0)); return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
def r2_block(X):
    # leave-operating-region-out: bin by RPM x Power terciles -> 9 groups
    rb=pd.qcut(rich["RPM"].rank(method="first"),3,labels=False)
    pb=pd.qcut(rich["Power"].rank(method="first"),3,labels=False)
    grp=(rb*3+pb).to_numpy()
    p=cross_val_predict(HGB(),X,z,cv=GroupKFold(5),groups=grp); return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
R["E7_bench_R2_all_shuffled"]=round(r2_shuf(rich[ALL].to_numpy()),3)
R["E7_bench_R2_all_blocked"]=round(r2_block(rich[ALL].to_numpy()),3)
R["E7_bench_R2_deploy_shuffled"]=round(r2_shuf(rich[DEPLOY].to_numpy()),3)
R["E7_bench_R2_deploy_blocked"]=round(r2_block(rich[DEPLOY].to_numpy()),3)

# ================= E8: counterfactual percentile sensitivity =================
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
AGG=["accel_pos_mean","accel_p95","jerk_rms","harsh_accel_per_km","harsh_brake_per_km","speed_std"]
y=v["fuel_per_100km"].to_numpy()
m=HGB().fit(v[BEHAV+ENV].to_numpy(),y)
base=m.predict(v[BEHAV+ENV].to_numpy())
for q in [0.10,0.20,0.30]:
    cf=v[BEHAV+ENV].copy()
    for c in AGG: cf[c]=v[c].quantile(q)
    comp=(base-m.predict(cf.to_numpy())).clip(min=0)
    R[f"E8_behavcomp_median_q{int(q*100)}"]=round(float(np.median(comp)),3)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p12_eaai_trim_bench.json"),"w"),indent=2)
print("wrote p12_eaai_trim_bench.json")
