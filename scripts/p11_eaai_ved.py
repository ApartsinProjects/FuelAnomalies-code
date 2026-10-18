"""P11 (EAAI revision, VED side):
E1  cross-fitted vehicle-baseline variance share (held-out, split-half, shrinkage, min-trip sweep)
E9  model comparison for expected fuel (mean, ridge, RF, HGB)
E10 diagnostic-feature leakage ablation (operational vs max-prediction model)
Lean: n_jobs=1.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

ROOT=os.path.join(os.path.dirname(__file__),"..")
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
DIAG=["rpm_mean","load_mean"]
y=v["fuel_per_100km"].to_numpy(); g=v["VehId"].to_numpy(); SS=np.sum((y-y.mean())**2)
HGB=lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0)
def oofpred(model,feats):
    return cross_val_predict(model,v[feats],y,cv=GroupKFold(5),groups=g)
def r2(p): return 1-np.sum((y-p)**2)/SS

R={}
# ---- E9 model comparison (expected fuel, behav+env) ----
F=BEHAV+ENV
imp=lambda m:make_pipeline(SimpleImputer(strategy="median"),m)
models={"mean":DummyRegressor(),"ridge":imp(Ridge(alpha=1.0)),
        "rf":imp(RandomForestRegressor(n_estimators=60,max_depth=14,max_samples=0.5,random_state=0,n_jobs=1)),"hgb":HGB()}
R["E9_model_r2"]={k:round(r2(oofpred(m,F)),3) for k,m in models.items()}

# ---- E10 leakage ablation ----
pred_op=oofpred(HGB(),BEHAV+ENV)                 # operational (no diagnostic)
pred_mx=oofpred(HGB(),BEHAV+ENV+DIAG)            # max-prediction (with rpm/load)
R["E10_r2_operational"]=round(r2(pred_op),3)
R["E10_r2_maxpred_with_rpm_load"]=round(r2(pred_mx),3)
# how much vehicle-baseline survives under each (crude, in-sample expanded)
def veh_share_of(pred):
    resid=y-pred; fe=pd.Series(resid).groupby(g).transform("mean").to_numpy()
    return np.sum((fe-resid.mean())**2)/SS
R["E10_vehshare_operational"]=round(veh_share_of(pred_op),3)
R["E10_vehshare_maxpred"]=round(veh_share_of(pred_mx),3)

# ---- E1 cross-fitted vehicle baseline ----
resid=y-pred_op
ybar=y.mean()
d=pd.DataFrame({"VehId":g,"resid":resid,"y":y})
d=d.reset_index(drop=True)
# naive in-sample share (for reference)
fe_all=d.groupby("VehId")["resid"].transform("mean").to_numpy()
R["E1_vehshare_insample"]=round(np.sum((fe_all-resid.mean())**2)/SS,3)
# split-half cross-fit: estimate v_j on half A, evaluate explained fuel variance on half B
rng=np.random.default_rng(0)
def crossfit_share(min_trips):
    num=0.0; den=0.0
    for vid,grp in d.groupby("VehId"):
        idx=grp.index.to_numpy()
        if len(idx)<min_trips: continue
        rng.shuffle(idx); half=len(idx)//2
        A,B=idx[:half],idx[half:]
        vj=d.loc[A,"resid"].mean()               # estimate offset on A
        rb=d.loc[B,"resid"].to_numpy(); yb=d.loc[B,"y"].to_numpy()
        # fuel variance uniquely explained by adding vj on held-out B
        num+=np.sum(rb**2)-np.sum((rb-vj)**2)
        den+=np.sum((yb-ybar)**2)
    return num/den if den>0 else np.nan
for mt in [5,10,15,20]:
    R[f"E1_crossfit_share_min{mt}"]=round(float(crossfit_share(mt)),3)
# split-half reliability of the baseline (correlation of v_j between two halves)
va,vb=[],[]
for vid,grp in d.groupby("VehId"):
    idx=grp.index.to_numpy()
    if len(idx)<10: continue
    rng.shuffle(idx); h=len(idx)//2
    va.append(d.loc[idx[:h],"resid"].mean()); vb.append(d.loc[idx[h:],"resid"].mean())
rel,_=pearsonr(va,vb)
R["E1_splithalf_reliability_r"]=round(float(rel),3); R["E1_nveh_ge10"]=len(va)
# empirical-Bayes shrinkage share (James-Stein-ish): shrink v_j toward 0 by reliability
# report shrunk between-vehicle share
vj_full=d.groupby("VehId")["resid"].mean()
nj=d.groupby("VehId").size()
within_var=d.groupby("VehId")["resid"].var().mean()
shrink=nj/(nj+within_var/max(vj_full.var(),1e-9))
vj_shr=vj_full*shrink
fe_shr=d["VehId"].map(vj_shr).to_numpy()
R["E1_vehshare_shrinkage"]=round(np.sum((fe_shr-fe_shr.mean())**2)/SS,3)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p11_eaai_ved.json"),"w"),indent=2)
print("wrote p11_eaai_ved.json")
