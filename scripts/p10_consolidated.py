"""P10: ONE-PASS consolidated VED numbers so all reported values are mutually
consistent (addresses review items 1,5,8 and reconciles the combustion-R2 split).
Single config: HistGBM(300, lr=0.05, depth=5, seed=0), GroupKFold(5) by vehicle.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import pearsonr

ROOT=os.path.join(os.path.dirname(__file__),"..")
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
y=v["fuel_per_100km"].to_numpy(); g=v["VehId"].to_numpy(); SS=np.sum((y-y.mean())**2)
GBM=lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0)
def oof(feats): return cross_val_predict(GBM(),v[feats],y,cv=GroupKFold(5),groups=g)
r2=lambda p:1-np.sum((y-p)**2)/SS

pf=oof(BEHAV+ENV); pe=oof(ENV); pb=oof(BEHAV)
R2_full,R2_env,R2_behav=r2(pf),r2(pe),r2(pb)
v["resid"]=y-pf; fe=v.groupby("VehId")["resid"].transform("mean")
veh_share=np.sum((fe-v["resid"].mean())**2)/SS
behav_marg=R2_full-R2_env; env_marg=R2_full-R2_behav; unexp=1-R2_full-veh_share

# ---- axis regressions on driver-excess (net of context), ONE config ----
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
KIN=["speed_std","accel_pos_mean","accel_p95","decel_p05","jerk_rms","harsh_accel_per_km",
     "harsh_brake_per_km","stops_per_km","idle_frac","vsp_mean"]
vt=v[v["ltft1_cov"]>=0.2].dropna(subset=["stft1_mean","ltft1_mean"]).copy()
gt=vt["VehId"].to_numpy()
def excess(col):
    yy=vt[col].to_numpy()
    return yy-cross_val_predict(GBM(),vt[CTX],yy,cv=GroupKFold(5),groups=gt)
def r2x(X,z):
    p=cross_val_predict(GBM(),X,z,cv=GroupKFold(5),groups=gt); return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
vt["fuel_corr"]=vt["fuel_per_100km"]*(1+(vt["stft1_mean"]+vt["ltft1_mean"])/100.0)
ex_blind=excess("fuel_per_100km"); ex_corr=excess("fuel_corr")
comb=vt[["stft1_mean","ltft1_mean"]].to_numpy()
comb_blind=r2x(comb,ex_blind); comb_corr=r2x(comb,ex_corr); kin_r2=r2x(vt[KIN].to_numpy(),ex_blind)

# ---- axis orthogonality (within-vehicle standardized) ----
gg=vt.groupby("VehId")
zk=((vt[KIN]-gg[KIN].transform("mean"))/(gg[KIN].transform("std")+1e-9)).mean(axis=1).to_numpy()
zc=((vt["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy()
ax_corr,_=pearsonr(zk,zc)

# ---- S_bench unrounded ----
S_bench=((14.7/13.68-1)*100-(14.7/14.07-1)*100)/vt.groupby("VehId")["ltft1_mean"].std().median()

R={"n_trips":int(len(v)),"n_veh":int(v.VehId.nunique()),
   "R2_full":round(R2_full,3),"R2_env":round(R2_env,3),"R2_behav":round(R2_behav,3),
   "behav_marg":round(behav_marg,3),"env_marg":round(env_marg,3),
   "veh_share":round(veh_share,3),"unexplained":round(unexp,3),
   "sum_check":round(R2_env+behav_marg+veh_share+unexp,3),
   "comb_R2_driverexcess_blind":round(comb_blind,3),
   "comb_R2_driverexcess_trimcorr":round(comb_corr,3),
   "kin_R2_driverexcess":round(kin_r2,3),
   "axis_corr_kin_comb":round(ax_corr,3),
   "S_bench":round(float(S_bench),2)}
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p10_consolidated.json"),"w"),indent=2)
print("wrote p10_consolidated.json")
