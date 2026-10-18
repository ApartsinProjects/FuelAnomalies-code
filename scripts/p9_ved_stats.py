"""P9 (revision): VED statistical rigor + W1 trim-correction + W2 range restriction.
Cluster-bootstrap CIs (by vehicle), decomposition seed stability, trim-corrected
dissociation, and high-trim-subset analysis. Lean: n_jobs=1, fit-once + bootstrap OOF.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = os.path.join(os.path.dirname(__file__), "..")
v = pd.read_parquet(os.path.join(ROOT, "data", "processed", "trip_features.parquet"))
lo, hi = v["fuel_per_100km"].quantile([.01, .99]); v = v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb"]
y=v["fuel_per_100km"].to_numpy(); g=v["VehId"].to_numpy(); SS=np.sum((y-y.mean())**2)
def gbm(seed=0): return HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=seed)
def oof(feats,seed=0): return cross_val_predict(gbm(seed),v[feats],y,cv=GroupKFold(5),groups=g)

pred_full=oof(BEHAV+ENV); pred_env=oof(ENV); pred_behav=oof(BEHAV)
v["resid"]=y-pred_full
r2=lambda p:1-np.sum((y-p)**2)/SS
r2_full,r2_env,r2_behav=r2(pred_full),r2(pred_env),r2(pred_behav)
fe=v.groupby("VehId")["resid"].transform("mean"); veh_share=np.sum((fe-v["resid"].mean())**2)/SS

# ---- cluster bootstrap (by vehicle) for CIs ----
vehs=v["VehId"].unique(); by={k:np.where(g==k)[0] for k in vehs}; rng=np.random.default_rng(0)
def boot_metric(fn,B=1000):
    out=[]
    for _ in range(B):
        samp=rng.choice(vehs,len(vehs),replace=True)
        idx=np.concatenate([by[k] for k in samp])
        out.append(fn(idx))
    return np.percentile(out,[2.5,97.5])
def r2_idx(p): return lambda idx:1-np.sum((y[idx]-p[idx])**2)/np.sum((y[idx]-y[idx].mean())**2)
def mape_idx(idx): return np.mean(np.abs(y[idx]-pred_full[idx])/y[idx])*100
ci_full=boot_metric(r2_idx(pred_full)); ci_mape=boot_metric(mape_idx)
# variance-share CIs
def share_env(idx):
    yy=y[idx]; return 1-np.sum((yy-pred_env[idx])**2)/np.sum((yy-yy.mean())**2)
def share_behavmarg(idx):
    yy=y[idx]; re=1-np.sum((yy-pred_env[idx])**2)/np.sum((yy-yy.mean())**2)
    rf=1-np.sum((yy-pred_full[idx])**2)/np.sum((yy-yy.mean())**2); return rf-re
def share_veh(idx):
    yy=y[idx]; r=yy-pred_full[idx]; feb=pd.Series(r).groupby(g[idx]).transform("mean").to_numpy()
    return np.sum((feb-r.mean())**2)/np.sum((yy-yy.mean())**2)
ci_env=boot_metric(share_env); ci_bm=boot_metric(share_behavmarg); ci_veh=boot_metric(share_veh)

# ---- decomposition seed stability ----
seeds=[r2(oof(BEHAV+ENV,s)) for s in range(5)]

# ---- W1: trim-corrected fuel dissociation ----
vt=v[v["ltft1_cov"]>=0.2].dropna(subset=["stft1_mean","ltft1_mean"]).copy()
KIN=["speed_std","accel_pos_mean","accel_p95","decel_p05","jerk_rms","harsh_accel_per_km",
     "harsh_brake_per_km","stops_per_km","idle_frac","vsp_mean"]
# excess net of context (env only)
ctx=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
def excess_of(target):
    yy=vt[target].to_numpy()
    p=cross_val_predict(gbm(),vt[ctx],yy,cv=GroupKFold(5),groups=vt["VehId"].to_numpy())
    return yy-p
def r2_pred(X,z):
    p=cross_val_predict(gbm(),X,z,cv=GroupKFold(5),groups=vt["VehId"].to_numpy())
    return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
vt["fuel_corr"]=vt["fuel_per_100km"]*(1+(vt["stft1_mean"]+vt["ltft1_mean"])/100.0)
exc_blind=excess_of("fuel_per_100km"); exc_corr=excess_of("fuel_corr")
comb=vt[["stft1_mean","ltft1_mean"]].to_numpy()
r2_comb_blind=r2_pred(comb,exc_blind); r2_comb_corr=r2_pred(comb,exc_corr)
r2_kin_blind=r2_pred(vt[KIN].to_numpy(),exc_blind)
trim_tot=(vt["stft1_mean"]+vt["ltft1_mean"])
# ---- W2: range restriction ----
veh_ltft=vt.groupby("VehId")["ltft1_mean"].mean()
p75=veh_ltft.abs().quantile(0.75)
hi_veh=veh_ltft[veh_ltft.abs()>p75].index
vt_hi=vt[vt["VehId"].isin(hi_veh)].copy()
k=min(5,vt_hi["VehId"].nunique())
exc_hi=vt_hi["fuel_per_100km"].to_numpy()-cross_val_predict(gbm(),vt_hi[ctx],
        vt_hi["fuel_per_100km"],cv=GroupKFold(k),groups=vt_hi["VehId"].to_numpy())
corr_hi=np.corrcoef(vt_hi["ltft1_mean"].to_numpy(),exc_hi)[0,1]

R={
 "r2_full":round(r2_full,3),"ci_full":[round(x,3) for x in ci_full],
 "mape":round(np.mean(np.abs(y-pred_full)/y)*100,2),"ci_mape":[round(x,2) for x in ci_mape],
 "share_env":round(r2_env,3),"ci_env":[round(x,3) for x in ci_env],
 "behav_marg":round(r2_full-r2_env,3),"ci_behav_marg":[round(x,3) for x in ci_bm],
 "veh_share":round(veh_share,3),"ci_veh":[round(x,3) for x in ci_veh],
 "seed_stability_r2":[round(s,4) for s in seeds],
 "W1_trim_median_pct":round(trim_tot.median(),2),"W1_trim_p90_pct":round(trim_tot.quantile(.9),2),
 "W1_comb_R2_driverexcess_blind":round(r2_comb_blind,3),
 "W1_comb_R2_driverexcess_trimcorrected":round(r2_comb_corr,3),
 "W1_kin_R2_driverexcess":round(r2_kin_blind,3),
 "W2_within_veh_LTFT_sd_median":round(vt.groupby("VehId")["ltft1_mean"].std().median(),2),
 "W2_between_veh_LTFT_sd":round(veh_ltft.std(),2),
 "W2_hightrim_corr_excess_vs_LTFT":round(corr_hi,3),"W2_hightrim_nveh":int(len(hi_veh)),
}
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p9_ved_stats.json"),"w"),indent=2)
print("\nwrote data/processed/p9_ved_stats.json")
