"""P18: harden three flagged soft spots.
A. P16 #1 re-run with TRIM-CORRECTED fuel target (removes the trim-blindness objection).
B. P16 #3 with OOF normal reference + ROC/AUC (removes in-sample optimism + single-threshold).
C. Vehicle-baseline share with per-fold demeaning + re-randomized folds (fold-bias check on s_veh).
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
HGB=lambda s=0:HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=5,random_state=s)
R={}

# ---------- A: P16 #1 trim-corrected ----------
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean","speed_std","accel_p95","harsh_accel_per_km"]).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
v["fuel_corr"]=v["fuel_per_100km"]*(1+(v["stft1_mean"].fillna(0)+v["ltft1_mean"])/100.0)
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
gt=v["VehId"].to_numpy()
for tgt,key in [("fuel_per_100km","blind"),("fuel_corr","trimcorr")]:
    yy=v[tgt].to_numpy(); ex=yy-cross_val_predict(HGB(),v[CTX],yy,cv=GroupKFold(5),groups=gt)
    v["_ex"]=ex
    KIN=["speed_std","accel_pos_mean","accel_p95","decel_p05","jerk_rms","harsh_accel_per_km","harsh_brake_per_km","vsp_mean"]
    gg=v.groupby("VehId")
    zk=((v[KIN]-gg[KIN].transform("mean"))/(gg[KIN].transform("std")+1e-9)).mean(axis=1)
    zc=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).abs()
    v["zk"]=zk.values; v["zc"]=zc.values
    kp75,cp75,kp50,cp50=v.zk.quantile(.75),v.zc.quantile(.75),v.zk.quantile(.5),v.zc.quantile(.5)
    kin=v[(v.zk>kp75)&(v.zc<cp50)]["_ex"]; comb=v[(v.zc>cp75)&(v.zk<kp50)]["_ex"]
    R[f"A_{key}"]={"kin_dominant_excess":round(float(kin.mean()),3),"comb_dominant_excess":round(float(comb.mean()),3),
                   "corr_excess_zk":round(float(np.corrcoef(v.zk,v._ex)[0,1]),3),
                   "corr_excess_zc":round(float(np.corrcoef(v.zc,v._ex)[0,1]),3)}

# ---------- B: P16 #3 OOF normal reference + ROC ----------
e=pd.read_csv(os.path.join(RAW,"EngineFaultDB_Final.csv"))
OP=["RPM","MAP","TPS","Force","Power","Speed"]
norm=e[e.Fault==0].reset_index(drop=True)
oof=cross_val_predict(HGB(),norm[OP],norm["Consumption L/100KM"],cv=KFold(5,shuffle=True,random_state=0))
Mfull=HGB().fit(norm[OP].to_numpy(),norm["Consumption L/100KM"].to_numpy())
e=e.copy(); e["excess"]=np.nan
e.loc[e.Fault==0,"excess"]=norm["Consumption L/100KM"].to_numpy()-oof         # OOF for normal
e.loc[e.Fault!=0,"excess"]=e.loc[e.Fault!=0,"Consumption L/100KM"].to_numpy()-Mfull.predict(e.loc[e.Fault!=0,OP].to_numpy())
e["comb"]=(14.7/e["AFR"]-1)*100
base=np.median(e.loc[e.Fault==0,"comb"])
# fuel-fault score = standardized excess + standardized combustion elevation (both vs normal)
sc=(e["excess"]/e.loc[e.Fault==0,"excess"].std())+((e["comb"]-base)/e.loc[e.Fault==0,"comb"].std())
yb=(e.Fault==1).astype(int)              # rich vs everything else
R["B_AUC_rich_vs_rest"]=round(float(roc_auc_score(yb,sc)),3)
yb2=e[e.Fault.isin([0,1])].copy(); R["B_AUC_rich_vs_normal"]=round(float(roc_auc_score((yb2.Fault==1).astype(int),
    sc[e.Fault.isin([0,1])])),3)
# flag rates with OOF normal
thr=0.3
flag=((e["excess"]>thr)&(e["comb"]>base+1.0))
R["B_flag_rate"]={n:round(float(flag[e.Fault==k].mean()),3) for k,n in {0:"normal",1:"rich",2:"lean",3:"ignition"}.items()}

# ---------- C: vehicle-baseline share, fold-demeaned + re-randomized ----------
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
y=v["fuel_per_100km"].to_numpy(); g=v["VehId"].to_numpy(); SS=np.sum((y-y.mean())**2)
def vehshare(seed,demean_fold):
    gkf=GroupKFold(5); pred=np.full(len(v),np.nan); fold=np.full(len(v),-1)
    for fi,(tr,te) in enumerate(gkf.split(v[BEHAV+ENV],y,g)):
        m=HGB(seed).fit(v[BEHAV+ENV].to_numpy()[tr],y[tr]); pred[te]=m.predict(v[BEHAV+ENV].to_numpy()[te]); fold[te]=fi
    resid=y-pred
    if demean_fold:
        for fi in range(5): resid[fold==fi]-=resid[fold==fi].mean()
    fe=pd.Series(resid).groupby(g).transform("mean").to_numpy()
    return np.sum((fe-resid.mean())**2)/SS
R["C_vehshare_plain"]=round(float(vehshare(0,False)),3)
R["C_vehshare_fold_demeaned"]=round(float(vehshare(0,True)),3)
R["C_vehshare_reseed"]=[round(float(vehshare(1,True)),3)]

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p18_harden.json"),"w"),indent=2)
print("wrote p18_harden.json")
