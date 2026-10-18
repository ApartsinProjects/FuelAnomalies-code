"""P24: construct-matched ranking metrics at the CORRECT asymmetric operating points.
p14's AUROC (0.951) is from the superseded SYMMETRIC, score-level injection at S=1.66; it must not be
quoted beside the honest 0.785 (d=0.65) accuracy. Here we recompute accuracy AND AUROC/AUPRC from the
SAME p17 feature-level injection (frozen UAH weights, asymmetric fault S_fault, unit-SD axes), at:
  - the conservative window-level operating point d_driver = 0.65
  - the realistic trip-level operating point d_driver = 1.3 (VED records whole trips; aggregation raises
    the aggressive-vs-normal effect from 0.65 to ~1.3, see 8.3)
Continuous score = combustion-evidence minus kinematic-evidence (the margin the argmax thresholds at 0).
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")

def kin(spd):
    v=np.asarray(spd,float)/3.6; a=np.diff(v) if len(v)>1 else np.array([0.]); j=np.diff(a) if len(a)>1 else np.array([0.])
    d=np.nansum(v)/1000.0 if len(v) else 1e-9
    return [np.nanstd(v)*3.6, np.nanpercentile(a,95) if len(a) else 0.0, np.sum(np.abs(a)>2.5)/max(d,1e-3), np.sqrt(np.nanmean(j**2)) if len(j) else 0.0]
ur,uy=[],[]
for f in sorted(glob.glob(os.path.join(RAW,"uah","D*_merged.csv"))):
    d=pd.read_csv(f); ts=d["Timestamp (seconds)"].to_numpy(); seg=np.concatenate([[0],np.cumsum(np.diff(ts)<0)]); d=d.assign(_s=seg)
    for _,rec in d.groupby("_s"):
        rec=rec.reset_index(drop=True); t0=rec["Timestamp (seconds)"].iloc[0]; wid=((rec["Timestamp (seconds)"]-t0)//30).astype(int)
        for _,w in rec.groupby(wid):
            if len(w)<15: continue
            r=[w["Ratio normal (base 1)"].sum(),w["Ratio drowsy (base 1)"].sum(),w["Ratio aggressive (base 1)"].sum()]; c=int(np.argmax(r))
            if c==1: continue
            ur.append(kin(w["Speed (km/h)"])); uy.append(1 if c==2 else 0)
Uu=np.array(ur); uy=np.array(uy); su=StandardScaler().fit(Uu); Uz=su.transform(Uu)
Wk=LogisticRegression(max_iter=1000).fit(Uz,uy).coef_[0]; dirn=Uz[uy==1].mean(0)-Uz[uy==0].mean(0)
sc=Uz@Wk; S_driver=abs(sc[uy==1].mean()-sc[uy==0].mean())/np.sqrt((sc[uy==1].var()+sc[uy==0].var())/2)

v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
sv=StandardScaler().fit(v[VF]); Zv=sv.transform(v[VF]); kraw=Zv@Wk; n=len(v)
cz0=((v["ltft1_mean"]-v.groupby("VehId")["ltft1_mean"].transform("mean"))/(v.groupby("VehId")["ltft1_mean"].transform("std")+1e-9)).to_numpy()
S_fault=3.49/v.groupby("VehId")["ltft1_mean"].std().median()  # LiU-ICE smallest measured fueling deviation, pp (p37)
ksd=kraw.std()
def z(x,ref): return (x-ref.mean())/ref.std()

def metrics(S_d):
    PERT=(S_d*ksd/(Wk@dirn))*dirn
    kd=z((Zv+PERT)@Wk,kraw); kn=z(kraw,kraw); cz=z(cz0,cz0); cf=z(cz0+S_fault,cz0)
    Xk=np.r_[kd,kn]; Xc=np.r_[cz,cf]; y=np.r_[np.zeros(n),np.ones(n)]
    score=Xc-Xk                    # continuous fault-minus-driver margin; argmax thresholds at 0
    return {"accuracy":round(float(accuracy_score(y,(score>0).astype(int))),3),
            "AUROC":round(float(roc_auc_score(y,score)),3),
            "AUPRC":round(float(average_precision_score(y,score)),3)}

R={"S_driver_window":round(float(S_driver),2),"S_fault":round(float(S_fault),2),
   "window_level_d0.65":metrics(0.65),"trip_level_d1.3":metrics(1.3),
   "note":"Construct-matched to p17 feature-level asymmetric injection. Supersedes p14 AUROC 0.951 (symmetric, score-level, superseded methodology)."}
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p24_operating_points.json"),"w"),indent=2)
print("wrote p24_operating_points.json")
