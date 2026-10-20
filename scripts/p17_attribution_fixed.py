"""P17: CORRECTED attribution experiment (fixes Fable BUG-1..4).
- Unit-SD axes (BUG-1).
- FEATURE-LEVEL driver injection along the real aggressive direction learned from UAH,
  scored with the FROZEN UAH weights, so shuffling the weights breaks it (BUG-2 falsification).
- MEASURED fuel-only baseline: both variants get the same +6% fuel, classified on it (BUG-3).
- ASYMMETRIC calibration: S_fault from the bench (2.98 pp LTFT); S_driver from UAH Cohen's d (BUG-4).
Common kinematic features derived from 1 Hz SPEED in BOTH UAH and VED (identical formulas).
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0); R={}

# ---------- common kinematic features from 1 Hz speed ----------
def kin_from_speed(spd_kmh):
    v=np.asarray(spd_kmh,float)/3.6; a=np.diff(v) if len(v)>1 else np.array([0.]); j=np.diff(a) if len(a)>1 else np.array([0.])
    dist_km=np.nansum(v)/1000.0 if len(v) else 1e-9
    return [np.nanstd(v)*3.6, np.nanpercentile(a,95) if len(a) else 0.0,
            (np.sum(np.abs(a)>2.5))/max(dist_km,1e-3), np.sqrt(np.nanmean(j**2)) if len(j) else 0.0]
FEAT=["speed_std","accel_p95","harsh_per_km","jerk_rms"]

# ---------- UAH: frozen driver signature ----------
ur,uy=[],[]
for f in sorted(glob.glob(os.path.join(RAW,"uah","D*_merged.csv"))):
    d=pd.read_csv(f); ts=d["Timestamp (seconds)"].to_numpy(); seg=np.concatenate([[0],np.cumsum(np.diff(ts)<0)])
    d=d.assign(_seg=seg)
    for _,rec in d.groupby("_seg"):
        rec=rec.reset_index(drop=True); t0=rec["Timestamp (seconds)"].iloc[0]
        wid=((rec["Timestamp (seconds)"]-t0)//30).astype(int)
        for _,w in rec.groupby(wid):
            if len(w)<15: continue
            rat=[w["Ratio normal (base 1)"].sum(),w["Ratio drowsy (base 1)"].sum(),w["Ratio aggressive (base 1)"].sum()]
            c=int(np.argmax(rat))
            if c==1: continue
            ur.append(kin_from_speed(w["Speed (km/h)"])); uy.append(1 if c==2 else 0)
Uu=np.array(ur); uy=np.array(uy)
su=StandardScaler().fit(Uu); Uz=su.transform(Uu)
drv=LogisticRegression(max_iter=1000).fit(Uz,uy); Wk=drv.coef_[0]
dirn=Uz[uy==1].mean(0)-Uz[uy==0].mean(0)            # real aggressive direction (standardized)
sc=Uz@Wk; S_driver=abs(sc[uy==1].mean()-sc[uy==0].mean())/np.sqrt((sc[uy==1].var()+sc[uy==0].var())/2)
R["S_driver_cohen_d"]=round(float(S_driver),2); R["driver_weights"]=[round(float(x),3) for x in Wk]

# ---------- VED trim subset ----------
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
v = v.drop(columns=["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]).merge(pd.read_parquet(os.path.join(ROOT,"data","processed","signature_features.parquet")), on=["VehId", "Trip"], how="left")   # signature features with the source (UAH) definition, p63
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VFEAT=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]   # same order as FEAT
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VFEAT).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
sv=StandardScaler().fit(v[VFEAT]); Zv=sv.transform(v[VFEAT])
ltft_sd=v.groupby("VehId")["ltft1_mean"].std().median(); S_fault=3.49/ltft_sd  # LiU-ICE smallest measured fueling deviation, pp (p37)
R["S_fault_bench"]=round(float(S_fault),2); R["ltft_within_sd"]=round(float(ltft_sd),2)
gg=v.groupby("VehId"); grp=v["VehId"].to_numpy(); n=len(v)
cz0=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy(); cz0 = cz0 / cz0.std()   # exact unit pooled variance (fault shift = S_fault exactly)

# FIXED driver perturbation vector, calibrated ONCE with the TRUE weights to shift the true
# score by S_driver SDs. The SAME feature perturbation is then scored by any weight vector, so a
# misaligned (shuffled/negated) weight vector genuinely captures less of the injected signal.
ksd_true=(Zv@Wk).std(); a_true=S_driver*ksd_true/(Wk@dirn); PERT=a_true*dirn   # feature-space step
def run(Wk_use, S_f):
    kraw=Zv@Wk_use                                   # scoring with these weights (unperturbed)
    kd=(Zv+PERT)@Wk_use                              # driver: FIXED feature perturbation, scored
    cf=cz0+S_f                                        # fault: combustion (single measured feature) +S_f
    def z(x,ref): return (x-ref.mean())/ref.std()
    X=np.vstack([np.c_[z(kd,kraw),z(cz0,cz0)], np.c_[z(kraw,kraw),z(cf,cz0)]])
    y=np.r_[np.zeros(n),np.ones(n)]; g=np.r_[grp,grp]
    return accuracy_score(y,(X[:,1]>X[:,0]).astype(int)),X,y,g

acc,X,y,g=run(Wk,S_fault)
R["frozen_argmax_acc"]=round(float(acc),3)
# trained logistic (unit-SD axes) upper bound
R["trained_acc"]=round(float(accuracy_score(y,cross_val_predict(
    make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),X,y,cv=GroupKFold(5),groups=g))),3)
# FALSIFICATION: shuffled and negated driver weights
sh=[]
for _ in range(20):
    Ws=Wk.copy(); rng.shuffle(Ws); sh.append(run(Ws,S_fault)[0])
R["shuffled_weights_acc_mean"]=round(float(np.mean(sh)),3)
R["negated_weights_acc"]=round(float(run(-Wk,S_fault)[0]),3)
# aggression-ORTHOGONAL weights = the true no-driver-information floor (fault axis alone)
orth=[]
for _ in range(20):
    w=rng.standard_normal(len(Wk)); w=w-(w@dirn)/(dirn@dirn)*dirn; orth.append(run(w,S_fault)[0])
R["orthogonal_weights_floor"]=round(float(np.mean(orth)),3)
# MEASURED fuel-only baseline: both classes get +6% fuel -> classify on delta-fuel
fe=np.r_[rng.normal(0.06,0.01,n),rng.normal(0.06,0.01,n)]
R["fuel_only_measured_acc"]=round(float(accuracy_score(y,cross_val_predict(
    make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),fe.reshape(-1,1),y,cv=GroupKFold(5),groups=g))),3)

# ---- driver-severity sweep (rebuild PERT per S_d): accuracy rises with aggression severity ----
def run_Sd(S_d):
    a=S_d*ksd_true/(Wk@dirn); P=a*dirn; kraw=Zv@Wk
    def z(x,ref): return (x-ref.mean())/ref.std()
    X=np.vstack([np.c_[z((Zv+P)@Wk,kraw),z(cz0,cz0)], np.c_[z(kraw,kraw),z(cz0+S_fault,cz0)]])
    yy=np.r_[np.zeros(n),np.ones(n)]
    return round(float(accuracy_score(yy,(X[:,1]>X[:,0]).astype(int))),3)
R["driver_severity_sweep"]={str(s):run_Sd(s) for s in [0.65,1.0,1.5,1.66,1.94,2.0,3.0]}

# ---- abstention (margin) at the operating point ----
mar=np.abs(X[:,1]-X[:,0]); pred=(X[:,1]>X[:,0]).astype(int)
R["abstention"]={}
for tau in [0.0,0.5,1.0,1.5,2.0]:
    k=mar>=tau; R["abstention"][str(tau)]=[round(float(k.mean()),3), round(float(accuracy_score(y[k],pred[k])),3) if k.sum() else None]

# ---- task-aligned baselines at the operating point ----
# raw-threshold rule: larger standardized deviation wins (== argmax) already = frozen; add LR-prototype
lr_proto=(-0.5*((X[:,0]-0)**2+(X[:,1]-S_fault)**2) - (-0.5*((X[:,0]-S_driver)**2+(X[:,1]-0)**2)))
R["baseline_lr_prototype"]=round(float(accuracy_score(y,(lr_proto>0).astype(int))),3)

# ---- corrected mixed-cause 2D grid (feature-level driver injection) ----
grid=[0,0.5,1.0,1.5,2.0,2.5,3.0]; kraw=Zv@Wk
def z(x,ref): return (x-ref.mean())/ref.std()
H=[]
for Sd in grid:
    row=[]
    for Sf in grid:
        if Sd==Sf: row.append(None); continue
        P=(Sd*ksd_true/(Wk@dirn))*dirn
        kd=z((Zv+P)@Wk,kraw); cc=z(cz0+Sf,cz0)
        pred=(cc>kd).astype(int); true=1 if Sf>Sd else 0
        row.append(round(float(np.mean(pred==true)),3))
    H.append(row)
R["grid_S"]=grid; R["grid_acc"]=H
json.dump(R,open(os.path.join(ROOT,"data","processed","p17_attribution_fixed.json"),"w"),indent=2)
print("frozen=%.3f trained=%.3f negated=%.3f fuel_only=%.3f"%(R["frozen_argmax_acc"],R["trained_acc"],R["negated_weights_acc"],R["fuel_only_measured_acc"]))
json.dump(R,open(os.path.join(ROOT,"data","processed","p17_attribution_fixed.json"),"w"),indent=2)
print("wrote p17_attribution_fixed.json")
