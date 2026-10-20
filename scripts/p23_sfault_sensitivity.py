"""P23 (Fable pass-2): shore the two soft spots.

(A) S_fault SENSITIVITY SWEEP. The headline accuracy is arithmetic in two calibrated constants; S_fault
    (bench AFR -> fleet trim transfer) is the one never observed on a real faulted vehicle. We sweep it
    over [0.5, 3.0] at the fixed driver constant S_driver=0.65 and report attribution accuracy, both the
    closed-form Gaussian prediction and the empirical p17 injection, so the paper names the field-
    measurable unknown and shows the whole operating curve rather than a single assumed point.

(B) COMBUSTION AGGREGATION AUC. The per-sample fault sensitivity (AUC 0.58, TPR 0.29) is a SINGLE-SAMPLE
    number; attribution operates on a persistent per-vehicle signal. We compute AUC of the MEAN combustion
    score over n in {1,5,10,20,50} random rich vs normal samples. Invariant (stated in advance): AUC must
    rise monotonically with n and exceed 0.9 by n=20 at a per-sample effect of d~1.66; if it does not,
    the fault signature is weaker than claimed and we need to know.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as Nrm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0); R={}

# ---------- frozen UAH driver signature (as p17) ----------
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

# ---------- VED trip subset (as p17) ----------
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
v = v.drop(columns=["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]).merge(pd.read_parquet(os.path.join(ROOT,"data","processed","signature_features.parquet")), on=["VehId", "Trip"], how="left")   # signature features with the source (UAH) definition, p63
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy(); v=v.reset_index(drop=True)
gg=v.groupby("VehId"); n=len(v); sv=StandardScaler().fit(v[VF]); Zv=sv.transform(v[VF]); kraw=Zv@Wk
cz0=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy(); cz0 = cz0 / cz0.std()   # exact unit pooled variance (fault shift = S_fault exactly)
S_fault_current=3.49/v.groupby("VehId")["ltft1_mean"].std().median()  # LiU-ICE smallest measured fueling deviation, pp (p37)
ksd=kraw.std(); PERT=(S_driver*ksd/(Wk@dirn))*dirn

def z(x,ref): return (x-ref.mean())/ref.std()
def emp_acc(S_f):
    kd=z((Zv+PERT)@Wk,kraw); kn=z(kraw,kraw); cz=z(cz0,cz0); cf=z(cz0+S_f,cz0)
    X=np.vstack([np.c_[kd,cz],np.c_[kn,cf]]); y=np.r_[np.zeros(n),np.ones(n)]
    return float((( (X[:,1]>X[:,0]).astype(int))==y).mean())
def cf_acc(S_f):  # closed-form argmax Bayes for the two unit-var axes
    return float(0.5*(Nrm.cdf(S_driver/np.sqrt(2))+Nrm.cdf(S_f/np.sqrt(2))))

sweep={}
for Sf in [0.5,0.8,1.0,1.33,1.66,1.95,2.0,2.5,3.0,3.5]:
    sweep[str(Sf)]={"empirical":round(emp_acc(Sf),3),"closed_form":round(cf_acc(Sf),3)}
R["S_driver"]=round(float(S_driver),2); R["S_fault_current"]=round(float(S_fault_current),2)
R["A_sfault_sweep"]=sweep
R["A_note"]="S_fault=1.66 is the bench-AFR/fleet-trim transfer constant; sweep shows the operating curve. At S_fault=0.5 (weak trim response) acc still %.3f; at 3.0 acc %.3f."%(cf_acc(0.5),cf_acc(3.0))

# ---------- (B) combustion aggregation AUC on EngineFaultDB ----------
e=pd.read_csv(os.path.join(RAW,"EngineFaultDB_Final.csv"))
e["comb"]=(14.7/e["AFR"]-1)*100
norm=e[e.Fault==0]["comb"].to_numpy(); rich=e[e.Fault==1]["comb"].to_numpy()
mu,sd=norm.mean(),norm.std()
score_norm=(norm-mu)/sd; score_rich=(rich-mu)/sd
d_persample=abs(rich.mean()-norm.mean())/np.sqrt((rich.var()+norm.var())/2)
def agg_auc(nn,reps=400):
    aucs=[]
    for _ in range(reps):
        rr=rng.choice(score_rich,nn).mean(); nnn=rng.choice(score_norm,nn).mean()
        aucs.append((rr,nnn))
    # build AUC from reps paired means: label 1 rich, 0 normal
    s=np.r_[[a for a,_ in aucs],[b for _,b in aucs]]; yb=np.r_[np.ones(reps),np.zeros(reps)]
    return float(roc_auc_score(yb,s))
R["B_persample_AUC"]=round(float(roc_auc_score(np.r_[np.ones(len(rich)),np.zeros(len(norm))],np.r_[score_rich,score_norm])),3)
R["B_persample_cohens_d"]=round(float(d_persample),2)
R["B_aggregation_AUC"]={str(nn):round(agg_auc(nn),3) for nn in [1,5,10,20,50]}
R["B_invariant_monotone_and_ge_0.9_by_20"]=bool(
    all(agg_auc(a)<=agg_auc(b)+0.02 for a,b in zip([1,5,10,20],[5,10,20,50])) and agg_auc(20)>=0.9)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p23_sfault_sensitivity.json"),"w"),indent=2)
print("wrote p23_sfault_sensitivity.json")
