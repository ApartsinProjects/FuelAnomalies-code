"""P19: principled fusion + conformal abstention (method upgrade for Algorithm 1).
Replace the margin-argmax with a source-calibrated likelihood-ratio fusion, and calibrate the
abstention threshold by SPLIT CONFORMAL so decisions carry a finite-sample selective-risk guarantee
P(error | decision issued) <= alpha. Reuses the corrected P17 frozen scores.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw"); rng=np.random.default_rng(0)

# ---- frozen UAH driver signature (as p17) ----
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
Uu=np.array(ur); uy=np.array(uy); su=StandardScaler().fit(Uu)
Wk=LogisticRegression(max_iter=1000).fit(su.transform(Uu),uy).coef_[0]; dirn=su.transform(Uu)[uy==1].mean(0)-su.transform(Uu)[uy==0].mean(0)
sc=su.transform(Uu)@Wk; S_driver=abs(sc[uy==1].mean()-sc[uy==0].mean())/np.sqrt((sc[uy==1].var()+sc[uy==0].var())/2)

# ---- VED scores + injection ----
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy(); cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
Zv=StandardScaler().fit_transform(v[VF]); gg=v.groupby("VehId")
cz0=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy()
S_fault=3.49/v.groupby("VehId")["ltft1_mean"].std().median(); n=len(v); grp=v["VehId"].to_numpy()  # LiU-ICE smallest measured fueling deviation, pp (p37)
kraw=Zv@Wk; ksd=kraw.std(); PERT=(S_driver*ksd/(Wk@dirn))*dirn
def z(x,ref): return (x-ref.mean())/ref.std()
kz_dr=z((Zv+PERT)@Wk,kraw); kz_no=z(kraw,kraw); cz_z=z(cz0,cz0); cz_f=z(cz0+S_fault,cz0)
# two classes: driver (kz elevated), fault (cz elevated)
KZ=np.r_[kz_dr,kz_no]; CZ=np.r_[cz_z,cz_f]; y=np.r_[np.zeros(n),np.ones(n)]; g=np.r_[grp,grp]

# ---- likelihood-ratio fusion (source-calibrated Gaussians) ----
ld=N.logpdf(KZ,S_driver,1)-N.logpdf(KZ,0,1)     # driver present vs not
lf=N.logpdf(CZ,S_fault,1)-N.logpdf(CZ,0,1)      # fault present vs not
pred=(lf>ld).astype(int); conf=np.abs(ld-lf)
acc=(pred==y).mean()
R={"S_driver":round(float(S_driver),2),"S_fault":round(float(S_fault),2),"LR_fusion_acc":round(float(acc),3)}

# ---- SPLIT-CONFORMAL selective risk with a FINITE-SAMPLE (Hoeffding) guarantee ----
# Calibration set = held-out VEHICLES' injected variants (labelled semi-synthetic trips; split by
# vehicle so exchangeability holds at the vehicle level). Choose the threshold with the most coverage
# whose Hoeffding UPPER confidence bound on selective error is <= alpha, so P(test error <= alpha) >= 1-delta.
vehs=np.unique(g); rng.shuffle(vehs); cut=len(vehs)//2
cal=np.isin(g,vehs[:cut]); te=~cal; delta=0.1
def tau_for(alpha):
    c=conf[cal]; e=(pred[cal]!=y[cal]).astype(int); order=np.argsort(-c)
    es=np.cumsum(e[order]); ns=np.arange(1,len(c)+1); ehat=es/ns
    K=100; pos=np.unique(np.clip((np.arange(1,K+1)/K*len(c)).astype(int),1,len(c)))-1   # fixed grid of candidate thresholds
    ucb=__import__('scipy.stats',fromlist=['beta']).beta.ppf(1-delta/K,np.cumsum(e[order])[pos]+1,np.maximum(ns[pos]-np.cumsum(e[order])[pos],1e-9))   # exact binomial (Clopper-Pearson) UCB, Bonferroni over K candidates
    ok=np.where(ucb<=alpha)[0]
    return c[order][pos[ok[-1]]] if len(ok) else np.inf
R["conformal_delta"]=delta; R["conformal"]={}
for a in [0.05,0.10,0.20]:
    t=tau_for(a); acc_mask=conf[te]>=t
    cov=float(acc_mask.mean()); err=float((pred[te][acc_mask]!=y[te][acc_mask]).mean()) if acc_mask.sum() else float("nan")
    R["conformal"][str(a)]={"coverage":round(cov,3),"selective_error":round(err,3),"target_alpha":a}
print(json.dumps(R,indent=2)); json.dump(R,open(os.path.join(ROOT,"data","processed","p19_conformal.json"),"w"),indent=2)
print("wrote p19_conformal.json")
