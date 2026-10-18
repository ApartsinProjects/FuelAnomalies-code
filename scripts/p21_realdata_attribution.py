"""P21 (Fable item-5): weak-label attribution on REAL VED trips (no injection).

VED carries no ground-truth cause labels (the paper's whole premise). We therefore build two
INDEPENDENT proxy label groups from the fleet's own signals and ask whether the FROZEN
signature pipeline (UAH driver weights + within-vehicle combustion axis) + likelihood-ratio
fusion agrees with them, and whether split-conformal abstention yields a usable
coverage/selective-error tradeoff on real (non-synthetic) trips.

Proxy labels (each requires a positive fuel excess, so we only attribute trips that are
actually anomalous):
  driver-caused proxy = high kinematic z (top decile) AND low combustion z AND excess>0
  fault-suspect proxy = sustained |LTFT| high (top decile of within-veh combustion) AND
                        low kinematic z AND excess>0
Trips matching neither or both are left UNLABELLED and excluded from the accuracy set.

HONESTY / INVARIANTS stated up front:
  * This is a consistency check, NOT independent validation: proxy labels and the classifier
    share the same two axes, so agreement is necessary-not-sufficient. We report it as such.
  * The fleet is healthy (P0/E1), so the fault-suspect group is expected to be THIN. We report
    the group sizes explicitly; if the fault side is too small we report the driver side alone.
  * Excess is out-of-fold (GroupKFold by vehicle) so it is not in-sample optimistic.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as N
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0)

# ---- frozen UAH driver signature (identical to p17/p19) ----
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
Wk=LogisticRegression(max_iter=1000).fit(su.transform(Uu),uy).coef_[0]
sc=su.transform(Uu)@Wk; S_driver=abs(sc[uy==1].mean()-sc[uy==0].mean())/np.sqrt((sc[uy==1].var()+sc[uy==0].var())/2)

# ---- VED real trips ----
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
v=v.reset_index(drop=True); grp=v["VehId"].to_numpy()

# out-of-fold excess (context model, GroupKFold by vehicle)
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
HGB=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0)
yy=v["fuel_per_100km"].to_numpy()
excess=yy-cross_val_predict(HGB,v[CTX],yy,cv=GroupKFold(5),groups=grp)
v["excess"]=excess

# frozen kinematic score (VED features standardized on VED, scored with UAH weights)
Zv=StandardScaler().fit_transform(v[VF]); kraw=Zv@Wk
kz=(kraw-kraw.mean())/kraw.std()                      # standardized driver score
# within-vehicle combustion score (|LTFT| deviation from the vehicle's own mean)
gg=v.groupby("VehId")
cz=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).abs().to_numpy()
czz=(cz-cz.mean())/cz.std()

# ---- SINGLE-AXIS proxy labels (require positive excess). No cross-axis clamp, so the two
# groups are NOT forced anti-correlated: a trip can be high on both axes (the contested/mixed
# case) and is then left UNLABELLED. This avoids the trivial-separation artifact that a
# mutually-exclusive definition would create. ----
khi=np.quantile(kz,0.90); chi=np.quantile(czz,0.90)
driver_proxy=(kz>=khi)&(excess>0)
fault_proxy =(czz>=chi)&(excess>0)
both=driver_proxy&fault_proxy                              # elevated on both -> contested, unlabelled
lab=np.full(len(v),-1)
lab[driver_proxy&~both]=0; lab[fault_proxy&~both]=1
mask=lab>=0
n_dr=int((lab==0).sum()); n_ft=int((lab==1).sum()); n_mixed=int(both.sum())

# ---- likelihood-ratio fusion (source-calibrated Gaussians, as p19) ----
ld=N.logpdf(kz,S_driver,1)-N.logpdf(kz,0,1)
lf=N.logpdf(czz,S_driver,1)-N.logpdf(czz,0,1)              # symmetric prior scale on real data
pred=(lf>ld).astype(int); conf=np.abs(ld-lf)
acc=float((pred[mask]==lab[mask]).mean()) if mask.sum() else float("nan")

R={"S_driver":round(float(S_driver),2),"n_driver_proxy":n_dr,"n_fault_proxy":n_ft,
   "n_mixed_contested":n_mixed,"n_labelled":int(mask.sum()),"n_total":int(len(v)),
   "LR_fusion_acc_on_proxy":round(acc,3),
   "driver_side_recall":round(float((pred[lab==0]==0).mean()),3) if n_dr else None,
   "fault_side_recall":round(float((pred[lab==1]==1).mean()),3) if n_ft else None}

# ---- split-conformal abstention on real proxy-labelled trips (Hoeffding UCB, delta=0.1) ----
li=np.where(mask)[0]; gi=grp[li]; vehs=np.unique(gi); rng.shuffle(vehs); cut=len(vehs)//2
cal=np.isin(gi,vehs[:cut]); te=~cal; delta=0.1
cL=conf[li]; pL=pred[li]; yL=lab[li]
def tau_for(alpha):
    c=cL[cal]; e=(pL[cal]!=yL[cal]).astype(int); order=np.argsort(-c)
    es=np.cumsum(e[order]); ns=np.arange(1,len(c)+1); ehat=es/ns
    ucb=ehat+np.sqrt(np.log(1.0/delta)/(2*ns))
    ok=np.where(ucb<=alpha)[0]
    return c[order][ok[-1]] if len(ok) else np.inf
R["conformal_delta"]=delta; R["conformal"]={}
for a in [0.05,0.10,0.20]:
    t=tau_for(a); am=cL[te]>=t
    cov=float(am.mean()); err=float((pL[te][am]!=yL[te][am]).mean()) if am.sum() else float("nan")
    R["conformal"][str(a)]={"coverage":round(cov,3),"selective_error":round(err,3),"target_alpha":a}

# ---- invariant check: driver-proxy trips should carry higher mean excess than fault-proxy? ----
R["mean_excess_driver_proxy"]=round(float(excess[lab==0].mean()),3) if n_dr else None
R["mean_excess_fault_proxy"]=round(float(excess[lab==1].mean()),3) if n_ft else None
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p21_realdata_attribution.json"),"w"),indent=2)
print("wrote p21_realdata_attribution.json")
