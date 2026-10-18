"""P13 (EAAI E3): FROZEN source-signature attribution + abstention (E4-lite).

Driver signature: logistic fit on the EXTERNAL aggressive-driving dataset (Zenodo) using
kinematic features that also exist in VED, then FROZEN. Fault signature: the combustion axis,
whose direction (rich = elevated fueling-correction) and effect size come from the bench.
Neither scoring function is fit to the VED injection labels.

Attribution rule (zero-shot, no injection-label training): predict FAULT iff the standardized
combustion score exceeds the standardized kinematic score (argmax); abstain within a margin tau.
Compared against: (a) a logistic TRAINED on the injection labels (upper bound, the old P8 setup),
(b) a fuel-magnitude baseline (chance). Same fuel increase for both causes.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score

ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0); R={}

# ---------- 1. Driver signature FROZEN from Zenodo (external) ----------
def zfeats(x,y,z,t):
    mag=np.sqrt(x*x+y*y+z*z); dur=t[-1]-t[0] if len(t)>1 else 1
    return [mag.std(), np.percentile(mag,95), np.sum(mag>3)/dur]   # var, p95, harsh-rate
rows,yb=[],[]
for i in [1,2,3]:
    a=pd.read_csv(f"{RAW}/zenodo_driving/Linear_Acceleration_{i}.csv",dtype=np.float32,usecols=[0,1,2,3])
    a.columns=["t","x","y","z"]; tv=a["t"].to_numpy(); ev=pd.read_csv(f"{RAW}/zenodo_driving/Labeled_events_{i}.csv")
    for _,r in ev.iterrows():
        m=(tv>=r["start"])&(tv<=r["end"])
        if m.sum()<20: continue
        w=a.loc[m]; rows.append(zfeats(w["x"].to_numpy(),w["y"].to_numpy(),w["z"].to_numpy(),w["t"].to_numpy()))
        yb.append(1 if int(r["target"])>=1 else 0)
    del a
Xz=np.array(rows); yb=np.array(yb)
zsc=StandardScaler().fit(Xz)                          # source standardization
drv=LogisticRegression(max_iter=1000).fit(zsc.transform(Xz),yb)   # FROZEN driver signature
Wz=drv.coef_[0]                                       # frozen weights on [std, p95, harsh]
R["driver_sig_weights"]=[round(float(x),3) for x in Wz]

# ---------- 2. VED trips: matched kinematic features + combustion axis ----------
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean","speed_std","accel_p95","harsh_accel_per_km"]).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
# VED features matched to Zenodo [variability, p95, harsh-rate]; within-vehicle standardized
KMAP=["speed_std","accel_p95","harsh_accel_per_km"]
gg=v.groupby("VehId")
Zk=((v[KMAP]-gg[KMAP].transform("mean"))/(gg[KMAP].transform("std")+1e-9))
# apply the FROZEN Zenodo weights to the (re-standardized to source scale) VED features
Xv=StandardScaler().fit_transform(Zk.to_numpy())     # unsupervised target standardization
kin_raw=Xv@Wz                                         # frozen driver score on VED
comb_raw=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy()
# standardize both scores to unit SD on unperturbed VED (unsupervised)
kz=(kin_raw-np.nanmean(kin_raw))/np.nanstd(kin_raw)
cz=(comb_raw-np.nanmean(comb_raw))/np.nanstd(comb_raw)
mask=np.isfinite(kz)&np.isfinite(cz); kz,cz=kz[mask],cz[mask]; grp=v["VehId"].to_numpy()[mask]; n=len(kz)
R["n_trips"]=int(n); R["n_veh"]=int(pd.unique(grp).size)

S_bench=((14.7/13.68-1)*100-(14.7/14.07-1)*100)/v.groupby("VehId")["ltft1_mean"].std().median()
R["S_bench"]=round(float(S_bench),2)

def build(S):
    X=np.vstack([np.c_[kz+S,cz], np.c_[kz,cz+S]]); y=np.r_[np.zeros(n),np.ones(n)]; g=np.r_[grp,grp]
    fe=np.r_[rng.normal(0.06,0.01,n),rng.normal(0.06,0.01,n)]
    return X,y,g,fe

# ---------- 3. FROZEN argmax rule vs TRAINED logistic vs fuel-only ----------
print(f"{'S':>4} {'frozen_argmax':>14} {'trained_logit':>14} {'fuel_only':>10}")
curve=[]
for S in [0.5,1.0,1.5,S_bench,2.0,2.5,3.0]:
    X,y,g,fe=build(S)
    frozen=(X[:,1]>X[:,0]).astype(int)               # predict fault iff comb>kin (zero-shot)
    a_fr=accuracy_score(y,frozen)
    a_tr=accuracy_score(y,cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
                         X,y,cv=GroupKFold(5),groups=g))
    a_fu=accuracy_score(y,cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
                         fe.reshape(-1,1),y,cv=GroupKFold(5),groups=g))
    curve.append((round(float(S),2),round(a_fr,3),round(a_tr,3),round(a_fu,3)))
    tag="*bench" if abs(S-S_bench)<1e-6 else ""
    print(f"{S:4.2f} {a_fr:14.3f} {a_tr:14.3f} {a_fu:10.3f} {tag}")
R["curve_S_frozen_trained_fuelonly"]=curve

# ---------- 4. abstention: coverage vs accuracy at S_bench (frozen margin) ----------
X,y,g,fe=build(S_bench); margin=np.abs(X[:,1]-X[:,0]); pred=(X[:,1]>X[:,0]).astype(int)
print("\nabstention (frozen, S_bench): tau  coverage  accuracy-on-covered")
absten=[]
for tau in [0.0,0.5,1.0,1.5,2.0]:
    keep=margin>=tau
    cov=keep.mean(); acc=accuracy_score(y[keep],pred[keep]) if keep.sum()>0 else float("nan")
    absten.append((tau,round(float(cov),3),round(float(acc),3)))
    print(f"  {tau:.1f}   {cov:.3f}    {acc:.3f}")
R["abstention_tau_coverage_acc"]=absten
R["frozen_acc_at_Sbench"]=next(c[1] for c in curve if abs(c[0]-round(float(S_bench),2))<0.02)

json.dump(R,open(os.path.join(ROOT,"data","processed","p13_frozen.json"),"w"),indent=2)
print("\nwrote p13_frozen.json  frozen@bench=%.3f"%R["frozen_acc_at_Sbench"])
