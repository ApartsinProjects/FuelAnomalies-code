"""P9 (revision) part 2: bench collinearity/overlap (W4, should-fix 8) + behaviour-side
statistics (AUC CI, 1000-permutation p, RF vs LogReg) + attribution-accuracy CI.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0); R={}

# ===== W4: bench combustion collinearity =====
e=pd.read_csv(os.path.join(RAW,"EngineFaultDB_Final.csv"))
OP=["RPM","MAP","TPS","Force","Power","Speed"]; norm=e[e.Fault==0]
M=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=6,random_state=0).fit(
    norm[OP].to_numpy(),norm["Consumption L/100KM"].to_numpy())
e["excess"]=e["Consumption L/100KM"].to_numpy()-M.predict(e[OP].to_numpy())
rich=e[e.Fault==1].copy()
def r2cv(X,z):
    # shuffled folds: rows are grouped by Fault and ordered within, so non-shuffled CV
    # splits into disjoint operating ranges and spuriously fails (rows are exchangeable here)
    p=cross_val_predict(HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0),
                        X,z,cv=KFold(5,shuffle=True,random_state=0)); return 1-np.sum((z-p)**2)/np.sum((z-z.mean())**2)
z=rich["excess"].to_numpy()
R["W4_bench_R2_all4"]=round(r2cv(rich[["AFR","Lambda","CO","HC"]].to_numpy(),z),3)
for c in ["AFR","Lambda","CO","HC"]:
    R[f"W4_bench_R2_{c}_only"]=round(r2cv(rich[[c]].to_numpy(),z),3)
R["W4_corr_AFR_Lambda"]=round(np.corrcoef(rich["AFR"],rich["Lambda"])[0,1],3)
# covariate overlap normal vs rich (standardized mean difference)
def smd(a,b):
    return round((a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2),2)
R["W4_overlap_SMD"]={c:smd(e.loc[e.Fault==0,c],e.loc[e.Fault==1,c]) for c in ["RPM","Power","Speed"]}
R["W4_fault0_lambda"]=round(e.loc[e.Fault==0,"Lambda"].mean(),3)
R["W4_fault0_CO_pct"]=round(e.loc[e.Fault==0,"CO"].mean(),2)

# ===== behaviour: AUC CI + 1000-perm + RF =====
def feats(x,y,z2,t):
    mag=np.sqrt(x*x+y*y+z2*z2); dt=np.median(np.diff(t)) if len(t)>1 else .0025
    jk=np.diff(mag)/dt if len(mag)>1 else np.array([0.],np.float32); dur=t[-1]-t[0] if len(t)>1 else 1
    return [mag.mean(),mag.std(),np.percentile(mag,95),mag.max(),np.sqrt(np.mean(jk*jk)),
            np.percentile(np.abs(jk),95),np.sum(mag>3)/dur,x.std(),y.std(),z2.std(),np.mean(mag*mag)]
rows,yb,grp=[],[],[]
for i in [1,2,3]:
    a=pd.read_csv(f"{RAW}/zenodo_driving/Linear_Acceleration_{i}.csv",dtype=np.float32,usecols=[0,1,2,3])
    a.columns=["t","x","y","z"]; tv=a["t"].to_numpy(); ev=pd.read_csv(f"{RAW}/zenodo_driving/Labeled_events_{i}.csv")
    for _,r in ev.iterrows():
        m=(tv>=r["start"])&(tv<=r["end"])
        if m.sum()<20: continue
        w=a.loc[m]; rows.append(feats(w["x"].to_numpy(),w["y"].to_numpy(),w["z"].to_numpy(),w["t"].to_numpy()))
        yb.append(1 if int(r["target"])>=1 else 0); grp.append(i)
    del a
X=np.array(rows); yb=np.array(yb); grp=np.array(grp)
def logo(y_,model):
    pr=np.full(len(X),np.nan)
    for tr,te in LeaveOneGroupOut().split(X,y_,grp):
        pr[te]=model().fit(X[tr],y_[tr]).predict_proba(X[te])[:,1]
    return pr
lr=lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
rf=lambda:RandomForestClassifier(n_estimators=300,random_state=0)
prob=logo(yb,lr); auc=roc_auc_score(yb,prob); auc_rf=roc_auc_score(yb,logo(yb,rf))
# bootstrap AUC CI (resample events, stratified-ish)
aucs=[]
idx=np.arange(len(X))
for _ in range(1000):
    s=rng.choice(idx,len(idx),replace=True)
    if len(np.unique(yb[s]))<2: continue
    aucs.append(roc_auc_score(yb[s],prob[s]))
R["behav_auc_logreg"]=round(auc,3); R["behav_auc_ci"]=[round(np.percentile(aucs,2.5),3),round(np.percentile(aucs,97.5),3)]
R["behav_auc_rf"]=round(auc_rf,3); R["behav_n"]=int(len(X)); R["behav_pos"]=int(yb.sum()); R["behav_neg"]=int((yb==0).sum())
# 1000 permutations
perm=[]
for _ in range(1000):
    yp=rng.permutation(yb); perm.append(roc_auc_score(yp,logo(yp,lr)))
R["behav_perm_mean"]=round(float(np.mean(perm)),3); R["behav_perm_p"]=round((1+np.sum(np.array(perm)>=auc))/(1+len(perm)),4)

# ===== attribution accuracy CI at bench-calibrated S =====
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean","stft1_mean"]).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
KIN=["accel_p95","jerk_rms","harsh_accel_per_km","harsh_brake_per_km","speed_std","vsp_mean"]
gg=v.groupby("VehId")
zk=((v[KIN]-gg[KIN].transform("mean"))/(gg[KIN].transform("std")+1e-9)).mean(axis=1).to_numpy()
zc=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy()
grp2=v["VehId"].to_numpy(); n=len(v)
S=(14.7/13.68-1)*100-(14.7/14.07-1)*100; S=S/ v.groupby("VehId")["ltft1_mean"].std().median()
Xa=np.vstack([np.c_[zk+S,zc],np.c_[zk,zc+S]]); ya=np.r_[np.zeros(n),np.ones(n)]; ga=np.r_[grp2,grp2]
pa=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),Xa,ya,cv=GroupKFold(5),groups=ga)
acc=accuracy_score(ya,pa)
# cluster bootstrap by vehicle
vv=np.unique(ga); byv={k:np.where(ga==k)[0] for k in vv}; accs=[]
for _ in range(1000):
    s=rng.choice(vv,len(vv),replace=True); ii=np.concatenate([byv[k] for k in s])
    accs.append(accuracy_score(ya[ii],pa[ii]))
R["attrib_acc"]=round(acc,3); R["attrib_acc_ci"]=[round(np.percentile(accs,2.5),3),round(np.percentile(accs,97.5),3)]
R["attrib_S_bench"]=round(float(S),2)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p9_bench_behav.json"),"w"),indent=2)
print("wrote data/processed/p9_bench_behav.json")
