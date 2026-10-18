"""P14 (EAAI E4 + E5): stress tests + task-aligned baselines for frozen attribution.

Reuses the frozen source-derived scores (driver signature from external data; combustion axis
from bench) built as in p13. All evaluation is zero-shot (no injection-label training) unless a
baseline is explicitly a trained upper bound.

E4: mixed-cause 2D grid (dominant-cause accuracy vs driver effect x fault effect); Gaussian sensor
    noise; systematic combustion bias; missing-trim; probabilistic metrics (AUROC/AUPRC/Brier).
E5: task-aligned baselines (raw-threshold rule; trained full-feature logistic; likelihood-ratio
    prototype from source-calibrated Gaussians) vs frozen and fuel-only.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0); R={}

# ---- frozen driver signature from Zenodo (as p13) ----
def zfeats(x,y,z,t):
    mag=np.sqrt(x*x+y*y+z*z); dur=t[-1]-t[0] if len(t)>1 else 1
    return [mag.std(),np.percentile(mag,95),np.sum(mag>3)/dur]
rows,yb=[],[]
for i in [1,2,3]:
    a=pd.read_csv(f"{RAW}/zenodo_driving/Linear_Acceleration_{i}.csv",dtype=np.float32,usecols=[0,1,2,3])
    a.columns=["t","x","y","z"]; tv=a["t"].to_numpy(); ev=pd.read_csv(f"{RAW}/zenodo_driving/Labeled_events_{i}.csv")
    for _,r in ev.iterrows():
        m=(tv>=r["start"])&(tv<=r["end"])
        if m.sum()<20: continue
        w=a.loc[m]; rows.append(zfeats(w["x"].to_numpy(),w["y"].to_numpy(),w["z"].to_numpy(),w["t"].to_numpy())); yb.append(1 if int(r["target"])>=1 else 0)
    del a
from sklearn.preprocessing import StandardScaler as SS2
zsc=SS2().fit(np.array(rows)); Wz=LogisticRegression(max_iter=1000).fit(zsc.transform(np.array(rows)),np.array(yb)).coef_[0]

# ---- VED frozen scores ----
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean","speed_std","accel_p95","harsh_accel_per_km"]).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
KMAP=["speed_std","accel_p95","harsh_accel_per_km"]; gg=v.groupby("VehId")
Zk=((v[KMAP]-gg[KMAP].transform("mean"))/(gg[KMAP].transform("std")+1e-9))
kin_raw=StandardScaler().fit_transform(Zk.to_numpy())@Wz
comb_raw=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy()
kz=(kin_raw-np.nanmean(kin_raw))/np.nanstd(kin_raw); cz=(comb_raw-np.nanmean(comb_raw))/np.nanstd(comb_raw)
m=np.isfinite(kz)&np.isfinite(cz); kz,cz=kz[m],cz[m]; grp=v["VehId"].to_numpy()[m]; n=len(kz)
# raw (unstandardized) features for rule/full baselines
rawk=v.loc[m,"harsh_accel_per_km"].to_numpy(); rawt=v.loc[m,"ltft1_mean"].abs().to_numpy()
Sb=((14.7/13.68-1)*100-(14.7/14.07-1)*100)/v.groupby("VehId")["ltft1_mean"].std().median()
R["S_bench"]=round(float(Sb),2)
def sig(x): return 1/(1+np.exp(-x))

# ================= E4: mixed-cause 2D dominant-cause grid =================
grid=[0,0.5,1.0,1.5,2.0,2.5,3.0]; H=np.full((len(grid),len(grid)),np.nan)
for i,Sd in enumerate(grid):
    for j,Sf in enumerate(grid):
        if Sd==Sf: continue
        pred=((cz+Sf)>(kz+Sd)).astype(int); true=1 if Sf>Sd else 0
        H[i,j]=np.mean(pred==true)
R["E4_grid_rows_Sd"]=grid; R["E4_grid_cols_Sf"]=grid
R["E4_grid_acc"]=[[None if not np.isfinite(x) else round(float(x),3) for x in row] for row in H]

# ================= E4: stress at S_bench (pure two classes) =================
def make(S):
    X=np.vstack([np.c_[kz+S,cz],np.c_[kz,cz+S]]); y=np.r_[np.zeros(n),np.ones(n)]; g=np.r_[grp,grp]; return X,y,g
X,y,g=make(Sb)
def frozen_acc(Xk,Xc): return accuracy_score(y,(Xc>Xk).astype(int))
# noise
R["E4_noise"]={}
for s in [0.0,0.5,1.0,1.5]:
    Xk=X[:,0]+rng.normal(0,s,len(y)); Xc=X[:,1]+rng.normal(0,s,len(y))
    R["E4_noise"][str(s)]=round(frozen_acc(Xk,Xc),3)
# systematic combustion bias
R["E4_bias"]={}
for b in [0.0,0.5,1.0]:
    R["E4_bias"][str(b)]=round(frozen_acc(X[:,0],X[:,1]+b),3)
# missing trim (combustion score -> 0)
R["E4_missing_trim"]={}
for miss in [0.0,0.2,0.5]:
    Xc=X[:,1].copy(); idx=rng.random(len(y))<miss; Xc[idx]=0.0
    R["E4_missing_trim"][str(miss)]=round(frozen_acc(X[:,0],Xc),3)
# probabilistic metrics at S_bench (frozen prob = sigmoid(cz-kz))
p=sig(X[:,1]-X[:,0])
R["E4_metrics"]={"AUROC":round(roc_auc_score(y,p),3),"AUPRC":round(average_precision_score(y,p),3),
                 "Brier":round(brier_score_loss(y,p),3),"accuracy":round(accuracy_score(y,(p>0.5).astype(int)),3)}

# ================= E5: task-aligned baselines at S_bench =================
B={}
# frozen (reference)
B["frozen_argmax"]=round(frozen_acc(X[:,0],X[:,1]),3)
# (1) raw-threshold rule: compare raw harsh-rate vs raw |trim|, each vs its own median; predict the exceeded one
rk=np.r_[rawk,rawk]; rt=np.r_[rawt,rawt]
# inject the same shifts on the raw axes proportionally (approx): driver adds to harsh, fault to trim
rk2=rk+np.r_[np.full(n,Sb*np.std(rawk)),np.zeros(n)]; rt2=rt+np.r_[np.zeros(n),np.full(n,Sb*np.std(rawt))]
rule=((rt2/np.median(rt))>(rk2/np.median(rk))).astype(int)
B["rule_raw_threshold"]=round(accuracy_score(y,rule),3)
# (2) trained full-feature logistic (upper bound): use both scores + raw features, GroupKFold
Xfull=np.column_stack([X[:,0],X[:,1],rk2,rt2])
B["trained_full_logit"]=round(accuracy_score(y,cross_val_predict(
    make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),Xfull,y,cv=GroupKFold(5),groups=g)),3)
# (3) likelihood-ratio prototype: source-calibrated Gaussian means (driver: kin~+Sb, comb~0; fault: kin~0, comb~+Sb), unit var
def loglik(xk,xc,mk,mc): return -0.5*((xk-mk)**2+(xc-mc)**2)
lr=loglik(X[:,0],X[:,1],0,Sb)-loglik(X[:,0],X[:,1],Sb,0)   # >0 -> fault
B["likelihood_ratio_prototype"]=round(accuracy_score(y,(lr>0).astype(int)),3)
# fuel-only chance
fe=np.r_[rng.normal(0.06,0.01,n),rng.normal(0.06,0.01,n)]
B["fuel_only"]=round(accuracy_score(y,cross_val_predict(
    make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),fe.reshape(-1,1),y,cv=GroupKFold(5),groups=g)),3)
R["E5_baselines_acc_at_Sbench"]=B

# ---- heatmap figure ----
RED="#7a2020"
fig,ax=plt.subplots(figsize=(4.8,4.2))
im=ax.imshow(H,origin="lower",cmap="RdYlGn",vmin=0.5,vmax=1.0,aspect="equal")
ax.set_xticks(range(len(grid))); ax.set_xticklabels(grid); ax.set_yticks(range(len(grid))); ax.set_yticklabels(grid)
ax.set_xlabel("fault effect $S_f$"); ax.set_ylabel("driver effect $S_d$")
ax.set_title("Dominant-cause accuracy (mixed causes)")
for i in range(len(grid)):
    for j in range(len(grid)):
        if np.isfinite(H[i,j]): ax.text(j,i,f"{H[i,j]:.2f}",ha="center",va="center",fontsize=7,
                                        color="black")
fig.colorbar(im,ax=ax,shrink=0.8,label="accuracy"); fig.tight_layout()
fig.savefig(os.path.join(ROOT,"figures","mixed_cause_grid.svg"),bbox_inches="tight")
fig.savefig(os.path.join(ROOT,"figures","mixed_cause_grid.png"),bbox_inches="tight",dpi=150)

print(json.dumps({k:R[k] for k in ["S_bench","E4_noise","E4_bias","E4_missing_trim","E4_metrics","E5_baselines_acc_at_Sbench"]},indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p14_stress_baselines.json"),"w"),indent=2)
print("wrote p14_stress_baselines.json + figures/mixed_cause_grid")
