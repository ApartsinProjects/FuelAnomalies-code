"""P15 (EAAI E6): behaviour-axis validation on a SECOND dataset at VED-like 1 Hz.
UAH-DriveSet processed 1 Hz streams (6 drivers), obtained credential-free from a public
GitHub mirror (official server down; not redistributed here). Window into 30 s segments,
label by the dominant semantic ratio, and test aggressive-vs-normal separation with the
driver held out (6-fold LODO) using 1 Hz kinematic features.
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu

RAW=os.path.join(os.path.dirname(__file__),"..","data","raw","uah")
KFX="Acceleration in X filtered by KF (Gs)"; KFY="Acceleration in Y filtered by KF (Gs)"; KFZ="Acceleration in Z filtered by KF (Gs)"
SPD="Speed (km/h)"; TS="Timestamp (seconds)"
RN,RD,RA="Ratio normal (base 1)","Ratio drowsy (base 1)","Ratio aggressive (base 1)"
WIN=30  # seconds per window

def winfeats(w):
    ax,ay,az=w[KFX].to_numpy(),w[KFY].to_numpy(),w[KFZ].to_numpy()
    mag=np.sqrt(ax*ax+ay*ay+az*az); sp=w[SPD].to_numpy()
    jerk=np.diff(mag)
    return [mag.std(),np.percentile(mag,95),ay.std(),az.std(),np.sqrt(np.mean(jerk**2)),
            np.mean(np.abs(ay)>0.2),sp.std(),sp.mean()]
FEAT=["mag_std","mag_p95","lat_std","lon_std","jerk_rms","harsh_lat_rate","speed_std","speed_mean"]

rows,lab,drv=[],[],[]
for f in sorted(glob.glob(os.path.join(RAW,"D*_merged.csv"))):
    di=int(os.path.basename(f)[1])
    d=pd.read_csv(f)
    ts=d[TS].to_numpy(); seg=np.concatenate([[0],np.cumsum(np.diff(ts)<0)])  # recording id
    d=d.assign(_seg=seg)
    for _,rec in d.groupby("_seg"):
        rec=rec.reset_index(drop=True); t0=rec[TS].iloc[0]
        wid=((rec[TS]-t0)//WIN).astype(int)
        for _,w in rec.groupby(wid):
            if len(w)<15: continue
            rat=np.array([w[RN].sum(),w[RD].sum(),w[RA].sum()]); cls=int(np.argmax(rat))  # 0 norm,1 drowsy,2 aggr
            if cls==1: continue  # aggressive-vs-normal
            rows.append(winfeats(w)); lab.append(1 if cls==2 else 0); drv.append(di)
X=np.array(rows); y=np.array(lab); g=np.array(drv)
print(f"windows={len(X)}  aggressive={int(y.sum())}  normal={int((y==0).sum())}  drivers={sorted(set(g))}")

R={"n_windows":int(len(X)),"aggressive":int(y.sum()),"normal":int((y==0).sum()),"n_drivers":int(len(set(g))),"win_s":WIN}
# univariate separation
R["univariate_auc"]={}
for j,c in enumerate(FEAT):
    try: a=roc_auc_score(y,X[:,j])
    except Exception: a=np.nan
    R["univariate_auc"][c]=round(float(a),3)
# LODO classifiers
def lodo(model):
    pr=np.full(len(X),np.nan)
    for tr,te in LeaveOneGroupOut().split(X,y,g):
        pr[te]=model().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pr
lr=lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
rf=lambda:RandomForestClassifier(n_estimators=100,max_depth=8,random_state=0,n_jobs=1)
plr=lodo(lr); R["auc_logreg_LODO"]=round(roc_auc_score(y,plr),3); R["auc_rf_LODO"]=round(roc_auc_score(y,lodo(rf)),3)
R["per_driver_auc"]={}
for di in sorted(set(g)):
    m=g==di
    if len(set(y[m]))==2: R["per_driver_auc"][f"D{di}"]=round(roc_auc_score(y[m],plr[m]),3)
# permutation (1000)
rng=np.random.default_rng(0); perm=[]
for _ in range(1000):
    yp=rng.permutation(y); perm.append(roc_auc_score(yp,lodo(lr)))
R["perm_mean"]=round(float(np.mean(perm)),3); R["perm_p"]=round((1+np.sum(np.array(perm)>=R["auc_logreg_LODO"]))/(1+len(perm)),4)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(os.path.dirname(__file__),"..","data","processed","p15_uah_e6.json"),"w"),indent=2)
print("wrote p15_uah_e6.json")
