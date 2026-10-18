import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
RAW='data/raw/zenodo_driving'
def feats(x,y,z,t):
    mag=np.sqrt(x*x+y*y+z*z); dt=np.median(np.diff(t)) if len(t)>1 else .0025
    jerk=np.diff(mag)/dt if len(mag)>1 else np.array([0.],np.float32); dur=t[-1]-t[0] if len(t)>1 else 1
    return [mag.mean(),mag.std(),np.percentile(mag,95),mag.max(),np.sqrt(np.mean(jerk**2)),
            np.percentile(np.abs(jerk),95),np.sum(mag>3)/dur,x.std(),y.std(),z.std(),np.mean(mag*mag)]
rows,yb,grp=[],[],[]
for i in [1,2,3]:
    a=pd.read_csv(f'{RAW}/Linear_Acceleration_{i}.csv',dtype=np.float32)
    a.columns=['t','x','y','z']; tv=a['t'].to_numpy()
    e=pd.read_csv(f'{RAW}/Labeled_events_{i}.csv')
    for _,r in e.iterrows():
        m=(tv>=r['start'])&(tv<=r['end'])
        if m.sum()<20: continue
        w=a.loc[m]; rows.append(feats(w['x'].to_numpy(),w['y'].to_numpy(),w['z'].to_numpy(),w['t'].to_numpy()))
        yb.append(1 if int(r['target'])>=1 else 0); grp.append(i)
    del a
X=np.array(rows,np.float64); yb=np.array(yb); grp=np.array(grp)
print('events',len(X),'nan/inf',np.isnan(X).any(),np.isinf(X).any(),'base_rate',round(yb.mean(),3))
def logo_auc(y):
    pr=np.full(len(X),np.nan)
    for tr,te in LeaveOneGroupOut().split(X,y,grp):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)); clf.fit(X[tr],y[tr])
        pr[te]=clf.predict_proba(X[te])[:,1]
    return roc_auc_score(y,pr)
print('REAL logreg LOGO AUC:',round(logo_auc(yb),3))
rng=np.random.default_rng(0)
perm=[logo_auc(rng.permutation(yb)) for _ in range(30)]
print('PERMUTED AUC: mean=%.3f  95th=%.3f  max=%.3f'%(np.mean(perm),np.percentile(perm,95),np.max(perm)))
p=(1+sum(pp>=logo_auc(yb) for pp in perm))/(1+len(perm))
print('permutation p-value:',round(p,4))
