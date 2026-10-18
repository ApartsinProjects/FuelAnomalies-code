"""Generate publication figures (SVG + PNG). Figs 1-3,5 plot validated scalars
(traceable to RESULTS.md and scripts p3/p5b/p6). Fig 4 recomputes the ROC in a
low-memory pass. Kept lean for a memory-constrained host.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RAW = os.path.join(ROOT, "data", "raw")

INK="#1a1a1a"; MUT="#666"; RED="#7a2020"; BLUE="#2c5f8a"; GREEN="#0b8a4b"; GRAY="#9a9a9a"; SAND="#d9cfc0"
plt.rcParams.update({
    "font.family":"serif","font.serif":["Georgia","Times New Roman","DejaVu Serif"],
    "font.size":11,"axes.edgecolor":"#888","axes.linewidth":0.8,
    "axes.titlesize":12.5,"axes.titleweight":"bold","figure.dpi":120,
    "svg.fonttype":"none","text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
def save(fig,name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG,name+".svg"),bbox_inches="tight")
    fig.savefig(os.path.join(FIG,name+".png"),bbox_inches="tight",dpi=150)
    plt.close(fig); print("wrote",name)

# ---- Fig 1: variance decomposition (p3) ----
parts=[("Environment / route",0.593,GRAY),("Driving behaviour (marginal)",0.080,RED),
       ("Vehicle-baseline",0.143,BLUE),("Unexplained",0.184,SAND)]
fig,ax=plt.subplots(figsize=(7,2.5)); left=0
for lab,val,col in parts:
    ax.barh(0,val,left=left,color=col,edgecolor="white",height=0.6)
    if val>0.03: ax.text(left+val/2,0,f"{val:.2f}",ha="center",va="center",
                         color="white" if col!=SAND else INK,fontsize=10,fontweight="bold")
    left+=val
ax.set_xlim(0,1); ax.set_ylim(-.5,.9); ax.set_yticks([]); ax.set_xlabel("share of trip fuel variance (R²)")
ax.set_title("Variance decomposition of trip fuel (VED, 197 vehicles)")
ax.legend([plt.Rectangle((0,0),1,1,color=c) for _,_,c in parts],[p[0] for p in parts],
          ncol=2,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.5,-0.55),frameon=False)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
save(fig,"fig1_variance_decomposition")

# ---- Fig 2: double dissociation (p6) ----
x=np.arange(2); w=0.36
kin=[0.084,np.nan]; comb=[0.0,0.903]   # ved_comb=-0.028 -> shown as 0 (silent)
fig,ax=plt.subplots(figsize=(5.6,3.6))
b1=ax.bar(x-w/2,kin,w,color=RED,label="Kinematic axis")
b2=ax.bar(x+w/2,comb,w,color=BLUE,label="Combustion axis")
ax.text(1-w/2,0.03,"n/a\n(steady-state)",ha="center",va="bottom",fontsize=8,color=MUT,style="italic")
ax.text(0+w/2,0.03,"~0 (silent)",ha="center",va="bottom",fontsize=8,color=MUT,style="italic")
for b,val in zip(b1,kin):
    if np.isfinite(val): ax.text(b.get_x()+b.get_width()/2,val+.02,f"{val:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.text(b2[1].get_x()+b2[1].get_width()/2,comb[1]+.02,f"{comb[1]:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Driver-excess\n(VED)","Fault-excess\n(bench)"]); ax.set_ylim(0,1.0)
ax.set_ylabel("variance of excess explained (R²)"); ax.set_title("Signature contrast")
ax.legend(fontsize=9,frameon=False,loc="upper left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
save(fig,"signature_contrast")

# ---- Fig 3: behaviour counterfactual by aggression decile (p3) ----
mono=[0.34,0.42,0.39,0.41,0.49,0.58,0.65,0.78,0.99,1.55]
fig,ax=plt.subplots(figsize=(5.6,3.2))
ax.plot(range(10),mono,"-o",color=RED,lw=2,ms=5); ax.fill_between(range(10),0,mono,color=RED,alpha=0.08)
ax.set_xlabel("aggression decile (0 = gentle, 9 = harsh)"); ax.set_ylabel("behaviour component (L/100km)")
ax.set_title("Behaviour component rises with aggression"); ax.set_xticks(range(10))
for s in ["top","right"]: ax.spines[s].set_visible(False)
save(fig,"fig3_behaviour_monotonicity")

# ---- Fig 5: rich-fault fuel signature (p5b) ----
names=["normal","rich","lean","ignition"]; exc=[0.0,0.498,-0.060,0.148]; afr=[14.07,13.68,14.43,14.39]
cols=[GRAY,RED,BLUE,"#b58a00"]
fig,ax=plt.subplots(figsize=(5.6,3.4))
b=ax.bar(names,exc,color=cols,edgecolor="white"); ax.axhline(0,color="#888",lw=0.8)
for i,bar in enumerate(b):
    ax.text(bar.get_x()+bar.get_width()/2, exc[i]+(0.02 if exc[i]>=0 else -0.04),
            f"AFR {afr[i]:.1f}",ha="center",va="bottom" if exc[i]>=0 else "top",fontsize=8,color=MUT)
ax.set_ylabel("excess fuel vs normal model (L/100km)")
ax.set_title("Rich mixture is the fuel-relevant fault (EngineFaultDB)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
save(fig,"fig5_fault_signature")

# ---- Fig 4: behaviour-axis validation ROC (recompute, low-memory) ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, roc_auc_score
def feats(x,y,z,t):
    mag=np.sqrt(x*x+y*y+z*z); dt=np.median(np.diff(t)) if len(t)>1 else .0025
    jk=np.diff(mag)/dt if len(mag)>1 else np.array([0.],np.float32); dur=t[-1]-t[0] if len(t)>1 else 1
    return [mag.mean(),mag.std(),np.percentile(mag,95),mag.max(),np.sqrt(np.mean(jk*jk)),
            np.percentile(np.abs(jk),95),np.sum(mag>3)/dur,x.std(),y.std(),z.std(),np.mean(mag*mag)]
rows,yb,grp=[],[],[]
for i in [1,2,3]:
    a=pd.read_csv(f"{RAW}/zenodo_driving/Linear_Acceleration_{i}.csv",dtype=np.float32,
                  usecols=[0,1,2,3]); a.columns=["t","x","y","z"]; tv=a["t"].to_numpy()
    ev=pd.read_csv(f"{RAW}/zenodo_driving/Labeled_events_{i}.csv")
    for _,r in ev.iterrows():
        m=(tv>=r["start"])&(tv<=r["end"])
        if m.sum()<20: continue
        w=a.loc[m]; rows.append(feats(w["x"].to_numpy(),w["y"].to_numpy(),w["z"].to_numpy(),w["t"].to_numpy()))
        yb.append(1 if int(r["target"])>=1 else 0); grp.append(i)
    del a
X=np.array(rows); yb=np.array(yb); grp=np.array(grp); probs=np.full(len(X),np.nan)
for tr,te in LeaveOneGroupOut().split(X,yb,grp):
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(X[tr],yb[tr])
    probs[te]=clf.predict_proba(X[te])[:,1]
fpr,tpr,_=roc_curve(yb,probs); auc=roc_auc_score(yb,probs)
fig,ax=plt.subplots(figsize=(4.4,4.2))
ax.plot(fpr,tpr,color=GREEN,lw=2.2,label=f"kinematic axis (AUC = {auc:.3f})")
ax.plot([0,1],[0,1],"--",color=GRAY,lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Behaviour-axis validation\n(aggressive vs normal, driver held out)")
ax.legend(fontsize=9,frameon=False,loc="lower right"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
for s in ["top","right"]: ax.spines[s].set_visible(False)
save(fig,"fig4_behaviour_roc")
print("done; ROC AUC=%.3f"%auc)
