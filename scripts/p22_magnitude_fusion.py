"""P22 (Fable Edit 6): DEMONSTRATE the conservative-floor claim.

The headline 78.5%/81.2% is measured under an equal-fuel injection that deliberately withholds
magnitude. In reality the two causes are NOT fuel-matched (p20: driver +1.07% median vs fault +6.2%),
so magnitude is cause-informative. Here we add excess-magnitude as a THIRD likelihood channel to the
p19 fusion, calibrated to the REAL per-cause fuel effects, and measure the accuracy gain.

INVARIANTS (stated in advance; a violation is a bug, not a finding):
  (I1) 3-channel accuracy >= 2-channel accuracy  (adding real information cannot hurt Bayes fusion).
  (I2) magnitude-only accuracy > 0.5             (because 1.07% and 6.2% are separated).
  (I3) with EQUAL magnitudes (both causes +6.2%), the magnitude channel adds ~0 (sanity: recovers floor).
The magnitude-channel NOISE is the real out-of-fold fuel residual (sd in %), so the gain is a
CONSERVATIVE lower bound: real flagged trips sit in the upper excess tail where separation is cleaner.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import norm as Nrm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
rng=np.random.default_rng(0)

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

# ---------- VED trip subset (identical filter to p17/p19/p20) ----------
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
v = v.drop(columns=["speed_std", "accel_p95", "harsh_accel_per_km", "jerk_rms"]).merge(pd.read_parquet(os.path.join(ROOT,"data","processed","signature_features.parquet")), on=["VehId", "Trip"], how="left")   # signature features with the source (UAH) definition, p63
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy(); v=v.reset_index(drop=True)
grp=v["VehId"].to_numpy(); n=len(v); gg=v.groupby("VehId")
sv=StandardScaler().fit(v[VF]); Zv=sv.transform(v[VF]); kraw=Zv@Wk
cz0=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).to_numpy(); cz0 = cz0 / cz0.std()   # exact unit pooled variance (fault shift = S_fault exactly)
S_fault=3.49/v.groupby("VehId")["ltft1_mean"].std().median()  # LiU-ICE smallest measured fueling deviation, pp (p37)

# ---------- per-trip DRIVER fuel effect (%) via model M and the p20 perturbation ----------
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
M=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0).fit(
    v[BEHAV+ENV].to_numpy(), v["fuel_per_100km"].to_numpy())
PERT=(S_driver*kraw.std()/(Wk@dirn))*dirn; raw_delta=PERT*sv.scale_
base=M.predict(v[BEHAV+ENV].to_numpy()); vp=v[BEHAV+ENV].copy()
for i,c in enumerate(VF): vp[c]=np.clip(v[c].to_numpy()+raw_delta[i], v[c].quantile(.001), v[c].quantile(.999))
dpct=(M.predict(vp.to_numpy())-base)/base*100.0            # per-trip driver fuel effect, %
mu_dr=float(np.median(dpct)); mu_ft=0.32                    # fault fuel change of the LiU-ICE anchor fault (p37 WLTP g=1.05)

# ---------- magnitude-channel NOISE = real OOF fuel residual (%) ----------
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
yfuel=v["fuel_per_100km"].to_numpy()
oof=cross_val_predict(HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0),
                      v[CTX],yfuel,cv=GroupKFold(5),groups=grp)
resid_pct=(yfuel-oof)/oof*100.0
s_mag=float(np.std(resid_pct))                             # honest per-trip magnitude noise (%)

# ---------- build the two-class injection (as p19) + magnitude observations ----------
def z(x,ref): return (x-ref.mean())/ref.std()
kz_dr=z((Zv+PERT)@Wk,kraw); kz_no=z(kraw,kraw); cz_z=z(cz0,cz0); cz_f=z(cz0+S_fault,cz0)
KZ=np.r_[kz_dr,kz_no]; CZ=np.r_[cz_z,cz_f]; y=np.r_[np.zeros(n),np.ones(n)]; g=np.r_[grp,grp]
eps=rng.choice(resid_pct,size=n,replace=True)              # one real base-trip noise per base trip
# observed magnitude the attributor sees: cause effect + the SAME base-trip noise for both variants
MAGr=np.r_[dpct+eps, mu_ft+eps]                            # class0 driver ~1%, class1 fault ~6.2%
MAGeq=np.r_[mu_ft+eps, mu_ft+eps]                          # I3 control: equal magnitudes

def acc_fuse(use_mag, MAG):
    lLd=Nrm.logpdf(KZ,S_driver,1)+Nrm.logpdf(CZ,0,1)
    lLf=Nrm.logpdf(KZ,0,1)+Nrm.logpdf(CZ,S_fault,1)
    if use_mag:
        lLd=lLd+Nrm.logpdf(MAG,mu_dr,s_mag); lLf=lLf+Nrm.logpdf(MAG,mu_ft,s_mag)
    pred=(lLf>lLd).astype(int); return float((pred==y).mean())
def acc_magonly(MAG):
    lLd=Nrm.logpdf(MAG,mu_dr,s_mag); lLf=Nrm.logpdf(MAG,mu_ft,s_mag)
    return float(((lLf>lLd).astype(int)==y).mean())

two=acc_fuse(False,MAGr); three=acc_fuse(True,MAGr); magonly=acc_magonly(MAGr); eqctl=acc_fuse(True,MAGeq)
R={"S_driver":round(float(S_driver),2),"S_fault":round(float(S_fault),2),
   "driver_fuel_pct_median":round(mu_dr,2),"fault_fuel_pct":mu_ft,"mag_noise_sd_pct":round(s_mag,2),
   "acc_2channel_equalmag_floor":round(two,3),"acc_3channel_with_magnitude":round(three,3),
   "gain":round(three-two,3),"acc_magnitude_only":round(magonly,3),
   "acc_3channel_equalmag_control":round(eqctl,3),
   "I1_three_ge_two":bool(three>=two-1e-9),"I2_magonly_above_chance":bool(magonly>0.5),
   "I3_equalmag_recovers_floor":bool(abs(eqctl-two)<0.01)}
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p22_magnitude_fusion.json"),"w"),indent=2)
print("wrote p22_magnitude_fusion.json")
