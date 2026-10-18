"""P20 (Fable item-4): what fuel increase does the realistic driver injection actually cause?
Push the P17 feature-level driver perturbation (calibrated to Cohen's d = 0.65) through the
expected-fuel model M, so the 'same fuel increase for both causes' premise becomes a measured
statement, and report the driver injection's real fuel cost vs the bench fault's +6.2%.
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")

# frozen UAH driver signature (as p17)
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

# VED: fit expected-fuel model M on behav+env; build the P17 driver perturbation
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
VF=["speed_std","accel_p95","harsh_accel_per_km","jerk_rms"]
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean"]+VF).copy(); cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
BEHAV=["speed_mean","speed_std","speed_p85","accel_pos_mean","accel_p95","decel_p05","jerk_rms",
       "harsh_accel_per_km","harsh_brake_per_km","idle_frac","stops_per_km","pct_hwy","vsp_mean"]
ENV=["oat_mean","ac_w_mean","heat_w_mean","weight_lb","dist_km","dur_min"]
M=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0).fit(
    v[BEHAV+ENV].to_numpy(), v["fuel_per_100km"].to_numpy())
sv=StandardScaler().fit(v[VF]); Zv=sv.transform(v[VF]); kraw=Zv@Wk
PERT=(S_driver*kraw.std()/(Wk@dirn))*dirn            # standardized-feature step
raw_delta=PERT*sv.scale_                              # un-standardize to raw VF units
# baseline vs perturbed fuel (perturb only the 4 VF columns, clip to real ranges)
base=M.predict(v[BEHAV+ENV].to_numpy())
vp=v[BEHAV+ENV].copy()
for i,c in enumerate(VF):
    vp[c]=np.clip(v[c].to_numpy()+raw_delta[i], v[c].quantile(.001), v[c].quantile(.999))
pert=M.predict(vp.to_numpy())
dpct=(pert-base)/base*100.0
R={"S_driver":round(float(S_driver),2),"raw_delta_per_feature":{c:round(float(raw_delta[i]),4) for i,c in enumerate(VF)},
   "driver_injection_fuel_pct_median":round(float(np.median(dpct)),2),
   "driver_injection_fuel_pct_mean":round(float(np.mean(dpct)),2),
   "driver_injection_fuel_pct_p90":round(float(np.percentile(dpct,90)),2),
   "bench_fault_fuel_pct":6.2}
print(json.dumps(R,indent=2)); json.dump(R,open(os.path.join(ROOT,"data","processed","p20_driver_fuelcost.json"),"w"),indent=2)
print("wrote p20_driver_fuelcost.json")
