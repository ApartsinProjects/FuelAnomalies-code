"""P16: two real-data strengthening analyses.
#1 High-trim real-fault proxy + specificity: among REAL VED trips, do kinematic-dominant
   trips carry fuel excess (driver signature -> real fuel) while combustion(trim)-dominant
   trips do not (healthy fleet / ECU compensates)? Real data, no injection.
#3 Fault-signature specificity: does the vehicle-fuel-fault signature fire only for the
   rich (fuel-relevant) fault and correctly reject lean/ignition on the real bench?
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from scipy.stats import mannwhitneyu

ROOT=os.path.join(os.path.dirname(__file__),".."); RAW=os.path.join(ROOT,"data","raw")
HGB=lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=5,random_state=0)
R={}

# ================= #1 real high-trim vs high-kinematic, within VED =================
v=pd.read_parquet(os.path.join(ROOT,"data","processed","trip_features.parquet"))
lo,hi=v["fuel_per_100km"].quantile([.01,.99]); v=v[(v.fuel_per_100km>=lo)&(v.fuel_per_100km<=hi)].copy()
v=v[v["ltft1_cov"]>=0.2].dropna(subset=["ltft1_mean","speed_std","accel_p95","harsh_accel_per_km"]).copy()
cnt=v.groupby("VehId")["ltft1_mean"].transform("size"); v=v[cnt>=15].copy()
CTX=["oat_mean","ac_w_mean","heat_w_mean","dist_km","dur_min","weight_lb","speed_mean","pct_hwy"]
gt=v["VehId"].to_numpy()
v["excess"]=v["fuel_per_100km"].to_numpy()-cross_val_predict(HGB(),v[CTX],v["fuel_per_100km"],cv=GroupKFold(5),groups=gt)
KIN=["speed_std","accel_pos_mean","accel_p95","decel_p05","jerk_rms","harsh_accel_per_km","harsh_brake_per_km","vsp_mean"]
gg=v.groupby("VehId")
zk=((v[KIN]-gg[KIN].transform("mean"))/(gg[KIN].transform("std")+1e-9)).mean(axis=1)
zc=((v["ltft1_mean"]-gg["ltft1_mean"].transform("mean"))/(gg["ltft1_mean"].transform("std")+1e-9)).abs()  # |trim dev|
v["zk"]=zk.values; v["zc"]=zc.values
kp75,cp75=v.zk.quantile(.75),v.zc.quantile(.75); kp50,cp50=v.zk.quantile(.5),v.zc.quantile(.5)
kin_dom=v[(v.zk>kp75)&(v.zc<cp50)]; comb_dom=v[(v.zc>cp75)&(v.zk<kp50)]; both_lo=v[(v.zk<kp50)&(v.zc<cp50)]
def m(df): return round(float(df["excess"].mean()),3),round(float(df["excess"].median()),3),len(df)
R["#1_excess_kinematic_dominant"]=m(kin_dom)
R["#1_excess_combustion_dominant"]=m(comb_dom)
R["#1_excess_both_low_ref"]=m(both_lo)
u,p=mannwhitneyu(kin_dom["excess"],comb_dom["excess"])
R["#1_MWU_kin_vs_comb_p"]=float(f"{p:.2e}")
# does excess track kinematic but not combustion, on REAL trips?
R["#1_corr_excess_zk"]=round(float(np.corrcoef(v.zk,v.excess)[0,1]),3)
R["#1_corr_excess_zc"]=round(float(np.corrcoef(v.zc,v.excess)[0,1]),3)
# within highest-combustion quartile, does excess track combustion? (real fault proxy)
hi_c=v[v.zc>cp75]
R["#1_hicomb_corr_excess_zc"]=round(float(np.corrcoef(hi_c.zc,hi_c.excess)[0,1]),3)
R["#1_hicomb_mean_excess"]=round(float(hi_c.excess.mean()),3)

# ================= #3 fault-signature specificity on the real bench =================
e=pd.read_csv(os.path.join(RAW,"EngineFaultDB_Final.csv"))
OP=["RPM","MAP","TPS","Force","Power","Speed"]; norm=e[e.Fault==0]
M=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=6,random_state=0).fit(
    norm[OP].to_numpy(),norm["Consumption L/100KM"].to_numpy())
e["excess"]=e["Consumption L/100KM"].to_numpy()-M.predict(e[OP].to_numpy())
e["comb"]=(14.7/e["AFR"]-1)*100  # combustion-axis magnitude (% fuel-air vs stoich)
names={0:"normal",1:"rich",2:"lean",3:"ignition"}
# fuel-fault signature score = excess AND elevated combustion (rich = high fuel-air)
base=e.loc[e.Fault==0,"comb"].median()
R["#3_by_fault"]={}
for k in [0,1,2,3]:
    sub=e[e.Fault==k]
    R["#3_by_fault"][names[k]]={"excess_mean":round(float(sub.excess.mean()),3),
        "comb_pct_mean":round(float(sub.comb.mean()),2),"comb_vs_normal":round(float(sub.comb.mean()-base),2)}
# specificity: flag a fuel-relevant fault if excess>thr AND comb elevated vs normal.
thr=0.3
def flag(sub): return float(((sub.excess>thr)&(sub.comb>base+1.0)).mean())
R["#3_flag_rate"]={names[k]:round(flag(e[e.Fault==k]),3) for k in [0,1,2,3]}
# TPR(rich) vs FPR(lean,ignition,normal)
R["#3_TPR_rich"]=R["#3_flag_rate"]["rich"]
R["#3_FPR_nonfuel"]=round(float(np.mean([R["#3_flag_rate"]["lean"],R["#3_flag_rate"]["ignition"],R["#3_flag_rate"]["normal"]])),3)

print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(ROOT,"data","processed","p16_strengthen.json"),"w"),indent=2)
print("wrote p16_strengthen.json")
