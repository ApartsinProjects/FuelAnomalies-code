import pandas as pd, numpy as np, matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.join(os.path.dirname(__file__),"..")
d=pd.read_csv(os.path.join(ROOT,"data","processed","trip_decomposition.csv"))
d["gentle"]=d["pred_actual"]-d["behav_comp"]; d["resid_final"]=d["fuel_per_100km"]-d["pred_actual"]-d["veh_fe"]
def row(vid,tr): return d[(d.VehId==vid)&(d.Trip==tr)].iloc[0]
A=row(416,776); B=row(394,185)
INK="#1a1a1a";RED="#7a2020";BLUE="#2c5f8a";GRAY="#9a9a9a";SAND="#cdbfae"
plt.rcParams.update({"font.family":"serif","font.serif":["Georgia","DejaVu Serif"],"font.size":10.5,
 "svg.fonttype":"none","axes.edgecolor":"#888","axes.titleweight":"bold"})
def waterfall(ax,r,title,verdict,vcol):
    steps=[("gentle\nbaseline",r.gentle,GRAY),("driving\nbehaviour",r.behav_comp,RED),
           ("vehicle\nbaseline",r.veh_fe,BLUE),("residual",r.resid_final,SAND)]
    x=0; base=0
    for i,(lab,val,col) in enumerate(steps):
        if i==0:
            ax.bar(i,val,color=col,edgecolor="white"); base=val
            ax.text(i,val+0.3,f"{val:.1f}",ha="center",fontsize=9)
        else:
            ax.bar(i,val,bottom=base,color=col,edgecolor="white")
            ax.text(i,base+val+(0.3 if val>=0 else -0.5),f"{val:+.1f}",ha="center",fontsize=9,
                    color=col if abs(val)>0.5 else "#888",fontweight="bold" if abs(val)>1 else "normal")
            base+=val
    ax.bar(4,r.fuel_per_100km,color="#333",edgecolor="white")
    ax.text(4,r.fuel_per_100km+0.3,f"{r.fuel_per_100km:.1f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels([s[0] for s in steps]+["actual\nfuel"],fontsize=8.5)
    ax.set_title(title,fontsize=11.5); ax.set_ylim(0,22)
    ax.text(0.5,0.93,verdict,transform=ax.transAxes,fontsize=10,fontweight="bold",color=vcol,ha="center",
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=vcol))
    for s in ["top","right"]: ax.spines[s].set_visible(False)
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4),sharey=True)
a1.set_ylabel("fuel (L/100km)")
waterfall(a1,A,"Trip A (vehicle 416)","driver-signature dominated",RED)
waterfall(a2,B,"Trip B (vehicle 394)","persistent vehicle residual",BLUE)
fig.suptitle("Same symptom, different cause: per-trip excess-fuel decomposition",fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(ROOT,"figures","fig7_worked_example.svg"),bbox_inches="tight")
fig.savefig(os.path.join(ROOT,"figures","fig7_worked_example.png"),bbox_inches="tight",dpi=150)
print("wrote fig7; A excess=%.1f (behav %.1f/veh %.1f), B excess=%.1f (behav %.1f/veh %.1f)"%(
 A.fuel_per_100km-A.gentle,A.behav_comp,A.veh_fe,B.fuel_per_100km-B.gentle,B.behav_comp,B.veh_fe))
