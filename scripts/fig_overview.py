import matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=os.path.join(os.path.dirname(__file__),"..")
RED="#7a2020";BLUE="#2c5f8a";GREEN="#0b8a4b";INK="#1a1a1a";GRAY="#555555"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Segoe UI","DejaVu Sans"],"svg.fonttype":"none"})
fig,ax=plt.subplots(figsize=(8.4,3.6)); ax.set_xlim(0,100); ax.set_ylim(-3,51); ax.axis("off")
def box(x,y,w,h,title,lines,ec,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",lw=1.3,ec=ec,fc=fc))
    ax.text(x+w/2,y+h-2.6,title,ha="center",va="top",fontsize=9.5,fontweight="bold",color=ec)
    for i,ln in enumerate(lines): ax.text(x+w/2,y+h-6.0-i*2.9,ln,ha="center",va="top",fontsize=7.8,color=INK)
def arrow(x1,y1,x2,y2,c="#555"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=11,lw=1,color=c))
box(1,38,27,11,"Driver source: UAH-DriveSet",["aggressive vs normal labels","speed-only signature, frozen"],RED,"#f3eded")
box(1,25,27,11,"Fault benches",["LiU-ICE: measured fueling faults","EngineFaultDB: emission signature"],BLUE,"#eef2f6")
box(1,12,27,11,"Target fleet: VED",["197 cars, 14,460 trips","fuel from air flow, no cause labels"],GRAY,"#f4f4f4")
box(1,-1,27,11,"Replications",["Kia, Sonata (fixed cars)","Opel fill-ups, Driving Events"],GREEN,"#eef5ee")
box(38,30,26,12,"Kinematic axis",["acceleration, jerk,","harsh events (1 Hz speed)"],RED,"#fff")
box(38,10,26,12,"Fuel-trim axis",["within-vehicle long-term trim","(bench: air-fuel ratio, CO, HC)"],BLUE,"#fff")
box(73,15,26,22,"Attribution",["driver vs fueling fault","likelihood-ratio fusion","conformal abstention","85% on injected trips"],GREEN,"#eef5ee")
arrow(28,43,37.5,37); arrow(28,30,37.5,17)
arrow(28,17,37.5,33,"#999"); arrow(28,17,37.5,15,"#999")
arrow(64,36,72.5,29); arrow(64,16,72.5,23)
fig.tight_layout()
fig.savefig(os.path.join(ROOT,"figures","overview.svg"),bbox_inches="tight")
fig.savefig(os.path.join(ROOT,"figures","overview.png"),bbox_inches="tight",dpi=150)
print("wrote overview")
