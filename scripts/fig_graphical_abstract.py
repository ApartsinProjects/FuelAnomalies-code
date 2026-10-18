import matplotlib, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=os.path.join(os.path.dirname(__file__),"..")
RED="#7a2020";BLUE="#2c5f8a";GREEN="#0b8a4b";INK="#1a1a1a";GRAY="#8a8a8a"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Segoe UI","DejaVu Sans"],"svg.fonttype":"none"})
fig,ax=plt.subplots(figsize=(13.28,5.31)); ax.set_xlim(0,133); ax.set_ylim(0,53); ax.axis("off")
def box(x,y,w,h,ec,fc): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=1.6",lw=1.6,ec=ec,fc=fc))
def arw(x1,y1,x2,y2,c="#555"): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,lw=1.6,color=c))
ax.text(66,50,"Driver or Fueling Fault? Attributing excess fuel by signature",ha="center",fontsize=15,fontweight="bold",color=INK)
# symptom
box(3,20,28,18,GRAY,"#f2f2f2")
ax.text(17,33,"Excess fuel",ha="center",fontsize=13,fontweight="bold",color=INK)
ax.text(17,27.5,"same symptom",ha="center",fontsize=10.5,color=INK)
ax.text(17,23,"driver?  fueling fault?",ha="center",fontsize=11,style="italic",color=GRAY)
# two signatures
box(46,31,42,15,RED,"#f3eded")
ax.text(67,41.5,"Kinematic signature  →  DRIVER",ha="center",fontsize=12,fontweight="bold",color=RED)
ax.text(67,35,"transient: acceleration, jerk, harsh events",ha="center",fontsize=10,color=INK)
box(46,10,42,15,BLUE,"#eef2f6")
ax.text(67,20.5,"Fueling signature  →  FUELING FAULT",ha="center",fontsize=12,fontweight="bold",color=BLUE)
ax.text(67,14,"steady-state: air-fuel ratio, long-term fuel trim",ha="center",fontsize=10,color=INK)
# result
box(100,20,30,18,GREEN,"#eef5ee")
ax.text(115,33.5,"Attribution",ha="center",fontsize=13,fontweight="bold",color=GREEN)
ax.text(115,28,"85%",ha="center",fontsize=17,fontweight="bold",color=GREEN)
ax.text(115,23.5,"injected cases; fuel size alone: chance",ha="center",fontsize=9.5,color=GRAY)
arw(31,29,45,38,RED); arw(31,29,45,18,BLUE)
arw(88,38,100,31,RED); arw(88,17,100,27,BLUE)
ax.text(66,4,"Frozen source signatures, likelihood-ratio fusion, conformal abstention  ·  public data: 197-car fleet, two engine benches, four driving datasets",
        ha="center",fontsize=9.5,color=GRAY)
fig.tight_layout()
fig.savefig(os.path.join(ROOT,"figures","graphical_abstract.png"),bbox_inches="tight",dpi=100)
fig.savefig(os.path.join(ROOT,"SUBMISSION","graphical_abstract.png"),bbox_inches="tight",dpi=110)
print("wrote graphical_abstract")
