import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
ROOT=os.path.join(os.path.dirname(__file__),"..")
c=pd.read_csv(os.path.join(ROOT,"data","processed","attribution_curve.csv"))
INK="#1a1a1a";RED="#7a2020";BLUE="#2c5f8a";GRAY="#9a9a9a";GREEN="#0b8a4b"
plt.rcParams.update({"font.family":"serif","font.serif":["Georgia","DejaVu Serif"],"font.size":11,
 "svg.fonttype":"none","axes.edgecolor":"#888","axes.titleweight":"bold"})
fig,ax=plt.subplots(figsize=(5.8,3.6))
ax.plot(c.S,c.signature_acc,"-o",color=GREEN,lw=2.2,ms=5,label="signature attributor (kinematic + combustion)")
ax.plot(c.S,c.fuel_only_acc,"--s",color=GRAY,lw=1.6,ms=4,label="fuel magnitude only (baseline)")
ax.axhline(0.5,color=GRAY,lw=0.7,ls=":")
ax.axvline(1.66,color=RED,lw=1.2,ls="--"); ax.plot(1.66,0.959,"*",color=RED,ms=15)
ax.annotate("bench-calibrated\nfault (S≈1.66): 96%",xy=(1.66,0.959),xytext=(1.9,0.72),
            fontsize=9,color=RED,arrowprops=dict(arrowstyle="->",color=RED,lw=1))
ax.set_xlabel("signature strength S (within-vehicle SD units)")
ax.set_ylabel("attribution accuracy"); ax.set_ylim(0.45,1.02)
ax.set_title("Attribution accuracy: signature vs fuel magnitude")
ax.legend(fontsize=8.5,frameon=False,loc="center right")
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(ROOT,"figures","fig6_attribution_accuracy.svg"),bbox_inches="tight")
fig.savefig(os.path.join(ROOT,"figures","fig6_attribution_accuracy.png"),bbox_inches="tight",dpi=150)
print("wrote fig6")
