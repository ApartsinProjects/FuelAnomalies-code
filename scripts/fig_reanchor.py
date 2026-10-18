"""Figures after the fault re-anchor (2026-10-18): (1) patch the fault-magnitude marker in fig_attribution_corrected.py
and rerun it; (2) signature contrast on three real datasets (p48); (3) bench operating conditions: emission signature
follows the measured mixture, not the label (p47)."""
import json, os, re, subprocess, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); FIG = os.path.join(ROOT, "figures")
p = os.path.join(ROOT, "scripts", "fig_attribution_corrected.py"); s = open(p, encoding="utf-8").read()
s = re.sub(r'ax\.axvline\(1\.66,.*?color=BLUE\)',
           'SF=R["S_fault_bench"]; ax.axvline(SF,color=BLUE,lw=1,ls="--",alpha=0.6); ax.text(SF+0.02,0.50,"fault magnitude"+chr(10)+"(measured, %.2f)"%SF,fontsize=8,color=BLUE)', s)
open(p, "w", encoding="utf-8").write(s)
subprocess.run([sys.executable, "-P", p], check=True)
RED = "#7a2020"; BLUE = "#2c5f8a"; GRAY = "#9a9a9a"; INK = "#1a1a1a"; GOLD = "#b58a00"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "svg.fonttype": "none", "axes.titleweight": "bold"})
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.svg", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=150); plt.close(fig)
# (2) signature contrast on real data
R = json.load(open(os.path.join(ROOT, "data", "processed", "p48_signature_contrast_real.json")))
ds = ["VED", "HCRL", "Sonata"]; lab = ["VED fleet\n136 cars, trips", "Kia Soul\n10 drivers, windows", "Hyundai Sonata\n4 drivers, windows"]
x = np.arange(3); w = 0.36; fig, ax = plt.subplots(figsize=(6.0, 3.6))
for off, key, col, name in ((-w / 2, "kin", RED, "Kinematic signature"), (w / 2, "trim", BLUE, "Fuel-trim (fueling-fault) axis")):
    r = [R[d][key]["r"] for d in ds]; lo = [R[d][key]["r"] - R[d][key]["ci95"][0] for d in ds]; hi = [R[d][key]["ci95"][1] - R[d][key]["r"] for d in ds]
    ax.bar(x + off, r, w, color=col, label=name, yerr=[lo, hi], capsize=3, ecolor=INK)
    for xi, d in zip(x + off, ds): rr = R[d][key]; ax.text(xi, rr["ci95"][1] + 0.02 if rr["r"] >= 0 else rr["ci95"][0] - 0.06, f"{rr['r']:+.2f}", ha="center", fontsize=8.5, fontweight="bold")
ax.axhspan(-0.1, 0.1, color=GRAY, alpha=0.15, lw=0); ax.text(2.55, 0.0, "±0.1\nequivalence\nmargin", fontsize=7, color="#555", va="center")
ax.axhline(0, color="#888", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylim(-0.2, 0.65); ax.set_xlim(-0.6, 2.95)
ax.set_ylabel("correlation with fuel excess"); ax.set_title("Signature contrast on real vehicles"); ax.legend(fontsize=8.5, frameon=False, loc="upper left")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
save(fig, "signature_contrast")
# (3) bench conditions
C = pd.read_csv(os.path.join(ROOT, "data", "processed", "p47_bench_conditions.csv"))
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for k, col, mk, name in ((0, GRAY, "o", "labelled normal"), (1, RED, "s", "labelled rich"), (2, BLUE, "^", "labelled lean")):
    c = C[C.F == k]; ax.scatter(c.fpa_dev, c.co, s=36, color=col, marker=mk, label=f"{name} ({len(c)} conditions)", edgecolor="white", lw=0.5, zorder=3)
ax.axvspan(12.0, 30, color=RED, alpha=0.06, lw=0); ax.text(13, 0.4, "mixture actually rich", fontsize=8, color=RED)
ax.set_xlabel("fuel per air vs normal mean (percentage points)"); ax.set_ylabel("carbon monoxide (%)")
ax.set_title("Bench conditions: emissions follow the mixture"); ax.legend(fontsize=8, frameon=False, loc="upper left")
for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
save(fig, "fig5_fault_signature")
print("figures written")
