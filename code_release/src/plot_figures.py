"""Plot the predicted-versus-assigned scatter plot and the ablation chart
from the output of ridge_evaluation.py.

Usage: python plot_figures.py RESULTS_JSON OUT_DIR
"""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUTDIR = sys.argv[1], sys.argv[2]
r = json.load(open(RES))
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

# Predicted vs actual, test set, split seed 0
p = r["preds"]["0"]; y, yh = np.array(p["y"]), np.array(p["pred"])
fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=600)
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
ax.scatter(y, yh, s=14, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.4, zorder=2)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.08, 1.08)
ax.set_xlabel("Harm score (label)"); ax.set_ylabel("Predicted harm score")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
s0 = r["0"]["ablation"]["full"]
ax.text(0.03, 0.97, f"Test set (n = {len(y)})\nMAE = {s0['MAE']:.4f}\n$R^2$ = {s0['R2']:.3f}",
        transform=ax.transAxes, va="top", color=INK, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/Fig_pred_vs_actual.png"); fig.savefig(f"{OUTDIR}/Fig_pred_vs_actual.tif")

# Ablation: mean R2 +/- SD over three splits
order = ["full", "no_image", "category+keyword", "image_only", "category_only", "title_tfidf_only", "price_only"]
names = {"full": "All features", "no_image": "All except image", "category+keyword": "Category + keyword",
         "image_only": "Image only", "category_only": "Category only", "title_tfidf_only": "Title text only",
         "price_only": "Price only"}
S = r["summary"]; m = np.array([S[k]["R2_mean"] for k in order]); sd = np.array([S[k]["R2_sd"] for k in order])
fig, ax = plt.subplots(figsize=(5.0, 2.9), dpi=600)
yy = np.arange(len(order))[::-1]
ax.barh(yy, np.clip(m, 0, None), height=0.55, color=BLUE, edgecolor="white", linewidth=1)
ax.errorbar(np.clip(m, 0, None), yy, xerr=sd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
for yv, mv, sv in zip(yy, m, sd):
    ax.text(max(mv, 0) + sv + 0.025, yv, f"{mv:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([names[k] for k in order])
ax.set_xlim(0, 1.12); ax.set_xlabel("Test $R^2$ (mean ± SD over three splits)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/Fig_ablation.png"); fig.savefig(f"{OUTDIR}/Fig_ablation.tif")
print("saved")
