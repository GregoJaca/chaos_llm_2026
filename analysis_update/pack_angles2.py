import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

ORDER = all_names() + ["Mamba-130m", "Pythia-410m", "randQwen"]
A = {n: angle(n) for n in ORDER}
A = {n: a for n, a in A.items() if a is not None}

fig, ax = plt.subplots(figsize=(12.0, 5.0))
labels, data = [], []
for n in ORDER:
    if n not in A:
        continue
    ang = A[n]["angle"]
    v = (ang[0, :, -1, :] if ang.ndim == 4 else ang).ravel()
    v = v[np.isfinite(v)]
    labels.append(n); data.append(v)
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL.get(labels[i], "gray")); b.set_alpha(0.55)
for i, n in enumerate(labels):
    med = np.median(data[i]); lo, hi = np.percentile(data[i], [25, 75])
    ax.plot([i + 1, i + 1], [lo, hi], color="k", lw=1.2)
    ax.plot(i + 1, med, "o", color="white", mec=COL.get(n, "k"), mew=2, ms=6)
ax.axhline(90, color="k", ls="--", lw=1.0, label="random directions (90$^\\circ$)")
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.6)
ax.set_ylabel(r"angle $\angle(\delta h_\ell,h_\ell)$  (deg)")
ax.set_title("Response-vs-state angle, all LLMs and controls (white dot = median, bar = IQR)", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_angles_violin_controls.png")

# angle vs depth, LLMs + controls
names = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Phi-4-mini",
         "Mamba-130m", "Pythia-410m", "randQwen"]
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for ax, n in zip(axes.flat, names):
    if n not in A:
        ax.axis("off"); continue
    ang = A[n]["angle"][0]          # (K, L, T)
    med = np.median(ang, axis=(0, 2))
    lo, hi = np.percentile(ang, [25, 75], axis=(0, 2))
    x = np.arange(len(med))
    ax.plot(x, med, "o-", ms=3, color=COL[n])
    ax.fill_between(x, lo, hi, color=COL[n], alpha=0.2, lw=0)
    ax.axhline(90, color="k", ls="--", lw=0.8)
    ax.set_ylim(60, 115)
    ax.set_xlabel("saved layer index $\\ell$", fontsize=7.6)
    ax.set_ylabel(r"angle (deg)", fontsize=7.6)
    ax.set_title(n, fontsize=8.6)
    ax.grid(alpha=0.3)
fig.suptitle("Response angle vs depth, LLMs and controls (median, band = IQR)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_angle_vs_depth_controls.png")

out = {n: dict(median=float(np.median((A[n]["angle"][0, :, -1, :] if A[n]["angle"].ndim == 4 else A[n]["angle"]).ravel())),
               n=int((A[n]["angle"][0, :, -1, :] if A[n]["angle"].ndim == 4 else A[n]["angle"]).size))
       for n in A}
json.dump(out, open(os.path.join(HERE, "analysis_update", "pack_angles_controls.json"), "w"), indent=1)
print(out)
