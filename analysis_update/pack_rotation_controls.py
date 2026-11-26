import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

FILES = {
    "DeepSeek-1.5B": "results/nonnorm_w100_big/nonnorm.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/nonnorm_w100/nonnorm.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/nonnorm_w0/nonnorm.npz",
    "Phi-4": "results_big_small/results_big/phi4/nonnorm_w0/nonnorm.npz",
    "randQwen": "results/update/nonnorm_randqwen3/nonnorm.npz",
}
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
ax = axes[0]
labels, data = [], []
for n, p in FILES.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    a = z["angles"].astype(float); a = a[np.isfinite(a)]
    labels.append(n); data.append(a)
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[n]); b.set_alpha(0.55)
for i, n in enumerate(labels):
    med = np.median(data[i]); lo, hi = np.percentile(data[i], [25, 75])
    ax.plot([i + 1, i + 1], [lo, hi], color="k", lw=1.2)
    ax.plot(i + 1, med, "o", color="white", mec=COL[n], mew=2, ms=6)
ax.axhline(90, color="k", ls="--", lw=1.0, label="random (90$^\\circ$)")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.6)
ax.set_ylabel("principal angle between consecutive top singular vectors (deg)")
ax.set_title("(a) Rotation angles: LLMs vs the random control", loc="left")
ax.legend(fontsize=7.2); ax.grid(alpha=0.3, axis="y")
ax = axes[1]
for n, a in zip(labels, data):
    x, F = ecdf(a)
    ax.plot(x, F, color=COL[n], lw=1.6, label=f"{n} ({np.median(a):.1f}$^\\circ$)")
ax.axvline(90, color="k", ls="--", lw=1.0)
ax.set_xlim(50, 95)
ax.set_xlabel("rotation angle (deg)")
ax.set_ylabel("cumulative fraction")
ax.set_title("(b) Same data, ECDF", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_rotation_controls.png")
json.dump({n: dict(median=float(np.median(a)), p25=float(np.percentile(a, 25)),
                   p75=float(np.percentile(a, 75)), n=int(len(a)))
           for n, a in zip(labels, data)},
          open(os.path.join(HERE, "analysis_update", "pack_rotation_controls.json"), "w"), indent=1)
print("done")
