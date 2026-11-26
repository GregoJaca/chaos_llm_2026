import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

DEP = {
    "DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
    "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
    "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
    "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
    "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
    "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz",
}
fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
ax = axes[0]
ratios = {}
for n, p in DEP.items():
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    g = z["local_gain"].astype(float)          # (token, layer)
    med = np.median(g, axis=0)
    mid = np.median(med[max(1, len(med)//4):3*len(med)//4])
    x = np.arange(len(med)) / (len(med) - 1)
    ax.plot(x[1:], med[1:] / mid, color=COL[n], label=n)
    ratios[n] = float(np.median(med[-max(1, len(med)//10):]) / mid)
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("normalized depth  $\\ell/(L-1)$  (block 0 excluded)")
ax.set_ylabel(r"$g_\ell$ / median(g)  (middle half)")
ax.set_title("(a) The late-depth rise, all models", loc="left")
ax.legend(fontsize=6.2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
names = list(ratios)
vals = [ratios[n] for n in names]
y = np.arange(len(names))
ax.barh(y, vals, color=[COL[n] for n in names], alpha=0.85)
ax.axvline(1, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.4)
ax.set_xlabel(r"$g_{\rm last10\%}\,/\,g_{\rm middle}$")
ax.set_title("(b) Late rise factor", loc="left")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_depth_late_rise.png")
json.dump(ratios, open(os.path.join(HERE, "analysis_update", "pack_depth_late_rise.json"), "w"), indent=1)
print(ratios)
