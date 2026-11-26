import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
for ax, n in zip(axes, ["DeepSeek-1.5B", "Qwen3-4B", "Mamba-130m"]):
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1))); med = z["med"][ri]; lay = z["layers"]
    elif n == "Mamba-130m":
        z = load("results_big_small/ctrl_Mamba_130m.npz"); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        med = z["dh_med"][ri]; lay = z["layers"]
    else:
        continue
    cmap = plt.cm.viridis(np.linspace(0, 1, med.shape[0]))
    for i in range(med.shape[0]):
        a = med[i]; a = a / max(a[1], 1e-30)
        ax.plot(np.arange(len(a)), a, color=cmap[i], lw=1.0)
    ax.set_yscale("log"); ax.set_xlim(0, 300)
    ax.set_xlabel("generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$")
    ax.set_title(f"{n} ({med.shape[0]} layers)", loc="left")
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Every saved layer: response vs time (layer 0 included but is the input)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_time_all_layers.png")
print("done")
