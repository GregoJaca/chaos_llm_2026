import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
for ax, W in zip(axes, [100, 4096]):
    p = os.path.join(HERE, f"results/update/attn_w{W}/attn_frac.npz")
    z = np.load(p, allow_pickle=True)
    frac = z["frac"]; lay = z["layers"]; P = int(z["prompt_len"])
    for i, l in enumerate(lay):
        ax.plot(np.arange(frac.shape[1]), frac[i], lw=1.3, label=f"layer {l}")
    if W - P < 200:
        ax.axvline(max(W - P, 0), color="k", ls="--", lw=1.0, label=r"$W-P$")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("generated token $t$")
    ax.set_ylabel("attention mass on prompt tokens")
    ax.set_title(f"W={W} (P={P})", loc="left")
    ax.legend(fontsize=6.6); ax.grid(alpha=0.3)
fig.suptitle("Where the attention goes: prompt vs generated tokens (measured, not schematic)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_attention_fraction.png")
print("done")
