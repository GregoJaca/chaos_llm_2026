import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)

def curve(n):
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        return z["med"][ri], z["layers"]
    if n in BIG:
        z = big(n); return z["dh_med"], z["layers"]
    z = load(CTRL[n]); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    return z["dh_med"][ri], z["layers"]

fig, ax = plt.subplots(figsize=(11.2, 5.2))
for n in ORDER:
    med, lay = curve(n)
    a = med[-1]; a = a / max(a[1], 1e-30)
    ax.plot(np.arange(len(a)), a, color=COL[n], lw=1.4, label=n)
ax.set_yscale("log"); ax.set_xlim(0, 300)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_{t,L}\|/\|\delta h_{1,L}\|$")
ax.set_title("Prompt response vs token: LLMs and controls", loc="left")
ax.legend(fontsize=6.4, ncol=3); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_time_all_controls.png")

fig, axes = plt.subplots(3, 4, figsize=(16.0, 10.4))
for ax, n in zip(axes.flat, ORDER):
    med, lay = curve(n)
    # drop layer 0, use first/middle/last
    idx = [1, len(lay) // 2, len(lay) - 1]
    for j, c in zip(idx, ["#1f77b4", "#ff7f0e", "#d62728"]):
        a = med[j]; a = a / max(a[1], 1e-30)
        ax.plot(np.arange(len(a)), a, color=c, lw=1.2, label=f"layer {lay[j]}")
    ax.set_yscale("log"); ax.set_xlim(0, 300)
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_title(n, fontsize=8.4)
    ax.set_xlabel("$t$", fontsize=7.4)
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$", fontsize=7.4)
    ax.legend(fontsize=5.8); ax.grid(True, which="both", alpha=0.25)
for j in range(len(ORDER), 12):
    axes.flat[j].axis("off")
fig.suptitle("Response vs time at three layers (layer 0 excluded), LLMs and controls", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_time_layers_controls.png")
print("done")
