"""Response vs time for every saved layer, one figure per model (with colourbar).

Colour encodes the layer index, so the depth ordering is readable directly.
Layer 0 is the input embedding and is dropped (its response is identically 0).
"""
import os
import sys
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)


def get(n, radius=0.1):
    if n in SMALL:
        z = a1(n)
        ri = int(np.argmin(np.abs(z["radii"] - radius)))
        return z["med"][ri], z["layers"].astype(float)
    if n in BIG:
        z = big(n)
        return z["dh_med"], z["layers"].astype(float)
    z = load(CTRL[n])
    ri = int(np.argmin(np.abs(z["radii"] - radius)))
    return z["dh_med"][ri], z["layers"].astype(float)


def panel(ax, n, fig=None):
    med, lay = get(n)
    keep = lay > 0
    med, lay = med[keep], lay[keep]
    cmap = plt.cm.viridis(np.linspace(0, 1, med.shape[0]))
    for i in range(med.shape[0]):
        a = med[i]
        a = a / max(a[1], 1e-30)
        ax.plot(np.arange(len(a)), np.maximum(a, 1e-6), color=cmap[i], lw=1.0)
    ax.set_yscale("log"); ax.set_xlim(0, med.shape[-1] - 1)
    ax.set_xlabel("generated token $t$")
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$")
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(1, color="k", ls=":", lw=0.7)
    if fig is not None:
        sm = ScalarMappable(norm=Normalize(lay.min(), lay.max()), cmap="viridis")
        cb = fig.colorbar(sm, ax=ax, pad=0.01)
        cb.set_label("layer $\\ell$")
    return med.shape[-1]


# combined three-panel view (kept because it was liked)
fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8))
for ax, n in zip(axes, ["DeepSeek-1.5B", "Qwen3-4B", "randQwen"]):
    Tn = panel(ax, n, fig)
    ax.set_title(f"{n}  ({Tn} tokens)", loc="left")
fig.suptitle("Every saved layer: response vs time (colour = layer)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_time_all_layers.png")

# one figure per model
for n in ORDER:
    try:
        med, lay = get(n)
    except Exception as exc:
        print("skip", n, exc)
        continue
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    panel(ax, n, fig)
    tag = "" if n not in CTRL else " (control)"
    ax.set_title(f"{n}{tag}: response vs time, every saved layer", loc="left")
    fig.tight_layout()
    save(fig, "pack_time_layers_" + n.replace("-", "_").replace(".", "") + ".png")
    print("wrote", n, lay[:5], "...", lay[-1])
