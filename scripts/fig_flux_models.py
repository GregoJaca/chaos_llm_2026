"""Flux-tube curves a(t)/a(1) for all four models, all saved layers, with a layer colourbar."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11})

RUNS = [
    ("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1, 27),
    ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1, 35),
    ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3, 31),
    ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1, 23),
]

fig, axs = plt.subplots(1, 4, figsize=(21, 4.6), sharey=False)
for ax, (name, path, r, last) in zip(axs, RUNS):
    d = np.load(path, allow_pickle=True)
    A = d["dh_norm"].astype(np.float64)
    L = np.asarray(d["layers"])
    ri = int(np.argmin(np.abs(d["radii"] - r)))
    a = np.maximum(np.median(A[ri], axis=0), 1e-12)     # [layer, T]
    cmap = plt.get_cmap("viridis")
    for li in range(1, len(L)):
        ax.semilogy(np.arange(a.shape[1]), a[li] / a[li, 1], lw=0.9,
                    color=cmap(L[li] / last), alpha=0.9)
    ax.set_ylim(1e-6, 1e3)
    ax.set_xlabel(r"generated token $t$")
    ax.set_title(f"{name}, $r={r}$")
    ax.grid(alpha=0.22, which="both")
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, last))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.045)
    cb.set_label(r"layer $\ell$")
axs[0].set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|\delta h_{1,\ell}\|$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flux_models_all_layers.png"), dpi=190)
print("saved")
