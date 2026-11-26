import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

PAIRS = {
    "DeepSeek-1.5B": ("results/depth2/deepseek/depth.npz", "results/update/depth_k32/depth.npz"),
    "Qwen2.5-3B": ("results/depth2/qwen25_3b/depth.npz", "results/update/depth_qwen25_k32/depth.npz"),
    "Phi-4-mini": ("results/depth2/phi4mini/depth.npz", "results/update/depth_phi_k32/depth.npz"),
}
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
for ax, (n, (p8, p32)) in zip(axes, PAIRS.items()):
    for lab, p, c, ls in [("k=8", p8, "#1f77b4", "-"), ("k=32", p32, "#d62728", "--")]:
        z = np.load(os.path.join(HERE, p), allow_pickle=True)
        lam = z["lam"].astype(float); L = lam.shape[1]
        cum = lam * (np.arange(L) + 1)[None, :, None]
        per_tok = np.median(cum, axis=2)
        med = np.median(per_tok, axis=0)
        lo, hi = np.percentile(per_tok, [25, 75], axis=0)
        x = np.arange(L) / (L - 1)
        ax.plot(x, med, ls, color=c, lw=1.4, label=lab)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xlabel("normalized depth")
    ax.set_ylabel(r"$\mathrm{cum}_\ell$")
    ax.set_title(n, loc="left")
    ax.legend(fontsize=7.4); ax.grid(alpha=0.3)
fig.suptitle("Direction count: k=8 vs k=32 (median and IQR over tokens)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "pack_depth_k32_all.png")
print("done")
