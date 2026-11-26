import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

Z = {}
for tag, p in [("k=8", "results/depth2/deepseek/depth.npz"),
               ("k=32", "results/update/depth_k32/depth.npz")]:
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    lam = z["lam"].astype(float); L = lam.shape[1]
    cum = lam * (np.arange(L) + 1)[None, :, None]
    Z[tag] = dict(local=np.median(z["local_gain"].astype(float), axis=0),
                  cum=np.median(cum, axis=(0, 2)),
                  lo=np.percentile(np.median(cum, axis=2), 25, axis=0),
                  hi=np.percentile(np.median(cum, axis=2), 75, axis=0))

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
ax = axes[0]
for tag, z in Z.items():
    ax.plot(np.arange(len(z["local"])), z["local"], "o-", ms=3, label=tag)
ax.set_yscale("log")
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"$g_\ell$  (median over tokens and directions)")
ax.set_title("(a) Local gain: 8 vs 32 tracked directions", loc="left")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for tag, z in Z.items():
    x = np.arange(len(z["cum"]))
    ax.plot(x, z["cum"], "-", label=tag)
    ax.fill_between(x, z["lo"], z["hi"], alpha=0.15, lw=0)
ax.set_xlabel("block index $\\ell$")
ax.set_ylabel(r"$\mathrm{cum}_\ell$")
ax.set_title("(b) Product gain: the curve is unchanged", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_depth_k32.png")
print("done")
