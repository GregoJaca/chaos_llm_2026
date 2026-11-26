import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

z = np.load(os.path.join(HERE, "results/local_data/a1_4k_test.npz"), allow_pickle=True)
dh = z["dh_norm"].astype(float)        # (r, dir, layer, T)
lay = z["layers"].astype(int)
a = np.median(dh[0, :, -1, :], axis=0)
a = a / max(a[1], 1e-30)
t = np.arange(len(a))
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
ax = axes[0]
ax.plot(t, np.maximum(a, 1e-12), color=COL["DeepSeek-1.5B"], lw=1.2)
ax.set_yscale("log")
ax.set_xlim(0, 4000)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_{t,L}\|/\|\delta h_{1,L}\|$  (log)")
ax.set_title("(a) 4000 tokens, lin-log", loc="left")
ax.grid(True, which="both", alpha=0.3)
ax = axes[1]
ax.loglog(np.maximum(t[1:], 1), np.maximum(a[1:], 1e-12), color=COL["DeepSeek-1.5B"], lw=1.2)
ax.set_xlabel("$t$  (log)")
ax.set_ylabel(r"$\|\delta h_{t,L}\|/\|\delta h_{1,L}\|$  (log)")
ax.set_title("(b) log-log", loc="left")
ax.grid(True, which="both", alpha=0.3)
ax = axes[2]
k = 20
lam = (np.log(np.maximum(a[k:], 1e-30)) - np.log(np.maximum(a[:-k], 1e-30))) / k
ax.plot(t[k:], lam, lw=1.0, color="#d62728")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(0, 4000); ax.set_ylim(-0.02, 0.02)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\lambda_{20}(t)$  (nats/token)")
ax.set_title("(c) Local rate: no sustained positive level", loc="left")
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_long_horizon2.png")
print(z["radii"], lay, a.min(), a.max())

