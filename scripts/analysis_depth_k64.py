"""k=8 vs k=64: does the spectrum stay flat with more directions?"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
for tag, c, name in [("deepseek_k64", "tab:blue", "DeepSeek-1.5B")]:
    z = np.load(os.path.join(ROOT, "results", "depth2", tag, "depth.npz"))
    L = z["layers"]
    lam = z["lam"]
    ax = axs[0]
    for l, cc in [(0, "0.2"), (5, "tab:orange"), (14, "tab:green"), (27, "tab:red")]:
        y = np.median(lam[:, l, :], axis=0)
        ax.loglog(np.arange(1, len(y) + 1), y, "o-", ms=3, color=cc, label=f"block {l}")
    ax.set_xlabel("singular-value index $i$")
    ax.set_ylabel(r"$\lambda_i$ at that block")
    ax.set_title(f"(a) {name}: top 64 exponents")
    ax.legend(); ax.grid(alpha=0.25, which="both")

    ax = axs[1]
    for l, cc in [(0, "0.2"), (5, "tab:orange"), (14, "tab:green"), (27, "tab:red")]:
        y = np.median(lam[:, l, :], axis=0)
        ax.plot(np.arange(1, len(y) + 1), y / y[0], "o-", ms=2, color=cc, label=f"block {l}")
    ax.set_xlabel("singular-value index $i$")
    ax.set_ylabel(r"$\lambda_i/\lambda_1$")
    ax.set_title("(b) The shape: flat out to $i=64$?")
    ax.legend(); ax.grid(alpha=0.25)

# ratio of the 64th to the 1st singular value of the product, per depth
z = np.load(os.path.join(ROOT, "results", "depth2", "deepseek_k64", "depth.npz"))
lam64 = np.median(z["lam"], axis=0)          # [layers, 64]
ax = axs[2]
ax.plot(z["layers"], lam64[:, -1] / lam64[:, 0], "o-", ms=3, color="tab:purple",
        label="k=64: $\\lambda_{64}/\\lambda_1$")
z8 = np.load(os.path.join(ROOT, "results", "depth2", "deepseek128", "depth.npz"))
lam8 = np.median(z8["lam"], axis=0)
ax.plot(z8["layers"], lam8[:, -1] / lam8[:, 0], "s--", ms=3, color="tab:gray",
        label="k=8: $\\lambda_8/\\lambda_1$")
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"block $\ell$"); ax.set_ylabel("ratio of exponents")
ax.set_title("(c) Anisotropy inside the tracked subspace")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_k64.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

out = dict(ratio_k64_last=float(lam64[-1, -1] / lam64[-1, 0]),
           ratio_k64_first=float(lam64[0, -1] / lam64[0, 0]))
json.dump(out, open(os.path.join(ROOT, "results", "depth_k64.json"), "w"), indent=1)
print(out)
