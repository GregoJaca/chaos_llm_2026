"""The whole depth spectrum: lambda_i vs i, and how its slope changes with depth."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})

z = np.load(os.path.join(ROOT, "results", "depth2", "deepseek", "depth.npz"))
lam, L = z["lam"], z["layers"]        # [tokens, layers, k]
nt, nL, k = lam.shape
med = np.median(lam, axis=0)          # [layers, k]

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
ax = axs[0]
idx = np.arange(1, k + 1)
for l, c in [(1, "tab:blue"), (5, "tab:orange"), (14, "tab:green"), (27, "tab:red")]:
    ax.loglog(idx, med[l], "o-", ms=4, color=c, label=f"layer {l}")
ax.set_xlabel("exponent index $i$"); ax.set_ylabel(r"$\lambda_i$ at that depth")
ax.set_title("(a) The spectrum is nearly flat in $i$")
ax.legend(); ax.grid(alpha=0.25, which="both")

# fit lambda_i ~ i^-alpha at each layer
alpha = np.full(nL, np.nan); alpha_se = np.full(nL, np.nan); a0 = np.full(nL, np.nan)
for l in range(nL):
    y = med[l]
    if np.all(y > 0):
        r = fit_curve(idx, y, families=["power"], log_mode="native")
        p = dict(zip(r.best.pnames, r.best.params))
        alpha[l] = -p["b"]; alpha_se[l] = r.best.se[1]; a0[l] = p["a"]
ax = axs[1]
ax.errorbar(L, alpha, yerr=alpha_se, fmt="o-", ms=3, color="tab:purple")
ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"$\alpha$ in $\lambda_i\propto i^{-\alpha}$")
ax.set_title(r"(b) Spectral slope is zero at every depth")
ax.grid(alpha=0.25)

ax = axs[2]
for l, c in [(1, "tab:blue"), (5, "tab:orange"), (14, "tab:green"), (27, "tab:red")]:
    ax.plot(idx, med[l] / med[l, 0], "o-", ms=4, color=c, label=f"layer {l}")
ax.set_xlabel("exponent index $i$"); ax.set_ylabel(r"$\lambda_i/\lambda_1$")
ax.set_title("(c) Normalised shape: all depths have the same profile")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_spectrum_shape.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("alpha per layer:", np.round(alpha, 2).tolist())
print("alpha mean", float(np.nanmean(alpha)))
