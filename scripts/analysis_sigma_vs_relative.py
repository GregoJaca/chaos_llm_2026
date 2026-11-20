"""Does the model make a perturbation more relevant? Maximum-possible stretch vs actual relative size."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

MODELS = [("DeepSeek-1.5B", "deepseek", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1, "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1, "tab:orange"),
          ("Phi-4-mini", "phi4mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3, "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1, "tab:purple")]

fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.8))
for name, tag, path, r0, c in MODELS:
    f = os.path.join(ROOT, "results", "depth2", tag, "depth.npz")
    if os.path.exists(f):
        z = np.load(f)
        L = z["layers"]
        s1 = np.exp(np.median(z["lam"][:, :, 0], axis=0) * (L + 1))
        axs[0].plot(L / L[-1], s1 / s1[1], "o-", ms=3, lw=1.2, color=c, label=name)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        A = d["dh_norm"].astype(float); Lm = np.asarray(d["layers"])
        h = d["norm_h0"].astype(float)
        ri = int(np.argmin(np.abs(d["radii"] - r0)))
        rel = np.median(A[ri, :, :, 50], axis=0) / np.maximum(h[np.arange(len(Lm)), 50], 1e-9)
        ok = rel > 0
        axs[1].plot(Lm[ok] / Lm[ok][-1], rel[ok] / rel[ok][0], "o-", ms=3, lw=1.2, color=c,
                    label=name)
axs[0].set_title("(a) Maximum stretch of the same-position tangent $\\sigma_1(\\ell)$\n"
                 "normalised at layer 1; computed from block Jacobians")
axs[0].set_xlabel("relative depth $\\ell/(L-1)$")
axs[0].set_ylabel("$\\sigma_1(\\ell)/\\sigma_1(1)$")
axs[0].set_yscale("log"); axs[0].legend(); axs[0].grid(alpha=0.25, which="both")

axs[1].set_title("(b) Actual relative size of the perturbation\n"
                 "$\\|\\delta h_{t,\\ell}\\|/\\|h_{t,\\ell}\\|$ at $t=50$, normalised at layer 1")
axs[1].set_xlabel("relative depth $\\ell/(L-1)$")
axs[1].set_ylabel("$[\\|\\delta h_\\ell\\|/\\|h_\\ell\\|] / [\\|\\delta h_1\\|/\\|h_1\\|]$")
axs[1].set_yscale("log"); axs[1].legend(); axs[1].grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sigma_vs_relative.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved")
