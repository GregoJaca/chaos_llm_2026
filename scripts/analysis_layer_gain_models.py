"""Layer-resolved gain for every model, from the all-layer prompt-perturbation runs."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})

MODELS = [("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1, "tab:blue"),
          ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1, "tab:orange"),
          ("Qwen2.5-3B", os.path.join(LD, "models", "qwen25_3b", "a1.npz"), 0.1, "tab:red"),
          ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3, "tab:green"),
          ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1, "tab:purple")]

fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
res = {}
for name, path, r, c in MODELS:
    if not os.path.exists(path):
        print("missing", name)
        continue
    d = np.load(path, allow_pickle=True)
    A = d["dh_norm"].astype(float); L = np.asarray(d["layers"])
    h = d["norm_h0"].astype(float)
    ri = int(np.argmin(np.abs(d["radii"] - r)))
    t = 50
    a = np.median(A[ri, :, :, t], axis=0) / r
    rel = L / L[-1]
    g = np.where(a[:-1] > 0, a[1:] / np.maximum(a[:-1], 1e-30), np.nan)
    axs[0].semilogy(rel[1:], g, "o-", ms=3, lw=1.1, color=c, label=name)
    gl = np.where(np.isfinite(g) & (g > 0), g, 1.0)
    axs[1].plot(rel[1:], np.cumsum(np.log(gl)), "o-", ms=3, lw=1.1,
                color=c, label=name)
    q = a / np.maximum(h[np.arange(len(L)), t], 1e-9)
    axs[2].semilogy(rel[1:], q[1:], "o-", ms=3, lw=1.1, color=c, label=name)
    res[name] = dict(median_gain=float(np.nanmedian(g)), total_log_gain=float(np.nansum(np.log(np.maximum(g, 1e-9)))),
                     rel_start=float(q[1]), rel_end=float(q[-1]))
axs[0].axhline(1, color="k", ls=":", lw=0.8)
axs[0].set_xlabel(r"relative depth $\ell/(L-1)$"); axs[0].set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{t,\ell-1}\|$")
axs[0].set_title(f"(a) Layer-to-layer gain, $t={50}$"); axs[0].legend(); axs[0].grid(alpha=0.25, which="both")
axs[1].set_xlabel(r"relative depth"); axs[1].set_ylabel(r"cumulative $\sum\log g$ (nats)")
axs[1].set_title("(b) Cumulative gain (the depth profile)"); axs[1].legend(); axs[1].grid(alpha=0.25, which="both")
axs[2].set_xlabel(r"relative depth"); axs[2].set_ylabel(r"$\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|$")
axs[2].set_title("(c) Relative response"); axs[2].legend(); axs[2].grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_gain_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

json.dump(res, open(os.path.join(ROOT, "results", "layer_gain_models.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
