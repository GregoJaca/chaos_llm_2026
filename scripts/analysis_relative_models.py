"""Relative response vs depth with 25-75% bands over directions, all models."""
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

MODELS = [("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1, "tab:blue", 20),
          ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1, "tab:orange", 20),
          ("Qwen2.5-3B", os.path.join(LD, "models", "qwen25_3b", "a1.npz"), 0.1, "tab:red", 20),
          ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3, "tab:green", 20),
          ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1, "tab:purple", 10)]

fig, ax = plt.subplots(figsize=(9, 4.8))
for name, path, r0, c, nd in MODELS:
    if not os.path.exists(path):
        continue
    d = np.load(path, allow_pickle=True)
    A = d["dh_norm"].astype(float); Lm = np.asarray(d["layers"])
    h = d["norm_h0"].astype(float)
    ri = int(np.argmin(np.abs(d["radii"] - r0)))
    rel = A[ri, :, :, 50] / np.maximum(h[np.arange(len(Lm)), 50][None, :], 1e-9)   # [dir, layer]
    med = np.median(rel, axis=0); lo = np.percentile(rel, 25, axis=0); hi = np.percentile(rel, 75, axis=0)
    ok = med > 0
    ax.semilogy(Lm[ok], med[ok], "o-", ms=3, lw=1.2, color=c, label=f"{name} ($r={r0}$, {nd} dirs)")
    ax.fill_between(Lm[ok], np.maximum(lo[ok], 1e-12), np.maximum(hi[ok], 1e-12), color=c, alpha=0.15)
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|h_{t,\ell}\|$ at $t=50$")
ax.set_title("Relative size of the perturbation vs depth\nline = median over directions, band = 25–75%")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "relative_response_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved")
