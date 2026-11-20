"""Full-dimension block sigma_max vs the chained product: does amplification compose?"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "block_sigma")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:purple")]

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
res = {}
for name, tag, c in MODELS:
    f = os.path.join(B, tag, "block_sigma.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    sm = z["smax"]                       # [tokens, layers]
    med = np.median(sm, axis=0)
    L = z["layers"]
    axs[0].semilogy(L, med, "o-", ms=3, lw=1.2, color=c, label=name)
    naive = np.sum(np.log10(med))
    prod = med[0] * np.prod(med[1:])     # upper bound if directions composed
    res[name] = dict(block0=float(med[0]), median_middle=float(np.median(med[1:-3])),
                     last=float(med[-1]), naive_log10=float(naive),
                     composed=float(prod))
axs[0].axhline(1, color="k", ls=":", lw=0.8)
axs[0].set_xlabel(r"block $\ell$")
axs[0].set_ylabel(r"$\sigma_{\max}$ of one block (full dimension)")
axs[0].set_title("(a) Every block amplifies: 3–6× in the middle")
axs[0].legend(); axs[0].grid(alpha=0.25, which="both")

# measured product from the depth map
ax = axs[1]
for name, tag, c in MODELS:
    f = os.path.join(ROOT, "results", "depth2", tag, "depth.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    L = z["layers"]
    s1 = np.exp(np.median(z["lam"][:, :, 0], axis=0) * (L + 1))
    ax.semilogy(L, s1, "o-", ms=3, lw=1.2, color=c, label=name)
ax.set_xlabel(r"block $\ell$"); ax.set_ylabel(r"$\sigma_1$ of the product")
ax.set_title("(b) The product grows only to ~20")
ax.legend(); ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")

ax = axs[2]
names = list(res)
vals = [res[n]["naive_log10"] for n in names]
ax.barh(np.arange(len(names)), vals, color=[dict((m[0], m[2]) for m in MODELS)[n] for n in names],
        alpha=0.85)
ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel(r"$\log_{10}\prod_\ell \sigma_{\max}(J_\ell)$  (if directions composed)")
ax.set_title("(c) Naive composition predicts $10^{18}$–$10^{23}$")
ax.grid(alpha=0.25, axis="x")
for i, n in enumerate(names):
    ax.text(vals[i] + 0.2, i, f"{vals[i]:.1f}", va="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "block_sigma_full.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
json.dump(res, open(os.path.join(ROOT, "results", "block_sigma_full.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
