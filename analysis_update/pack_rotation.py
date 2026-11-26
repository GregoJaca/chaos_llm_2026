import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

NN = {
    "DeepSeek-1.5B": "results/nonnorm_w100_big/nonnorm.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/nonnorm_w100/nonnorm.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/nonnorm_w0/nonnorm.npz",
    "Phi-4": "results_big_small/results_big/phi4/nonnorm_w0/nonnorm.npz",
}
Z = {n: load(p) for n, p in NN.items()}

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))
ax = axes[0, 0]
for n, z in Z.items():
    a = z["angles"].astype(float)
    a = a[np.isfinite(a)]
    x, F = ecdf(a)
    ax.plot(x, F, color=COL[n], lw=1.6, label=f"{n} (median {np.median(a):.1f}$^\\circ$)")
ax.axvline(90, color="k", ls="--", lw=1.0, label="random directions (90$^\\circ$)")
ax.set_xlim(50, 95)
ax.set_xlabel("principal angle between top singular vectors of consecutive blocks (deg)")
ax.set_ylabel("cumulative fraction")
ax.set_title("(a) Amplification directions rotate by 78-87$^\\circ$", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)

ax = axes[0, 1]
for n, z in Z.items():
    s = z["full_smax"].astype(float)
    r = z["full_rho"].astype(float)
    k = z["full_kappa"].astype(float)
    ok = np.isfinite(s) & np.isfinite(r)
    ax.scatter(r[ok], s[ok], s=14, alpha=0.6, color=COL[n], label=n)
    ax.scatter(r[ok], k[ok], s=8, alpha=0.2, color=COL[n], marker="x")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"spectral radius $\rho(J_\ell)$")
ax.set_ylabel(r"$\sigma_{\max}(J_\ell)$  (dot) and $\kappa=\sigma_{\max}/\rho$  (x)")
ax.set_title(r"(b) Strong non-normality: $\sigma_{\max}\gg\rho$", loc="left")
ax.legend(fontsize=6.6); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 0]
for n, z in Z.items():
    s = np.nanmedian(z["full_smax"].astype(float), axis=0)
    r = np.nanmedian(z["full_rho"].astype(float), axis=0)
    sel = np.where(np.isfinite(s))[0]
    ax.plot(sel, s[sel], "o-", color=COL[n], label=f"{n} $\\sigma_{{\\max}}$")
    ax.plot(sel, r[sel], "s--", color=COL[n], alpha=0.5)
ax.set_yscale("log")
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel("median over tokens")
ax.set_title("(c) Per-block amplification potential vs depth", loc="left")
ax.legend(fontsize=6.2); ax.grid(True, which="both", alpha=0.3)

# product comparison for the three models with all-layer block spectra
ax = axes[1, 1]
BS = {"DeepSeek-1.5B": "results/block_sigma/deepseek/block_sigma.npz",
      "Qwen3-4B": "results/block_sigma/qwen3_4b/block_sigma.npz",
      "Phi-4-mini": "results/block_sigma/phi4mini/block_sigma.npz"}
DEP = {"DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
       "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
       "Phi-4-mini": "results/depth2/phi4mini/depth.npz"}
labels, naive, measured = [], [], []
for n in BS:
    z = load(BS[n]); smax = z["smax"].astype(float)
    nv = np.exp(np.sum(np.log(np.median(smax, axis=0))))
    d = load(DEP[n]); lam = d["lam"].astype(float)
    L = lam.shape[1]
    cum = lam * (np.arange(L) + 1)[None, :, None]
    mv = float(np.exp(np.median(cum[:, -1, :])))
    labels.append(n); naive.append(nv); measured.append(mv)
x = np.arange(len(labels))
ax.bar(x - 0.18, naive, width=0.36, color="#d62728", label=r"$\prod_\ell\sigma_{\max}(J_\ell)$")
ax.bar(x + 0.18, measured, width=0.36, color="#2ca02c", label="measured product gain")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("gain (log scale)")
ax.set_title("(d) Composing the per-block maxima overpredicts by 16-23 dex", loc="left")
ax.legend(fontsize=6.8); ax.grid(alpha=0.3, axis="y", which="both")
for i, (a, b) in enumerate(zip(naive, measured)):
    ax.text(i, max(a, b) * 2, f"{np.log10(a/b):.0f} dex", ha="center", fontsize=7)

fig.suptitle("Why the composition is stable although every block can amplify", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_rotation_composition.png")
print("done")
