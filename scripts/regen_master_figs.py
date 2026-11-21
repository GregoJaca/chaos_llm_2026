"""Regenerate the master.md figures with percentile bands, from the local data."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

MODELS = [("DeepSeek-1.5B", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.1, "tab:blue"),
          ("Qwen3-4B", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.1, "tab:orange"),
          ("Phi-4-mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3, "tab:green"),
          ("Qwen3.5-2B", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.1, "tab:purple")]


def load(path, r, li=-1):
    d = np.load(path, allow_pickle=True)
    A = d["dh_norm"].astype(float)
    ri = int(np.argmin(np.abs(d["radii"] - r)))
    return d, A[ri], np.asarray(d["layers"])


# ---------------------------------------------------------------- 1. flux: overlay + three layers with bands
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
for name, path, r, c in MODELS:
    d, A, L = load(path, r)
    a = np.maximum(np.median(A, axis=0), 1e-6)
    cmap = plt.get_cmap("viridis")
    for i in range(1, len(L)):
        ax.semilogy(np.arange(a.shape[1]), a[i] / a[i, 1], lw=0.7,
                    color=cmap(L[i] / L[-1]), alpha=0.55)
    ax.text(0.02, 0.95 - 0.09 * MODELS.index((name, path, r, c)) if False else 0.95,
            "", transform=ax.transAxes)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$")
ax.set_title("(a) All layers, four models (colour = relative depth)")
ax.grid(alpha=0.22, which="both")

ax = axs[1]
for name, path, r, c in MODELS:
    d, A, L = load(path, r)
    for frac, ls in [(0.15, ":"), (0.85, "-")]:
        i = int(round(frac * (len(L) - 1)))
        curve = A[:, i, :]
        med = np.median(curve, axis=0) / np.median(curve[:, 1]) if False else np.median(curve, axis=0)
        med = np.maximum(med, 1e-6) / max(np.median(curve[:, 1]), 1e-12)
        lo = np.maximum(np.percentile(curve, 10, axis=0), 1e-6) / max(np.median(curve[:, 1]), 1e-12)
        hi = np.maximum(np.percentile(curve, 90, axis=0), 1e-6) / max(np.median(curve[:, 1]), 1e-12)
        ax.semilogy(np.arange(len(med)), med, ls, lw=1.3, color=c,
                    label=f"{name} L{int(L[i])}")
        ax.fill_between(np.arange(len(med)), lo, hi, color=c, alpha=0.12)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$")
ax.set_title("(b) One shallow (: ) and one deep (−) layer per model\nband = 10–90% over directions")
ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.22, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flux_models_all_layers.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. lambda with bands
fig, ax = plt.subplots(figsize=(9, 4.8))
for name, path, r, c in MODELS:
    d, A, L = load(path, 0.3)
    k = 10
    lam = []
    for di in range(A.shape[0]):
        x = np.maximum(A[di, -1, :], 1e-12)
        t = np.arange(k, len(x) - k)
        lam.append(np.log(x[t + k] / x[t - k]) / (2 * k))
    lam = np.array(lam)
    t = np.arange(k, A.shape[-1] - k)
    ax.plot(t, np.median(lam, axis=0), lw=1.3, color=c, label=name)
    ax.fill_between(t, np.percentile(lam, 25, axis=0), np.percentile(lam, 75, axis=0),
                    color=c, alpha=0.15)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\lambda_{10}(t)$ (nats/token)")
ax.set_title("Finite-time rate at the last layer, $r=0.3$\nline = median, band = 25–75% over directions")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "model_lambda_compare.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. block sigma with band
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for name, tag, c in [("DeepSeek-1.5B", "deepseek", "tab:blue"), ("Qwen3-4B", "qwen3_4b", "tab:orange"),
                     ("Phi-4-mini", "phi4mini", "tab:green")]:
    f = os.path.join(ROOT, "results", "block_sigma", tag, "block_sigma.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); sm = z["smax"]; L = z["layers"]
    axs[0].semilogy(L, np.median(sm, axis=0), "o-", ms=3, lw=1.2, color=c, label=name)
    axs[0].fill_between(L, sm.min(axis=0), sm.max(axis=0), color=c, alpha=0.15)
    axs[1].semilogy(L, np.median(sm, axis=0) ** 1, "o-", ms=3, lw=1.0, color=c)
axs[0].axhline(1, color="k", ls=":", lw=0.8)
axs[0].set_xlabel(r"block $\ell$"); axs[0].set_ylabel(r"$\sigma_{\max}$ of one block")
axs[0].set_title("(a) Full-dimension top singular value\nline = median, band = min–max over tokens")
axs[0].legend(); axs[0].grid(alpha=0.25, which="both")
axs[1].set_xlabel(r"block $\ell$"); axs[1].set_ylabel(r"$\sigma_{\max}$ (log)")
axs[1].set_title("(b) Same, log scale")
axs[1].grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fits", "block_sigma_full.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4. window curves with bands
fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
for ax, model, name, layers in [(axs[0], "qwen3_4b", "Qwen3-4B", [1]),
                                (axs[1], "phi4mini", "Phi-4-mini", [1])]:
    for w, c in [(60, "tab:blue"), (100, "tab:orange"), (200, "tab:green"), (4096, "0.35")]:
        f = os.path.join(LD, "window4d", model, f"w{w}", "a1.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); A = z["dh_norm"][0]
        L = list(z["layers"]); li = L.index(layers[0])
        med = np.median(A[:, li, :], axis=0)
        ax.semilogy(np.arange(A.shape[-1]), np.maximum(med, 1e-10), lw=1.2, color=c, label=f"$W={w}$")
        ax.fill_between(np.arange(A.shape[-1]), np.maximum(np.percentile(A[:, li, :], 10, axis=0), 1e-10),
                        np.maximum(np.percentile(A[:, li, :], 90, axis=0), 1e-10), color=c, alpha=0.12)
    ax.set_xlabel("generated token $t$")
    ax.set_title(f"{name}, layer 1, $r=0.1$, 6 directions\nband = 10–90%")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
axs[0].set_ylabel(r"$\|\delta h_{t,\ell=1}\|/r$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window4d_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 5. peak lag with IQR
B = os.path.join(LD, "probe2")
fig, ax = plt.subplots(figsize=(9, 4.6))
for name, tag, c in [("DeepSeek-1.5B", "deepseek", "tab:blue"), ("Qwen3-4B", "qwen3_4b", "tab:orange"),
                     ("Qwen2.5-3B", "qwen25_3b", "tab:red"), ("Phi-4-mini", "phi4mini", "tab:green"),
                     ("Qwen3.5-2B", "qwen3_5", "tab:purple")]:
    d0 = os.path.join(B, tag)
    if not os.path.isdir(d0):
        continue
    S, med, lo, hi = [], [], [], []
    for sub in sorted(os.listdir(d0)):
        f = os.path.join(d0, sub, "layer_probe.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); p = int(z["pos"]); r = float(z["r"])
        curves = z["dh"][0][:, -1, p + 1:p + 51].T / r    # [tau, dir]
        k = np.argmax(curves, axis=0) + 1
        S.append(p); med.append(np.median(k)); lo.append(np.percentile(k, 25)); hi.append(np.percentile(k, 75))
    o = np.argsort(S)
    S = np.array(S)[o]; med = np.array(med)[o]; lo = np.array(lo)[o]; hi = np.array(hi)[o]
    ax.plot(S, med, "o-", ms=4, lw=1.2, color=c, label=name)
    ax.fill_between(S, lo, hi, color=c, alpha=0.15)
ax.set_xlabel("position $s$ of the perturbation")
ax.set_ylabel(r"lag of the peak $\tau^*$ (tokens)")
ax.set_title("Lag at which each perturbation is amplified\nline = median over directions, band = 25–75%")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "positions", "peak_is_absolute.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("regenerated 5 figures with bands")
