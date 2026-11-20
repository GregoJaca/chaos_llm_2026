"""Amplification of a one-position perturbation: correct statistics (per-direction max over lag,
then median and 10-90% across directions). Replaces the earlier mislabelled versions."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "probe2")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Qwen2.5-3B", "qwen25_3b", "tab:red"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:purple")]


def load(tag, pos):
    f = os.path.join(B, tag, f"pos{pos}", "layer_probe.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return z, float(z["r"]), int(z["pos"])


def per_direction_max(dh, p, r, tau=50, inj=0):
    """[n_dir] array: for each direction, max over t=p+1..p+tau of ||dh||/r at the last layer."""
    return dh[inj][:, -1, p + 1:p + 1 + tau].max(axis=1) / r


# ---------------------------------------------------------------- figure 1: vs position
fig, ax = plt.subplots(figsize=(9.2, 4.9))
for name, tag, c in MODELS:
    xs, med, lo, hi = [], [], [], []
    d0 = os.path.join(B, tag)
    for sub in sorted(os.listdir(d0)) if os.path.isdir(d0) else []:
        if not sub.startswith("pos"):
            continue
        z, r, p = load(tag, sub[3:])
        if z is None:
            continue
        v = per_direction_max(z["dh"], p, r)
        xs.append(p); med.append(np.median(v))
        lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
    if not xs:
        continue
    o = np.argsort(xs)
    xs = np.array(xs)[o]; med = np.array(med)[o]; lo = np.array(lo)[o]; hi = np.array(hi)[o]
    ax.semilogy(xs, med, "o-", ms=4, lw=1.3, color=c, label=f"{name}")
    ax.fill_between(xs, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3), color=c, alpha=0.12)
ax.set_xlabel("position $s$ of the perturbation (generated tokens)")
ax.set_ylabel(r"$\max_{1\leq\tau\leq50}\|\delta h_{s+\tau,\,L}\|/r$   (layer $L$ = last)")
ax.set_title("Amplification of a one-position perturbation vs position\n"
             "line = median over directions, band = 10–90%")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "perturbation_vs_position.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2: vs injection depth
fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
for ax, pos in zip(axs, [16, 50]):
    for name, tag, c in MODELS:
        z, r, p = load(tag, str(pos))
        if z is None:
            continue
        dh = z["dh"]
        med, lo, hi = [], [], []
        for i in range(dh.shape[0]):
            v = per_direction_max(dh, p, r, inj=i)
            med.append(np.median(v)); lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
        inj = z["layers"]
        ax.semilogy(inj, med, "o-", ms=4, lw=1.2, color=c, label=name)
        ax.fill_between(inj, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3), color=c, alpha=0.12)
    ax.set_xlabel(r"block where the perturbation is injected, $\ell_0$")
    ax.set_title(f"perturbation at position $s={pos}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5)
axs[0].set_ylabel(r"$\max_{1\leq\tau\leq50}\|\delta h_{s+\tau,\,L}\|/r$   (last layer)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "perturbation_vs_depth.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved perturbation_vs_position.png and perturbation_vs_depth.png")
