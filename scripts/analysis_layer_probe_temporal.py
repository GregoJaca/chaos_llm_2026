"""Layer probe (b): same-position response vs transported response, with 10-90% bands."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "probe2")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 7.5})

MODELS = [("DeepSeek-1.5B", "deepseek", 0.1, "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", 0.1, "tab:orange"),
          ("Phi-4-mini", "phi4mini", 0.3, "tab:green")]

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True)
for ax, (name, tag, r0, c) in zip(axs, MODELS):
    f = os.path.join(B, tag, "pos16", "layer_probe.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); r = float(z["r"]); p = int(z["pos"])
    for off, ls, lab in [(0, "-", "same position"), (5, "--", "$t+5$"),
                         (20, ":", "$t+20$")]:
        v = z["dh"][0][:, :, p + off] / r      # [dir, block]
        med = np.median(v, axis=0)
        ax.semilogy(np.arange(v.shape[1]), np.maximum(med, 1e-4), ls, lw=1.4, color=c,
                    label=f"{name} {lab}")
        ax.fill_between(np.arange(v.shape[1]),
                        np.maximum(np.percentile(v, 10, axis=0), 1e-4),
                        np.maximum(np.percentile(v, 90, axis=0), 1e-4), color=c, alpha=0.10)
    ax.set_xlabel("block $\\ell$")
    ax.set_title(f"{name}, perturbation at $s=16$, $r={r0}$")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7)
axs[0].set_ylabel(r"$\|\delta h_{16+\mathrm{offset},\ell}\|/r$")
fig.suptitle("Where the response lives: at the perturbed position, 5 tokens later, 20 tokens later\n"
             "line = median over directions, band = 10–90%", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_probe_temporal.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved")
