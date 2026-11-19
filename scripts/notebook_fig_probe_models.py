"""Cross-model single-position kick: downstream gain vs injection layer."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "probe_models")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})

MODELS = [("DeepSeek-1.5B", "deepseek_ds1", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:red")]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, pos in zip(axs, ["pos16", "pos50"]):
    for name, tag, c in MODELS:
        f = os.path.join(B, tag, pos + ".npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        dh = d["dh"]; inj = np.asarray(d["layers"]); r = float(d["r"]); p = int(d["pos"])
        G = np.median(dh[:, :, -1, p + 1:p + 51], axis=(1, 2)) / r
        ax.semilogy(inj, np.maximum(G, 1e-3), "o-", ms=4, lw=1.2, color=c, label=name)
    ax.set_xlabel(r"injection layer $\ell_0$")
    ax.set_ylabel(r"$\max_{1\leq\tau\leq50}\|\delta h_{s+\tau,\,L}\|\,/\,r$")
    ax.set_title(f"(a) kick at generated position $s={p}$" if pos == "pos16"
                 else f"(b) kick at generated position $s={p}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "probe_models.png"), dpi=190)
print("saved")
