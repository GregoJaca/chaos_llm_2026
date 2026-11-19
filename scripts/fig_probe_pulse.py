"""Growth of a single-position kick with lag, per model, with 10-90% bands."""
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
        dh = d["dh"]; r = float(d["r"]); p = int(d["pos"])
        inj0 = 0                       # kick at block 0
        x = dh[inj0, :, -1, p + 1:p + 51] / r     # [dir, lag]
        lag = np.arange(1, x.shape[1] + 1)
        ax.semilogy(lag, np.median(x, axis=0), lw=1.5, color=c, label=name)
        ax.fill_between(lag, np.percentile(x, 10, axis=0), np.percentile(x, 90, axis=0),
                        color=c, alpha=0.13)
    ax.set_xlabel(r"lag $\tau$ after the kick")
    ax.set_ylabel(r"$\|\delta h_{s+\tau,L}\|\,/\,r$")
    ax.set_title(f"kick at $s={int(np.load(os.path.join(B, 'deepseek_ds1', pos + '.npz'))['pos'])}$, "
                 f"injection at block 0")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
fig.suptitle("Single-token kick: median and 10–90% over 10 directions (last layer)", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "probe_pulse.png"), dpi=190, bbox_inches="tight")
print("saved")
