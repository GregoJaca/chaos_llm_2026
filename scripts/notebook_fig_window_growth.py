"""Growth rate vs attention window across prompts."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def collect(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(B, folder, "*", "a1.npz"))):
        d = np.load(p, allow_pickle=True)
        L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
        a = np.maximum(np.median(d["dh_norm"][ri, :, li, :], axis=0), 1e-12)
        k = 10; t = np.arange(k, 300 - k)
        loc = np.log(a[t + k] / a[t - k]) / (2 * k)
        out[os.path.basename(os.path.dirname(p))] = (loc.max(), t[loc.argmax()], (loc > 0).mean())
    return out


w100 = collect("rate_prompts")
w50 = collect("rate_w50")
keys = sorted(set(w100) & set(w50))

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
x = np.arange(len(keys))
for ax, idx, ylab, title in [
        (axs[0], 0, r"peak growth rate $\max_t \lambda_{10}(t)$", "(a) Peak growth rate"),
        (axs[1], 2, r"fraction of tokens with $\lambda_{10}(t)>0$", "(b) Fraction of growing tokens")]:
    ax.bar(x - 0.2, [w50[k][idx] for k in keys], 0.4, label="$W=50<P$", color="tab:red")
    ax.bar(x + 0.2, [w100[k][idx] for k in keys], 0.4, label="$W=100>P$", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels([f"prompt {k}" for k in keys])
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_growth_summary.png"), dpi=190)
print("saved")
