"""Mean stretching rate by window, per prompt and per direction, with percentiles."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def per_dir(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(B, folder, "*", "a1.npz"))):
        d = np.load(p, allow_pickle=True)
        L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
        A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)
        k = 10
        t = np.arange(k, 300 - k)
        lam = np.log(A[:, t + k] / A[:, t - k]) / (2 * k)
        sel = (t >= 10) & (t <= 100)
        out[os.path.basename(os.path.dirname(p))] = lam[:, sel].mean(axis=1)
    return out


w100 = per_dir("rate_prompts")
w50 = per_dir("rate_w50")
prompts = sorted(set(w100) & set(w50))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axs[0]
for i, p in enumerate(prompts):
    ax.plot([0, 1], [w50[p].mean(), w100[p].mean()], "o-", lw=1, color="0.7")
ax.boxplot([np.concatenate([w50[p] for p in prompts]),
            np.concatenate([w100[p] for p in prompts])],
           positions=[0, 1], widths=0.35, showfliers=False)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["$W=50<P$", "$W=100>P$"])
ax.set_ylabel(r"mean $\lambda_{10}$ over $t\in[10,100]$ (nats/token)")
ax.set_title(f"(a) {len(prompts)} prompts, all directions pooled\n(thin lines join the same prompt)")
ax.grid(alpha=0.25, axis="y")

ax = axs[1]
data = [np.concatenate([w50[p] for p in prompts]), np.concatenate([w100[p] for p in prompts])]
for i, v in enumerate(data):
    ax.hist(v, bins=24, alpha=0.6, label=["$W=50$", "$W=100$"][i])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"mean $\lambda_{10}$ per direction")
ax.set_ylabel("count")
ax.set_title("(b) Shifted but overlapping: the window moves the mean rate by ~0.003 nats/token")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_growth_errors.png"), dpi=190)
from scipy.stats import mannwhitneyu
p = mannwhitneyu(data[0], data[1], alternative="two-sided").pvalue
print("saved; n=", len(data[0]), len(data[1]), "p=", p,
      "means", round(float(data[0].mean()), 4), round(float(data[1].mean()), 4))
