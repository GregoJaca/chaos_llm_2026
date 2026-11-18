"""Per-direction mean stretching rate by window, with tests."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def mean_rate(w, t0=10, t1=100):
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)
    k = 10
    out = []
    for di in range(A.shape[0]):
        a = A[di]
        t = np.arange(k, 300 - k)
        lam = np.log(a[t + k] / a[t - k]) / (2 * k)
        sel = (t >= t0) & (t <= t1)
        out.append(lam[sel].mean())
    return np.array(out)


wins = [50, 100, 200]
data = {w: mean_rate(w) for w in wins}

fig, ax = plt.subplots(figsize=(7.6, 4.6))
rng = np.random.default_rng(0)
for i, w in enumerate(wins):
    v = data[w]
    x = i + rng.uniform(-0.08, 0.08, len(v))
    ax.scatter(x, v, s=28, color="tab:blue", alpha=0.7, zorder=3)
    ax.hlines(np.median(v), i - 0.25, i + 0.25, color="tab:red", lw=2.5, zorder=4)
    ax.vlines(i, np.percentile(v, 25), np.percentile(v, 75), color="tab:red", lw=1.2, alpha=0.6)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(wins))); ax.set_xticklabels([f"$W={w}$" for w in wins])
ax.set_ylabel(r"mean $\lambda_{10}$ over $t\in[10,100]$  (nats/token)")
p1 = mannwhitneyu(data[50], data[100], alternative="two-sided").pvalue
p2 = mannwhitneyu(data[100], data[200], alternative="two-sided").pvalue
ax.set_title(f"Direction-resolved rates (n=10 each)\n$W{{=}}50$ vs 100: $p={p1:.1e}$; 100 vs 200: $p={p2:.1e}$", fontsize=10)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "lambda_window_test.png"), dpi=190)
print("saved", {w: round(float(np.median(v)), 4) for w, v in data.items()})
