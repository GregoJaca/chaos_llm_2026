"""Finite-time rate with percentile bands, per window."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
WINS = [50, 100, 200]
COL = {50: "tab:red", 100: "tab:blue", 200: "tab:green"}
K = 10

fig, ax = plt.subplots(figsize=(9.2, 4.8))
for w in WINS:
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)      # [dirs, T]
    t = np.arange(K, 300 - K)
    lam = np.log(A[:, t + K] / A[:, t - K]) / (2 * K)       # [dirs, t]
    ax.plot(t, np.median(lam, 0), lw=1.8, color=COL[w], label=f"$W={w}$ (median)")
    ax.fill_between(t, np.percentile(lam, 10, 0), np.percentile(lam, 90, 0),
                    color=COL[w], alpha=0.15, label=f"$W={w}$ 10–90%")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$\lambda_{10}(t)=\frac{1}{20}\log\frac{a(t+10)}{a(t-10)}$  (nats/token)")
ax.set_title("Local stretching rate with direction percentiles\n(ocean-currents prompt, $r=0.1$, layer 27, 10 directions)")
ax.legend(fontsize=7.5, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "lambda_bands_windows.png"), dpi=190)
print("saved")
