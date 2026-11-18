"""Post-clip dynamics: rate vs time since the context exceeded the window."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
P = 63

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
for w, c in [(50, "tab:red"), (100, "tab:blue"), (200, "tab:green"), (400, "tab:orange")]:
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)
    k = 10
    t = np.arange(k, 300 - k)
    lam = np.log(np.median(A[:, t + k], 0) / np.median(A[:, t - k], 0)) / (2 * k)
    tau = t - (w - P)
    ax.plot(tau, lam, lw=1.2, color=c, label=f"$W={w}$")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-40, 160)
ax.set_xlabel(r"$\tau = t-(W-P)$: tokens since the context exceeded the window")
ax.set_ylabel(r"$\lambda_{10}$ (median over directions)")
ax.set_title("(a) The post-clip dynamics is not universal:\n$W=100$ rises after clipping, $W=200$ does not")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axs[1]
for w, c in [(50, "tab:red"), (100, "tab:blue"), (200, "tab:green"), (400, "tab:orange")]:
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)
    k = 10
    t = np.arange(k, 300 - k)
    lam = np.log(np.median(A[:, t + k], 0) / np.median(A[:, t - k], 0)) / (2 * k)
    sel = (t >= 10) & (t <= 100)
    ax.bar(str(w), lam[sel].mean(), color=c)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"window $W$")
ax.set_ylabel(r"mean $\lambda_{10}$ over $t\in[10,100]$")
ax.set_title("(b) Naive pooled comparison is confounded by\nwhere the clip falls in the interval")
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_tau_rate.png"), dpi=190)
print("saved")
