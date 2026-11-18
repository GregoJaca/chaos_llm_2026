"""Local stretching rate of the perturbation, and how aggregation changes it."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(B, allow_pickle=True)
L = list(d["layers"]); radii = d["radii"]
li = L.index(27); ri = int(np.argmin(np.abs(radii - 0.1)))
A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)   # [dirs, T]

k = 10                                    # half-window in tokens
t = np.arange(k, 300 - k)
loc_each = np.log(A[:, t + k] / A[:, t - k]) / (2 * k)      # [dirs, t]
loc_mean = np.log(A.mean(0)[t + k] / A.mean(0)[t - k]) / (2 * k)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axs[0]
ax.plot(t, loc_each.T, color="0.8", lw=0.4, alpha=0.5)
ax.plot(t, loc_mean, lw=2.2, color="tab:blue", label=r"$\lambda$ of the mean curve")
ax.plot(t, np.median(loc_each, 0), lw=2.0, color="tab:red", label=r"median of per-direction $\lambda$")
ax.fill_between(t, np.percentile(loc_each, 25, 0), np.percentile(loc_each, 75, 0),
                color="tab:red", alpha=0.12, label="inter-quartile range")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"local stretching rate $\frac{1}{20}\log\frac{a(t+10)}{a(t-10)}$")
ax.set_title(r"Prompt perturbation, $r=0.1$, layer 27, 20 directions")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)

ax = axs[1]
ax.plot(t, loc_mean - np.median(loc_each, 0), lw=1.4, color="tab:purple")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$\lambda$ of the mean $-$ median of $\lambda$")
ax.set_title("The two aggregations differ by up to 0.01 nats/token;\nreporting one without the other is misleading")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "rate_aggregation.png"), dpi=190)
print("saved")
for tt in [20, 50, 100, 150, 200, 250]:
    i = int(np.argmin(np.abs(t - tt)))
    print(f"t={tt:3d}  lam(mean)={loc_mean[i]:+.4f}  median={np.median(loc_each[:, i]):+.4f}  "
          f"IQR=[{np.percentile(loc_each[:, i], 25):+.4f},{np.percentile(loc_each[:, i], 75):+.4f}]")
print("fraction of tokens with positive rate_of_mean:", float((loc_mean > 0).mean()))
