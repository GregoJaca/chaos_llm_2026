"""Full 28-layer depth profile: absolute response, relative response, and the state norm."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big\all_layers\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})

d = np.load(B)
A = d["dh_norm"]; layers = np.asarray(d["layers"]); h = d["norm_h0"]
ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
r = d["radii"][ri]
times = [(1, "tab:cyan"), (5, "tab:blue"), (20, "tab:orange"), (50, "tab:green"),
         (100, "tab:red"), (200, "tab:purple")]

fig, axs = plt.subplots(1, 3, figsize=(17.5, 4.6))

ax = axs[0]
for t, c in times:
    q = np.median(A[ri, :, :, t], axis=0) / r
    ax.semilogy(layers, np.maximum(q, 1e-6), "o-", ms=3, lw=1.1, color=c, label=f"$t={t}$")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,r$")
ax.set_title("(a) Absolute response grows with depth")
ax.legend(ncol=2)
ax.grid(alpha=0.25, which="both")

ax = axs[1]
for t, c in times:
    q = np.median(A[ri, :, :, t], axis=0) / np.maximum(h[:, t], 1e-9)
    ax.semilogy(layers, np.maximum(q, 1e-12), "o-", ms=3, lw=1.1, color=c, label=f"$t={t}$")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$\|\delta h_{t,\ell}\|\,/\,\|h_{t,\ell}\|$")
ax.set_title("(b) Relative response is flat after layer 2")
ax.legend(ncol=2)
ax.grid(alpha=0.25, which="both")

ax = axs[2]
for t, c in times:
    ax.semilogy(layers, h[:, t], "o-", ms=3, lw=1.1, color=c, label=f"$t={t}$")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$\|h_{t,\ell}\|$")
ax.set_title("(c) The state norm grows with depth\n(this is what (a) follows)")
ax.legend(ncol=2)
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(OUT + r"\depth_full_profile.png", dpi=190)
print("saved")
print("gain r=0.1, t=5:", [round(float(x), 3) for x in (np.median(A[ri, :, :, 5], axis=0) / r)[::4]])
print("rel t=5:", ["%.1e" % float(x) for x in (np.median(A[ri, :, :, 5], axis=0) / h[:, 5])[::4]])
