"""Full 28-layer depth profile of the response."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big\all_layers\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(B)
A = d["dh_norm"]; layers = np.asarray(d["layers"])
ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
r = d["radii"][ri]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axs[0]
for t, c in [(5, "tab:blue"), (20, "tab:orange"), (50, "tab:green"), (100, "tab:red"), (200, "tab:purple")]:
    a = np.median(A[ri, :, :, t], axis=0) / r
    ax.semilogy(layers, np.maximum(a, 1e-6), "o-", ms=3, lw=1.1, color=c, label=f"$t={t}$")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$a(t,\ell)/r$")
ax.set_title("(a) Response vs depth at fixed times")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

ax = axs[1]
g = np.median(A[ri, :, :, :50] / r, axis=(0, 2))
ax.semilogy(layers, g, "o-", ms=3, color="tab:blue")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"median $a/r$ over the first 50 tokens")
ax.set_title("(b) Depth gain (all 28 layers)")
ax.grid(alpha=0.25, which="both")

ax = axs[2]
for t, c in [(5, "tab:blue"), (50, "tab:green"), (200, "tab:purple")]:
    q = np.median(A[ri, :, :, t] / np.maximum(d["norm_h0"][:, t], 1e-9), axis=0)
    ax.semilogy(layers, np.maximum(q, 1e-9), "o-", ms=3, lw=1.1, color=c, label=f"$t={t}$")
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel(r"$a(t,\ell)/\|h_{t,\ell}\|$")
ax.set_title("(c) Relative response vs depth")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(OUT + r"\depth_full_profile.png", dpi=190)
print("saved; gain L0..L27:", [round(float(x), 3) for x in g[::4]])
