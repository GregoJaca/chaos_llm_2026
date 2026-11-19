"""Long-horizon finite-time rate: 4000 tokens."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\a1_4k_test.npz")
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

A = np.maximum(d["dh_norm"][0], 1e-12)      # [dirs, layers, T]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axs[0]
t = np.arange(1, A.shape[-1])
for li, c in zip([0, 1, 2, 3], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    a = np.median(A[:, li, :], axis=0)
    ax.semilogy(t, a[1:], lw=1.0, color=c, label=f"layer {int(d['layers'][li])}")
ax.set_xscale("log")
ax.set_xlabel(r"generated token $t$ (log)")
ax.set_ylabel(r"$a(t,\ell)=\|\delta h_{t,\ell}\|$")
ax.set_title("(a) Response over 4000 tokens")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

ax = axs[1]
for li, c in zip([0, 1, 2, 3], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    a = np.median(A[:, li, :], axis=0)
    lam = np.log(a[1:] / a[1]) / t
    ax.plot(t, lam, lw=1.0, color=c, label=f"layer {int(d['layers'][li])}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"horizon $t$ (log)")
ax.set_ylabel(r"$\Lambda(t)=\frac{1}{t}\log\frac{a(t)}{a(1)}$")
ax.set_title("(b) The rate is positive early then\nsettles at about $-0.007$ nats/token")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(OUT + r"\long_horizon.png", dpi=190)
print("saved")
