"""The window flush: the response dies layer by layer as the perturbed prompt
leaves the sliding window."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(B, allow_pickle=True)
L = list(d["layers"]); radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.1)))
curves = {}
for l in [1, 2, 3, 27]:
    li = L.index(l)
    curves[l] = np.median(d["dh_norm"][ri, :, li, :], axis=0)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
for l, c in zip([1, 2, 3], ["tab:red", "tab:orange", "tab:green"]):
    ax.semilogy(np.arange(300), np.maximum(curves[l], 1e-8), lw=1.4, color=c,
                label=fr"layer $\ell={l}$")
ax.semilogy(np.arange(300), np.maximum(curves[27], 1e-8), lw=1.0, color="0.5",
            label=r"layer $\ell=27$")
for t, c in [(92, "tab:red"), (191, "tab:orange"), (290, "tab:green")]:
    ax.axvline(t, color=c, ls="--", lw=1)
ax.set_ylim(1e-8, 3)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$a(t,\ell)=\|\delta h_{t,\ell}\|_2$")
ax.set_title("(a) The response dies layer by layer, 99 tokens apart\n(the window is $W=100$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axs[1]
for l, c in zip([1, 2, 3], ["tab:red", "tab:orange", "tab:green"]):
    m = curves[l] > 0
    ax.semilogy(np.arange(300)[m], np.maximum(curves[l][m], 1e-8), lw=1.4, color=c,
                label=fr"$\ell={l}$")
ax.set_xlim(80, 120)
ax.set_ylim(1e-5, 1e-1)
ax.set_xlabel(r"$t$ (zoom on the first drop)")
ax.set_ylabel(r"$a(t,\ell)$")
ax.set_title("(b) Zoom: layer 1 drops to exactly zero at $t=92$")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_flush.png"), dpi=190)
print("saved")
