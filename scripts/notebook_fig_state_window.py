"""Single-position perturbation under different attention windows."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\state_window"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

WINS = [50, 200, 4096]
cols = {50: "tab:red", 200: "tab:green", 4096: "0.4"}

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
data = {w: np.load(os.path.join(B, f"w{w}", "state2.npz"), allow_pickle=True)
        for w in WINS if os.path.exists(os.path.join(B, f"w{w}", "state2.npz"))}

for w, d in data.items():
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1))); li = layers.index(27)
    for s in [16, 50]:
        si = steps.index(s)
        a = np.median(d["dh"][ri, si, :, li, :], axis=0)
        axs[0].semilogy(np.arange(300) - s, np.maximum(a, 1e-12), lw=1.1,
                        color=cols[w], ls="-" if s == 16 else "--",
                        label=f"$W={w}$, $s={s}$")
axs[0].set_xlim(-5, 150)
axs[0].axvline(0, color="k", lw=0.6)
axs[0].set_xlabel(r"lag $\tau=t-s$")
axs[0].set_ylabel(r"$a(s+\tau,27)$")
axs[0].set_title("(a) Decay of a single-position perturbation")
axs[0].legend(fontsize=6.5, ncol=2)
axs[0].grid(alpha=0.25)

for w, d in data.items():
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1)))
    vals = [np.median(d["dh"][ri, steps.index(16), :, layers.index(l), 16]) for l in layers]
    axs[1].semilogy(layers, vals, "o-", color=cols[w], label=f"$W={w}$")
axs[1].set_xlabel(r"layer $\ell$")
axs[1].set_ylabel(r"$a(16,\ell)$")
axs[1].set_title("(b) Depth profile of the same perturbation")
axs[1].legend(fontsize=8)
axs[1].grid(alpha=0.25)

# (c) immediate size and half-life vs window, s=16
xs, imm, half = [], [], []
for w, d in data.items():
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1))); li = layers.index(27)
    si = steps.index(16)
    a = np.median(d["dh"][ri, si, :, li, :], axis=0)
    seg = a[16:]
    below = np.where(seg < 0.5 * seg[0])[0]
    xs.append(w); imm.append(a[16]); half.append(int(below[0]) if len(below) else -1)
xb = np.arange(len(xs))
axs[2].bar(xb, imm, color=["tab:red", "tab:green", "0.5"])
for i, (v, h) in enumerate(zip(imm, half)):
    axs[2].text(i, v + 0.3, f"{v:.1f}\nhalf-life {h}", ha="center", fontsize=8)
axs[2].set_xticks(xb); axs[2].set_xticklabels([f"$W={w}$" for w in xs])
axs[2].set_ylabel(r"$a(16,27)$ at the perturbation")
axs[2].set_ylim(0, max(imm) * 1.25)
axs[2].set_title("(c) $W<P$ amplifies the immediate response\nand lengthens its half-life")
axs[2].grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_window.png"), dpi=190)
print("saved", list(data))
print("immediate", dict(zip(xs, [round(float(v), 2) for v in imm])), "half-life", dict(zip(xs, half)))
