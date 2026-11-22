"""Part 8a: transport kernel comparison - Mamba (no attention) vs the LLMs.

Data: results/ctrl_local/mamba130/kernel/kernel.npy (own run, r=0.1, 8 dirs,
teacher-forced 300 tokens, last layer) and results/ctrl_local/kernels/*.npz
(the LLM kernels, same protocol, W=100 for DeepSeek, r=0.1).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERN = np.load(os.path.join(ROOT, "results", "ctrl_local", "mamba130", "kernel", "kernel.npy"))  # [P, K, T]
SUP = json.load(open(os.path.join(ROOT, "results", "ctrl_local", "mamba130", "kernel", "superposition.json")))
KS = os.path.join(ROOT, "results", "ctrl_local", "kernels")
OUT = os.path.join(ROOT, "notebook", "figures_controls")

C_MAM = "#0072B2"
C_DS = "#D55E00"
C_PHI = "#009E73"

Kmam = np.median(KERN, axis=1) / 0.1  # [P, T]
Pm = Kmam.shape[0]

zds = np.load(os.path.join(KS, "deepseek.npz"))
Kds = np.median(zds["dh"][:, :, -1, :], axis=1) / float(zds["r"])  # [P, T]
Pd = Kds.shape[0]
zphi = np.load(os.path.join(KS, "phi4mini.npz"))
Kphi = np.median(zphi["dh"][:, :, -1, :], axis=1) / float(zphi["r"])

fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
ax = axs[0, 0]
im = ax.imshow(np.log10(np.maximum(Kmam, 1e-6)), aspect="auto", origin="lower",
               cmap="viridis", extent=[0, Kmam.shape[1], 0, Pm])
ax.plot(Kmam.argmax(axis=1), np.arange(Pm), "w-", lw=1.2)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"perturbed prompt position $p$")
ax.set_title("(a) Mamba-130m $\\log_{10}K(p,t)$: no front - every position's\nresponse enters once and decays; peak (white) is always at $t\\approx0$")
plt.colorbar(im, ax=ax, label=r"$\log_{10}K$")

ax = axs[0, 1]
im = ax.imshow(np.log10(np.maximum(Kds, 1e-6)), aspect="auto", origin="lower",
               cmap="viridis", extent=[0, Kds.shape[1], 0, Pd])
ax.plot(Kds.argmax(axis=1), np.arange(Pd), "w-", lw=1.2)
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"perturbed prompt position $p$")
ax.set_title("(b) DeepSeek-1.5B $\\log_{10}K(p,t)$ (master's kernel): persistent\nbackground + travelling front (white: peak $t$ moves with $p$)")
plt.colorbar(im, ax=ax, label=r"$\log_{10}K$")

ax = axs[1, 0]
# peak time vs position
ax.plot(np.arange(Pm) / Pm, Kmam.argmax(axis=1), "o-", ms=2, color=C_MAM, label="Mamba-130m")
ax.plot(np.arange(Pd) / Pd, Kds.argmax(axis=1), "o-", ms=2, color=C_DS, label="DeepSeek-1.5B")
ax.set_xlabel(r"$p/P$")
ax.set_ylabel(r"peak time $t^*(p)$ of $K(p,t)$")
ax.set_title("(c) Peak time: flat at 0 for the control; the front slope $\\approx1$\nis an attention transport property")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 1]
# background: first 5 positions
bgm = np.median(Kmam[:5], axis=0)
bgd = np.median(Kds[:5], axis=0)
bgp = np.median(Kphi[:5], axis=1) if False else np.median(Kphi[:5], axis=0)
ax.semilogy(np.arange(len(bgm)), bgm, color=C_MAM, label="Mamba-130m")
ax.semilogy(np.arange(len(bgd)), bgd, color=C_DS, label="DeepSeek-1.5B")
ax.semilogy(np.arange(len(bgp)), bgp, color=C_PHI, alpha=0.7, label="Phi-4-mini")
ax.set_xlabel(r"generated token $t$")
ax.set_ylabel(r"$K(p\leq5,t)$ (background)")
ax.set_title("(d) Persistent background: decays in DeepSeek, rises in Phi,\ndecays slowly in the control - all well above zero")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_kernel.png"), dpi=190); plt.close(fig)

# superposition figure
fig, ax = plt.subplots(figsize=(8.6, 4.6))
rows = [
    ("Mamba-130m r=0.01/pos", SUP["0.01"]["rel_err"], SUP["0.01"]["cos"]),
    ("Mamba-130m r=0.03/pos", SUP["0.03"]["rel_err"], SUP["0.03"]["cos"]),
    ("Mamba-130m r=0.1/pos", SUP["0.1"]["rel_err"], SUP["0.1"]["cos"]),
]
ds_tab = [("DeepSeek r=0.01/pos", 0.11), ("DeepSeek r=0.03/pos", 0.068), ("DeepSeek r=0.1/pos", 0.155)]
xs = np.arange(len(rows))
for i, (lab, rel, cos) in enumerate(rows):
    ax.bar(i, np.median(rel), 0.55, color=C_MAM, alpha=0.85)
    ax.annotate("cos=%.3f" % np.median(cos), (i, np.median(rel)), ha="center", va="bottom", fontsize=8)
    ax.bar(i + 0.6, ds_tab[i][1], 0.55, color=C_DS, alpha=0.85)
ax.set_xticks(xs + 0.3)
ax.set_xticklabels([f"{r[0].split(' ')[1]}\n(total norm {v})" for r, v in zip(rows, ["0.08", "0.24", "0.81"])], fontsize=8)
ax.set_ylabel(r"median $\|A-S\|/\|A\|$ (superposition error)")
ax.set_title("Superposition $A=\\sum_p B_p$: the control decomposes at least as well\nas the trained transformer (blue Mamba, orange DeepSeek from master.md \\u00a77.2)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_superposition.png"), dpi=190); plt.close(fig)

print("mamba kernel: background", bgm[1], "->", bgm[100], "->", bgm[299])
print("ds kernel: background", bgd[1], "->", bgd[100], "->", bgd[299])
print("ds peak t at p=0:", int(Kds[0].argmax()), " at p=P-1:", int(Kds[-1].argmax()))
print("mamba peak t at p=0:", int(Kmam[0].argmax()), " at p=P-1:", int(Kmam[-1].argmax()))
print("superposition medians:", {k: round(float(np.median(v["rel_err"])), 4) for k, v in SUP.items()})
print("wrote", OUT)
