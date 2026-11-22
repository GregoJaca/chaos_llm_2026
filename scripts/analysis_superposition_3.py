"""Superposition: the three panels the user chose (a: scatter, b: ratio, f: error vs amplitude)."""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "superposition")
OUT = os.path.join(ROOT, "notebook", "figures", "transport")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

data = {}
for d in sorted(glob.glob(os.path.join(B, "deepseek_r*"))) + [os.path.join(B, "deepseek")]:
    f = os.path.join(d, "superposition.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    L = list(z["layers"])[-1]
    data[float(z["r"])] = dict(nA=z[f"nA_L{L}"], nS=z[f"nS_L{L}"], err=z[f"err_L{L}"],
                               cos=z[f"cos_L{L}"])
rs = sorted(data)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axs[0]
for r in rs:
    x = data[r]
    ax.loglog(x["nA"].ravel(), x["nS"].ravel(), ".", ms=2, alpha=0.35, label=f"$r={r}$")
ax.plot([1e-4, 1e2], [1e-4, 1e2], "k--", lw=1.1, label="perfect prediction")
ax.set_xlabel(r"$\|A\|$   (response to all positions perturbed at once)")
ax.set_ylabel(r"$\|S\|$   (sum of the 65 single-position responses)")
ax.set_title("(a) Predicted vs measured amplitude\none point per (token, trial)")
ax.legend(fontsize=7, markerscale=3); ax.grid(alpha=0.25, which="both")

ax = axs[1]
for r in rs:
    ratio = (data[r]["nS"] / np.maximum(data[r]["nA"], 1e-12)).ravel()
    ax.hist(np.log10(np.maximum(ratio, 1e-8)), bins=70, histtype="step", lw=1.3, label=f"$r={r}$")
ax.axvline(0, color="k", ls=":", lw=1.0, label="exact")
ax.set_xlabel(r"$\log_{10}(\|S\|/\|A\|)$")
ax.set_ylabel("count (tokens × trials)")
ax.set_title("(b) Ratio of the predicted to the measured norm\n0 = perfect")
ax.legend(fontsize=7); ax.grid(alpha=0.25)

ax = axs[2]
for r in rs:
    a = data[r]["nA"].ravel(); e = data[r]["err"].ravel()
    bins = np.geomspace(max(a.min(), 1e-6), a.max(), 12)
    idx = np.digitize(a, bins) - 1
    med = [np.median(e[idx == i]) if np.any(idx == i) else np.nan for i in range(len(bins) - 1)]
    ax.semilogx(0.5 * (bins[1:] + bins[:-1]), med, "o-", ms=4, lw=1.2, label=f"$r={r}$")
ax.set_xlabel(r"amplitude of the joint response $\|A\|$")
ax.set_ylabel(r"relative error $\|A-S\|/\|A\|$")
ax.set_title("(c) Relative error vs amplitude\nthe error is large only when the response is at the noise floor")
ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "superposition_3.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved superposition_3.png")
