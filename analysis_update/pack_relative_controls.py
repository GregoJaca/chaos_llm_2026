import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

CTRL = {"Mamba-130m": "results_big_small/ctrl_Mamba_130m.npz",
        "Pythia-410m": "results_big_small/ctrl_Pythia_410m.npz",
        "randQwen": "results_big_small/ctrl_randQwen.npz"}
ORDER = all_names() + list(CTRL)
fig, ax = plt.subplots(figsize=(10.8, 5.2))
for n in ORDER:
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        t = min(50, z["dh"].shape[-1] - 1)
        med = z["rel"][ri, :, t]
        lo, hi = np.percentile(z["dh"][ri, :, :, t], [25, 75], axis=0) / np.maximum(z["hn"][:, t], 1e-30)
        base = max(med[1], 1e-30)
        med, lo, hi = med[1:] / base, lo[1:] / base, hi[1:] / base
        x = np.arange(len(med)) / max(len(med) - 1, 1)
        ax.plot(x, med, color=COL[n], label=n)
        ax.fill_between(x, lo, hi, color=COL[n], alpha=0.15, lw=0)
    else:
        if n in BIG:
            z = big(n); med = z["rel_med"][:, 50]
        else:
            z = load(CTRL[n]); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
            med = z["dh_med"][ri, :, 50] / np.maximum(z["norm_h0"][:, 50], 1e-30)
        med = med[1:] / max(med[1], 1e-30)
        x = np.arange(len(med)) / max(len(med) - 1, 1)
        ax.plot(x, med, color=COL[n], label=n, lw=1.8)
ax.axhline(1, color="k", ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("normalized depth  $\\ell/(L-1)$")
ax.set_ylabel(r"$[\|\delta h_\ell\|/\|h_\ell\|]$ / layer 1  ($t=50$)")
ax.set_title("Relative response vs depth: LLMs and controls (band = IQR)", loc="left")
ax.legend(fontsize=6.4, ncol=3); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_relative_controls.png")
print("done")
