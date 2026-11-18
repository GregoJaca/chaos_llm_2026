"""Directions differ in sign, not just magnitude -- tested at ALL near-ties,
not only the one at t=123.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
m = d["margins"]; radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.3)))
dm = d["dm"][ri] / radii[ri]        # per unit perturbation radius
ties = np.argsort(m)[:8]
X = dm[:, ties].T                    # [8 ties, 20 dirs]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})

ax = axs[0]
vmax = np.abs(X).max()
im = ax.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(20)); ax.set_xticklabels(range(20), fontsize=7)
ax.set_yticks(range(len(ties)))
ax.set_yticklabels([f"t={int(t)}, m={m[t]:.4f}" for t in ties], fontsize=8)
ax.set_xlabel("perturbation direction $u$ (index)")
ax.set_ylabel("near-tie token $t$ (sorted by margin $m_t$)")
ax.set_title(r"(a) $\delta m_t(u)/r$ at the eight smallest-gap tokens" "\n"
             "red = gap widens (no flip), blue = gap shrinks (can flip)", fontsize=10)
cb = plt.colorbar(im, ax=ax, pad=0.01)
cb.set_label(r"$\delta m_t(u)/r$  (logit gap change per unit $r$)")

ax = axs[1]
neg = (X < 0).sum(axis=0)
ax.bar(range(20), neg, color=["tab:blue" if n >= 5 else "0.7" for n in neg])
ax.axhline(4, color="k", ls="--", lw=1)
ax.text(0.5, 4.15, "4/8 = no bias", fontsize=8)
ax.set_xticks(range(20)); ax.set_xticklabels(range(20), fontsize=7)
ax.set_xlabel("direction index")
ax.set_ylabel("number of the 8 near-ties\nwhere the gap shrinks (out of 8)")
ax.set_title("(b) No direction is consistently\ngood or bad across ties", fontsize=10)
ax.set_ylim(0, 8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "directions_all_ties.png"), dpi=190)
print("saved")
C = spearmanr(X.T).correlation
print("mean off-diagonal Spearman across ties:", round(float((C.sum() - 8) / 56), 3))
print("neg counts", list(neg))
