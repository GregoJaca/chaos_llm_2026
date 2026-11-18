"""Directions are not equivalent: at the same close call they push the decision
in opposite directions, and the flips are stratified by direction.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

d = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
fr = np.load(os.path.join(B, "fr_w100_large.npz"), allow_pickle=True)
margins = d["margins"]; radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.3)))
T123 = 123
dm = d["dm"][ri, :, T123]          # margin change per direction at the close call
order = np.argsort(-dm)
flip = fr["obs"]                    # [3 radii, 20]
fr_radii = fr["radii"]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axs[0]
cols = ["#2a9d8f" if dm[i] < 0 else "#b0b0b0" for i in order]
ax.bar(range(len(order)), dm[order], color=cols)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(-margins[T123], color="tab:red", ls="--", lw=1)
ax.text(0.2, -margins[T123] - 0.006, f"gap at word 123 = {margins[T123]:.4f}\n(a bar below this line flips the word)",
        color="tab:red", fontsize=8, va="top")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{i}" for i in order], fontsize=8)
ax.set_xlabel("direction (sorted by its effect at word 123)")
ax.set_ylabel("change of the top1-top2 gap at word 123\n(negative = pushes the runner-up ahead)")
ax.set_title("(a) At the same close call, directions disagree in sign\n(7 shrink the gap, 13 widen it; only one crosses the flip line)")
ax.grid(alpha=0.25, axis="y")

ax = axs[1]
for k, r in enumerate(fr_radii):
    obs = fr["obs"][k]
    xs = np.where(obs >= 0)[0]
    ax.scatter(xs, [r] * len(xs), s=70, label=f"r={r:g}: {len(xs)}/20 flip", zorder=3)
ax.set_yticks(list(fr_radii))
ax.set_yticklabels([f"r={r:g}" for r in fr_radii])
ax.set_xlabel("direction index")
ax.set_ylabel("kick size")
ax.set_xlim(-1, 20)
ax.set_title("(b) Flips are stratified by direction:\nthe same 2–3 directions break first at every size")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, axis="x")
for xi in [10, 11, 12]:
    ax.axvspan(xi - 0.4, xi + 0.4, color="tab:red", alpha=0.08)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "directions_are_not_equivalent.png"), dpi=190)
print("saved")
print("signed dm at t=123, sorted:", [round(float(x), 4) for x in dm[order]])
print("negative count", int((dm < 0).sum()))
