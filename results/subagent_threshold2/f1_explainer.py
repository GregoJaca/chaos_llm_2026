"""fig1 -- WHAT IS PREDICTED, one-panel explainer built from real data.

w100 (propulsion-review prompt), nudge r=0.5:
  black  = decision gap m_t (winner logit - runner-up logit), no nudge
  red    = same gap nudged in fragile direction d12 -> dips below 0 at t=123
  green  = same gap nudged in safe direction d18 -> widens at the near-tie,
           never dips below 0
Observed free-run first change for d12 is exactly step 123; d18 never changes.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common2 as C

c = C.load_case("w100")
r = 0.5
ir = int(np.argmin(np.abs(c.radii - r)))
d_red, d_green = 12, 18
m = c.m
mr = m + c.dm[ir, d_red]
mg = m + c.dm[ir, d_green]

first_cross = int(np.where(mr <= 0)[0][0])
assert first_cross == 123 and (mg > 0).all()
obs = np.asarray([e for e in C.load_free("all")
                  if e["source"] == "fr_w100_large.npz" and e["r"] == r][0]["obs"])
assert obs[d_red] == 123 and obs[d_green] == -1

fig, ax = plt.subplots(figsize=(13, 6.6))
t = np.arange(c.T)

ax.plot(t, m, color="0.15", lw=1.7, zorder=3,
        label="decision gap without any nudge = winner logit - runner-up logit")
ax.plot(t, mr, color="#d62728", lw=1.4, ls="--", zorder=4,
        label=f"gap with a nudge of strength 0.5, fragile direction d{d_red}")
ax.plot(t, mg, color="#2ca02c", lw=1.3, ls=":", zorder=4,
        label=f"gap with the same nudge, safe direction d{d_green}")
ax.axhline(0, color="k", lw=1.3, zorder=2,
           label="overtake line: below zero the runner-up word wins and the text changes")

ax.plot([123], [mr[123]], "o", color="#d62728", ms=10, zorder=6,
        markeredgecolor="k", markeredgewidth=0.9)

ax.annotate("near-tie at step 123:\ngap is 0.020, about 240x smaller\n"
            "than the typical gap (~4.8).\nEverything hinges on steps like this.",
            xy=(123, m[123]), xytext=(0.40, 0.90), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.2),
            fontsize=10.6, color="0.15",
            bbox=dict(fc="white", ec="0.7", alpha=0.95))

ax.annotate("predicted first changed word = step 123.\n"
            "The real generation from the nudged prompt\n"
            "changed exactly at step 123. The safe direction\n"
            "never changed any word (verified over 300 words).",
            xy=(123, mr[123]), xytext=(0.655, 0.86), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
            fontsize=10.6, color="#a01717",
            bbox=dict(fc="white", ec="#d62728", alpha=0.95))

dm123 = c.dm[ir, d_red, 123]
rc = m[123] * r / (-dm123)
box = ("How big a nudge flips the word at step 123?\n"
       "threshold = gap / (gap-shrink per unit nudge)\n"
       f"          = {m[123]:.3f} / ({-dm123:.4f} / {r})  =  {rc:.2f}\n"
       "Any nudge stronger than 0.25 swaps that word; the first\n"
       "step whose threshold falls below the nudge is the\n"
       "predicted first change. Only steps where the nudge\n"
       "SHRINKS the gap can flip - the green curve WIDENS\n"
       "the gap at the tie, so it can never flip anything.")
ax.text(1.0, 1.9, box, transform=ax.transData, ha="right", va="top",
        fontsize=10.4, family="monospace",
        bbox=dict(fc="#fffbe8", ec="0.6"))

ax.set_yscale("symlog", linthresh=0.01, linscale=0.3)
ax.set_xlim(-4, 304)
ax.set_ylim(-0.04, 140)
ax.set_xlabel("generation step (one generated word per step)", fontsize=12.5)
ax.set_ylabel("decision gap between the model's top two word choices\n"
              "(log scale so gaps from 0.02 to 13 all fit)", fontsize=12.5)
ax.set_title("What the threshold prediction predicts: the decision gap at every "
             "generated word,\nand how one fixed nudge to the prompt shifts it "
             "(DeepSeek-R1-Distill-Qwen-1.5B, real data)",
             fontsize=13)
ax.legend(loc="upper left", fontsize=10.6, framealpha=0.95, borderpad=0.7)
ax.grid(alpha=0.22, which="both", lw=0.5)
fig.tight_layout()
fig.savefig("fig1_what_is_predicted.png", dpi=170)
print("saved fig1_what_is_predicted.png; crossing", first_cross,
      "obs", obs[d_red], "green min", round(float(mg.min()), 4))
