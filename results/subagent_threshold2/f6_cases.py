"""fig6 -- three individual directions: how the threshold tube reshapes as the
nudge grows, and how the flip follows the reshaped tube.

Each panel: the signed threshold r_c-(t) = smallest nudge that flips the word
at step t, drawn twice:
  gray dashed: measured at the smallest clean nudge (r=0.001) - the 'linear'
               tube one would naively extrapolate;
  colored solid: measured at the radius actually generated;
  horizontal line: that radius. The first dip below the line = predicted first
  changed word; the black star = what the real generation did.

  d10 (fragile)   : flips at word 16 for r>=0.13; at r=1 the real run flips at
                    the very first word - teacher-forcing along the unperturbed
                    text cannot see a flip at the first position (2nd of 260
                    remaining errors in fig2).
  d12 (near-tie)  : the small-nudge tube has its minimum at word 14 (needs
                    r=1.27); the r=0.5 tube develops a deep new dip at word
                    123 - the observed flip, predicted exactly.
  d13 (resistant) : the naive tube would predict a flip at word 143 for r=2;
                    the true r=2 tube never dips below 2 - no flip, as
                    observed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common2 as C

c = C.load_case("w100")
free = {e["r"]: e for e in C.load_free("all") if e["geom"] == "w100"}
rc0 = c.rc_signed()[0]              # linear tube from r=0.001
ir_of = {float(r): i for i, r in enumerate(c.radii)}

CASES = [
    dict(d=10, r=1.0, title="fragile direction d10 - flips early, and at r=1\n"
         "the real run flips at the very first word",
         note="predicted from the r=1 tube: word 22 (orange dot)\n"
              "observed: word 0 (black star) - one of only 2 misses\n"
              "of 260 runs; a 7x-oversized nudge breaks teacher-forcing"),
    dict(d=12, r=0.5, title="near-tie direction d12 - the flip appears only\n"
         "because the tube reshapes between r=0.001 and r=0.5",
         note="the r=0.5 tube dips below the line first at word 123\n"
              "(predicted, orange) - the real run changed exactly at\n"
              "word 123 (star). The naive small-nudge tube (gray)\n"
              "would have predicted NO flip: its minimum is word 14,\n"
              "needs r=1.27"),
    dict(d=13, r=2.0, title="resistant direction d13 - never flips, and the\n"
         "true r=2 tube explains why",
         note="naive gray tube: minimum at word 143, r_c=1.26 < 2,\n"
              "so linear extrapolation wrongly predicts a flip.\n"
              "The true r=2 tube never dips below 2 (its dip filled\n"
              "in): no flip predicted, and indeed no flip observed"),
]

fig, axes = plt.subplots(3, 1, figsize=(13.5, 12.2), sharex=True)
XMAX = 175
for ax, case in zip(axes, CASES):
    d, r = case["d"], case["r"]
    ir = ir_of[float(r)]
    tube = c.rc_signed()[ir][d]          # (T,)
    lin = rc0[d]
    obs = int(np.asarray(free[float(r)]["obs"])[d])
    tt = np.arange(c.T)

    ax.plot(tt, lin, color="0.55", lw=1.5, ls="--",
            label="threshold tube measured at the smallest clean nudge "
                  "(r=0.001)\n- what a linear extrapolation would use")
    fin = np.isfinite(tube)
    ax.plot(tt[fin], tube[fin], color="#08519c", lw=2.2,
            label=f"threshold tube measured at the radius actually generated "
                  f"(r={r:g})")
    ax.axhline(r, color="#cb181d", lw=1.6,
               label=f"nudge strength r={r:g} (horizontal line)")
    # first crossing of the at-radius tube
    hit = np.where(tube <= r)[0]
    if len(hit):
        tp = int(hit[0])
        ax.plot([tp], [r], "o", color="#e08214", ms=13, mec="k", mew=0.9,
                zorder=6, label=f"predicted first changed word: {tp}")
    else:
        ax.text(0.985, r * 1.9, "at-radius tube never dips below the line:\n"
                "no flip predicted", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9.6, color="#08519c")
    # observed
    if obs >= 0:
        ax.plot([obs], [r], "*", color="k", ms=19, zorder=7,
                label=f"observed first changed word: {obs}")
    else:
        ax.text(0.015, r * 1.9, "observed: no word changed in all 300",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=9.8, color="k")
    ax.text(0.985, 0.965, case["note"], transform=ax.transAxes, ha="right",
            va="top", fontsize=9.7,
            bbox=dict(fc="#fffbe8", ec="0.65"))
    ax.set_yscale("log")
    ax.set_ylim(0.06, 40)
    ax.set_xlim(0, XMAX)
    ax.set_ylabel("smallest nudge that flips\nthe word at that step\n"
                  "(log scale; lower = more fragile)", fontsize=10.5)
    ax.set_title(case["title"], fontsize=11.6, loc="left")
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.93, ncol=2)
    ax.grid(alpha=0.3, which="both", lw=0.5)

axes[-1].set_xlabel("word index in the 300-word generation "
                     "(shown to word 175)", fontsize=12)
fig.suptitle("The threshold rule on three individual directions - the tube "
             "reshapes as the nudge grows (superlinear response),\nand the "
             "first flip follows the reshaped tube, not the small-nudge one",
             fontsize=13.5)
fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.06, hspace=0.34)
fig.savefig("fig6_three_directions.png", dpi=165)
print("saved fig6_three_directions.png")
