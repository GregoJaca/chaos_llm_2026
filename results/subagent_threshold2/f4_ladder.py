"""fig4 -- DANGER LADDER: which near-tie controls each direction, made readable.

One horizontal row per perturbation direction, sorted from most to least
fragile. The row shows
  - a gray bar reaching to r_* = the smallest nudge (built from the clean small-
    nudge response, r=0.001/0.01) that flips any word along that direction;
  - a colored dot at r_*: its color gives the word index of the controlling
    near-tie (where the predicted first flip lands), also printed as text;
  - an open gray circle where the sign-blind |gap-shrink| rule would wrongly
    put the threshold (steps whose gap-shift is actually positive - the nudge
    WIDENS the gap there and nothing can flip);
  - dashed vertical lines: the nudge strengths actually generated, annotated
    with how many of the 20 directions really flipped there.
Two prompts: (a) interstellar propulsion review, (b) cybersecurity review.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common2 as C


def ladder_data(nm):
    c = C.load_case(nm)
    rc = c.rc_signed()[0]
    rc[~np.isfinite(rc)] = 1e30
    rstar = rc.min(axis=1)
    tstar = rc.argmin(axis=1)
    rcA = c.rc_adv()[0]
    rcA[~np.isfinite(rcA)] = 1e30
    rstarA = rcA.min(axis=1)
    return c, rstar, tstar, rstarA


def draw(ax, nm, radii_obs, xlo, xhi, cb_ticks):
    c, rstar, tstar, rstarA = ladder_data(nm)
    order = np.argsort(rstar)
    cmap = plt.cm.plasma
    vmin, vmax = 0, max(cb_ticks)
    for yi, d in enumerate(order):
        # gray bar
        ax.plot([xlo, rstar[d]], [yi, yi], color="#bdbdbd", lw=6.5,
                solid_capstyle="butt", zorder=2)
        # sign-blind threshold if it differs
        if rstarA[d] < 0.97 * rstar[d]:
            ax.plot([rstarA[d], rstar[d]], [yi, yi], color="0.45", lw=1.0,
                    ls=":", zorder=3)
            ax.plot([rstarA[d]], [yi], "o", mfc="none", mec="0.35", ms=8.5,
                    mew=1.4, zorder=4)
        # colored dot at the signed threshold
        ax.plot([rstar[d]], [yi], "o", color=cmap((tstar[d] - vmin) / (vmax - vmin)),
                mec="k", ms=9.5, mew=0.7, zorder=5)
        ax.text(xhi * 1.06, yi, f"word {tstar[d]:3d}, gap {c.m[tstar[d]]:.3f}",
                fontsize=8.8, va="center", ha="left", color="0.15")
        ax.text(xlo * 0.94, yi, f"d{d:2d}", fontsize=8.8, va="center", ha="right",
                color="0.35")
    # observed free-run radii
    for k, (r, nflip) in enumerate(radii_obs):
        ax.axvline(r, color="0.25", lw=1.1, ls="--", alpha=0.85, zorder=1)
        ytxt = -1.1 - 0.95 * (k % 2)
        ax.text(r, ytxt, f"r={r:g}\n{nflip}/20 flipped", fontsize=8.8,
                ha="center", va="center", color="0.15",
                bbox=dict(fc="white", ec="0.75", alpha=0.9, pad=1.5))
    ax.set_xscale("log")
    ax.set_xlim(xlo * 0.55, xhi * 4.4)
    ax.set_ylim(len(order) - 0.4, -3.0)
    ax.set_yticks([])
    ax.set_xlabel("smallest nudge that flips any word in that direction "
                  "(log scale)\nlonger bar = more robust; open circle = "
                  "sign-blind rule cries wolf", fontsize=11.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.16, ticks=cb_ticks)
    cb.set_label("word index of the controlling near-tie\n(the word the first "
                 "flip lands on)", fontsize=9.6)
    cb.ax.tick_params(labelsize=8.5)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(17, 7.4),
                               gridspec_kw=dict(width_ratios=[1, 1]))

# (a) propulsion w100 (skip the tiny radii: they are off-scale on this ladder)
c100 = C.load_case("w100")
obs100 = []
for e in C.load_free("all"):
    if e["geom"] == "w100" and e["r"] >= 0.03:
        obs100.append((e["r"], int((np.asarray(e["obs"]) >= 0).sum())))
obs100 = sorted(obs100)
draw(axa, "w100", obs100, 0.09, 3.6, [2, 14, 16, 116, 123, 143])
_, rs100, ts100, _ = ladder_data("w100")
import collections
tc = collections.Counter(ts100.tolist())
top_ties = ", ".join(f"word {t} ({n} dirs)" for t, n in
                     tc.most_common(3))
axa.set_title(f"(a) propulsion-review prompt, window 100:\n"
              f"top controlling near-ties: {top_ties}", fontsize=11.8)

# (b) cyber
cy = C.load_case("cyber")
obs_cy = [(float(e["r"]), int((np.asarray(e["obs"]) >= 0).sum()))
          for e in C.load_free("all") if e.get("prompt") == "cyber"]
draw(axb, "cyber", sorted(obs_cy), 0.09, 10.9, [1, 17, 34, 39, 51, 79])
_, _, ts_cy, _ = ladder_data("cyber")
tcc = collections.Counter(ts_cy.tolist())
top_cy = ", ".join(f"word {t} ({n})" for t, n in tcc.most_common(3))
axb.set_title(f"(b) cybersecurity prompt:\n"
              f"fragility spread over several ties: {top_cy}", fontsize=11.8)

fig.suptitle("Danger ladder: every direction sorted by the nudge that first "
             "breaks it, and the word where it breaks\n(replaces the "
             "threshold control map; all quantities from real teacher-forced "
             "data, smallest clean nudge)", fontsize=13.5, y=0.985)
handles = [
    Line2D([], [], marker="o", ls="", mfc="0.35", mec="0.35", ms=9,
           label="colored dot: true (signed) threshold + controlling word"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="0.35", ms=9,
           label="open circle: sign-blind |shift| threshold (unphysical)"),
    Line2D([], [], color="0.25", ls="--", lw=1.2,
           label="dashed lines: nudge strengths actually generated"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.2,
           frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.text(0.5, -0.055,
         "red flag: d14's bar says any nudge > 0.33 flips it at word 123, but "
         "nothing flipped at r=0.5 or 1 - at r=2 it finally flips at word 2, "
         "not 123:\nits gap-shrink at word 123 turns into a gap-widening at "
         "large nudges (same superlinear drift as figs 5-6); the ladder is "
         "built at the small clean nudge, so treat it as the r < 0.3 danger map",
         fontsize=10, ha="center", color="#a01717")
fig.subplots_adjust(left=0.025, right=0.975, top=0.86, bottom=0.145, wspace=0.45)
fig.savefig("fig4_danger_ladder.png", dpi=165, bbox_inches="tight")
print("saved fig4_danger_ladder.png; w100 obs radii:", obs100,
      "; cyber:", sorted(obs_cy))
print("w100 tie counts:", dict(tc), "| cyber tie counts:", dict(tcc))
