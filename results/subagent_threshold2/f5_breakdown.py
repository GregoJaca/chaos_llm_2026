"""fig5 -- WHEN IT BREAKS (and what exactly breaks).

(a) gap-shrink per unit nudge |dm|/r vs nudge strength r, pooled over all
    (direction, word) pairs, merging the two twin runs so r spans 1e-6..2:
       r < ~3e-4 : apparent response is float32 round-off (|dm| ~ 1e-5), the
                   'gain' inflates as noise/r - thresholds are meaningless;
       3e-3..0.3: response is cleanly proportional (gain flat ~0.0085) - this
                   is where the threshold is a real quantity;
       r > 0.5   : response grows faster than the nudge (median 1.55x by r=2),
                   and different near-ties drift at different rates (shown for
                   words 2, 123, 143) - so which tie flips first gets re-ranked.
(b) observed number of flipped directions vs the predictions of three
    variants, w100:
       same-radius signed rule  -> exact at every nudge strength (8/8 at r=1)
       linear extrapolation of the small-nudge (r=0.001) threshold field ->
                                  counts drift, steps badly wrong at r>=0.5
       delivered sign-blind rule -> over-predicts (15 vs 8 at r=1).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common2 as C

c = C.load_case("w100")          # radii 1e-3 .. 2
ct = C.load_case("w100tiny")     # radii 1e-6 .. 1e-2 (exact-twin baseline)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.6),
                              gridspec_kw=dict(width_ratios=[1.15, 1]))

# ---------------------------------------------------------------- (a) gain vs r
allr, allmed, lo, hi, src = [], [], [], [], []
for case, tag in [(ct, "twin run, tiny nudges"), (c, "main run")]:
    for i, r in enumerate(case.radii):
        g = np.abs(case.dm[i]).ravel() / r
        allr.append(r); allmed.append(np.median(g))
        lo.append(np.percentile(g, 25)); hi.append(np.percentile(g, 75))
        src.append(tag)
allr, allmed = np.array(allr), np.array(allmed)
lo, hi = np.array(lo), np.array(hi)
ax.fill_between(allr, lo, hi, color="#9ecae1", alpha=0.45, lw=0,
                label="middle 50% of all (direction, word) pairs")
ax.plot(allr, allmed, "o-", color="#08519c", lw=1.8, ms=5,
        label="median gap-shrink per unit nudge")

# per-tie gain curves
for t, col, lab in [(2, "#d62728", "word 2 tie"), (123, "#31a354", "word 123 tie"),
                    (143, "#963696", "word 143 tie")]:
    rs, gs = [], []
    for case in [ct, c]:
        for i, r in enumerate(case.radii):
            v = case.dm[i, :, t]
            neg = v[v < 0]
            if len(neg):
                rs.append(r); gs.append(np.median(np.abs(neg)) / r)
    ax.plot(rs, gs, "s--", color=col, lw=1.5, ms=5, label=lab)

# zones
ax.axvspan(3e-4, 3e-3, color="0.85", alpha=0.5, lw=0)
ax.axvspan(0.5, 2.05, color="#fdbb84", alpha=0.35, lw=0)
ax.text(2.4e-5, 1.9,
        "float32 noise floor:\nthe measured shift saturates\nat |dm| ~ 1e-5 "
        "(round-off\nof logits of size ~10-30),\nso the apparent shrink/r\n"
        "inflates as nudge -> 0.\nThresholds are meaningless here.",
        fontsize=9.6, ha="center", va="top",
        bbox=dict(fc="white", ec="0.7"))
ax.text(0.031, 2.8,
        "clean linear window (0.003 - 0.3):\nshrink is exactly proportional "
        "to\nthe nudge (flat line), so one small\nmeasurement predicts any "
        "nudge\nin this range - 99.2% exact (fig2).",
        fontsize=9.6, ha="center", va="top",
        bbox=dict(fc="white", ec="0.5"))
ax.text(1.02, 30,
        "superlinear:\nmedian 1.55x by r=2,\nties drift apart\n(word 2 doubles,\n"
        "word 123 stays flat)\n-> the first flip\nmoves to a different\nword",
        fontsize=9.6, ha="left", va="top", color="#8c4a04",
        bbox=dict(fc="#fff3e0", ec="#fdbb84"))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("nudge strength r added to the prompt (log scale)", fontsize=12)
ax.set_ylabel("gap-shrink per unit nudge  |change of decision gap| / r\n"
              "(log scale; flat = response proportional to the nudge)",
              fontsize=11.5)
ax.set_title("(a) how the decision-gap response scales with the nudge\n"
             "(26 nudge strengths, 1e-6 to 2; twin runs merged)", fontsize=12.3)
ax.legend(loc="lower left", fontsize=9.8, framealpha=0.95)
ax.grid(alpha=0.3, which="both", lw=0.5)
ax.set_ylim(1e-3, 60)

# ---------------------------------------------------------------- (b) counts
free = C.load_free("all")
w100_rows = [e for e in free if e["geom"] == "w100"]
rs = np.array([e["r"] for e in w100_rows])
obs = np.array([int((np.asarray(e["obs"]) >= 0).sum()) for e in w100_rows])
pred_sgn_all = c.pred("signed")                    # (n_r, K)
ir_of = {float(r): i for i, r in enumerate(c.radii)}
same = np.array([int((pred_sgn_all[ir_of[float(e["r"])]] >= 0).sum())
                 for e in w100_rows])

def extrap_pred(case, ir_from, r):
    rc = case.rc_signed()[ir_from]
    return np.array([int(np.argmax(rc[d] <= r)) if (rc[d] <= r).any() else -1
                     for d in range(case.K)])

ext = np.array([int((extrap_pred(c, 0, r) >= 0).sum()) for r in rs])
adv = np.array([int((np.asarray(e["pred"]) >= 0).sum()) for e in w100_rows])

ax2.plot(rs, obs, "o-", color="k", lw=2.2, ms=8, label="observed (real generations)")
ax2.plot(rs, same, "s", color="#cb181d", ms=9, mew=1.2, mfc="none",
         label="signed rule, gap-shift measured AT the nudge being predicted")
ax2.plot(rs, ext, "^--", color="#e08214", lw=1.4, ms=8,
         label="linear extrapolation of the r=0.001 threshold field")
ax2.plot(rs, adv, "D--", color="0.45", lw=1.2, ms=6.5,
         label="delivered rule, as shipped (sign-blind |shift|; the three\n"
               "small radii were even scored against the wrong teacher-forced row)")
ax2.set_xscale("log")
ax2.set_ylim(-1, 21)
ax2.set_xticks(rs)
ax2.set_xticklabels([f"{r:g}" for r in rs])
ax2.set_xlabel("nudge strength r", fontsize=12)
ax2.set_ylabel("number of the 20 directions whose text changed", fontsize=12)
ax2.set_title("(b) propulsion prompt: how many directions flip, observed vs "
              "three prediction variants", fontsize=12.3)
ax2.legend(loc="upper left", fontsize=9.8, framealpha=0.95)
ax2.grid(alpha=0.3, which="both", lw=0.5)
ax2.text(0.032, -0.55,
         "same-radius rule: exact at every nudge (red squares sit on black "
         "circles).\nLinear extrapolation gets the COUNT roughly right but the "
         "STEP wrong: at r=2 only 2\nof 16 flipping steps are exact - worst "
         "error 129 words (predicted word 14, real\nflip at word 143; others "
         "predicted word 123, real flip at word 2).\nThe sign-blind rule "
         "counts steps that cannot flip at all (over-predicts: 15 vs 8 "
         "at r=1).",
         fontsize=9.8, va="top", ha="left",
         bbox=dict(fc="#f7f7f7", ec="0.6"))

fig.suptitle("When the prediction breaks - and it is the EXTRAPOLATION that "
             "breaks, not the rule", fontsize=13.5, y=0.985)
fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.19, wspace=0.26)
fig.savefig("fig5_when_it_breaks.png", dpi=165)
print("saved fig5; ext counts:", ext.tolist(), "| obs:", obs.tolist(),
      "| same:", same.tolist(), "| adv:", adv.tolist())
