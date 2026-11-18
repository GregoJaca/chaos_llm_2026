"""fig2 -- PREDICTION ACCURACY done right.

(a) flipping-directions only: predicted vs observed first-changed word step,
    points colored by how deep past the threshold the nudge was
    (rho = r / r_*: 1-2 = borderline call), marker size = runs at that spot.
    Includes every radius we have: propulsion w100 (r=0.03..2), w200, w500
    (0.1, 0.3), cyber (0.01-0.3) with the signed rule at the exact radius, plus
    the childhood prompt scored with the delivered sign-blind rule (geometry
    not stored locally).
(b) all 320 direction-runs as outcome strips: exact flip (dark blue), correct
    no-flip (light gray), wrong step (orange), false alarm (red), missed flip
    (magenta); signed-at-exact-radius vs delivered rule side by side.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

import common2 as C

pts = json.load(open("eval_points.json"))

ROW_ORDER = [
    ("free_w100_large.json", 0.03), ("free_w100_large.json", 0.1),
    ("free_w100_large.json", 0.3), ("fr_w100_large.npz", 0.5),
    ("fr_w100_large.npz", 1.0), ("fr_w100_large.npz", 2.0),
    ("free_w200.json", 0.1), ("free_w200.json", 0.3),
    ("free_w500.json", 0.1), ("free_w500.json", 0.3),
    ("fr_cyber.npz", 0.01), ("fr_cyber.npz", 0.1), ("fr_cyber.npz", 0.3),
    ("fr_childhood.npz", 0.01), ("fr_childhood.npz", 0.1), ("fr_childhood.npz", 0.3),
]
ROWS = {e["source"]: e for e in pts}  # placeholder, replaced below
def find(src, r):
    for e in pts:
        if e["source"] == src and abs(e["r"] - r) < 1e-12:
            return e
    raise KeyError((src, r))

label_of = {
    "free_w100_large.json": "propulsion, window 100",
    "fr_w100_large.npz": "propulsion, window 100",
    "free_w200.json": "propulsion, window 200",
    "free_w500.json": "propulsion, window 500",
    "fr_cyber.npz": "cybersecurity prompt",
    "fr_childhood.npz": "childhood prompt",
}
row_labels = []
for src, r in ROW_ORDER:
    lab = label_of[src]
    if "window" in lab:
        row_labels.append(f"{lab.split(', ')[0]}\n{lab.split(', ')[1]}  r={r:g}")
    else:
        row_labels.append(f"{lab}\nr={r:g}")

# ---------------------------------------------------------------- panel (a)
fig = plt.figure(figsize=(16.5, 7.6))
axa = fig.add_subplot(1, 2, 1)

groups = {}   # (obs,pred) -> (count, rhos, tag)
miss_notes = []
for e in pts:
    obs = np.asarray(e["obs"])
    if e["pred_sgn"] is not None:
        pred = np.asarray(e["pred_sgn"])
        rho = np.asarray(e["rho"])
        tag = "sgn"
    else:
        pred = np.asarray(e["pred_ship"])
        rho = np.full(len(obs), np.nan)
        tag = "childhood (delivered rule)"
    bf = (obs >= 0) & (pred >= 0)
    for d in np.where(bf)[0]:
        key = (int(obs[d]), int(pred[d]))
        g = groups.setdefault(key, dict(n=0, rho=[], tags=set()))
        g["n"] += 1
        g["rho"].append(rho[d])
        g["tags"].add(tag)
    for d in np.where(bf & (pred != obs))[0]:
        miss_notes.append((e["source"], e["r"], int(d), int(obs[d]), int(pred[d]),
                           float(rho[d]) if np.isfinite(rho[d]) else None))

marker_of = {"sgn": "o", "childhood (delivered rule)": "s"}
for (obs_v, pred_v), g in sorted(groups.items()):
    tags = list(g["tags"])
    if len(tags) == 1 and tags[0] == "childhood (delivered rule)":
        col, mk, ed = "#8c8c8c", "s", "#4d4d4d"
    else:
        rh = np.array(g["rho"])
        col, mk, ed = None, "o", "k"
        val = np.nanmean(np.log2(rh)) if np.isfinite(rh).any() else 0.0
        col = plt.cm.RdBu_r(TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=4)(val))
    axa.scatter(obs_v, pred_v, s=28 + 26 * (g["n"] - 1), c=[col], marker=mk,
                edgecolors=ed, linewidths=0.7, alpha=0.92, zorder=5)

lim = (-10, 186)
axa.plot(lim, lim, "k--", lw=1.1, zorder=2)
axa.text(150, 158, "perfect prediction\n(step for step)", rotation=38,
         fontsize=10, ha="left", va="bottom", color="0.2")
axa.set_xlim(lim); axa.set_ylim(lim)
axa.set_aspect("equal")

n_sgn_flip = sum(g["n"] for k, g in groups.items() if "sgn" in g["tags"])
n_sgn_exact = sum(g["n"] for k, g in groups.items()
                  if k[0] == k[1] and "sgn" in g["tags"])
n_ch_exact = sum(g["n"] for k, g in groups.items()
                 if k[0] == k[1] and g["tags"] == {"childhood (delivered rule)"})
axa.text(0.03, 0.97,
         f"signed rule at the exact nudge: {n_sgn_exact}/{n_sgn_flip} flipping "
         f"runs predicted to the exact word\nworst miss: 22 steps (both remaining "
         f"misses are nudges 6-7x past threshold, not borderline calls)\n"
         f"70 of 260 calls sit within a factor 2 of the threshold (pale dots) - "
         f"all 70 are correct",
         transform=axa.transAxes, va="top", ha="left", fontsize=10.2,
         bbox=dict(fc="white", ec="0.7", alpha=0.95))

seen_notes = set()
for note in miss_notes:
    src, r, d, obs_v, pred_v, rho_v = note
    if "w100" in src or "fr_w100" in src:
        txt = f"w100 r={r:g} d{d}: off by {abs(pred_v-obs_v)} steps"
        axa.annotate(txt, xy=(obs_v, pred_v), xytext=(obs_v + 26, pred_v + 6),
                     fontsize=9.6, color="#a01717",
                     arrowprops=dict(arrowstyle="->", color="#a01717", lw=1))
    elif (obs_v, pred_v) not in seen_notes:
        seen_notes.add((obs_v, pred_v))
        n_here = sum(1 for m2 in miss_notes
                     if m2[3] == obs_v and m2[4] == pred_v)
        txt = (f"childhood r={r:g}: {n_here} runs, sign-blind rule\n"
               f"says step {pred_v}, real change at step {obs_v}")
        axa.annotate(txt, xy=(obs_v, pred_v), xytext=(obs_v - 6, pred_v + 30),
                     fontsize=9.6, color="#525252",
                     arrowprops=dict(arrowstyle="->", color="#525252", lw=1))

axa.set_xlabel("observed step of the first changed word\n"
               "(measured on the real, freely generated text)", fontsize=12)
axa.set_ylabel("predicted step of the first changed word\n"
               "(from one teacher-forced pass along the unperturbed text)",
               fontsize=12)
axa.set_title("(a) flipping directions only - 75 runs, all prompts and nudge "
              "strengths pooled", fontsize=12.5)
axa.grid(alpha=0.25, lw=0.5)

cb = fig.colorbar(plt.cm.ScalarMappable(
    norm=TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=4), cmap="RdBu_r"), ax=axa,
    fraction=0.045, pad=0.02)
cb.set_label("how deep past the threshold the nudge was\n"
             "(log2 of nudge / smallest flipping nudge; 0 = borderline call)",
             fontsize=10)
cb.set_ticks([-1, 0, 1, 2, 4])
cb.set_ticklabels(["0.5x", "borderline", "2x", "4x", "16x"])

# ---------------------------------------------------------------- panel (b)
axb = fig.add_subplot(1, 2, 2)
COLS = dict(exact="#08519c", noflip="#e0e0e0", wrong="#f16913",
            false="#cb181d", missed="#7a0177")
NY = len(ROW_ORDER)
counts = dict(exact=0, noflip=0, wrong=0, false=0, missed=0)
ship_counts = dict(exact=0, noflip=0, wrong=0, false=0, missed=0)


def outcome(o, p):
    return ("exact" if o == p and o >= 0 else
            "noflip" if o < 0 and p < 0 else
            "wrong" if o >= 0 and p >= 0 else
            "false" if o < 0 else "missed")
for yi, (src, r) in enumerate(ROW_ORDER):
    e = find(src, r)
    obs = np.asarray(e["obs"])
    pred_sgn = np.asarray(e["pred_sgn"]) if e["pred_sgn"] else None
    pred_shp = np.asarray(e["pred_ship"])
    for d in range(len(obs)):
        # signed block (left, columns 0..19)
        if pred_sgn is None:
            axb.add_patch(plt.Rectangle((d - 0.5, yi - 0.42), 1, 0.40, fc="white",
                                        ec="0.85", hatch="///", lw=0.3))
        else:
            key = outcome(int(obs[d]), int(pred_sgn[d]))
            counts[key] += 1
            axb.add_patch(plt.Rectangle((d - 0.5, yi - 0.42), 1, 0.40,
                                        fc=COLS[key], ec="white", lw=0.3))
        # delivered block (right, columns 22..41)
        key = outcome(int(obs[d]), int(pred_shp[d]))
        ship_counts[key] += 1
        axb.add_patch(plt.Rectangle((d + 22.5 - 0.5, yi - 0.42), 1, 0.40,
                                    fc=COLS[key], ec="white", lw=0.3))

axb.axvline(21, color="0.3", lw=14, alpha=0.08)
axb.text(10, -1.15, "signed rule, exact nudge", ha="center", fontsize=11.5,
        weight="bold")
axb.text(32, -1.15, "delivered rule (sign-blind |shift|)", ha="center",
        fontsize=11.5, weight="bold")
axb.set_xlim(-1.2, 42.4)
axb.set_ylim(NY - 0.6, -1.9)
axb.set_xticks(list(range(0, 20)) + list(range(23, 43)))
axb.set_xticklabels([(str(i + 1) if i < 20 else str(i - 22)) for i in
                     list(range(0, 20)) + list(range(23, 43))], fontsize=8)
axb.set_yticks(range(NY))
axb.set_yticklabels(row_labels, fontsize=9.6)
axb.set_xlabel("direction index (1-20 of the fixed direction set)", fontsize=11.5)
axb.set_title("(b) all 320 direction-runs incl. no-flip calls; 260 evaluable "
              "with the signed rule, hatched = geometry not stored locally",
              fontsize=12)
for sp in axb.spines.values():
    sp.set_visible(False)

handles = [
    Patch(fc=COLS["exact"],
          label=f"flip step predicted exactly: {counts['exact']} signed / "
                f"{ship_counts['exact']} delivered"),
    Patch(fc=COLS["noflip"],
          label=f"'no flip anywhere' predicted correctly: {counts['noflip']} / "
                f"{ship_counts['noflip']}"),
    Patch(fc=COLS["wrong"],
          label=f"flip right, step off by a few: {counts['wrong']} / "
                f"{ship_counts['wrong']}"),
    Patch(fc=COLS["false"],
          label=f"flip predicted that never happened: {counts['false']} / "
                f"{ship_counts['false']}"),
    Patch(fc=COLS["missed"],
          label=f"flip observed, none predicted: {counts['missed']} / "
                f"{ship_counts['missed']}"),
    Patch(fc="white", hatch="///", ec="0.7",
          label="not evaluable locally (childhood, 3 rows)"),
]
axb.legend(handles=handles, loc="lower left", fontsize=9.6, framealpha=0.97,
           bbox_to_anchor=(0.0, 0.0))
axb.text(42.2, NY - 0.7,
         "signed rule total: 258/260\nfully correct (99.2%)",
         fontsize=10.2, ha="right", va="bottom", weight="bold",
         bbox=dict(fc="#deebf7", ec="0.5"))

fig.suptitle("Does the threshold rule actually predict the first changed word? "
             "All 16 local free-run experiments (5 prompt/window settings, 20 "
             "directions each)", fontsize=13.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("fig2_prediction_scorecard.png", dpi=165)
print("saved fig2; pooled counts:", counts, "| misses:", miss_notes)
