"""Fig 5: predicted vs observed first flips, all configs.

Left: class-count table (rows = window@radius configs, cols = outcome class
none/2/16/44/55/123/other). Cell shows observed count (background color) and
predicted count (text 'obs | pred'), so errors read as color/text mismatch.
Right: diagonal plot for flipping directions only (x = observed flip step,
y = predicted step), one marker per direction-config, duplicates jittered.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from common import FREE, Case, OUT

CLASSES = ["none", "2", "16", "44", "55", "123", "other"]
CFG = []          # (label, obs array, pred array)
cases = {"w100": "w100", "w200": "w200", "w500": "w500", "w100L": "w100L"}


def load_cfg():
    for label, key in [("w100@0.03", "w100"), ("w100@0.1", "w100"), ("w100@0.3", "w100"),
                       ("w200@0.1", "w200"), ("w200@0.3", "w200"),
                       ("w500@0.1", "w500"), ("w500@0.3", "w500"),
                       ("w100@0.5", "w100L"), ("w100@1.0", "w100L"), ("w100@2.0", "w100L")]:
        j = json_load(key)
        rs = j["radii"]
        r = float(label.split("@")[1])
        i = rs.index(r)
        CFG.append((label, np.array(j["obs"][i]), np.array(j["pred"][i])))


def json_load(key):
    import json
    with open(FREE[key]) as f:
        return json.load(f)


load_cfg()


def cls(v):
    if v < 0:
        return 0
    if v == 2:
        return 1
    if v == 16:
        return 2
    if v == 44:
        return 3
    if v == 55:
        return 4
    if v == 123:
        return 5
    return 6


fig = plt.figure(figsize=(14, 7.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22, figure=fig)

# ---------- table ----------
ax = fig.add_subplot(gs[0])
n_r, n_c = len(CFG), len(CLASSES)
M_obs = np.zeros((n_r, n_c))
M_pred = np.zeros((n_r, n_c))
for i, (lab, obs, pred) in enumerate(CFG):
    for v in obs:
        M_obs[i, cls(v)] += 1
    for v in pred:
        M_pred[i, cls(v)] += 1
im = ax.imshow(M_obs, cmap="YlOrRd", vmin=0, vmax=20, aspect="auto")
for i in range(n_r):
    for j in range(n_c):
        col = "white" if M_obs[i, j] > 12 else "black"
        ax.text(j, i, f"{int(M_obs[i, j])} | {int(M_pred[i, j])}", ha="center", va="center",
                fontsize=10, color=col)
ax.set_xticks(range(n_c))
ax.set_xticklabels(CLASSES, fontsize=9)
ax.set_yticks(range(n_r))
ax.set_yticklabels([f"{l}   {int((o == p).sum())}/20" for l, o, p in CFG], fontsize=9)
ax.set_xlabel("first-flip step class   (cell: observed | predicted count, of 20)")
ax.set_title("Observed vs predicted flip classes per config", fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("observed count", fontsize=8)

# ---------- diagonal plot ----------
ax = fig.add_subplot(gs[1])
rng = np.random.default_rng(0)
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(CFG)))
steps = [0, 2, 16, 24, 44, 55, 116, 123, 143, 242]
for ci, ((lab, obs, pred), col) in enumerate(zip(CFG, colors)):
    fl = np.where(obs >= 0)[0]
    for d in fl:
        jit = rng.uniform(-1.2, 1.2, 2)
        ax.plot(obs[d] + jit[0], pred[d] + jit[1], "o", ms=7, color=col, alpha=0.9,
                mec="k", mew=0.5)
    agree = int((obs == pred).sum())
    ax.annotate(f"{lab}: {agree}/20", xy=(0.99, 1.0 - 0.055 * (ci + 1)),
                xycoords="axes fraction", ha="right", fontsize=7.5, color=col)
ax.plot([-5, 300], [-5, 300], "k--", lw=0.8, label="perfect")
ax.axhline(-1, color="0.7", lw=0.6)
ax.set_xlim(-5, 260)
ax.set_ylim(-8, 260)
ax.set_xlabel("observed first-flip step (free run)")
ax.set_ylabel("predicted first-flip step (linear $|\\delta m|$ rule)")
ax.set_title("Flipping directions only: predicted vs observed step", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Predicted vs observed first flips: near-perfect for r<=0.1, borderline band "
             "0.25--0.4, systematic breakdown for r>=1", fontsize=11.5)
fig.savefig(OUT + r"\fig5_pred_vs_obs.png", dpi=160, bbox_inches="tight")
print("saved fig5")
for lab, obs, pred in CFG:
    print(f"{lab:>10}: flips obs {int((obs >= 0).sum()):>2} pred {int((pred >= 0).sum()):>2} "
          f"exact {int((obs == pred).sum()):>2}/20")
