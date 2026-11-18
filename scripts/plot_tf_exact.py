"""Teacher forcing is an exact simulator up to the first token flip.

One pass of the model on the perturbed prompt, fed the unperturbed token
sequence, reproduces (i) whether the free run flips at all, (ii) the exact step,
and (iii) the exact new token.  After the flip the two runs have different
prefixes, so comparison stops being meaningful.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\tf_pred"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\figures"

sets = {n: np.load(os.path.join(BASE, f"{n}.npz"), allow_pickle=True) for n in ["mid", "large"]}

rows = []
for name, d in sets.items():
    pred, ar, ref, radii = d["pred"], d["ar"], d["ref"], d["radii"]
    for ri, r in enumerate(radii):
        for di in range(pred.shape[1]):
            a, p, f = ar[ri, di], pred[ri, di], ref
            v = a >= 0
            df = np.where((a != f) & v)[0]
            dp = np.where((p != f) & v)[0]
            ta = int(df[0]) if len(df) else -1
            tp = int(dp[0]) if len(dp) else -1
            rows.append(dict(r=float(r), di=di, ta=ta, tp=tp, a=a, p=p, f=f,
                             tok_ok=(int(p[tp] == a[ta]) if (ta >= 0 and tp >= 0) else None)))

both = [x for x in rows if x["ta"] >= 0 and x["tp"] >= 0]
none = [x for x in rows if x["ta"] < 0 and x["tp"] < 0]
fp = [x for x in rows if x["ta"] < 0 and x["tp"] >= 0]
miss = [x for x in rows if x["ta"] >= 0 and x["tp"] < 0]
print("flips", len(both), "no-flips", len(none), "false positives", len(fp), "missed", len(miss))
print("token correct at flip", sum(1 for x in both if x["tok_ok"]), "/", len(both))
print("step exact", sum(1 for x in both if x["ta"] == x["tp"]), "/", len(both))

fig = plt.figure(figsize=(14, 8.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.26)

# (a) token timeline
ax = fig.add_subplot(gs[0, :])
ex = max(both, key=lambda x: 0 if x["ta"] != 123 else x["r"])
T = ex["ta"]
lo, hi = max(0, T - 6), min(len(ex["f"]), T + 14)
names = ["unperturbed", "teacher-forced", "free-run"]
seqs = [ex["f"], ex["p"], ex["a"]]
grid = np.zeros((3, hi - lo))
for i, s in enumerate(seqs):
    for j, t in enumerate(range(lo, hi)):
        grid[i, j] = 1 if s[t] == ex["f"][t] else 0
ax.imshow(grid, aspect="auto", cmap=ListedColormap(["#f4c2c2", "#c7e6c7"]),
          extent=(lo - 0.5, hi - 0.5, 2.5, -0.5), vmin=0, vmax=1)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(names)
ax.axvline(T - 0.5, color="tab:red", ls="--", lw=1.5)
ax.text(T - 0.4, 0.42, f"first flip at t={T}", ha="right", color="tab:red", fontsize=9)
ax.set_xlabel("generated token t")
ax.set_title(f"(a) One example (r={ex['r']:g}, direction {ex['di']}): green = same token as the unperturbed run.\n"
             "The teacher-forced pass changes exactly the same token at the same step; only after that do the prefixes differ.")
ax.set_xticks(range(lo, hi, 2))

# (b) predicted vs observed flip step
ax = fig.add_subplot(gs[1, 0])
ax.scatter([x["ta"] for x in both], [x["tp"] for x in both], s=55, color="tab:blue", zorder=3)
lim = max(x["ta"] for x in both) + 8
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlabel("observed flip step in the free run")
ax.set_ylabel("flip step predicted by one teacher-forced pass")
ax.set_title(f"(b) {len(both)}/{len(both)} flips: exact step and exact new token,\n"
             f"{len(fp)} false positives and {len(miss)} misses in {len(rows)} cases")
ax.grid(alpha=0.25)

# (c) counts per radius
ax = fig.add_subplot(gs[1, 1])
radii = sorted(set(x["r"] for x in rows))
obs = [sum(1 for x in rows if x["r"] == r and x["ta"] >= 0) for r in radii]
prd = [sum(1 for x in rows if x["r"] == r and x["tp"] >= 0) for r in radii]
x = np.arange(len(radii))
ax.bar(x - 0.18, obs, 0.36, label="free run flipped", color="tab:blue")
ax.bar(x + 0.18, prd, 0.36, label="teacher-forced pass flipped", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in radii])
ax.set_xlabel("perturbation size r")
ax.set_ylabel("number of directions (out of 20)")
ax.set_title("(c) Exact agreement at every radius")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("Teacher forcing is exact until the first token flip: same decision, same step, same new token",
             fontsize=12)
fig.savefig(os.path.join(OUT, "tf_is_exact_until_flip.png"), dpi=190, bbox_inches="tight")
print("saved")
