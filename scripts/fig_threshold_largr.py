"""Predicted (signed margin rule, teacher-forced) vs observed (autoregressive) flip times at large r."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 11,
                     "legend.fontsize": 9})

RADII = [0.5, 0.8, 1.2, 2.0]
COLS = {0.5: "tab:blue", 0.8: "tab:orange", 1.2: "tab:green", 2.0: "tab:red"}

fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
for ax, p in zip(axs, ["p1", "p2"]):
    d = np.load(os.path.join(B, "bigr", "deepseek", p, "a1.npz"))
    m = d["margins"].astype(float); dm = d["dm"].astype(float); radii = d["radii"]
    for r in RADII:
        ri = int(np.argmin(np.abs(radii - r)))
        z = np.load(os.path.join(B, "ar", f"ar_{p}_r{r}", "free_run.npz"), allow_pickle=True)
        obs = z["obs"][0]
        pred = []
        for di in range(dm.shape[1]):
            hit = np.where(m + dm[ri, di] < 0)[0]
            pred.append(int(hit[0]) if len(hit) else -1)
        pred = np.array(pred)
        ok = (pred >= 0) & (obs >= 0)
        ax.scatter(pred[ok], obs[ok], s=38, color=COLS[r], alpha=0.8,
                   label=f"$r={r}$ (exact {int((pred==obs).sum())}/20)")
        no = (~ok)
        ax.scatter(np.zeros(no.sum()) + 3, np.zeros(no.sum()) + 3, marker="x", s=20,
                   color=COLS[r], alpha=0.5)
    ax.plot([0, 300], [0, 300], "k--", lw=1.0)
    ax.set_xlabel(r"predicted first flip $t$ (signed rule, teacher-forced)")
    ax.set_ylabel(r"observed first flip $t$ (autoregressive)")
    ax.set_title(f"prompt {p[-1]}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("The threshold prediction is exact at small $r$ and breaks at $r\\geq0.5$", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "threshold_largr.png"), dpi=190, bbox_inches="tight")
print("saved")
