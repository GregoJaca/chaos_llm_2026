"""Where a single-position perturbation acts: position sensitivity and flip times."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

m = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)["margins"]
runs = [("r=0.1", os.path.join(B, "state_freerun", "sf.npz"), "tab:blue"),
        ("r=0.1 (other positions)", os.path.join(B, "state_freerun", "sf01b.npz"), "tab:blue"),
        ("r=0.3", os.path.join(B, "state_freerun", "sf03.npz"), "tab:red")]

rows = []
for label, path, col in runs:
    d = np.load(path, allow_pickle=True)
    ref = d["ref"]
    for si, s in enumerate(d["steps"]):
        flips = []
        for di in range(d["pred"].shape[1]):
            p = d["pred"][si, di]; v = p >= 0
            mm = np.where((p != ref) & v)[0]
            if len(mm):
                flips.append(int(mm[0]))
        rows.append((label, int(s), len(flips), flips, col))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.25]})

ax = axs[0]
ss = sorted(set(r[1] for r in rows))
for label, col in [("r=0.1", "tab:blue"), ("r=0.3", "tab:red")]:
    xs, ys = [], []
    for lab, s, n, fl, c in rows:
        if lab.startswith(label):
            xs.append(s); ys.append(n / 24)
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color=col, label=label)
ax.set_xlabel(r"perturbation position $s$ (generated token index)")
ax.set_ylabel("fraction of the 24 directions\nthat change any later token")
ax.set_title("(a) Position, not size, decides whether the text changes")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.set_ylim(-0.02, 0.8)

ax = axs[1]
for lab, s, n, fl, col in rows:
    if not fl:
        continue
    ax.scatter([s] * len(fl), fl, s=45, color=col, alpha=0.8)
for t in [40, 116, 123, 143]:
    ax.axhline(t, color="0.6", lw=0.7, ls=":")
    ax.text(150, t + 1.2, f"$t={t}$, $m_t={m[t]:.4f}$", fontsize=7.5, color="0.35")
ax.set_xlabel(r"perturbation position $s$")
ax.set_ylabel(r"token $t$ where the first change happens")
ax.set_title("(b) When it acts, it acts at one of the close calls\n(40 = early, response still large; 116/123/143 = smallest gaps)")
ax.set_xlim(0, 205)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_position_map.png"), dpi=190)
print("saved")
for r in rows:
    print(r[0], "s=", r[1], "flips", r[2], "at", sorted(set(r[3])))
