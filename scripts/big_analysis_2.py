"""Analyses of the big batch: state-window grid and multi-position layer probes."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

# ---------------- state window grid ----------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
wins, imm, half = [], [], []
for p in sorted(glob.glob(os.path.join(B, "state", "w*", "state2.npz"))):
    w = int(p.split("w")[-1].split(os.sep)[0])
    d = np.load(p, allow_pickle=True)
    steps = list(d["steps"]); radii = d["radii"]; layers = list(d["layers"])
    ri = int(np.argmin(np.abs(radii - 0.1))); li = layers.index(27)
    si = steps.index(16) if 16 in steps else 1
    a = np.median(d["dh"][ri, si, :, li, :], axis=0)
    s = steps[si]
    seg = a[s:]
    below = np.where(seg < 0.5 * seg[0])[0]
    wins.append(w); imm.append(float(a[s])); half.append(int(below[0]) if len(below) else -1)

ax = axs[0]
ax.semilogx(wins, imm, "o-", color="tab:blue")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"$a(s,27)$ at the perturbation ($s=16$)")
ax.set_title("(a) Immediate response vs window")
ax.grid(alpha=0.25, which="both")

ax = axs[1]
ax.semilogx(wins, half, "o-", color="tab:red")
ax.set_xlabel("window $W$")
ax.set_ylabel("half-life (tokens)")
ax.set_title("(b) Half-life vs window")
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "state_window_grid.png"), dpi=190)
print("state", list(zip(wins, [round(x, 1) for x in imm], half)))

# ---------------- multi-position probes ----------------
fig, ax = plt.subplots(figsize=(8.8, 4.6))
pos_list, gains, errs = [], [], []
for p in sorted(glob.glob(os.path.join(B, "probe", "p*.npz")), key=lambda x: int(os.path.basename(x)[1:-4])):
    d = np.load(p)
    pos = int(d["pos"]); inj = list(d["layers"]); r = float(d["r"]); dh = d["dh"]
    vals = []
    for i, L in enumerate(inj):
        if L + 1 >= 28:
            continue
        vals.append(float(np.median(dh[i, :, 27, pos]) / r))
    pos_list.append(pos)
    gains.append(np.mean(vals)); errs.append(np.std(vals))
ax.errorbar(pos_list, gains, yerr=errs, fmt="o-", capsize=3)
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel(r"perturbation position $s$")
ax.set_ylabel(r"$a(27)/r$ (mean over injection layers)")
ax.set_title("Only the near-tie position amplifies the perturbation")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_probe_positions_all.png"), dpi=190)
print("probe", list(zip(pos_list, [round(g, 2) for g in gains])))
