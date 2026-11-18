"""Layer-resolved amplification across positions: statistics."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

files = {"pos16": os.path.join(B, "layer_probe", "pos16.npz"),
         "pos50": os.path.join(B, "layer_probe", "pos50.npz")}
for p in [32, 75, 100, 150]:
    f = os.path.join(B, "layer_probe_multi", f"p{p}.npz")
    if os.path.exists(f):
        files[f"pos{p}"] = f

# gather the total gain at the last layer, per injection layer and position
rows = []
for name, path in files.items():
    d = np.load(path)
    dh = d["dh"]; inj = list(d["layers"]); pos = int(d["pos"]); r = float(d["r"])
    for i, L in enumerate(inj):
        if L + 1 >= 28:
            continue
        vals = np.median(dh[i, :, 27, pos]) / r
        rows.append((pos, L, float(vals), name))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axs[0]
for pos in sorted(set(r[0] for r in rows)):
    sel = [r for r in rows if r[0] == pos]
    sel.sort(key=lambda r: r[1])
    ax.semilogx([r[1] + 1 for r in sel], [r[2] for r in sel], "o-", ms=4,
                label=f"$s={pos}$")
ax.axhline(1, color="k", lw=0.8)
ax.set_xlabel(r"injection layer $L+1$")
ax.set_ylabel(r"$a(27)/r$ at the injected position")
ax.set_title("(a) Total gain to the last layer vs injection layer")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25, which="both")

ax = axs[1]
pos_unique = sorted(set(r[0] for r in rows))
means, mins, maxs = [], [], []
for pos in pos_unique:
    sel = [r[2] for r in rows if r[0] == pos]
    means.append(np.mean(sel)); mins.append(np.min(sel)); maxs.append(np.max(sel))
x = np.arange(len(pos_unique))
ax.errorbar(x, means, yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            fmt="o", capsize=4, color="tab:blue")
ax.axhline(1, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"{p}" for p in pos_unique])
ax.set_xlabel(r"perturbation position $s$")
ax.set_ylabel(r"mean $a(27)/r$ over injection layers")
ax.set_title("(b) Amplification depends on position, not on where in depth you inject")
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "layer_probe_positions.png"), dpi=190)
print("saved")
for pos in pos_unique:
    sel = [r[2] for r in rows if r[0] == pos]
    print(f"s={pos}: gain range {min(sel):.2f}-{max(sel):.2f}, mean {np.mean(sel):.2f}")
