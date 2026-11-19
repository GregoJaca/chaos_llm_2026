"""Window boundary: the immediate response collapses when plotted vs W-P."""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\big\scan"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

rows = []
for d in sorted(glob.glob(os.path.join(B, "*_w*"))):
    f = os.path.join(d, "a1.npz")
    if not os.path.exists(f):
        continue
    name = os.path.basename(d); base, W = name.split("_w"); W = int(W)
    if base == "scanB_1":
        continue
    z = np.load(f, allow_pickle=True)
    L = list(z["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
    P = int(z["prompt_ids"].shape[0])
    a = np.median(z["dh_norm"][ri, :, li, :], axis=0)
    rows.append(dict(prompt=base, P=P, W=W, a0=float(a[0]), a10=float(a[10]), a50=float(a[50])))

fig, ax = plt.subplots(figsize=(8.8, 4.8))
for p in sorted(set(r["prompt"] for r in rows)):
    sel = [r for r in rows if r["prompt"] == p]
    sel.sort(key=lambda r: r["W"])
    P = sel[0]["P"]
    sat = np.mean([r["a0"] for r in sel if r["W"] - P >= 20])
    ax.plot([r["W"] - P for r in sel], [r["a0"] / sat for r in sel], "o-", label=f"{p} ($P={P}$)")
ax.axvline(0, color="k", ls="--", lw=1)
ax.axhline(1, color="k", lw=0.6)
ax.set_xlabel(r"$W-P$ (negative: prompt is clipped)")
ax.set_ylabel(r"$a(0,27)$ normalised by its value for $W\gg P$")
ax.set_title("The window acts as a step at $W=P$:\nsame collapse for three prompts")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "window_boundary.png"), dpi=190)
print("saved")
for p in sorted(set(r["prompt"] for r in rows)):
    sel = sorted([r for r in rows if r["prompt"] == p], key=lambda r: r["W"])
    print(p, [(r["W"] - r["P"], round(r["a0"], 3)) for r in sel])
