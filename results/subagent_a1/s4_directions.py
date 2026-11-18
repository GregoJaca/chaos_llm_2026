"""A1 direction dependence (w100): 20 near-orthogonal simplex directions.

Q: are there special directions; does the direction change only the gain?

Outputs:
  fig6_directions.png
  direction_stats.csv
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

D = C.load("w100")
dh, radii, layers = D["dh"], D["radii"], D["layers"]
safe_idx = np.where(D["safe"])[0]
small_idx = np.where(D["small"])[0]
LG = [li for li, l in enumerate(layers) if l > 0]
ll = [int(layers[li]) for li in LG]
li27 = LG[-1]
T = 300

# per-direction gain at every layer, and per-direction universal curve at L27
A = np.zeros((len(LG), 20))
for i, li in enumerate(LG):
    for idir in range(20):
        A[i, idir] = np.median(dh[safe_idx, idir, li, 0] / radii[safe_idx])
Gdir = np.zeros((20, T))
for idir in range(20):
    Gdir[idir] = np.array([np.median(dh[safe_idx, idir, li27, t] / radii[safe_idx])
                           for t in range(T)])

# pairwise Pearson correlation of log-response curves (r=0.1, pre-flush)
ir10 = int(np.where(np.isclose(radii, 0.1))[0][0])
X = np.log(dh[ir10, :, li27, :85])
Xc = X - X.mean(axis=1, keepdims=True)
corr = (Xc @ Xc.T) / np.outer(np.sqrt((Xc ** 2).sum(1)), np.sqrt((Xc ** 2).sum(1)))
off = corr[~np.eye(20, dtype=bool)]
print(f"pairwise corr of dir curves (r=0.1, t<85): median={np.median(off):.3f}, "
      f"min={off.min():.3f}")

# full-horizon correlation (envelope) and eigen-spectrum across directions
X250 = np.log(dh[ir10, :, li27, :250])
Xc = X250 - X250.mean(axis=1, keepdims=True)
corr250 = (Xc @ Xc.T) / np.outer(np.sqrt((Xc ** 2).sum(1)), np.sqrt((Xc ** 2).sum(1)))
off250 = corr250[~np.eye(20, dtype=bool)]
Mvar = (np.log(Gdir[:, :85]) - np.log(Gdir[:, :85]).mean(0, keepdims=True))
ev = np.linalg.eigvalsh(Mvar.T @ Mvar / 85)[::-1]
ev_share = ev / ev.sum()
print(f"pairwise corr over full horizon t<250: median={np.median(off250):.3f}")
print("eigen-share of direction covariance (log space):",
      np.round(ev_share[:4], 3))

# shape identity across directions (normalize each dir by its own t=0 gain)
S = Gdir / Gdir[:, :1]
dev = S.std(axis=0) / S.mean(axis=0)
print(f"shape spread across dirs after normalizing by own gain: "
      f"median CV={np.median(dev[:85])*100:.1f}% (t<85), "
      f"max={dev[:85].max()*100:.1f}%")

# simplex sanity: near-orthogonal directions
simp = D["simplex"]
print("simplex diag:", np.round(np.diag(simp)[:4], 3),
      " off-diag:", np.round(simp[0, 1:4], 3), "(~ -1/19: regular simplex)")
print("gain at L27: min/med/max =", A[-1].min(), np.median(A[-1]), A[-1].max(),
      " argmax:", A[-1].argmax(), " ids:", D["subspace"][A[-1].argmax()],
      D["subspace"][np.argsort(A[-1])[-2]])
print("gain ratio max/min per layer:",
      dict(zip(ll, np.round(A.max(1) / A.min(1), 1).tolist())))
print("top-3 dirs at L27:", np.argsort(A[-1])[-3:].tolist(),
      "bottom-3:", np.argsort(A[-1])[:3].tolist())

with open(os.path.join(C.OUT, "direction_stats.csv"), "w") as f:
    f.write("layer," + ",".join(f"gain_dir{j}" for j in range(20)) + "\n")
    for i, l in enumerate(ll):
        f.write(f"{l}," + ",".join(f"{v:.4f}" for v in A[i]) + "\n")

# ---------------------------------------------------------------- fig 6
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))

# (a) gain spectrum at L27
ax = axes[0, 0]
order = np.argsort(A[-1])
colors = ["crimson" if j in (10, 11, 1) else "0.55" for j in order]
ax.bar(range(20), A[-1][order], color=colors)
ax.axhline(np.median(A[-1]), color="k", ls="--", lw=1,
           label=f"median = {np.median(A[-1]):.2f}")
ax.axhline(A[-1].min(), color="b", ls=":", lw=1,
           label=f"min = {A[-1].min():.2f}")
for j in (10, 11, 1):
    k = int(np.where(order == j)[0][0])
    ax.annotate(f"dir {j}", xy=(k, A[-1][j] + 0.3), ha="center", fontsize=9)
ax.set_xlabel("direction (sorted)"); ax.set_ylabel("gain at L27, t=0: $\\delta h/r$")
ax.set_title("(a) directions differ by a $\\times$16 scalar gain:\nmax at dir 10 "
             "$\\approx$ 10x median (dirs 1, 11 next)", loc="left")
ax.legend()

# (b) pairwise correlation matrix
ax = axes[0, 1]
im = ax.imshow(corr, cmap="viridis", vmin=-0.3, vmax=1.0)
ax.set_xlabel("direction"); ax.set_ylabel("direction")
ax.set_title(f"(b) token-level shapes decorrelate:\nmedian r = "
             f"{np.median(off):.2f} (t<85), but r = {np.median(off250):.2f} "
             "over t<250", loc="left")
fig.colorbar(im, ax=ax, pad=0.015, label="Pearson r of log$\\,\\delta h(t)$")

# (c) gain vs layer for weak / typical / strong + median band
ax = axes[1, 0]
for idir, lab in [(6, "dir 6 (weakest)"), (0, "dir 0 (typical)"), (10, "dir 10 (strongest)")]:
    ax.plot(ll, A[:, idir], "o-", label=lab)
ax.fill_between(ll, np.percentile(A, 25, axis=1), np.percentile(A, 75, axis=1),
                color="0.8", label="IQR over 20 dirs")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(ll); ax.set_xticklabels([str(x) for x in ll])
ax.set_xlabel("layer"); ax.set_ylabel("gain $\\delta h(t{=}0,l)/r$")
ax.set_title("(c) anisotropy exists from layer 1 and the\nmax/min ratio is "
             "depth-stable ($\\times$12-16)", loc="left")
ax.legend(fontsize=8.5)

# (d) shape after own-gain normalization: 3 dirs + aggregate
ax = axes[1, 1]
for idir, col in [(0, "tab:blue"), (6, "tab:green"), (10, "crimson")]:
    ax.plot(S[idir], lw=1.1, color=col, alpha=0.8, label=f"dir {idir}")
ax.plot(np.median(S, axis=0), "k-", lw=2.0, label="median of 20")
ax.set_yscale("log")
ax.set_xlabel("generated token t")
ax.set_ylabel("$G_j(t)\\,/\\,G_j(0)$ (each dir by its own gain)")
ax.set_title(f"(d) after own-gain normalization: shared envelope\n"
             f"({ev_share[0]*100:.0f}% of log-variance) + dir-specific\n"
             f"oscillations (CV {np.median(dev[:85])*100:.0f}%)", loc="left")
ax.legend(fontsize=8.5, loc="lower left")
ax.set_ylim(3e-3, 8)

fig.suptitle("A1 direction dependence (w100): near-orthogonal simplex directions, "
             "yet gains span $\\times$16;\nto first order direction = scalar gain "
             "on a shared envelope; token-level oscillations are direction-specific "
             "(2nd mode 8%)", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
C.save(fig, "fig6_directions.png")
