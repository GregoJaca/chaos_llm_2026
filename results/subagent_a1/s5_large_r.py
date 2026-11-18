"""A1 large-r regime (w100): r >= 0.5. Saturation? Regime change? Flips?

Outputs:
  fig7_large_r.png
  large_r_stats.txt
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

D = C.load("w100")
dh, dhm, radii, layers = D["dh"], D["dhm"], D["radii"], D["layers"]
G = D["G"]
li27 = int(np.where(layers == 27)[0][0])
large_idx = np.where(D["large"])[0]
small_idx = np.where(D["small"])[0]
T = 300
lr = np.log(radii[large_idx])

# shape stability at large r: dh(t)/dh(t=0) per radius vs linear-regime shape
shapes = {}
for ir in [4, 7, 9, 11]:  # r = 0.1, 0.5, 1, 2
    shapes[radii[ir]] = dhm[ir, li27, :] / dhm[ir, li27, 0]
cv_shape = []
for t in range(85):
    v = np.array([shapes[r][t] for r in shapes])
    cv_shape.append(v.std() / v.mean())
print("shape CV across r in {0.1,0.5,1,2} on t<85: median =",
      f"{np.median(cv_shape)*100:.1f}%")

# NOTE: dh[large_idx, j, li27, 0] is already a vector over the 5 large radii
alphas_dir = np.array([np.polyfit(lr, np.log(dh[large_idx, j, li27, 0]), 1)[0]
                       for j in range(20)])
print("per-dir large-r exponent at L27, t=0: min/med/max =",
      round(alphas_dir.min(), 2), round(np.median(alphas_dir), 2),
      round(alphas_dir.max(), 2))
gains_dir = np.array([np.median(dh[np.where(D["safe"])[0], j, li27, 0] /
                                radii[np.where(D["safe"])[0]]) for j in range(20)])
r_gain_alpha = np.corrcoef(np.log(gains_dir), alphas_dir)[0, 1]
print("corr(linear gain, large-r exponent) across dirs:", round(r_gain_alpha, 2))

# where is the r=2 response biggest, and how far does the state rotate?
cos = D["d"]["cos"]
i2 = int(np.where(np.isclose(radii, 2.0))[0][0])
j3, t3 = np.unravel_index(dh[i2, :, li27, :].argmax(), dh[i2, :, li27, :].shape)
print(f"max dh at r=2: {dh[i2].max():.1f} at L27, dir {j3}, t={t3} (the surge); "
      f"min cos at r=2, L27: {cos[i2, :, li27, :].min():.3f}")

# prediction flips
pT = D["pred_T"]
m = D["margins"]
flip_t = np.unique(pT[pT >= 0])
flip_margin = {int(t): float(m[t]) for t in flip_t}
order = np.argsort(m)
lowest = order[:5]
frac_flip = [np.mean(pT[ir] >= 0) for ir in range(len(radii))]
print("flip fraction by radius:", dict(zip(radii.tolist(),
                                            np.round(frac_flip, 2).tolist())))
print("flip times and their base-run margins:",
      {int(t): round(float(m[t]), 3) for t in flip_t})
print("5 lowest-margin tokens:", [(int(t), round(float(m[t]), 3)) for t in lowest])

lines = [
    f"shape CV across r in {{0.1,0.5,1,2}}, t<85: {np.median(cv_shape)*100:.1f}%\n",
    f"per-dir large-r exponent at L27,t=0: min={alphas_dir.min():.2f} "
    f"med={np.median(alphas_dir):.2f} max={alphas_dir.max():.2f}\n",
    f"corr(linear gain, large-r exponent) across dirs: {r_gain_alpha:.2f}\n",
    f"max dh at r=2: {dh[i2].max():.1f} at L27 dir {j3} t={t3}; "
    f"min cos at r=2, L27: {cos[i2, :, li27, :].min():.3f}\n",
    "flip fraction by radius: " +
    str(dict(zip(radii.tolist(), np.round(frac_flip, 3).tolist()))) + "\n",
    "flip times vs base margins: " +
    str({int(t): round(float(m[t]), 3) for t in flip_t}) + "\n",
    "5 lowest-margin tokens: " +
    str([(int(t), round(float(m[t]), 3)) for t in lowest]) + "\n",
]
C.txt("large_r_stats.txt", "".join(lines))

# ---------------------------------------------------------------- fig 7
fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))

# (a) aggregate L27: large r vs linear prediction
ax = axes[0, 0]
for ir in large_idx:
    r = radii[ir]
    ax.plot(dhm[ir, li27, :], color=C.rcolor(r, radii), lw=1.5,
            label=f"r={r:g}")
    ax.plot(r * G[li27], color=C.rcolor(r, radii), ls="--", lw=0.9)
ax.plot(dhm[4, li27, :], color="k", lw=1.5, label="r=0.1 (linear regime)")
ax.plot(0.1 * G[li27], "k--", lw=0.9)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("$\\delta h(t,L27)$")
ax.set_title("(a) large r vs its linear-regime prediction $r\\,G(t)$ (dashed):\n"
             "same envelope, superlinear amplitude", loc="left")
ax.legend(fontsize=8, ncol=2)
ax.annotate("dashed = $r\\times$G(t) extrapolated\nfrom the linear regime",
            xy=(190, 0.5), fontsize=9, color="0.25")
ax.annotate(f"r=2 per-dir max: $\\delta h\\approx$ {dh[i2].max():.0f}\n"
            f"(dir 3, t={t3}; $\\approx$0.7$\\times$base norm,\n"
            f"cos$\\to${cos[i2, :, li27, :].min():.2f}); aggregate\nshown peaks "
            f"$\\approx$20", xy=(96, 90), fontsize=8.5, color="0.2")

# (b) shape collapse across the full r range
ax = axes[0, 1]
for r, sh in shapes.items():
    ax.plot(sh, lw=1.4, color=C.rcolor(r, radii), label=f"r={r:g}")
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("$\\delta h(t)\\,/\\,\\delta h(0)$")
ax.set_title(f"(b) envelope is r-independent even at r=2:\nmedian CV over r = "
             f"{np.median(cv_shape)*100:.0f}% (t<85)", loc="left")
ax.legend(fontsize=8.5)
ax.annotate("no change of TIME regime\nat large r — only amplitude",
            xy=(150, 0.5), fontsize=9, color="0.25")

# (c) three individual dirs at r=2
ax = axes[1, 0]
for idir, lab in [(0, f"dir 0 ($\\alpha$={alphas_dir[0]:.1f})"),
                  (6, f"dir 6 ($\\alpha$={alphas_dir[6]:.1f})"),
                  (11, f"dir 11 ($\\alpha$={alphas_dir[11]:.1f})"),
                  (10, f"dir 10 ($\\alpha$={alphas_dir[10]:.1f})")]:
    ax.plot(dh[i2, idir, li27, :], lw=1.3, label=lab)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("$\\delta h(t,L27)$, r=2")
ax.set_title("(c) r=2, individual directions: the exponent is\n"
             "DIRECTION-dependent (0.5 to 2.0)", loc="left")
ax.legend(fontsize=8.5)
ax.annotate(f"$\\alpha$ over r$\\in$[0.5,2] at t=0:\nmin={alphas_dir.min():.2f} "
            f"(saturating) to max={alphas_dir.max():.2f};\nuncorrelated with the "
            f"linear gain\n(Pearson r = {r_gain_alpha:.2f})",
            xy=(130, 1.2), fontsize=9, color="0.25")

# (d) prediction flips vs radius
ax = axes[1, 1]
ax.bar([str(r) for r in radii], frac_flip, color="0.55")
ax.set_xlabel("perturbation radius r"); ax.set_ylabel("fraction of directions flipped")
ax.set_title("(d) flips onset at r$\\approx$0.2, reach 95% at r$\\geq$1.5;\n"
             "flip times are the lowest-margin tokens", loc="left")
ax.annotate("flip times (pred$T$) $\\subset$ {0, 2, 14, 16, 123, 143}\n"
            "all with base margin $\\leq$ 0.33\n(median margin $\\approx$1.9; "
            "t=123, 143 have\nthe two smallest margins, 0.02)",
            xy=(0.02, 0.45), fontsize=9, color="0.2")

fig.suptitle("A1 large-r regime (w100): no saturation — superlinear with a "
             "direction-dependent exponent (0.5-2.0 at L27);\nthe biggest "
             "deviations sit in the pre-flush surge; prediction flips are "
             "decided by the BASE run's margins, not by $\\delta h$",
             y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
C.save(fig, "fig7_large_r.png")
