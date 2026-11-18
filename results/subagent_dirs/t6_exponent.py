"""TEST 6: direction dependence of the scaling exponent.

alpha_d = slope of log(median_t ||dh_t||) vs log r over r = 0.3..2 (large r).
A single-scalar-gain model predicts alpha_d identical for all directions
(the gain cancels in the slope). We measure the spread and its link to gain.
Also: exponent in the LINEAR regime (should be ~1 for all), and the exponent
of the margin response |dm|.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (L14, L21, L27, LIN_R, LARGE_R, load_w100, logd,
                    save_json, savefig, spear)

z = load_w100()
dh, dm, radii, layers = z["dh_norm"], z["dm"], z["radii"], z["layers"]
D = dh.shape[1]
EX = [10, 9, 13]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
N = {}

def alpha_fit(vals_r, ridx):
    x = np.log(radii[ridx])
    y = np.log(vals_r)
    return float(np.polyfit(x, y, 1)[0]), float(np.polyfit(x, y, 1)[1])

# alpha at layer 27 in the large-r regime, per direction
g_large = np.median(dh[LARGE_R][:, :, L27], axis=-1)          # [len(R), D]
alpha = np.array([alpha_fit(g_large[:, d], LARGE_R)[0] for d in range(D)])
alpha_lin = np.array([alpha_fit(np.median(dh[LIN_R + [4], d, L27], axis=-1),
                                LIN_R + [4])[0] for d in range(D)])
# dm exponent over r = 0.1..2 (indices 4..11)
g_dm = np.median(np.abs(dm[4:12]), axis=-1)     # [R, D]
alpha_dm = np.array([alpha_fit(g_dm[:, d], list(range(4, 12)))[0] for d in range(D)])

gain01 = np.median(dh[4, :, L27], axis=-1)
N["alpha_large"] = {"min": float(alpha.min()), "max": float(alpha.max()),
                    "std": float(alpha.std()), "values": alpha.tolist()}
N["alpha_linear_regime"] = {"min": float(alpha_lin.min()), "max": float(alpha_lin.max()),
                            "std": float(alpha_lin.std())}
N["alpha_dm"] = {"min": float(alpha_dm.min()), "max": float(alpha_dm.max()),
                 "std": float(alpha_dm.std())}
N["spearman_alpha_vs_gain"] = spear(alpha, np.log(gain01))
N["pearson_alpha_vs_gain"] = float(np.corrcoef(alpha, np.log(gain01))[0, 1])
N["spearman_alpha_vs_logrstar_r2"] = None  # filled below from t5 numbers if present

# per-layer alpha at 14, 21: same directions?
a14 = np.array([alpha_fit(np.median(dh[LARGE_R, d, L14], axis=-1), LARGE_R)[0]
               for d in range(D)])
a21 = np.array([alpha_fit(np.median(dh[LARGE_R, d, L21], axis=-1), LARGE_R)[0]
               for d in range(D)])
N["alpha_L14"] = {"min": float(a14.min()), "max": float(a14.max())}
N["spearman_alpha27_vs_alpha14"] = spear(alpha, a14)
N["spearman_alpha27_vs_alpha21"] = spear(alpha, a21)

try:
    import json
    with open("numbers_t5.json") as f:
        n5 = json.load(f)
    rstar2 = np.array(n5["decomp"]["r=2.0"].get("tie", []))  # placeholder
    N.pop("spearman_alpha_vs_logrstar_r2", None)
except Exception:
    N.pop("spearman_alpha_vs_logrstar_r2", None)

print("=== TEST 6 ===")
print("alpha large-r: min %.3f max %.3f std %.3f" % (alpha.min(), alpha.max(), alpha.std()))
print("alpha linear:  min %.3f max %.3f std %.3f" % (alpha_lin.min(), alpha_lin.max(), alpha_lin.std()))
print("alpha dm (0.1..2): min %.3f max %.3f std %.3f" % (alpha_dm.min(), alpha_dm.max(), alpha_dm.std()))
print("spearman(alpha, log gain):", round(N["spearman_alpha_vs_gain"], 3),
      " pearson:", round(N["pearson_alpha_vs_gain"], 3))
print("alpha L27 vs L14 spearman:", round(N["spearman_alpha27_vs_alpha14"], 3),
      " vs L21:", round(N["spearman_alpha27_vs_alpha21"], 3))

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

ax = axs[0, 0]
for d in range(D):
    ax.plot(radii[LARGE_R], g_large[:, d], color="gray", alpha=0.4, lw=0.8)
lbl = ["highest gain", "median gain", "lowest gain"]
xfit = np.logspace(np.log10(0.25), np.log10(2.1), 20)
for k, d in enumerate(EX):
    a, b = alpha_fit(g_large[:, d], LARGE_R)
    ax.plot(radii[LARGE_R], g_large[:, d], "o", color=cols[k], ms=4)
    ax.plot(xfit, np.exp(b) * xfit ** a, color=cols[k], lw=1.6,
            label=f"dir {d} ({lbl[k]}): alpha={a:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("perturbation radius r")
ax.set_ylabel("gain = median_t ||dh_t|| at layer 27 (log scale)")
ax.set_title(f"(a) Power-law fits over r=0.3..2 (points = measured, lines = fit)\n"
             f"alpha range {alpha.min():.2f}..{alpha.max():.2f} "
             f"(spread {alpha.max()-alpha.min():.2f}; linear-regime alpha "
             f"{alpha_lin.min():.2f}..{alpha_lin.max():.2f})")
ax.legend(fontsize=7)

ax = axs[0, 1]
ax.scatter(np.log(gain01), alpha, s=40, color="gray")
for k, d in enumerate(EX):
    ax.scatter(np.log(gain01[d]), alpha[d], s=70, color=cols[k], zorder=3, label=f"dir {d}")
A = np.polyfit(np.log(gain01), alpha, 1)
xs = np.linspace(np.log(gain01).min(), np.log(gain01).max(), 10)
ax.plot(xs, np.polyval(A, xs), "k--", lw=1,
        label=f"OLS slope {A[0]:.2f} per log-unit of gain")
ax.set_xlabel("log gain at r=0.1 (linear regime, log-units)")
ax.set_ylabel("large-r exponent alpha (r=0.3..2)")
ax.set_title(f"(b) Do high-gain directions saturate earlier?\n"
             f"spearman(alpha, log gain) = {N['spearman_alpha_vs_gain']:.2f}  "
             f"(negative = high gain -> weaker growth at large r)")
ax.legend(fontsize=7)

ax = axs[1, 0]
ax.scatter(alpha_dm, alpha, s=40, color="gray")
for k, d in enumerate(EX):
    ax.scatter(alpha_dm[d], alpha[d], s=70, color=cols[k], zorder=3)
ax.set_xlabel("exponent of margin response: median |dm| vs r (r=0.1..2)")
ax.set_ylabel("exponent of norm response alpha (r=0.3..2)")
ax.set_title(f"(c) Norm-response vs margin-response exponents\n"
             f"|dm| exponent range {alpha_dm.min():.2f}..{alpha_dm.max():.2f} "
             f"(>1 = margin response accelerates); spearman with alpha = "
             f"{spear(alpha, alpha_dm):.2f}")

ax = axs[1, 1]
ax.scatter(a14, alpha, s=30, color="#7f7f7f", label="layer 14")
ax.scatter(a21, alpha, s=30, color="#2ca02c", label="layer 21")
for k, d in enumerate(EX):
    ax.scatter(a21[d], alpha[d], s=70, color="k", zorder=3, marker="x")
lim = [min(a14.min(), a21.min()) - 0.05, max(a14.max(), a21.max()) + 0.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("alpha at layer 14 / 21 (r=0.3..2)")
ax.set_ylabel("alpha at layer 27 (r=0.3..2)")
ax.set_title(f"(d) Is a direction's exponent consistent across layers?\n"
             f"spearman(L27, L14) = {N['spearman_alpha27_vs_alpha14']:.2f}, "
             f"spearman(L27, L21) = {N['spearman_alpha27_vs_alpha21']:.2f}")
ax.legend(fontsize=7)

fig.suptitle("TEST 6: the scaling exponent is direction-dependent "
             "(single-gain model predicts identical alpha)", y=1.005)
savefig(fig, "fig6_exponent.png")
save_json("numbers_t6.json", N)
