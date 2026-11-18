"""Gain vs radius: concave (U-shaped) dependence with percentile bands."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
radii = d["radii"]; dm = d["dm"]; m = d["margins"]
G = np.abs(dm) / radii[:, None, None]          # [n_r, 20 dirs, 300 t]

q = {p: np.percentile(G.reshape(len(radii), -1), p, axis=1) for p in [10, 25, 50, 75, 90]}

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))

ax = axs[0]
ax.fill_between(radii, q[10], q[90], color="tab:blue", alpha=0.10, label="10–90%")
ax.fill_between(radii, q[25], q[75], color="tab:blue", alpha=0.20, label="25–75%")
ax.loglog(radii, q[50], "o-", color="tab:blue", lw=1.4, label="median")
lr = np.log(radii)
coef = np.polyfit(lr, np.log(q[50]), 2)
rr = np.logspace(np.log10(radii.min()), np.log10(radii.max()), 100)
ax.loglog(rr, np.exp(np.polyval(coef, np.log(rr))), "k--",
          label="parabola in $\\log r$")
pred = np.exp(np.polyval(coef, lr))
r2 = 1 - np.sum((np.log(q[50]) - np.log(pred)) ** 2) / np.sum((np.log(q[50]) - np.log(q[50]).mean()) ** 2)
ax.text(0.03, 0.06, f"$R^2={r2:.3f}$", transform=ax.transAxes, fontsize=9)
ax.set_xlabel(r"perturbation radius $r$")
ax.set_ylabel(r"$g=|\delta m_t|/r$  (logit-gap change per unit $r$)")
ax.set_title("(a) Concave dependence: a floor near $r\\approx0.03$,\nrising at both ends")
ax.legend(fontsize=7.5, loc="upper center")
ax.grid(alpha=0.25, which="both")

ax = axs[1]
ri_lo = int(np.argmin(np.abs(radii - 0.1)))
ri_hi = int(np.argmin(np.abs(radii - 2.0)))
ratio = G[ri_hi] / np.maximum(G[ri_lo], 1e-12)
r = ratio[np.isfinite(ratio) & (ratio > 0) & (ratio < 1e6)]
ax.hist(np.log10(r), bins=60, color="tab:red", alpha=0.8)
ax.axvline(0, color="k", lw=0.8)
med = np.median(r); p10, p90 = np.percentile(r, [10, 90])
ax.axvline(np.log10(med), color="k", ls="--", lw=1.2)
ax.set_xlabel(r"$\log_{10}[g(r=2)/g(r=0.1)]$  at one step and direction")
ax.set_ylabel("count")
ax.set_title(f"(b) Heterogeneous: median $\\times${med:.2f},\n10–90% range {p10:.2f}–{p90:.1f}")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "threshold_gain_scaling.png"), dpi=190)
print("saved; parabola coefficients", coef, "R2", round(float(r2), 3))
