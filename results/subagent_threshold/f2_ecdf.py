"""Fig 2: distribution of local thresholds.

Panel A: ECDF of per-direction thresholds r_*^- = min_t r_c^- (signed, destabilizing)
         and r_* from |dm|, for windows 100/200/500.
Panel B: ECDF of all local thresholds r_c^-(d,t) pooled over directions and steps
         (the "threshold field"), same windows.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Case, OUT

COL = {"w100": "tab:blue", "w200": "tab:orange", "w500": "tab:green"}


def ecdf(ax, x, color, label, ls="-", lw=1.8):
    x = np.sort(np.asarray(x, dtype=float))
    x = np.clip(x, 1e-4, 1e2)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, ls=ls, lw=lw, label=label)
    return x


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

# ---------- Panel A: per-direction r_* ----------
ax = axes[0]
for name in ["w100", "w200", "w500"]:
    c = Case(name)
    rc_abs = c.rc_ref()
    rc_neg = np.where(c.dm[c.ir_ref] < 0, rc_abs, np.inf)
    rstar_abs = rc_abs.min(axis=1)
    rstar_neg = rc_neg.min(axis=1)
    ecdf(ax, rstar_neg, COL[name], f"{name}  signed $r_*^-$")
    ecdf(ax, rstar_abs, COL[name], f"{name}  $|\\delta m|$ $r_*$", ls=":", lw=1.3)
    med = np.median(np.clip(rstar_neg, 1e-4, 1e2))
    ax.plot(med, 0.5, "o", color=COL[name], ms=6, mec="k", mew=0.6)
    ax.annotate(f"{name}: med {med:.3g}", xy=(med, 0.5), xytext=(6, 4),
                textcoords="offset points", fontsize=7.5, color=COL[name])
for r in [0.1, 0.3]:
    ax.axvline(r, color="0.4", lw=0.8, ls="--")
    ax.annotate(f"r={r}", xy=(r, 0.97), fontsize=8, color="0.3",
                ha="center", xytext=(0, -3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlim(3e-3, 1e2)
ax.set_ylim(0, 1.02)
ax.set_xlabel("local threshold  $r_* = \\min_t r_c$   per direction")
ax.set_ylabel("ECDF over 20 directions")
ax.set_title("Per-direction thresholds", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, which="both")

# ---------- Panel B: pooled local thresholds ----------
ax = axes[1]
stats = {}
for name in ["w100", "w200", "w500"]:
    c = Case(name)
    rc_neg = np.where(c.dm[c.ir_ref] < 0, c.rc_ref(), np.inf)
    fin = rc_neg[np.isfinite(rc_neg)]
    ecdf(ax, fin, COL[name], f"{name}  ({len(fin)}/{rc_neg.size} destab.)")
    q = np.quantile(fin, [0.05, 0.5, 0.95])
    ax.plot(q[1], 0.5, "o", color=COL[name], ms=6, mec="k", mew=0.6)
    f01 = (fin <= 0.1).mean()
    f03 = (fin <= 0.3).mean()
    stats[name] = (q, f01, f03)
    ax.annotate(f"{name}: 5%/{q[0]:.2g}   med/{q[1]:.2g}\n"
                f"   frac$\\leq$0.1: {f01:.4f}   $\\leq$0.3: {f03:.4f}",
                xy=(0.02, 0.74 - 0.20 * ["w100", "w200", "w500"].index(name)),
                xycoords="axes fraction", fontsize=8, color=COL[name])
for r in [0.1, 0.3]:
    ax.axvline(r, color="0.4", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlim(3e-3, 1e2)
ax.set_ylim(0, 1.02)
ax.set_xlabel("local threshold  $r_c^-(d,t)$   pooled over dirs x steps")
ax.set_ylabel("ECDF over (d,t) pairs")
ax.set_title("Threshold field (all local thresholds; mostly safe -- danger is the left tail)",
             fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, which="both")

fig.suptitle("Local thresholds over directions and steps: signed (margin-destroying) response, "
             "$r_c^- = r\\,m_t/\\max(-\\delta m_t,\\,0)$", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT + r"\fig2_threshold_ecdf.png", dpi=160)
print("saved fig2")
for k, v in stats.items():
    print(k, "median", round(v[0][1], 4), "frac<=0.1", round(v[1], 3), "frac<=0.3", round(v[2], 3))
