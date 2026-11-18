"""Fig 1: physical margin-gain scatter.

x = baseline margin m_t, y = gain |dm_t|/r (r = r_ref = 0.01, linear regime).
Iso-threshold lines g = m / r_c for r_c = 0.03/0.1/0.3/1: a point above the line
r_c = r means a kick of radius r flips the argmax there. Overlay the points that
actually flipped in the free runs (r = 0.1 and 0.3).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from common import Case, OUT

R_FILL = 100.0  # clip for display


def scatter_panel(ax, c, r_hi=0.3, r_lo=0.1, has_lo=True):
    m = np.repeat(c.m[None, :], c.n_dirs, axis=0)          # (20, T)
    g = np.abs(c.dm[c.ir_ref]) / c.r_ref                    # (20, T)
    rc_neg = np.where(c.dm[c.ir_ref] < 0, c.rc_ref(), np.inf)

    # density background: all (d, t) points
    ok = np.isfinite(rc_neg)
    ax.hist2d(m.ravel(), g.ravel(), bins=[np.logspace(-2.7, 1.6, 45), np.logspace(-4, 1, 45)],
              norm=LogNorm(), cmap="Greys", alpha=0.85, cmin=1)

    # iso-threshold lines g = m / r_c
    for rc_v, style in [(0.03, ":"), (0.1, "--"), (0.3, "-"), (1.0, "-")]:
        xs = np.logspace(-2.7, 1.6, 10)
        ax.plot(xs, xs / rc_v, style, color="0.35", lw=1.0 if rc_v != 0.3 else 1.4,
                zorder=3)
        m_lab = min(30.0, 8.0 * rc_v)
        ax.annotate(f"$r_c$={rc_v}", xy=(m_lab, m_lab / rc_v), fontsize=7.5, color="0.25",
                    ha="left", va="top", xytext=(0, -2), textcoords="offset points")

    # observed flips, from free runs
    obs_hi = {r_hi: None}
    j = c.free()
    for i, r in enumerate(j["radii"]):
        if abs(r - r_hi) < 1e-9:
            obs_hi = np.array(j["obs"][i])
    fl = np.where(obs_hi >= 0)[0]
    for d in fl:
        t = obs_hi[d]
        ax.plot(m[d, t], g[d, t], marker="*", ms=13, color="tab:green",
                mec="k", mew=0.6, ls="none", zorder=6)
        ax.annotate(f"d{d}@{t}", xy=(m[d, t], g[d, t]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", zorder=6)
    if has_lo:
        for i, r in enumerate(j["radii"]):
            if abs(r - r_lo) < 1e-9:
                obs_lo = np.array(j["obs"][i])
        fl = np.where(obs_lo >= 0)[0]
        for d in fl:
            t = obs_lo[d]
            ax.plot(m[d, t], g[d, t], marker="^", ms=9, color="tab:blue",
                    mec="k", mew=0.6, ls="none", zorder=7)

    # non-flippers at their signed controlling point
    nf = [d for d in range(c.n_dirs) if obs_hi[d] < 0]
    for d in nf:
        t = int(np.argmin(rc_neg[d]))
        if not np.isfinite(rc_neg[d, t]):
            t = int(np.argmin(c.rc_ref()[d]))
        ax.plot(m[d, t], g[d, t], "x", ms=6, color="tab:red", mew=1.2, alpha=0.85, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-3, 40)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel("margin  $m_t$   (top1$-$top2 logit gap)")
    ax.set_ylabel("gain  $|g_t| = |\\delta m_t|/r$")
    ax.set_title(f"window {c.window}   (r_ref={c.r_ref}; ties: "
                 f"t={c.m.argmin()} m={c.m.min():.4g})", fontsize=10)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, name in zip(axes, ["w100", "w200", "w500"]):
    c = Case(name)
    scatter_panel(ax, c)
axes[0].plot([], [], ls="none", marker="*", ms=12, color="tab:green", mec="k", label="observed flip @ r=0.3")
axes[0].plot([], [], ls="none", marker="^", ms=9, color="tab:blue", mec="k", label="observed flip @ r=0.1")
axes[0].plot([], [], ls="none", marker="x", ms=7, color="tab:red", mew=1.5, label="no flip @ 0.3 (at its weakest tie)")
axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
fig.suptitle("Margin-gain geometry: a sustained kick of radius r flips the argmax wherever "
             "$|g_t| \\geq m_t/r$\n(flips live above the iso-$r_c$ lines, at the deep near-ties)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT + r"\fig1_margin_gain_scatter.png", dpi=160)
print("saved fig1")
