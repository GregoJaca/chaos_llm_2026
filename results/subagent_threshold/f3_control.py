"""Fig 3: which near-tie controls each direction.

Per window: top = baseline margin trace m_t (log); bottom = heatmap of the local
signed threshold r_c^-(d,t) with rows (directions) sorted by their weakest tie
r_*^- (most dangerous on top). White dot = that direction's controlling step
t* = argmin_t r_c^-; green star = observed first flip in the free run at r=0.3.
Rows are labelled by direction id and r_*^- (no meaningless direction index axis).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from common import Case, OUT

COL = {"w100": "tab:blue", "w200": "tab:orange", "w500": "tab:green"}
R_OBS = 0.3


def window_block(fig, gs, name):
    c = Case(name)
    rc_neg = np.where(c.dm[c.ir_ref] < 0, c.rc_ref(), np.inf)
    rstar = rc_neg.min(axis=1)
    tstar = rc_neg.argmin(axis=1)
    order = np.argsort(rstar)                      # most dangerous first
    Z = np.log10(np.clip(rc_neg[order], 1e-4, 1e3))  # (20, T)

    j = c.free()
    obs = None
    for i, r in enumerate(j["radii"]):
        if abs(r - R_OBS) < 1e-9:
            obs = np.array(j["obs"][i])
    if obs is None:  # w100 free file has r=0.3 too; fallback
        obs = np.full(c.n_dirs, -1)

    # --- margin trace ---
    axm = fig.add_subplot(gs[0])
    t = np.arange(c.T)
    axm.semilogy(t, np.maximum(c.m, 1e-4), lw=0.8, color="0.4")
    axm.set_ylim(1e-3, 60)
    axm.set_xticklabels([])
    axm.set_ylabel("$m_t$", fontsize=8)
    axm.grid(alpha=0.25, which="both")
    for tt in np.unique(tstar):
        axm.plot(tt, max(c.m[tt], 1e-4), "o", ms=5, color="tab:red", mec="k", mew=0.5, zorder=5)
    tmin = int(np.argmin(c.m))
    axm.annotate(f"tie t={tmin}: m={c.m[tmin]:.4g}", xy=(tmin, max(c.m[tmin], 1e-4)),
                 xytext=(6, 6), textcoords="offset points", fontsize=8, color="tab:red")
    axm.set_title(f"window {c.window}", fontsize=10, color=COL[name])

    # --- threshold heatmap ---
    axh = fig.add_subplot(gs[1], sharex=axm)
    masked = np.ma.masked_invalid(Z)
    im = axh.pcolormesh(t, np.arange(c.n_dirs), masked, cmap="RdYlBu_r",
                         vmin=-2.0, vmax=1.0, rasterized=True, shading="auto")
    for row, d in enumerate(order):
        axh.plot(tstar[d], row, "o", ms=4, mfc="white", mec="k", mew=0.5)
        if obs[d] >= 0:
            axh.plot(obs[d], row, "*", ms=10, color="tab:green",
                     mec="k", mew=0.5)
    axh.set_ylim(20, 0)
    axh.set_xlim(0, c.T)
    axh.set_yticks(np.arange(20) + 0.5)
    axh.set_yticklabels([f"d{d}  {rstar[d]:.3g}" for d in order], fontsize=6)
    axh.set_xlabel("generation step $t$")
    axh.set_ylabel("directions sorted by $r_*^-$", fontsize=8)
    cnt = {}
    for tt in tstar:
        cnt[int(tt)] = cnt.get(int(tt), 0) + 1
    txt = "controlling steps: " + ", ".join(
        f"t={k}: {v}/20" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))
    axh.text(0.5, -0.22, txt, transform=axh.transAxes, ha="center", va="top", fontsize=8)
    return im, rstar, tstar, obs, order


fig = plt.figure(figsize=(13, 11))
outer = gridspec.GridSpec(3, 1, hspace=0.42, figure=fig)
info = []
for i, name in enumerate(["w100", "w200", "w500"]):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                             height_ratios=[1.15, 2.6], hspace=0.06)
    info.append(window_block(fig, inner, name))

cax = fig.add_axes([0.92, 0.09, 0.012, 0.2])
cb = fig.colorbar(info[0][0], cax=cax)
cb.set_label("$\\log_{10} r_c^-$  (low = fragile; gray = stabilizing $\\delta m>0$)", fontsize=8)
fig.suptitle("Which near-tie controls each direction: local signed threshold field "
             "$r_c^-(d,t)$, rows sorted by danger", fontsize=12, y=0.985)
fig.savefig(OUT + r"\fig3_control_map.png", dpi=160, bbox_inches="tight")
print("saved fig3")
