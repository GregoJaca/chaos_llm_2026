"""Closed loop vs network-only replay, done properly.

Definitions used everywhere in the report
    a_c(d,t) = ||delta h_t|| / ||delta h_1||   for direction d, closed loop
    a_r(d,t) = same for the replay of the SAME recorded orbit
    L(d,t)   = log( a_c(d,t) / a_r(d,t) )      per direction, then median/IQR
The ratio is always formed per direction and only then summarised, so the band is
an honest spread over probe directions.  L(1) = 0 exactly (both curves are
normalised at t=1), which is why the fit is L(t) = Linf (1 - exp(-t/tau)) with
NO free offset.

Data: every (loop, replay) pair in the same directory, at the same T, all chunks
pooled (a chunk = one run with `--dirs K`; the reference orbit is identical across
chunks because row 0 is unperturbed), so 8-direction legacy runs and 256-direction
v3 runs are handled by the same code.
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from looputil import TAGS, LEGACY, gather, pairs
from style import plt, save


def sat(tt, L, tau):
    return L * (1.0 - np.exp(-tt / tau))
def band(x, axis=0):
    with np.errstate(all="ignore"):
        return (np.nanmedian(x, axis=axis),
                (np.nanpercentile(x, 25, axis=axis), np.nanpercentile(x, 75, axis=axis)))
summary = {}
MODELS = [m for m in TAGS]
fig, axes = plt.subplots(3, 4, figsize=(16.6, 10.6))
for ax, name in zip(axes.flat, MODELS):
    ac, ar, ds = gather(name)
    if ac is None:
        ax.axis("off"); continue
    mc, (clo, chi) = band(ac, 0)
    mr, (rlo, rhi) = band(ar, 0)
    t = np.arange(len(mc))
    ax.fill_between(t[1:], np.maximum(clo[1:], 1e-8), chi[1:], color="#d62728", alpha=0.12, lw=0)
    ax.fill_between(t[1:], np.maximum(rlo[1:], 1e-8), rhi[1:], color="#1f77b4", alpha=0.12, lw=0)
    ax.plot(t[1:], mc[1:], color="#d62728", lw=1.2, label=f"closed ({ac.shape[0]} dirs)")
    ax.plot(t[1:], mr[1:], color="#1f77b4", lw=1.0, ls="--", label="replay")
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{name}  ({len(ds)} run{'s' if len(ds) > 1 else ''})", fontsize=8.8)
    ax.set_xlabel("step $t$", fontsize=7.2)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.2)
    ax.legend(fontsize=5.6); ax.grid(True, which="both", alpha=0.24)
    summary[name] = dict(n_dirs=int(ac.shape[0]), steps=int(len(mc)),
                         growth_closed=float(mc[-1]), growth_replay=float(mr[-1]))
for j in range(len(MODELS), 12):
    axes.flat[j].axis("off")
fig.suptitle("Closed loop vs replay at $T=1$: median over directions, band = IQR over directions", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_all.png")

# ---------------------------------------------------------------- per model: response + ratio + fit
for name in MODELS:
    ac, ar, ds = gather(name)
    if ac is None:
        continue
    t = np.arange(ac.shape[1])
    Ld = np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))   # (dir, t), NaN where padded
    Lm, (Llo, Lhi) = band(Ld, 0)
    fig, axes2 = plt.subplots(1, 2, figsize=(13.2, 4.5))
    ax = axes2[0]
    for arr, c, ls, lab in [(ac, "#d62728", "-", "closed loop"),
                            (ar, "#1f77b4", "--", "replay (network only)")]:
        m, (lo, hi) = band(arr, 0)
        ax.fill_between(t[1:], np.maximum(lo[1:], 1e-8), hi[1:], color=c, alpha=0.12, lw=0)
        ax.plot(t[1:], m[1:], color=c, lw=1.4, ls=ls, label=lab)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
    ax.set_title(f"{name}: response ($T=1$, {ac.shape[0]} directions)", loc="left", fontsize=9.6)
    ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.28)

    ax = axes2[1]
    ax.fill_between(t[1:], Llo[1:], Lhi[1:], color="#2ca02c", alpha=0.14, lw=0)
    ax.plot(t[1:], Lm[1:], color="#2ca02c", lw=1.6,
            label=r"$\log(\|\delta h_t^{\rm closed}\|/\|\delta h_t^{\rm replay}\|)$")
    tau = float("nan"); Linf = float("nan"); r2 = float("nan")
    sel = (t >= 1) & np.isfinite(Lm)
    try:
        popt, _ = curve_fit(sat, t[sel], Lm[sel], p0=[max(Lm[sel][-1], 0.5), max(len(t) / 4, 5)],
                            bounds=([0.0, 0.5], [60.0, 5e4]), maxfev=40000)
        Linf, tau = float(popt[0]), float(popt[1])
        pred = sat(t[sel], *popt)
        r2 = 1 - np.nansum((Lm[sel] - pred) ** 2) / np.nansum((Lm[sel] - np.nanmean(Lm[sel])) ** 2)
        tt = np.linspace(1, t[-1], 300)
        ax.plot(tt, sat(tt, *popt), "k--", lw=1.1,
                label=rf"$L_\infty$={Linf:.1f} nats, $\tau$={tau:.0f} steps ($R^2$={r2:.2f})")
    except Exception as exc:
        print("fit failed", name, exc)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$L(t)=\log(\|\delta h_t^{\rm closed}\|/\|\delta h_t^{\rm replay}\|)$  (nats)")
    ax.set_title("feedback gain, saturating fit", loc="left", fontsize=9.6)
    ax.legend(fontsize=6.8); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, f"pack_loop_{TAGS[name]}.png")

    summary[name].update(L_inf=Linf, tau=tau, sat_r2=r2,
                         L_end=float(np.nanmedian(Ld[:, int(0.8 * len(t)):])),
                         params=[p.replace(HERE, "") for p in ds])
    print("%-14s dirs=%-4d steps=%-5d L_inf=%5.2f tau=%7.1f r2=%5.2f  L_end=%5.2f" % (
        name, ac.shape[0], len(t), Linf, tau, r2, summary[name]["L_end"]))

json.dump(summary, open(os.path.join(HERE, "analysis_update", "pack_loop_core.json"), "w"), indent=1)
print("done")
