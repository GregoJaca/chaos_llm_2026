#!/usr/bin/env python
"""Deep analysis of a large-model A1 run (signed thresholds, gain geometry).

Reads ``a1.npz`` from ``--run`` (default: the DS-R1-14B 1000-direction run) and
writes five figures plus a JSON with every number quoted in the notebook
section.  No GPU / torch is needed: all inputs are npz/json.

Definitions (r = run radius, u = direction index, t = generated step):

  m_t                 top1-top2 logit gap of the unperturbed greedy run (nats)
  dm_t(u)             change of that gap under the perturbation (nats)
  g_t(u) = |dm_t(u)|/r   signed local sensitivity (nats per unit radius)
  r_c^-(t,u) = m_t*r / max(-dm_t(u), 0)   radius at which this step flips
  r_*(u) = min_t r_c^-(t,u)               direction threshold radius
  t_*(u) = argmin_t r_c^-(t,u)            controlling step
  G(u) = median_t ||dh_t^last|| / r       typical last-state response gain

Only dm < 0 can destroy a margin; the run-sized flip test is r_*(u) <= r.

Run from the repository root:
    python scripts_spark/figs_a1_deep.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, NullFormatter

# --------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

C_MED = "#1f4e79"
C_BAND = "#7fb3d5"
C_OBS = "#c0392b"
C_ALT = "#e67e22"
C_GREEN = "#2e8b57"
C_GREY = "#8a8a8a"
C_LAYERS = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
            "#41ab5d", "#e67e22", "#d94801", "#a50f15"]


# ------------------------------------------------------------------ helpers
def load_run(run_dir):
    with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    d = np.load(os.path.join(run_dir, "a1.npz"))
    return d, meta


def qdict(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
           "min": float(x.min()), "max": float(x.max())}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def jf(o):
    """JSON-safe conversion."""
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    return o


def safe_pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def safe_spearman(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    return xs, np.arange(1, xs.size + 1) / xs.size


def derive(d, r):
    """All derived arrays for one run radius."""
    margins = d["margins"].astype(np.float64)
    dm = d["dm"][0].astype(np.float64)                    # [K,T]
    dh = d["dh_norm"][0].astype(np.float64)               # [K,nl,T]
    layers = d["layers"].astype(int)
    K, nl, T = dh.shape

    # signed local threshold; +inf where dm >= 0 (margin cannot be destroyed)
    rc = np.where(dm < 0.0,
                  margins[None, :] * r / np.maximum(-dm, 1e-300),
                  np.inf)                                  # [K,T]
    tstar = np.argmin(rc, axis=1)
    rstar = rc[np.arange(K), tstar]
    flip = rstar <= r

    # gain at the final stored state (layer 47 = output of block 46)
    G47 = np.median(dh[:, -1, :], axis=1) / r              # [K]
    gstar = np.abs(dm[np.arange(K), tstar]) / r            # local sensitivity [K]

    return dict(margins=margins, dm=dm, dh=dh, layers=layers, K=K, nl=nl, T=T,
                rc=rc, tstar=tstar, rstar=rstar, flip=flip,
                G47=G47, gstar=gstar)


def cos_block(d, r):
    """Response-vector geometry from the 24 stored final-norm vectors."""
    dhl = d["dh_last"][0].astype(np.float64)               # [nvec,T,D]
    nvec, T, D = dhl.shape
    n = np.linalg.norm(dhl, axis=2)                        # [nvec,T]
    tmed = np.argmin(np.abs(n - np.median(n, axis=1, keepdims=True)), axis=1)
    V = np.stack([dhl[i, tmed[i]] for i in range(nvec)])
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = Vn @ Vn.T
    iu = np.triu_indices(nvec, 1)
    pair = C[iu]
    # common components
    vm = Vn.mean(axis=0)
    cos_mean = Vn @ (vm / np.linalg.norm(vm))
    # PCA of the response directions
    sv = np.linalg.svd(Vn, compute_uv=False)
    ev = sv ** 2
    pca = ev / ev.sum()
    # pairwise blocks: dirs 0-19 share one input subspace, dirs 20-23 another
    within = C[:20, :20][np.triu_indices(20, 1)]
    cross = C[20:, :20].ravel()
    # temporal coherence of each direction's response vector
    lags = {}
    for lag in (1, 10, 50):
        cs = []
        for i in range(nvec):
            a = dhl[i, :-lag]
            b = dhl[i, lag:]
            cs.append(np.sum(a * b, 1) /
                      np.maximum(np.linalg.norm(a, axis=1) *
                                 np.linalg.norm(b, axis=1), 1e-30))
        lags[lag] = np.concatenate(cs)
    scale = np.linalg.norm(V, axis=1)
    return dict(dhl=dhl, norms=n, tmed=tmed, V=V, Vn=Vn, C=C, pair=pair,
                iu=iu, cos_mean=cos_mean, pca=pca, within=within,
                cross=cross, lags=lags, scale=scale, nvec=nvec)


# ------------------------------------------------------------------ figure 1
def fig_controlling(dv, margin_steps, top_steps, path):
    m = dv["margins"]
    K, T = dv["K"], dv["T"]
    tstar, flip = dv["tstar"], dv["flip"]
    allc = np.bincount(tstar, minlength=T)
    flc = np.bincount(tstar[flip], minlength=T)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    steps = np.arange(T)
    ax.bar(steps, allc, width=1.0, color="#c9d6e3",
           label=f"all {K} directions")
    ax.bar(steps, flc, width=1.0, color=C_MED,
           label=f"flip at r={0.1:g} ({int(flip.sum())})")
    for t in top_steps[:3]:
        n_all = int(allc[t])
        n_fl = int(flc[t])
        off = {21: (38, 575), 19: (78, 470), 189: (205, 225)}.get(t, (t + 25, n_all + 25))
        ax.annotate(f"t={t}: {n_all} all / {n_fl} flip",
                    xy=(t, n_all), xytext=off,
                    fontsize=7.8, color=C_MED,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=C_MED,
                                    shrinkA=0, shrinkB=2))
    ax.set_xlabel(r"$t$  (baseline generated step, 0 = first token)")
    ax.set_ylabel(r"number of directions with $t_*(u)=t$")
    ax.set_xlim(-4, T + 2)
    ax.set_ylim(0, max(allc) * 1.32)
    ax.legend(loc="upper right")
    ax.set_title("(a) every direction's controlling step")

    ax = axes[1]
    x = np.arange(len(margin_steps))
    cnt = np.array([allc[t] for t in margin_steps])
    fcnt = np.array([flc[t] for t in margin_steps])
    bars = ax.bar(x, cnt, color=C_MED, width=0.72,
                  label="all directions controlled here")
    ax.bar(x, fcnt, color=C_ALT, width=0.72,
           label="of which flip at r=0.1")
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(cnt) * 3.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n{m[t]:.1e}".replace("e-0", "e-")
                        for t in margin_steps], fontsize=7.0)
    ax.set_xlabel(r"step $t$  (second line: baseline margin $m_t$, nats)")
    ax.set_ylabel("number of directions (log scale)")
    for xi, c in zip(x, cnt):
        ax.text(xi, c * 1.18, f"{int(c)}", ha="center", va="bottom",
                fontsize=7.8, color=C_MED)
    ax.legend(loc="upper right")
    ax.set_title("(b) 10 smallest-margin steps")

    f = int(flip.sum())
    fig.suptitle(f"All {f} flips at r=0.1 are decided by two steps "
                 f"(t=21 and t=19); the ten near-ties are the shared weak points",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_sign(dv, steps, path):
    m = dv["margins"]
    dm = dv["dm"]
    r = 0.1
    flip = dv["flip"]

    fig = plt.figure(figsize=(13.0, 8.6))
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.28)

    for j, t in enumerate(steps):
        ax = fig.add_subplot(gs[0, j])
        s = dm[:, t] / r
        lo, hi = np.percentile(s, [0.5, 99.5])
        lim = max(abs(lo), abs(hi)) * 1.08
        bins = np.linspace(-lim, lim, 61)
        ax.hist(s[dm[:, t] < 0], bins=bins, color=C_OBS, alpha=0.85,
                label=r"$\delta m_t<0$ (margin destroyed)")
        ax.hist(s[dm[:, t] >= 0], bins=bins, color=C_MED, alpha=0.75,
                label=r"$\delta m_t\geq0$ (margin widened)")
        ax.axvline(0.0, color="k", lw=0.9)
        thr = -m[t] / r
        ax.axvline(thr, color="k", lw=1.3, ls="--")
        ax.axvspan(-lim, min(thr, lim), color=C_OBS, alpha=0.08)
        neg = float((dm[:, t] < 0).mean())
        bel = float((dm[:, t] <= -m[t]).mean())
        ax.text(0.97, 0.97,
                f"red: destroys ({100 * neg:.1f}%)\n"
                f"blue: widens ({100 * (1 - neg):.1f}%)\n"
                f"$m_t$={m[t]:.2e} nats\n"
                f"pass flip threshold: {100 * bel:.1f}%\n"
                f"(dashed line at $-m_t/r$)",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.4,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                          pad=1.5))
        ax.set_xlabel(r"$\delta m_t(u)/r$  (nats per unit radius)")
        ax.set_ylabel("directions per bin")
        ax.set_xlim(-lim, lim)
        ax.set_title(f"(a{j + 1:d}) step t={t}")

    ax = fig.add_subplot(gs[1, :])
    t1, t2 = steps[0], steps[2]
    x = dm[:, t1] * 1e3
    y = dm[:, t2] * 1e3
    ax.scatter(x[~flip], y[~flip], s=9, color=C_GREY, alpha=0.45,
               label=f"no flip (r*>0.1), n={int((~flip).sum())}")
    ax.scatter(x[flip], y[flip], s=10, color=C_OBS, alpha=0.65,
               label=f"flip (r*<=0.1), n={int(flip.sum())}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    rp = safe_pearson(dm[:, t1], dm[:, t2])
    rs = safe_spearman(dm[:, t1], dm[:, t2])
    rpf = safe_pearson(dm[flip, t1], dm[flip, t2])
    q = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            q[(sx, sy)] = int(((np.sign(dm[:, t1]) == sx) &
                               (np.sign(dm[:, t2]) == sy)).sum())
    ax.text(0.015, 0.97,
            f"Pearson r={rp:+.3f} (all), {rpf:+.3f} (flips); "
            f"Spearman r={rs:+.3f}\n"
            f"quadrants (sign at {t1}, sign at {t2}): "
            f"--/-- {q[(-1, -1)]}, --/+ {q[(-1, 1)]}, "
            f"+/-- {q[(1, -1)]}, +/+ {q[(1, 1)]}",
            transform=ax.transAxes, va="top", fontsize=8.6,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                      pad=1.5))
    ax.set_xlabel(rf"$\delta m_{{{t1}}}(u)$  (margin change, millinats)")
    ax.set_ylabel(rf"$\delta m_{{{t2}}}(u)$  (margin change, millinats)")
    ax.legend(loc="upper right")
    ax.set_title(f"(b) the two near-ties are not controlled by the same "
                 f"directions (r={rp:+.2f})")

    fig.suptitle("At a near-tie about half the directions destroy the margin "
                 "and half widen it; the two ties are almost independent",
                 fontsize=12, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_gain(dv, cb, path):
    G = dv["G47"]
    K = dv["K"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax = axes[0]
    lo, hi = G.min() * 0.95, G.max() * 1.05
    bins = np.geomspace(lo, hi, 42)
    ax.hist(G, bins=bins, color=C_MED, alpha=0.82)
    ax.set_xscale("log")
    ax.axvline(np.median(G), color=C_OBS, lw=1.4)
    ax.axvspan(np.percentile(G, 25), np.percentile(G, 75),
               color=C_OBS, alpha=0.08)
    ax.set_xlabel(r"$G(u)=\mathrm{median}_t\,\|\delta h_t^{47}\|/r$  "
                  r"(state-response gain, per unit radius)")
    ax.set_ylabel("directions per bin")
    ax.text(0.03, 0.97,
            f"p5={np.percentile(G, 5):.3f}\n"
            f"median={np.median(G):.3f}\n"
            f"p95={np.percentile(G, 95):.3f}\n"
            f"max/min={G.max() / G.min():.1f}x\n"
            f"red line=median; band=IQR",
            transform=ax.transAxes, va="top", fontsize=8.4)
    ax2 = ax.twinx()
    xs, ys = ecdf(G)
    ax2.plot(xs, ys, color=C_ALT, lw=1.4)
    ax2.set_ylabel("ECDF (fraction of directions)", color=C_ALT)
    ax2.tick_params(axis="y", colors=C_ALT)
    ax2.grid(False)
    ax.set_title("(a) gain is spread over a 14x range")

    ax = axes[1]
    pair = cb["pair"]
    bins = np.linspace(-0.20, 0.25, 55)
    ax.hist(pair, bins=bins, color=C_MED, alpha=0.82)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.median(pair), color=C_OBS, lw=1.3)
    ax.set_xlabel(r"$\cos(\delta h_i,\delta h_j)$  (angle between two "
                  r"response vectors)")
    ax.set_ylabel("pairs per bin")
    ax.text(0.03, 0.97,
            f"mean cos={pair.mean():+.3f} (red: median {np.median(pair):+.3f})\n"
            f"|cos|>0.1: {100 * np.mean(np.abs(pair) > 0.1):.1f}%\n"
            f"top-1 PC: {100 * cb['pca'][0]:.1f}% of variance\n"
            f"(1/24={100 / 24:.1f}% if isotropic)",
            transform=ax.transAxes, va="top", fontsize=8.4)
    ax.set_title("(b) responses are nearly orthogonal, not one shared mode")

    ins = ax.inset_axes([0.62, 0.38, 0.34, 0.30])
    ins.bar(np.arange(1, 9), 100 * cb["pca"][:8], color=C_GREEN)
    ins.axhline(100 / 24, color=C_OBS, ls="--", lw=0.9)
    ins.set_xlabel("PC", fontsize=7)
    ins.set_ylabel("% var", fontsize=7)
    ins.tick_params(labelsize=6.5)

    fig.suptitle("Responses vary 14x in strength but point in almost "
                 "orthogonal directions: no single response mode",
                 fontsize=12, y=1.02)
    fig.text(0.985, -0.015,
             "angle of a response vector vs the baseline state: not recoverable "
             "(baseline state vectors were not stored; the stored state cosine "
             "$\\cos(h',h_0)=0.99998\\ldots1.0000004$ is 1 at fp32 resolution)",
             ha="right", va="top", fontsize=7.2, color=C_GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_shape(dv, path, taus=(0, 10, 50, 100, 299)):
    dh = dv["dh"]
    layers = dv["layers"]
    K, nl, T = dh.shape
    r = 0.1
    tt = np.arange(T)

    fig = plt.figure(figsize=(14.2, 10.2))
    gs = GridSpec(3, 4, width_ratios=[1, 1, 1, 1.55], hspace=0.38, wspace=0.30)

    abs_prof = {}
    rel_prof = {}
    for li, L in enumerate(layers):
        row, col = divmod(li, 3)
        ax = fig.add_subplot(gs[row, col])
        p = dh[:, li, :] / r
        med = np.median(p, axis=0)
        q25, q75 = np.percentile(p, [25, 75], axis=0)
        if L == 0:
            ax.plot(tt, med, color=C_LAYERS[li], lw=1.0)
            ax.set_ylim(-0.01, 0.01)
            ax.text(0.5, 0.55,
                    "exactly 0 at every t\n(prompt-only perturbation\n"
                    "does not touch generated-token\nembeddings)",
                    transform=ax.transAxes, ha="center", fontsize=7.8)
        else:
            ax.fill_between(tt, q25, q75, color=C_BAND, alpha=0.35, lw=0)
            ax.plot(tt, med, color=C_LAYERS[li], lw=1.2)
            pk = int(np.argmax(med))
            ax.plot([pk], [med[pk]], marker="v", ms=5, color=C_OBS)
            t90 = int(np.argmax(med >= 0.9 * med[pk]))
            ax.text(0.03, 0.96,
                    f"peak t={pk} ({med[pk]:.2f})\n"
                    f"90% at t={t90}; end/peak={med[-1] / med[pk]:.2f}",
                    transform=ax.transAxes, va="top", fontsize=7.2)
        ax.set_title(f"state {L}" + (" (embedding)" if L == 0 else
                                     " (block output)" if L < 47 else
                                     " (deepest stored)"))
        if col == 0:
            ax.set_ylabel(r"$\|\delta h_t^{l}\|/r$")
        if row == 2 or (row < 2 and li + 3 >= nl):
            ax.set_xlabel(r"$t$ (generated step)")
        ax.set_xlim(0, T - 1)
        abs_prof[L] = med

    # relative response panel: heatmap state x step (9 layers, 300 steps)
    axr = fig.add_subplot(gs[:, 3])
    rows = [li for li, L in enumerate(layers) if L > 0]
    M = np.stack([np.median(dv["dh"][:, li, :] / dv["norm_h0"][li][None, :],
                            axis=0) for li in rows])
    im = axr.pcolormesh(tt, np.arange(len(rows)), M, shading="auto",
                        cmap="viridis",
                        norm=LogNorm(vmin=5e-5, vmax=5e-4))
    axr.set_yticks(np.arange(len(rows)))
    axr.set_yticklabels([f"state {layers[li]}" for li in rows], fontsize=8)
    axr.set_xlabel(r"$t$ (generated step)")
    axr.set_ylabel(r"state index $l$")
    axr.grid(False)
    cb_ = fig.colorbar(im, ax=axr, pad=0.02, fraction=0.05)
    cb_.set_label(r"$\|\delta h_t^{l}\|/\|h_t^{l}\|$  (relative response)")
    axr.set_title("(d) relative response (state 0: exactly 0, omitted)")
    for li, L in enumerate(layers):
        if L == 0:
            continue
        rel = np.median(dv["dh"][:, li, :], axis=0) / dv["norm_h0"][li]
        rel_prof[L] = rel

    fig.suptitle("The response grows 8.4x with depth, but the baseline state "
                 "grows 20.7x, so the relative response falls 2.5x",
                 fontsize=12.5, y=0.985)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return abs_prof, rel_prof


# ------------------------------------------------------------------ figure 5
def fig_rstar(dv, path):
    rstar, G, tstar = dv["rstar"], dv["G47"], dv["tstar"]
    m = dv["margins"]
    ok = np.isfinite(rstar) & (rstar < 1.0)
    groups = [21, 19, 189, 273]
    cols = {21: C_OBS, 19: C_ALT, 189: C_GREEN, 273: C_MED}
    r = 0.1

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    other = ok & ~np.isin(tstar, groups)
    ax.scatter(rstar[other], G[other], s=7, color=C_GREY, alpha=0.35,
               label=f"other controlling steps (n={int(other.sum())})")
    for t in groups:
        sel = ok & (tstar == t)
        if sel.sum():
            slope = float(np.polyfit(np.log(G[sel]), np.log(rstar[sel]), 1)[0])
            ax.scatter(rstar[sel], G[sel], s=9, color=cols[t], alpha=0.7,
                       label=f"t*={t} (n={int(sel.sum())}, slope {slope:+.2f})")
            c = float(np.median(rstar[sel] * G[sel]))
            xx = np.geomspace(max(rstar[ok].min(), 1e-4), 1.0, 50)
            ax.plot(xx, c / xx, color=cols[t], lw=1.0, ls="--", alpha=0.8)
    xx = np.geomspace(max(rstar[ok].min(), 1e-4), 1.0, 50)
    ax.plot(xx, m.min() / xx, color="k", lw=1.1, ls=":",
            label=r"naive $r_*=m_{\min}/G$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_*(u)=\min_t r_c^-(t,u)$  (threshold radius)")
    ax.set_ylabel(r"$G(u)$  (median state-response gain)")
    ax.legend(loc="lower left", fontsize=7.0, ncol=2)
    rp = safe_pearson(np.log(rstar[ok]), np.log(G[ok]))
    miss = float(np.median(rstar[ok] * G[ok]) / m.min())
    ax.text(0.03, 0.97,
            f"log-log Pearson r={rp:+.3f} (n={int(ok.sum())})\n"
            f"Spearman={safe_spearman(np.log(rstar[ok]), np.log(G[ok])):+.3f}\n"
            f"naive line misses the median by {miss:.0f}x\n"
            f"dashed: $r_*=$const$/G$ fitted per t* group",
            transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                      pad=1.5))
    ax.set_title("(a) threshold radius vs gain (r*<1 only)")

    ax = axes[1]
    order = [21, 19, 189, 273, 247, 255, 98, 0, 171]
    data, labels, colors = [], [], []
    gmed = {}
    for t in order:
        sel = np.isfinite(rstar) & (tstar == t)
        if sel.sum() >= 4:
            prod_t = rstar[sel] * G[sel]
            gmed[t] = float(np.median(prod_t))
            data.append(prod_t)
            labels.append(f"{t}\n(n={int(sel.sum())})")
            colors.append(cols.get(t, C_GREY))
    bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                    medianprops=dict(color="k", lw=1.2),
                    flierprops=dict(marker=".", ms=2, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_xlabel(r"controlling step $t_*$")
    ax.set_ylabel(r"$r_*(u)\cdot G(u)$  (nats)")
    allprod = rstar[ok] * G[ok]
    ax.axhline(np.median(allprod), color="k", ls="--", lw=1.0,
               label=f"all: median {np.median(allprod):.3f}")
    ax.text(0.03, 0.97,
            f"all: 5-95% = {np.percentile(allprod, 5):.4f}-"
            f"{np.percentile(allprod, 95):.3f}\n"
            f"log-std = {np.std(np.log(allprod)):.2f}\n"
            f"group medians span {max(gmed.values()) / min(gmed.values()):.0f}x",
            transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                      pad=1.5))
    ax.legend(loc="lower right")
    ax.set_title("(b) r*·G is roughly constant within a tie, not across ties")

    fig.suptitle("The threshold radius is set by which near-tie controls the "
                 "direction, not by the response gain; r*·G spans "
                 f"{max(gmed.values()) / min(gmed.values()):.0f}x across ties",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ numbers
def build_numbers(d, meta, dv, cb, margin_steps, top_steps, run_dir):
    m = dv["margins"]
    dm = dv["dm"]
    rstar, tstar, flip, G = dv["rstar"], dv["tstar"], dv["flip"], dv["G47"]
    K, T = dv["K"], dv["T"]
    allc = np.bincount(tstar, minlength=T)
    flc = np.bincount(tstar[flip], minlength=T)

    order = np.argsort(m)
    margin_table = [{"step": int(t), "margin": float(m[t]),
                     "controlling_all": int(allc[t]),
                     "controlling_flip": int(flc[t])}
                    for t in order[:10]]

    counts_all = [{"step": int(t), "count": int(allc[t]),
                   "count_flip": int(flc[t]), "margin": float(m[t])}
                  for t in np.argsort(-allc)[:10]]

    nsteps = {}
    for t in margin_steps:
        s = dm[:, t] / 0.1
        nsteps[str(t)] = {
            "margin": float(m[t]),
            "frac_destroyed": float((dm[:, t] < 0).mean()),
            "frac_widened": float((dm[:, t] >= 0).mean()),
            "frac_below_flip_threshold": float((dm[:, t] <= -m[t]).mean()),
            "frac_destroyed_among_flips": float((dm[flip, t] < 0).mean()),
            "dm_over_r": qdict(s),
            "dm_nats": qdict(dm[:, t]),
            "controlling_all": int(allc[t]),
            "controlling_flip": int(flc[t]),
            "local_gain_median": float(np.median(np.abs(dm[:, t]) / 0.1)),
        }

    corr = {}
    for a, b in [(21, 189), (21, 19), (19, 189)]:
        x, y = dm[:, a], dm[:, b]
        quad = {f"{sx:+d},{sy:+d}": int(((np.sign(x) == sx) &
                                         (np.sign(y) == sy)).sum())
                for sx in (-1, 1) for sy in (-1, 1)}
        corr[f"{a}_vs_{b}"] = {
            "pearson_all": safe_pearson(x, y),
            "spearman_all": safe_spearman(x, y),
            "pearson_flip": safe_pearson(x[flip], y[flip]),
            "spearman_flip": safe_spearman(x[flip], y[flip]),
            "quadrant_counts_all": quad,
        }

    # per-step local gain table
    step_gain = {}
    for t in margin_steps:
        g = np.abs(dm[:, t]) / 0.1
        step_gain[str(t)] = {
            "margin": float(m[t]),
            "g_median": float(np.median(g)),
            "g_p10": float(np.percentile(g, 10)),
            "g_p90": float(np.percentile(g, 90)),
            "g_over_G_median": float(np.median(g / G)),
        }

    # r* vs gain
    ok = np.isfinite(rstar) & (rstar < 1.0)
    logr, logG = np.log(rstar[ok]), np.log(G[ok])
    lm = np.log(m[tstar][ok])
    X = np.c_[lm, logG, np.ones(int(ok.sum()))]
    beta = np.linalg.lstsq(X, logr, rcond=None)[0]
    resid = logr - X @ beta
    r2 = 1.0 - resid.var() / logr.var()

    groups = {}
    for t in [21, 19, 189, 273, 247, 255, 98, 0, 171]:
        sel = np.isfinite(rstar) & (tstar == t)
        if sel.sum() < 4:
            continue
        prod = rstar[sel] * G[sel]
        slope = float(np.polyfit(np.log(G[sel]), np.log(rstar[sel]), 1)[0])
        groups[str(t)] = {
            "n": int(sel.sum()),
            "pearson_loglog": safe_pearson(np.log(rstar[sel]), np.log(G[sel])),
            "slope_logr_vs_logG": slope,
            "rstar_G_median": float(np.median(prod)),
            "rstar_G_p25": float(np.percentile(prod, 25)),
            "rstar_G_p75": float(np.percentile(prod, 75)),
            "rstar_median": float(np.median(rstar[sel])),
            "G_median": float(np.median(G[sel])),
            "gstar_over_G_median": float(np.median(dv["gstar"][sel] / G[sel])),
        }

    prod = rstar[ok] * G[ok]
    gmeds = [rstar[np.isfinite(rstar) & (tstar == t)] *
             G[np.isfinite(rstar) & (tstar == t)] for t in
             [21, 19, 189, 273, 247, 255, 98, 0, 171]
             if np.sum(np.isfinite(rstar) & (tstar == t)) >= 4]
    gmeds = [float(np.median(x)) for x in gmeds]
    gmargins = [float(m[t]) for t in [21, 19, 189, 273, 247, 255, 98, 0, 171]
                if np.sum(np.isfinite(rstar) & (tstar == t)) >= 4]

    # token-locked peak steps of the last-state response
    peak_steps = np.argmax(dv["dh"][:, -1, :], axis=1)
    pu, pc = np.unique(peak_steps, return_counts=True)
    peak_top = [{"step": int(pu[i]), "count": int(pc[i])}
                for i in np.argsort(-pc)[:10]]

    layer_table = []
    for li, L in enumerate(dv["layers"]):
        p = dv["dh"][:, li, :] / 0.1
        med = np.median(p, axis=0)
        rel = np.median(dv["dh"][:, li, :] / dv["norm_h0"][li][None, :],
                        axis=0)
        row = {
            "layer": int(L),
            "median_gain": float(np.median(dv["dh"][:, li, :] / 0.1)),
            "peak_step_of_median": int(np.argmax(med)) if L else 0,
            "peak_value": float(med.max()),
            "end_over_peak": float(med[-1] / med.max()) if med.max() > 0 else 0.0,
            "rel_median": float(np.median(dv["dh"][:, li, :] /
                                          dv["norm_h0"][li][None, :])),
            "rel_t0": float(rel[0]),
            "rel_t299": float(rel[-1]),
        }
        for tau in (0, 10, 50, 100, 258, 299):
            row[f"abs_t{tau}"] = float(med[tau])
        layer_table.append(row)

    # gain comparison final norm vs state 47 (24 directions)
    Gfn = np.median(cb["norms"], axis=1) / 0.1
    gain_compare = {
        "G47_median_24": float(np.median(dv["G47"][:cb["nvec"]])),
        "Gfn_median_24": float(np.median(Gfn)),
        "ratio_median": float(np.median(dv["G47"][:cb["nvec"]] / Gfn)),
        "ratio_p5": float(np.percentile(dv["G47"][:cb["nvec"]] / Gfn, 5)),
        "ratio_p95": float(np.percentile(dv["G47"][:cb["nvec"]] / Gfn, 95)),
        "pearson_log": safe_pearson(np.log(dv["G47"][:cb["nvec"]]),
                                    np.log(Gfn)),
    }

    state_cos = d["cos"][0, :, -1, :]
    numbers = {
        "run_dir": os.path.abspath(run_dir),
        "model": meta["model"],
        "dtype": meta["dtype"],
        "window": int(meta["window"]),
        "prompt_len": int(meta["prompt_len"]),
        "n_generated": int(T),
        "n_directions": int(K),
        "dirs_per_block": int(meta["dirs_per_block"]),
        "n_blocks": int(np.ceil(K / meta["dirs_per_block"])),
        "radius": 0.1,
        "layers": [int(x) for x in meta["layers"]],
        "baseline_margin": {
            "median": float(np.median(m)),
            "min": float(m.min()), "min_step": int(np.argmin(m)),
            "mean": float(m.mean()),
            "n_below_0.1": int((m < 0.1).sum()),
            "n_below_0.01": int((m < 0.01).sum()),
            "smallest_10": margin_table,
        },
        "controlling_steps": {
            "n_distinct_all": int(np.sum(allc > 0)),
            "n_distinct_flip": int(np.sum(flc > 0)),
            "top_all": counts_all,
            "flips_only_steps": [int(t) for t in np.unique(tstar[flip])],
            "flips_only_counts": [int(x) for x in
                                  np.unique(tstar[flip], return_counts=True)[1]],
            "top3_all_share": float(allc[np.argsort(-allc)[:3]].sum() / K),
            "top2_flip_share": float(flc[np.argsort(-flc)[:2]].sum() /
                                     max(int(flip.sum()), 1)),
        },
        "flips": {
            "n": int(flip.sum()), "fraction": float(flip.mean()),
            "rstar_all": qdict(rstar),
            "rstar_flip": qdict(rstar[flip]),
            "rstar_gt_1": int((rstar > 1).sum()),
            "rstar_gt_run": int((rstar > 0.1).sum()),
        },
        "sign_structure": {
            "steps": nsteps,
            "correlations": corr,
            "flip_frac_negative_at_21": float((dm[flip, 21] < 0).mean()),
            "flip_frac_negative_at_19": float((dm[flip, 19] < 0).mean()),
            "flip_frac_negative_at_189": float((dm[flip, 189] < 0).mean()),
        },
        "gain": {
            "G47": qdict(G),
            "G47_log10": qdict(np.log10(G)),
            "max_over_min": float(G.max() / G.min()),
            "mean_gain": qdict(np.mean(dv["dh"][:, -1, :], axis=1) / 0.1),
            "final_norm_24": qdict(Gfn),
            "compare_state47_vs_final": gain_compare,
        },
        "response_geometry": {
            "n_vectors": int(cb["nvec"]),
            "median_steps": [int(x) for x in cb["tmed"]],
            "vector_norms": qdict(cb["scale"]),
            "pairwise_cos": qdict(cb["pair"]),
            "frac_abs_cos_gt_0.1": float(np.mean(np.abs(cb["pair"]) > 0.1)),
            "within_block_cos_mean": float(cb["within"].mean()),
            "cross_block_cos_mean": float(cb["cross"].mean()),
            "input_within_block_cos": -1.0 / 19.0,
            "cos_to_mean_response": qdict(cb["cos_mean"]),
            "pca_explained": [float(x) for x in cb["pca"]],
            "pca_top1": float(cb["pca"][0]),
            "temporal_cos": {str(k): qdict(v) for k, v in cb["lags"].items()},
            "state_cosine_perturbed_vs_baseline": qdict(state_cos),
            "state_cosine_frac_exact_one": float((state_cos == 1.0).mean()),
        },
        "rstar_vs_gain": {
            "n_finite_below_1": int(ok.sum()),
            "pearson_loglog": safe_pearson(logr, logG),
            "spearman_loglog": safe_spearman(logr, logG),
            "pearson_logr_logmargin": safe_pearson(logr, lm),
            "r2_logmargin": safe_pearson(logr, lm) ** 2,
            "regression_coeffs": {"log_margin": float(beta[0]),
                                  "log_G": float(beta[1]),
                                  "intercept": float(beta[2])},
            "regression_r2": float(r2),
            "product_quantiles": qdict(prod),
            "product_log_std": float(np.std(np.log(prod))),
            "product_cv": float(prod.std() / prod.mean()),
            "product_p95_over_p5": float(np.percentile(prod, 95) /
                                         np.percentile(prod, 5)),
            "group_median_span": float(max(gmeds) / min(gmeds)),
            "group_margin_span": float(max(gmargins) / min(gmargins)),
            "groups": groups,
            "gstar": qdict(dv["gstar"]),
            "gstar_over_G": qdict(dv["gstar"] / G),
            "corr_log_gstar_logG": safe_pearson(np.log(dv["gstar"]),
                                                np.log(G)),
            "local_peak_gain_median": float(np.median(
                dv["dh"][:, -1, :].max(axis=1) / 0.1)),
            "step_gain": step_gain,
        },
        "layers_table": layer_table,
        "summary": {
            "abs_gain_state6_to_47": float(layer_table[8]["median_gain"] /
                                           layer_table[1]["median_gain"]),
            "rel_median_state6_to_47": float(layer_table[1]["rel_median"] /
                                             layer_table[8]["rel_median"]),
            "peak_state6": layer_table[1]["peak_value"],
            "peak_state47": layer_table[8]["peak_value"],
            "per_direction_peak_step_top": peak_top,
            "frac_peak_at_258": float(np.mean(peak_steps == 258)),
            "frac_peak_at_100": float(np.mean(peak_steps == 100)),
        },
        "notes": {
            "signed_vs_stored_rc": (
                "run_a1_big stores r_c with |dm|; this analysis recomputes the "
                "signed r_c^- with max(-dm,0). Unsigned threshold would call "
                "810 directions flips at r=0.1 instead of 464."),
            "state47_vs_final_norm": (
                "dh_norm[:,-1] is state 47 (output of block 46). dh_last "
                "vectors are final-norm differences (lm_head input) and are "
                "~4.06x smaller; they correlate 0.996 in log."),
            "state_angle_not_recoverable": (
                "Baseline state vectors were not stored, only norms, so the "
                "angle between a response vector and h0 cannot be computed. "
                "The stored state cosine cos(h',h0) is 1 to fp32 precision."),
            "layer0_zero": "dh_norm at layer 0 is exactly zero for every "
                           "direction and step.",
        },
    }
    return numbers


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results_big/ds14/a1_inter_w100")
    ap.add_argument("--tag", default="ds14")
    ap.add_argument("--outdir", default="notebook_big/figures")
    ap.add_argument("--radius", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d, meta = load_run(args.run)
    r = args.radius
    dv = derive(d, r)
    dv["norm_h0"] = d["norm_h0"].astype(np.float64)
    cb = cos_block(d, r)

    m = dv["margins"]
    margin_steps = [int(t) for t in np.argsort(m)[:10]]
    top_steps = [int(t) for t in np.argsort(-np.bincount(dv["tstar"],
                                                        minlength=dv["T"]))[:3]]

    fig_controlling(dv, margin_steps, top_steps,
                    os.path.join(args.outdir, f"{args.tag}_controlling_steps.png"))
    fig_sign(dv, [21, 19, 189],
             os.path.join(args.outdir, f"{args.tag}_sign_structure.png"))
    fig_gain(dv, cb,
             os.path.join(args.outdir, f"{args.tag}_gain_structure.png"))
    fig_shape(dv, os.path.join(args.outdir, f"{args.tag}_shape_by_layer.png"))
    fig_rstar(dv, os.path.join(args.outdir, f"{args.tag}_rstar_vs_gain.png"))

    numbers = build_numbers(d, meta, dv, cb, margin_steps, top_steps, args.run)
    jpath = os.path.join(args.outdir, f"{args.tag}_a1_deep_numbers.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(jf(numbers), f, indent=2)

    print("wrote:")
    for name in ("controlling_steps", "sign_structure", "gain_structure",
                 "shape_by_layer", "rstar_vs_gain"):
        print("  ", os.path.join(args.outdir, f"{args.tag}_{name}.png"))
    print("  ", jpath)
    print(f"K={dv['K']} T={dv['T']} flips={int(dv['flip'].sum())} "
          f"medianG={np.median(dv['G47']):.3f} "
          f"rho(log r*,log G)={numbers['rstar_vs_gain']['pearson_loglog']:+.3f}")


if __name__ == "__main__":
    main()
