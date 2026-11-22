#!/usr/bin/env python
"""State kicks, depth tangent spectrum and non-normality, DS-R1-14B (true W=100).

Reads, all local:
  results_big/ds14/state_w100/state2.npz   single-position kicks (2 radii x
                                           10 positions x 24 directions)
  results_big/ds14/depth_w100/depth.npz    tangent spectrum, 8 tokens x 48 blocks
  results_big/ds14/nonnorm_w100/nonnorm.npz MLP / full-block Jacobian metrics
  results_big/ds14/a1_inter_w100/a1.npz     baseline state norms only

Writes
  notebook_big/figures/ds14sd_state_kicks.png
  notebook_big/figures/ds14sd_depth_spectrum.png
  notebook_big/figures/ds14sd_nonnorm.png
  notebook_big/figures/ds14sd_numbers.json

Definitions
-----------
State kicks
  s              generated position at which the kick is injected
                 (embedding of token s, residual stream, layer 0)
  r              kick radius (0.03, 0.1)
  delta h^l_t    h^l_t(perturbed) - h^l_t(baseline), l = saved state
  immediate      ||delta h^47_s|| / r: response at the injection position and
                 the last saved state, median over the 24 random directions
  half-life      per direction, first t > s with ||delta h^47_t|| below half
                 ||delta h^47_s||; the figure plots the median over directions
  r_c^-(t,u)     m_t r / max(-delta m_t(u), 0): radius at which step t flips
  flip count     directions with min_{t>=s} r_c^-(t,u) <= 1

Depth
  lam[t,l,k]     (1/(l+1)) * sum_{b<=l} log|R_kk(b)| after QR each block.
                 It is the running average per-block exponent, not the local
                 gain and not the cumulative sum.
  e[t,l,k]       per-block exponent = (l+1)lam[l] - l lam[l-1] (e[0]=lam[0])
  h_depth        sum_i max(final lam_i, 0)
  local_gain     mean ||J q_i|| over the 8 orthonormal tangent directions
  norm_h0        baseline ||h^l_t|| at states 0,6,...,42,47 (a1 run)
  relative       e - d log||h||/dl, computed on 6-block chunks because the a1
                 run saved norms only every 6 blocks (plus state 47)

Non-normality
  sigma_max      top singular value of the block Jacobian
  rho            spectral radius (largest |eigenvalue|)
  kappa          sigma_max / rho (1 = normal)
  angle          principal angle between top singular directions of blocks
                 4 apart (last pair 44 -> 47)

No GPU, no torch.  Run from the repository root:
    python scripts_spark/figs_state_depth.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results_big", "ds14")
STATE = os.path.join(D, "state_w100", "state2.npz")
DEPTH = os.path.join(D, "depth_w100", "depth.npz")
NONNORM = os.path.join(D, "nonnorm_w100", "nonnorm.npz")
A1 = os.path.join(D, "a1_inter_w100", "a1.npz")
FIGDIR = os.path.join(ROOT, "notebook_big", "figures")

C_BLUE = "#1f4e79"
C_LBLUE = "#6baed6"
C_RED = "#c0392b"
C_ORANGE = "#e67e22"
C_GREEN = "#2e7d32"
C_PURPLE = "#6a3d9a"
C_GREY = "#6f6f6f"
C_DARK = "#222222"

# 1.5B reference numbers quoted in the brief
REF15 = dict(depth_h_low=0.6, depth_h_high=1.4,
             depth_lam_low=0.08, depth_lam_high=0.18,
             rel_worst=-0.034,
             nn_block0_sigma=(95.0, 195.0), nn_block0_rho=(11.0, 39.0),
             nn_block0_kappa=(5.0, 8.0),
             nn_mid_sigma=(2.7, 3.6), nn_mid_kappa=(1.7, 2.5),
             nn_last_sigma=(6.0, 9.0), nn_last_kappa=(3.0, 5.3),
             nn_last_mlp_sigma=(58.0, 93.0),
             angle_median=83.6)


# ------------------------------------------------------------------ helpers
def jf(o):
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = float(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    return o


def q(x, axis=0):
    x = np.asarray(x, float)
    return (np.nanmedian(x, axis=axis),
            np.nanpercentile(x, 25, axis=axis),
            np.nanpercentile(x, 75, axis=axis))


def band(ax, x, med, lo, hi, color, label=None, alpha=0.18, lw=1.7, z=3):
    ax.plot(x, med, color=color, lw=lw, zorder=z, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=z - 1)


# ------------------------------------------------------------- state metrics
def state_metrics():
    d = np.load(STATE)
    dh = d["dh"].astype(float)          # [r, s, u, layer, t]
    dm = d["dm"].astype(float)          # [r, s, u, t]
    m = d["margins"].astype(float)      # [t]
    steps = d["steps"].astype(int)
    radii = d["radii"].astype(float)
    layers = d["layers"].astype(int)
    li = int(np.where(layers == 47)[0][0])
    T = dh.shape[-1]

    nr, ns, K = dh.shape[0], dh.shape[1], dh.shape[2]
    imm = np.zeros((nr, ns))
    imm_lo = np.zeros((nr, ns))
    imm_hi = np.zeros((nr, ns))
    hl = np.full((nr, ns), np.nan)
    hl_frac = np.zeros((nr, ns))
    late_ratio = np.full((nr, ns), np.nan)
    flip = np.zeros((nr, ns), dtype=int)
    flip_early = np.zeros((nr, ns), dtype=int)
    flip_late = np.zeros((nr, ns), dtype=int)
    flip_297 = np.zeros((nr, ns), dtype=int)
    rc_min_med = np.full((nr, ns), np.nan)
    rc_min_min = np.full((nr, ns), np.nan)
    peak_t = np.zeros((nr, ns), dtype=int)

    for ri in range(nr):
        for si, s in enumerate(steps):
            v = dh[ri, si, :, li, s]     # immediate response, 24 directions
            imm[ri, si] = np.median(v) / radii[ri]
            imm_lo[ri, si] = np.percentile(v, 25) / radii[ri]
            imm_hi[ri, si] = np.percentile(v, 75) / radii[ri]

            per_dir = []
            per_late = []
            for k in range(K):
                traj = dh[ri, si, k, li, :]
                im = traj[s]
                later = np.where((np.arange(T) > s) & (traj < im / 2.0))[0]
                per_dir.append(later[0] - s if len(later) else np.nan)
                per_late.append(np.max(traj[s:]) / max(im, 1e-30))
            per_dir = np.asarray(per_dir, float)
            hl[ri, si] = np.nanmedian(per_dir) if np.isfinite(per_dir).any() else np.nan
            hl_frac[ri, si] = np.isfinite(per_dir).mean()
            late_ratio[ri, si] = np.median(per_late)

            # first later step at which some direction's threshold <= 1
            for k in range(K):
                dd = dm[ri, si, k, s:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rc = np.where(dd < 0, m[s:] * radii[ri] / (-dd), np.inf)
                mn = rc.min()
                rc_min_min[ri, si] = mn if not np.isfinite(rc_min_min[ri, si]) \
                    else min(rc_min_min[ri, si], mn)
                hit = np.where(rc <= 1.0)[0]
                if len(hit):
                    flip[ri, si] += 1
                    t = int(s + hit[0])
                    if t in (19, 20, 21):
                        flip_early[ri, si] += 1
                    elif t == 297:
                        flip_297[ri, si] += 1
                    else:
                        flip_late[ri, si] += 1
                if k == 0:
                    rc_min_med[ri, si] = np.nan
            rc_all = []
            for k in range(K):
                dd = dm[ri, si, k, s:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rc = np.where(dd < 0, m[s:] * radii[ri] / (-dd), np.inf)
                rc_all.append(rc.min())
            rc_min_med[ri, si] = np.median(rc_all)

            vmed = np.median(dh[ri, si, :, li, :], axis=0)
            peak_t[ri, si] = int(np.argmax(vmed[s:]) + s)

    return dict(dh=dh, dm=dm, m=m, steps=steps, radii=radii, layers=layers,
                T=T, K=K, imm=imm, imm_lo=imm_lo, imm_hi=imm_hi,
                hl=hl, hl_frac=hl_frac, late_ratio=late_ratio,
                flip=flip, flip_early=flip_early, flip_late=flip_late,
                flip_297=flip_297, rc_min_med=rc_min_med,
                rc_min_min=rc_min_min, peak_t=peak_t)


# ------------------------------------------------------------- depth metrics
def depth_metrics():
    d = np.load(DEPTH)
    lam = d["lam"].astype(float)        # [token, 48, k]
    lg = d["local_gain"].astype(float)  # [token, 48]
    hd = d["h_depth"].astype(float)
    tok = d["tokens"].astype(int)
    nl = lam.shape[1]
    l = np.arange(nl)

    cum = lam * (l + 1)[None, :, None]
    e = np.empty_like(lam)
    e[:, 0, :] = cum[:, 0, :]
    e[:, 1:, :] = cum[:, 1:, :] - cum[:, :-1, :]
    eb = e.mean(2)                      # mean over k, per block

    a = np.load(A1)
    nh = a["norm_h0"].astype(float)     # [9, 300]
    al = a["layers"].astype(int)

    norms = np.zeros((len(tok), len(al)))
    for ti, t in enumerate(tok):
        norms[ti] = nh[:, t]

    # 6-block chunks between saved states (the only state-normalization
    # possible with norm_h0 at 0,6,...,42,47)
    chunks = []
    for i in range(len(al) - 1):
        L0, L1 = int(al[i]), int(al[i + 1])
        c0 = cum[:, L0 - 1, :].mean(1) if L0 > 0 else np.zeros(len(tok))
        c1 = cum[:, L1 - 1, :].mean(1)
        te = (c1 - c0) / (L1 - L0)
        ne = np.log(norms[:, i + 1] / norms[:, i]) / (L1 - L0)
        chunks.append(dict(l0=L0, l1=L1, blocks=L1 - L0,
                           tangent=te, norm=ne, relative=te - ne))
    rel = np.stack([c["relative"] for c in chunks], 1)  # [token, chunk]
    tang_chunk = np.stack([c["tangent"] for c in chunks], 1)
    norm_chunk = np.stack([c["norm"] for c in chunks], 1)
    chunk_center = np.array([(c["l0"] + c["l1"] - 1) / 2 for c in chunks])

    return dict(lam=lam, lg=lg, hd=hd, tok=tok, nl=nl, cum=cum, e=e, eb=eb,
                norms=norms, al=al, chunks=chunks, rel=rel,
                tang_chunk=tang_chunk, norm_chunk=norm_chunk,
                chunk_center=chunk_center,
                total_growth=np.exp(cum[:, -1, :].mean(1)),
                norm_growth=norms[:, -1] / norms[:, 0])


# ---------------------------------------------------------- nonnorm metrics
def nonnorm_metrics():
    d = np.load(NONNORM)
    tok = d["tokens"].astype(int)
    sel = d["sel_layers"].astype(int)
    angle_layers = d["angle_layers"].astype(int)
    meta = json.load(open(os.path.join(os.path.dirname(NONNORM), "meta.json")))
    full_layers = np.asarray(meta["full_layers"], int)
    return dict(tok=tok, sel=sel, full_layers=full_layers,
                angle_layers=angle_layers,
                ms=d["mlp_smax"].astype(float),
                mr=d["mlp_rho"].astype(float),
                mk=d["mlp_kappa"].astype(float),
                fs=d["full_smax"].astype(float),
                fr=d["full_rho"].astype(float),
                fk=d["full_kappa"].astype(float),
                ang=d["angles"].astype(float))


# ------------------------------------------------------------------ figure 1
def fig_state(S):
    steps, radii = S["steps"], S["radii"]
    fig = plt.figure(figsize=(13.2, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

    # (a) immediate response vs s
    ax = fig.add_subplot(gs[0, 0])
    cols = [C_ORANGE, C_BLUE]
    for ri, r in enumerate(radii):
        ax.errorbar(steps, S["imm"][ri], yerr=[S["imm"][ri] - S["imm_lo"][ri],
                                               S["imm_hi"][ri] - S["imm"][ri]],
                    fmt="o" if ri else "s", ms=5.5, lw=1.4, capsize=0,
                    color=cols[ri], mfc="white" if ri == 0 else cols[ri],
                    label=fr"$r={r:g}$")
    ax.set_yscale("log")
    ax.set_xticks(steps)
    ax.set_xlabel("generated position $s$ where the kick is injected")
    ax.set_ylabel(r"$\Vert\delta h^{47}_s\Vert\,/\,r$   (state units per unit radius)")
    ax.set_title("(a) the immediate response depends on where the kick lands")
    ax.annotate("largest: $s=50$", xy=(50, S["imm"][1, 3]), xytext=(74, 14),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))
    ax.annotate("smallest: $s=32$", xy=(32, S["imm"][1, 2]), xytext=(58, 2.15),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))
    ax.legend(loc="upper right", title="kick radius")

    # (b) initial half-life, with the late re-amplification ratio
    ax = fig.add_subplot(gs[0, 1])
    axb = ax.twinx()
    x = np.arange(len(steps))
    bars = []
    for si, s in enumerate(steps):
        h = S["hl"][1, si]
        if np.isfinite(h):
            b = ax.bar(x[si], h, width=0.68, color=C_LBLUE,
                       edgecolor=C_BLUE, lw=0.8,
                       hatch="///" if S["hl_frac"][1, si] < 1 else None)
        else:
            b = ax.bar(x[si], 0.0, width=0.68, color="white",
                       edgecolor=C_BLUE, lw=0.8, hatch="xxx")
        bars.append(b)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("kick position $s$")
    ax.set_ylabel("steps until the response first falls below half\nits value at $s$ (median over directions)")
    ax.set_ylim(0, 32)
    ax.set_title("(b) usually 1-5 steps, but late kicks re-amplify")
    axb.semilogy(x, S["late_ratio"][1], "v", ms=6, color=C_RED, ls="none",
                 label="largest later response / response at $s$")
    axb.set_ylabel("largest later response / immediate (x)", color=C_RED)
    axb.tick_params(axis="y", colors=C_RED)
    axb.set_ylim(0.8, 250)
    axb.spines["right"].set_visible(True)
    axb.text(0.98, 0.90, "red triangles: largest later\nresponse / immediate",
             transform=axb.transAxes, ha="right", va="top", fontsize=7.4,
             color=C_RED)
    ax.annotate("4 of 24 directions\ndecay at all", xy=(8, 6.2), xytext=(4.6, 14),
                fontsize=7.6, color=C_GREY)
    ax.annotate("no later step", xy=(9, 0.4), xytext=(7.4, 9.0),
                fontsize=7.6, color=C_GREY)
    ax.text(0.02, 0.98, "hatched = not every direction crosses half",
            transform=ax.transAxes, va="top", fontsize=7.4, color=C_GREY)

    # (c) which later tie is crossed
    ax = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(steps))
    cats = [(S["flip_early"][1], C_RED, "step 19 or 21 (razor ties)"),
            (S["flip_late"][1], C_ORANGE, "step 145-276 (late burst / ties)"),
            (S["flip_297"][1], C_PURPLE, "step 297 (razor tie)")]
    for vals, col, lab in cats:
        frac = vals / S["K"]
        ax.bar(x, frac, bottom=bottom, width=0.68, color=col, label=lab)
        bottom += frac
    ax.bar(x, 1.0 - bottom, bottom=bottom, width=0.68, color="#dddddd",
           label="no flip at any radius $\\leq 1$")
    ax.plot(x, S["flip"][1] / S["K"], "o", ms=5, color=C_DARK, zorder=5,
            label="total")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("kick position $s$")
    ax.set_ylabel("fraction of the 24 directions")
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) a kick before step 21 can flip the razor tie 13 steps later")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3,
              fontsize=7.4, columnspacing=1.2, handlelength=1.4)

    # (d) response trajectories
    ax = fig.add_subplot(gs[1, 1])
    show = [0, 3, 6, 8]
    cols2 = [C_LBLUE, C_BLUE, C_GREEN, C_RED]
    for si, col in zip(show, cols2):
        v = np.median(S["dh"][1, si, :, 4, :], axis=0)
        ax.semilogy(np.arange(steps[si], S["T"]), v[steps[si]:], lw=1.4,
                    color=col, label=f"$s={steps[si]}$")
        ax.axvline(steps[si], color=col, ls=":", lw=0.8, alpha=0.8)
    ax.axvline(291, color=C_GREY, ls="--", lw=1.0)
    ax.text(205, 25, "late burst\n$t\\approx291$", ha="left", fontsize=7.8,
            color=C_GREY)
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel(r"$\Vert\delta h^{47}_t\Vert$   (median over directions)")
    ax.set_title("(d) the response decays, then bursts again near the end")
    ax.legend(loc="upper left", title="kick at $s$", ncol=2)

    fig.suptitle("The 14B absorbs a single-position kick in a few steps, but its "
                 "influence survives and re-amplifies at the late near-ties",
                 fontsize=12.2, y=0.995)
    p = os.path.join(FIGDIR, "ds14sd_state_kicks.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 2
def fig_depth(Dm):
    lam, tok, nl = Dm["lam"], Dm["tok"], Dm["nl"]
    l = np.arange(nl)
    fig = plt.figure(figsize=(15.2, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.26)

    # (a),(b) running-average spectra, four tokens each
    groups = [(slice(0, 4), "(a) tokens 0-100: block 0 dominates, then a slow decay"),
              (slice(4, 8), "(b) tokens 150-299: quieter, but same shape")]
    colors = [C_LBLUE, C_RED, C_GREEN, C_BLUE,
              C_ORANGE, C_PURPLE, C_GREY, C_DARK]
    for gi, (sl, title) in enumerate(groups):
        ax = fig.add_subplot(gs[0, gi])
        for j, ti in enumerate(range(sl.start, sl.stop)):
            med, lo, hi = q(lam[ti], axis=1)
            col = colors[ti]
            band(ax, l, med, lo, hi, col, label=f"token {tok[ti]}",
                 alpha=0.14, lw=1.6)
        ax.set_yscale("log")
        ax.set_xlim(-1, 48)
        ax.set_ylim(0.012, 4.0)
        ax.set_xlabel("block index $l$   (0 = first block)")
        ax.set_ylabel(r"$\lambda_i(l)$   (running mean log gain per block)")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=1, fontsize=8)
    # highlight the tie token
    ax0 = fig.axes[0]
    ax0.text(16, 0.30, "token 21 (the razor tie)\nhas the highest curve",
             fontsize=8, color=C_RED)

    # (c) h_depth
    ax = fig.add_subplot(gs[0, 2])
    xx = np.arange(len(tok))
    cols = [C_RED if t == 21 else C_BLUE for t in tok]
    ax.bar(xx, Dm["hd"], color=cols, width=0.66)
    ax.axhspan(REF15["depth_h_low"], REF15["depth_h_high"], color=C_GREY,
               alpha=0.14, lw=0)
    ax.text(7.4, REF15["depth_h_low"] + 0.06, "1.5B range 0.6-1.4",
            fontsize=7.8, color=C_GREY, ha="right")
    ax.set_xticks(xx)
    ax.set_xticklabels([str(t) for t in tok], rotation=0)
    ax.set_xlabel("probe token $t$")
    ax.set_ylabel(r"$h_{\rm depth}=\sum_i\max(\lambda_i,0)$   (nats)")
    ax.set_ylim(0, 1.55)
    ax.set_title("(c) the tie token at 21 grows most")
    for i, v in enumerate(Dm["hd"]):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=7.6)
    ax.text(0.02, 0.80, "median $\\lambda$ at the last block:\n"
            "0.02-0.09 nats/block (1.5B: 0.08-0.18)",
            transform=ax.transAxes, fontsize=7.8, color=C_GREY)

    # (d) local gain and state-norm growth per block
    ax = fig.add_subplot(gs[1, 0])
    med, lo, hi = q(Dm["lg"], axis=0)
    band(ax, l, med, lo, hi, C_BLUE, label="tangent gain (8 directions)", alpha=0.16)
    ax.plot(Dm["chunk_center"], np.exp(Dm["norm_chunk"].mean(0)), "s-",
            color=C_RED, ms=4, lw=1.5, label="state-norm growth (6-block chunks)")
    ax.axhline(1.0, color=C_GREY, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlim(-1, 48)
    ax.set_ylim(0.85, 15)
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("growth factor per block   (x)")
    ax.set_title("(d) tangent gain near 1, state norm grows")
    ax.legend(loc="upper right")
    ax.annotate("block 0:\n3.2-11.1x", xy=(0, med[0]), xytext=(7, 6.5),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))
    ax.text(0.30, 0.42, "48 blocks:\ntangent volume $\\times$3-46\n"
            "state norm $\\times$655-833", transform=ax.transAxes,
            fontsize=8, color=C_DARK)

    # (e) absolute vs relative per-block exponent
    ax = fig.add_subplot(gs[1, 1])
    mede, loe, hie = q(Dm["eb"], axis=0)
    ax.plot(l[1:], mede[1:], color=C_GREY, lw=1.6,
            label="absolute, per block (median over tokens)")
    ax.fill_between(l[1:], loe[1:], hie[1:], color=C_GREY, alpha=0.16, lw=0)
    rel_med, rel_lo, rel_hi = q(Dm["rel"], axis=0)
    ax.errorbar(Dm["chunk_center"], rel_med,
                yerr=[rel_med - rel_lo, rel_hi - rel_med],
                fmt="o", ms=5, lw=1.4, capsize=2.5, color=C_BLUE,
                label="relative to state-norm growth, 6-block chunks")
    ax.axhline(0.0, color=C_DARK, lw=1)
    ax.axhline(REF15["rel_worst"], color=C_RED, ls=":", lw=1.2)
    ax.text(46.8, REF15["rel_worst"] + 0.006, "1.5B least-negative -0.034",
            fontsize=7.4, color=C_RED, ha="right")
    ax.set_xlim(-1, 48)
    ax.set_ylim(-0.16, 0.24)
    ax.set_xlabel("block index $l$   (relative points at chunk centres)")
    ax.set_ylabel("log gain per block   (nats/block)")
    ax.set_title("(e) normalized, 61 of 64 chunks contract")
    ax.text(38, 0.075, "absolute", fontsize=8, color=C_GREY, ha="center")
    ax.text(11, -0.115, "relative (6-block)", fontsize=8, color=C_BLUE)
    ax.text(0.02, 0.70, "block 0 absolute: +1.22\n(range +0.98 to +2.40, off axis)",
            transform=ax.transAxes, fontsize=7.6, color=C_GREY)

    # (f) final spectrum per token
    ax = fig.add_subplot(gs[1, 2])
    kk = np.arange(1, 9)
    for ti, t in enumerate(tok):
        col = C_RED if t == 21 else C_GREY
        ax.plot(kk, lam[ti, -1], "-o", ms=3.4, lw=1.0 if t != 21 else 2.0,
                color=col, alpha=1.0 if t == 21 else 0.55,
                label=f"token {t}" if t == 21 else None)
    ax.axhspan(REF15["depth_lam_low"], REF15["depth_lam_high"],
               color=C_GREY, alpha=0.14, lw=0)
    ax.text(1.0, 0.155, "1.5B 0.08-0.18", fontsize=7.6, color=C_GREY)
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0.0, 0.20)
    ax.set_xlabel("tangent direction $i$   (1 = largest gain)")
    ax.set_ylabel(r"final $\lambda_i$   (nats/block)")
    ax.set_title("(f) flat spectra; only token 21 reaches the 1.5B band")
    ax.legend(loc="upper right")

    fig.suptitle("Absolute tangent growth is small and mostly one block plus a "
                 "late tail; divided by state growth, the tangent contracts",
                 fontsize=12.2, y=1.00)
    p = os.path.join(FIGDIR, "ds14sd_depth_spectrum.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ figure 3
def fig_nonnorm(N):
    fig = plt.figure(figsize=(13.2, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

    # (a) MLP sigma_max and rho
    ax = fig.add_subplot(gs[0, 0])
    for arr, col, lab, mk in [(N["ms"], C_BLUE, r"$\sigma_{\max}$", "o"),
                              (N["mr"], C_ORANGE, r"$\rho$", "s")]:
        med, lo, hi = q(arr[:, N["sel"]], axis=0)
        x = N["sel"]
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt=mk + "-", ms=5,
                    lw=1.5, capsize=2.5, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(-2, 49)
    ax.set_ylim(1.0, 400)
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("Jacobian gain   (x)")
    ax.set_title("(a) MLP block gain: quiet middle, huge last block")
    ax.legend(loc="upper left")
    ax.annotate("last block\n$\\sigma_{\\max}$ 94-147", xy=(47, 118),
                xytext=(30, 270), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))

    # (b) kappa, MLP and full
    ax = fig.add_subplot(gs[0, 1])
    med, lo, hi = q(N["mk"][:, N["sel"]], axis=0)
    ax.errorbar(N["sel"], med, yerr=[med - lo, hi - med], fmt="o-", ms=5,
                lw=1.6, capsize=2.5, color=C_BLUE, label="MLP block (9 blocks)")
    med, lo, hi = q(N["fk"][:, N["full_layers"]], axis=0)
    ax.errorbar(N["full_layers"], med, yerr=[med - lo, hi - med], fmt="s--",
                ms=5, lw=1.3, capsize=2.5, color=C_RED,
                label="full block (5 blocks)")
    ax.fill_between([6, 40], REF15["nn_mid_kappa"][0], REF15["nn_mid_kappa"][1],
                    color=C_GREY, alpha=0.14, lw=0)
    ax.fill_between([44, 48], REF15["nn_last_kappa"][0], REF15["nn_last_kappa"][1],
                    color=C_PURPLE, alpha=0.14, lw=0)
    ax.text(43.5, 1.55, "1.5B mid 1.7-2.5", fontsize=7.4, color=C_GREY, ha="right")
    ax.text(38.0, 5.6, "1.5B last 3-5.3", fontsize=7.2, color=C_PURPLE,
            ha="center")
    ax.set_xlim(-2, 49)
    ax.set_ylim(1.0, 12.0)
    ax.set_yticks([1, 2, 3, 5, 7, 10, 12])
    ax.set_xlabel("block index $l$")
    ax.set_ylabel(r"$\kappa=\sigma_{\max}/\rho$")
    ax.set_title("(b) non-normality is U-shaped; last block is worse")
    ax.legend(loc="upper center", ncol=2)

    # (c) sigma vs rho, full blocks
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap("viridis")
    fl = N["full_layers"]
    for j, l in enumerate(fl):
        col = cmap(j / max(len(fl) - 1, 1))
        ax.loglog(N["fr"][:, l], N["fs"][:, l], "o", ms=7, color=col,
                  alpha=0.9, label=f"block {l}")
    rr = np.array([1.2, 40.0])
    for kap, ls, xl in [(1.0, "-", 28.0), (3.0, "--", 28.0), (10.0, ":", 19.0)]:
        ax.loglog(rr, kap * rr, ls, color=C_GREY, lw=1.0)
        ax.text(xl, kap * xl * 1.25, f"$\\kappa={kap:g}$", fontsize=7.4,
                color=C_GREY, ha="left", va="bottom")
    ax.set_xlim(1.2, 40)
    ax.set_ylim(2.0, 320)
    ax.set_xticks([2, 3, 5, 10, 20, 30])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "30"])
    ax.set_yticks([3, 10, 30, 100, 300])
    ax.set_yticklabels(["3", "10", "30", "100", "300"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"spectral radius $\rho$   (largest $|$eigenvalue$|$)")
    ax.set_ylabel(r"$\sigma_{\max}$   (top singular value)")
    ax.set_title("(c) the last block amplifies transients far above its eigenvalues")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.annotate("block 0", xy=(12.7, 96), xytext=(5.0, 135), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))
    ax.annotate("block 47", xy=(1.7, 14.2), xytext=(2.1, 25), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GREY))

    # (d) principal angles
    ax = fig.add_subplot(gs[1, 1])
    ang = N["ang"].ravel()
    ax.hist(ang, bins=np.arange(65, 91.5, 2.0), color=C_LBLUE,
            edgecolor="white", lw=0.6)
    med_all = float(np.nanmedian(ang))
    ax.axvline(med_all, color=C_BLUE, lw=1.8,
               label=f"14B median {med_all:.1f}$^\\circ$")
    ax.axvline(REF15["angle_median"], color=C_RED, ls="--", lw=1.5,
               label=f"1.5B median {REF15['angle_median']:.1f}$^\\circ$")
    for i, t in enumerate(N["tok"]):
        mt = float(np.nanmedian(N["ang"][i]))
        ax.plot([mt], [0.8], marker="|", ms=11, mew=1.8,
                color=C_DARK, alpha=0.85)
    ax.set_xlim(65, 91)
    ax.set_ylim(0, 22)
    ax.set_xlabel("angle between top singular directions of blocks 4 apart  (deg)")
    ax.set_ylabel("number of block pairs")
    ax.set_title("(d) consecutive blocks point nearly orthogonally")
    ax.legend(loc="upper left")
    ax.text(0.02, 0.60, "per-token medians: 84.1, 87.8, 87.7, 87.0, 85.4$^\\circ$\n"
            "(1.5B: 83.6$^\\circ$)", transform=ax.transAxes, fontsize=8,
            color=C_GREY)

    fig.suptitle("Non-normal at both ends, normal in the middle: the same U-shape "
                 "as the 1.5B, with a more extreme last block", fontsize=12.2, y=0.995)
    p = os.path.join(FIGDIR, "ds14sd_nonnorm.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------- main
def main():
    S = state_metrics()
    Dm = depth_metrics()
    N = nonnorm_metrics()

    p1 = fig_state(S)
    p2 = fig_depth(Dm)
    p3 = fig_nonnorm(N)

    # ------------------------------------------------------------ JSON
    steps = S["steps"]
    radii = S["radii"]
    out = {
        "protocol": {
            "model": "DeepSeek-R1-Distill-Qwen-14B",
            "blocks": 48, "hidden": 5120, "dtype": "float32",
            "window": 100, "prompt": "interstellar_travel", "prompt_len": 65,
            "T": 300, "baseline": "greedy, TRUE-window-100 A1 tokens, teacher forced",
            "state_kick_files": "results_big/ds14/state_w100/state2.npz",
            "depth_file": "results_big/ds14/depth_w100/depth.npz",
            "nonnorm_file": "results_big/ds14/nonnorm_w100/nonnorm.npz",
            "norms_file": "results_big/ds14/a1_inter_w100/a1.npz (norm_h0)",
            "state_kick_setup": "one random unit direction in the 5120-d residual stream "
                                "at the embedding of generated position s; 24 directions; "
                                "r=0.03 and 0.1; continuation teacher-forced on the true "
                                "window-100 baseline tokens",
            "depth_setup": "8 orthonormal tangent vectors at one generated token, "
                           "JVP through each of 48 blocks, QR renormalized each block",
            "nonnorm_setup": "MLP analytic SwiGLU Jacobian at 5 tokens x 9 blocks; "
                             "full-block power iteration + Arnoldi at 5 tokens x 5 blocks; "
                             "principal angles between blocks 4 apart",
            "tie_margins": {"t19": float(S["m"][19]), "t21": float(S["m"][21]),
                            "t297": float(S["m"][297])},
        },
        "state_kicks": {
            "steps": steps, "radii": radii,
            "immediate_median_over_dirs_over_r": S["imm"],
            "immediate_iqr_lo_over_r": S["imm_lo"],
            "immediate_iqr_hi_over_r": S["imm_hi"],
            "half_life_median_dirs": S["hl"],
            "half_life_fraction_dirs_defined": S["hl_frac"],
            "late_peak_over_immediate": S["late_ratio"],
            "flip_count_r01": S["flip"][1],
            "flip_count_r003": S["flip"][0],
            "flip_fraction_r01": S["flip"][1] / S["K"],
            "flip_fraction_r003": S["flip"][0] / S["K"],
            "flip_earliest_19_21_r01": S["flip_early"][1],
            "flip_earliest_145_276_r01": S["flip_late"][1],
            "flip_earliest_297_r01": S["flip_297"][1],
            "min_rc_median_dirs_r01": S["rc_min_med"][1],
            "min_rc_min_dirs_r01": S["rc_min_min"][1],
            "global_peak_step_median_curve_r01": S["peak_t"][1],
            "dm_linearity_ratio_median": float(np.median(np.abs(S["dm"][1]) /
                                                          np.abs(S["dm"][0]))),
        },
        "depth": {
            "tokens": Dm["tok"],
            "h_depth": Dm["hd"],
            "final_lambda": Dm["lam"][:, -1, :],
            "final_lambda_median": np.median(Dm["lam"][:, -1, :], axis=1),
            "lambda_running_mean_median_over_k": np.median(Dm["lam"], axis=2),
            "local_gain_median_over_k": np.median(Dm["lg"], axis=0),
            "local_gain_median_over_tokens": np.median(Dm["lg"], axis=0),
            "absolute_exponent_per_block_median_over_tokens": np.median(Dm["eb"], axis=0),
            "chunks": [dict(l0=c["l0"], l1=c["l1"],
                            tangent_median=float(np.median(c["tangent"])),
                            norm_median=float(np.median(c["norm"])),
                            relative_median=float(np.median(c["relative"])),
                            relative_min=float(np.min(c["relative"])),
                            relative_max=float(np.max(c["relative"]))) for c in Dm["chunks"]],
            "relative_per_token_chunk": Dm["rel"],
            "n_contract_cells": int((Dm["rel"] < 0).sum()),
            "n_cells": int(Dm["rel"].size),
            "relative_median_by_token": np.median(Dm["rel"], axis=1),
            "relative_overall_by_token": Dm["rel"].mean(1),
            "total_tangent_growth": Dm["total_growth"],
            "state_norm_growth": Dm["norm_growth"],
            "normalized_total_growth": Dm["total_growth"] / Dm["norm_growth"],
            "norms": Dm["norms"],
            "norm_layers": Dm["al"],
            "normalization_note": "norm_h0 is saved only at states 0,6,...,42,47, "
                                  "so the relative exponent is a 6-block average; "
                                  "per-block state normalization is not possible",
        },
        "nonnorm": {
            "tokens": N["tok"],
            "mlp_layers": N["sel"],
            "full_layers": N["full_layers"],
            "angle_layers": N["angle_layers"],
            "mlp_smax": N["ms"][:, N["sel"]],
            "mlp_rho": N["mr"][:, N["sel"]],
            "mlp_kappa": N["mk"][:, N["sel"]],
            "full_smax": N["fs"][:, N["full_layers"]],
            "full_rho": N["fr"][:, N["full_layers"]],
            "full_kappa": N["fk"][:, N["full_layers"]],
            "angles": N["ang"],
            "angle_median_per_token": np.nanmedian(N["ang"], axis=1),
            "angle_median_overall": float(np.nanmedian(N["ang"])),
        },
        "reference_1p5B": REF15,
    }
    jp = os.path.join(FIGDIR, "ds14sd_numbers.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(jf(out), fh, indent=1)

    # ---------------------------------------------------------- printout
    print("=== state kicks ===")
    print("s       imm/r r=.03   imm/r r=.1   HL(med,frac)   flip/24  min rc med")
    for si, s in enumerate(steps):
        print(f"{s:3d}  {S['imm'][0, si]:10.2f}  {S['imm'][1, si]:10.2f}   "
              f"{S['hl'][1, si]!s:>5} {S['hl_frac'][1, si]:4.2f}   "
              f"{S['flip'][1, si]:2d}/24   {S['rc_min_med'][1, si]:.4g}")
    print("flip categories r=.1 (s: early19/21, late145-276, 297, none):")
    for si, s in enumerate(steps):
        print(f"  s={s:3d}: {S['flip_early'][1, si]:2d} / "
              f"{S['flip_late'][1, si]:2d} / {S['flip_297'][1, si]:2d} / "
              f"{S['K'] - S['flip'][1, si]:2d}")
    print()
    print("=== depth ===")
    for ti, t in enumerate(Dm["tok"]):
        print(f"token {t:3d}: h_depth {Dm['hd'][ti]:.3f}  median final lam "
              f"{np.median(Dm['lam'][ti, -1]):.3f}  total growth "
              f"{Dm['total_growth'][ti]:.2f}x  norm {Dm['norm_growth'][ti]:.0f}x  "
              f"relative mean {Dm['rel'][ti].mean():+.3f}")
    print("relative chunks (median over tokens):",
          " ".join(f"{v:+.3f}" for v in np.median(Dm["rel"], axis=0)))
    print(f"contracting cells {(Dm['rel'] < 0).sum()}/{Dm['rel'].size}; "
          f"positive values {sorted(Dm['rel'][Dm['rel'] > 0].round(4).tolist())}")
    print()
    print("=== non-normality ===")
    for l in N["sel"]:
        i = int(np.where(N["sel"] == l)[0][0])
        print(f"MLP  l={l:2d}: sigma {np.median(N['ms'][:, l]):7.2f} "
              f"({N['ms'][:, l].min():.2f}-{N['ms'][:, l].max():.2f})  "
              f"rho {np.median(N['mr'][:, l]):6.2f}  kappa "
              f"{np.median(N['mk'][:, l]):5.2f}")
    for l in N["full_layers"]:
        print(f"FULL l={l:2d}: sigma {np.median(N['fs'][:, l]):7.2f}  "
              f"rho {np.median(N['fr'][:, l]):6.2f}  kappa "
              f"{np.median(N['fk'][:, l]):5.2f}")
    print("angle medians per token:", np.round(np.nanmedian(N["ang"], axis=1), 2))
    print("angle median overall:", float(np.nanmedian(N["ang"])))
    print()
    print("wrote", p1)
    print("wrote", p2)
    print("wrote", p3)
    print("wrote", jp)


if __name__ == "__main__":
    main()
