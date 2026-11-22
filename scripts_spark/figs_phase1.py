#!/usr/bin/env python
"""Phase-1 figure builder for the large-model perturbation experiments.

Reads an ``a1.npz`` produced by ``src/chaos_llm/run_a1_big.py`` (optionally
``free_run.npz`` from ``run_free_big.py``) and writes the seven phase-1 figures.

Conventions used here (see ``results/subagent_threshold2/common2.py``):

* ``margins[t]``            -- top-1 minus top-2 logit gap (nats) on the greedy run.
* ``dm[ri,d,t]``            -- change of that margin when the prompt embedding is
                               perturbed by ``radii[ri]`` along direction ``d``,
                               teacher-forced on the unperturbed tokens.
* A flip at step t needs ``m_t + dm_t <= 0``; positive dm can never flip.
* signed threshold radius   ``r_c_minus(t,d) = r * m_t / max(-dm_t, 0)``
  (``+inf`` where dm >= 0).  A direction's local threshold is
  ``r_* = min_t r_c_minus``; it is predicted to flip at radius r iff r_* <= r.
* predicted first flip      first t with ``r_c_minus <= r``, else -1.
* free run obs              first generated step at which the perturbed run
                            differs from its r=0 reference, -1 if never.

The response is linear in the radius, so big-model ensembles store a single
radius.  Everything below works with ``nr == 1`` and with the multi-radius
fixture; when a requested radius is not stored, the response is taken from the
reference radius and scaled linearly.

Run:
    python scripts_spark/figs_phase1.py --run <dir> [--free <dir|npz>] \
        [--tag ds14] [--outdir notebook_big/figures]

No GPU / torch is needed: the inputs are npz/json.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
import warnings

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, NullFormatter

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
    "legend.frameon": False,
})

C_MED = "#1f4e79"
C_BAND = "#7fb3d5"
C_OBS = "#c0392b"
C_PRED = "#1f4e79"
C_ALT = "#e67e22"
C_THIRD = "#2e8b57"


# ----------------------------------------------------------------- io
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_free_npz(path):
    """Accept a directory (free_run.npz inside) or a direct .npz path."""
    if path is None:
        return None
    if os.path.isdir(path):
        p = os.path.join(path, "free_run.npz")
        return p if os.path.exists(p) else None
    return path if os.path.exists(path) else None


def load_free(path):
    """Load free-run obs/pred, tolerating the schema variants seen locally.

    Primary schema (run_free_big.py):
        obs (n_r, K_free), pred (n_r, K_free), radii (n_r,), dirs (K_free,)
    ``dirs`` may be absent (all directions, in order) and ``pred`` may be
    absent (we recompute the signed prediction ourselves anyway).
    Sequences in ``ref_ids``/``gen_ids`` are not required.
    """
    npz = find_free_npz(path)
    if npz is not None:
        z = np.load(npz, allow_pickle=True)
        if "obs" not in z:
            warnings.warn(f"{npz}: no 'obs' array; free validation skipped")
            return None
        obs = np.asarray(z["obs"], dtype=int)
        radii = np.asarray(z["radii"], dtype=float).ravel()
        dirs = np.asarray(z["dirs"], dtype=int).ravel() if "dirs" in z else None
        return dict(path=npz, radii=radii, obs=obs, dirs=dirs, kind="npz",
                    pred=np.asarray(z["pred"], int) if "pred" in z else None)

    # fallback: a directory whose summary.json carries obs/pred lists
    if path is not None and os.path.isdir(path):
        sj = os.path.join(path, "summary.json")
        if os.path.exists(sj):
            j = load_json(sj)
            obs = np.asarray(j["obs"], dtype=int)
            radii = np.asarray(j["radii"], dtype=float).ravel()
            dirs = np.asarray(j["dirs"], dtype=int).ravel() if "dirs" in j else None
            return dict(path=sj, radii=radii, obs=obs, dirs=dirs, kind="json",
                        pred=np.asarray(j["pred"], int) if "pred" in j else None)
    warnings.warn(f"{path}: no free_run.npz / summary.json with obs; free validation skipped")
    return None


# ----------------------------------------------------------------- model
def _match_radius(radii, r):
    """Index of a stored radius equal to r (within 1e-6 relative), else None."""
    hit = np.where(np.isclose(radii, r, rtol=1e-6, atol=0))[0]
    return int(hit[0]) if len(hit) else None


def choose_gain_index(radii):
    """Reference radius for the linear gain dm/r (closest to 0.03 on log scale).

    Tiny radii suffer float32 cancellation; large ones are already nonlinear.
    """
    if len(radii) == 1:
        return 0
    pos = np.where(np.asarray(radii) > 0)[0]
    if not len(pos):
        return 0
    return int(pos[np.argmin(np.abs(np.log(np.asarray(radii)[pos] / 0.03)))])


def choose_run_radius(radii, meta, cli_radius):
    """Radius the figures describe: explicit CLI > single meta/npz radius > 0.3."""
    if cli_radius is not None:
        return float(cli_radius)
    mradii = meta.get("radii") if meta else None
    if mradii is not None and len(np.atleast_1d(mradii)) == 1:
        return float(np.atleast_1d(mradii)[0])
    le = np.asarray(radii)[np.asarray(radii) <= 0.3 + 1e-12]
    if len(le):
        return float(le.max())
    return float(np.min(radii))


class Run:
    """One teacher-forced a1 geometry (margins, response, signed thresholds)."""

    def __init__(self, run_dir):
        self.dir = os.path.abspath(run_dir)
        self.npz = os.path.join(self.dir, "a1.npz")
        if not os.path.exists(self.npz):
            raise FileNotFoundError(self.npz)
        d = np.load(self.npz, allow_pickle=True)
        self.d = d
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.meta = load_json(self.meta_path) if os.path.exists(self.meta_path) else {}

        self.m = np.asarray(d["margins"], dtype=float).ravel()
        self.T = len(self.m)
        self.radii = np.asarray(d["radii"], dtype=float).ravel()
        self.nr = len(self.radii)
        self.dm = np.asarray(d["dm"], dtype=float)
        self.K = self.dm.shape[1]
        if self.dm.shape != (self.nr, self.K, self.T):
            raise ValueError(f"dm shape {self.dm.shape} inconsistent with "
                             f"nr={self.nr}, K={self.K}, T={self.T}")
        self.dh = np.asarray(d["dh_norm"], dtype=float)          # (nr, K, nl, T)
        self.layers = np.asarray(d["layers"], dtype=int).ravel()
        self.nl = self.dh.shape[2]
        if self.dh.shape[:2] != (self.nr, self.K) or self.dh.shape[3] != self.T:
            raise ValueError(f"dh_norm shape {self.dh.shape} inconsistent")
        if self.nl != len(self.layers):
            self.layers = np.arange(self.nl)
        self.igain = choose_gain_index(self.radii)
        self.g = self.dm[self.igain] / self.radii[self.igain]  # margin change / radius
        self.gen_ids = d["gen_ids"] if "gen_ids" in d else None

    # -- geometry ---------------------------------------------------------
    def rc_signed(self, r):
        """Signed threshold radius field (K, T): m*r / max(-dm, 0), +inf if dm>=0."""
        ir = _match_radius(self.radii, r)
        if ir is not None:
            den = -self.dm[ir]
            with np.errstate(divide="ignore", invalid="ignore"):
                rc = r * self.m[None, :] / den
        else:
            den = -self.g
            with np.errstate(divide="ignore", invalid="ignore"):
                rc = self.m[None, :] / den     # radius cancels via linear scaling
        rc[den <= 0] = np.inf
        rc[~np.isfinite(rc)] = np.inf
        return rc

    def dh_over_r(self, r):
        """Response ||dh_t^l||/r (K, nl, T); linear fallback when r is not stored."""
        ir = _match_radius(self.radii, r)
        if ir is None:
            ir = self.igain
        return self.dh[ir] / self.radii[ir]

    @staticmethod
    def first_below(rc, r):
        """First t with rc[t] <= r per row, else -1 (rc: K x T)."""
        hit = rc <= r
        out = np.full(rc.shape[0], -1, dtype=np.int32)
        any_ = hit.any(axis=1)
        out[any_] = hit.argmax(axis=1)[any_]
        return out

    def summary_stats(self, r):
        rc = self.rc_signed(r)
        rs = rc.min(axis=1)
        ts = rc.argmin(axis=1)
        rs[~np.isfinite(rs)] = np.inf
        pred = self.first_below(rc, r)
        return rc, rs, ts, pred

    def label(self):
        model = self.meta.get("model") or self.meta.get("model_path") or ""
        model = os.path.basename(str(model).rstrip("/\\"))
        window = self.meta.get("window")
        bits = [model] if model else []
        if window:
            bits.append(f"window {window}")
        return ", ".join(bits)


# ----------------------------------------------------------------- helpers
def run_note(R):
    lab = R.label()
    return f" — {lab}" if lab else ""


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / max(1, len(x))


def pct(x, nd=0):
    return f"{100.0 * x:.{nd}f}%"


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def short_num(x):
    if not np.isfinite(x):
        return "inf"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000 or a < 1e-3:
        return f"{x:.1e}"
    if a >= 10:
        return f"{x:.0f}"
    if a >= 1:
        return f"{x:.1f}"
    return f"{x:.3g}"


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def log_ticks(ax, lo, hi, n=5):
    """Readable log-x ticks: 1/2/3/5 x powers of ten inside [lo, hi]."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    cand = []
    for e in range(-12, 13):
        for m in (1.0, 2.0, 5.0):
            v = m * 10.0 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                cand.append(v)
    if len(cand) > n:
        pick = np.unique(np.linspace(0, len(cand) - 1, n).round().astype(int))
        cand = [cand[i] for i in pick]
    if len(cand) < 3:
        # narrow range: linear ticks are more legible than one 10^k tick
        cand = [float(v) for v in np.linspace(lo, hi, 4)]
    ax.set_xticks(cand)
    ax.set_xticklabels([short_num(v) for v in cand])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)
    return path


def panel_title(ax, text, fontsize=9.5):
    ax.set_title(text, fontsize=fontsize, loc="left")


# ----------------------------------------------------------------- figure 1
def fig_margins(R, outdir, tag, n_small=6):
    t = np.arange(R.T)
    m = np.asarray(R.m, dtype=float)
    order = np.argsort(m)
    n_small = int(min(max(n_small, 1), R.T))
    small = order[:n_small]
    mmin, tmin = float(m[order[0]]), int(order[0])
    n_frag = int((m < 0.1).sum())

    fig = plt.figure(figsize=(12.5, 4.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.75, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])

    ax.plot(t, np.maximum(m, 1e-4), lw=0.9, color=C_MED)
    ax.scatter(t[small], np.maximum(m[small], 1e-4), s=22, color=C_OBS, zorder=4)
    ymax = float(np.nanmax(m))
    xs = np.sort(small)
    groups = [[int(xs[0])]]
    for ti in xs[1:]:
        if ti - groups[-1][-1] < 10:
            groups[-1].append(int(ti))
        else:
            groups.append([int(ti)])
    for gi, group in enumerate(groups):
        dy = 16 if gi % 2 == 0 else 34
        for k, ti in enumerate(group):
            ax.axvline(ti, color=C_OBS, lw=0.7, alpha=0.45)
            frac = 0.5 if len(group) == 1 else k / (len(group) - 1)
            dx = (frac - 0.5) * 2 * 24
            ax.annotate(f"t={ti}", xy=(ti, m[ti]), xytext=(dx, dy),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=C_OBS,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=C_OBS,
                                        alpha=0.6, shrinkA=1, shrinkB=2))
    ax.set_yscale("log")
    ax.set_xlabel(r"generated step $t$")
    ax.set_ylabel(r"$m_t$  (top-1 $-$ top-2 logit gap, nats)")
    ax.set_ylim(max(1e-4, mmin / 3), ymax * 4.5)
    ax.set_title(
        wrap(f"Margins: {n_frag} of {R.T} steps are within 0.1 nats; these set the "
             f"lowest flip thresholds (tightest {short_num(mmin)} at t={tmin})", 80),
        loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    xs, ys = ecdf(m)
    ax2.step(xs, ys, where="post", color=C_MED, lw=1.6)
    for q, yq in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]:
        v = float(np.quantile(m, q))
        ax2.axvline(v, color="#888888", lw=0.7, ls=":")
        ax2.text(v, 0.02, f"p{int(q*100)}={short_num(v)}", rotation=90,
                 fontsize=7.5, color="#555555", ha="right", va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$m_t$  (top-1 $-$ top-2 logit gap, nats, log scale)")
    ax2.set_ylabel(r"fraction of steps with margin $\leq x$")
    ax2.set_ylim(0, 1.02)
    ax2.set_title(
        wrap(f"ECDF: median margin {short_num(np.median(m))} nats; "
             f"{pct(np.mean(m < 0.1), 1)} of steps below 0.1 nats", 46),
        loc="left", fontsize=10)
    fig.suptitle(
        "Margin spectrum: a few fragile near-ties set the lowest flip thresholds"
        + run_note(R), x=0.01, ha="left", fontsize=12.5, y=1.02)
    return save(fig, outdir, f"{tag}_margins.png")


# ----------------------------------------------------------------- figure 2
def fig_response(R, r, outdir, tag):
    rr = R.dh_over_r(r)                          # (K, nl, T)
    t = np.arange(R.T)
    med = np.median(rr, axis=0)                  # (nl, T)
    lo = np.percentile(rr, 10, axis=0)
    hi = np.percentile(rr, 90, axis=0)

    nl = R.nl
    ncols = nl if nl <= 4 else min(4, nl)
    nrows = int(np.ceil(nl / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.0 * nrows + 0.9),
                             squeeze=False)
    zero_layers = []
    for li in range(nl):
        ax = axes[li // ncols, li % ncols]
        if not np.any(hi[li] > 0):
            zero_layers.append(li)
        ax.fill_between(t, lo[li], hi[li], color=C_BAND, alpha=0.45,
                        lw=0, label="10–90% of directions")
        ax.plot(t, med[li], color=C_MED, lw=1.7, label="median direction")
        ax.set_xlabel(r"generated step $t$")
        ax.set_ylabel(r"$\|\delta h_t^{\ell}\|/r$  (change per unit radius)")
        layer_name = int(R.layers[li]) if li < len(R.layers) else li
        note = " — no change recorded" if li in zero_layers else ""
        panel_title(ax, f"layer {layer_name}{note}")
        ax.set_ylim(bottom=0)
    for li in range(nl, nrows * ncols):
        axes[li // ncols, li % ncols].axis("off")

    # conclusion: peak median response per layer (late-run values decay to ~0)
    peak = med.max(axis=1)
    valid = np.where(peak > 1e-9)[0]
    if len(valid) >= 2:
        lo_i, hi_i = int(valid[0]), int(valid[-1])
        a, b = float(peak[lo_i]), float(peak[hi_i])
        t_peak = int(med[hi_i].argmax())
        if b > 1.25 * a:
            concl = (f"peak median response grows {b / a:.0f}× from layer "
                     f"{int(R.layers[lo_i])} to {int(R.layers[hi_i])} and "
                     f"decays after t≈{t_peak}")
        else:
            concl = (f"peak response is depth-flat ({short_num(a)}–"
                     f"{short_num(b)}) and decays after t≈{t_peak}")
    else:
        concl = "response per layer"
    fig.suptitle(f"Perturbation response at $r={r:g}$: {concl}",
                 x=0.01, ha="left", fontsize=12.5, y=1.015)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=1.8)
    return save(fig, outdir, f"{tag}_response_curves.png")


# ----------------------------------------------------------------- figure 3
def fig_rc_ladder(R, r, outdir, tag):
    rc, rs, ts, pred = R.summary_stats(r)
    order = np.argsort(rs)
    mat = rc[order]
    K, T = mat.shape
    finite = np.isfinite(mat)
    if not finite.any():
        warnings.warn("no finite signed thresholds; rc ladder is empty")
        vals = np.array([r])
    else:
        vals = mat[finite]
    vmin = max(min(np.percentile(vals, 1), r / 20.0), 1e-4)
    vmax = max(np.percentile(vals, 99), r * 20.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[4.0, 1.35, 0.17],
                  wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="lower",
                   extent=[-0.5, T - 0.5, -0.5, K - 0.5], cmap=cmap,
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
    ax.grid(False)
    # flip boundary r_c = r (cells with smaller r_c are the predicted flips)
    n_flip_cells = int((mat <= r).sum())
    if n_flip_cells:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.contour(np.arange(T), np.arange(K),
                       np.where(finite, mat, vmax * 10),
                       levels=[r], colors="white", linewidths=1.1,
                       linestyles="--", alpha=0.9)
    rows_flip = np.where(rs[order] <= r)[0]
    if len(rows_flip) <= 100:
        ax.plot(np.full(len(rows_flip), T - 0.5), rows_flip, marker=">", ls="none",
                color=C_OBS, ms=5, clip_on=False, zorder=5)
    ax.set_xlabel(r"generated step $t$")
    ax.set_ylabel(r"direction rank $u$  (sorted by $r_*$; smallest at bottom)")
    flip_frac = float(np.mean(rs <= r))
    rs_finite = rs[np.isfinite(rs)]
    rs_max = float(rs_finite.max()) if len(rs_finite) else float("inf")
    ax.set_title(
        wrap(f"Signed threshold ladder: {pct(flip_frac, 1)} of {K} directions flip at "
             f"r={r:g} (white dashed = r_c=r; brighter = easier to flip)", 78),
        loc="left", fontsize=10)
    panel_title(ax, "each row is one perturbation direction", fontsize=8.5)

    axh = fig.add_subplot(gs[0, 1])
    rsf = rs[np.isfinite(rs) & (rs > 1e-6)]
    if len(rsf):
        lo = max(min(float(rsf.min()), r) / 1.3, 1e-4)
        hi = max(float(rsf.max()), r) * 1.3
        bins = np.logspace(np.log10(lo), np.log10(hi), 30)
        axh.hist(rsf, bins=bins, color=C_MED, alpha=0.8)
        axh.set_xscale("log")
        log_ticks(axh, lo, hi, n=4)
    axh.yaxis.set_major_locator(MaxNLocator(integer=True))
    axh.axvline(r, color=C_OBS, lw=1.4, ls="--")
    axh.annotate(f"run r={r:g}", xy=(r, 0.97), xycoords=("data", "axes fraction"),
                 xytext=(4, 0), textcoords="offset points", color=C_OBS,
                 fontsize=7.5, va="top")
    axh.set_xlabel(r"$r_*=\min_t r_c^-$  (threshold radius, log)")
    axh.set_ylabel("direction count")
    panel_title(axh, wrap(f"threshold distribution ({flip_frac*100:.1f}% left of "
                          f"line = flip)", 26), fontsize=8.5)
    n_inf = int((~np.isfinite(rs)).sum())
    if n_inf:
        axh.text(0.03, 0.97, f"{n_inf} direction(s) never flip\n($dm\\geq0$ everywhere)",
                 transform=axh.transAxes, va="top", fontsize=7.5, color="#555")

    axc = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=axc)
    cb.set_label(r"$r_c^-$  (signed flip threshold radius, log)"
                 "\nwhite line = run radius; below it = predicted flips")
    cb.ax.axhline(r, color="white", lw=1.6)
    cb.ax.axhline(r, color=C_OBS, lw=0.9, ls="--")
    fig.suptitle(
        wrap(f"Threshold ladder sorted by r*: the bottom {flip_frac*100:.1f}% of "
             f"directions flip at the run radius; the median needs "
             f"r={short_num(safe_percentile(rs, 50))}, the hardest "
             f"r={short_num(rs_max)}", 105),
        x=0.01, ha="left", fontsize=12.5, y=1.02)
    return save(fig, outdir, f"{tag}_rc_ladder.png")


# ----------------------------------------------------------------- figure 4
def fig_rstar_ecdf(R, r, outdir, tag):
    rc, rs, ts, pred = R.summary_stats(r)
    fin = np.isfinite(rs) & (rs > 0)
    frac = float(np.mean(rs <= r))
    n_inf = int((~np.isfinite(rs)).sum())

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if fin.any():
        xs, ys = ecdf(rs[fin])
        ax.step(xs, ys * (1 - n_inf / max(1, len(rs))), where="post",
                color=C_MED, lw=1.8, label="directions with a finite threshold")
        ax.scatter(xs, ys * (1 - n_inf / max(1, len(rs))), s=10,
                   color=C_MED, alpha=0.5, zorder=3)
    ax.axvline(r, color=C_OBS, lw=1.6, ls="--", label=f"run radius r={r:g}")
    ax.axhline(frac, color=C_OBS, lw=0.8, ls=":", alpha=0.7)
    ax.annotate(f"predicted flip fraction = {pct(frac, 1)}\n"
                f"({int((rs <= r).sum())} of {len(rs)} directions have $r_*<r$)",
                xy=(r, frac), xytext=(0.05, 0.60), textcoords="axes fraction",
                fontsize=9, color=C_OBS,
                arrowprops=dict(arrowstyle="->", color=C_OBS, lw=0.9))
    for q in (10, 50, 90):
        v = safe_percentile(rs, q)
        if np.isfinite(v):
            ax.axvline(v, color="#999999", lw=0.6, ls=":")
            ax.text(v, 0.055, f"p{q}={short_num(v)}", rotation=90,
                    fontsize=7.5, color="#555", ha="right", va="bottom")
    ax.set_xscale("log")
    lo = max(1e-4, float(np.nanmin(rs[fin] if fin.any() else [r])) / 1.5)
    hi = float(np.nanmax(rs[fin] if fin.any() else [r])) * 1.6
    log_ticks(ax, lo, hi, n=5)
    ax.set_xlabel(r"$r_*=\min_t r_c^-(t,u)$  (per-direction threshold radius, log)")
    ax.set_ylabel(r"fraction of directions with $r_*\leq x$  (ECDF)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper left")
    extra = f"; {n_inf} direction(s) can never flip" if n_inf else ""
    ax.set_title(
        wrap(f"Thresholds: {pct(frac, 1)} of directions flip at r={r:g}; the median "
             f"direction needs r={short_num(safe_percentile(rs, 50))}{extra}", 66),
        loc="left", fontsize=10)
    fig.tight_layout()
    return save(fig, outdir, f"{tag}_rstar_ecdf.png")


# ----------------------------------------------------------------- figure 5
def fig_gains(R, r, outdir, tag):
    rr = R.dh_over_r(r)[:, -1, :]                # (K, T) last stored layer
    G = np.median(rr, axis=1)                    # gain per direction
    ok = np.isfinite(G) & (G > 0)
    Gfin = G[ok]
    q = np.percentile(Gfin, [0, 50, 100])
    fold = q[2] / q[0] if q[0] > 0 else np.inf

    fig = plt.figure(figsize=(13.0, 4.8))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.1, 0.95, 1.5, 1.25], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    if q[2] > q[0] * 1.001:
        bins = np.logspace(np.log10(max(q[0], 1e-6)), np.log10(q[2]), 32)
    else:
        bins = 20
    ax1.hist(Gfin, bins=bins, color=C_MED, alpha=0.85)
    ax1.set_xscale("log")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.axvline(q[1], color=C_OBS, lw=1.2, ls="--")
    ax1.annotate(f"median {short_num(q[1])}", xy=(q[1], 0.96),
                 xycoords=("data", "axes fraction"), xytext=(4, 0),
                 textcoords="offset points", color=C_OBS, fontsize=7.5,
                 va="top")
    log_ticks(ax1, q[0] / 1.4, q[2] * 1.4, n=4)
    ax1.set_xlabel(r"$G(u)$  (median$_t\ \|\delta h_t^{last}\|/r$)")
    ax1.set_ylabel("direction count")
    panel_title(ax1, f"gain histogram ({len(Gfin)} directions)", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    xs, ys = ecdf(Gfin)
    ax2.step(xs, ys, where="post", color=C_MED, lw=1.6)
    ax2.set_xscale("log")
    log_ticks(ax2, q[0] / 1.4, q[2] * 1.4, n=4)
    ax2.set_xlabel(r"$G(u)$  (gain, log)")
    ax2.set_ylabel(r"fraction of directions with $G\leq x$")
    ax2.set_ylim(0, 1.02)
    panel_title(ax2, f"gain ECDF: {fold:.1f}× spread", fontsize=9)

    # individual curves vs median, each divided by its own median over t
    ax3 = fig.add_subplot(gs[0, 2])
    t = np.arange(R.T)
    med_curve = np.median(rr, axis=0)
    denom_med = np.median(med_curve)
    norm = rr / np.maximum(np.median(rr, axis=1, keepdims=True), 1e-12)
    vidx = np.where(ok)[0]
    gorder = vidx[np.argsort(G[vidx])] if len(vidx) else np.array([], dtype=int)
    pick = ([int(gorder[int(0.10 * (len(gorder) - 1))]),
             int(gorder[len(gorder) // 2]),
             int(gorder[int(0.90 * (len(gorder) - 1))])] if len(gorder) else [])
    colors = [C_OBS, C_ALT, C_THIRD]
    for j, di in enumerate(pick):
        ax3.plot(t, norm[di], color=colors[j], lw=1.0, alpha=0.9,
                 label=f"dir {di} (G={short_num(G[di])})")
    if denom_med > 0:
        ax3.plot(t, med_curve / denom_med, color="black", lw=2.2,
                 label="median direction")
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel(r"generated step $t$")
    ax3.set_ylabel(r"response $\|\delta h_t^{last}\|/r$ ÷ own median (shape)")
    corrs = [np.corrcoef(norm[di][R.T // 4:], (med_curve / denom_med)[R.T // 4:])[0, 1]
             if denom_med > 0 else np.nan for di in pick]
    cmin = np.nanmin(corrs) if np.isfinite(corrs).any() else np.nan
    if not np.isfinite(cmin):
        shape_msg = "normalised shapes: compare each direction with the median"
    elif cmin >= 0.7:
        shape_msg = f"late-run shape matches the median (r≥{cmin:.2f}); early spikes differ"
    elif cmin >= 0.3:
        shape_msg = f"late-run shapes broadly match (r≥{cmin:.2f}); early spikes differ"
    else:
        shape_msg = f"normalised shapes vary (r≥{cmin:.2f}); spikes are direction-specific"
    panel_title(ax3, wrap(shape_msg, 52), fontsize=9)
    ax3.legend(fontsize=7, loc="upper right")

    ax4 = fig.add_subplot(gs[0, 3])
    sg = np.sort(Gfin)[::-1]
    cum = np.cumsum(sg) / sg.sum()
    rank = np.arange(1, len(sg) + 1) / len(sg)
    ax4.plot(rank, cum, color=C_MED, lw=1.7)
    k10 = int(np.ceil(0.1 * len(sg)))
    ax4.axvline(0.1, color=C_OBS, lw=0.9, ls="--")
    ax4.annotate(f"top 10% of directions\ncarry {cum[k10 - 1]*100:.0f}% of total gain",
                 xy=(0.1, cum[k10 - 1]), xytext=(0.22, max(0.25, cum[k10 - 1] - 0.25)),
                 fontsize=7.5, color=C_OBS,
                 arrowprops=dict(arrowstyle="->", color=C_OBS, lw=0.8))
    ax4.set_xlim(0, 1.02)
    ax4.set_ylim(0, 1.02)
    ax4.set_xlabel("fraction of directions (sorted strongest first)")
    ax4.set_ylabel("cumulative share of total gain")
    panel_title(ax4, "gain is concentrated in a few directions", fontsize=9)

    fig.suptitle(
        f"Direction gains span {fold:.1f}× (G = {short_num(q[0])} to "
        f"{short_num(q[2])}); the top 10% of directions carry "
        f"{cum[k10 - 1]*100:.0f}% of the response",
        x=0.01, ha="left", fontsize=12.5, y=1.03)
    return save(fig, outdir, f"{tag}_direction_gains.png")


# ----------------------------------------------------------------- free helpers
def aligned_free(R, free):
    """Free obs + signed predictions aligned to this run's direction indices.

    The free run may cover only a subset of directions (``dirs``); the returned
    arrays are restricted to that subset and ordered like the free run.
    """
    obs_all = np.asarray(free["obs"], dtype=int)
    if free.get("dirs") is not None:
        dirs_all = np.asarray(free["dirs"], dtype=int).ravel()
    else:
        dirs_all = np.arange(obs_all.shape[1])
    keep = (dirs_all >= 0) & (dirs_all < R.K)
    if not keep.all():
        warnings.warn("free run references direction indices outside the a1 run; "
                      "those columns are dropped")
    obs = obs_all[:, keep]
    dirs = dirs_all[keep]
    radii = np.asarray(free["radii"], dtype=float).ravel()
    pred = np.vstack([R.first_below(R.rc_signed(rr), rr)[dirs] for rr in radii])
    return radii, obs, dirs, pred


# ----------------------------------------------------------------- figure 6
def fig_pred_vs_obs(R, free, outdir, tag):
    radii, obs, dirs, pred = aligned_free(R, free)
    n_r = len(radii)
    T = R.T

    ncols = 1 if n_r == 1 else min(n_r, 4)
    nrows_s = int(np.ceil(n_r / ncols))
    fig = plt.figure(figsize=(7.0 if n_r == 1 else 3.7 * ncols,
                              3.0 * nrows_s + 4.2))
    gs = GridSpec(nrows_s + 1, ncols, figure=fig, hspace=0.55, wspace=0.40,
                  height_ratios=[1.0] * nrows_s + [1.55])

    tg = np.arange(T)
    for ri in range(n_r):
        ax = fig.add_subplot(gs[ri // ncols, ri % ncols])
        o = obs[ri]
        p = pred[ri]
        so = np.array([np.mean((o > t) | (o < 0)) for t in tg])
        sp = np.array([np.mean((p > t) | (p < 0)) for t in tg])
        ax.step(tg, so, where="post", color=C_OBS, lw=1.7, label="observed (free run)")
        ax.step(tg, sp, where="post", color=C_PRED, lw=1.5, ls="--",
                label="predicted ($r_c^-$)")
        exc = int((o == p).sum())
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"step $t$")
        ax.set_ylabel(r"$S(T>t)$  (fraction unflipped)")
        panel_title(ax, f"r={radii[ri]:g}: {int((o >= 0).sum())} obs / "
                        f"{int((p >= 0).sum())} pred flips\n{exc}/{len(o)} exact",
                    fontsize=8.5)
        if ri == 0:
            handles, labels = ax.get_legend_handles_labels()

    # scatter / hexbin panel across all radii
    ax = fig.add_subplot(gs[nrows_s, :])
    x = np.where(obs >= 0, obs, T).ravel().astype(float)
    y = np.where(pred >= 0, pred, T).ravel().astype(float)
    n = len(x)
    cat_hit = (obs >= 0) & (pred >= 0) & (obs == pred)
    cat_wrong = (obs >= 0) & (pred >= 0) & (obs != pred)
    cat_miss = (obs >= 0) & (pred < 0)
    cat_fa = (obs < 0) & (pred >= 0)
    cat_none = (obs < 0) & (pred < 0)
    if n <= 600:
        ax.scatter(x[cat_hit.ravel()], y[cat_hit.ravel()], s=22, color=C_THIRD,
                   alpha=0.85, label="hit (obs = pred)", zorder=3)
        ax.scatter(x[cat_wrong.ravel()], y[cat_wrong.ravel()], s=26, marker="D",
                   color=C_PRED, alpha=0.85, label="wrong step (both flip)", zorder=3)
        ax.scatter(x[cat_miss.ravel()], y[cat_miss.ravel()], s=30, marker="v",
                   color=C_ALT, alpha=0.9, label="miss (obs flips, pred does not)", zorder=3)
        ax.scatter(x[cat_fa.ravel()], y[cat_fa.ravel()], s=30, marker="^",
                   color=C_OBS, alpha=0.9, label="false alarm (obs never)", zorder=3)
        ax.scatter(x[cat_none.ravel()], y[cat_none.ravel()], s=14, marker="s",
                   color="#bbbbbb", alpha=0.8, label="both no flip", zorder=2)
    else:
        hb = ax.hexbin(x, y, gridsize=36, bins="log", mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="direction × radius count (log)")
        ax.scatter(x[cat_miss.ravel()], y[cat_miss.ravel()], s=8, marker="v",
                   color=C_ALT, alpha=0.4, label="miss")
        ax.scatter(x[cat_fa.ravel()], y[cat_fa.ravel()], s=8, marker="^",
                   color=C_OBS, alpha=0.4, label="false alarm")
    ax.plot([0, T], [0, T], color="black", lw=1.0, ls="--", alpha=0.6)
    ax.set_xlim(-2, T + 2)
    ax.set_ylim(-2, T + 2)
    ax.set_xlabel(r"observed first-flip step (free run; $T$ = no flip within the run)")
    ax.set_ylabel(r"predicted first-flip step ($r_c^-$ rule; $T$ = no flip)")
    n_all = obs.size
    exact = int((obs == pred).sum())
    n_obs_flip = int((obs >= 0).sum())
    exact_flip = int(((obs == pred) & (obs >= 0)).sum())
    ax.set_title(
        wrap(f"Prediction vs free run over {n_r} radii: {exact}/{n_all} exact "
             f"({pct(exact / max(1, n_all), 1)}), {exact_flip}/{max(1, n_obs_flip)} "
             f"exact among observed flips, {int(cat_wrong.sum())} wrong step, "
             f"{int(cat_fa.sum())} false alarms, {int(cat_miss.sum())} misses", 100),
        loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.955),
               ncol=2, fontsize=8)
    fig.suptitle(
        f"Free-run validation: signed thresholds predict {pct(exact / max(1, n_all), 1)} "
        f"of direction/radius cases exactly", x=0.01, ha="left",
        fontsize=12.5, y=1.01)
    return save(fig, outdir, f"{tag}_pred_vs_obs.png")


# ----------------------------------------------------------------- figure 7
def fig_flip_stats(R, free, outdir, tag):
    radii, obs, dirs, pred = aligned_free(R, free)
    n = obs.shape[1]

    flip_o = (obs >= 0).mean(axis=1)
    flip_p = (pred >= 0).mean(axis=1)
    mean_o = np.array([obs[i][obs[i] >= 0].mean() if (obs[i] >= 0).any() else np.nan
                       for i in range(len(radii))])
    sd_o = np.array([obs[i][obs[i] >= 0].std(ddof=1) if (obs[i] >= 0).sum() > 1 else 0.0
                     for i in range(len(radii))])
    nflip = (obs >= 0).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.fill_between(radii, 0, flip_o, step="post", color=C_OBS, alpha=0.22, lw=0)
    ax.step(radii, flip_o, where="post", color=C_OBS, lw=1.8, marker="o",
            label="observed (free run)")
    ax.step(radii, flip_p, where="post", color=C_PRED, lw=1.6, ls="--",
            marker="s", ms=4, label="predicted ($r_c^-$)")
    for rr, fo in zip(radii, flip_o):
        ax.text(rr, fo + 0.025, f"{fo*100:.0f}%", ha="center", fontsize=8,
                color=C_OBS)
    ax.set_xscale("log")
    ax.set_xticks(radii)
    ax.set_xticklabels([short_num(v) for v in radii])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(float(np.min(radii) / 1.5), float(np.max(radii) * 1.5))
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("perturbation radius $r$ (log)")
    ax.set_ylabel("flip fraction (share of directions that flip)")
    ax.set_title(f"Flips start near the predicted threshold; n={n} directions per radius",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ok = np.isfinite(mean_o)
    ax.errorbar(radii[ok], mean_o[ok], yerr=sd_o[ok], fmt="o-", color=C_OBS,
                lw=1.6, capsize=3, label="observed mean first-flip step")
    pred_mean = np.array([pred[i][pred[i] >= 0].mean() if (pred[i] >= 0).any()
                          else np.nan for i in range(len(radii))])
    ax.plot(radii, pred_mean, "s--", color=C_PRED, lw=1.4,
            label="predicted mean first-flip step")
    ymax = np.nanmax(mean_o[ok] + sd_o[ok]) if ok.any() else 1.0
    for rr, mo, nf in zip(radii, mean_o, nflip):
        if np.isfinite(mo):
            ax.annotate(f"n_flip={int(nf)}", (rr, mo), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=C_OBS)
        else:
            ax.annotate(f"n_flip=0 (n={n})", (rr, 0.02), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5, color="#666")
    ax.set_xscale("log")
    ax.set_xticks(radii)
    ax.set_xticklabels([short_num(v) for v in radii])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(float(np.min(radii) / 1.5), float(np.max(radii) * 1.5))
    ax.set_ylim(0, max(1.15 * ymax, 1.0))
    ax.set_xlabel("perturbation radius $r$ (log)")
    ax.set_ylabel("mean first-flip step among flipped runs")
    ax.set_title("Flip timing: n annotated; error bars = ±1 s.d. across directions",
                 loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="best")

    if np.isclose(flip_o[0], flip_o[-1]):
        trend = (f"flip fraction stays at {flip_o[0]*100:.0f}% across radii "
                 f"(predicted {flip_p[0]*100:.0f}% → {flip_p[-1]*100:.0f}%)")
    else:
        trend = (f"observed flips go from {flip_o[0]*100:.0f}% at r={radii[0]:g} "
                 f"to {flip_o[-1]*100:.0f}% at r={radii[-1]:g} "
                 f"(predicted {flip_p[0]*100:.0f}% → {flip_p[-1]*100:.0f}%)")
    fig.suptitle("Free-run flip statistics: " + trend,
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return save(fig, outdir, f"{tag}_flip_stats.png")


# ----------------------------------------------------------------- summary
def build_summary(R, r, free, free_path):
    rc, rs, ts, pred = R.summary_stats(r)
    order = np.argsort(R.m)
    n_small = int(min(6, R.T))
    summ = {
        "run_dir": R.dir,
        "model": R.meta.get("model"),
        "window": R.meta.get("window"),
        "meta_json_present": bool(R.meta),
        "n_steps": int(R.T),
        "n_directions": int(R.K),
        "n_layers": int(R.nl),
        "layers": [int(x) for x in R.layers],
        "radii": [float(x) for x in R.radii],
        "run_radius": float(r),
        "gain_reference_radius": float(R.radii[R.igain]),
        "min_margin": {"step": int(order[0]), "value": float(R.m[order[0]])},
        "smallest_margins": [
            {"step": int(t), "value": float(R.m[t])} for t in order[:n_small]
        ],
        "n_steps_below_0.1": int((R.m < 0.1).sum()),
        "r_star_quantiles": {
            "p10": safe_percentile(rs, 10),
            "p50": safe_percentile(rs, 50),
            "p90": safe_percentile(rs, 90),
        },
        "n_directions_infinite_threshold": int((~np.isfinite(rs)).sum()),
        "predicted_flip_fraction_at_run_radius": float(np.mean(rs <= r)),
        "predicted_flip_directions_at_run_radius": int((rs <= r).sum()),
    }
    if free is None:
        return summ

    radii, obs, _dirs, pred = aligned_free(R, free)
    per = []
    for i, rr in enumerate(radii):
        o, p = obs[i], pred[i]
        n = len(o)
        exact = int((o == p).sum())
        n_obs = int((o >= 0).sum())
        exact_flip = int(((o == p) & (o >= 0)).sum())
        fa = int(((o < 0) & (p >= 0)).sum())
        miss = int(((o >= 0) & (p < 0)).sum())
        per.append({
            "radius": float(rr),
            "n_directions": int(n),
            "n_observed_flips": n_obs,
            "n_predicted_flips": int((p >= 0).sum()),
            "n_exact": exact,
            "exact_fraction_all": exact / max(1, n),
            "exact_fraction_among_flipped": exact_flip / max(1, n_obs),
            "false_alarms": fa,
            "misses": miss,
            "mean_observed_first_flip_step": (
                float(o[o >= 0].mean()) if n_obs else None),
            "mean_predicted_first_flip_step": (
                float(p[p >= 0].mean()) if (p >= 0).any() else None),
        })
    summ["free"] = {
        "path": free_path,
        "n_radii": int(len(radii)),
        "exact_fraction_all": float((obs == pred).mean()),
        "false_alarms_total": int(((obs < 0) & (pred >= 0)).sum()),
        "misses_total": int(((obs >= 0) & (pred < 0)).sum()),
        "per_radius": per,
    }
    return summ


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="directory with a1.npz (+ optional meta.json)")
    ap.add_argument("--free", default=None, help="directory with free_run.npz (or a direct npz)")
    ap.add_argument("--tag", default="ds14", help="filename prefix / figure tag")
    ap.add_argument("--outdir", default=os.path.join("notebook_big", "figures"))
    ap.add_argument("--radius", type=float, default=None,
                    help="radius the figures describe (default: auto from meta/npz)")
    ap.add_argument("--n-small", type=int, default=6,
                    help="number of smallest-margin steps ticked in the margin figure")
    args = ap.parse_args(argv)

    R = Run(args.run)
    r = choose_run_radius(R.radii, R.meta, args.radius)
    if _match_radius(R.radii, r) is None:
        warnings.warn(f"requested radius {r:g} is not stored; response scaled "
                      f"linearly from r={R.radii[R.igain]:g}")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"run={R.dir}  K={R.K}  T={R.T}  radii={R.radii.tolist()}  "
          f"run r={r:g}  gain ref r={R.radii[R.igain]:g}")

    made = []
    made.append(fig_margins(R, args.outdir, args.tag, args.n_small))
    made.append(fig_response(R, r, args.outdir, args.tag))
    made.append(fig_rc_ladder(R, r, args.outdir, args.tag))
    made.append(fig_rstar_ecdf(R, r, args.outdir, args.tag))
    made.append(fig_gains(R, r, args.outdir, args.tag))

    free = load_free(args.free) if args.free else None
    if args.free and free is None:
        warnings.warn("--free given but no usable obs/pred found")
    if free is not None:
        print(f"free: {free['path']}  radii={np.asarray(free['radii']).tolist()}  "
              f"K_free={np.asarray(free['obs']).shape[1]}")
        made.append(fig_pred_vs_obs(R, free, args.outdir, args.tag))
        made.append(fig_flip_stats(R, free, args.outdir, args.tag))

    summ = build_summary(R, r, free, args.free)
    spath = os.path.join(args.outdir, f"{args.tag}_phase1_summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print("wrote", spath)
    print(json.dumps(summ, indent=2))
    return made


if __name__ == "__main__":
    main()
