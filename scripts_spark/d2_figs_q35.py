#!/usr/bin/env python
"""The three notebook figures for the Qwen3.5-9B D2 section.

Consumes results_big/qwen35/d2 (the GPU distance matrices + summary.json) and
the pipeline JSON written by ``d2_local.py --tag q35d2``.  All estimators are
the validated ones from ``d2_local.py``; this script only arranges them.

    python scripts_spark/d2_figs_q35.py

Outputs (notebook_big/figures):
    q35d2_overview.png       C(eps) + local slope, 4 trajectories
    q35d2_summary.png        (a) D2/TwoNN  (b) loop screen  (c) N dependence
    q35d2_time_shuffle.png   raw vs index-shuffled distance matrix, 2 runs
    q35d2_numbers.json       every number quoted in the section

The "chord" quoted here is evaluated on a 900-point eps grid rather than the
pipeline's fixed 90-point grid: at d=4096 the admissible 10/N -> 0.1 range is
only 2-3 pipeline grid steps wide, so the 90-point chord is undefined for
7/12 runs.  Both are reported; the estimator (OLS slope of log C on log eps
over the admissible range) is identical.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2_local as L  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2DIR = os.path.join(ROOT, "results_big", "qwen35", "d2")
FIG = os.path.join(ROOT, "notebook_big", "figures")
PIPE = os.path.join(FIG, "d2pipe_q35d2.json")
TAG = "q35d2"

COL = ["#2166ac", "#d6604d", "#1b7837"]
C_GREY = "#8a8a8a"
C_LOOP = "#b2182b"
C_OK = "#2166ac"
C_CHORD = "#762a83"
C_T4 = "#8a8a8a"
C_ORANGE = "#e08214"
BOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78)

# 1.5B reference numbers (notebook_big/sections/section_d2_pipeline.md)
T15_TWONN = (5.44, 6.23)          # four validated runs, loops included
T15_LOOP_D2 = (2.978, 3.894)      # T4 max-R^2 chords, period-53/69 loops
T15_HEALTHY_D2 = (9.195, 11.353)  # T4 chords, no plateau (0.08-0.10 dec)
T15_OLD = 5.75                    # docs/MEASUREMENTS.md raw D2 (shuffled 5.69)
T15_LOOP_PROM = 0.45              # worst 1.5B full-loop prominence
T15_HEALTHY_PROM = 0.91           # best 1.5B healthy prominence

# DS-14B reference numbers (notebook_big/sections/section_d2_ds14.md)
DS14_TWONN = (4.66, 8.90)
DS14_TWONN_MED = 8.19
DS14_CHORD = (8.10, 10.27)
DS14_T4 = (8.69, 11.75)
DS14_PROM = (0.961, 0.992)


def load_pipe():
    with open(PIPE) as fh:
        pipe = json.load(fh)
    return {t["name"]: t for t in pipe["trajectories"]}, pipe


def load_D(name, n=None):
    D = np.load(os.path.join(D2DIR, f"dist_{name}_h_last.npy"))
    D = np.asarray(D, dtype=np.float32)
    return D if n is None else D[:n, :n]


def curve(D, n=None, theiler=0, d_typ=None):
    """(eps/d_typ, C, local slope, d_typ) at the pipeline grid."""
    if n is not None:
        D = D[:n, :n]
    dt = L.typical_scale(D) if d_typ is None else d_typ
    eps = dt * np.geomspace(L.EPS_LO, L.EPS_HI, L.N_EPS)
    C, _ = L.correlation_curve(D, eps, theiler=min(theiler, D.shape[0] - 1))
    e, s = L.local_slope(eps, C)
    return e / dt, C[C > 0], s, dt


def win_from_C(n, e, C, s):
    return L.valid_window(n, e, C, s)


def window_at(D, n, theiler=0):
    e, C, s, _ = curve(D, n, theiler)
    return win_from_C(n, e, C, s)


def fine_chord(D, n=None, d_typ=None, theiler=0, n_eps=900, lo=0.15, hi=1.7):
    """Chord over the admissible range on a fine eps grid (the 90-pt grid is
    only 2-3 points wide here).  Also the exact 10/N and 0.1 crossings."""
    if n is not None:
        D = D[:n, :n]
    dt = L.typical_scale(D) if d_typ is None else d_typ
    N = D.shape[0]
    eps = dt * np.geomspace(lo, hi, n_eps)
    C, _ = L.correlation_curve(D, eps, theiler=min(theiler, N - 1))
    e = eps / dt
    m = (C >= 10.0 / N) & (C <= 0.1)
    ok = bool(m.sum() >= 3)
    if not ok:
        return dict(chord=float("nan"), span=0.0, npts=int(m.sum()),
                    eps_lo=float("nan"), eps_hi=float("nan"))
    return dict(
        chord=float(np.polyfit(np.log(e[m]), np.log(C[m]), 1)[0]),
        span=float(np.log10(e[m][-1] / e[m][0])),
        npts=int(m.sum()), eps_lo=float(e[m][0]), eps_hi=float(e[m][-1]),
    )


def dist_shape(D):
    """Pairwise-distance distribution shape (CV and quantiles, in d_typ units)."""
    dt = L.typical_scale(D)
    n = D.shape[0]
    ut = np.concatenate([D[i, i + 1:] for i in range(n - 1)]) / dt
    q = np.quantile(ut, [0.01, 0.10, 0.50, 0.90])
    return dict(cv=float(ut.std() / ut.mean()), q01=float(q[0]),
                q10=float(q[1]), q50=float(q[2]), q90=float(q[3]),
                min=float(ut.min()))


def n_for_plateau(D, fc, target_dec=0.40):
    """Order-of-magnitude N at which the admissible range would be target_dec
    wide, extrapolating the fine-grid chord below the depletion floor."""
    if not np.isfinite(fc["chord"]) or fc["span"] <= 0:
        return float("nan")
    dt = L.typical_scale(D)
    N = D.shape[0]
    eps = dt * np.geomspace(0.15, 1.7, 900)
    C, _ = L.correlation_curve(D, eps, theiler=0)
    e = eps / dt
    m = (C >= 10.0 / N) & (C <= 0.1)
    b, a = np.polyfit(np.log(e[m]), np.log(C[m]), 1)
    e_tgt = e[m][-1] / 10 ** target_dec
    C_tgt = np.exp(a) * e_tgt ** b
    return float(10.0 / C_tgt)


def log_ticks(ax):
    ax.set_xlim(8e-3, 3.0)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ overview
def fig_overview(pipe, names, path):
    fig, axs = plt.subplots(2, 4, figsize=(17.0, 7.4), sharex="col")
    for j, name in enumerate(names):
        D = load_D(name)
        n = D.shape[0]
        r = pipe[name]
        for i, W in enumerate((0, 50, 500)):
            e, C, s, _ = curve(D, theiler=W)
            axs[0, j].loglog(e, C, color=COL[i], lw=1.25, label=f"W={W}")
            axs[1, j].semilogx(e, s, color=COL[i], lw=1.05, label=f"W={W}")
        q = r["theiler"]["0"]
        dep, sat = q["eps_depletion"], q["eps_saturation"]
        ax = axs[0, j]
        ax.axhline(10.0 / n, color=C_GREY, ls=":", lw=1.0)
        ax.axhline(0.1, color=C_GREY, ls=":", lw=1.0)
        ax.axvspan(dep, sat, color=C_GREY, alpha=0.16, lw=0)
        ax.set_title(f"{name}   N={n}")
        ax.set_ylabel(r"$C(\epsilon)$")
        ax.text(dep, 1.6e-2, r"$10/N$", color=C_GREY, fontsize=7, ha="center",
                bbox=BOX)
        ax.text(sat, 0.13, r"$C=0.1$", color=C_GREY, fontsize=7, ha="center",
                bbox=BOX)
        if j == 0:
            ax.legend(loc="lower right", title="Theiler", fontsize=7)

        fc = fine_chord(D)
        ax = axs[1, j]
        ax.axhline(r["slope_ceiling_2log10N"], color=C_CHORD, ls=":", lw=1.0,
                   label=r"$2\log_{10}N$")
        ax.axhline(fc["chord"], color=C_GREY, ls="-.", lw=1.0,
                   label="chord over admissible")
        for xv in (dep, sat):
            ax.axvline(xv, color=C_GREY, ls=":", lw=0.9)
        log_ticks(ax)
        ax.set_xlabel(r"$\epsilon/d_{typ}$")
        ax.set_ylabel(r"$d\log C/d\log\epsilon$")
        ax.set_ylim(0, 22.0)
        txt = (f"no scaling region\n"
               f"window 0.00 dec\n"
               f"admissible {fc['span']:.2f} dec\n"
               f"chord $D_2$ {fc['chord']:.2f}\n"
               f"T4 max-$R^2$ {r['t4_style']['d2']:.2f}\n"
               f"TwoNN {r['twonn']['mle']:.2f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=7.2, family="monospace", bbox=BOX)
        if j == 0:
            ax.text(0.97, 0.02,
                    "period-7 dip at lag 7,\nlast 200 states,\ncoverage 14%",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6.8, color=C_ORANGE, bbox=BOX)
    fig.suptitle(
        "Qwen3.5-9B (FP32, greedy, no window, last layer $d=4096$, 1000 tokens): "
        r"$C(\epsilon)$ (top) and local slope (bottom). Grey band = measurable "
        "range $10/N\\to C=0.1$. No valid window in any run.", fontsize=10.4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------- summary
def fig_summary(pipe, names, ndep, path):
    fig = plt.figure(figsize=(16.8, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.11,
                          wspace=0.30, left=0.05, right=0.985, bottom=0.20,
                          top=0.87)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c1 = fig.add_subplot(gs[0, 2])
    ax_c2 = fig.add_subplot(gs[1, 2], sharex=ax_c1)

    # (a) TwoNN, chord and T4, sorted by TwoNN
    ax = ax_a
    rows = sorted((pipe[n]["twonn"]["mle"], n,
                   ind["chord"], ind["t4"]) for n, ind in ndep["individual"].items())
    x = np.arange(len(rows))
    tw = np.array([r[0] for r in rows])
    ch = np.array([r[2] for r in rows])
    t4 = np.array([r[3] for r in rows])
    ax.axhspan(*T15_TWONN, color=C_GREY, alpha=0.20, lw=0,
               label="1.5B TwoNN 5.4-6.2")
    ax.axhspan(*T15_HEALTHY_D2, color=C_OK, alpha=0.10, lw=0,
               label="1.5B healthy T4 9.2-11.4")
    ax.axhspan(*DS14_TWONN, color="#4393c3", alpha=0.10, lw=0,
               label="14B TwoNN 4.7-8.9")
    ax.plot(x, tw, "o", ms=6, color=C_OK, label="9B TwoNN $1/\\langle\\ln\\mu\\rangle$")
    ax.plot(x, ch, "s", ms=5.5, mfc="none", mec=C_CHORD, mew=1.1,
            label="9B chord (fine grid)")
    ax.plot(x, t4, "x", ms=5.5, color=C_T4, mew=1.1,
            label="9B T4 max-$R^2$")
    for xi, (a, b, c) in enumerate(zip(tw, ch, t4)):
        dy = -0.85 if abs(b - a) < 1.0 else 0.40
        ax.text(xi, a + dy, f"{a:.2f}", ha="center", fontsize=6.0, color=C_OK)
        ax.text(xi, c + 0.35, f"{c:.0f}", ha="center", fontsize=5.9,
                color=C_T4)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], rotation=38, ha="right", fontsize=7.2)
    ax.set_ylabel(r"$\hat D$ / TwoNN")
    ax.set_ylim(0, 25.0)
    ax.set_title("(a) 9B: 0/12 valid windows; point-cloud values")
    ax.legend(loc="lower right", fontsize=5.6, ncol=2)
    ax.text(0.02, 0.985, "chord spans 0.07-0.16 dec\n"
                         "T4 spans 0.013-0.066 dec",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.7,
            color=C_CHORD, bbox=BOX)

    # (b) loop screen
    ax = ax_b
    rows = sorted((pipe[n]["period"]["period_prominence"],
                   pipe[n]["period"]["coverage"], n) for n in names)
    x = np.arange(len(rows))
    pr = np.array([r[0] for r in rows])
    cv = np.array([r[1] for r in rows])
    cols = [C_ORANGE if p == pr.min() else C_OK for p in pr]
    ax.axhspan(0.0, T15_LOOP_PROM, color=C_LOOP, alpha=0.14, lw=0)
    ax.axhline(0.6, color=C_LOOP, ls="--", lw=1.2)
    ax.axhline(0.8, color=C_ORANGE, ls="--", lw=0.9)
    ax.axhline(T15_HEALTHY_PROM, color=C_GREY, ls=":", lw=1.1,
               label="1.5B healthy 0.91")
    ax.axhline(DS14_PROM[0], color="#4393c3", ls=":", lw=1.0,
               label="14B 0.961")
    ax.bar(x, pr, color=cols, width=0.68)
    for xi, (p, c) in enumerate(zip(pr, cv)):
        ax.text(xi, p + 0.010, f"{p:.3f}", ha="center", va="bottom",
                fontsize=5.8, rotation=90)
        if c > 0.01:
            ax.text(xi, 0.04, f"cov\n{100*c:.0f}%", ha="center", va="bottom",
                    fontsize=6.2, color=C_ORANGE, fontweight="bold")
    ax.text(len(rows) - 0.45, 0.50, "loop cut 0.6", color=C_LOOP, fontsize=7,
            ha="right", bbox=BOX)
    ax.text(len(rows) - 0.45, 0.705, "partial 0.8", color=C_ORANGE,
            fontsize=7, ha="right", bbox=BOX)
    ax.text(0.03, 0.47, "1.5B loops $\\leq0.45$", color=C_LOOP,
            fontsize=6.8, ha="left", bbox=BOX)
    ax.legend(loc="upper left", fontsize=6.2, ncol=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[2] for r in rows], rotation=38, ha="right", fontsize=7.2)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("lag-dip prominence (lower = loop)")
    ax.set_title("(b) loop screen: one partial period-7 repeat")
    ax.text(0.02, 0.02,
            "great_emu_war: prominence 0.849,\n"
            "period-7 dip in states 800-990 only,\n"
            "13.8% of positions within 0.1 $d_{typ}$",
            transform=ax.transAxes, fontsize=6.6, va="bottom", ha="left",
            color=C_ORANGE, bbox=BOX)

    # (c) N dependence, split in TwoNN and chord D2
    ns = [500, 750, 1000]
    cols = (C_OK, C_LOOP)
    for k, (name, col) in enumerate(zip(ndep["names"], cols)):
        d = ndep["series"][name]
        ax_c1.plot(ns, [d[str(n)]["twonn"] for n in ns], "o-", color=col,
                   lw=1.5, label=name)
        ax_c2.plot(ns, [d[str(n)]["chord"] for n in ns], "s--",
                   color=col, lw=1.2, mfc="none", label=name)
    ax_c1.axhspan(*T15_TWONN, color=C_GREY, alpha=0.18, lw=0)
    ax_c1.axhspan(*DS14_TWONN, color="#4393c3", alpha=0.12, lw=0)
    ax_c1.text(520, 5.62, "1.5B TwoNN", color=C_GREY, fontsize=6.6, bbox=BOX)
    ax_c1.text(520, 8.62, "14B TwoNN", color="#4393c3", fontsize=6.6, bbox=BOX)
    for nm, col in (("great_emu_war", C_OK), ("robot_friendship", C_LOOP)):
        ax_c1.text(1010, ndep["series"][nm]["1000"]["twonn"] + 0.25, nm,
                   color=col, fontsize=6.4, ha="right")
        ax_c2.text(1010, ndep["series"][nm]["1000"]["chord"] + 0.55, nm,
                   color=col, fontsize=6.4, ha="right")
    ax_c1.set_ylabel("TwoNN")
    ax_c2.set_ylabel("chord $D_2$ (fine grid)")
    ax_c2.set_xlabel("N (nested prefixes)")
    ax_c2.set_xticks(ns)
    ax_c1.set_xlim(440, 1080)
    ax_c2.set_xlim(440, 1080)
    ax_c1.set_ylim(4.0, 10.2)
    ax_c2.set_ylim(5.0, 17.0)
    ax_c1.tick_params(labelbottom=False)
    ax_c1.set_title("(c) N dependence: no window at any N")
    ax_c2.text(0.03, 0.06, "valid window 0.00 dec at every N\n"
                           "admissible span 0.05 -> 0.16 dec",
               transform=ax_c2.transAxes, fontsize=6.6, family="monospace",
               bbox=BOX)
    ax_c2.text(0.97, 0.30,
               "great_emu_war: the drop at N=1000 is the\n"
               "period-7 repeat entering at t$\\approx$800,\n"
               "not finite-N bias",
               transform=ax_c2.transAxes, fontsize=6.4, ha="right", va="top",
               color=C_ORANGE, bbox=BOX)

    fig.suptitle("Qwen3.5-9B D2: estimates, loop screen, N dependence "
                 "(1.5B and DS-14B reference values marked)", fontsize=10.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------- time shuffle
def fig_shuffle(pipe, names, out, path):
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.6))
    rng = np.random.default_rng(20261121)
    handles = None
    for j, name in enumerate(names):
        D = load_D(name)
        n = D.shape[0]
        perm = rng.permutation(n)
        Dp = D[np.ix_(perm, perm)]
        rec = {}
        ax = axs[j]
        for i, W in enumerate((0, 50, 500)):
            e, C, s, _ = curve(D, theiler=W)
            ep, Cp, sp, _ = curve(Dp, theiler=W)
            ax.semilogx(e, s, color=COL[i], lw=1.35, label=f"raw W={W}")
            ax.semilogx(ep, sp, color=COL[i], lw=1.0, ls="--",
                        label=f"shuffled W={W}")
        # direct comparison on the identical eps grid: W=0 is exactly invariant
        dt = L.typical_scale(D)
        eps = dt * np.geomspace(L.EPS_LO, L.EPS_HI, L.N_EPS)
        for W in (0, 50, 500):
            Cr, _ = L.correlation_curve(D, eps, theiler=W)
            Cs, _ = L.correlation_curve(Dp, eps, theiler=W)
            er, sr = L.local_slope(eps, Cr)
            es, ss = L.local_slope(eps, Cs)
            vr = win_from_C(n, er / dt, Cr[Cr > 0], sr)
            vs = win_from_C(n, es / dt, Cs[Cs > 0], ss)
            rec[str(W)] = dict(
                d2_raw=vr["d2_admissible"], dec_raw=vr["decades_admissible"],
                d2_shuf=vs["d2_admissible"], dec_shuf=vs["decades_admissible"],
                chord_raw=fine_chord(D, d_typ=dt, theiler=W)["chord"],
                chord_shuf=fine_chord(Dp, d_typ=dt, theiler=W)["chord"],
                max_abs_dC=float(np.max(np.abs(Cr - Cs))),
                max_rel_dC=float(np.max(np.abs(Cr - Cs) / np.maximum(Cr, 1e-12))),
                valid_raw=bool(vr["valid"]), valid_shuf=bool(vs["valid"]),
            )
        rec["twonn_raw"] = float(L.twonn(D)["mle"])
        rec["twonn_shuf"] = float(L.twonn(Dp)["mle"])
        rec["permutation_seed"] = 20261121
        out["shuffle"][name] = rec

        q = pipe[name]
        dep, sat = q["theiler"]["0"]["eps_depletion"], q["theiler"]["0"]["eps_saturation"]
        for xv in (dep, sat):
            ax.axvline(xv, color=C_GREY, ls=":", lw=0.9)
        ax.axhline(q["slope_ceiling_2log10N"], color=C_CHORD, ls=":", lw=0.9)
        log_ticks(ax)
        ax.set_xlabel(r"$\epsilon/d_{typ}$")
        ax.set_ylabel(r"$d\log C/d\log\epsilon$")
        ax.set_ylim(0, 22.0)
        ax.set_title(f"{name}: raw vs index-shuffled (same point cloud)")
        txt = ("chord $D_2$ raw / shuffled\n"
               f"  W=0    {rec['0']['chord_raw']:5.2f} / {rec['0']['chord_shuf']:5.2f}\n"
               f"  W=50   {rec['50']['chord_raw']:5.2f} / {rec['50']['chord_shuf']:5.2f}\n"
               f"  W=500  {rec['500']['chord_raw']:5.2f} / {rec['500']['chord_shuf']:5.2f}\n"
               f"TwoNN {rec['twonn_raw']:.3f} / {rec['twonn_shuf']:.3f}\n"
               "W=0 exact: $\\Delta C=0$")
        ax.text(0.36, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=7.0, family="monospace", bbox=BOX)
        if j == 0:
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=6.8,
                   bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Point-cloud control: permuting the time indices leaves the "
                 "point cloud unchanged; only the Theiler exclusion moves.",
                 fontsize=10.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    pipe, _ = load_pipe()
    names = sorted(pipe)
    os.makedirs(FIG, exist_ok=True)
    extra = {"shuffle": {}, "ndep": {}, "individual": {}}

    with open(os.path.join(D2DIR, "summary.json")) as fh:
        gpu = {r["name"]: r for r in json.load(fh)}

    numbers = dict(
        tag=TAG,
        model="Qwen3.5-9B",
        n=1000, dim=4096,
        source="results_big/qwen35/d2 (dist_<name>_h_last.npy + summary.json)",
        pipeline_json=os.path.basename(PIPE),
        references=dict(
            t15=dict(twonn=T15_TWONN, loop_d2=T15_LOOP_D2,
                     healthy_t4=T15_HEALTHY_D2, old=T15_OLD,
                     loop_prom=T15_LOOP_PROM, healthy_prom=T15_HEALTHY_PROM),
            ds14=dict(twonn=DS14_TWONN, twonn_median=DS14_TWONN_MED,
                      chord=DS14_CHORD, t4=DS14_T4, prominence=DS14_PROM),
        ),
        trajectories={}, ndep=None, shuffle=None,
    )

    for name in names:
        r = pipe[name]
        D = load_D(name)
        fc = fine_chord(D)
        g = gpu.get(name, {})
        q0 = r["theiler"]["0"]
        extra["individual"][name] = dict(chord=fc["chord"], t4=r["t4_style"]["d2"])
        numbers["trajectories"][name] = dict(
            n=r["n"], d_typ=r["d_typ"],
            distance=dist_shape(D),
            lag1_rel=float(np.diagonal(D, 1).mean() / r["d_typ"]),
            n_for_plateau=n_for_plateau(D, fc),
            twonn=dict(mle=r["twonn"]["mle"], cdf_fit=r["twonn"]["cdf_fit"],
                       log2_bug=r["twonn"]["log2_bug"], n=r["twonn"]["n"],
                       n_dropped=r["twonn"]["n_dropped"]),
            mle={k: v["d2"] for k, v in r["mle"].items()},
            chord=fc,
            pipeline_chord=dict(d2=q0["d2_admissible"],
                                decades=q0["decades_admissible"]),
            admissible=dict(eps_depletion=q0["eps_depletion"],
                            eps_saturation=q0["eps_saturation"],
                            slope_at_depletion=q0["slope_at_depletion"],
                            slope_at_saturation=q0["slope_at_saturation"],
                            slope_min=q0["slope_min"], slope_max=q0["slope_max"]),
            window=dict(d2=q0["d2"], d2_flat=q0["d2_flat"],
                        decades=q0["decades"], valid=q0["valid"],
                        eps_lo=q0["eps_lo"], eps_hi=q0["eps_hi"]),
            t4={k: r["t4_style"][k] for k in
                ("d2", "r2", "fit_decades", "n_fit_points", "pairs_in_window")},
            period=r["period"], rep_screen=r["rep_screen"],
            gpu_screen=g,
            verdict=r["verdict"],
        )

    overview = ["great_emu_war", "healthy_habits", "interstellar",
                "robot_friendship"]
    fig_overview(pipe, overview, os.path.join(FIG, f"{TAG}_overview.png"))

    ndep_names = ["great_emu_war", "robot_friendship"]
    series = {}
    for name in ndep_names:
        D = load_D(name)
        series[name] = {}
        for n in (500, 750, 1000):
            v = window_at(D, n, 0)
            fc = fine_chord(D, n)
            series[name][str(n)] = dict(
                twonn=float(L.twonn(D[:n, :n])["mle"]),
                chord=float(fc["chord"]),
                chord_span=float(fc["span"]),
                d2_admissible=float(v["d2_admissible"]),
                decades_admissible=float(v["decades_admissible"]),
                window_decades=float(v["decades"]),
                valid=bool(v["valid"]),
                t4=float(L._t4_d2_curve(D[:n, :n])["d2"]),
            )
    extra["ndep"] = {"names": ndep_names, "series": series, "individual": extra["individual"]}

    fig_summary(pipe, names, extra["ndep"], os.path.join(FIG, f"{TAG}_summary.png"))
    fig_shuffle(pipe, ["interstellar", "great_emu_war"], extra,
                os.path.join(FIG, f"{TAG}_time_shuffle.png"))

    numbers["ndep"] = {"names": ndep_names, "series": series}
    numbers["shuffle"] = extra["shuffle"]
    with open(os.path.join(FIG, f"{TAG}_numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, default=float)
    print("wrote", TAG, "figures and", f"{TAG}_numbers.json", flush=True)


if __name__ == "__main__":
    main()
