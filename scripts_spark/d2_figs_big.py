#!/usr/bin/env python
"""The three notebook figures for the DS-14B D2 section.

Consumes results_big/ds14/d2 (the GPU distance matrices + summary.json) and the
pipeline JSON written by ``d2_local.py --tag ds14d2``.  All estimators are the
validated ones from ``d2_local.py``; this script only arranges them.

    python scripts_spark/d2_figs_big.py

Outputs (notebook_big/figures):
    ds14d2_overview.png       C(eps) + local slope, 4 trajectories
    ds14d2_summary.png        (a) D2/TwoNN  (b) loop screen  (c) N dependence
    ds14d2_time_shuffle.png   raw vs index-shuffled distance matrix, 2 runs
    ds14d2_extra.json         the numbers behind (c) and the shuffle control
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
D2DIR = os.path.join(ROOT, "results_big", "ds14", "d2")
FIG = os.path.join(ROOT, "notebook_big", "figures")
PIPE = os.path.join(FIG, "d2pipe_ds14d2.json")
TAG = "ds14d2"

COL = ["#2166ac", "#d6604d", "#1b7837"]
C_GREY = "#8a8a8a"
C_LOOP = "#b2182b"
C_OK = "#2166ac"
C_CHORD = "#762a83"
BOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78)

# 1.5B reference numbers (notebook_big/sections/section_d2_pipeline.md)
T15_TWONN = (5.44, 6.23)          # four validated runs, loops included
T15_LOOP_D2 = (2.978, 3.894)      # T4 max-R^2 chords, period-53/69 loops
T15_HEALTHY_D2 = (9.195, 11.353)  # T4 chords, no plateau (0.08-0.10 dec)
T15_OLD = 5.75                    # docs/MEASUREMENTS.md raw D2 (shuffled 5.69)
T15_LOOP_PROM = 0.45              # worst 1.5B full-loop prominence
T15_HEALTHY_PROM = 0.91           # best 1.5B healthy prominence


def load_pipe():
    with open(PIPE) as fh:
        pipe = json.load(fh)
    return {t["name"]: t for t in pipe["trajectories"]}, pipe


def load_D(name, n=1000):
    D = np.load(os.path.join(D2DIR, f"dist_{name}_h_last.npy"))
    return np.asarray(D[:n, :n], dtype=np.float32)


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

        ax = axs[1, j]
        ax.axhline(r["slope_ceiling_2log10N"], color=C_CHORD, ls=":", lw=1.0,
                   label=r"$2\log_{10}N$")
        ax.axhline(q["d2_admissible"], color=C_GREY, ls="-.", lw=1.0,
                   label="chord over admissible")
        for xv in (dep, sat):
            ax.axvline(xv, color=C_GREY, ls=":", lw=0.9)
        log_ticks(ax)
        ax.set_xlabel(r"$\epsilon/d_{typ}$")
        ax.set_ylabel(r"$d\log C/d\log\epsilon$")
        ax.set_ylim(0, max(11.0, 1.35 * q["d2_admissible"]))
        txt = (f"no scaling region\n"
               f"window 0.00 dec\n"
               f"admissible {q['decades_admissible']:.2f} dec\n"
               f"chord $D_2$ {q['d2_admissible']:.2f}\n"
               f"TwoNN {r['twonn']['mle']:.2f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=7.4, family="monospace", bbox=BOX)
        if j == 0:
            ax.legend(loc="lower right", fontsize=6.8)
    fig.suptitle(
        "DS-14B (FP32, greedy, W=4096, last layer, 1000 tokens): "
        r"$C(\epsilon)$ (top) and local slope (bottom). "
        "Grey band = measurable range $10/N\\to C=0.1$. "
        "No trajectory has a valid window.", fontsize=10.2)
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

    # (a) TwoNN and D2, sorted
    ax = ax_a
    rows = sorted((pipe[n]["twonn"]["mle"], n,
                   pipe[n]["theiler"]["0"]["d2_admissible"],
                   pipe[n]["theiler"]["0"]["decades_admissible"])
                  for n in names)
    x = np.arange(len(rows))
    tw = np.array([r[0] for r in rows])
    ch = np.array([r[2] for r in rows])
    dec = np.array([r[3] for r in rows])
    ax.axhspan(*T15_TWONN, color=C_GREY, alpha=0.20, lw=0,
               label="1.5B TwoNN 5.4-6.2")
    ax.axhspan(*T15_LOOP_D2, color=C_LOOP, alpha=0.16, lw=0,
               label="1.5B loop $D_2$ 3.0-3.9")
    ax.axhspan(*T15_HEALTHY_D2, color=C_OK, alpha=0.12, lw=0,
               label="1.5B healthy T4 chord 9.2-11.4")
    ax.axhline(T15_OLD, color="k", ls="--", lw=1.1,
               label="1.5B old quote $D_2\\approx5.75$")
    ax.plot(x, tw, "o", ms=6, color=C_OK,
            label="14B TwoNN $1/\\langle\\ln\\mu\\rangle$")
    ax.plot(x, ch, "s", ms=5.5, mfc="none", mec=C_CHORD, mew=1.1,
            label="14B $D_2$ chord over admissible range")
    for xi, (a, b) in enumerate(zip(tw, ch)):
        ax.text(xi, a + 0.24, f"{a:.2f}", ha="center", fontsize=6.3, color=C_OK)
        ax.text(xi, b - 0.50, f"{b:.1f}", ha="center", fontsize=6.1,
                color=C_CHORD)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], rotation=38, ha="right", fontsize=7.2)
    ax.set_ylabel(r"$\hat D$ / TwoNN")
    ax.set_ylim(0, 13.6)
    ax.set_title("(a) 14B: 0/13 valid windows; point-cloud values")
    ax.legend(loc="upper left", fontsize=6.2)
    ax.text(0.99, 0.02, f"chord windows {dec.min():.2f}-{dec.max():.2f} dec\n"
                        "no value to plot as a plateau",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=C_CHORD, bbox=BOX)

    # (b) loop screen
    ax = ax_b
    rows = sorted((pipe[n]["period"]["period_prominence"],
                   pipe[n]["period"]["coverage"], n) for n in names)
    x = np.arange(len(rows))
    pr = np.array([r[0] for r in rows])
    ax.axhspan(0.0, T15_LOOP_PROM, color=C_LOOP, alpha=0.14, lw=0)
    ax.axhline(0.6, color=C_LOOP, ls="--", lw=1.2)
    ax.axhline(0.8, color="#e08214", ls="--", lw=0.9)
    ax.axhline(T15_HEALTHY_PROM, color=C_GREY, ls=":", lw=1.1)
    ax.bar(x, pr, color=C_OK, width=0.68)
    for xi, p in enumerate(pr):
        ax.text(xi, p + 0.010, f"{p:.2f}", ha="center", va="bottom",
                fontsize=6.2, rotation=90)
    ax.text(len(rows) - 0.45, 0.50, "loop cut 0.6", color=C_LOOP, fontsize=7,
            ha="right", bbox=BOX)
    ax.text(len(rows) - 0.45, 0.705, "partial 0.8", color="#e08214",
            fontsize=7, ha="right", bbox=BOX)
    ax.text(len(rows) - 0.45, 0.925, "1.5B healthy 0.91", color=C_GREY,
            fontsize=6.8, ha="right", bbox=BOX)
    ax.text(-0.35, 0.23, "1.5B loops $\\leq0.45$", color=C_LOOP,
            fontsize=6.8, bbox=BOX)
    ax.set_xticks(x)
    ax.set_xticklabels([r[2] for r in rows], rotation=38, ha="right", fontsize=7.2)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("lag-dip prominence (lower = loop)")
    ax.set_title("(b) loop screen: no dip in any trajectory")
    ax.text(0.99, 0.02, "coverage 0% in all 13",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
            bbox=BOX)

    # (c) N dependence, split in TwoNN and chord D2
    ns = [500, 750, 1000]
    cols = (C_OK, C_LOOP)
    for k, (name, col) in enumerate(zip(ndep["names"], cols)):
        d = ndep["series"][name]
        ax_c1.plot(ns, [d[str(n)]["twonn"] for n in ns], "o-", color=col,
                   lw=1.5, label=name)
        ax_c2.plot(ns, [d[str(n)]["d2_admissible"] for n in ns], "s--",
                   color=col, lw=1.2, mfc="none", label=name)
    ax_c1.axhspan(*T15_TWONN, color=C_GREY, alpha=0.18, lw=0)
    ax_c1.text(520, 5.62, "1.5B TwoNN", color=C_GREY, fontsize=6.6, bbox=BOX)
    ax_c1.text(1010, ndep["series"]["great_emu_war"]["1000"]["twonn"] + 0.25,
               "great_emu_war", color=C_OK, fontsize=6.6, ha="right")
    ax_c1.text(1010, ndep["series"]["interstellar"]["1000"]["twonn"] + 0.25,
               "interstellar", color=C_LOOP, fontsize=6.6, ha="right")
    ax_c2.text(1010, ndep["series"]["great_emu_war"]["1000"]["d2_admissible"] + 0.06,
               "great_emu_war", color=C_OK, fontsize=6.6, ha="right")
    ax_c2.text(1010, ndep["series"]["interstellar"]["1000"]["d2_admissible"] - 0.16,
               "interstellar", color=C_LOOP, fontsize=6.6, ha="right")
    ax_c1.set_ylabel("TwoNN")
    ax_c2.set_ylabel("chord $D_2$")
    ax_c2.set_xlabel("N (nested prefixes)")
    ax_c2.set_xticks(ns)
    ax_c1.set_xlim(440, 1080)
    ax_c2.set_xlim(440, 1080)
    ax_c1.set_ylim(4.0, 10.2)
    ax_c2.set_ylim(9.0, 10.8)
    ax_c1.tick_params(labelbottom=False)
    ax_c1.set_title("(c) N dependence: no window at any N")
    ax_c2.text(0.03, 0.06, "valid window 0.00 dec at every N\n"
                           "admissible span 0.05 -> 0.08 dec\n"
                           "interstellar chord undefined at N=500",
               transform=ax_c2.transAxes, fontsize=6.6, family="monospace",
               bbox=BOX)

    fig.suptitle("DS-14B D2: estimates, repetition screen, N dependence "
                 "(1.5B reference values marked)", fontsize=10.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------- time shuffle
def fig_shuffle(pipe, names, out, path):
    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.4))
    rng = np.random.default_rng(20261121)
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
        ax.set_ylim(0, 12.0)
        ax.set_title(f"{name}: raw vs index-shuffled (same point cloud)")
        txt = ("W=0: $\\Delta C\\equiv0$, TwoNN identical\n"
               f"  chord $D_2$ raw/shuf  {rec['0']['d2_raw']:.2f}/{rec['0']['d2_shuf']:.2f}\n"
               f"W=50:           {rec['50']['d2_raw']:.2f}/{rec['50']['d2_shuf']:.2f}\n"
               f"W=500:          {rec['500']['d2_raw']:.2f}/{rec['500']['d2_shuf']:.2f}\n"
               f"TwoNN raw/shuf  {rec['twonn_raw']:.3f}/{rec['twonn_shuf']:.3f}\n"
               "valid window  none in either")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
                fontsize=7.2, family="monospace", bbox=BOX)
        ax.legend(loc="lower right", fontsize=6.2, ncol=2)
    fig.suptitle("Point-cloud control: permuting the time indices leaves the "
                 "point cloud unchanged; only the Theiler exclusion moves.",
                 fontsize=10.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    pipe, _ = load_pipe()
    names = sorted(pipe)
    os.makedirs(FIG, exist_ok=True)
    extra = {"shuffle": {}, "ndep": {}}

    overview = ["great_emu_war", "healthy_habits", "interstellar", "transformers"]
    fig_overview(pipe, overview, os.path.join(FIG, f"{TAG}_overview.png"))

    # N dependence for two trajectories: lowest TwoNN and the interstellar run
    ndep_names = ["great_emu_war", "interstellar"]
    series = {}
    for name in ndep_names:
        D = load_D(name)
        series[name] = {}
        for n in (500, 750, 1000):
            v = window_at(D, n, 0)
            series[name][str(n)] = dict(
                twonn=float(L.twonn(D[:n, :n])["mle"]),
                d2_admissible=float(v["d2_admissible"]),
                decades_admissible=float(v["decades_admissible"]),
                window_decades=float(v["decades"]),
                valid=bool(v["valid"]),
            )
    extra["ndep"] = {"names": ndep_names, "series": series}

    fig_summary(pipe, names, extra["ndep"], os.path.join(FIG, f"{TAG}_summary.png"))
    fig_shuffle(pipe, ["interstellar", "great_emu_war"], extra,
                os.path.join(FIG, f"{TAG}_time_shuffle.png"))

    with open(os.path.join(FIG, f"{TAG}_extra.json"), "w") as fh:
        json.dump(extra, fh, indent=1)
    print("wrote", TAG, "figures;", f"{TAG}_extra.json", flush=True)


if __name__ == "__main__":
    main()
