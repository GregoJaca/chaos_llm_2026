#!/usr/bin/env python
"""The FP32 forward-pass rounding floor per saved layer (DS-R1-14B calibration).

Reads ``results_big/ds14/calib/calib.npz`` (and, for baseline state norms only,
``results_big/ds14/a1_inter_full/a1.npz``, same model / same prompt / same saved
layers).  Writes two figures and one JSON with every number quoted in
``notebook_big/sections/section_floor.md``.  No GPU, no torch.

Definitions
-----------
  r                 perturbation radius (9 values, 1e-4 .. 1)
  u                 direction index (8 directions)
  t                 generated step; all medians use t in [100, 400)
  l                 saved state index: 0 = embedding output, l = output of
                    block l-1, 47 = input to the final norm
  dh_norm[r,u,l,t]  ||h^l_t(u) - h^l_t(0)||: norm of the state displacement
  M[r,l]            median over u and t>=100 of dh_norm
  F_l               floor = M at r=1e-4 (the smallest radius)
                    (checked: M at 3e-4 agrees within ~1%, i.e. both are on the
                    plateau)
  G_l               gain = M at r=0.1 divided by 0.1
  r_min(l)          F_l / G_l: radius at which the linear response equals the
                    floor
  n_l               median over t>=100 of ||h^l_t(0)|| (baseline state norm),
                    taken from the a1 run because calib.npz stores no norms

Run from the repository root:
    python scripts_spark/figs_floor.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CALIB = os.path.join(ROOT, "results_big", "ds14", "calib", "calib.npz")
A1 = os.path.join(ROOT, "results_big", "ds14", "a1_inter_full", "a1.npz")
FIGDIR = os.path.join(ROOT, "notebook_big", "figures")

# dark -> light blue for the four non-zero states; state 0 is zero and only
# annotated, never drawn on a log axis
C_LAYERS = ["#a6cee3", "#6baed6", "#4292c6", "#2171b5", "#08306b"]
C_GREY = "#8a8a8a"
C_OBS = "#c0392b"
C_ALT = "#e67e22"
EPS32 = float(np.finfo(np.float32).eps)


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


def log_fit(x, y):
    """y = A * x**p in log space; returns p, A, rms log residual, max |log res|."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([np.ones_like(x), np.log(x)]).T
    c, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    pred = np.exp(c[0]) * x ** c[1]
    res = np.log(pred / y)
    return float(c[1]), float(np.exp(c[0])), float(np.sqrt(np.mean(res ** 2))), float(np.max(np.abs(res)))


def exp_fit(l, y):
    A = np.vstack([np.ones_like(l, dtype=float), np.asarray(l, float)]).T
    c, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    pred = np.exp(c[0]) * np.exp(c[1] * np.asarray(l, float))
    res = np.log(pred / y)
    return float(c[1]), float(np.exp(c[0])), float(np.sqrt(np.mean(res ** 2))), float(np.max(np.abs(res)))


def slope1_label(ax, x0, y0, x1, y1, text, color=C_GREY):
    """Label a line with the text rotated to its on-screen angle."""
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(text, xy=(x1, y1), xytext=(-4, -2), textcoords="offset points",
                rotation=ang, rotation_mode="anchor", ha="right", va="top",
                fontsize=7.6, color=color)


# --------------------------------------------------------------------- data
def main():
    d = np.load(CALIB)
    radii = d["radii"].astype(float)
    layers = d["layers"].astype(np.int64)
    dh = d["dh_norm"].astype(float)                  # [9, 8, 5, 400]
    T = dh.shape[-1]
    t0 = 100

    # M[r, l] = median over directions and over t >= 100
    M = np.array([[np.median(dh[ri, :, li, t0:]) for li in range(len(layers))]
                  for ri in range(len(radii))])

    # ---- 0. is the state-0 floor exactly zero, and is 1e-4 really a plateau?
    state0_max = float(np.max(dh[:, :, 0, :]))
    plateau_ratio = M[1] / np.where(M[0] > 0, M[0], np.nan)
    # direction spread of the floor: min/max of the per-direction medians
    dir_med = np.array([[np.median(dh[ri, u, li, t0:]) for u in range(dh.shape[1])]
                        for ri in range(len(radii)) for li in range(len(layers))])
    dir_med = dir_med.reshape(len(radii), len(layers), -1)
    dir_spread = dir_med[0].max(axis=1) / np.maximum(dir_med[0].min(axis=1), 1e-300)
    # time stability: median of the two halves of [100, 400)
    t_first = np.array([np.median(dh[0, :, li, 100:250]) for li in range(len(layers))])
    t_second = np.array([np.median(dh[0, :, li, 250:]) for li in range(len(layers))])
    t_ratio = t_second / np.where(t_first > 0, t_first, np.nan)

    F = M[0].copy()                                   # floor, by definition
    plateau = np.median(M[:2], axis=0)                # median of two smallest radii
    sig01 = M[6].copy()                               # signal at the campaign radius
    G = sig01 / radii[6]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_min = np.where(G > 0, F / G, 0.0)
        r_min3 = np.where(G > 0, 3.0 * F / G, 0.0)
        sig_over_floor = np.where(F > 0, sig01 / F, np.inf)

    # ---- per-t signal/floor ratio at the campaign radius
    per_t = {}
    for li, l in enumerate(layers):
        if l == 0:
            continue
        s = np.median(dh[6, :, li, t0:], axis=0)
        f = np.median(dh[0, :, li, t0:], axis=0)
        q = s / f
        per_t[int(l)] = {
            "median": float(np.median(q)),
            "p10": float(np.percentile(q, 10)),
            "p5": float(np.percentile(q, 5)),
            "min": float(q.min()),
            "t_at_min": int(t0 + np.argmin(q)),
            "frac_below_3": float(np.mean(q < 3.0)),
        }

    # ---- local log-log slope of the response curve
    slopes = np.full((len(radii) - 1, len(layers)), np.nan)
    for li in range(len(layers)):
        for ri in range(len(radii) - 1):
            if M[ri, li] > 0 and M[ri + 1, li] > 0:
                slopes[ri, li] = (np.log(M[ri + 1, li] / M[ri, li])
                                  / np.log(radii[ri + 1] / radii[ri]))

    # ---- baseline state norms (a1 run: same model, same prompt, same layers)
    norms = None
    norm_source = "absent"
    if os.path.exists(A1):
        a = np.load(A1)
        if "norm_h0" in a.files:
            al = a["layers"].astype(np.int64)
            nh = a["norm_h0"].astype(float)
            norms = np.array([np.median(nh[np.where(al == l)[0][0], t0:])
                              for l in layers])
            if np.array_equal(a["prompt_ids"], d["prompt_ids"]):
                norm_source = "a1_inter_full/a1.npz norm_h0 (identical prompt_ids)"
            else:
                norm_source = "a1_inter_full/a1.npz norm_h0 (prompt differs!)"

    # ---- depth law fits on the non-zero layers
    m = layers > 0
    l_ = layers[m].astype(float)
    f_ = F[m]
    fits = {}

    # (i) independent per-block rounding of equal absolute size -> sqrt(l)
    A = float(np.sum(f_ * np.sqrt(l_)) / np.sum(l_))
    pred = A * np.sqrt(l_)
    res = np.log(pred / f_)
    fits["sqrt_l"] = {"law": "F = A*sqrt(l)", "A": A,
                      "rms_log": float(np.sqrt(np.mean(res ** 2))),
                      "max_abs_log": float(np.max(np.abs(res))),
                      "pred": pred, "ratio": f_ / pred}

    # (ii) free power of l
    p, A, rms, mx = log_fit(l_, f_)
    fits["power_l"] = {"law": f"F = A*l^{p:.3f}", "p": p, "A": A,
                       "rms_log": rms, "max_abs_log": mx,
                       "pred": A * l_ ** p, "ratio": f_ / (A * l_ ** p)}

    # (iii) multiplicative growth, exp(c*l)
    c, A, rms, mx = exp_fit(l_, f_)
    fits["exp_l"] = {"law": f"F = A*exp({c:.4f}*l)", "c": c, "A": A,
                     "blocks_per_e": 1.0 / c,
                     "rms_log": rms, "max_abs_log": mx,
                     "pred": A * np.exp(c * l_), "ratio": f_ / (A * np.exp(c * l_))}

    # (iv) proportional to the baseline state norm (free exponent)
    if norms is not None:
        n_ = norms[m]
        p, A, rms, mx = log_fit(n_, f_)
        fits["power_n"] = {"law": f"F = A*n^{p:.3f}", "p": p, "A": A,
                           "rms_log": rms, "max_abs_log": mx,
                           "pred": A * n_ ** p, "ratio": f_ / (A * n_ ** p)}

        # (v) through-origin proportionality, the one-parameter version
        kappa = float(np.sum(f_ * n_) / np.sum(n_ ** 2))
        pred = kappa * n_
        res = np.log(pred / f_)
        fits["prop_n"] = {"law": "F = kappa*n", "kappa": kappa,
                          "kappa_over_eps": kappa / EPS32,
                          "rms_log": float(np.sqrt(np.mean(res ** 2))),
                          "max_abs_log": float(np.max(np.abs(res))),
                          "pred": pred, "ratio": f_ / pred}

        # (vi) quadrature accumulation, per-block noise proportional to the
        # block's own state norm.  Approximated on the 9-layer grid of the a1
        # run (6-block spacing), cumulative in l.
        an = np.load(A1)
        al = an["layers"].astype(np.int64).astype(float)
        grid_n = np.array([np.median(an["norm_h0"][i, t0:]) for i in range(len(al))])
        q = np.sqrt(np.cumsum(grid_n ** 2))
        q_at = np.array([q[np.where(al == l)[0][0]] for l in layers])
        kq = float(np.sum(f_ * q_at[m]) / np.sum(q_at[m] ** 2))
        pred = kq * q_at[m]
        res = np.log(pred / f_)
        fits["quadrature_n"] = {
            "law": "F = kq*sqrt(sum_b n_b^2)", "kq": kq,
            "kq_over_eps": kq / EPS32,
            "grid": "saved layers 0,6,...,47 of a1 (6-block spacing)",
            "rms_log": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_log": float(np.max(np.abs(res))),
            "pred": pred, "ratio": f_ / pred,
        }

    # ---- per-block variance increments between measured layers
    inc = []
    for i in range(1, len(layers) - 1):
        j = i + 1
        nb = int(layers[j] - layers[i])
        d_f2 = (F[j] ** 2 - F[i] ** 2) / nb
        inc.append({"l_from": int(layers[i]), "l_to": int(layers[j]),
                    "blocks": nb, "dF2_per_block": float(d_f2)})

    # ---- does the floor add in quadrature to the linear response?
    # pred(r) = sqrt(F_l^2 + (G_l*r)^2), no free parameter (F from r=1e-4,
    # G from r=0.1), tested at the other seven radii.
    quad_err = np.full((len(radii), len(layers)), np.nan)
    lin_err = np.full((len(radii), len(layers)), np.nan)
    for li, l in enumerate(layers):
        if l == 0:
            continue
        pq = np.sqrt(F[li] ** 2 + (G[li] * radii) ** 2)
        pl = F[li] + G[li] * radii
        quad_err[:, li] = M[:, li] / pq - 1.0
        lin_err[:, li] = M[:, li] / pl - 1.0
    win = radii <= radii[7]                      # r <= 0.3, before superlinearity
    qe = quad_err[win][:, layers > 0]
    le = lin_err[win][:, layers > 0]

    # ------------------------------------------------------------------ json
    out = {
        "protocol": {
            "npz": "results_big/ds14/calib/calib.npz",
            "model": "DeepSeek-R1-Distill-Qwen-14B",
            "dtype": "float32",
            "arch_blocks": 48,
            "prompt": "interstellar_travel",
            "P": int(d["prompt_ids"].size),
            "T": int(T),
            "t_window": [t0, T],
            "radii": radii,
            "layers": layers,
            "n_directions": int(dh.shape[1]),
            "state_convention": "0 = embedding output, l = output of block l-1, 47 = input to final norm",
            "campaign_radius": float(radii[6]),
            "fp32_eps": EPS32,
        },
        "median_dh": {"rows": "radius", "cols": "layer", "value": M},
        "state0": {"max_abs_dh": state0_max,
                   "statement": "identically zero at every (r,u,t): the perturbation touches only prompt positions"},
        "plateau_check": {
            "ratio_3e4_over_1e4": plateau_ratio,
            "ratio_1e3_over_1e4": M[2] / np.where(M[0] > 0, M[0], np.nan),
            "max_abs_dev_3e4": float(np.nanmax(np.abs(plateau_ratio - 1.0))),
            "direction_spread_at_1e4": dir_spread,
            "t_second_half_over_first_half": t_ratio,
        },
        "floor": {
            "definition": "F_l = median_{u, t>=100} ||dh^l_t(u)|| at r=1e-4",
            "F": F,
            "plateau_two_smallest": plateau,
            "relative_floor_no_norm": "not computable from this npz",
        },
        "local_loglog_slope": {
            "rows": "interval between consecutive radii",
            "r_lo": radii[:-1], "r_hi": radii[1:], "value": slopes,
        },
        "signal_over_floor_r0p1": {
            "signal": sig01, "gain_G": G, "ratio": sig_over_floor,
        },
        "per_t_ratio_r0p1": per_t,
        "depth_law": {
            "norms": None if norms is None else norms,
            "norm_source": norm_source,
            "npz_has_norm_h0": bool("norm_h0" in d.files),
            "fits": {k: {kk: vv for kk, vv in v.items()} for k, v in fits.items()},
            "per_block_increments": inc,
            "F_ratio_47_over_12": float(F[-1] / F[1]),
            "sqrt_l_predicted_ratio": float(np.sqrt(layers[-1] / layers[1])),
            "n_ratio_47_over_12": None if norms is None else float(norms[-1] / norms[1]),
        },
        "r_min": {
            "definition": "r_min(l) = F_l / G_l with G_l = median dh at r=0.1 divided by 0.1",
            "r_min": r_min,
            "r_min_3x_floor": r_min3,
        },
        "floor_plus_linear_model": {
            "model": "||dh||(r) = sqrt(F_l^2 + (G_l r)^2), no free parameter",
            "rel_err_quadrature": quad_err,
            "rel_err_linear_sum": lin_err,
            "rms_abs_quad_r_le_0p3": float(np.sqrt(np.mean(qe ** 2))),
            "max_abs_quad_r_le_0p3": float(np.max(np.abs(qe))),
            "rms_abs_linear_r_le_0p3": float(np.sqrt(np.mean(le ** 2))),
            "max_abs_linear_r_le_0p3": float(np.max(np.abs(le))),
            "note": "both statistics over r<=0.3 and the 4 non-zero layers (32 points)",
        },
    }

    # --------------------------------------------------------------- figure 1
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12.6, 5.0),
                                  gridspec_kw={"wspace": 0.26,
                                               "width_ratios": [1.35, 1.0]})

    # (a) response curves
    dcol = [C_LAYERS[i] for i in range(len(layers))]
    for li, l in enumerate(layers):
        if l == 0:
            continue
        ax.plot(radii, M[:, li], "-o", ms=3.6, lw=1.5,
                color=dcol[li], label=f"state $l={l}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"perturbation radius $r$   (embedding units)")
    ax.set_ylabel(r"median$_{u,t\geq100}\,\Vert\delta h^l_t(u)\Vert$")
    ax.set_title("(a) response vs radius: a plateau below $r\\approx3\\times10^{-4}$")

    # slope-1 reference, parked in the empty corner so it never hides data
    xs = np.array([1.5e-2, 2.0e-1])
    ys = 1.1e-2 * xs
    ax.plot(xs, ys, ls="--", lw=1.1, color=C_GREY, alpha=0.9, zorder=1,
            label="slope 1 reference")
    slope1_label(ax, xs[0], ys[0], xs[-1], ys[-1], "slope 1")
    ax.axvspan(radii[0] / 1.6, radii[1] * 1.5, color="#f2c14e", alpha=0.20, lw=0)
    ax.axvline(radii[6], color=C_OBS, ls=":", lw=1.2)
    ax.text(radii[6] * 0.72, 1.3e-3, r"campaign $r=0.1$", color=C_OBS,
            fontsize=8, rotation=90, va="bottom", ha="right")
    ax.text(1.1e-4, 1.05e-3, "plateau\n$r\\leq3\\times10^{-4}$", fontsize=8,
            color="#8a6d1a", va="top")
    ax.text(1.1e-4, 1.32e-4,
            "state $l=0$: $\\Vert\\delta h\\Vert=0$ exactly, every $r,t$ (not drawn)",
            fontsize=8, color="#444444", va="bottom")
    ax.set_ylim(1.15e-4, 1.0)
    ax.legend(loc="upper left", ncol=1)

    # (b) floor vs layer index
    lf = layers[m].astype(float)
    axb.plot(lf, F[m], "o", ms=7, color=C_OBS, zorder=4,
             label=r"$F_l$ (measured, $r=10^{-4}$)")
    axb.plot(lf, plateau[m], "s", ms=5, mfc="white", mew=1.3, color=C_OBS,
             zorder=4, label=r"median of $r=10^{-4},3\times10^{-4}$")
    if norms is not None:
        an = np.load(A1)
        al = an["layers"].astype(int)
        grid_n = np.array([np.median(an["norm_h0"][i, t0:]) for i in range(len(al))])
        axb.plot(al, fits["prop_n"]["kappa"] * grid_n, "-", lw=1.1,
                 color="#1f4e79", alpha=0.85,
                 label=r"$2.61\times10^{-6}\,\Vert h^l_t\Vert$ (from a1 norms)")
    axb.set_yscale("log")
    axb.set_xlim(-2, 50)
    axb.set_ylim(1.25e-4, 8e-3)
    axb.set_xlabel("saved state index $l$   (0 = embedding, 47 = last)")
    axb.set_ylabel(r"floor $F_l$   (state units)")
    axb.set_title("(b) the floor accumulates with depth")
    axb.annotate("$l=0$: exactly 0\n(no arithmetic yet)", xy=(2.0, 1.45e-4),
                 fontsize=8, color="#444444", va="bottom")
    axb.legend(loc="upper left")
    for li, l in enumerate(layers):
        if l == 0:
            continue
        axb.annotate(f"{F[li]:.2e}", (l, F[li]), xytext=(5, 6),
                     textcoords="offset points", fontsize=7.4, color=C_OBS)

    fig.suptitle("FP32 rounding floor of the 14B: 0 at the embedding, "
                 "$2.7\\times10^{-3}$ at the last state, and proportional to the "
                 "state norm in between", fontsize=12.0, y=1.00)
    p1 = os.path.join(FIGDIR, "ds14floor_perlayer.png")
    fig.savefig(p1, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------- figure 2
    fig2 = plt.figure(figsize=(13.0, 4.9))
    gs = fig2.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.25], wspace=0.30)

    # (c) floor vs baseline state norm
    axc = fig2.add_subplot(gs[0, 0])
    if norms is not None:
        nn = norms[m]
        axc.loglog(nn, F[m], "o", ms=7, color=C_OBS, zorder=4)
        for l in lf.astype(int):
            axc.annotate(f"$l={l}$", (norms[np.where(layers == l)[0][0]],
                                      F[np.where(layers == l)[0][0]]),
                         xytext=(5, -3), textcoords="offset points",
                         fontsize=7.6)
        xs = np.array([70.0, 1500.0])
        axc.plot(xs, fits["power_n"]["A"] * xs ** fits["power_n"]["p"],
                 "-", lw=1.4, color="#1f4e79",
                 label=f"fit: $F\\propto n^{{{fits['power_n']['p']:.2f}}}$")
        axc.plot(xs, fits["power_n"]["A"] * xs, "--", lw=1.0, color=C_GREY,
                 label="slope 1")
        axc.set_xlabel(r"baseline state norm $\Vert h^l_t\Vert$ (median $t\geq100$)")
        axc.set_ylabel(r"floor $F_l$")
        axc.set_title("(c) floor $\u221d$ state norm")
        axc.legend(loc="upper left")
        axc.set_xlim(60, 1800)
        axc.set_ylim(1.4e-4, 5e-3)

    # (d) residuals of the candidate laws
    axd = fig2.add_subplot(gs[0, 1])
    laws = [("sqrt_l", r"$F\propto l^{1/2}$"),
            ("power_l", r"$F\propto l^{p}$"),
            ("exp_l", r"$F\propto e^{cl}$")]
    if norms is not None:
        laws += [("power_n", r"$F\propto n^{p}$"),
                 ("prop_n", r"$F\propto n$"),
                 ("quadrature_n", r"$F\propto\sqrt{\sum_b n_b^2}$")]
    rms = np.array([fits[k]["rms_log"] * 100 for k, _ in laws])
    mx = np.array([fits[k]["max_abs_log"] * 100 for k, _ in laws])
    yy = np.arange(len(laws))[::-1]
    axd.barh(yy, rms, height=0.62,
             color=[C_OBS if v > 25 else "#1f4e79" for v in rms], alpha=0.85)
    for y, v, m in zip(yy, rms, mx):
        if max(v, m) > 70:
            axd.text(3.0, y, f"{v:.0f}% (max {m:.0f}%)", va="center",
                     fontsize=7.8, color="white", fontweight="bold")
        else:
            axd.text(max(v, m) + 2.5, y, f"{v:.0f}% (max {m:.0f}%)",
                     va="center", fontsize=7.8)
    axd.set_yticks(yy)
    axd.set_yticklabels([lab for _, lab in laws])
    axd.axvline(10.0, color=C_GREY, ls=":", lw=1.0)
    axd.text(11.0, yy[0] + 0.45, "10%", fontsize=7.6, color=C_GREY)
    axd.set_xlim(0, 135)
    axd.set_xlabel("rms log residual over the 4 measured layers  (%)")
    axd.set_title("(d) which depth law fits")
    axd.grid(axis="y", visible=False)

    # (e) per-step signal/floor ratio
    axe = fig2.add_subplot(gs[0, 2])
    for li, l in enumerate(layers):
        if l == 0:
            continue
        s = np.median(dh[6, :, li, t0:], axis=0)
        f = np.median(dh[0, :, li, t0:], axis=0)
        axe.semilogy(np.arange(t0, T), s / f, lw=1.0, color=dcol[li],
                     label=f"$l={l}$ (median {np.median(s / f):.1f})")
    axe.axhline(1.0, color=C_GREY, ls="--", lw=1.0)
    axe.axhline(3.0, color=C_GREY, ls=":", lw=1.0)
    axe.text(102, 3.3, "3x floor", fontsize=7.6, color=C_GREY)
    axe.set_xlabel("generated step $t$")
    axe.set_ylabel(r"signal$(r=0.1)$ / local floor$(r=10^{-4})$")
    axe.set_title("(e) the ratio step by step, not just its median")
    axe.legend(loc="lower left", ncol=2)
    axe.set_ylim(1.0, 120)

    fig2.suptitle("Depth law and the per-step check: the floor is a fixed fraction "
                  "of the state norm, and no step's $r=0.1$ signal is near it",
                  fontsize=12.0, y=1.00)
    p2 = os.path.join(FIGDIR, "ds14floor_law.png")
    fig2.savefig(p2, dpi=170, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------------------------------------------ json
    jp = os.path.join(FIGDIR, "ds14floor_numbers.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(jf(out), fh, indent=1)

    # ---------------------------------------------------------------- report
    print("median ||dh|| per (r, l)      layers:", layers.tolist())
    for ri, r in enumerate(radii):
        print(f"  r={r:<9.1e} " + "  ".join(f"{v:10.4g}" for v in M[ri]))
    print()
    print(f"state-0 max |dh| = {state0_max:.3e}")
    print("plateau check  M(3e-4)/M(1e-4):",
          "  ".join(f"{v:.3f}" for v in plateau_ratio))
    print("M(1e-3)/M(1e-4):",
          "  ".join(f"{v:.3f}" for v in M[2] / np.where(M[0] > 0, M[0], np.nan)))
    print("floor direction spread (max/min):",
          "  ".join(f"{v:.3f}" for v in dir_spread))
    print("floor second/first half of t:",
          "  ".join(f"{v:.3f}" for v in t_ratio))
    print()
    print("F_l        :", "  ".join(f"{v:.4e}" for v in F))
    print("plateau    :", "  ".join(f"{v:.4e}" for v in plateau))
    print("signal 0.1 :", "  ".join(f"{v:.4e}" for v in sig01))
    print("G_l        :", "  ".join(f"{v:.4e}" for v in G))
    print("sig/floor  :", "  ".join(("inf" if np.isinf(v) else f"{v:.2f}")
                                     for v in sig_over_floor))
    print("r_min      :", "  ".join(f"{v:.3e}" for v in r_min))
    print("r_min(3F)  :", "  ".join(f"{v:.3e}" for v in r_min3))
    print(f"quadrature model r<=0.3: rms {np.sqrt(np.mean(qe ** 2)) * 100:.2f}%  "
          f"max {np.max(np.abs(qe)) * 100:.2f}%   "
          f"(linear sum: rms {np.sqrt(np.mean(le ** 2)) * 100:.1f}%, "
          f"max {np.max(np.abs(le)) * 100:.1f}%)")
    print()
    if norms is not None:
        print("norms      :", "  ".join(f"{v:.4g}" for v in norms))
        print("F/n        :", "  ".join(("nan" if not np.isfinite(v) else f"{v:.3e}")
                                        for v in F / np.where(norms > 0, norms, np.nan)))
    for k, v in fits.items():
        print(f"fit {k:<15s} {v['law']:<28s} rms(log)={v['rms_log']:.4f} "
              f"max={v['max_abs_log']:.4f}")
    print()
    print("local log-log slope below floor / in window / at the top, per layer:")
    for li, l in enumerate(layers):
        if l == 0:
            continue
        print(f"  l={l:<3d} r1e-4..1e-3 {np.nanmean(slopes[0:2, li]):5.2f}   "
              f"r0.01..0.3 {np.nanmean(slopes[4:7, li]):5.2f}   "
              f"r0.3..1 {slopes[7, li]:5.2f}")
    print()
    print("per-t ratio r=0.1 / floor:")
    for k, v in per_t.items():
        print(f"  l={k}: median {v['median']:.1f}  p10 {v['p10']:.1f}  "
              f"min {v['min']:.1f} at t={v['t_at_min']}  "
              f"frac<3 {v['frac_below_3']:.0%}")
    print()
    print("wrote", p1)
    print("wrote", p2)
    print("wrote", jp)


if __name__ == "__main__":
    main()
