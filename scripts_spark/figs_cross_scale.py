#!/usr/bin/env python
"""Cross-scale comparison: DeepSeek-R1-Distill-Qwen-1.5B vs -14B, same A1 protocol.

Reads the two A1 ensembles (no torch / no GPU needed) and writes five figures
plus ``xscale_numbers.json`` into ``notebook_big/figures``.

Both runs: same prompt (``interstellar_travel``, P=65), attention window 100,
T=300 teacher-forced baseline tokens, FP32, greedy decoding, prompt-only
perturbation of total norm r, r=0.1.

Definitions (r = run radius, u = direction index, t = generated step, m_t in nats):

  m_t                  top1-top2 logit gap of the unperturbed run
  dm_t(u)              change of that gap under the perturbation
  r_c^-(t,u) = m_t r / max(-dm_t(u), 0)   radius at which step t flips
  r_*(u)     = min_t r_c^-(t,u)           flux-tube radius (smallest radius that
                                          flips some token for direction u)
  t_*(u)     = argmin_t r_c^-(t,u)        controlling step
  G(u)       = median_t ||dh_t^last|| / r          absolute gain
  G_rel(u)   = median_t ||dh_t^last|| / ||h_t^last||   relative gain at r (=r*G(u)/s_last)
  s_emb      = median_t ||h_t^(0)||  (layer-0 state norm, = token embedding norm)

Run from the repository root:
    python scripts_spark/figs_cross_scale.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "notebook_big", "figures")
OUTJSON = os.path.join(FIGDIR, "xscale_numbers.json")

D15 = os.path.join(os.path.expanduser("~"), "research", "chaos_llm_2026",
                   "results", "local_data", "w100_layers", "a1.npz")
D14 = os.path.join(HERE, "results_big", "ds14", "a1_inter_w100", "a1.npz")

# matched relative depths (state index / last state index)
LAYERS_PICK = {"1.5B": [3, 14, 27], "14B": [6, 24, 47]}
PANEL_DEPTH = ["early depth (11% / 13%)", "middle depth (52% / 51%)",
               "last saved state (penultimate exit)"]

FP32_FLOOR = {"1.5B": 1.0e-3, "14B": 2.7e-3}     # ||dh|| noise floor, last state
CLEAN_R = {"1.5B": (1e-3, 0.1), "14B": (0.03, 0.3)}

COL = {"1.5B": "#1f4e79", "14B": "#c0392b"}
COL_LIGHT = {"1.5B": "#7fb3d5", "14B": "#e5989b"}

T_FLUSH = 35          # W - P = 100 - 65

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


# ------------------------------------------------------------------ helpers
def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    return xs, np.arange(1, xs.size + 1) / xs.size


def q(x, ps=(10, 50, 90)):
    x = np.asarray(x, float)
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def qfull(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), min=float(x.min()), max=float(x.max()),
                mean=float(x.mean()), **q(x, (1, 5, 10, 25, 50, 75, 90, 95, 99)))


def jf(o):
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    return o


def load_model(label, npz_path, ri=None):
    d = np.load(npz_path)
    radii = d["radii"].astype(float)
    if ri is None:
        ri = int(np.argmin(np.abs(radii - 0.1)))
    r = float(radii[ri])
    assert abs(r - 0.1) < 1e-9, f"{label}: nearest radius is {r}, not 0.1"
    margins = d["margins"].astype(float)
    dm = d["dm"][ri].astype(float)
    dh = d["dh_norm"][ri].astype(float)          # [K, nl, T]
    layers = d["layers"].astype(int)
    K, nl, T = dh.shape

    rc = np.where(dm < 0.0, margins[None, :] * r / np.maximum(-dm, 1e-300), np.inf)
    tstar = rc.argmin(axis=1)
    rstar = rc[np.arange(K), tstar]

    h0 = d["norm_h0"].astype(float)              # [nl, T]
    s_emb = float(np.median(h0[0]))
    s_last = h0[-1]
    G = np.median(dh[:, -1, :], axis=1) / r
    G_rel = np.median(dh[:, -1, :] / np.maximum(s_last[None, :], 1e-30), axis=1)
    # same, but response normalised by the embedding scale (for reference only)
    G_emb = G / s_emb

    return dict(label=label, d=d, r=r, ri=ri, margins=margins, dm=dm, dh=dh,
                layers=layers, K=K, nl=nl, T=T, rc=rc, tstar=tstar, rstar=rstar,
                norm_h0=h0, s_emb=s_emb, s_last=s_last, G=G, G_rel=G_rel,
                G_emb=G_emb, P=int(d["prompt_ids"].shape[0]))


# ---------------------------------------------------------------- figure 1
def fig_margins(models, path):
    fig = plt.figure(figsize=(12.4, 6.6))
    gs = GridSpec(2, 2, height_ratios=[1.55, 1.0], hspace=0.52, wspace=0.24,
                  figure=fig)
    for col, (label, mv) in enumerate(models.items()):
        m = mv["margins"]
        T = m.size
        steps = np.arange(T)
        ax = fig.add_subplot(gs[0, col])
        ax.semilogy(steps, m, lw=0.8, color=COL[label])
        order = np.argsort(m)[:5]
        ax.plot(order, m[order], "o", ms=6.0, mfc="white", mew=1.4,
                color=COL[label], zorder=4)
        offs = [(9, -2), (-17, 10), (9, 12), (-16, -8), (9, -11)]
        for j, t in enumerate(order):
            ax.annotate(str(j + 1), (t, m[t]), xytext=offs[j],
                        textcoords="offset points", fontsize=7.6,
                        fontweight="bold", color=COL[label])
        ax.axvline(T_FLUSH, color="#8a8a8a", ls=":", lw=1.0)
        ax.text(T_FLUSH + 3, 0.035, r"$t^*=W-P=35$", fontsize=7.5,
                color="#8a8a8a", rotation=90, va="bottom",
                transform=ax.get_xaxis_transform())
        ax.set_xlim(-3, T + 1)
        ax.set_ylim(0.6 * m.min(), 8 * m.max())
        ax.set_xlabel(r"generated step $t$  (0 = first generated token)")
        ax.set_ylabel(r"margin $m_t=\ell^{(1)}_t-\ell^{(2)}_t$   (nats)")
        ax.set_title(f"({'ab'[col]}) {label}: 300 baseline margins "
                     f"(median {np.median(m):.2f}, min {m.min():.2e} at t={m.argmin()})")

        ax2 = fig.add_subplot(gs[1, col])
        for other, ov in models.items():
            xs, ys = ecdf(ov["margins"])
            if other == label:
                ax2.step(xs, ys, where="post", color=COL[other], lw=2.0,
                         label=f"{other} (own, K={ov['K']})")
            else:
                ax2.step(xs, ys, where="post", color=COL_LIGHT[other], lw=1.3,
                         ls="--", label=f"{other} (reference, K={ov['K']})")
        ax2.set_xscale("log")
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel(r"margin $m_t$   (nats, log scale)")
        ax2.set_ylabel(r"ECDF  $P(m_t\leq x)$")
        ax2.axvline(0.1, color="#8a8a8a", ls=":", lw=0.9)
        ax2.legend(loc="lower right")
        ax2.set_title(f"({'cd'[col]}) ECDF of the same 300 margins (log $x$)")
        lines = [r"rank   $t$        $m_t$ (nats)"]
        for j, t in enumerate(order):
            lines.append(f"{j + 1}      {int(t):>3d}      {m[t]:.3e}")
        ax2.text(0.03, 0.96, "\n".join(lines), transform=ax2.transAxes,
                 fontsize=6.9, va="top", family="monospace",
                 bbox=dict(fc="white", ec="#cccccc", lw=0.6, alpha=0.9))

    fig.suptitle("Cross-scale margins: the 14B's weakest tie (9.7e-5 nats) is 200x "
                 "thinner than anything the 1.5B meets in 300 steps (2.0e-2)",
                 fontsize=12.2, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_rstar(models, path):
    fig = plt.figure(figsize=(12.2, 7.6))
    gs = GridSpec(2, 2, height_ratios=[1.45, 1.0], hspace=0.42, wspace=0.22,
                  figure=fig)
    specs = [(False, r"$r_*(u)$   (absolute embedding units)",
              "(a) raw threshold radius"),
             (True, r"$r_*(u)\,/\,s_{\rm emb}$   (dimensionless)",
              "(b) threshold radius in units of the embedding scale")]
    for k, (norm, xlab, ttl) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        axs = fig.add_subplot(gs[1, k])
        for label, mv in models.items():
            v = mv["rstar"] / (mv["s_emb"] if norm else 1.0)
            v = v[np.isfinite(v)]
            xs, ys = ecdf(v)
            ax.step(xs, ys, where="post", color=COL[label], lw=2.0,
                    label=f"{label}  (K={mv['K']}, median {np.median(v):.3g})")
            ax.plot(xs, ys, ".", ms=3.0, color=COL[label], alpha=0.45)
            axs.loglog(xs, 1.0 - ys + 1.0 / v.size, color=COL[label], lw=1.6)
            for qq in (10, 50, 90):
                ax.plot(np.percentile(v, qq), qq / 100.0, "|", ms=8, mew=1.4,
                        color=COL[label])
            ax.axvline(0.1 / (mv["s_emb"] if norm else 1.0),
                       color=COL[label], ls="--", lw=0.9, alpha=0.85,
                       label=f"run radius, {label}")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"ECDF  $P(r_*(u)\leq x)$")
        ax.set_title(ttl)
        ax.legend(loc="lower right", fontsize=7.8)
        axs.set_xlabel(xlab)
        axs.set_ylabel(r"tail  $P(r_*(u)>x)$")
        axs.set_title(f"({'cd'[k]}) same data, log-log survival (1-ECDF + 1/K)")
        axs.grid(True, alpha=0.2, which="both")
        axs.text(0.03, 0.93, "ticks on the ECDF above: p10 / p50 / p90",
                 transform=ax.transAxes, fontsize=7.2, color="#555555")
    fig.suptitle("Threshold radii are 13x smaller in the 14B (17x after dividing by "
                 "the embedding scale): 464/1000 directions flip at r=0.1, 0/20",
                 fontsize=11.8, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_gain(models, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    panels = [
        ("G", r"$G(u)=\mathrm{median}_t\,\Vert\delta h_t^{\rm last}\Vert/r$   (state units per unit $r$)",
         "(a) absolute gain", False, "G"),
        ("G_rel", r"$G_{\rm rel}(u)=\mathrm{median}_t\,\Vert\delta h_t^{\rm last}\Vert/\Vert h_t^{\rm last}\Vert$   (dimensionless)",
         "(b) relative gain (response / baseline state norm)", True, "G_{\\rm rel}"),
    ]
    for ax, (key, xlab, ttl, relative, sym) in zip(axes, panels):
        for label, mv in models.items():
            v = mv[key]
            xs, ys = ecdf(v)
            ax.step(xs, ys, where="post", color=COL[label], lw=2.0,
                    label=f"{label}  (K={mv['K']}, median {np.median(v):.3g})")
            ax.plot(xs, ys, ".", ms=3.0, color=COL[label], alpha=0.45)
            floor = FP32_FLOOR[label] / (mv["s_last"].mean() if relative else mv["r"])
            ax.axvline(floor, color=COL[label], ls=":", lw=1.0, alpha=0.9)
            ax.text(floor * 1.08, 0.06, f"{label} fp32 floor", rotation=90,
                    fontsize=7.0, color=COL[label])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(xlab)
        ax.set_ylabel(rf"ECDF  $P({sym}\leq x)$")
        ax.set_title(ttl)
        ax.legend(loc="upper left")
    fig.suptitle("Absolute gain is 2.3x SMALLER in the 14B, relative gain 4.6x "
                 "smaller: wider residual stream, weaker relative displacement",
                 fontsize=11.8, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_shape(models, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True)
    for k, ax in enumerate(axes):
        lo, hi = np.inf, 0.0
        for label, mv in models.items():
            l = LAYERS_PICK[label][k]
            li = list(mv["layers"]).index(l)
            med = np.median(mv["dh"][:, li, :] / mv["r"], axis=0)
            base = np.median(med)
            y = med / base
            ax.semilogy(np.arange(y.size), y, color=COL[label], lw=1.5,
                        label=f"{label} L{l}")
            lo, hi = min(lo, y[y > 0].min()), max(hi, y.max())
            pre = med[30:35].mean()
            post = med[36:41].mean()
            ax.plot([289], [y[-11]], "o", ms=4, color=COL[label])
            ax.annotate(f"{label} L{l}", (289, y[-11]), xytext=(-2, 12),
                        textcoords="offset points", fontsize=8.2,
                        color=COL[label], ha="left", va="bottom")
            ax.text(0.035, 0.975 - 0.075 * (label == "14B"),
                    f"{label}: 36-40 / 30-34 = {post / pre:.2f}",
                    transform=ax.transAxes, fontsize=7.8, color=COL[label],
                    va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        ax.axvline(T_FLUSH, color="#8a8a8a", ls=":", lw=1.1)
        ax.text(T_FLUSH + 4, 0.04, r"flush $t=W-P=35$", fontsize=7.5,
                color="#8a8a8a", rotation=90, va="bottom",
                transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color="#bbbbbb", lw=0.8)
        ax.set_ylim(lo * 0.55, hi * 6.0)
        ax.set_xlim(-3, 302)
        ax.set_xlabel(r"generated step $t$")
        if k == 0:
            ax.set_ylabel(r"$\mathrm{median}_u\Vert\delta h_t^{l}\Vert/r$  "
                          r"(divided by its own median over $t$)")
        ax.set_title(f"({'abc'[k]}) {PANEL_DEPTH[k]}  "
                     f"(1.5B L{LAYERS_PICK['1.5B'][k]}, 14B L{LAYERS_PICK['14B'][k]})")
    fig.suptitle("The 1.5B response collapses after t=150 (late/early 0.19) and "
                 "surges 1.70x at t=35; the 14B persists (0.68) and surges 1.11x",
                 fontsize=11.6, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_dm(models, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    for ax, (label, mv) in zip(axes, models.items()):
        t = int(mv["margins"].argmin())          # own smallest-margin step
        v = mv["dm"][:, t] / mv["r"]
        m = mv["margins"][t]
        bins = np.linspace(v.min(), v.max(), 41)
        ax.hist(v, bins=bins, color=COL_LIGHT[label], edgecolor=COL[label],
                lw=0.6, label=f"all {mv['K']} directions")
        ax.axvline(-m / mv["r"], color="#c0392b", ls="--", lw=1.6,
                   label=r"$-m_t/r$: flip threshold at $r=0.1$")
        ax.axvline(-m, color="#555555", ls="-.", lw=1.2,
                   label=r"$-m_t$: margin in nats (slope only if $r=1$)")
        ax.axvline(0, color="#333333", lw=0.8)
        passed = int((v <= -m / mv["r"]).sum())
        ax.set_xlabel(r"$\delta m_t(u)/r$   (nats per unit radius)")
        ax.set_ylabel("number of directions")
        ax.set_title(f"{label}: t={t}, $m_t$={m:.2e} nats, {passed}/{mv['K']} "
                     f"directions reach $-m_t/r$")
        ax.legend(loc="upper left", fontsize=7.6)
    # concrete gap annotations
    t15 = int(models["1.5B"]["margins"].argmin())
    v15 = models["1.5B"]["dm"][:, t15] / models["1.5B"]["r"]
    axes[0].annotate("worst direction: %.3f\nneeded: %.3f\n= %.0f%% of threshold"
                     % (v15.min(),
                        -models['1.5B']['margins'][t15] / 0.1,
                        100 * abs(v15.min()) / (models['1.5B']['margins'][t15] / 0.1)),
                     xy=(v15.min(), 1.0), xytext=(0.44, 0.80),
                     textcoords="axes fraction", fontsize=8.2, va="top",
                     color="#1f4e79",
                     arrowprops=dict(arrowstyle="->", lw=0.9, color="#1f4e79"))
    t14 = int(models["14B"]["margins"].argmin())
    v14 = models["14B"]["dm"][:, t14] / models["14B"]["r"]
    axes[1].annotate(f"{(v14 <= -models['14B']['margins'][t14] / 0.1).mean() * 100:.1f}% "
                     "of directions\nlie left of the red dashed line\n"
                     "(which sits at -9.7e-4, inside the peak)",
                     xy=(-models["14B"]["margins"][t14] / 0.1, 150),
                     xytext=(0.47, 0.78), textcoords="axes fraction",
                     fontsize=8.2, va="top", color="#c0392b",
                     arrowprops=dict(arrowstyle="->", lw=0.9, color="#c0392b"))
    fig.suptitle("At its own weakest step the 14B margin sits inside the response "
                 "distribution (427/1000 cross); the 1.5B's does not (0/20)",
                 fontsize=11.6, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    M15 = load_model("1.5B", D15)
    M14 = load_model("14B", D14)
    models = {"1.5B": M15, "14B": M14}

    fig_margins(models, os.path.join(FIGDIR, "xscale_margins.png"))
    fig_rstar(models, os.path.join(FIGDIR, "xscale_rstar.png"))
    fig_gain(models, os.path.join(FIGDIR, "xscale_gain.png"))
    fig_shape(models, os.path.join(FIGDIR, "xscale_shape.png"))
    fig_dm(models, os.path.join(FIGDIR, "xscale_dm.png"))

    out = dict(
        protocol=dict(prompt="interstellar_travel (65 tokens)", window=100,
                      P=65, T=300, dtype="float32", decode="greedy",
                      r=0.1, force="teacher-forced on baseline tokens",
                      t_flush=int(T_FLUSH)),
        models={}, flush={}, cross={},
    )
    for label, mv in models.items():
        m = mv["margins"]
        fin = np.isfinite(mv["rstar"])
        rs = mv["rstar"][fin]
        rsn = rs / mv["s_emb"]
        cnt = np.bincount(mv["tstar"], minlength=m.size)
        top3 = np.argsort(cnt)[::-1][:3]
        out["models"][label] = dict(
            npz=("chaos_llm_2026/results/local_data/w100_layers/a1.npz" if label == "1.5B"
                 else "results_big/ds14/a1_inter_w100/a1.npz"),
            d_model=(1536 if label == "1.5B" else 5120),
            n_layers=(28 if label == "1.5B" else 48),
            K=mv["K"], P=mv["P"], T=mv["T"], r=mv["r"], layers=mv["layers"].tolist(),
            last_state=int(mv["layers"][-1]), n_simplex_blocks=int(mv["K"] // 20),
            margins=qfull(m), margin_min_step=int(m.argmin()),
            margin_min=float(m.min()),
            margin_steps_below_0p1=int((m < 0.1).sum()),
            smallest5=[dict(t=int(t), m=float(m[t])) for t in np.argsort(m)[:5]],
            rstar_raw=dict(n=int(rs.size), finite=int(fin.sum()),
                           **{k: v for k, v in qfull(rs).items() if k != "n"}),
            rstar_norm=dict(**qfull(rsn)),
            flip_count=int((mv["rstar"] <= mv["r"]).sum()),
            s_emb=mv["s_emb"], s_last_median=float(np.median(mv["s_last"])),
            gain_raw=qfull(mv["G"]), gain_rel=qfull(mv["G_rel"]),
            gain_emb_norm=qfull(mv["G_emb"]),
            gain_floor_raw=FP32_FLOOR[label] / mv["r"],
            gain_floor_rel=FP32_FLOOR[label] / float(np.median(mv["s_last"])),
            gain_over_floor=float(np.median(mv["G"]) / (FP32_FLOOR[label] / mv["r"])),
            fp32_floor_last=FP32_FLOOR[label],
            clean_r_window=CLEAN_R[label],
            tstar_top3=[dict(t=int(t), n=int(cnt[t])) for t in top3],
            tstar_n_distinct=int((cnt > 0).sum()),
            smallest_margin_step_quantities=dict(
                t=int(m.argmin()), m=float(m.min()),
                dm_over_r=qfull(mv["dm"][:, int(m.argmin())] / mv["r"]),
                needed_slope=float(m.min() / mv["r"]),
                n_pass=int((mv["dm"][:, int(m.argmin())] / mv["r"]
                            <= -m.min() / mv["r"]).sum()),
                n_dm_negative=int((mv["dm"][:, int(m.argmin())] < 0).sum()),
            ),
        )

        # flush jump, per saved layer and at the last state
        layers = mv["layers"]
        jump = {}
        for li, l in enumerate(layers):
            med = np.median(mv["dh"][:, li, :] / mv["r"], axis=0)
            pre = float(med[30:35].mean())
            post = float(med[36:41].mean())
            jump[f"L{int(l)}"] = dict(
                pre=pre, post=post,
                ratio=(post / pre if pre > 0 else float("nan")),
                local_step35_over34=(float(med[35] / med[34]) if med[34] > 0
                                     else float("nan")))
        out["flush"][label] = dict(per_layer=jump,
                                   last=float(mv["layers"][-1]),
                                   ratio_last=jump[f"L{int(mv['layers'][-1])}"]["ratio"])

        # rise/decay summary at the three matched depths (median of the
        # direction-median response, per time window, in units of its own median)
        decay = {}
        for k, l in enumerate(LAYERS_PICK[label]):
            li = list(mv["layers"]).index(l)
            med = np.median(mv["dh"][:, li, :] / mv["r"], axis=0)
            early = float(med[0:50].mean())
            peak = float(med[50:150].max())
            late = float(med[200:300].mean())
            decay[f"L{l}"] = dict(early=early, peak_50_150=peak, late=late,
                                  peak_t=int(50 + med[50:150].argmax()),
                                  late_over_early=(late / early if early > 0 else float("nan")),
                                  peak_over_late=(peak / late if late > 0 else float("nan")))
        out["models"][label]["shape_at_matched_depth"] = decay

        # is the continuation periodic? (the 1.5B degenerates into a 4-token loop)
        g = mv["d"]["gen_ids"].astype(np.int64)
        n = g.size
        per = {}
        for p in range(2, 17):
            per[p] = float((g[:n - p] == g[p:]).mean())
        best_p = max(per, key=per.get)
        start = None
        if per[best_p] > 0.5:
            eq = (g[:n - best_p] == g[best_p:])
            bad = np.where(~eq)[0]
            start = int(bad[-1] + 1) if bad.size else 0
        out["models"][label]["continuation"] = dict(
            match_rate_by_period={str(k): v for k, v in per.items()},
            best_period=int(best_p), best_match=per[best_p],
            exact_cycle_start_t=start,
            cycle_tokens=(g[start:start + best_p].tolist() if start is not None and best_p <= 8 else None),
        )



    out["cross"] = dict(
        rstar_median_ratio_raw=float(np.median(M15["rstar"]) / np.median(M14["rstar"])),
        rstar_median_ratio_norm=float((np.median(M15["rstar"]) / M15["s_emb"]) /
                                      (np.median(M14["rstar"]) / M14["s_emb"])),
        gain_median_ratio=float(np.median(M15["G"]) / np.median(M14["G"])),
        gain_rel_median_ratio=float(np.median(M15["G_rel"]) / np.median(M14["G_rel"])),
        margin_median_ratio=float(np.median(M15["margins"]) / np.median(M14["margins"])),
        margin_min_ratio=float(M15["margins"].min() / M14["margins"].min()),
        s_emb_ratio=float(M15["s_emb"] / M14["s_emb"]),
        s_last_ratio=float(np.median(M15["s_last"]) / np.median(M14["s_last"])),
        flush_ratio_15=out["flush"]["1.5B"]["ratio_last"],
        flush_ratio_14=out["flush"]["14B"]["ratio_last"],
        d_model=(1536, 5120), n_layers=(28, 48),
    )
    with open(OUTJSON, "w", encoding="utf-8") as f:
        json.dump(jf(out), f, indent=2)
    print(json.dumps(jf(out["cross"]), indent=2))
    for label in models:
        print(f"\n[{label}]")
        print(json.dumps(jf({k: v for k, v in out["models"][label].items()
                             if k not in ("npz", "layers", "smallest5",
                                          "dm_over_r")}), indent=2))


if __name__ == "__main__":
    main()
