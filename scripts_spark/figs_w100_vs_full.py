#!/usr/bin/env python
"""True window-100 versus full attention on the same 14B prompt/directions.

Reads the three local a1 runs and writes five figures plus one JSON:

  results_big/ds14/a1_inter_full/a1.npz    full attention (window inactive)
  results_big/ds14/a1_inter_w100/a1.npz    TRUE sliding window 100 (patched)
  results_big/ds14/a1_inter_w4096/a1.npz   no-clipping reference, 32 dirs
  results_big/ds14/calib/calib.npz         radius-linearity calibration

Definitions (r = run radius 0.1, u = direction, t = generated step, all
indices as stored by the engine):

  m_t                         top1-top2 logit gap of the run's own baseline
  dm_t(u) = m_t(u)-m_t        change of that gap under direction u
  r_c^-(t,u) = m_t r/max(-dm_t(u),0)   signed radius that would flip step t
  r_*(u) = min_t r_c^-(t,u)   direction threshold radius; flips at r iff <= r
  t_*(u) = argmin_t r_c^-(t,u)        controlling step
  resp_t^l(u) = ||dh_t^l(u)||/r       stored response norm per unit radius
  G(u) = median_t resp_t^47(u)        typical last-state gain

Indexing note: the engine stores the response at absolute position P+t
(``dh_norm[...][P:P+T]``) and the margin of the token predicted from position
P+t-1. For P=65, W=100 the first clipped key (prompt position 0) leaves the
window at response index t=35 and at margin index t=36; that one-step offset
is visible in the data (states depart at 35, margins at 36).

Run from the repository root:  python scripts_spark/figs_w100_vs_full.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# --------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

C_FULL = "#1f4e79"     # full attention
C_WIN = "#c0392b"      # true window 100
C_BAND_F = "#7fb3d5"
C_BAND_W = "#f0a396"
C_GREY = "#8a8a8a"
C_GREEN = "#2e8b57"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_big", "ds14")
FIG = os.path.join(ROOT, "notebook_big", "figures")
R = 0.1            # measurement radius (all a1 runs)
P_LEN = 65         # prompt length
WIN = 100          # true window width
TSTAR = WIN - P_LEN   # 35: first response index with a clipped prompt key


# ------------------------------------------------------------------ helpers
def load(name):
    run = os.path.join(RES, name)
    with open(os.path.join(run, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    d = np.load(os.path.join(run, "a1.npz"))
    return d, meta


def derive(d, r=R):
    """Signed thresholds and response arrays of one run."""
    m = d["margins"].astype(np.float64)
    dm = d["dm"][0].astype(np.float64)                 # [K,T]
    dh = d["dh_norm"][0].astype(np.float64) / r        # [K,nl,T]
    K, nl, T = dh.shape
    rc = np.where(dm < 0.0,
                  m[None, :] * r / np.maximum(-dm, 1e-300),
                  np.inf)
    tstar = np.argmin(rc, axis=1)
    rstar = rc[np.arange(K), tstar]
    flip = rstar <= r
    return dict(m=m, dm=dm, dh=dh, K=K, nl=nl, T=T, rc=rc, tstar=tstar,
                rstar=rstar, flip=flip, layers=d["layers"].astype(int),
                gen_ids=d["gen_ids"].astype(int),
                norm_h0=d["norm_h0"].astype(float))


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    return xs, np.arange(1, xs.size + 1) / xs.size


def q(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"n": int(x.size), "mean": float(x.mean()),
           "min": float(x.min()), "max": float(x.max())}
    for p in ps:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def jf(o):
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def small_steps(m, n=8):
    order = np.argsort(m)[:n]
    return [(int(t), float(m[t])) for t in order]


# ------------------------------------------------------------------ figure 1
def fig_margins(F, W, path):
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.4), sharex=True)
    info = {}
    for ax, dv, col, lab, key in [
            (axes[0], F, C_FULL, "full attention", "full"),
            (axes[1], W, C_WIN, "true sliding window 100", "w100")]:
        m = dv["m"]
        t = np.arange(m.size)
        ax.plot(t, m, color=col, lw=0.9)
        ax.set_yscale("log")
        ax.set_ylim(6e-5, 40)
        ax.axvspan(0, TSTAR, color=C_GREY, alpha=0.10, lw=0)
        ax.axvline(TSTAR, color="k", ls="--", lw=1.0)
        for thr in (0.05, 0.1, 0.5):
            ax.axhline(thr, color=C_GREY, ls=":", lw=0.8)
        # annotate three smallest ties
        offsets = {21: (2, 3.0e-4), 19: (23, 2.2e-3), 189: (176, 1.7e-2),
                   297: (252, 2.2e-3), 273: (258, 3.2e-2)}
        for t0, v0 in small_steps(m, 3):
            dx, dy = offsets.get(t0, (t0 + 8, v0 * 4.5))
            ax.annotate(f"t={t0}\n{v0:.2g}", xy=(t0, v0),
                        xytext=(dx, dy), fontsize=7.6, color=col,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
        n05 = int((m < 0.05).sum())
        n10 = int((m < 0.1).sum())
        n50 = int((m < 0.5).sum())
        txt = "\n".join(f"{tt}: {vv:.4g}" for tt, vv in small_steps(m, 5))
        ax.text(0.985, 0.965,
                f"5 smallest gaps (step: nats)\n{txt}\n"
                f"gaps < 0.05 / 0.1 / 0.5: {n05} / {n10} / {n50} of 300",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.86, edgecolor="none",
                          pad=2.0))
        ax.set_ylabel("top-1 minus top-2 logit gap  (nats, log)")
        ax.set_title(f"({'a' if key == 'full' else 'b'}) {lab}"
                     + (" — identical to the window run for t <= 35"
                        if key == "full" else ""))
        info[key] = dict(smallest_8=small_steps(m, 8),
                         n_below_0p05=n05, n_below_0p1=n10,
                         n_below_0p5=n50,
                         min=float(m.min()), argmin=int(m.argmin()))
    axes[1].set_xlabel("generated step t  (0 = first generated token)")
    axes[1].set_xlim(0, 300)
    axes[0].text(TSTAR + 1, 33, "t=35 = W-P", fontsize=7.8, va="top")
    diff = np.abs(F["m"] - W["m"])
    fig.suptitle("The same two razor ties (t=21, t=19) survive the window; "
                 "every later near-tie is new and the windowed text itself "
                 "changes at t=45", fontsize=11.6, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info, diff


# ------------------------------------------------------------------ figure 2
def fig_response(F, W, path):
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    li = F["nl"] - 1                       # stored state 8 = layer 47
    series = {}
    for dv, col, band, lab in [(F, C_FULL, C_BAND_F, "full attention"),
                               (W, C_WIN, C_BAND_W, "true window 100")]:
        x = dv["dh"][:, li, :]             # [K,T], already /r
        med = np.median(x, axis=0)
        lo = np.percentile(x, 25, axis=0)
        hi = np.percentile(x, 75, axis=0)
        t = np.arange(med.size)
        ax.fill_between(t, lo, hi, color=band, alpha=0.30, lw=0)
        ax.plot(t, med, color=col, lw=1.4, label=lab)
        series[lab] = dict(med=med, lo=lo, hi=hi)
    ax.axvline(TSTAR, color="k", ls="--", lw=1.0)
    ax.axvspan(0, TSTAR, color=C_GREY, alpha=0.08, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(0.15, 25)
    ax.set_xlim(0, 300)
    ax.set_xlabel("generated step t  (0 = first generated token)")
    ax.set_ylabel(r"$\|\delta h_t^{47}\|/r$   (median, band = 25–75%, log)")
    ax.legend(loc="lower left")
    # numbers in the corner
    mf = series["full attention"]["med"]
    mw = series["true window 100"]["med"]
    rf = np.median(mf[36:46]) / np.median(mf[25:35])
    rw = np.median(mw[36:46]) / np.median(mw[25:35])
    ax.text(0.015, 0.965,
            "median ratio, t in [36,45] vs [25,34]:\n"
            f"  full   {rf:.2f}   window {rw:.2f}\n"
            f"t=35 is the first clipped position (W-P)",
            transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2.5))
    ax.annotate("x2.2 flush one step after\nthe first clipped token",
                xy=(37.0, 2.0), xytext=(50, 5.0),
                fontsize=8.0, color=C_WIN,
                arrowprops=dict(arrowstyle="->", color=C_WIN, lw=0.9))
    ax.annotate("bursts in the windowed run\n(repetitive stretch of its text)",
                xy=(120, 7.88), xytext=(158, 12.0), fontsize=8.0,
                color=C_WIN,
                arrowprops=dict(arrowstyle="->", color=C_WIN, lw=0.9))
    ax.set_title("The true window makes the last-layer response jump x2.2 "
                 "across t=35 (the 1.5B showed x1.73)", fontsize=11.6)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dict(rf=float(rf), rw=float(rw),
                med_full=mf, med_win=mw, lo_full=series["full attention"]["lo"],
                hi_full=series["full attention"]["hi"],
                lo_win=series["true window 100"]["lo"],
                hi_win=series["true window 100"]["hi"])


# ------------------------------------------------------------------ figure 3
def fig_rstar(F, W, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for dv, col, lab in [(F, C_FULL, "full attention"),
                         (W, C_WIN, "true window 100")]:
        rs = dv["rstar"]
        xs, ys = ecdf(rs[np.isfinite(rs) & (rs < 60)])
        ax.step(xs, ys, where="post", color=col, lw=1.5, label=lab)
        frac = float(dv["flip"].mean())
        ax.axhline(frac, color=col, ls=":", lw=0.8)
        ax.text(1.4e-3, frac + 0.012, f"{100 * frac:.1f}%", color=col,
                fontsize=8.2)
    ax.axvline(R, color="k", ls="--", lw=1.0)
    ax.text(R * 1.08, 0.03, "r = 0.1\nmeasurement radius", fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlim(2e-3, 60)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$r_*(u)=\min_t r_c^-(t,u)$   (threshold radius, log)")
    ax.set_ylabel("fraction of the 1000 directions")
    ax.legend(loc="upper left")
    ax.set_title("(a) predicted flips at r=0.1: 46.4% -> 49.3%")

    ax = axes[1]
    steps = [21, 19, 189, 273, 297, 276, 247, 255]
    cf, cw = [], []
    for s in steps:
        cf.append(int((F["tstar"] == s).sum()))
        cw.append(int((W["tstar"] == s).sum()))
    y = np.arange(len(steps))
    h = 0.38
    ax.barh(y + h / 2, cf, height=h, color=C_FULL, alpha=0.85,
            label="full attention")
    ax.barh(y - h / 2, cw, height=h, color=C_WIN, alpha=0.85,
            label="true window 100")
    for yy, vf, vw in zip(y, cf, cw):
        if vf:
            ax.text(vf + 6, yy + h / 2, str(vf), va="center", fontsize=7.4,
                    color=C_FULL)
        if vw:
            ax.text(vw + 6, yy - h / 2, str(vw), va="center", fontsize=7.4,
                    color=C_WIN)
    ax.set_yticks(y)
    ax.set_yticklabels([f"t={s}" for s in steps])
    ax.invert_yaxis()
    ax.set_xlim(0, 590)
    ax.set_xlabel("directions whose controlling tie is this step")
    ax.legend(loc="lower right")
    ax.set_title("(b) t=21 and t=19 still control almost every direction")
    fig.suptitle("The window leaves t=21 and t=19 in charge of 93% of the "
                 "flips and adds 29 late flips at t=297",
                 fontsize=11.6, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dict(steps=steps, counts_full=cf, counts_win=cw)


# ------------------------------------------------------------------ figure 4
def fig_gain(F, W, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    G = {}
    for dv, col, lab in [(F, C_FULL, "full attention"),
                         (W, C_WIN, "true window 100")]:
        g = np.median(dv["dh"][:, dv["nl"] - 1, :], axis=1)
        G[lab] = g
        xs, ys = ecdf(g)
        ax.step(xs, ys, where="post", color=col, lw=1.5, label=lab)
        ax.axvline(np.median(g), color=col, ls=":", lw=0.9)
        ax.text(np.median(g) * 1.04, 0.05, f"{np.median(g):.2f}", color=col,
                fontsize=8.2, rotation=90)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 5.0)
    ax.set_xticks([0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "3", "5"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$G(u)=\mathrm{median}_t\ \|\delta h_t^{47}\|/r$  (log)")
    ax.set_ylabel("fraction of directions")
    ax.legend(loc="upper left")
    ax.set_title("(a) median gain 0.58 -> 1.49 (x2.6)")

    ax = axes[1]
    for dv, col, lab in [(F, C_FULL, "full attention"),
                         (W, C_WIN, "true window 100")]:
        x = dv["dh"][:, dv["nl"] - 1, :]
        ratio = np.median(x[:, 150:300], axis=1) / \
            np.maximum(np.median(x[:, 0:50], axis=1), 1e-30)
        xs, ys = ecdf(ratio)
        ax.step(xs, ys, where="post", color=col, lw=1.5, label=lab)
        ax.axvline(np.median(ratio), color=col, ls=":", lw=0.9)
    ax.axvline(1.0, color="k", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 4.0)
    ax.set_xticks([0.2, 0.3, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("late / early response per direction:  "
                  r"median$_{t\in[150,300)}$ / median$_{t\in[0,50)}$  (log)")
    ax.set_ylabel("fraction of directions")
    ax.legend(loc="upper left")
    ax.set_title("(b) median late/early 0.57 -> 0.92")
    fig.suptitle("The window raises the typical response gain 2.6x and nearly "
                 "removes the late-time decay", fontsize=11.6, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dict(G_full=G["full attention"], G_win=G["true window 100"])


# ------------------------------------------------------------------ figure 5
def fig_depth(F, W, path):
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    prof = {}
    pre = None
    for dv, col, lab in [(F, C_FULL, "full attention"),
                         (W, C_WIN, "true window 100")]:
        p = np.median(dv["dh"], axis=(0, 2))       # median over u and t
        prof[lab] = p
        if pre is None:
            pre = np.median(dv["dh"][:, :, :TSTAR], axis=(0, 2))
            ax.plot(np.arange(1, pre.size), pre[1:], "s--", color=C_GREY,
                    lw=1.1, ms=4.0,
                    label="both runs, t < 35 (identical)")
        ax.plot(np.arange(1, p.size), p[1:], "o-", color=col, lw=1.6, ms=5.0,
                label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0.028, 2.6)
    ax.set_xlim(0.6, 8.4)
    ax.set_xlabel("saved state index  (1 = block 0 output, 8 = layer 47)")
    ax.set_ylabel(r"$\|\delta h^l\|/r$   (median over t and directions, log)")
    ax.set_xticks(np.arange(1, 9))
    ax.set_xticklabels([f"{l}\n(L{ll})" for l, ll in
                        zip(range(1, 9), F["layers"][1:])], fontsize=8.0)
    ax.legend(loc="upper left")
    pf, pw = prof["full attention"], prof["true window 100"]
    ax.annotate(f"x{pw[-1] / pw[1]:.1f} growth\nfrom state 1 to 8",
                xy=(7.9, pw[-1]), xytext=(5.7, 1.55), fontsize=8.4,
                color=C_WIN,
                arrowprops=dict(arrowstyle="->", color=C_WIN, lw=0.9))
    ax.annotate(f"x{pf[-1] / pf[1]:.1f} growth",
                xy=(7.9, pf[-1]), xytext=(5.7, 0.55), fontsize=8.4,
                color=C_FULL,
                arrowprops=dict(arrowstyle="->", color=C_FULL, lw=0.9))
    ax.set_title("The window halves the first-block response but doubles the "
                 "last-layer one: depth growth x8.4 -> x38.6",
                 fontsize=11.4)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dict(prof_full=pf, prof_win=pw)


# ------------------------------------------------------------------ numbers
def build_numbers(F, W, G4096, calib, diffs):
    out = {
        "runs": {
            "full": {"file": "results_big/ds14/a1_inter_full/a1.npz",
                     "attention": "full (window setting inactive)",
                     "P": P_LEN, "T": int(F["T"]), "dirs": int(F["K"])},
            "w100": {"file": "results_big/ds14/a1_inter_w100/a1.npz",
                     "attention": "true sliding window 100 (layer_types patch)",
                     "P": P_LEN, "T": int(W["T"]), "dirs": int(W["K"])},
        },
        "verification": {
            "first_response_difference_t": int(diffs["first_dh"]),
            "first_margin_difference_t": int(diffs["first_m"]),
            "first_dm_difference_t": int(diffs["first_dm"]),
            "first_baseline_token_difference_t": int(diffs["first_gen"]),
            "states_bit_identical_before_t": int(diffs["first_dh"]),
            "predicted_first_clip_t": TSTAR,
            "first_gen_tokens": [int(F["gen_ids"][diffs["first_gen"]]),
                                 int(W["gen_ids"][diffs["first_gen"]])],
        },
        "margins": {},
        "response_shape": {},
        "thresholds": {},
        "gain": {},
        "depth": {},
        "calib_check": {},
        "w4096_check": {},
    }
    for key, dv in [("full", F), ("w100", W)]:
        m = dv["m"]
        out["margins"][key] = {
            "smallest_8": small_steps(m, 8),
            "n_below_0p05": int((m < 0.05).sum()),
            "n_below_0p1": int((m < 0.1).sum()),
            "n_below_0p5": int((m < 0.5).sum()),
            "min": float(m.min()), "argmin": int(m.argmin()),
            "median": float(np.median(m)),
        }
        dh = dv["dh"][:, dv["nl"] - 1, :]
        med = np.median(dh, axis=0)
        hi = np.where(med > 3.0)[0]
        runs = []
        if hi.size:
            start = prev = int(hi[0])
            for t in hi[1:]:
                if int(t) == prev + 1:
                    prev = int(t)
                else:
                    runs.append([start, prev])
                    start = prev = int(t)
            runs.append([start, prev])
        out["response_shape"][key] = {
            "median_profile": med,
            "median_25_34": float(np.median(dh[:, 25:35])),
            "median_36_45": float(np.median(dh[:, 36:46])),
            "ratio_36_45_over_25_34": float(np.median(dh[:, 36:46]) /
                                            np.median(dh[:, 25:35])),
            "per_state_flush_ratio": [
                float(np.median(dv["dh"][:, l, 36:46]) /
                      np.median(dv["dh"][:, l, 25:35]))
                for l in range(1, dv["nl"])],
            "n_steps_median_above_3": int(hi.size),
            "median_above_3_ranges": runs,
            "max_step_median": float(med.max()),
            "max_step_median_t": int(med.argmax()),
            "baseline_norm_state8_late": float(np.median(
                dv["norm_h0"][dv["nl"] - 1, 150:300])),
            "median_0_50": float(np.median(dh[:, 0:50])),
            "median_150_300": float(np.median(dh[:, 150:300])),
            "late_over_early": float(np.median(dh[:, 150:300]) /
                                     np.median(dh[:, 0:50])),
        }
        rs = dv["rstar"]
        out["thresholds"][key] = {
            "flip_fraction": float(dv["flip"].mean()),
            "flip_count": int(dv["flip"].sum()),
            "rstar": q(rs),
            "rstar_sorted": rs,
            "controlling_step_counts": {
                int(s): int((dv["tstar"] == s).sum())
                for s in np.unique(dv["tstar"])},
            "flips_by_controlling_step": {
                int(s): int(((dv["tstar"] == s) & dv["flip"]).sum())
                for s in np.unique(dv["tstar"])},
        }
        g = np.median(dh, axis=1)
        ratio = np.median(dh[:, 150:300], axis=1) / \
            np.maximum(np.median(dh[:, 0:50], axis=1), 1e-30)
        ratio25 = np.median(dh[:, 150:300], axis=1) / \
            np.maximum(np.median(dh[:, 25:35], axis=1), 1e-30)
        out["gain"][key] = {
            "G": q(g), "G_sorted": g,
            "per_direction_late_over_early": q(ratio),
            "per_direction_late_over_preeearly": q(ratio25),
            "median_late_over_early": float(np.median(ratio)),
            "median_late_over_preeearly": float(np.median(ratio25)),
            "frac_late_over_early_gt1": float(np.mean(ratio > 1.0)),
        }
        p = np.median(dv["dh"], axis=(0, 2))
        out["depth"][key] = {
            "profile": p,
            "growth_state1_to_8": float(p[-1] / p[1]),
            "early_profile_t_lt_35": np.median(dv["dh"][:, :, :TSTAR],
                                               axis=(0, 2)),
            "late_profile_t_ge_36": np.median(dv["dh"][:, :, TSTAR + 1:],
                                              axis=(0, 2)),
        }

    # set-level comparisons
    fF, fW = F["flip"], W["flip"]
    dhf = F["dh"][:, F["nl"] - 1, :]
    dhw = W["dh"][:, W["nl"] - 1, :]
    out["response_shape"]["window_over_full_median"] = {
        str(t): float(np.median(dhw[:, t]) / np.median(dhf[:, t]))
        for t in (34, 35, 36, 37, 38, 45)}
    out["thresholds"]["comparison"] = {
        "both_flip": int((fF & fW).sum()),
        "only_full": int((fF & ~fW).sum()),
        "only_w100": int((fW & ~fF).sum()),
        "extra_flip_controlling_steps": {
            int(s): int(((W["tstar"] == s) & fW & ~fF).sum())
            for s in np.unique(W["tstar"][fW & ~fF])},
        "same_direction_flip_sets_note":
            "all 464 full-attention flips are also flips under w100",
    }
    out["margins"]["comparison"] = {
        "max_abs_difference": float(np.abs(F["m"] - W["m"]).max()),
        "argmax_difference_t": int(np.abs(F["m"] - W["m"]).argmax()),
        "first_difference_t": int(diffs["first_m"]),
        "identical_up_to_t": int(diffs["first_m"]) - 1,
    }

    # calibration: is it really the true window?
    out["calib_check"] = {
        "file": "results_big/ds14/calib/calib.npz",
        "meta_window": 100,
        "baseline_tokens_equal_to_full_run": bool(
            np.array_equal(calib["gen_ids"][:300], F["gen_ids"])),
        "baseline_tokens_equal_to_w100_run": bool(
            np.array_equal(calib["gen_ids"][:300], W["gen_ids"])),
        "max_margin_difference_vs_full": float(
            np.abs(calib["margins"][:300].astype(float) - F["m"]).max()),
        "r_linearity_dh_over_r": {
            f"layer_{int(lay)}": [
                float(np.median(calib["dh_norm"][i, :, li, :]) /
                      calib["radii"][i])
                for i in range(len(calib["radii"]))]
            for li, lay in enumerate(calib["layers"])},
        "radii": calib["radii"],
        "layers": calib["layers"],
    }
    out["w4096_check"] = {
        "file": "results_big/ds14/a1_inter_w4096/a1.npz",
        "baseline_tokens_equal_to_full_run_first_300": bool(
            np.array_equal(G4096["gen_ids"][:300], F["gen_ids"])),
        "max_margin_difference_vs_full_first_300": float(
            np.abs(G4096["margins"][:300].astype(float) - F["m"]).max()),
        "dirs": int(G4096["dh_norm"].shape[1]),
    }
    return out


# --------------------------------------------------------------------- main
def main():
    os.makedirs(FIG, exist_ok=True)
    dF, mF = load("a1_inter_full")
    dW, mW = load("a1_inter_w100")
    dG, _mG = load("a1_inter_w4096")
    calib = np.load(os.path.join(RES, "calib", "calib.npz"))

    F = derive(dF)
    W = derive(dW)

    diffs = {}
    diffs["first_dh"] = int(np.argmax(
        np.abs(dF["dh_norm"][0] - dW["dh_norm"][0]).max(axis=(0, 1)) > 0))
    diffs["first_m"] = int(np.argmax(
        np.abs(dF["margins"].astype(float) -
               dW["margins"].astype(float)) > 0))
    diffs["first_dm"] = int(np.argmax(
        np.abs(dF["dm"][0] - dW["dm"][0]).max(axis=0) > 0))
    diffs["first_gen"] = int(np.argmax(dF["gen_ids"] != dW["gen_ids"]))

    minfo, mdiff = fig_margins(F, W, os.path.join(FIG, "ds14reg_margins.png"))
    rinfo = fig_response(F, W, os.path.join(FIG, "ds14reg_response_shape.png"))
    tinfo = fig_rstar(F, W, os.path.join(FIG, "ds14reg_rstar.png"))
    ginfo = fig_gain(F, W, os.path.join(FIG, "ds14reg_direction_gain.png"))
    dinfo = fig_depth(F, W, os.path.join(FIG, "ds14reg_depth_gain.png"))

    numbers = build_numbers(F, W, dG, calib, diffs)
    numbers["figures"] = {
        "response_shape": {"ratio_full": rinfo["rf"],
                           "ratio_w100": rinfo["rw"]},
        "rstar_controlling_steps": tinfo,
        "depth_profiles": {"full": dinfo["prof_full"],
                           "w100": dinfo["prof_win"]},
        "gain_medians": {"full": float(np.median(ginfo["G_full"])),
                         "w100": float(np.median(ginfo["G_win"]))},
        "margin_summary": minfo,
    }
    with open(os.path.join(FIG, "ds14reg_numbers.json"), "w",
              encoding="utf-8") as f:
        json.dump(jf(numbers), f, indent=1)
    print("wrote 5 figures + ds14reg_numbers.json")
    print("diffs:", diffs)
    for key in ("full", "w100"):
        th = numbers["thresholds"][key]
        rs = numbers["response_shape"][key]
        print(f"{key}: flips {th['flip_count']} ({100 * th['flip_fraction']:.1f}%)  "
              f"flush {rs['ratio_36_45_over_25_34']:.2f}  "
              f"late/early {rs['late_over_early']:.2f}  "
              f"G {numbers['gain'][key]['G']['p50']:.3f}  "
              f"growth {numbers['depth'][key]['growth_state1_to_8']:.1f}")


if __name__ == "__main__":
    main()
