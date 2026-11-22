#!/usr/bin/env python
"""Window-flush program on DS-R1-Distill-Qwen-14B: does the response of a
perturbed prompt position die at t* = p_last + (W-P), and does the cascade
follow t*_l = t*_1 + (l-1)(W-1)?

Reads (all optional except the six fixed-token sweep runs):

  results_big/ds14/a1_inter_w4096/a1.npz               32 dirs, no clipping
  results_big/ds14/a1_inter_w{50..500}_fixed/a1.npz    240 dirs, 500 tokens
  results_big/ds14/a1_inter_w100_posearly_{0-22,22-44,44-65}/a1.npz
  results_big/ds14/a1_{short,long}_prompt_w{50,100,200,300,500}_fixed/a1.npz
  results_big/ds14/a1_{short,long}_prompt_w4096/a1.npz   (fixed-token source)
  results_big/ds14/a1_inter_w100_cascade/a1.npz         1500 tokens, 8 dirs, states 0-14
  1.5B reference: chaos_llm_2026/results/local_data/flush_sweep/w*/a1.npz

Writes notebook_big/figures/ds14win_*.png and ds14win_numbers.json.
Run from the repository root:  python scripts_spark/figs_window.py
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
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

C = {50: "#8c510a", 100: "#c0392b", 150: "#d8a13a", 200: "#2e8b57",
     300: "#1f6f9e", 500: "#6a3d9a", 4096: "#555555"}
C_SUP = {"0-22": "#c0392b", "22-44": "#2e8b57", "44-65": "#1f6f9e"}
GREEN = "#2e8b57"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_big", "ds14")
FIG = os.path.join(ROOT, "notebook_big", "figures")
REF15 = os.path.join(os.path.dirname(ROOT), "chaos_llm_2026", "results",
                     "local_data", "flush_sweep")

P_DEFAULT = 65
D_MODEL = 5120
R = 0.1
WS = [50, 100, 150, 200, 300, 500]
MIN_TOKENS_RESOLVED = 11      # need 11 further steps for a death to count


# ------------------------------------------------------------------ helpers
def jf(o):
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def load_run(name, res=RES):
    path = os.path.join(res, name, "a1.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    meta = {}
    mp = os.path.join(res, name, "meta.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return dict(d=d, meta=meta, name=name, path=path)


def plast(d, D=D_MODEL):
    """Largest 0-indexed prompt position carrying support of each direction."""
    simp = d["simplex"]
    sub = d["subspace"]
    out = np.empty(simp.shape[0], dtype=int)
    for u in range(simp.shape[0]):
        out[u] = int((sub[simp[u] != 0] // D).max())
    return out


def state_floors(dh, norm_h0, rel_floor=1.0e-6):
    """Round-off floor per saved state, in absolute response units.

    fp32 batched SDPA leaves a residue even where the response is
    mathematically zero.  Its size scales with the residual-stream norm, so
    the floor is measured as the 5th percentile of the last 60 steps (where
    the response is at the floor for every resolved state) and floored at
    rel_floor * median state norm.
    """
    fl = np.zeros(dh.shape[1])
    for st in range(dh.shape[1]):
        tail = float(np.percentile(dh[:, st, -60:], 5))
        fl[st] = max(tail, rel_floor * float(np.median(norm_h0[st])))
    return fl


def death_time(x1, fl, streak=30):
    """First t that starts the final excursion below the round-off floor.

    Condition: x[t] <= 2*floor and at least 90% of the next `streak` steps
    stay <= 4*floor.  -1 if the run ends first.
    """
    T = x1.size
    thr = 2.0 * fl
    for i in range(T):
        if x1[i] <= thr and (x1[i:i + streak] <= 2.0 * thr).mean() >= 0.9:
            return i
    return -1


def death_table(run, W=None, P=None):
    """Per-state death times for every direction and the resolved mask."""
    d = run["d"]
    dh = d["dh_norm"][0].astype(np.float64)
    nl, T = dh.shape[1], dh.shape[2]
    P = int(P if P is not None else run["meta"].get("prompt_len", P_DEFAULT))
    W = int(W if W is not None else run["meta"].get("window", 0))
    pl = plast(d)
    t1 = pl + (W - P)
    fl = state_floors(dh, d["norm_h0"].astype(np.float64))
    out = dict(W=W, P=P, T=T, nl=nl, p_last=pl, t1=t1, floors=fl,
               death=np.full((nl, pl.size), -1, dtype=int),
               peak=np.zeros((nl, pl.size)),
               resolved=np.zeros((nl, pl.size), dtype=bool))
    for st in range(nl):
        x = dh[:, st, :]
        for u in range(pl.size):
            out["death"][st, u] = death_time(x[u], fl[st])
        if st == 0:
            continue
        if st == 1:
            pred = t1
            pre = np.array([np.median(x[u, max(0, pred[u] - 10):pred[u] + 1])
                            for u in range(pl.size)])
        else:
            pred = t1 + (st - 1) * (W - 1)
            pre = np.array([np.median(x[u, t1[u]:min(T, t1[u] + 10)])
                            for u in range(pl.size)])
        out["peak"][st] = pre
        out["resolved"][st] = ((out["death"][st] >= 0) &
                               (pre >= 10.0 * fl[st]) &
                               (pred <= T - 1 - MIN_TOKENS_RESOLVED))
    out["pred"] = {1: t1}
    for st in range(2, nl):
        out["pred"][st] = t1 + (st - 1) * (W - 1)
    out["dh"] = dh
    return out


def dev_stats(death, pred, resolved):
    m = resolved & (death >= 0)
    if not m.any():
        return dict(n=0)
    dev = (death - pred)[m]
    return dict(n=int(m.sum()),
                median=float(np.median(dev)),
                p5=float(np.percentile(dev, 5)),
                p95=float(np.percentile(dev, 95)),
                min=int(dev.min()), max=int(dev.max()),
                frac_on=float((dev == 0).mean()),
                frac_within2=float((np.abs(dev) <= 2).mean()),
                frac_at_or_before=float((dev <= 0).mean()))


# -------------------------------------------------------------------- data
def load_all():
    runs = {}
    for W in WS:
        r = load_run(f"a1_inter_w{W}_fixed")
        if r is not None:
            runs[W] = r
    ref = load_run("a1_inter_w4096")
    sup = {}
    for s in ("0-22", "22-44", "44-65"):
        r = load_run(f"a1_inter_w100_posearly_{s}")
        if r is not None:
            sup[s] = r
    plen = {}
    for p in ("short_prompt", "long_prompt"):
        for W in WS:
            r = load_run(f"a1_{p}_w{W}_fixed")
            if r is not None:
                plen[(p, W)] = r
    casc = load_run("a1_inter_w100_cascade")
    return runs, ref, sup, plen, casc


# ---------------------------------------------------------------- figures
def fig_departure(runs, tabs, path, ref15=None):
    states = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.0))
    info = {}
    for ax, st in zip(axes, states):
        lim = 0
        for W, tb in tabs.items():
            if st >= tb["nl"]:
                continue
            pred = tb["pred"][st]
            m = tb["resolved"][st] & (tb["death"][st] >= 0)
            if not m.any():
                continue
            x, y = pred[m], tb["death"][st][m]
            lim = max(lim, int(x.max()), int(y.max()))
            ax.scatter(x, y, s=9, alpha=0.75, color=C[W], lw=0,
                       label=f"W={W} (n={m.sum()})")
        if ref15 is not None and st <= 2:
            xs = ref15[f"st{st-1}"]["pred"]
            ys = ref15[f"st{st-1}"]["death"]
            ax.scatter(xs, ys, s=44, facecolors="none", edgecolors="k",
                       lw=1.0, marker="s",
                       label="1.5B exact zero" if st == 1 else None)
            lim = max(lim, int(np.max(xs)))
        lim = max(lim, 20) + 20
        ax.plot([0, lim], [0, lim], color="k", lw=0.9, ls="--", zorder=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("predicted death step  " +
                      ("$p_{last}+(W-P)$" if st == 1
                       else f"$t^*_1+{st-1}(W-1)$"))
        ax.set_ylabel("measured death step  "
                      "(first step at the round-off floor)")
        ax.set_title(f"({'abc'[states.index(st)]}) saved state {st}"
                     + (" = block-0 output" if st == 1 else ""))
        ax.legend(loc="upper left", markerscale=1.6)
        if st == 1:
            lines = ["W   on-line/n   max|dev|"]
            for W, tb in tabs.items():
                d = dev_stats(tb["death"][st], tb["pred"][st],
                              tb["resolved"][st])
                if not d.get("n"):
                    continue
                n_on = int(round(d["frac_on"] * d["n"]))
                mx = max(abs(d["min"]), abs(d["max"]))
                lines.append(f"{W:<4d}{n_on:>6d}/{d['n']:<5d}{mx:>7d}")
            ax.text(0.985, 0.02, "\n".join(lines), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.4, family="monospace",
                    color=GREEN,
                    bbox=dict(facecolor="white", alpha=0.88, edgecolor="none"))
        else:
            lines = ["W   at/before   median gap"]
            for W, tb in tabs.items():
                d = dev_stats(tb["death"][st], tb["pred"][st],
                              tb["resolved"][st])
                if not d.get("n"):
                    continue
                lines.append(f"{W:<4d}{100*d['frac_at_or_before']:>7.1f}%"
                             f"{d['median']:>11.0f}")
            ax.text(0.985, 0.02, "\n".join(lines), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.4, family="monospace",
                    color=C[100],
                    bbox=dict(facecolor="white", alpha=0.88, edgecolor="none"))
        ax.grid(True, alpha=0.25)
        info[f"state{st}"] = {
            str(W): dev_stats(tb["death"][st], tb["pred"][st],
                              tb["resolved"][st])
            for W, tb in tabs.items() if st < tb["nl"]}
    fig.suptitle("State 1 dies exactly where the window law says; states 2-3 "
                 "decay to the fp32 floor at or before the cascade bound",
                 fontsize=11.8, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


def fig_cascade(tabs, path, casc=None):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    info = {}
    for W, tb in tabs.items():
        lw = 1.9 if W in (100, 200) else 1.1
        alpha = 1.0 if W in (100, 200) else 0.65
        ls = [1, 2, 3, 4, 6, 8]
        ms, bs = [], []
        sts = []
        for st in ls:
            if st >= tb["nl"]:
                continue
            m = tb["resolved"][st] & (tb["death"][st] >= 0)
            if not m.any():
                continue
            sts.append(st)
            ms.append(float(np.median(tb["death"][st][m])))
            bs.append(float(np.median(tb["pred"][st][m])))
        if not sts:
            continue
        ax.plot(sts, bs, ls="--", lw=lw * 0.8, color=C[W], alpha=0.55)
        ax.plot(sts, ms, ls="-", marker="o", ms=4.5, lw=lw, color=C[W],
                alpha=alpha, label=f"W={W} measured")
        if W in (100, 200):
            info[str(W)] = dict(states=sts, measured_median=ms,
                                bound_median=bs,
                                slope_measured=float(np.polyfit(sts, ms, 1)[0]),
                                slope_bound=float(W - 1))
    ax.set_xlabel("saved state index $l$  (1 = block-0 output)")
    ax.set_ylabel("death step $t$  (median over directions)")
    ax.set_title("(a) measured death vs the cascade bound "
                 "$t^*_l=t^*_1+(l-1)(W-1)$")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.text(0.985, 0.03,
            "dashed = upper bound; markers = time to\n"
            "reach the fp32 floor (always earlier)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    ax2 = axes[1]
    if casc is not None:
        tb = death_table(casc)
        x = tb["dh"]
        ms, ms_lo, ms_hi, bs, zs = [], [], [], [], []
        sts = []
        for st in range(1, tb["nl"]):
            m = tb["resolved"][st] & (tb["death"][st] >= 0)
            if not m.any():
                continue
            sts.append(st)
            ms.append(float(np.median(tb["death"][st][m])))
            ms_lo.append(float(np.percentile(tb["death"][st][m], 25)))
            ms_hi.append(float(np.percentile(tb["death"][st][m], 75)))
            bs.append(float(np.median(tb["pred"][st][m])))
            zs.append([int(np.argmax(tr == 0)) if (tr == 0).any() else -1
                       for tr in x[:, st]])
        ax2.plot(sts, bs, ls="--", color=C[100], lw=1.1,
                 label="bound $t^*_1+(l-1)(W-1)$")
        ax2.plot(sts, [np.median(z) for z in zs], ls="none", marker="o",
                 ms=9.0, mfc="none", mec="#333333", mew=0.9,
                 label="first bitwise zero (all 8 dirs, on the bound)")
        ax2.errorbar(sts, ms, yerr=[np.array(ms) - np.array(ms_lo),
                                    np.array(ms_hi) - np.array(ms)], fmt="o",
                     ms=4.5, color=C[100], lw=1.1,
                     label="threshold death (1e-6 rel floor)")
        # median gap of the threshold death, and the measured slope
        dev = [a - b for a, b in zip(ms, bs)]
        slope = float(np.polyfit(sts, ms, 1)[0])
        ax2.text(0.03, 0.97,
                 "threshold death: slope %.0f vs 99 tokens/state\n"
                 "median gap %.0f at l=2 to %.0f at l=14\n"
                 "all %d resolved direction-states at/before bound"
                 % (slope, dev[1], dev[-1], int(sum(
                     (tb["resolved"][st] & (tb["death"][st] >= 0)).sum()
                     for st in sts))),
                 transform=ax2.transAxes, ha="left", va="top", fontsize=7.8,
                 family="monospace",
                 bbox=dict(facecolor="white", alpha=0.86, edgecolor="none"))
        ax2.text(0.02, 0.35,
                 "floor rule as in (a); the tail is bitwise zero,\n"
                 "so the 1e-6 cap sets it: death = 1e-6 crossing",
                 transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.3,
                 bbox=dict(facecolor="white", alpha=0.86, edgecolor="none"))
        ax2.legend(loc="lower right", fontsize=7.6)
        ax2.set_xlabel("saved state index $l$")
        ax2.set_ylabel("death step $t$")
        ax2.set_title("(b) 1500-token cascade run, W=100, $P=65$")
        info["cascade1500"] = dict(
            states=sts, measured=ms, bound=bs,
            iqr_lo=ms_lo, iqr_hi=ms_hi,
            dev_median=dev,
            first_zero=zs,
            n_literal_zero_on_bound=int(sum(
                int(v == b) for z, b in zip(zs, bs) for v in z)),
            n_literal_zero=int(sum(len(z) for z in zs)),
            floor=[float(tb["floors"][st]) for st in sts],
            floor_rel=[float(tb["floors"][st] /
                           np.median(casc["d"]["norm_h0"].astype(float)[st]))
                       for st in sts],
            peak_over_floor=[float(np.median(tb["peak"][st]) /
                                   tb["floors"][st]) for st in sts],
            slope_measured=slope, slope_bound=float(tb["W"] - 1),
            resolved=[int((tb["resolved"][st] &
                           (tb["death"][st] >= 0)).sum()) for st in sts])
    else:
        ax2.text(0.5, 0.55,
                 "1500-token cascade run\n(window 100, 8 dirs, states 0-14)\n"
                 "PENDING — file not yet local",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=9.5, color="#777777",
                 bbox=dict(facecolor="#f4f4f4", edgecolor="#cccccc"))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("(b) deeper cascade: pending")
        info["cascade1500"] = dict(pending=True)
    fig.suptitle("Threshold death runs early and saturates with depth; the "
                 "first bitwise zero lands exactly on the cascade bound",
                 fontsize=11.8, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


def fig_support(sup, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    info = {}
    lim = 0
    for s, run in sup.items():
        tb = death_table(run, W=100)
        m = tb["resolved"][1] & (tb["death"][1] >= 0)
        x, y = tb["pred"][1][m], tb["death"][1][m]
        ax.scatter(x, y, s=14, alpha=0.75, color=C_SUP[s], lw=0,
                   label=f"prompt positions {s} (n={m.sum()}, "
                         f"$p_{{last}}$={tb['p_last'].min()}-{tb['p_last'].max()})")
        info[s] = dict(dev=dev_stats(tb["death"][1], tb["pred"][1],
                                     tb["resolved"][1]),
                       p_last=[int(tb["p_last"].min()), int(tb["p_last"].max())],
                       n=int(m.sum()))
        lim = max(lim, int(np.max(x)))
    lim += 15
    ax.plot([0, lim], [0, lim], "k--", lw=0.9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted death step  $p_{last}+35$  (W=100, P=65)")
    ax.set_ylabel("measured death step (state 1)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.set_title("Moving only the perturbation's support moves the death "
                 "step to $p_{last}+35$", fontsize=11.0)
    ax.text(0.985, 0.02,
            "identical W, P, tokens and directions;\n"
            "only the prompt positions differ",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


def plen_flush(tb):
    """Surge ratio at t = W-P in the last saved state, if the kink is visible.

    Same definition as fig_shape: ratio of the median response over the ten
    steps after the kink to the ten steps before, pooled and per direction.
    """
    k, T = tb["W"] - tb["P"], tb["T"]
    if k - 10 < 0 or k + 10 > T - 1:
        return dict(kink=k, pooled=None, per_direction=None)
    x = tb["dh"][:, -1, :] / R
    pooled = float(np.median(x[:, k + 1:k + 11]) /
                   max(float(np.median(x[:, k - 10:k])), 1e-300))
    per = (np.median(x[:, k + 1:k + 11], axis=1) /
           np.maximum(np.median(x[:, k - 10:k], axis=1), 1e-300))
    return dict(kink=k, pooled=pooled,
                per_direction=float(np.median(per)))


def fig_plen(plen, path):
    """State-1 death step vs p_last+(W-P) for the two prompt lengths.

    One panel per prompt.  Marker colour is the window.  Cases with no
    resolved direction (deaths at or beyond the 300-step horizon) are
    annotated, not plotted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    titles = {
        "short_prompt": "(a) short prompt, $P=8$: $W\\gg P$ in every window",
        "long_prompt": "(b) long prompt, $P=190$: $W<P$ (50, 100), "
                       "$W>P$ (200, 300)",
    }
    notes = {
        "short_prompt": "W=300: $t^*=298$-$299$, 1-2 steps before the end\n"
                        "W=500: $t^*=492$-$499$, beyond the 300-step run",
        "long_prompt": "W=50: $t^*=p_{last}-140$, W=100: $p_{last}-90$\n"
                       "  (positions below $P-W$ are never visible)\n"
                       "W=200: $p_{last}+10$, W=300: $p_{last}+110$\n"
                       "W=500: $p_{last}+310>299$: no position ever leaves",
    }
    info = {}
    for ax, p in zip(axes, ("short_prompt", "long_prompt")):
        ref = load_run(f"a1_{p}_w4096")
        mr = ref["d"]["margins"] if ref is not None else None
        lim = 0
        for W in WS:
            run = plen.get((p, W))
            if run is None:
                continue
            tb = death_table(run, W=W)
            m = tb["resolved"][1] & (tb["death"][1] >= 0)
            d = dev_stats(tb["death"][1], tb["pred"][1], tb["resolved"][1])
            x1 = tb["dh"][:, 1, :]
            e = dict(
                P=tb["P"], W=W, T=tb["T"],
                regime=("P>W" if tb["P"] > W else "P<W"),
                n_dir=int(tb["p_last"].size),
                p_last=[int(tb["p_last"].min()), int(tb["p_last"].max())],
                p_last_values=[int(v) for v in np.unique(tb["p_last"])],
                t1=[int(tb["t1"].min()), int(tb["t1"].max())],
                n_dead=int((tb["death"][1] >= 0).sum()),
                n_no_death=int((tb["death"][1] < 0).sum()),
                n_horizon_ok=int((tb["t1"] <= tb["T"] - 1 -
                                  MIN_TOKENS_RESOLVED).sum()),
                floor_contaminated=bool(tb["floors"][1] > 1e-4),
                n_resolved=int(m.sum()), dev=d,
                floor=float(tb["floors"][1]),
                literal=dict(
                    n_exact_zero=int((x1 <= 0.0).any(1).sum()),
                    n_below_1e6=int((x1 <= 1e-6).any(1).sum()),
                    min_over_t_median=float(np.median(x1.min(1)))),
                flush=plen_flush(tb))
            if mr is not None:
                mm = run["d"]["margins"]
                same = np.array_equal(mm, mr)
                first = int(np.argmax(mm != mr)) if not same else None
                e["margins"] = dict(
                    identical_to_4096=bool(same), first_diff=first,
                    expected_first_diff=int(W - tb["P"] + 1),
                    gen_ids_identical=bool(np.array_equal(
                        run["d"]["gen_ids"], ref["d"]["gen_ids"])),
                    median=float(np.median(mm)), min=float(mm.min()),
                    argmin=int(mm.argmin()),
                    n_below_0p1=int((mm < 0.1).sum()),
                    n_below_0p5=int((mm < 0.5).sum()))
            info[f"{p}_w{W}"] = e
            if m.any():
                xs, ys = tb["pred"][1][m], tb["death"][1][m]
                ax.scatter(xs, ys, s=14, alpha=0.8, lw=0, color=C[W],
                           marker="o" if p == "short_prompt" else "^",
                           label=f"W={W}: {d['n']} of {e['n_dir']} resolved")
                lim = max(lim, int(xs.max()), int(ys.max()))
        lim = max(lim, 40) + 15
        if p == "long_prompt":
            lim = max(lim, 310)
        ax.plot([0, lim], [0, lim], "k--", lw=0.9, zorder=0)
        if 299 < lim:
            ax.axvline(299, color="#999999", ls=":", lw=1.0)
            ax.text(0.985, 0.40, "300-step horizon", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.0, color="#888888")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("predicted death step  $p_{last}+(W-P)$")
        ax.set_ylabel("measured death step (state 1)")
        ax.set_title(titles[p], fontsize=10.4)
        ax.legend(loc="upper left", fontsize=7.8)
        ax.text(0.985, 0.02, notes[p], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.3, family="monospace",
                bbox=dict(facecolor="white", alpha=0.86, edgecolor="none"))
    fig.suptitle("State 1 dies at $p_{last}+(W-P)$ for a 8-token and a "
                 "190-token prompt; the only failures are at the horizon",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


def fig_shape(runs, ref, path):
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    info = {}
    for W, run in sorted(runs.items()):
        dh = run["d"]["dh_norm"][0].astype(np.float64) / R
        x = dh[:, -1, :]
        med = np.median(x, axis=0)
        t = np.arange(med.size)
        if W == 100:
            ax.fill_between(t, np.percentile(x, 25, axis=0),
                            np.percentile(x, 75, axis=0),
                            color=C[W], alpha=0.18, lw=0)
        ax.plot(t, med, color=C[W], lw=1.35, label=f"W={W}")
        k = W - 65
        if k > 3:
            pooled = float(np.median(x[:, k + 1:k + 11]) /
                           np.median(x[:, k - 10:k]))
            per_dir = float(np.median(
                np.median(x[:, k + 1:k + 11], axis=1) /
                np.maximum(np.median(x[:, k - 10:k], axis=1), 1e-300)))
            ax.axvline(k, color=C[W], ls=":", lw=0.9, alpha=0.8)
            ax.annotate(f"x{pooled:.2f}", xy=(k, med[k] * 1.02),
                        xytext=(k + 4, med[k] * (0.62 if W % 200 else 1.6)),
                        fontsize=7.8, color=C[W])
        else:
            pooled = per_dir = None
        info[str(W)] = dict(kink_step=k if k > 0 else None,
                            flush_ratio_pooled=pooled,
                            flush_ratio_per_direction=per_dir,
                            median=med)
    if ref is not None:
        dh = ref["d"]["dh_norm"][0].astype(np.float64) / R
        med = np.median(dh[:, -1, :], axis=0)
        t = np.arange(med.size)
        ax.plot(t, med, color=C[4096], lw=1.1, ls="--",
                label="W=4096 (no clipping)")
        info["4096"] = dict(median=med)
    ax.set_yscale("log")
    ax.set_xlabel("generated step t  (0 = first generated token)")
    ax.set_ylabel("$\\|\\delta h_t^{47}\\|/r$  (median over directions, log)")
    ax.legend(loc="upper right", ncol=2, fontsize=8.0)
    ax.set_title("Every window shows a flush surge one step after "
                 "$t=W-P$; x2.19 at W=100 (core run x2.20), x3.95 at W=300",
                 fontsize=11.2)
    ax.text(0.012, 0.03,
            "vertical dotted lines: $t=W-P$ for each W\n"
            "numbers: median ratio of steps $[t^*+1,t^*+10]$ to $[t^*-10,t^*-1]$",
            transform=ax.transAxes, fontsize=7.8,
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


def fig_margins(runs, ref, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    info = {}
    ws = sorted(runs)
    med = [float(np.median(runs[W]["d"]["margins"])) for W in ws]
    mn = [float(np.min(runs[W]["d"]["margins"])) for W in ws]
    ax.plot(ws, med, "o-", color=C[100], lw=1.4, ms=5, label="median margin")
    ax.plot(ws, mn, "s-", color=C[300], lw=1.4, ms=5, label="minimum margin")
    if ref is not None:
        mref = ref["d"]["margins"]
        ax.axhline(float(np.median(mref)), color=C[4096], ls="--", lw=0.9)
        ax.axhline(float(np.min(mref)), color=C[4096], ls=":", lw=0.9)
        ax.text(500, float(np.median(mref)) * 1.08, "W=4096 median",
                ha="right", fontsize=7.6, color=C[4096])
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("window width W (tokens)")
    ax.set_ylabel("top-1 minus top-2 gap (nats, log)")
    ax.legend(loc="lower right")
    ax.set_title("(a) same tokens and directions; the window moves the "
                 "whole margin spectrum")

    ax = axes[1]
    for W in ws:
        m = runs[W]["d"]["margins"]
        ax.plot(m, color=C[W], lw=0.55, alpha=0.55)
    ax.axvline(35, color="k", ls="--", lw=1.0)
    ax.text(36, 0.02, "t=35=W-P (W=100)", fontsize=7.8)
    ax.set_yscale("log")
    ax.set_xlabel("generated step t")
    ax.set_ylabel("margin (nats, log)")
    ax.set_title("(b) margin traces, one line per W")
    for W in ws:
        m = runs[W]["d"]["margins"]
        info[str(W)] = dict(median=float(np.median(m)), min=float(m.min()),
                            argmin=int(m.argmin()),
                            n_below_0p1=int((m < 0.1).sum()),
                            n_below_0p5=int((m < 0.5).sum()))
    if ref is not None:
        info["4096"] = dict(median=float(np.median(ref["d"]["margins"])),
                            min=float(ref["d"]["margins"].min()),
                            argmin=int(ref["d"]["margins"].argmin()),
                            n_below_0p1=int((ref["d"]["margins"] < 0.1).sum()),
                            n_below_0p5=int((ref["d"]["margins"] < 0.5).sum()))
    fig.suptitle("The window moves the decision landscape: W=50 (prompt does "
                 "not fit) loses the t=21 tie; W>=100 keeps it to the bit",
                 fontsize=11.6, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return info


# --------------------------------------------------------------- 1.5B ref
def load_ref15():
    """Exact-zero cascade points from the 1.5B flush sweep (read-only)."""
    out = {}
    preds = {st: [] for st in range(3)}
    deaths = {st: [] for st in range(3)}
    per_w = {}
    if not os.path.isdir(REF15):
        return None
    for W in (100, 150, 200, 300, 500):
        p = os.path.join(REF15, f"w{W}", "a1.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        P = len(d["prompt_ids"])
        pl = plast(d, D=int(d["dh_last"].shape[-1]) if d["dh_last"].size else 1536)
        dh = d["dh_norm"][0]
        t1 = pl + (W - P)
        per_w[W] = dict(pred=[int(x) for x in t1], P=int(P))
        for st in range(min(3, dh.shape[1])):
            x = dh[:, st, :]
            z = np.where((x <= 0).any(1), (x <= 0).argmax(1), -1)
            pred = t1 if st == 0 else t1 + st * (W - 1)
            per_w[W][f"st{st}"] = dict(pred=[int(v) for v in pred],
                                       death=[int(v) for v in z])
            if (z >= 0).all():
                preds[st].extend([int(v) for v in pred])
                deaths[st].extend([int(v) for v in z])
    if not per_w:
        return None
    for st in range(3):
        out[f"st{st}"] = dict(pred=preds[st], death=deaths[st])
    out["per_run"] = per_w
    return out


# -------------------------------------------------------------------- main
def main():
    os.makedirs(FIG, exist_ok=True)
    runs, ref, sup, plen, casc = load_all()
    tabs = {}
    for W in sorted(runs):
        if W >= 500:
            pass
        tabs[W] = death_table(runs[W], W=W)

    ref15 = load_ref15()

    numbers = dict(setup=dict(model="DeepSeek-R1-Distill-Qwen-14B",
                              dtype="float32", radius=R,
                              prompt="interstellar_travel",
                              P_default=P_DEFAULT, W=sorted(runs),
                              T={str(W): int(tabs[W]["T"]) for W in tabs},
                              directions={str(W): int(tabs[W]["p_last"].size)
                                          for W in tabs},
                              death_rule=dict(
                                  floor="5th percentile of dh_norm over the "
                                        "last 60 steps, per state; at least "
                                        "1e-6 x median state norm",
                                  criterion="first t with dh_norm(t) <= "
                                            "2 x floor and 90% of the next 30 "
                                            "steps <= 4 x floor",
                                  resolved="peak signal >= 10 x floor and at "
                                           "least 11 further steps")))

    # literal zero / 1e-6 test, before any floor calibration
    literal = {}
    for W, tb in tabs.items():
        x = tb["dh"][:, 1, :]
        literal[str(W)] = dict(
            n_dirs=int(x.shape[0]),
            n_exact_zero=int((x <= 0.0).any(1).sum()),
            n_below_1e6=int((x <= 1e-6).any(1).sum()),
            min_over_t_median=float(np.median(x.min(1))),
            min_over_t_min=float(x.min()),
            floor_abs=float(tb["floors"][1]),
            floor_rel=float(tb["floors"][1] /
                            np.median(runs[W]["d"]["norm_h0"].astype(float)[1])))
    numbers["literal_death_test"] = literal

    # departure / cascade statistics
    dep = {}
    for W, tb in tabs.items():
        per = {}
        for st in range(1, tb["nl"]):
            m = tb["resolved"][st] & (tb["death"][st] >= 0)
            per[f"state{st}"] = dict(
                dev=dev_stats(tb["death"][st], tb["pred"][st],
                              tb["resolved"][st]),
                measured_median=(float(np.median(tb["death"][st][m]))
                                 if m.any() else None),
                predicted_median=(float(np.median(tb["pred"][st][m]))
                                  if m.any() else None),
                bound_median=(float(np.median(tb["pred"][st][m]))
                              if m.any() else None))
            # drop entries with no resolved direction to keep the JSON small
            if not m.any():
                per[f"state{st}"] = dict(dev=dict(n=0))
        pl = tb["p_last"]
        dep[str(W)] = dict(P=tb["P"], T=tb["T"], nl=tb["nl"],
                           p_last=dict(min=int(pl.min()), max=int(pl.max()),
                                       median=float(np.median(pl)),
                                       values=[int(v) for v in np.unique(pl)]),
                           states=per)
    numbers["departure"] = dep

    numbers["fig1_departure"] = fig_departure(
        runs, tabs, os.path.join(FIG, "ds14win_departure.png"), ref15)
    numbers["fig2_cascade"] = fig_cascade(
        tabs, os.path.join(FIG, "ds14win_cascade.png"), casc)
    numbers["fig3_support"] = (fig_support(
        sup, os.path.join(FIG, "ds14win_support.png")) if sup else {"pending": True})
    numbers["fig4_plen"] = fig_plen(
        plen, os.path.join(FIG, "ds14win_plen.png"))
    numbers["fig5_response_shape"] = fig_shape(
        runs, ref, os.path.join(FIG, "ds14win_response_shape.png"))
    numbers["fig6_margins"] = fig_margins(
        runs, ref, os.path.join(FIG, "ds14win_margins_by_w.png"))

    # continuation / margins cross-checks vs the no-clipping reference
    checks = {}
    if ref is not None:
        for W, tb in tabs.items():
            m = runs[W]["d"]["margins"]
            mr = ref["d"]["margins"]
            same = np.array_equal(m, mr)
            first = int(np.argmax(m != mr)) if not same else None
            h0 = runs[W]["d"]["h0_last"].astype(np.float64)
            h0r = ref["d"]["h0_last"].astype(np.float64)
            dmax = np.abs(h0 - h0r).max(1)
            firsth = int(np.argmax(dmax > 0))
            x = tb["dh"][:, -1, :] / R
            k = W - P_DEFAULT
            flush = (float(np.median(np.median(x[:, k + 1:k + 11], 1) /
                                     np.maximum(np.median(x[:, k - 10:k], 1),
                                                1e-300))) if k > 3 else None)
            checks[str(W)] = dict(gen_ids_identical=bool(
                np.array_equal(runs[W]["d"]["gen_ids"], ref["d"]["gen_ids"])),
                margins_identical=bool(same),
                margins_first_diff=first,
                h0_last_first_diff=firsth,
                margins_identical_upto=(first - 1 if first else None),
                flush_ratio_last_state=flush)
        checks["4096_ref"] = dict(
            margins_median=float(np.median(ref["d"]["margins"])),
            margins_min=float(ref["d"]["margins"].min()),
            margins_argmin=int(ref["d"]["margins"].argmin()))
    numbers["continuation_checks"] = checks

    # 1.5B reference numbers
    if ref15 is not None:
        numbers["ref_1p5B"] = {
            W: {k: v for k, v in d.items()}
            for W, d in ref15["per_run"].items()}

    numbers["pending"] = dict(
        prompt_length=[f"a1_{p}_w{W}_fixed" for p in
                       ("short_prompt", "long_prompt") for W in WS
                       if (p, W) not in plen],
        cascade=["a1_inter_w100_cascade"] if casc is None else [])

    with open(os.path.join(FIG, "ds14win_numbers.json"), "w",
              encoding="utf-8") as f:
        json.dump(jf(numbers), f, indent=1)
    print("wrote figures + ds14win_numbers.json")


if __name__ == "__main__":
    main()
