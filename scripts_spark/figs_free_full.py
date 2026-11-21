#!/usr/bin/env python
"""Free-run validation of the signed flip rule, DS-R1-Distill-Qwen-14B.

Validates the teacher-forced, full-attention A1 run (1000 directions, r=0.1)
against the free-running greedy continuations at rho = 0.05, 0.1, 0.3.

Reads
    results_big/ds14/a1_inter_full/a1.npz
    results_big/ds14/free_inter_full/free_run.npz
    results_big/ds14/free_inter_full/summary.json
    assets/tok_ds14/tokenizer.json

Writes
    notebook_big/figures/ds14free_{scorecard,survival,latency,resync,flipword}.png
    notebook_big/figures/ds14free_numbers.json

Definitions (u = direction, t = generated step, rho = free-run radius,
r = 0.1 = radius of the teacher-forced A1 run, m_t in nats):

  m_t                 top1-top2 logit gap of the unperturbed A1 run
  dm_t(u)             change of that gap under direction u at radius r
  r_c^-(t,u) = m_t r / max(-dm_t(u), 0)       radius at which t would flip
  r_*(u)     = min_t r_c^-(t,u)               flux-tube radius
  t_*(u)     = argmin_t r_c^-(t,u)            controlling tie
  T_pred(rho,u) = min{t : rho >= r_c^-(t,u)}  first crossing in TIME (signed rule)
  obs[ri,u]   first mismatching generated token in the free run (-1 = never)
  A_after(u)  mean_t>obs [gen_ids[t] == baseline_gen_ids[t]]
  A_late(u)   the same for t > obs + 45 (after the echoed prompt sentence)

The brief asks for the sign-blind comparison as well: the npz's stored ``r_c``
uses |dm| and is therefore symmetric.  It is recomputed here from ``dm`` to
make the sign convention explicit.

Run from the repository root:
    python scripts_spark/figs_free_full.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "notebook_big", "figures")
OUTJSON = os.path.join(FIGDIR, "ds14free_numbers.json")

A1 = os.path.join(HERE, "results_big", "ds14", "a1_inter_full", "a1.npz")
FREE = os.path.join(HERE, "results_big", "ds14", "free_inter_full", "free_run.npz")
SUMMARY = os.path.join(HERE, "results_big", "ds14", "free_inter_full", "summary.json")
TOK = os.path.join(HERE, "assets", "tok_ds14", "tokenizer.json")

# radii, in the order stored in free_run.npz
RHOS = (0.05, 0.1, 0.3)

# the free-run writer overwrites all_gen once per chunk, so gen_ids contains
# only the final chunk.  We verify the mapping below instead of assuming it.
GEN_DIR0 = 989
LATE_OFFSET = 45          # t > obs + 45 is past the echoed task sentence
RESYNC_MIN_RUN = 20       # "re-synchronised" = >= 20 identical consecutive tokens
LIT_1P5B_POSTFLIP = 0.027  # literature value quoted in the brief (1.5B, after flip)

COL = {
    "exact": "#2e7d32",
    "wrong_step": "#e08a1e",
    "false_alarm": "#c0392b",
    "miss": "#7b4fa0",
    "both_none": "#cfcfcf",
}
RCOL = {0.05: "#1f77b4", 0.1: "#d95f02", 0.3: "#6a3d9a"}

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


def runs_of(mask, start):
    """Maximal True runs [(first, last), ...] of a boolean vector."""
    out, s = [], None
    for k, ok in enumerate(mask, start=start):
        if ok and s is None:
            s = k
        elif (not ok) and s is not None:
            out.append((s, k - 1))
            s = None
    if s is not None:
        out.append((s, start + len(mask) - 1))
    return out


def first_mismatch(ref, row):
    n = min(len(ref), len(row))
    mm = np.where(np.asarray(row[:n]) != np.asarray(ref[:n]))[0]
    return int(mm[0]) if len(mm) else -1


def counts(y, lo, hi):
    return int(np.sum((y >= lo) & (y <= hi)))


# ------------------------------------------------------------------ data
def load():
    a = np.load(A1)
    f = np.load(FREE, allow_pickle=True)
    summ = json.load(open(SUMMARY))

    m = a["margins"].astype(np.float64)
    dm = a["dm"][0].astype(np.float64)
    r0 = float(a["radii"][0])
    ref = a["gen_ids"].astype(np.int64)
    top1 = a["top1"].astype(np.int64)
    top2 = a["top2"].astype(np.int64)
    assert np.array_equal(top1, ref), "A1 top1 and gen_ids disagree"

    obs = f["obs"].astype(np.int64)
    agreement = f["agreement"].astype(np.float64)
    radii = f["radii"].astype(np.float64)
    gen = f["gen_ids"]

    assert np.allclose(radii, RHOS)
    assert gen.shape[0] == len(RHOS) and gen.shape[1] == 11
    assert summ["T"] == 300 and summ["n_dirs"] == 1000

    # signed threshold radius, and the npz's sign-blind one
    rc_signed = np.where(dm < 0.0, m[None, :] * r0 / np.maximum(-dm, 1e-300), np.inf)
    rc_abs = np.where(np.abs(dm) > 0.0,
                      m[None, :] * r0 / np.maximum(np.abs(dm), 1e-300), np.inf)
    rc_stored = a["r_c"][0].astype(np.float64)
    assert np.allclose(rc_abs, rc_stored, rtol=1e-6, atol=1e-12), \
        "stored r_c is not m*r/|dm|"

    tstar = rc_signed.argmin(axis=1)
    rstar = rc_signed.min(axis=1)

    # T_pred(rho) = first time in index order at which rho >= r_c
    pred = {}
    for rho in RHOS:
        cross = rc_signed <= rho
        pred[rho] = np.where(cross.any(axis=1), np.argmax(cross, axis=1), -1).astype(np.int64)

    pred_abs = {}
    for rho in RHOS:
        cross = rc_abs <= rho
        pred_abs[rho] = np.where(cross.any(axis=1), np.argmax(cross, axis=1), -1).astype(np.int64)

    return dict(a=a, f=f, summ=summ, m=m, dm=dm, r0=r0, ref=ref, top1=top1,
                top2=top2, obs=obs, agreement=agreement, gen=gen,
                rc_signed=rc_signed, rc_abs=rc_abs, rc_stored=rc_stored,
                tstar=tstar, rstar=rstar, pred=pred, pred_abs=pred_abs)


def check_gen_mapping(D):
    """The writer kept only the last chunk; recover which directions it is."""
    obs = D["obs"]
    ref = D["ref"]
    matches = []
    for ri in range(len(RHOS)):
        got = np.array([first_mismatch(ref, D["gen"][ri, j]) for j in range(D["gen"].shape[1])])
        ok = np.array_equal(got, obs[ri, GEN_DIR0:GEN_DIR0 + 11])
        matches.append(bool(ok))
    assert all(matches), (
        "gen_ids rows do not match obs[...,989:1000]; " + repr(matches))
    # independent check: non-flipped rows are bit-identical to the A1 baseline
    for ri in range(len(RHOS)):
        for j in range(D["gen"].shape[1]):
            u = GEN_DIR0 + j
            row = np.asarray(D["gen"][ri, j]).astype(np.int64)
            if obs[ri, u] < 0:
                assert np.array_equal(row, ref), (ri, u)
    return matches


def classify(D):
    out = {}
    for ri, rho in enumerate(RHOS):
        pred = D["pred"][rho]
        o = D["obs"][ri]
        flip = o >= 0
        exact = int(np.sum(flip & (pred >= 0) & (o == pred)))
        wrong = int(np.sum(flip & (pred >= 0) & (o != pred)))
        false = int(np.sum((~flip) & (pred >= 0)))
        miss = int(np.sum(flip & (pred < 0)))
        none = int(np.sum((~flip) & (pred < 0)))
        assert exact + wrong + false + miss + none == 1000
        out[rho] = dict(exact=exact, wrong_step=wrong, false_alarm=false,
                        miss=miss, both_none=none,
                        n_obs_flip=int(flip.sum()), n_pred_flip=int((pred >= 0).sum()),
                        accuracy=(exact + none) / 1000.0)
    return out


def scorecard_abs(D):
    out = {}
    for ri, rho in enumerate(RHOS):
        pred = D["pred_abs"][rho]
        o = D["obs"][ri]
        flip = o >= 0
        exact = int(np.sum(flip & (pred >= 0) & (o == pred)))
        wrong = int(np.sum(flip & (pred >= 0) & (o != pred)))
        false = int(np.sum((~flip) & (pred >= 0)))
        miss = int(np.sum(flip & (pred < 0)))
        none = int(np.sum((~flip) & (pred < 0)))
        out[rho] = dict(pred_flip=int((pred >= 0).sum()), exact=exact, wrong_step=wrong,
                        false_alarm=false, miss=miss, both_none=none,
                        false_alarm_rate=false / max(int((pred >= 0).sum()), 1))
    return out


def latency(D):
    out = {}
    for ri, rho in enumerate(RHOS):
        o = D["obs"][ri]
        flip = o >= 0
        d = o - D["tstar"]
        p = D["pred"][rho]
        err = o - p
        vals, cnt = np.unique(d[flip], return_counts=True)
        evals, ecnt = np.unique(err[flip & (p >= 0)], return_counts=True)
        out[rho] = dict(
            obs_minus_tstar={int(a): int(b) for a, b in zip(vals, cnt)},
            immediate=int(np.sum(d[flip] == 0)),
            earlier_than_tstar=int(np.sum(d[flip] < 0)),
            delayed=int(np.sum(d[flip] > 0)),
            obs_minus_pred={int(a): int(b) for a, b in zip(evals, ecnt)},
            mean_flip_step=float(o[flip].mean()),
            median_flip_step=float(np.median(o[flip])),
            mean_tstar_flipped=float(D["tstar"][flip].mean()),
            n_flipped=int(flip.sum()),
        )
    return out


def survival(D):
    out = {}
    tgrid = np.arange(0, 301)
    INF = 10 ** 9
    for ri, rho in enumerate(RHOS):
        o = np.where(D["obs"][ri] < 0, INF, D["obs"][ri])
        p = np.where(D["pred"][rho] < 0, INF, D["pred"][rho])
        s_obs = np.array([np.mean(o > t) for t in tgrid])
        s_pred = np.array([np.mean(p > t) for t in tgrid])
        out[rho] = dict(
            t=tgrid, observed=s_obs, predicted=s_pred,
            s19_obs=float(np.mean(o > 19)), s19_pred=float(np.mean(p > 19)),
            s21_obs=float(np.mean(o > 21)), s21_pred=float(np.mean(p > 21)),
            s189_obs=float(np.mean(o > 189)), s189_pred=float(np.mean(p > 189)),
            max_abs_diff=float(np.max(np.abs(s_obs - s_pred))),
            t_max_diff=int(tgrid[int(np.argmax(np.abs(s_obs - s_pred)))]),
        )
    return out


def resync(D):
    """Per-run agreement after the first flip, plus the population algebra."""
    ref = D["ref"]
    obs = D["obs"]
    T = len(ref)
    recs = []
    for ri, rho in enumerate(RHOS):
        for j in range(D["gen"].shape[1]):
            u = GEN_DIR0 + j
            o = int(obs[ri, u])
            row = np.asarray(D["gen"][ri, j]).astype(np.int64)
            if o < 0:
                assert np.array_equal(row, ref)
                continue
            post = row[o + 1:] == ref[o + 1:]
            rr = runs_of(post, o + 1)
            longest = max([e - s + 1 for s, e in rr], default=0)
            inlong = sum(e - s + 1 for s, e in rr if e - s + 1 >= 5)
            npost = len(post)
            ntot = int(post.sum())
            late = row[o + 1 + LATE_OFFSET:] == ref[o + 1 + LATE_OFFSET:]
            # identical suffix length
            k = 0
            while k < T and row[-1 - k] == ref[-1 - k]:
                k += 1
            recs.append(dict(
                radius=float(rho), ri=ri, u=int(u), flip_step=o,
                post_agreement=float(post.mean()),
                late_agreement=float(late.mean()),
                whole_run_agreement=float((row == ref).mean()),
                stored_agreement=float(D["agreement"][ri, u]),
                n_post=npost, n_match=ntot,
                n_match_in_runs_ge5=int(inlong),
                longest_match_run=int(longest),
                n_runs_ge5=int(sum(1 for s, e in rr if e - s + 1 >= 5)),
                window_match={str(k): float((row[o + 1:o + 1 + k] ==
                                             ref[o + 1:o + 1 + k]).mean())
                              for k in (5, 10, 20, 50)},
                identical_suffix=int(k),
                runs=[[int(s), int(e)] for s, e in rr],
                resenthesised=bool(k >= RESYNC_MIN_RUN or longest >= RESYNC_MIN_RUN),
            ))
    # ensemble synchronisation: how many distinct continuations are there?
    uniq = {}
    all_rows = []
    for ri, rho in enumerate(RHOS):
        rows, flipped = [], []
        for j in range(D["gen"].shape[1]):
            row = tuple(np.asarray(D["gen"][ri, j]).astype(np.int64).tolist())
            rows.append(row)
            all_rows.append(row)
            if obs[ri, GEN_DIR0 + j] >= 0:
                flipped.append(np.asarray(row))
        uniq[rho] = dict(
            n_rows=len(rows), n_unique_all=len(set(rows)),
            n_flipped=len(flipped), n_unique_flipped=len(set(map(tuple, flipped))),
        )
        if len(flipped) > 1:
            pw = [float(np.mean(flipped[a] == flipped[b]))
                  for a in range(len(flipped)) for b in range(a + 1, len(flipped))]
            uniq[rho]["pairwise_agreement_flipped_min"] = float(min(pw))
            uniq[rho]["pairwise_agreement_flipped_mean"] = float(np.mean(pw))
    uniq["all_rows"] = dict(n_rows=len(all_rows), n_unique=len(set(all_rows)))

    # population decomposition from the stored whole-run means
    pop = {}
    for ri, rho in enumerate(RHOS):
        f = float((obs[ri] >= 0).mean())
        mean_all = float(D["agreement"][ri].mean())
        mean_flipped_whole = (mean_all - (1.0 - f)) / f
        mean_obs = float(obs[ri][obs[ri] >= 0].mean())
        post = (mean_flipped_whole * T - mean_obs) / (T - mean_obs)
        pop[rho] = dict(
            flip_fraction=f,
            mean_whole_run_all=mean_all,
            mean_whole_run_flipped=float(mean_flipped_whole),
            mean_first_flip_step_flipped=mean_obs,
            mean_postflip_agreement_inferred=float(post),
            summary_field_name="mean_postflip_agreement",
            summary_field_value=float(D["summ"]["mean_postflip_agreement"][ri]),
        )
    return recs, pop, uniq


def flip_words(D, recs):
    tok = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(TOK)
    except Exception as exc:  # pragma: no cover - environment issue
        print("WARNING: tokenizer unavailable:", exc)

    def dec(i):
        if tok is None:
            return f"<id {int(i)}>"
        return tok.decode([int(i)])

    def dec_ids(ids):
        if tok is None:
            return "<ids>"
        return tok.decode([int(x) for x in np.ravel(ids)])

    out = {}
    for name, t in [("t19", 19), ("t21", 21), ("t189", 189), ("t273", 273)]:
        out[name] = dict(
            t=int(t),
            top1_id=int(D["top1"][t]), top1=dec(D["top1"][t]),
            top2_id=int(D["top2"][t]), top2=dec(D["top2"][t]),
            margin_nats=float(D["m"][t]),
            context_baseline=dec_ids(D["ref"][max(0, t - 2):t + 5]),
        )
    # which token the saved flipped runs actually took
    obs = D["obs"]
    taken = {}
    for ri, rho in enumerate(RHOS):
        for j in range(D["gen"].shape[1]):
            u = GEN_DIR0 + j
            o = int(obs[ri, u])
            row = np.asarray(D["gen"][ri, j]).astype(np.int64)
            if o in (19, 21):
                key = f"t{o}"
                d = taken.setdefault(key, {"n": 0, "top1": 0, "top2": 0,
                                           "top1_ids": [], "top2_ids": []})
                d["n"] += 1
                gid = int(row[o])
                if gid == int(D["top1"][o]):
                    d["top1"] += 1
                    d["top1_ids"].append(u)
                elif gid == int(D["top2"][o]):
                    d["top2"] += 1
                    d["top2_ids"].append(u)
    out["taken_by_saved_flips"] = taken
    # all-dir observed flip counts at those steps
    out["obs_flip_counts"] = {
        str(rho): {str(t): int(np.sum(obs[ri] == t)) for t in (19, 21, 189, 273)}
        for ri, rho in enumerate(RHOS)
    }
    out["pred_flip_counts"] = {
        str(rho): {str(t): int(np.sum(D["pred"][rho] == t)) for t in (19, 21, 189, 273)}
        for rho in RHOS
    }
    return out


# ------------------------------------------------------------------ figures
def fig_scorecard(D, cls):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = [f"$\\rho={r:g}$" for r in RHOS]
    x = np.arange(len(RHOS))
    keys = [("exact", "exact"), ("wrong_step", "wrong step"),
            ("false_alarm", "false alarm"), ("miss", "miss"),
            ("both_none", "both none")]
    bottom = np.zeros(len(RHOS))
    small = {rho: [] for rho in RHOS}
    for k, lab in keys:
        vals = np.array([cls[r][k] for r in RHOS], float)
        ax.bar(x, vals, bottom=bottom, width=0.58, color=COL[k], edgecolor="white",
               linewidth=0.8, label=lab)
        for xi, vi, bi, rho in zip(x, vals, bottom, RHOS):
            if vi >= 30:
                ax.text(xi, bi + vi / 2, f"{int(vi)}", ha="center", va="center",
                        fontsize=7.5, color="white" if k != "both_none" else "#333333")
            elif vi > 0:
                small[rho].append((lab, int(vi), bi + vi / 2))
        bottom += vals
    for xi, rho in zip(x, RHOS):
        if not small[rho]:
            continue
        yy = np.mean([s[2] for s in small[rho]])
        txt = ",  ".join(f"{lab} {n}" for lab, n, _ in small[rho])
        ax.annotate(txt, xy=(xi + 0.29, yy), xytext=(xi + 0.36, yy),
                    fontsize=6.8, va="center", ha="left", color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666"))
    for xi, r in zip(x, RHOS):
        ax.text(xi, 1010, f"obs flips {cls[r]['n_obs_flip']}/1000\n"
                          f"pred {cls[r]['n_pred_flip']}/1000",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1085)
    ax.set_ylabel("directions (of 1000)")
    ax.set_xlabel("free-run radius $\\rho$ (A1 calibrated at $r=0.1$)")
    ax.set_title("Free-run scorecard: signed-rule prediction vs observed first flip")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ds14free_scorecard.png"), dpi=170)
    plt.close(fig)


def fig_survival(D, surv):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharey=True)
    for ax, (ri, rho) in zip(axes, enumerate(RHOS)):
        s = surv[rho]
        ax.plot(s["t"], s["observed"], color="black", lw=1.9, drawstyle="steps-post",
                label="observed")
        ax.plot(s["t"], s["predicted"], color=COL["false_alarm"], lw=1.7, ls="--",
                drawstyle="steps-post", label="predicted (signed)")
        for tt, c in [(19, "#777777"), (21, "#777777")]:
            ax.axvline(tt, color=c, lw=0.8, ls=":", zorder=0)
        ax.set_title(f"$\\rho={rho:g}$   flips {int(D['summ']['flip_frac'][ri]*1000)}"
                     f" vs pred {int((D['pred'][rho]>=0).sum())}", fontsize=9.5)
        ax.set_xlim(0, 300)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("generated step $t$")
        if ri == 0:
            ax.set_ylabel("$S(t)=P(\\mathrm{first\\ flip}>t)$")
            ax.legend(loc="upper right")
        ax.text(0.04, 0.55,
                f"$S(19)$ {s['s19_obs']:.3f} / {s['s19_pred']:.3f}\n"
                f"$S(21)$ {s['s21_obs']:.3f} / {s['s21_pred']:.3f}\n"
                f"$S(189)$ {s['s189_obs']:.3f} / {s['s189_pred']:.3f}\n"
                f"max $|\\Delta S|$ {s['max_abs_diff']:.3f}",
                transform=ax.transAxes, fontsize=7.4, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
    fig.suptitle("Survival of the baseline continuation under a free-run perturbation "
                 "(full-attention A1 calibration)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, "ds14free_survival.png"), dpi=170)
    plt.close(fig)


def fig_latency(D, lat):
    STEPS = [-1, 19, 21, 189, 273]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.0))
    for ri, rho in enumerate(RHOS):
        ax = axes[0, ri]
        o = D["obs"][ri]
        vals = o[o >= 0]
        obs_vals = [19, 21, 189, 273]
        hh = [counts(vals, t, t) for t in obs_vals]
        xx = np.arange(len(obs_vals))
        ax.bar(xx, hh, width=0.62, color=RCOL[rho], alpha=0.9)
        for xi, vi in zip(xx, hh):
            if vi > 0:
                ax.text(xi, vi * 1.35, str(vi), ha="center", va="bottom", fontsize=8.5)
        ax.axvspan(-0.45, 1.45, color="#f2e6d8", alpha=0.55, zorder=0)
        ax.text(0.58, 0.66, "the two razor-thin ties", transform=ax.transAxes,
                ha="left", va="center", fontsize=7.2, color="#8a5a20")
        ax.set_yscale("log")
        ax.set_xticks(xx)
        ax.set_xticklabels([str(t) for t in obs_vals])
        ax.set_xlim(-0.6, len(obs_vals) - 0.4)
        ax.set_ylim(0.7, 1800)
        ax.set_title(f"$\\rho={rho:g}$: observed flip steps (n={len(vals)})")
        ax.set_xlabel("first mismatch step (observed values only)")
        if ri == 0:
            ax.set_ylabel("directions (log)")
        ax.text(0.97, 0.95,
                f"never flipped: {int(np.sum(o<0))}\n"
                f"mean {vals.mean():.1f}, median {np.median(vals):.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))

        # categorical bubble matrix: predicted vs observed, all 1000 directions
        ax = axes[1, ri]
        p = D["pred"][rho]
        idx = {v: k for k, v in enumerate(STEPS)}
        pairs = {}
        for pp, oo in zip(p, o):
            pairs[(idx[int(pp)], idx[int(oo)])] = pairs.get((idx[int(pp)], idx[int(oo)]), 0) + 1
        xs = np.array([k[0] for k in pairs])
        ys = np.array([k[1] for k in pairs])
        cs = np.array([pairs[k] for k in pairs], float)
        ax.plot([-0.5, len(STEPS) - 0.5], [-0.5, len(STEPS) - 0.5],
                color="#888888", lw=0.9, ls="--", zorder=0)
        ax.scatter(xs, ys, s=12 + 120 * np.sqrt(cs),
                   c=np.log10(cs), cmap="viridis",
                   vmin=0, vmax=np.log10(max(cs.max(), 2)),
                   edgecolor="white", linewidth=0.6, zorder=3)
        for xx_, yy_, cc in zip(xs, ys, cs):
            ax.text(xx_ + 0.13, yy_ + 0.11, str(int(cc)), fontsize=7.0, color="#111111",
                    zorder=4)
        ax.set_xlim(-0.55, len(STEPS) - 0.35)
        ax.set_ylim(-0.55, len(STEPS) - 0.35)
        ax.set_xticks(range(len(STEPS)))
        ax.set_yticks(range(len(STEPS)))
        ax.set_xticklabels([str(t) for t in STEPS])
        ax.set_yticklabels([str(t) for t in STEPS])
        ax.set_xlabel("predicted first flip $T_{\\rm pred}(\\rho,u)$")
        if ri == 0:
            ax.set_ylabel("observed first flip ${\\rm obs}(\\rho,u)$")
        ax.set_title("prediction vs observation (bubble area $\\propto\\sqrt{n}$)")
        e = lat[rho]["obs_minus_pred"]
        ax.text(0.03, 0.97,
                f"exact {e.get(0,0)}   $\\pm2$ {e.get(-2,0)+e.get(2,0)}\n"
                f"$-1$ means never",
                transform=ax.transAxes, va="top", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
    fig.suptitle("Latency: the flips happen at the two razor-thin ties, and the "
                 "signed rule gets the step right", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGDIR, "ds14free_latency.png"), dpi=170)
    plt.close(fig)


def fig_resync(D, recs, pop):
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8),
                             gridspec_kw={"height_ratios": [1.2, 0.72]})
    if hasattr(axes, "ravel"):
        axes = axes.ravel()
    else:
        axes = np.asarray(axes).ravel()
    for ri, rho in enumerate(RHOS):
        sub = [r for r in recs if r["radius"] == rho]
        ax = axes[ri]
        for lab, sel, col, mk in [
            ("all flipped", sub, "black", "o"),
            ("flip at 21", [r for r in sub if r["flip_step"] == 21], RCOL[rho], "s"),
            ("flip at 19", [r for r in sub if r["flip_step"] == 19], "#2ca02c", "^"),
        ]:
            if not sel:
                continue
            xs = np.sort([r["post_agreement"] for r in sel])
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=col, lw=1.4, marker=mk, ms=4,
                    label=lab)
        ax.axvline(LIT_1P5B_POSTFLIP, color="#c0392b", lw=0.9, ls=":",
                   label="1.5B lit. 2.7%")
        ax.axvline(pop[rho]["mean_postflip_agreement_inferred"], color="#555555",
                   lw=0.8, ls="--", label="all-dir inference")
        late = float(np.mean([r["late_agreement"] for r in sub]))
        if np.isfinite(late):
            ax.axvline(late, color="#1b7a3a", lw=0.8, ls="-.", label="after $t>obs+45$")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"$\\rho={rho:g}$ — ECDF of $A_{{\\rm after}}$", fontsize=9.5)
        ax.set_xlabel("agreement after the first flip, $A_{\\rm after}(u)$")
        if ri == 0:
            ax.set_ylabel("ECDF over flipped directions")
        ax.text(0.97, 0.05,
                f"flipped {len(sub)}/{len([r for r in recs if r['radius']==rho])}\n"
                f"re-sync {sum(r['resenthesised'] for r in sub)}/{len(sub)}\n"
                f"longest run {max([r['longest_match_run'] for r in sub], default=0)}",
                transform=ax.transAxes, fontsize=7.4, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))

        # decomposition of the post-flip positions, normalised
        ax = axes[3 + ri]
        npost = sum(r["n_post"] for r in sub)
        inrun = sum(r["n_match_in_runs_ge5"] for r in sub)
        nmatch = sum(r["n_match"] for r in sub)
        iso = nmatch - inrun
        mism = npost - nmatch
        parts = [("inside runs $\\geq5$", inrun, "#7fb3d5"),
                 ("isolated matches", iso, "#f0b27a"),
                 ("mismatches", mism, "#cfcfcf")]
        left = 0.0
        for lab, val, col in parts:
            frac = val / max(npost, 1)
            ax.barh([0], [frac], left=left, color=col, edgecolor="white", height=0.55,
                    label=lab)
            if frac > 0.03:
                ax.text(left + frac / 2, 0, f"{100*frac:.0f}%", ha="center",
                        va="center", fontsize=8)
            left += frac
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("fraction of post-flip token positions")
        ax.set_title(f"$\\rho={rho:g}$ — decomposition", fontsize=9.5)
        ax.text(0.02, 0.99,
                f"$A_{{\\rm after}}$ {np.mean([r['post_agreement'] for r in sub]):.3f}   "
                f"$A_{{\\rm late}}$ {late:.3f}   "
                f"{100*inrun/max(nmatch,1):.0f}% of matches in runs $\\geq5$\n"
                f"identical suffix 0; runs $\\geq20$: 0",
                transform=ax.transAxes, va="top", fontsize=7.4)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7.8,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Post-flip agreement: the 73/62/48% headline includes non-flipped runs; "
                 "among flips it is ~10%, and 84% of that is shared boilerplate",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, "ds14free_resync.png"), dpi=170)
    plt.close(fig)


def fig_flipword(D, fw):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    info = [("t19", "step $t=19$:  \" reference\" vs \" research\"", 19),
            ("t21", "step $t=21$:  \"Alright\" vs \"Okay\"", 21)]
    for ax, (key, title, t) in zip(axes, info):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        w = fw[key]
        # two token cards
        for x0, lab, txt, tid, col in [
            (0.03, "top-1 (baseline)", w["top1"], w["top1_id"], "#2e7d32"),
            (0.53, "top-2 (the competitor)", w["top2"], w["top2_id"], "#c0392b"),
        ]:
            ax.add_patch(Rectangle((x0, 0.62), 0.44, 0.28, transform=ax.transAxes,
                                   fc="white", ec=col, lw=1.4, zorder=1))
            ax.text(x0 + 0.02, 0.84, lab, transform=ax.transAxes, fontsize=8,
                    color=col, fontweight="bold")
            ax.text(x0 + 0.22, 0.71, f"\"{txt}\"", transform=ax.transAxes,
                    fontsize=17 if len(txt) <= 9 else 15, ha="center", va="center")
            ax.text(x0 + 0.02, 0.645, f"id {tid}", transform=ax.transAxes,
                    fontsize=7.5, color="#666666")
        ax.text(0.5, 0.56, f"margin $m_t$ = {w['margin_nats']:.3e} nats",
                transform=ax.transAxes, ha="center", fontsize=8.5)
        ctx = w["context_baseline"].replace("\n", "\\n")
        ax.text(0.5, 0.47, "baseline text:  ..." + ctx + "...", transform=ax.transAxes,
                ha="center", fontsize=7.4, style="italic", color="#333333")

        # counts at this step
        obs_counts = [fw["obs_flip_counts"][f"{rho:g}"][str(t)] for rho in RHOS]
        ax.text(0.03, 0.34, "observed flips at this step (of 1000):",
                transform=ax.transAxes, fontsize=8, fontweight="bold")
        for k, (rho, c) in enumerate(zip(RHOS, obs_counts)):
            ax.text(0.08 + 0.29 * k, 0.26, f"$\\rho={rho:g}$", transform=ax.transAxes,
                    fontsize=8, ha="center")
            ax.text(0.08 + 0.29 * k, 0.19, str(c), transform=ax.transAxes,
                    fontsize=12, ha="center", fontweight="bold", color=RCOL[rho])
        tk = fw["taken_by_saved_flips"].get(key)
        if tk:
            ax.text(0.03, 0.09,
                    f"saved 11-direction sample: {tk['n']} flips took top-2, "
                    f"{tk['top1']} took top-1", transform=ax.transAxes, fontsize=7.6)
        else:
            ax.text(0.03, 0.09, "no saved sample at this step",
                    transform=ax.transAxes, fontsize=7.6)
    fig.suptitle("What the two razor-thin ties actually are: interchangeable-looking "
                 "words, different continuations", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, "ds14free_flipword.png"), dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    D = load()
    mapping_ok = check_gen_mapping(D)
    cls = classify(D)
    cls_abs = scorecard_abs(D)
    lat = latency(D)
    surv = survival(D)
    recs, pop, uniq = resync(D)
    fw = flip_words(D, recs)

    # verification of the headline summary numbers
    flip_frac = [float((D["obs"][ri] >= 0).mean()) for ri in range(3)]
    mean_step = [float(D["obs"][ri][D["obs"][ri] >= 0].mean()) for ri in range(3)]
    assert np.allclose(flip_frac, D["summ"]["flip_frac"], atol=1e-9)
    assert np.allclose(mean_step, D["summ"]["mean_flip_step"], atol=0.06)
    for ri, rho in enumerate(RHOS):
        assert np.isclose(pop[rho]["summary_field_value"],
                          D["summ"]["mean_postflip_agreement"][ri])

    # per-run resync is measured only on the 11 saved directions; the population
    # inference from the stored whole-run means reproduces it
    sample_post = {}
    for rho in RHOS:
        sub = [r for r in recs if r["radius"] == rho]
        sample_post[rho] = float(np.mean([r["post_agreement"] for r in sub]))

    # decode the longest shared run for the report (it is the echoed prompt phrase)
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOK)
    longest_rec = max(recs, key=lambda r: r["longest_match_run"])
    ls, le = max(longest_rec["runs"], key=lambda se: se[1] - se[0])
    longest_text = _tok.decode([int(x) for x in D["ref"][ls:le + 1]])
    prompt_text = _tok.decode([int(x) for x in D["a"]["prompt_ids"]])

    nums = dict(
        data=dict(
            a1=A1.replace(HERE + os.sep, "").replace("\\", "/"),
            free=FREE.replace(HERE + os.sep, "").replace("\\", "/"),
            summary=SUMMARY.replace(HERE + os.sep, "").replace("\\", "/"),
            model="DeepSeek-R1-Distill-Qwen-14B",
            precision="FP32",
            attention="full attention (window-100 setting inactive; see correction note)",
            prompt="interstellar_travel (P=65 tokens)",
            prompt_text=prompt_text,
            T=300, radii=list(RHOS), n_dirs=1000,
            note="free-run validation is against the FULL-ATTENTION A1 run; a "
                 "true-window version is being generated",
        ),
        definitions=dict(
            m_t="top1-top2 logit gap of the unperturbed A1 run (nats)",
            dm="stored change of that gap at r=0.1 (nats)",
            rc_signed="m_t * 0.1 / max(-dm_t(u), 0); +inf if dm>=0",
            rstar="min_t rc_signed(t,u) (flux-tube radius)",
            tstar="argmin_t rc_signed(t,u) (controlling tie)",
            T_pred="min{t: rho >= rc_signed(t,u)}; -1 if no t crosses",
            obs="first t with free-run token != baseline token; -1 = never",
            A_after="mean over t>obs of (free token == baseline token)",
            A_late="A_after for t > obs+45 (past the echoed task sentence)",
            resync=">=20 identical consecutive post-flip tokens, or identical suffix >=20",
        ),
        headline=dict(
            flip_fraction_observed=flip_frac,
            flip_fraction_pred_signed=[float((D["pred"][r] >= 0).mean()) for r in RHOS],
            flip_fraction_pred_signblind=[float((D["pred_abs"][r] >= 0).mean()) for r in RHOS],
            mean_flip_step_observed=mean_step,
            median_flip_step_observed=[float(np.median(D["obs"][ri][D["obs"][ri] >= 0]))
                                       for ri in range(3)],
            summary_mean_postflip_field=[float(x) for x in D["summ"]["mean_postflip_agreement"]],
            summary_mean_postflip_field_note=(
                "the stored field is a mean over ALL directions and the whole 300-token run, "
                "not a post-flip mean; non-flipped directions contribute 1.0"),
        ),
        scorecard_signed={f"{r:g}": cls[r] for r in RHOS},
        scorecard_signblind={f"{r:g}": cls_abs[r] for r in RHOS},
        latency={f"{r:g}": lat[r] for r in RHOS},
        survival={f"{r:g}": {k: v for k, v in surv[r].items() if k not in ("t",)}
                  for r in RHOS},
        survival_curves={f"{r:g}": dict(t=surv[r]["t"].tolist(),
                                        observed=surv[r]["observed"].tolist(),
                                        predicted=surv[r]["predicted"].tolist())
                         for r in RHOS},
        resync=dict(
            sample_dirs=[GEN_DIR0, GEN_DIR0 + 10],
            sample_dirs_note=("free_run.npz stores raw gen_ids only for the final writer "
                              "chunk (directions 989-999); rows verified against obs"),
            per_run=recs,
            population_inference={f"{r:g}": pop[r] for r in RHOS},
            sample_mean_postflip={f"{r:g}": sample_post[r] for r in RHOS},
            decomposition=dict(
                late_offset=LATE_OFFSET,
                pooled_postflip=float(np.mean([r["post_agreement"] for r in recs])),
                pooled_late=float(np.mean([r["late_agreement"] for r in recs])),
                match_fraction_inside_runs_ge5=float(
                    sum(r["n_match_in_runs_ge5"] for r in recs) /
                    max(sum(r["n_match"] for r in recs), 1)),
                longest_match_run=max(r["longest_match_run"] for r in recs),
                longest_match_run_t=[int(ls), int(le)],
                longest_match_run_text=longest_text,
                longest_match_run_is_prompt_phrase=bool(
                    longest_text.split(" The user")[0] in prompt_text),
                n_runs_ge_RESYNC=sum(r["resenthesised"] for r in recs),
                n_flipped_sample=len(recs),
            ),
            unique_continuations=dict(
                note="all flipped rows at a given gate are bit-identical; "
                     "flips at 19 and 21 differ only at t=19 (agreement 0.9967)",
                by_radius={f"{r:g}": uniq[r] for r in RHOS},
                all_rows=uniq["all_rows"],
            ),
        ),
        flip_words=fw,
        verification=dict(
            gen_mapping_ok=mapping_ok,
            gen_dirs=[GEN_DIR0, GEN_DIR0 + 10],
            stored_rc_is_abs_dm=True,
            top1_equals_gen_ids=True,
            summary_flip_frac_reproduced=True,
            summary_mean_step_reproduced=True,
        ),
    )
    with open(OUTJSON, "w") as fh:
        json.dump(jf(nums), fh, indent=1)

    fig_scorecard(D, cls)
    fig_survival(D, surv)
    fig_latency(D, lat)
    fig_resync(D, recs, pop)
    fig_flipword(D, fw)

    print("flip fractions obs:", [round(x, 4) for x in flip_frac])
    print("flip fractions pred signed:", [round(float((D['pred'][r] >= 0).mean()), 4) for r in RHOS])
    print("scorecard:", {r: cls[r] for r in RHOS})
    print("latency immediate/early/delayed:",
          {r: (lat[r]['immediate'], lat[r]['earlier_than_tstar'], lat[r]['delayed']) for r in RHOS})
    print("post-flip agreement: sample", {r: round(sample_post[r], 4) for r in RHOS},
          "population", {r: round(pop[r]['mean_postflip_agreement_inferred'], 4) for r in RHOS})
    print("late agreement:", {r: round(float(np.mean([x['late_agreement'] for x in recs if x['radius'] == r])), 4) for r in RHOS})
    print("resync:", {r: (sum(x['resenthesised'] for x in recs if x['radius'] == r),
                           sum(1 for x in recs if x['radius'] == r)) for r in RHOS})
    print("flip words:", {k: (fw[k]['top1'], fw[k]['top2']) for k in ('t19', 't21')})
    print("wrote", OUTJSON)


if __name__ == "__main__":
    main()
