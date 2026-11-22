#!/usr/bin/env python
"""Free-run validation of the signed flip rule, DS-R1-Distill-Qwen-14B,
TRUE sliding window 100 (verified actuation).

Validates the teacher-forced, window-100 A1 run (1000 directions, r=0.1)
against the free-running greedy continuations at rho = 0.1, 0.3.

Reads
    results_big/ds14/a1_inter_w100/a1.npz
    results_big/ds14/free_inter_w100/free_run.npz
    results_big/ds14/free_inter_w100/summary.json
    assets/tok_ds14/tokenizer.json
    (for the comparison table) results_big/ds14/a1_inter_full/a1.npz and
    results_big/ds14/free_inter_full/{free_run.npz,summary.json}

Writes
    notebook_big/figures/ds14w100free_{scorecard,survival,latency,resync,flipword}.png
    notebook_big/figures/ds14w100free_numbers.json

Difference to figs_free_full.py: this run's ``gen_ids`` is complete for all
1000 directions at both radii (the accumulation bug was fixed), so the
post-flip / re-synchronisation analysis is a full census, not a sample.

Definitions (u = direction, t = generated step, rho = free-run radius,
r = 0.1 = radius of the teacher-forced A1 run, m_t in nats):

  m_t                 top1-top2 logit gap of the unperturbed A1 run
  dm_t(u)             change of that gap under direction u at radius r
  r_c^-(t,u) = m_t r / max(-dm_t(u), 0)       radius at which t would flip
  r_*(u)     = min_t r_c^-(t,u)               flux-tube radius
  t_*(u)     = argmin_t r_c^-(t,u)            controlling tie
  T_pred(rho,u) = min{t : rho >= r_c^-(t,u)}  first crossing in TIME (signed)
  obs[ri,u]   first mismatching generated token in the free run (-1 = never)
  A_after(u)  mean_t>obs [gen_ids[t] == baseline_gen_ids[t]]
  A_late(u)   the same for t > obs + 45 (after the echoed prompt sentence)

Run from the repository root:
    python scripts_spark/figs_free_w100.py
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
OUTJSON = os.path.join(FIGDIR, "ds14w100free_numbers.json")
TAG = "ds14w100free"

A1 = os.path.join(HERE, "results_big", "ds14", "a1_inter_w100", "a1.npz")
FREE = os.path.join(HERE, "results_big", "ds14", "free_inter_w100", "free_run.npz")
SUMMARY = os.path.join(HERE, "results_big", "ds14", "free_inter_w100", "summary.json")
TOK = os.path.join(HERE, "assets", "tok_ds14", "tokenizer.json")

A1_FULL = os.path.join(HERE, "results_big", "ds14", "a1_inter_full", "a1.npz")
FREE_FULL = os.path.join(HERE, "results_big", "ds14", "free_inter_full", "free_run.npz")
SUM_FULL = os.path.join(HERE, "results_big", "ds14", "free_inter_full", "summary.json")

RHOS = (0.1, 0.3)
GATES = (19, 21, 271, 297)      # every step that occurs in obs or pred
MAIN_GATES = (19, 21, 297)      # the three razor-thin ties
LATE_OFFSET = 45
RESYNC_MIN_RUN = 20
LIT_1P5B_POSTFLIP = 0.027

COL = {
    "exact": "#2e7d32",
    "wrong_step": "#e08a1e",
    "false_alarm": "#c0392b",
    "miss": "#7b4fa0",
    "both_none": "#cfcfcf",
}
RCOL = {0.1: "#d95f02", 0.3: "#6a3d9a"}

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
    if isinstance(o, (np.bool_,)):
        return bool(o)
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


def signed_pred(m, dm, r0, rho):
    rc = np.where(dm < 0.0, m[None, :] * r0 / np.maximum(-dm, 1e-300), np.inf)
    cross = rc <= rho
    pred = np.where(cross.any(axis=1), np.argmax(cross, axis=1), -1).astype(np.int64)
    return rc, pred


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
    gen = f["gen_ids"]          # cache once: NpzFile.__getitem__ re-reads the zip

    assert np.allclose(radii, RHOS), radii
    assert gen.shape == (len(RHOS), 1000, 300), gen.shape
    assert np.array_equal(f["dirs"], np.arange(1000))
    assert summ["T"] == 300 and summ["n_dirs"] == 1000 and summ["window"] == 100

    # signed threshold radius, and the npz's sign-blind one
    rc_signed = np.where(dm < 0.0, m[None, :] * r0 / np.maximum(-dm, 1e-300), np.inf)
    rc_abs = np.where(np.abs(dm) > 0.0,
                      m[None, :] * r0 / np.maximum(np.abs(dm), 1e-300), np.inf)
    rc_stored = a["r_c"][0].astype(np.float64)
    assert np.allclose(rc_abs, rc_stored, rtol=1e-6, atol=1e-12), \
        "stored r_c is not m*r/|dm|"

    tstar = rc_signed.argmin(axis=1)
    rstar = rc_signed.min(axis=1)

    pred, pred_abs = {}, {}
    for rho in RHOS:
        cross = rc_signed <= rho
        pred[rho] = np.where(cross.any(axis=1), np.argmax(cross, axis=1), -1).astype(np.int64)
        cross = rc_abs <= rho
        pred_abs[rho] = np.where(cross.any(axis=1), np.argmax(cross, axis=1), -1).astype(np.int64)

    return dict(a=a, f=f, summ=summ, m=m, dm=dm, r0=r0, ref=ref, top1=top1,
                top2=top2, obs=obs, agreement=agreement, gen=gen,
                rc_signed=rc_signed, rc_abs=rc_abs, rc_stored=rc_stored,
                tstar=tstar, rstar=rstar, pred=pred, pred_abs=pred_abs)


def check_gen_mapping(D):
    """Full census: every row of gen_ids must reproduce obs and the baseline."""
    obs = D["obs"]
    ref = D["ref"]
    g = D["gen"]
    out = {}
    for ri, rho in enumerate(RHOS):
        n_ok = 0
        n_nonflip = 0
        n_nonflip_identical = 0
        for u in range(1000):
            row = np.asarray(g[ri, u]).astype(np.int64)
            assert row.shape == ref.shape, (ri, u, row.shape)
            assert first_mismatch(ref, row) == int(obs[ri, u]), (ri, u)
            n_ok += 1
            if obs[ri, u] < 0:
                n_nonflip += 1
                if np.array_equal(row, ref):
                    n_nonflip_identical += 1
        out[f"{rho:g}"] = dict(rows_checked=1000, obs_reproduced=n_ok,
                               nonflipped=n_nonflip,
                               nonflipped_bit_identical=n_nonflip_identical)
    # stored whole-run agreement must be reproducible from gen_ids
    for ri, rho in enumerate(RHOS):
        rec = np.array([float((np.asarray(g[ri, u]).astype(np.int64) == ref).mean())
                        for u in range(1000)])
        d = float(np.max(np.abs(rec - D["agreement"][ri])))
        out[f"{rho:g}"]["stored_agreement_max_abs_diff"] = d
        assert d < 1e-6, (rho, d)
    return out


def classify(D, pred_key="pred"):
    out = {}
    for ri, rho in enumerate(RHOS):
        pred = D[pred_key][rho]
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
                        false_alarm_rate=false / max(int((pred >= 0).sum()), 1),
                        accuracy=(exact + none) / 1000.0,
                        exact_over_pred=exact / max(int((pred >= 0).sum()), 1),
                        exact_over_obs=exact / max(int(flip.sum()), 1))
    return out


def gate_table(D):
    out = {}
    for ri, rho in enumerate(RHOS):
        o = D["obs"][ri]
        p = D["pred"][rho]
        g = D["gen"]
        rows = {}
        for t in GATES:
            n_obs = int(np.sum(o == t))
            n_pred = int(np.sum(p == t))
            hit = int(np.sum((o == t) & (p == t)))
            is_flip = (o == t)
            took_top1 = took_top2 = took_other = 0
            for u in np.where(is_flip)[0]:
                gid = int(np.asarray(g[ri, u])[t])
                if gid == int(D["top1"][t]):
                    took_top1 += 1
                elif gid == int(D["top2"][t]):
                    took_top2 += 1
                else:
                    took_other += 1
            if n_obs or n_pred:
                rows[str(t)] = dict(obs=n_obs, pred=n_pred, agree=hit,
                                    took_top1=took_top1, took_top2=took_top2,
                                    took_other=took_other)
        out[f"{rho:g}"] = rows
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
        rec = dict(
            t=tgrid, observed=s_obs, predicted=s_pred,
            max_abs_diff=float(np.max(np.abs(s_obs - s_pred))),
            t_max_diff=int(tgrid[int(np.argmax(np.abs(s_obs - s_pred)))]),
        )
        for t in (19, 21, 271, 297):
            rec[f"s{t}_obs"] = float(np.mean(o > t))
            rec[f"s{t}_pred"] = float(np.mean(p > t))
        out[rho] = rec
    return out


def resync(D, tok):
    """Full-census post-flip agreement, re-synchronisation and ensemble collapse."""
    ref = D["ref"]
    obs = D["obs"]
    g = D["gen"]
    T = len(ref)
    recs = []
    for ri, rho in enumerate(RHOS):
        for u in range(1000):
            o = int(obs[ri, u])
            if o < 0:
                continue
            row = np.asarray(g[ri, u]).astype(np.int64)
            post = row[o + 1:] == ref[o + 1:]
            rr = runs_of(post, o + 1)
            longest = max([e - s + 1 for s, e in rr], default=0)
            inlong = sum(e - s + 1 for s, e in rr if e - s + 1 >= 5)
            late = row[o + 1 + LATE_OFFSET:] == ref[o + 1 + LATE_OFFSET:]
            k = 0
            while k < T and row[-1 - k] == ref[-1 - k]:
                k += 1
            recs.append(dict(
                radius=float(rho), ri=ri, u=int(u), flip_step=o,
                n_post=len(post), n_match=int(post.sum()),
                post_agreement=float(post.mean()) if len(post) else float("nan"),
                late_agreement=float(late.mean()) if len(late) else float("nan"),
                whole_run_agreement=float((row == ref).mean()),
                stored_agreement=float(D["agreement"][ri, u]),
                n_match_in_runs_ge5=int(inlong),
                longest_match_run=int(longest),
                n_runs_ge5=int(sum(1 for s, e in rr if e - s + 1 >= 5)),
                identical_suffix=int(k),
                resenthesised=bool(k >= RESYNC_MIN_RUN or longest >= RESYNC_MIN_RUN),
                runs=[[int(s), int(e)] for s, e in rr],
                row_id=int(u),
            ))

    def agg(items):
        if not items:
            return dict(n=0)
        post = [r["post_agreement"] for r in items if np.isfinite(r["post_agreement"])]
        late = [r["late_agreement"] for r in items if np.isfinite(r["late_agreement"])]
        nmatch = sum(r["n_match"] for r in items)
        return dict(
            n=len(items),
            mean_post=float(np.mean(post)) if post else None,
            mean_late=float(np.mean(late)) if late else None,
            mean_whole_run=float(np.mean([r["whole_run_agreement"] for r in items])),
            match_fraction_inside_runs_ge5=float(
                sum(r["n_match_in_runs_ge5"] for r in items) / max(nmatch, 1)),
            longest_match_run=max(r["longest_match_run"] for r in items),
            n_resync=sum(r["resenthesised"] for r in items),
            n_long_horizon=sum(1 for r in items if r["n_post"] >= 100),
        )

    summary = {}
    for ri, rho in enumerate(RHOS):
        sub = [r for r in recs if r["radius"] == rho]
        by_gate = {}
        for t in MAIN_GATES + (271,):
            gate = [r for r in sub if r["flip_step"] == t]
            if gate:
                by_gate[str(t)] = agg(gate)
        # ensemble: how many distinct continuations, per gate and pooled
        rows = [tuple(np.asarray(g[ri, u]).astype(np.int64).tolist()) for u in range(1000)]
        flip_rows = [tuple(np.asarray(g[ri, u]).astype(np.int64).tolist())
                     for u in range(1000) if obs[ri, u] >= 0]
        gate_uniq = {}
        for t in MAIN_GATES + (271,):
            sel = [tuple(np.asarray(g[ri, u]).astype(np.int64).tolist())
                   for u in range(1000) if obs[ri, u] == t]
            if sel:
                gate_uniq[str(t)] = len(set(sel))
        # pairwise agreement over a deterministic sample of flipped rows
        idx = [u for u in range(1000) if obs[ri, u] >= 0]
        step = max(1, len(idx) // 60)
        sample = idx[::step][:60]
        pw = []
        for a_i in range(len(sample)):
            ra = np.asarray(g[ri, sample[a_i]]).astype(np.int64)
            for b_i in range(a_i + 1, len(sample)):
                rb = np.asarray(g[ri, sample[b_i]]).astype(np.int64)
                pw.append(float(np.mean(ra == rb)))
        summary[f"{rho:g}"] = dict(
            n_flipped=len(sub),
            overall=agg(sub),
            by_gate=by_gate,
            flip_steps={str(t): int(np.sum(obs[ri] == t)) for t in GATES},
            ensemble=dict(
                n_rows=1000, n_unique_all=len(set(rows)),
                n_flipped=len(flip_rows), n_unique_flipped=len(set(flip_rows)),
                n_unique_per_gate=gate_uniq,
                pairwise_agreement_sample_n=len(pw),
                pairwise_agreement_min=float(min(pw)) if pw else None,
                pairwise_agreement_mean=float(np.mean(pw)) if pw else None,
            ),
        )

    # per-gate examples (one per gate per radius) for the JSON
    examples = []
    for rho in RHOS:
        for t in MAIN_GATES:
            for r in recs:
                if r["radius"] == rho and r["flip_step"] == t:
                    examples.append({k: v for k, v in r.items() if k != "runs"})
                    break

    # longest shared run text, and where it sits (the echoed prompt sentence)
    longest_rec = max(recs, key=lambda r: r["longest_match_run"])
    ls, le = max(longest_rec["runs"], key=lambda se: se[1] - se[0])
    longest_text = tok.decode([int(x) for x in ref[ls:le + 1]]) if tok else ""
    prompt_text = tok.decode([int(x) for x in D["a"]["prompt_ids"]]) if tok else ""

    # population algebra from the stored whole-run means, as a cross-check
    pop = {}
    for ri, rho in enumerate(RHOS):
        f = float((obs[ri] >= 0).mean())
        mean_all = float(D["agreement"][ri].mean())
        mean_flipped_whole = (mean_all - (1.0 - f)) / f
        mean_obs = float(obs[ri][obs[ri] >= 0].mean())
        post = (mean_flipped_whole * T - mean_obs) / (T - mean_obs)
        pop[rho] = dict(flip_fraction=f, mean_whole_run_all=mean_all,
                        mean_whole_run_flipped=float(mean_flipped_whole),
                        mean_first_flip_step_flipped=mean_obs,
                        mean_postflip_agreement_inferred=float(post),
                        summary_field_value=float(D["summ"]["mean_postflip_agreement"][ri]))
    return recs, summary, examples, pop, dict(
        longest_match_run=longest_rec["longest_match_run"],
        longest_match_run_t=[int(ls), int(le)],
        longest_match_run_text=longest_text,
        longest_match_run_is_prompt_phrase=bool(longest_text.split(" The user")[0] in prompt_text),
    )


def flip_words(D, tok):
    def dec(i):
        return tok.decode([int(i)]) if tok else f"<id {int(i)}>"

    def dec_ids(ids):
        return tok.decode([int(x) for x in np.ravel(ids)]) if tok else "<ids>"

    out = {}
    for name, t in [("t19", 19), ("t21", 21), ("t271", 271), ("t297", 297)]:
        out[name] = dict(
            t=int(t),
            top1_id=int(D["top1"][t]), top1=dec(D["top1"][t]),
            top2_id=int(D["top2"][t]), top2=dec(D["top2"][t]),
            margin_nats=float(D["m"][t]),
            context_baseline=dec_ids(D["ref"][max(0, t - 4):t + 7]),
        )
    out["obs_flip_counts"] = {
        str(rho): {str(t): int(np.sum(D["obs"][ri] == t)) for t in GATES}
        for ri, rho in enumerate(RHOS)
    }
    out["pred_flip_counts"] = {
        str(rho): {str(t): int(np.sum(D["pred"][rho] == t)) for t in GATES}
        for rho in RHOS
    }
    # which token the flipped runs actually took at each gate
    took = {}
    for ri, rho in enumerate(RHOS):
        o = D["obs"][ri]
        for t in MAIN_GATES + (271,):
            sel = np.where(o == t)[0]
            if not len(sel):
                continue
            d = took.setdefault(str(t), {"n": 0, "top1": 0, "top2": 0, "other": 0})
            for u in sel:
                gid = int(np.asarray(D["gen"][ri, u])[t])
                d["n"] += 1
                if gid == int(D["top1"][t]):
                    d["top1"] += 1
                elif gid == int(D["top2"][t]):
                    d["top2"] += 1
                else:
                    d["other"] += 1
    out["taken_by_flips_all_dirs"] = took
    return out


def full_attention_reference():
    """Same scorecard for the full-attention run, for the comparison table."""
    a = np.load(A1_FULL)
    f = np.load(FREE_FULL, allow_pickle=True)
    summ = json.load(open(SUM_FULL))
    m = a["margins"].astype(np.float64)
    dm = a["dm"][0].astype(np.float64)
    r0 = float(a["radii"][0])
    obs = f["obs"].astype(np.int64)
    agreement = f["agreement"].astype(np.float64)
    radii = list(f["radii"].astype(float))
    out = {}
    for rho in (0.1, 0.3):
        ri = radii.index(rho)
        _, pred = signed_pred(m, dm, r0, rho)
        o = obs[ri]
        flip = o >= 0
        exact = int(np.sum(flip & (pred >= 0) & (o == pred)))
        wrong = int(np.sum(flip & (pred >= 0) & (o != pred)))
        false = int(np.sum((~flip) & (pred >= 0)))
        miss = int(np.sum(flip & (pred < 0)))
        none = int(np.sum((~flip) & (pred < 0)))
        fr = float(flip.mean())
        mean_all = float(agreement[ri].mean())
        mean_whole_flip = (mean_all - (1 - fr)) / fr
        mo = float(o[flip].mean())
        post = (mean_whole_flip * 300 - mo) / (300 - mo)
        out[f"{rho:g}"] = dict(
            n_obs_flip=int(flip.sum()), n_pred_flip=int((pred >= 0).sum()),
            exact=exact, wrong_step=wrong, false_alarm=false, miss=miss,
            both_none=none, accuracy=(exact + none) / 1000.0,
            flip_fraction=fr,
            postflip_agreement_inferred=float(post),
            flip_steps={str(t): int(np.sum(o == t)) for t in GATES},
            mean_flip_step=mo,
        )
    out["_meta"] = dict(
        free_gen_ids_dirs=int(f["gen_ids"].shape[1]),
        summary_window=summ.get("window"),
        note="full-attention free run stores gen_ids for the final chunk only",
    )
    return out


# ------------------------------------------------------------------ figures
def fig_scorecard(D, cls):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [f"$\\rho={r:g}$" for r in RHOS]
    x = np.arange(len(RHOS))
    keys = [("exact", "exact"), ("wrong_step", "wrong step"),
            ("false_alarm", "false alarm"), ("miss", "miss"),
            ("both_none", "both none")]
    bottom = np.zeros(len(RHOS))
    for k, lab in keys:
        vals = np.array([cls[r][k] for r in RHOS], float)
        ax.bar(x, vals, bottom=bottom, width=0.5, color=COL[k], edgecolor="white",
               linewidth=0.8, label=lab)
        for xi, vi, bi in zip(x, vals, bottom):
            if vi >= 30:
                ax.text(xi, bi + vi / 2, f"{int(vi)}", ha="center", va="center",
                        fontsize=8, color="white" if k != "both_none" else "#333333")
        bottom += vals
    for xi, rho, r in zip(x, RHOS, RHOS):
        c = cls[rho]
        ax.text(xi + 0.29, 900,
                f"exact {c['exact']}\nwrong {c['wrong_step']}\n"
                f"false {c['false_alarm']}\nmiss {c['miss']}\n"
                f"obs {c['n_obs_flip']}\npred {c['n_pred_flip']}",
                fontsize=6.8, va="top", ha="left", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(RHOS) - 0.05)
    ax.set_ylim(0, 1000)
    ax.set_ylabel("directions (of 1000)")
    ax.set_xlabel("free-run radius $\\rho$ (A1 calibrated at $r=0.1$)")
    ax.set_title("True window 100: signed-rule prediction vs observed first flip")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{TAG}_scorecard.png"), dpi=170)
    plt.close(fig)


def fig_survival(D, surv):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (ri, rho) in zip(axes, enumerate(RHOS)):
        s = surv[rho]
        ax.plot(s["t"], s["observed"], color="black", lw=1.9, drawstyle="steps-post",
                label="observed")
        ax.plot(s["t"], s["predicted"], color=COL["false_alarm"], lw=1.7, ls="--",
                drawstyle="steps-post", label="predicted (signed)")
        for tt in (19, 21, 297):
            ax.axvline(tt, color="#777777", lw=0.8, ls=":", zorder=0)
        ax.set_xlim(0, 300)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("generated step $t$")
        ax.set_title(f"$\\rho={rho:g}$   obs {int(D['summ']['flip_frac'][ri]*1000)}"
                     f" vs pred {int((D['pred'][rho]>=0).sum())}", fontsize=9.5)
        if ri == 0:
            ax.set_ylabel("$S(t)=P(\\mathrm{first\\ flip}>t)$")
            ax.legend(loc="upper right")
        ax.text(0.04, 0.42,
                f"$S(19)$ {s['s19_obs']:.3f} / {s['s19_pred']:.3f}\n"
                f"$S(21)$ {s['s21_obs']:.3f} / {s['s21_pred']:.3f}\n"
                f"$S(271)$ {s['s271_obs']:.3f} / {s['s271_pred']:.3f}\n"
                f"$S(297)$ {s['s297_obs']:.3f} / {s['s297_pred']:.3f}\n"
                f"max $|\\Delta S|$ {s['max_abs_diff']:.3f}",
                transform=ax.transAxes, fontsize=7.4, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
    fig.suptitle("Survival under a free-run perturbation, true window 100 "
                 "(gates at 19, 21, 297)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGDIR, f"{TAG}_survival.png"), dpi=170)
    plt.close(fig)


def fig_latency(D, lat):
    STEPS = [-1, 19, 21, 271, 297]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8))
    for ri, rho in enumerate(RHOS):
        ax = axes[0, ri]
        o = D["obs"][ri]
        vals = o[o >= 0]
        counts_ = {t: int(np.sum(vals == t)) for t in (19, 21, 271, 297)}
        xx = np.arange(4)
        hh = [counts_[t] for t in (19, 21, 271, 297)]
        ax.bar(xx, hh, width=0.62, color=RCOL[rho], alpha=0.9)
        for xi, vi in zip(xx, hh):
            if vi > 0:
                ax.text(xi, vi * 1.35, str(vi), ha="center", va="bottom", fontsize=8.5)
        ax.axvspan(-0.45, 1.45, color="#f2e6d8", alpha=0.55, zorder=0)
        ax.set_yscale("log")
        ax.set_xticks(xx)
        ax.set_xticklabels(["19", "21", "271", "297"])
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0.7, 2500)
        ax.set_title(f"$\\rho={rho:g}$: observed flip steps (n={len(vals)})")
        ax.set_xlabel("first mismatch step (shaded: the razor-thin ties)")
        if ri == 0:
            ax.set_ylabel("directions (log)")
        ax.text(0.97, 0.95,
                f"never flipped: {int(np.sum(o<0))}\n"
                f"mean {vals.mean():.1f}, median {np.median(vals):.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))

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
        ax.scatter(xs, ys, s=12 + 40 * np.sqrt(cs),
                   c=np.log10(cs), cmap="viridis",
                   vmin=0, vmax=np.log10(max(cs.max(), 2)),
                   edgecolor="white", linewidth=0.6, zorder=3)
        for xx_, yy_, cc in zip(xs, ys, cs):
            ax.text(xx_ + 0.13, yy_ + 0.11, str(int(cc)), fontsize=6.8, color="#111111",
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
        ax.set_title("prediction vs observation")
        e = lat[rho]["obs_minus_pred"]
        ax.text(0.03, 0.97,
                f"exact {e.get(0,0)}   $\\pm2$ {e.get(-2,0)+e.get(2,0)}\n"
                f"$-1$ means never",
                transform=ax.transAxes, va="top", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
    fig.suptitle("Latency, true window 100", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGDIR, f"{TAG}_latency.png"), dpi=170)
    plt.close(fig)


def fig_resync(D, recs, summary):
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6),
                             gridspec_kw={"height_ratios": [1.2, 0.72]})
    axes = np.asarray(axes).ravel()
    for ri, rho in enumerate(RHOS):
        sub = [r for r in recs if r["radius"] == rho]
        # long-horizon flips only: obs <= 21 leaves >= 278 post-flip tokens
        lh = [r for r in sub if r["flip_step"] <= 21]
        ax = axes[ri]
        for lab, sel, col in [
            ("all long-horizon", lh, "black"),
            ("flip at 21", [r for r in sub if r["flip_step"] == 21], RCOL[rho]),
            ("flip at 19", [r for r in sub if r["flip_step"] == 19], "#2ca02c"),
        ]:
            if not sel:
                continue
            xs = np.sort([r["post_agreement"] for r in sel])
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=col, lw=1.6, label=f"{lab} (n={len(sel)})")
        ax.axvline(LIT_1P5B_POSTFLIP, color="#c0392b", lw=0.9, ls=":",
                   label="1.5B lit. 2.7%")
        late = float(np.mean([r["late_agreement"] for r in lh]))
        ax.axvline(late, color="#1b7a3a", lw=0.9, ls="-.", label="after $t>obs+45$")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"$\\rho={rho:g}$ — ECDF of $A_{{\\rm after}}$, flips at 19/21")
        ax.set_xlabel("agreement after the first flip")
        if ri == 0:
            ax.set_ylabel("ECDF over flipped directions")
        ax.legend(loc="lower right", fontsize=7.0)

        # decomposition, pooled over the long-horizon flips (obs <= 21) at this radius
        ax = axes[2 + ri]
        npost = sum(r["n_post"] for r in lh)
        inrun = sum(r["n_match_in_runs_ge5"] for r in lh)
        nmatch = sum(r["n_match"] for r in lh)
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
            if frac > 0.04:
                ax.text(left + frac / 2, 0, f"{100*frac:.0f}%", ha="center",
                        va="center", fontsize=8)
            left += frac
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("fraction of post-flip token positions")
        ax.set_title(f"$\\rho={rho:g}$ — decomposition, flips at 19/21 (n={len(lh)})",
                     fontsize=9.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3,
                  fontsize=7.0)
        a = summary[f"{rho:g}"]["overall"]
        ax.text(0.02, 0.95,
                f"$A_{{\\rm after}}$ (19/21) {np.mean([r['post_agreement'] for r in lh]):.3f}   "
                f"$A_{{\\rm late}}$ {late:.3f}\n"
                f"re-sync {a['n_resync']}/{len(sub)}   longest run {a['longest_match_run']}",
                transform=ax.transAxes, va="top", fontsize=7.4)
    fig.suptitle("Post-flip agreement, true window 100: full census over all flipped "
                 "directions", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGDIR, f"{TAG}_resync.png"), dpi=170)
    plt.close(fig)


def fig_flipword(D, fw):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    info = [("t19", "step $t=19$", 19), ("t21", "step $t=21$", 21),
            ("t297", "step $t=297$", 297)]
    for ax, (key, title, t) in zip(axes, info):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        w = fw[key]
        ax.set_title(f"{title}:  \"{w['top1']}\" vs \"{w['top2']}\"", fontsize=10)
        for x0, lab, txt, tid, col in [
            (0.03, "top-1 (baseline)", w["top1"], w["top1_id"], "#2e7d32"),
            (0.53, "top-2 (competitor)", w["top2"], w["top2_id"], "#c0392b"),
        ]:
            ax.add_patch(Rectangle((x0, 0.66), 0.44, 0.24, transform=ax.transAxes,
                                   fc="white", ec=col, lw=1.4, zorder=1))
            ax.text(x0 + 0.02, 0.85, lab, transform=ax.transAxes, fontsize=7.6,
                    color=col, fontweight="bold")
            ax.text(x0 + 0.22, 0.74, f"\"{txt}\"", transform=ax.transAxes,
                    fontsize=15, ha="center", va="center")
            ax.text(x0 + 0.02, 0.68, f"id {tid}", transform=ax.transAxes,
                    fontsize=7.2, color="#666666")
        ax.text(0.5, 0.60, f"margin $m_t$ = {w['margin_nats']:.3e} nats",
                transform=ax.transAxes, ha="center", fontsize=8.5)
        ctx = w["context_baseline"].replace("\n", "\\n")
        ax.text(0.5, 0.51, "baseline:  ..." + ctx + "...", transform=ax.transAxes,
                ha="center", fontsize=7.0, style="italic", color="#333333")
        obs_counts = [fw["obs_flip_counts"][f"{rho:g}"][str(t)] for rho in RHOS]
        ax.text(0.03, 0.36, "observed first flips (of 1000):",
                transform=ax.transAxes, fontsize=8, fontweight="bold")
        for k, (rho, c) in enumerate(zip(RHOS, obs_counts)):
            ax.text(0.14 + 0.3 * k, 0.27, f"$\\rho={rho:g}$", transform=ax.transAxes,
                    fontsize=8, ha="center")
            ax.text(0.14 + 0.3 * k, 0.19, str(c), transform=ax.transAxes,
                    fontsize=13, ha="center", fontweight="bold", color=RCOL[rho])
        tk = fw["taken_by_flips_all_dirs"].get(str(t))
        if tk:
            ax.text(0.03, 0.07,
                    f"all flipped runs: {tk['top2']} took top-2, "
                    f"{tk['top1']} top-1, {tk['other']} other",
                    transform=ax.transAxes, fontsize=7.4)
    fig.suptitle("What the three window-100 ties compete between "
                 "(all 1186 flipped rows decoded, 1000 directions $\\times$ 2 radii)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGDIR, f"{TAG}_flipword.png"), dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    D = load()
    mapping = check_gen_mapping(D)
    cls = classify(D)
    cls_abs = classify(D, "pred_abs")
    gate = gate_table(D)
    lat = latency(D)
    surv = survival(D)
    recs, rsum, examples, pop, longrun = resync(D, None)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    fw = flip_words(D, tok)
    recs, rsum, examples, pop, longrun = resync(D, tok)

    flip_frac = [float((D["obs"][ri] >= 0).mean()) for ri in range(len(RHOS))]
    mean_step = [float(D["obs"][ri][D["obs"][ri] >= 0].mean()) for ri in range(len(RHOS))]
    assert np.allclose(flip_frac, D["summ"]["flip_frac"], atol=1e-9)
    assert np.allclose(mean_step, D["summ"]["mean_flip_step"], atol=0.06)
    for ri, rho in enumerate(RHOS):
        assert np.isclose(pop[rho]["summary_field_value"],
                          D["summ"]["mean_postflip_agreement"][ri])

    long_horizon = {}
    for rho in RHOS:
        lh = [r for r in recs if r["radius"] == rho and r["flip_step"] <= 21]
        nmatch = sum(r["n_match"] for r in lh)
        long_horizon[f"{rho:g}"] = dict(
            n=len(lh),
            mean_post=float(np.mean([r["post_agreement"] for r in lh])),
            mean_late=float(np.mean([r["late_agreement"] for r in lh])),
            match_fraction_inside_runs_ge5=float(
                sum(r["n_match_in_runs_ge5"] for r in lh) / max(nmatch, 1)),
            longest_match_run=max(r["longest_match_run"] for r in lh),
            n_resync=sum(r["resenthesised"] for r in lh),
            n_post_positions=sum(r["n_post"] for r in lh),
            n_matches=nmatch,
        )

    nums = dict(
        data=dict(
            a1=A1.replace(HERE + os.sep, "").replace("\\", "/"),
            free=FREE.replace(HERE + os.sep, "").replace("\\", "/"),
            summary=SUMMARY.replace(HERE + os.sep, "").replace("\\", "/"),
            model="DeepSeek-R1-Distill-Qwen-14B",
            precision="FP32",
            attention="TRUE sliding window 100 (verified actuation)",
            prompt="interstellar_travel (P=65 tokens)",
            prompt_text=tok.decode([int(x) for x in D["a"]["prompt_ids"]]),
            T=300, radii=list(RHOS), n_dirs=1000,
            gen_ids="complete: 2 x 1000 x 300 (accumulation bug fixed)",
        ),
        definitions=dict(
            m_t="top1-top2 logit gap of the unperturbed window-100 A1 run (nats)",
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
                                       for ri in range(len(RHOS))],
            summary_mean_postflip_field=[float(x) for x in D["summ"]["mean_postflip_agreement"]],
            summary_mean_postflip_field_note=(
                "stored field is a mean over ALL directions and the whole 300-token run, "
                "not a post-flip mean; non-flipped directions contribute 1.0"),
        ),
        scorecard_signed={f"{r:g}": cls[r] for r in RHOS},
        scorecard_signblind={f"{r:g}": cls_abs[r] for r in RHOS},
        gate_table={k: v for k, v in gate.items()},
        latency={f"{r:g}": lat[r] for r in RHOS},
        survival={f"{r:g}": {k: v for k, v in surv[r].items() if k != "t"}
                  for r in RHOS},
        survival_curves={f"{r:g}": dict(t=surv[r]["t"].tolist(),
                                        observed=surv[r]["observed"].tolist(),
                                        predicted=surv[r]["predicted"].tolist())
                         for r in RHOS},
        resync=dict(
            per_radius=rsum,
            examples=examples,
            long_horizon=long_horizon,
            population_inference={f"{r:g}": pop[r] for r in RHOS},
            longest_run=longrun,
        ),
        flip_words=fw,
        verification=dict(
            gen_mapping=mapping,
            top1_equals_gen_ids=True,
            stored_rc_is_abs_dm=True,
            summary_flip_frac_reproduced=True,
            summary_mean_step_reproduced=True,
            stored_agreement_reproduced_from_gen_ids=True,
        ),
        comparison_full_attention_vs_window=dict(
            full_attention=full_attention_reference(),
            true_window={f"{r:g}": dict(
                n_obs_flip=cls[r]["n_obs_flip"], n_pred_flip=cls[r]["n_pred_flip"],
                exact=cls[r]["exact"], wrong_step=cls[r]["wrong_step"],
                false_alarm=cls[r]["false_alarm"], miss=cls[r]["miss"],
                both_none=cls[r]["both_none"], accuracy=cls[r]["accuracy"],
                flip_fraction=float((D["obs"][list(RHOS).index(r)] >= 0).mean()),
                postflip_agreement_measured=long_horizon[f"{r:g}"]["mean_post"],
                postflip_agreement_inferred=pop[r]["mean_postflip_agreement_inferred"],
                mean_flip_step=mean_step[list(RHOS).index(r)],
                flip_steps={str(t): int(np.sum(D["obs"][list(RHOS).index(r)] == t))
                            for t in GATES},
            ) for r in RHOS},
        ),
    )
    with open(OUTJSON, "w") as fh:
        json.dump(jf(nums), fh, indent=1)

    fig_scorecard(D, cls)
    fig_survival(D, surv)
    fig_latency(D, lat)
    fig_resync(D, recs, rsum)
    fig_flipword(D, fw)

    print("gen mapping:", json.dumps(mapping))
    print("flip fractions obs:", [round(x, 4) for x in flip_frac])
    print("flip fractions pred signed:", [round(float((D['pred'][r] >= 0).mean()), 4)
                                          for r in RHOS])
    print("scorecard signed:", json.dumps({f"{r:g}": cls[r] for r in RHOS}, indent=0))
    print("scorecard signblind:", json.dumps({f"{r:g}": cls_abs[r] for r in RHOS}, indent=0))
    print("gate table:", json.dumps(gate, indent=0))
    print("latency:", json.dumps({f"{r:g}": {k: lat[r][k] for k in
                                              ('immediate', 'earlier_than_tstar', 'delayed',
                                               'obs_minus_pred', 'mean_flip_step',
                                               'median_flip_step')} for r in RHOS}, indent=0))
    print("survival:", json.dumps({f"{r:g}": {k: round(v, 4) for k, v in surv[r].items()
                                               if k != 't' and isinstance(v, float)}
                                   for r in RHOS}, indent=0))
    print("resync overall:", json.dumps({k: v["overall"] for k, v in rsum.items()}, indent=0))
    print("resync ensemble:", json.dumps({k: v["ensemble"] for k, v in rsum.items()}, indent=0))
    print("long horizon:", json.dumps(long_horizon, indent=0))
    print("population:", json.dumps({f"{r:g}": pop[r] for r in RHOS}, indent=0))
    print("longest run:", json.dumps(longrun, indent=0))
    print("flip words:", json.dumps({k: (fw[k]["top1"], fw[k]["top2"], fw[k]["margin_nats"])
                                     for k in ("t19", "t21", "t271", "t297")}, indent=0))
    print("taken:", json.dumps(fw["taken_by_flips_all_dirs"]))
    print("wrote", OUTJSON)


if __name__ == "__main__":
    main()
