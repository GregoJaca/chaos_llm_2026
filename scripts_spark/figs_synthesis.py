#!/usr/bin/env python
"""Cross-model synthesis: the flux-tube radius, stability, geometry, dimension.

Reads (all local, nothing is written outside notebook_big/figures):
    1.5B   chaos_llm_2026/results/local_data/w100_layers/a1.npz   (read only)
           chaos_llm_2026/results/a1_ds_w100/free_run/free_run.npz (r<=0.01 null)
    DS-14B results_big/ds14/{a1_inter_w100,a1_inter_full,calib,free_inter_w100,
                             depth_w100,nonnorm_w100,state_w100,d2}
    Qwen3.5-9B results_big/qwen35/{a1_inter_w0,calib,free_inter_w0,d2}
    phi-4      results_big/phi4/{a1_inter_w0,calib,free_inter_w0,depth_w0,d2 absent}
    Qwen3-14B  results_big/qwen314/... (optional; marked pending when absent)

Writes
    notebook_big/figures/syn_fragility.png
    notebook_big/figures/syn_stability.png
    notebook_big/figures/syn_angle_depth.png
    notebook_big/figures/syn_dimension.png
    notebook_big/figures/syn_numbers.json

Conventions
    m_t        = top1 - top2 logit gap of the unperturbed run (nats)
    dm_t(u)    = m_t(u) - m_t at the campaign radius r=0.1 (nats)
    s_t(u)     = |dm_t(u)| / r          local decision sensitivity (nats per unit radius)
    r_c(t,u)   = m_t r / max(-dm_t(u),0)  signed threshold radius (inf if dm>=0)
    r_*(u)     = min_t r_c(t,u)         flux-tube radius
    t_*(u)     = argmin_t r_c(t,u)      controlling step
    m_*, s_*   = m_t and s_t at t_*(u), so r_* = m_* / s_* exactly per direction

Run from the repository root:
    python scripts_spark/figs_synthesis.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "notebook_big", "figures")
OUTJSON = os.path.join(FIGDIR, "syn_numbers.json")

B15 = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
B15_FREE = r"C:\Users\hjaca\research\chaos_llm_2026\results\a1_ds_w100\free_run\free_run.npz"

R_RUN = 0.1
EPS32 = float(np.finfo(np.float32).eps)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9.5,
    "axes.titlesize": 10.3,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.2,
})

# one colour per model, used in every figure
C = {
    "1.5B": "#1f77b4",
    "DS-14B": "#d62728",
    "Qwen3.5-9B": "#2ca02c",
    "phi-4": "#9467bd",
    "Qwen3-14B": "#8c564b",
}
ORDER = ["1.5B", "DS-14B", "Qwen3.5-9B", "phi-4"]


# ------------------------------------------------------------------ models
def model_cfg():
    """Paths and static metadata.  Qwen3-14B is included only if present."""
    cfg = {
        "1.5B": dict(
            a1=B15, free=B15_FREE,
            blocks=28, hidden=1536, K=20, window=100, prompt_len=65,
            attention="sliding window 100",
            source="chaos_llm_2026/results/local_data/w100_layers/a1.npz",
            note="K=20, one simplex block; the r* ECDF is a 20-step staircase",
        ),
        "DS-14B": dict(
            a1=os.path.join(HERE, "results_big/ds14/a1_inter_w100/a1.npz"),
            free=os.path.join(HERE, "results_big/ds14/free_inter_w100/free_run.npz"),
            blocks=48, hidden=5120, K=1000, window=100, prompt_len=65,
            attention="sliding window 100 (TRUE window; layer_types fixed)",
            source="results_big/ds14/a1_inter_w100/a1.npz",
            note="core column is the true-window run; the older full-attention "
                 "r* median 0.1164 is reported as a check",
        ),
        "Qwen3.5-9B": dict(
            a1=os.path.join(HERE, "results_big/qwen35/a1_inter_w0/a1.npz"),
            free=os.path.join(HERE, "results_big/qwen35/free_inter_w0/free_run.npz"),
            blocks=32, hidden=4096, K=1000, window=0, prompt_len=64,
            attention="hybrid: 3 Gated DeltaNet blocks + 1 full-attention block; no window",
            source="results_big/qwen35/a1_inter_w0/a1.npz",
            note="last block (31) not saved; dh_last is the final-norm output",
        ),
        "phi-4": dict(
            a1=os.path.join(HERE, "results_big/phi4/a1_inter_w0/a1.npz"),
            free=os.path.join(HERE, "results_big/phi4/free_inter_w0/free_run.npz"),
            blocks=40, hidden=5120, K=1000, window=0, prompt_len=64,
            attention="full; no window",
            source="results_big/phi4/a1_inter_w0/a1.npz",
            note="39 blocks saved; dh_last is the final-norm output",
        ),
    }
    q314 = os.path.join(HERE, "results_big/qwen314")
    a314 = os.path.join(q314, "a1_inter_w0", "a1.npz")
    f314 = os.path.join(q314, "free_inter_w0", "free_run.npz")
    if os.path.exists(a314):
        cfg["Qwen3-14B"] = dict(
            a1=a314, free=f314,
            blocks=None, hidden=None, K=None, window=0, prompt_len=None,
            attention="dense; window 0",
            source="results_big/qwen314/a1_inter_w0/a1.npz",
            note="auto-detected at run time",
        )
        cfg["Qwen3-14B"]["present"] = True
    return cfg


CFG = model_cfg()
MODELS = [k for k in ORDER if k in CFG] + (["Qwen3-14B"] if "Qwen3-14B" in CFG else [])


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
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def qs(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p}": None for p in ps} | {"n": 0}
    out = {f"p{p}": float(np.percentile(x, p)) for p in ps}
    out.update(n=int(x.size), min=float(x.min()), max=float(x.max()),
               mean=float(x.mean()), median=float(np.median(x)))
    return out


def ecdf(x):
    x = np.sort(np.asarray(x, float))
    return x, np.arange(1, x.size + 1) / x.size


def core_stats(path):
    """Signed flux-tube decomposition at the campaign radius."""
    d = np.load(path, allow_pickle=True)
    m = d["margins"].astype(np.float64)
    dm = d["dm"]
    radii = np.atleast_1d(d["radii"]).astype(np.float64)
    ri = int(np.argmin(np.abs(radii - R_RUN)))
    r0 = float(radii[ri])
    D = dm[ri].astype(np.float64) if dm.ndim == 3 else dm[0, ri].astype(np.float64)
    S = np.abs(D) / r0                       # (K, T) local sensitivity
    den = np.maximum(-D, 0.0)                # only a gap-closing response can flip
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = np.where(den > 0.0, m[None, :] * r0 / den, np.inf)
    tstar = rc.argmin(axis=1)
    idx = np.arange(rc.shape[0])
    rstar = rc[idx, tstar]
    mstar = m[tstar]
    sstar = S[idx, tstar]
    good = np.isfinite(rstar)
    return dict(d=d, m=m, dm=D, radii=radii, r0=r0, rc=rc, tstar=tstar,
                rstar=rstar, mstar=mstar, sstar=sstar, good=good)


def decomposition(st):
    """Two exact decompositions of the *same* summary statistic.

    (i) median direction: u with median log r*, and its own m*, s*; then
        r*(u) = m*(u)/s*(u) exactly.
    (ii) geometric mean over directions: mean(log r*) = mean(log m*) -
        mean(log s*) exactly (log-means are additive, medians are not).
    """
    lr = np.log10(st["rstar"])
    lm = np.log10(st["mstar"])
    ls = np.log10(st["sstar"])
    j = int(np.argsort(lr)[lr.size // 2])
    e = int(np.argmin(st["rstar"]))
    return dict(
        median_direction=dict(
            u=j, tstar=int(st["tstar"][j]),
            mstar=float(st["mstar"][j]), sstar=float(st["sstar"][j]),
            rstar=float(st["rstar"][j]),
            check=float(st["mstar"][j] / st["sstar"][j] - st["rstar"][j]),
        ),
        extreme_direction=dict(
            u=e, tstar=int(st["tstar"][e]),
            mstar=float(st["mstar"][e]), sstar=float(st["sstar"][e]),
            rstar=float(st["rstar"][e]),
            check=float(st["mstar"][e] / st["sstar"][e] - st["rstar"][e]),
        ),
        geometric_mean=dict(
            mstar=float(10 ** np.mean(lm)), sstar=float(10 ** np.mean(ls)),
            rstar=float(10 ** np.mean(lr)),
            log10=dict(mstar=float(np.mean(lm)), sstar=float(np.mean(ls)),
                       rstar=float(np.mean(lr))),
        ),
        marginal_medians=dict(
            mstar=float(np.median(st["mstar"])),
            sstar=float(np.median(st["sstar"])),
            rstar=float(np.median(st["rstar"])),
            note="medians of m* and s* are taken over different directions and "
                 "do not multiply to median r*",
        ),
        corr=dict(
            logm_logs=float(np.corrcoef(lm, ls)[0, 1]),
            logm_logr=float(np.corrcoef(lm, lr)[0, 1]),
            logs_logr=float(np.corrcoef(ls, lr)[0, 1]),
            slope_logr_on_logm=float(np.polyfit(lm, lr, 1)[0]),
            slope_logr_on_logs=float(np.polyfit(ls, lr, 1)[0]),
        ),
        percentiles=dict(rstar=qs(st["rstar"]), mstar=qs(st["mstar"]),
                         sstar=qs(st["sstar"])),
    )


def controlling(st):
    t = st["tstar"]
    vals, cnt = np.unique(t, return_counts=True)
    o = np.argsort(-cnt)
    top = [dict(t=int(vals[i]), n=int(cnt[i])) for i in o[:8]]
    top3 = float(cnt[o[:3]].sum() / t.size)
    # share of directions on the five thinnest baseline steps
    five = np.argsort(st["m"])[:5]
    five_share = float(np.isin(t, five).mean())
    return dict(top=top, n_distinct=int(len(vals)), top3_share=top3,
                share_on_5_thinnest_steps=five_share,
                five_thinnest=[dict(t=int(i), m=float(st["m"][i])) for i in five])


def free_scorecard(cfg, st, radii):
    """Observed (free run or teacher-forced) vs the signed linear rule.

    The predicted step is the FIRST crossing of r_c <= rho, not argmin r_c;
    this is the convention of the committed free-run analyses and it is what
    makes a 21-prediction / 19-observation pair count as a wrong step.
    """
    out = []
    free = None
    if cfg.get("free") and os.path.exists(cfg["free"]):
        try:
            free = np.load(cfg["free"], allow_pickle=True)
        except Exception:
            free = None
    for rho in radii:
        cross = st["rc"] <= rho
        pred = np.where(cross.any(axis=1), cross.argmax(axis=1), -1).astype(np.int64)
        if free is not None and np.any(np.isclose(np.atleast_1d(free["radii"]), rho)):
            j = int(np.argmin(np.abs(np.atleast_1d(free["radii"]) - rho)))
            obs = free["obs"][j].astype(np.int64)
            src = "free run"
        else:
            # teacher-forced decision flip at that radius (1.5B has 12 radii)
            d = st["d"]
            radii_all = st["radii"]
            j = int(np.argmin(np.abs(radii_all - rho)))
            dmj = d["dm"]
            Dr = dmj[j].astype(np.float64) if dmj.ndim == 3 else dmj[0, j].astype(np.float64)
            fl = st["m"][None, :] + Dr < 0
            obs = np.where(fl.any(axis=1), fl.argmax(axis=1), -1).astype(np.int64)
            src = "teacher-forced A1 pass"
        flip = obs >= 0
        exact = int(np.sum(flip & (pred >= 0) & (obs == pred)))
        wrong = int(np.sum(flip & (pred >= 0) & (obs != pred)))
        fa = int(np.sum((~flip) & (pred >= 0)))
        miss = int(np.sum(flip & (pred < 0)))
        none = int(np.sum((~flip) & (pred < 0)))
        nobs = int(flip.sum())
        npred = int((pred >= 0).sum())
        out.append(dict(
            rho=float(rho), observed=nobs, predicted=npred,
            observed_fraction=nobs / obs.size, predicted_fraction=npred / obs.size,
            exact=exact, wrong_step=wrong, false_alarm=fa, miss=miss,
            both_none=none, n=int(obs.size),
            accuracy=(exact + none) / obs.size,
            exact_over_obs=(exact / nobs if nobs else None),
            exact_over_pred=(exact / npred if npred else None),
            source=src,
        ))
    return out


def resync(cfg):
    """Re-synchronisation to the baseline: 20 identical consecutive tokens
    (anywhere after the flip) or a 20-token identical suffix."""
    if not cfg.get("free") or not os.path.exists(cfg["free"]):
        return dict(resynced=0, flipped=0, longest_post_flip_run=0,
                    note="no free-run file")
    f = np.load(cfg["free"], allow_pickle=True)
    base = np.load(cfg["a1"], allow_pickle=True)["gen_ids"].astype(np.int64)
    g = f["gen_ids"]
    obs = f["obs"].astype(np.int64)
    res = 0
    flipped = 0
    best = 0
    detail = []
    for i in range(obs.shape[0]):
        for u in range(obs.shape[1]):
            o = int(obs[i, u])
            if o < 0:
                continue
            flipped += 1
            row = np.asarray(g[i, u]).astype(np.int64)
            start = min(o + 1, row.size)
            # longest identical run after the flip
            run = 0
            cur = 0
            for k in range(start, row.size):
                if row[k] == base[k]:
                    cur += 1
                    run = max(run, cur)
                else:
                    cur = 0
            # identical suffix
            suf = 0
            for k in range(row.size - 1, -1, -1):
                if row[k] == base[k]:
                    suf += 1
                else:
                    break
            best = max(best, run)
            if run >= 20 or suf >= 20:
                res += 1
                detail.append(dict(rho=float(f["radii"][i]), dir=int(u),
                                   flip=int(o), longest_run=int(run), suffix=int(suf)))
    return dict(resynced=res, flipped=flipped, longest_post_flip_run=int(best),
                detail=detail,
                note="2/4 for phi-4 are one-token substitutions that re-join "
                     "the baseline at o+1; all other models 0")


def angle_stats(path):
    d = np.load(path, allow_pickle=True)
    if "h0_last" not in d.files or "dh_last" not in d.files:
        return None
    h = d["h0_last"].astype(np.float64)
    dh = d["dh_last"].astype(np.float64)
    dh = dh[0] if dh.ndim == 4 else dh
    hn = np.linalg.norm(h, axis=1)[None, None, :]
    cos = (dh * h[None, None, :, :]).sum(-1) / (
        np.linalg.norm(dh, axis=-1) * hn + 1e-300)
    a = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return dict(n=int(a.size), median=float(np.median(a)), mean=float(a.mean()),
                percentiles={f"p{p}": float(np.percentile(a, p))
                             for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
                frac_gt_90=float(np.mean(a > 90)),
                frac_within_87_pm_5=float(np.mean(np.abs(a - 87) < 5)),
                dh_over_h_median=float(np.median(np.linalg.norm(dh, axis=-1) / hn[0, 0])),
                values=a)


def depth_stats(path, norms_path):
    d = np.load(path, allow_pickle=True)
    tokens = d["tokens"].astype(int).tolist()
    h = d["h_depth"].astype(float)
    lam = d["lam"].astype(float)
    A = np.load(norms_path, allow_pickle=True)
    nrm = np.median(A["norm_h0"].astype(float), axis=1)
    alayers = A["layers"].astype(int).tolist()
    # relative exponent: tangent log growth per block minus state-norm growth
    # per block, on the saved-state intervals (same convention as committed
    # sections; the last block has no saved output state).
    rel = []
    for ti in range(len(tokens)):
        total_t = float(np.sum(np.mean(lam[ti], axis=1)))
        # state norms only at saved states; use log-linear per-block rate
        g_t = total_t / lam.shape[1]
        ln0, ln1 = np.log(nrm[0]), np.log(nrm[-1])
        # saved span in block units
        span = max(alayers[-1] - alayers[0], 1)
        g_n = (ln1 - ln0) / span
        rel.append(float(g_t - g_n))
    return dict(tokens=tokens, h_depth=h.tolist(),
                h_depth_q=qs(h), n=int(h.size),
                relative_overall_by_token=rel,
                final_lambda_median=np.median(lam[:, -1, :], axis=1).tolist(),
                n_blocks=int(lam.shape[1]), k=int(np.atleast_1d(d["k"])[0]),
                source=os.path.relpath(path, HERE))


# ------------------------------------------------------------------ compute
def compute():
    print("computing synthesis ...")
    out = dict(
        tag="syn", campaign_radius=R_RUN, units="absolute embedding units for r; nats for m",
        models={k: {kk: vv for kk, vv in CFG[k].items()
                    if kk not in ("a1", "free")} for k in MODELS},
        pending=[], fragility={}, stability={}, angles={}, depth={},
        nonnorm={}, dimension={}, floor={},
    )
    if "Qwen3-14B" not in CFG:
        out["pending"].append(dict(
            what="Qwen3-14B", status="pending",
            detail="results_big/qwen314/a1_inter_w0/a1.npz not present at run time; "
                   "the column is omitted from every panel"))

    stats = {}
    for k in MODELS:
        print("  core", k)
        st = core_stats(CFG[k]["a1"])
        stats[k] = st
        dec = decomposition(st)
        ctl = controlling(st)
        s_emb = float(np.median(st["d"]["norm_h0"][0].astype(float)))
        m5 = np.argsort(st["m"])[:5]
        out["fragility"][k] = dict(
            K=int(st["rstar"].size), n_finite=int(st["good"].sum()),
            flip_fraction_at_run=float(np.mean(st["rstar"] < R_RUN)),
            decomposition=dec, controlling=ctl,
            s_emb=s_emb,
            rstar_over_s_emb=float(np.median(st["rstar"]) / s_emb),
            margins=qs(st["m"]),
            margins_below_0p1=int(np.sum(st["m"] < 0.1)),
            five_smallest_margins=[dict(t=int(t), m=float(st["m"][t])) for t in m5],
        )

    # relative decomposition against the 1.5B (exact at geometric-mean level)
    ref = out["fragility"]["1.5B"]["decomposition"]["geometric_mean"]["log10"]
    for k in MODELS:
        g = out["fragility"][k]["decomposition"]["geometric_mean"]["log10"]
        dm_ = g["mstar"] - ref["mstar"]
        ds_ = g["sstar"] - ref["sstar"]
        out["fragility"][k]["decomposition"]["vs_1p5B"] = dict(
            delta_log10_rstar=g["rstar"] - ref["rstar"],
            from_margin=dm_, from_sensitivity=-ds_,
            check=float((g["rstar"] - ref["rstar"]) - (dm_ - ds_)),
            margin_factor=float(10 ** dm_), sensitivity_factor=float(10 ** ds_),
            rstar_factor=float(10 ** (g["rstar"] - ref["rstar"])),
        )
        for kind in ("median_direction", "extreme_direction"):
            md = out["fragility"][k]["decomposition"][kind]
            r0 = out["fragility"]["1.5B"]["decomposition"][kind]
            out["fragility"][k]["decomposition"][kind + "_vs_1p5B"] = dict(
                delta_log10_rstar=float(np.log10(md["rstar"]) - np.log10(r0["rstar"])),
                from_margin=float(np.log10(md["mstar"]) - np.log10(r0["mstar"])),
                from_sensitivity=float(-(np.log10(md["sstar"]) - np.log10(r0["sstar"]))),
                mstar=md["mstar"], sstar=md["sstar"], rstar=md["rstar"],
                tstar=md["tstar"],
            )

    # ---------------- stability
    print("  stability")
    RAD = {
        "1.5B": [0.1, 0.3, 1.0],
        "DS-14B": [0.1, 0.3],
        "Qwen3.5-9B": [0.05, 0.1, 0.3],
        "phi-4": [0.1, 0.3],
        "Qwen3-14B": [0.1, 0.3],
    }
    for k in MODELS:
        radii = RAD.get(k, [R_RUN])
        # only radii actually present in the free run or the A1 radius ladder
        radii = [r for r in radii
                 if (CFG[k].get("free") and os.path.exists(CFG[k]["free"]) and
                     np.any(np.isclose(np.load(CFG[k]["free"], allow_pickle=True)["radii"], r)))
                 or np.any(np.isclose(stats[k]["radii"], r))]
        rows = free_scorecard(CFG[k], stats[k], radii)
        out["stability"][k] = dict(rows=rows, resync=resync(CFG[k]))
    out["stability"]["1.5B"]["free_run_note"] = (
        "the 1.5B free run tested only r<=0.01 (20/20 rows non-flipped at all "
        "three radii); the r=0.3 and r=1 rows are teacher-forced A1 passes, "
        "so they are decision flips, not free continuations")

    # ---------------- angles
    print("  angles")
    for k in MODELS:
        a = angle_stats(CFG[k]["a1"])
        if a is not None:
            out["angles"][k] = dict(
                median=a["median"], mean=a["mean"], percentiles=a["percentiles"],
                frac_gt_90=a["frac_gt_90"],
                frac_within_87_pm_5=a["frac_within_87_pm_5"],
                dh_over_h_median=a["dh_over_h_median"], n=a["n"],
                values=a["values"],
                source=CFG[k]["source"], kind="measured (h0_last, dh_last)")
        else:
            out["angles"][k] = dict(
                median=None, kind="not measurable from this run",
                reason="no h0_last vectors stored")
    out["angles"]["1.5B"] = dict(
        median=87.0, kind="literature value, not measured here",
        reason="the 1.5B A1 dump stores state norms and cos(h',h0), not h0_last; "
               "the T4 analysis quoted ~87 deg. The 14B full-attention run has "
               "the same limitation; the true-window-100 run does have h0_last.",
        n=None, source="section_cross_scale/T4 reference; not re-measured")

    # ---------------- depth
    print("  depth")
    ref15 = json.load(open(os.path.join(FIGDIR, "ds14sd_numbers.json"),
                           encoding="utf-8"))["reference_1p5B"]
    out["depth"]["1.5B"] = dict(
        h_depth_range=[ref15["depth_h_low"], ref15["depth_h_high"]],
        absolute_exponent_range=[ref15["depth_lam_low"], ref15["depth_lam_high"]],
        relative_worst=ref15["rel_worst"],
        verdict="absolute expansion, normalized every block contracts (worst -0.034)",
        kind="reference ranges (no per-token dump in this worktree)",
        source="ds14sd_numbers.json reference_1p5B")
    for k, dpath in [("DS-14B", os.path.join(HERE, "results_big/ds14/depth_w100/depth.npz")),
                     ("phi-4", os.path.join(HERE, "results_big/phi4/depth_w0/depth.npz"))]:
        if os.path.exists(dpath):
            ref = json.load(open(os.path.join(
                FIGDIR, "ds14sd_numbers.json" if k == "DS-14B" else "phi4_numbers.json"),
                encoding="utf-8"))
            dd = depth_stats(dpath, CFG[k]["a1"])
            if k == "DS-14B":
                dd["relative_overall_by_token"] = ref["depth"]["relative_overall_by_token"]
                dd["n_negative_cells"] = ref["depth"]["n_contract_cells"]
                dd["n_cells"] = ref["depth"]["n_cells"]
                dd["verdict"] = "61/64 six-block chunks contract"
            else:
                dd["relative_overall_by_token"] = ref["depth"]["relative_overall_by_token"]
                dd["n_negative_cells"] = ref["depth"]["relative_n_negative"]
                dd["n_cells"] = ref["depth"]["relative_n_cells"]
                dd["verdict"] = "93/96 one-to-four-block intervals contract"
            out["depth"][k] = dd
        else:
            out["depth"][k] = dict(kind="pending", reason="depth file absent")
    for k in ("Qwen3.5-9B", "Qwen3-14B"):
        if k in CFG:
            out["depth"][k] = dict(
                kind="pending", reason="no depth/JVP run in results_big/%s"
                % ("qwen35" if k == "Qwen3.5-9B" else "qwen314"),
                note="the layer profile measures the response, not the tangent "
                     "contraction; it is not substituted here")

    # ---------------- non-normality
    print("  non-normality")
    nn = json.load(open(os.path.join(FIGDIR, "ds14sd_numbers.json"),
                        encoding="utf-8"))
    blocks = nn["nonnorm"]["full_layers"]
    smax = np.array(nn["nonnorm"]["full_smax"], float)
    kap = np.array(nn["nonnorm"]["full_kappa"], float)
    out["nonnorm"]["DS-14B"] = dict(
        blocks=blocks,
        sigma_max_median=np.median(smax, axis=0).tolist(),
        sigma_max_min=smax.min(axis=0).tolist(),
        sigma_max_max=smax.max(axis=0).tolist(),
        kappa_median=np.median(kap, axis=0).tolist(),
        kappa_min=kap.min(axis=0).tolist(),
        kappa_max=kap.max(axis=0).tolist(),
        mlp_smax_median=np.median(np.array(nn["nonnorm"]["mlp_smax"]), axis=0).tolist(),
        mlp_layers=nn["nonnorm"]["mlp_layers"],
        tokens=nn["nonnorm"]["tokens"],
        reference_1p5B=nn["reference_1p5B"],
        source="results_big/ds14/nonnorm_w100/nonnorm.npz")
    for k in ("phi-4", "Qwen3-14B", "Qwen3.5-9B"):
        if k in CFG:
            out["nonnorm"][k] = dict(
                kind="pending",
                reason=("phi-4 fused-MLP Jacobian fix still generating"
                        if k == "phi-4" else "no non-normality run"))

    # ---------------- dimension
    print("  dimension")
    d2 = {
        "1.5B": os.path.join(FIGDIR, "d2pipe_validation.json"),
        "DS-14B": os.path.join(FIGDIR, "d2pipe_ds14d2.json"),
        "Qwen3.5-9B": os.path.join(FIGDIR, "d2pipe_q35d2.json"),
    }
    for k, p in d2.items():
        if not os.path.exists(p):
            continue
        D = json.load(open(p, encoding="utf-8"))
        ts = D["trajectories"]
        tw = [t["twonn"]["mle"] for t in ts]
        mle = [t["mle"]["10"]["d2"] for t in ts]
        t4 = [t["t4_style"]["d2"] for t in ts]
        prom = [t["period"]["period_prominence"] for t in ts]
        cov = [t["period"]["coverage"] for t in ts]
        out["dimension"][k] = dict(
            trajectories=[t["name"] for t in ts], n=len(ts),
            twonn=tw, twonn_range=[float(min(tw)), float(max(tw))],
            twonn_median=float(np.median(tw)),
            mle10=mle, mle10_range=[float(min(mle)), float(max(mle))],
            t4_chord=t4, t4_range=[float(min(t4)), float(max(t4))],
            loop_prominence=prom, loop_coverage=cov,
            valid_windows=int(sum(bool(t["theiler"]["0"]["valid"]) for t in ts)),
            valid_window_verdict="none (%d/%d pass the 0.40-decade rule)"
                                 % (sum(bool(t["theiler"]["0"]["valid"]) for t in ts), len(ts)),
            source=os.path.basename(p))
    # the 1.5B file includes two loop runs; flag them and separate the spurious
    # window on the period-53 loop from the healthy-run verdict
    if "1.5B" in out["dimension"]:
        D = json.load(open(d2["1.5B"], encoding="utf-8"))
        loops = [t["name"] for t in D["trajectories"]
                 if t["period"]["coverage"] > 0.5]
        healthy = [t for t in D["trajectories"] if t["period"]["coverage"] <= 0.5]
        out["dimension"]["1.5B"]["loop_runs"] = loops
        out["dimension"]["1.5B"]["healthy_runs"] = [t["name"] for t in healthy]
        out["dimension"]["1.5B"]["healthy_valid_windows"] = int(
            sum(bool(t["theiler"]["0"]["valid"]) for t in healthy))
        out["dimension"]["1.5B"]["valid_window_verdict"] = (
            "none on the two healthy runs (0/2); the single passing flag in the "
            "file is a spurious 0.54-decade window on the period-53 loop")
        out["dimension"]["1.5B"]["healthy_mle10_range"] = [
            float(min(t["mle"]["10"]["d2"] for t in healthy)),
            float(max(t["mle"]["10"]["d2"] for t in healthy))]
    out["dimension"]["DS-14B"]["loop_counts"] = dict(full=0, partial=0, n=13)
    out["dimension"]["Qwen3.5-9B"]["loop_counts"] = dict(full=0, partial=1, n=12)
    out["dimension"]["1.5B"]["loop_counts"] = dict(
        full=9, partial=2, n=20, equal_horizon=dict(full=6, partial=2, n=16),
        source="section_cross_scale.md / section_d2_ds14.md")
    if "phi-4" in CFG:
        out["dimension"]["phi-4"] = dict(
            kind="pending", reason="D2 distance matrices not computed in results_big/phi4")

    # ---------------- floor
    print("  floor")
    fl = json.load(open(os.path.join(FIGDIR, "ds14floor_numbers.json"),
                        encoding="utf-8"))
    layers = fl["protocol"]["layers"]
    F = fl["floor"]["F"]
    norms_full = fl["depth_law"]["norms"]
    A = np.load(CFG["DS-14B"]["a1"], allow_pickle=True)
    nrm_w = {int(L): float(np.median(A["norm_h0"][i].astype(float)))
             for i, L in enumerate(A["layers"].astype(int))}
    rows = []
    for j, L in enumerate(layers):
        nw = nrm_w[L]
        rows.append(dict(layer=int(L), F=float(F[j]),
                         norm_fullattn=float(norms_full[j]),
                         norm_w100=float(nw),
                         frac_fullattn=float(F[j] / norms_full[j]) if norms_full[j] else None,
                         frac_w100=float(F[j] / nw) if nw else None,
                         ulp_fullattn=float(F[j] / norms_full[j] / EPS32) if norms_full[j] else None,
                         ulp_w100=float(F[j] / nw / EPS32) if nw else None))
    out["floor"]["DS-14B"] = dict(rows=rows, definition=fl["floor"]["definition"],
                                  kappa_established=fl["depth_law"]["fits"]["prop_n"]["kappa"],
                                  note="the established normalization used the "
                                       "full-attention norms; the matched true-window "
                                       "norms give frac_w100, up to 16% lower (the "
                                       "norm itself differs by up to 19%), same law")
    for k, cal, a1 in [("Qwen3.5-9B", "results_big/qwen35/calib/calib.npz",
                        "results_big/qwen35/a1_inter_w0/a1.npz"),
                       ("phi-4", "results_big/phi4/calib/calib.npz",
                        "results_big/phi4/a1_inter_w0/a1.npz")]:
        Cp = os.path.join(HERE, cal)
        Ap = os.path.join(HERE, a1)
        C = np.load(Cp, allow_pickle=True)
        A2 = np.load(Ap, allow_pickle=True)
        rmin = float(C["radii"].min())
        i0 = int(np.argmin(C["radii"]))
        Fk = np.median(C["dh_norm"][i0], axis=(0, 2))
        nrm = {int(L): float(np.median(A2["norm_h0"][i].astype(float)))
               for i, L in enumerate(A2["layers"].astype(int))}
        rr = []
        for j, L in enumerate(C["layers"].astype(int)):
            if L not in nrm:
                continue
            rr.append(dict(layer=int(L), F=float(Fk[j]), norm=float(nrm[L]),
                           frac=float(Fk[j] / nrm[L]) if nrm[L] else None,
                           ulp=float(Fk[j] / nrm[L] / EPS32) if nrm[L] else None))
        out["floor"][k] = dict(rows=rr, r_floor=rmin,
                               definition="median ||dh_l|| over dirs,t at the smallest tested radius",
                               note=("phi-4's smallest tested radius is 1e-3, and "
                                     "the 3e-3 value is 22% higher, so F is an upper "
                                     "bound on the true round-off floor"
                                     if k == "phi-4" else
                                     "the 3e-4/1e-4 ratio is 1.05, so F is at the plateau"))
    x = json.load(open(os.path.join(FIGDIR, "xscale_numbers.json"), encoding="utf-8"))
    out["floor"]["1.5B"] = dict(
        rows=[dict(layer=27, F=float(x["models"]["1.5B"]["fp32_floor_last"]),
                   norm=float(x["models"]["1.5B"]["s_last_median"]),
                   frac=float(x["models"]["1.5B"]["gain_floor_rel"]),
                   ulp=float(x["models"]["1.5B"]["gain_floor_rel"] / EPS32))],
        note="last saved state only; gain_floor_rel = floor / s_last",
        source="xscale_numbers.json")

    # ---------------- headline table
    head = []
    for k in MODELS:
        f = out["fragility"][k]
        p = f["decomposition"]["percentiles"]
        md = f["decomposition"]["median_direction"]
        head.append(dict(
            model=k, blocks=CFG[k]["blocks"], hidden=CFG[k]["hidden"], K=CFG[k]["K"],
            rstar_p50=p["rstar"]["p50"], mstar_p50=p["mstar"]["p50"],
            sstar_p50=p["sstar"]["p50"], mstar_meddir=md["mstar"],
            sstar_meddir=md["sstar"], rstar_meddir=md["rstar"],
            s_emb=f["s_emb"], rstar_over_s_emb=f["rstar_over_s_emb"],
            flip_at_0p1=f["flip_fraction_at_run"],
            angle_deg=out["angles"].get(k, {}).get("median"),
            twonn_median=out["dimension"].get(k, {}).get("twonn_median"),
            loops=out["dimension"].get(k, {}).get("loop_counts"),
            resync=out["stability"][k]["resync"],
        ))
    out["headline"] = head
    return out, stats


# ------------------------------------------------------------------ figure 1
def fig_fragility(out, stats):
    fig = plt.figure(figsize=(14.2, 9.0))
    gs = GridSpec(2, 6, figure=fig, hspace=0.34, wspace=0.44,
                  left=0.075, right=0.985, top=0.93, bottom=0.09)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:6])
    axc = fig.add_subplot(gs[1, 0:3])
    axd = fig.add_subplot(gs[1, 3:6])

    # (a) ECDF of r*
    for k in MODELS:
        r = stats[k]["rstar"]
        r = r[np.isfinite(r)]
        x, y = ecdf(r)
        axa.step(x, y, where="post", color=C[k], lw=1.9 if k != "1.5B" else 2.4,
                 label=f"{k} (n={r.size})")
    axa.set_xscale("log")
    axa.set_xlim(1.5e-3, 40)
    axa.set_xlabel(r"flux-tube radius $r_*$  (embedding units)")
    axa.set_ylabel("ECDF over directions")
    axa.set_title("(a) fragility: $r_*=m_{t_*}/s_{t_*}$ per direction", loc="left")
    axa.axvline(R_RUN, color="#555555", ls=":", lw=1)
    axa.text(R_RUN * 1.12, 0.05, r"$r=0.1$", color="#555555", fontsize=8)
    axa.set_ylim(-0.02, 1.03)
    axa.legend(loc="upper left", fontsize=7.4)

    # (b) m* vs s*, iso-r* diagonals
    for k in MODELS:
        s = stats[k]
        good = s["good"]
        m = s["mstar"][good]
        ss = s["sstar"][good]
        if m.size > 450:
            sel = np.linspace(0, m.size - 1, 450).astype(int)
            m, ss = m[sel], ss[sel]
        axb.scatter(ss, m, s=5.5, alpha=0.20 if k != "1.5B" else 0.55,
                    color=C[k], linewidths=0, rasterized=True)
        md = out["fragility"][k]["decomposition"]["median_direction"]
        axb.scatter([md["sstar"]], [md["mstar"]], s=150, marker="*",
                    color=C[k], edgecolor="white", linewidth=0.7, zorder=6)
    # iso-r* lines
    sv = np.logspace(-4, 0.2, 50)
    for rr, lab in [(0.1, "0.1"), (1.0, "1"), (10.0, "10")]:
        axb.plot(sv, rr * sv, color="#777777", lw=0.7, ls="--", zorder=1)
        xt = 0.004
        axb.text(xt, rr * xt * 1.12, rf"$r_*={lab}$", color="#777777",
                 fontsize=7.4, ha="center", va="bottom",
                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(6e-4, 1.2)
    axb.set_ylim(6e-5, 0.62)
    axb.set_xlabel(r"controlling sensitivity $s_{t_*}=|\delta m_{t_*}|/r$  (nats per unit radius)")
    axb.set_ylabel(r"controlling margin $m_{t_*}$  (nats)")
    axb.set_title("(b) the margin axis orders the hierarchy; stars = median-$r_*$ direction",
                  loc="left")
    handles = []
    for k in MODELS:
        c = out["fragility"][k]["decomposition"]["corr"]["logm_logs"]
        handles.append(Line2D([], [], color=C[k], marker="o", ls="", ms=4,
                              label=f"{k}  (corr {c:+.2f})"))
    axb.legend(handles=handles, loc="upper left", ncol=2)

    # (c) s* marginal ECDF
    for k in MODELS:
        s = stats[k]["sstar"]
        s = s[np.isfinite(s)]
        x, y = ecdf(s)
        axc.step(x, y, where="post", color=C[k], lw=1.9)
    axc.set_xscale("log")
    axc.set_xlabel(r"local sensitivity at the controlling step, $s_{t_*}$")
    axc.set_ylabel("ECDF over directions")
    axc.set_title("(c) sensitivity alone does not order the models", loc="left")
    axc.set_ylim(-0.02, 1.03)
    txt = "\n".join(
        f"{k}: med {out['fragility'][k]['decomposition']['percentiles']['sstar']['p50']:.4g}"
        for k in MODELS)
    axc.text(0.03, 0.97, txt, transform=axc.transAxes, va="top", fontsize=7.8,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5, alpha=0.9))

    # (d) exact decomposition of the median r*, relative to the 1.5B
    y0 = np.arange(len(MODELS))
    h = 0.34
    for i, k in enumerate(MODELS):
        d = out["fragility"][k]["decomposition"]["median_direction_vs_1p5B"]
        axd.barh(i + h / 2, d["from_margin"], height=h, color="#3b6ea5", alpha=0.88)
        axd.barh(i - h / 2, d["from_sensitivity"], height=h, color="#e08a1e", alpha=0.88)
        axd.scatter([d["delta_log10_rstar"]], [i], s=90, marker="D",
                    color="black", zorder=5)
        axd.text(d["delta_log10_rstar"], i + 0.31,
                 f"net {d['delta_log10_rstar']:+.2f}", va="bottom", ha="center",
                 fontsize=7.8,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    axd.axvline(0, color="#333333", lw=0.9)
    axd.set_yticks(y0)
    axd.set_yticklabels(MODELS)
    axd.set_xlabel(r"$\Delta\log_{10}$ relative to the 1.5B  (dex)")
    axd.set_title(r"(d) exact split of median $r_*$: margin (blue) + sensitivity (orange) = net (diamond)",
                  loc="left")
    axd.set_xlim(-2.2, 1.35)
    axd.grid(axis="y", alpha=0.0)
    axd.text(0.99, 0.03, "margins span 55x across models;\nco-selected s spans 1.7x",
             transform=axd.transAxes, ha="right", fontsize=7.8,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5, alpha=0.9))

    fig.suptitle("Flux-tube radius decomposition: the controlling margin, not the response, "
                 "places each model", x=0.075, ha="left", fontsize=12.5, y=0.985)
    fig.savefig(os.path.join(FIGDIR, "syn_fragility.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_stability(out):
    fig = plt.figure(figsize=(15.2, 4.9))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30, left=0.06, right=0.985,
                  top=0.86, bottom=0.20)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    for k in MODELS:
        rows = out["stability"][k]["rows"]
        rr = [r["rho"] for r in rows]
        ob = [r["observed_fraction"] for r in rows]
        pr = [r["predicted_fraction"] for r in rows]
        axa.plot(rr, ob, "-o", color=C[k], ms=6.5, lw=1.6, label=k)
        axa.plot(rr, pr, "--s", color=C[k], ms=5.5, lw=1.1, alpha=0.80,
                 mfc="white")
    axa.set_xscale("log")
    axa.set_xlim(0.04, 1.25)
    axa.set_ylim(-0.02, 0.78)
    axa.set_xlabel(r"perturbation radius $\rho$")
    axa.set_ylabel("fraction of directions that flip")
    axa.set_title("(a) observed (filled) vs signed-rule prediction (open)", loc="left")
    axa.legend(ncol=2, loc="upper left", fontsize=7.6)

    # (b) horizontal bars: one row per model-radius
    labels = []
    rows_flat = []
    for k in MODELS:
        for row in out["stability"][k]["rows"]:
            labels.append(f"{k}  @{row['rho']:g}")
            rows_flat.append((k, row))
    yy = np.arange(len(rows_flat))[::-1]
    for y, (k, row) in zip(yy, rows_flat):
        obs = row["exact_over_obs"]
        pred = row["exact_over_pred"]
        if obs is None or row["observed"] == 0:
            axb.barh(y + 0.17, 0.0, height=0.34, color="#cccccc",
                     hatch="//", edgecolor="#999999")
            axb.text(0.02, y + 0.17, f"0 obs  (acc {row['accuracy']:.3f})",
                     va="center", fontsize=6.4, color="#666666")
        else:
            axb.barh(y + 0.17, obs, height=0.34, color=C[k], alpha=0.92)
            axb.text(obs + 0.015, y + 0.17, f"{obs:.2f}  (n={row['observed']})",
                     va="center", fontsize=6.6)
        if pred is None or row["predicted"] == 0:
            axb.barh(y - 0.17, 0.0, height=0.34, color="#cccccc",
                     hatch="..", edgecolor="#999999")
        else:
            axb.barh(y - 0.17, pred, height=0.34, color=C[k], alpha=0.42)
            axb.text(pred + 0.015, y - 0.17, f"{pred:.2f}", va="center",
                     fontsize=6.4)
    axb.set_yticks(yy)
    axb.set_yticklabels(labels, fontsize=7.0)
    axb.set_xlim(0, 1.28)
    axb.set_xlabel("exact step / observed (dark) or / predicted (light)")
    axb.set_title("(b) prediction accuracy, per model-radius", loc="left")
    axb.axvline(1.0, color="#999999", lw=0.8, ls=":")
    axb.grid(axis="y", alpha=0.0)

    # (c) resync verdict
    vals = []
    txt = []
    for k in MODELS:
        rs = out["stability"][k]["resync"]
        vals.append(rs["resynced"] / rs["flipped"] if rs["flipped"] else 0.0)
        txt.append(f"{rs['resynced']}/{rs['flipped']}")
    axc.bar(np.arange(len(MODELS)), vals, color=[C[k] for k in MODELS], alpha=0.9)
    for i, (v, t) in enumerate(zip(vals, txt)):
        axc.text(i, v + 0.02, t, ha="center", fontsize=8.5)
    axc.set_xticks(np.arange(len(MODELS)))
    axc.set_xticklabels(MODELS, fontsize=8)
    axc.set_ylim(0, 0.62)
    axc.set_ylabel("re-synchronised / flipped runs")
    axc.set_title("(c) re-synchronisation: only phi-4, 2 of 4", loc="left")
    axc.text(0.02, 0.96,
             "definition: 20 identical consecutive tokens\nor a 20-token identical suffix",
             ha="left", va="top", transform=axc.transAxes, fontsize=7.2,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5))
    axc.annotate("one-token substitutions\nthat re-join the baseline",
                 xy=(2.86, 0.48), xytext=(1.25, 0.30), fontsize=7.0,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))

    fig.suptitle("Stability scorecard: the signed linear rule predicts flip counts, "
                 "not late-flip identity", x=0.06, ha="left", fontsize=12.5, y=0.985)
    fig.savefig(os.path.join(FIGDIR, "syn_stability.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_angle_depth(out):
    fig = plt.figure(figsize=(15.0, 8.8))
    gs = GridSpec(2, 6, figure=fig, hspace=0.40, wspace=0.52,
                  left=0.065, right=0.985, top=0.925, bottom=0.085)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:4])
    axc = fig.add_subplot(gs[0, 4:6])
    axd = fig.add_subplot(gs[1, 1:3])
    axe = fig.add_subplot(gs[1, 3:5])

    # (a) angle distributions
    rng = np.random.default_rng(4)
    for i, k in enumerate(MODELS):
        a = out["angles"].get(k, {})
        vals = a.get("values")
        if vals is not None:
            v = np.asarray(vals).ravel()
            sel = rng.choice(v.size, size=min(1400, v.size), replace=False)
            axa.scatter(v[sel], i + rng.normal(0, 0.055, sel.size), s=2.2,
                        color=C[k], alpha=0.14, linewidths=0, rasterized=True)
            axa.scatter([a["median"]], [i], s=95, marker="D", color=C[k],
                        edgecolor="white", linewidth=0.7, zorder=5)
            axa.text(a["median"], i + 0.22, f"{a['median']:.2f}°", ha="center",
                     fontsize=8)
        else:
            axa.scatter([a.get("median", 87.0)], [i], s=110, marker="*",
                        facecolor="white", edgecolor=C[k], linewidth=1.3, zorder=5)
            axa.text(a.get("median", 87.0), i + 0.22,
                     f"{a.get('median', 87.0):.0f}° (lit.)", ha="center",
                     fontsize=7.6)
    axa.axvline(90, color="#666666", lw=0.9, ls=":")
    axa.axvline(87, color="#999999", lw=0.8, ls="--")
    axa.set_yticks(np.arange(len(MODELS)))
    axa.set_yticklabels(MODELS)
    axa.set_xlim(60, 120)
    axa.set_xlabel(r"angle between $\delta h$ and $h_0$  (degrees)")
    axa.set_title(r"(a) response-vs-state angle", loc="left")
    axa.grid(axis="y", alpha=0.0)
    axa.text(0.97, 0.03, "1.5B: literature value (no vectors)\nothers: 24 dirs x 300 steps",
             transform=axa.transAxes, va="bottom", ha="right", fontsize=7.0,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5))

    # (b) absolute h_depth per token
    if "1.5B" in out["depth"]:
        lo, hi = out["depth"]["1.5B"]["h_depth_range"]
        axb.axhspan(lo, hi, color="#1f77b4", alpha=0.10, zorder=0)
        axb.text(0.03, hi * 1.02, "1.5B band 0.6-1.4", color="#1f77b4",
                 fontsize=7.2, transform=axb.get_yaxis_transform())
    for k in ("DS-14B", "phi-4"):
        if k in out["depth"] and "tokens" in out["depth"][k]:
            d = out["depth"][k]
            axb.plot(d["tokens"], d["h_depth"], "-o", color=C[k], ms=5,
                     label=f"{k} ({d['n_blocks']} blocks)")
    axb.set_yscale("log")
    axb.set_xlabel("probe token $t$")
    axb.set_ylabel(r"$h_{\rm depth}=\sum_i\max(\lambda_i,0)$  (nats)")
    axb.set_title("(b) absolute tangent volume", loc="left")
    axb.legend(loc="upper right", fontsize=7.4)
    axb.text(0.98, 0.03, "Qwen3.5-9B / Qwen3-14B: no depth run",
             transform=axb.transAxes, fontsize=7.0, color="#666666", ha="right")

    # (c) state-normalized contraction rate
    if "1.5B" in out["depth"]:
        axc.axhline(out["depth"]["1.5B"]["relative_worst"], color="#1f77b4",
                    lw=1.1, ls="--")
        axc.text(0.02, out["depth"]["1.5B"]["relative_worst"] - 0.005,
                 "1.5B least-negative = -0.034", ha="left", va="top",
                 transform=axc.get_yaxis_transform(), fontsize=7.2,
                 color="#1f77b4")
    for k in ("DS-14B", "phi-4"):
        if k in out["depth"] and "relative_overall_by_token" in out["depth"][k]:
            d = out["depth"][k]
            axc.plot(d["tokens"], d["relative_overall_by_token"], "-o",
                     color=C[k], ms=5, label=k)
            off = 0.004 if k == "DS-14B" else -0.004
            axc.text(d["tokens"][-1] + 4,
                     d["relative_overall_by_token"][-1] + off,
                     k, color=C[k], fontsize=7.4, va="center")
    axc.axhline(0, color="#333333", lw=0.8)
    axc.set_xlabel("probe token $t$")
    axc.set_xlim(-5, 385)
    axc.set_ylabel("state-normalized tangent rate  (nats/block)")
    axc.set_title("(c) normalized: every model contracts", loc="left")

    # (d) sigma_max
    nn = out["nonnorm"]["DS-14B"]
    b = np.array(nn["blocks"])
    axd.fill_between(b, nn["sigma_max_min"], nn["sigma_max_max"],
                     color=C["DS-14B"], alpha=0.18, lw=0)
    axd.plot(b, nn["sigma_max_median"], "-o", color=C["DS-14B"], ms=5,
             label="DS-14B full block")
    ref = nn["reference_1p5B"]
    axd.errorbar([0], [np.mean(ref["nn_block0_sigma"])],
                 yerr=[[np.mean(ref["nn_block0_sigma"]) - ref["nn_block0_sigma"][0]],
                       [ref["nn_block0_sigma"][1] - np.mean(ref["nn_block0_sigma"])]],
                 fmt="s", mfc="white", mec="#1f77b4", ecolor="#1f77b4", ms=6,
                 capsize=2, label="1.5B block 0")
    axd.errorbar([12, 24, 36], [np.mean(ref["nn_mid_sigma"])] * 3,
                 yerr=[[np.mean(ref["nn_mid_sigma"]) - ref["nn_mid_sigma"][0]] * 3,
                       [ref["nn_mid_sigma"][1] - np.mean(ref["nn_mid_sigma"])] * 3],
                 fmt="s", mfc="white", mec="#1f77b4", ecolor="#1f77b4", ms=5,
                 capsize=2, alpha=0.7)
    axd.errorbar([47], [np.mean(ref["nn_last_sigma"])],
                 yerr=[[np.mean(ref["nn_last_sigma"]) - ref["nn_last_sigma"][0]],
                       [ref["nn_last_sigma"][1] - np.mean(ref["nn_last_sigma"])]],
                 fmt="s", mfc="white", mec="#1f77b4", ecolor="#1f77b4", ms=6,
                 capsize=2, label="1.5B reference band")
    axd.set_yscale("log")
    axd.set_xticks(b)
    axd.set_xlabel("block index")
    axd.set_ylabel(r"full-block $\sigma_{\max}$")
    axd.set_title(r"(d) non-normality: $\sigma_{\max}$ (phi-4 / Qwen3-14B pending)",
                  loc="left", fontsize=9.6)
    axd.legend(loc="upper right", fontsize=7.2)

    # (e) kappa
    axe.fill_between(b, nn["kappa_min"], nn["kappa_max"],
                     color=C["DS-14B"], alpha=0.18, lw=0)
    axe.plot(b, nn["kappa_median"], "-o", color=C["DS-14B"], ms=5)
    axe.set_xticks(b)
    axe.set_xlabel("block index")
    axe.set_ylabel(r"$\kappa=\sigma_{\max}/\rho$")
    axe.set_title(r"(e) non-normality: condition number $\kappa$", loc="left")
    axe.text(0.02, 0.96, "1.5B: block 0 5-8, mid 1.7-2.5, last 3-5.3",
             transform=axe.transAxes, va="top", fontsize=7.2,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5))

    fig.suptitle("Response geometry and depth: perpendicular responses, contracting "
                 "tangents, non-normal Jacobians", x=0.065, ha="left",
                 fontsize=12.5, y=0.985)
    fig.savefig(os.path.join(FIGDIR, "syn_angle_depth.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_dimension(out):
    fig = plt.figure(figsize=(15.2, 5.0))
    gs = GridSpec(1, 5, figure=fig, wspace=0.52, left=0.06, right=0.985,
                  top=0.84, bottom=0.18)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:3])
    axc = fig.add_subplot(gs[0, 3:5])

    # (a) TwoNN per trajectory
    rng = np.random.default_rng(2)
    for i, k in enumerate(MODELS):
        D = out["dimension"].get(k, {})
        if "twonn" not in D:
            axa.scatter([i], [8.5], marker="x", s=48, color="#999999")
            axa.text(i, 8.15, "no D2 run", ha="center", va="top", fontsize=7.2,
                     color="#666666")
            axa.text(i, 10.55, "pending", ha="center", fontsize=7.2,
                     color="#777777")
            continue
        tw = np.asarray(D["twonn"])
        axa.scatter(i + rng.normal(0, 0.07, tw.size), tw, s=22, color=C[k],
                    alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
        axa.plot([i - 0.22, i + 0.22], [D["twonn_median"]] * 2,
                 color=C[k], lw=2.2, zorder=4)
        axa.text(i + 0.26, D["twonn_median"], f"{D['twonn_median']:.2f}",
                 va="center", fontsize=7.6)
        axa.text(i, 10.55, "none", ha="center", fontsize=7.2, color="#b22222")
    axa.axhspan(5.44, 6.23, color="#1f77b4", alpha=0.08, zorder=0)
    axa.set_xticks(np.arange(len(MODELS)))
    axa.set_xticklabels(MODELS, fontsize=8)
    axa.set_ylim(3.8, 11.1)
    axa.set_ylabel(r"TwoNN dimension estimate $\hat d$")
    axa.set_title(r"(a) TwoNN per trajectory; valid $D_2$ window: none", loc="left")
    axa.text(0.99, 0.03, "grey band: 1.5B validation range 5.44-6.23",
             transform=axa.transAxes, ha="right", fontsize=7.0)

    # (b) loop counts
    full = []
    part = []
    ntot = []
    for k in MODELS:
        lc = out["dimension"].get(k, {}).get("loop_counts")
        if lc is None:
            full.append(0)
            part.append(0)
            ntot.append(0)
        else:
            full.append(lc["full"])
            part.append(lc.get("partial", 0))
            ntot.append(lc["n"])
    clean = [n - f - p for n, f, p in zip(ntot, full, part)]
    xx = np.arange(len(MODELS))
    axb.bar(xx, clean, color="#d9d9d9", label="clean")
    axb.bar(xx, full, bottom=clean, color="#c0392b", label="full loop")
    axb.bar(xx, part, bottom=np.array(clean) + np.array(full),
            color="#e08a1e", label="partial")
    for i, (n, f, p) in enumerate(zip(ntot, full, part)):
        if n == 0:
            axb.text(i, 1.0, "pending", rotation=90, ha="center", fontsize=7,
                     color="#666666")
        else:
            axb.text(i, n + 0.5, f"{f}+{p}/{n}", ha="center", fontsize=7.6)
    axb.set_xticks(xx)
    axb.set_xticklabels(MODELS, rotation=60, ha="right", fontsize=7.2)
    axb.set_ylabel("trajectories")
    axb.set_ylim(0, 22.6)
    axb.set_title("(b) repetition loops (full + partial)", loc="left")
    axb.legend(loc="upper right", fontsize=7.0)

    # (c) fp32 floor fraction of the state norm
    for k in ("DS-14B", "Qwen3.5-9B", "phi-4"):
        D = out["floor"].get(k, {})
        rows = D.get("rows", [])
        ukey = "ulp_w100" if k == "DS-14B" else "ulp"
        rows = [r for r in rows if r.get(ukey) is not None and r.get("layer") != 0]
        if not rows:
            continue
        L = [r["layer"] for r in rows]
        u = [r[ukey] for r in rows]
        axc.plot(L, u, "-o", color=C[k], ms=6 if k != "phi-4" else 7,
                 mfc="white" if k == "phi-4" else C[k], label=k)
    D = out["floor"].get("1.5B", {})
    if D.get("rows"):
        axc.scatter([27], [D["rows"][0]["ulp"]], marker="*", s=130,
                    color=C["1.5B"], edgecolor="white", zorder=5,
                    label="1.5B (last state)")
    axc.axhline(1, color="#999999", lw=0.8, ls=":")
    for m in (10, 20, 30):
        axc.axhline(m, color="#bbbbbb", lw=0.7, ls="--")
        axc.text(0.015, m * 1.03, f"{m} ulp", fontsize=6.8, color="#777777",
                 ha="left", transform=axc.get_yaxis_transform())
    axc.set_yscale("log")
    axc.set_xticks([8, 16, 24, 32, 40])
    axc.set_xlabel("block index")
    axc.set_ylabel(r"$F_l / \|h_l\|$  (fp32 ulp units)")
    axc.set_title("(c) round-off floor: a fixed fraction of the state norm", loc="left")
    axc.set_ylim(5, 60)
    axc.legend(loc="lower right", ncol=2, fontsize=7.2)
    axc.text(0.02, 0.97,
             "14B: 18-24 ulp (w100 norms)\nQ35: 12-26 ulp; phi-4: 21-29 ulp",
             transform=axc.transAxes, va="top", fontsize=7.2,
             bbox=dict(fc="white", ec="#cccccc", lw=0.5))

    fig.suptitle("Dimension and decay: no scaling window, no full loops above the 1.5B, "
                 "and a universal round-off floor", x=0.06, ha="left",
                 fontsize=12.5, y=0.97)
    fig.savefig(os.path.join(FIGDIR, "syn_dimension.png"), dpi=170,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    out, stats = compute()
    print("making figures ...")
    fig_fragility(out, stats)
    fig_stability(out)
    fig_angle_depth(out)
    fig_dimension(out)
    # strip the raw angle vectors before writing the JSON
    for k, a in out["angles"].items():
        a.pop("values", None)
    with open(OUTJSON, "w", encoding="utf-8") as f:
        json.dump(jf(out), f, indent=1)
    print("wrote", OUTJSON)
    print("\nHEADLINE")
    for r in out["headline"]:
        print("  %-11s r*=%.4g m*=%.4g s*=%.4g | med-dir m=%.4g s=%.4g "
              "r=%.4g | flip@0.1=%.3f | angle=%s | TwoNN=%s"
              % (r["model"], r["rstar_p50"], r["mstar_p50"], r["sstar_p50"],
                 r["mstar_meddir"], r["sstar_meddir"], r["rstar_meddir"],
                 r["flip_at_0p1"], r["angle_deg"], r["twonn_median"]))


if __name__ == "__main__":
    main()
