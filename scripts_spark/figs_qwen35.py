#!/usr/bin/env python
"""Qwen3.5-9B core analysis: margins, thresholds, hybrid-attention layer profile.

Reads (local copies pulled from the Spark):
    results_big/qwen35/calib/calib.npz          8 dirs x 9 radii, 400 tokens
    results_big/qwen35/a1_inter_w0/a1.npz       1000 dirs, r=0.1, T=300, 32 states
    results_big/qwen35/free_inter_w0/free_run.npz
    results_big/qwen35/free_inter_w0/summary.json
    assets/tok_qwen35/tokenizer.json            (only for reading the tie words)

Writes
    notebook_big/figures/q35_calib.png
    notebook_big/figures/q35_layer_profile.png
    notebook_big/figures/q35_margins_rstar.png
    notebook_big/figures/q35_free_validation.png
    notebook_big/figures/q35_gain_structure.png
    notebook_big/figures/q35_numbers.json

Index conventions (checked against src/chaos_llm/big_engine.py:Capture):
    state 0       = embedding output
    state l >= 1  = output of BLOCK l-1
    therefore block b is visible as state b+1, and the saved array 0..31
    covers blocks 0..30.  The last block (31, full attention) was not saved:
    its output would be state 32.  Its post-final-norm response is available
    for 24 directions through dh_last (final-norm output = lm_head input).
    Full-attention BLOCKS are 3,7,11,15,19,23,27,31; among the saved states
    their outputs are states 4,8,12,16,20,24,28 (block 31 -> state 32, absent).

Run from the repository root:
    python scripts_spark/figs_qwen35.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "notebook_big", "figures")
OUTJSON = os.path.join(FIGDIR, "q35_numbers.json")

CALIB = os.path.join(HERE, "results_big", "qwen35", "calib", "calib.npz")
A1 = os.path.join(HERE, "results_big", "qwen35", "a1_inter_w0", "a1.npz")
FREE = os.path.join(HERE, "results_big", "qwen35", "free_inter_w0", "free_run.npz")
SUMMARY = os.path.join(HERE, "results_big", "qwen35", "free_inter_w0", "summary.json")
TOK = os.path.join(HERE, "assets", "tok_qwen35", "tokenizer.json")

R_RUN = 0.1                      # campaign radius of the A1 ensemble
T_ALL = 300                      # A1 generated horizon
T_CAL = 400                      # calibration horizon
TWIN = (100, 400)                # calibration averaging window, as in Part 0
CLEAN = (1e-2, 0.3)              # measured linear window (see calib panel)
FULL_BLOCKS = [3, 7, 11, 15, 19, 23, 27, 31]
FULL_SAVED = [b for b in FULL_BLOCKS if b <= 30]      # visible in the npz
RHOS = (0.05, 0.1, 0.3)

C_FULL = "#c0392b"
C_LIN = "#2c7fb8"
C_OBS = "#111111"
C_PRED = "#d95f02"
C_ABS = "#7b4fa0"
C_GREY = "#8a8a8a"

COL = {
    "exact": "#2e7d32",
    "wrong_step": "#e08a1e",
    "false_alarm": "#c0392b",
    "miss": "#7b4fa0",
    "both_none": "#d4d4d4",
}

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
    "legend.fontsize": 8.4,
})


# ------------------------------------------------------------------ helpers
def jf(o):
    """JSON-safe recursive conversion."""
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    return o


def qsummary(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    x = np.asarray(x, float)
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} | {
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "n": int(x.size)}


def ecdf(x):
    xs = np.sort(np.asarray(x, float))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def dec(ids):
    if TOK_DECODE is None:
        return " ".join(str(int(i)) for i in np.atleast_1d(ids))
    s = TOK_DECODE([int(i) for i in np.atleast_1d(ids)])
    return s.replace("\n", "\\n")


TOK_DECODE = None
if os.path.exists(TOK):
    try:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(TOK)
        TOK_DECODE = _tok.decode
    except Exception as exc:            # pragma: no cover - optional dependency
        print("tokenizer unavailable:", exc)


# ------------------------------------------------------------------ data
def load():
    z = np.load(CALIB)
    a = np.load(A1)
    f = np.load(FREE, allow_pickle=True)
    summ = json.load(open(SUMMARY))

    D = {}
    # ---------------- calibration
    D["cal_radii"] = z["radii"].astype(float)
    D["cal_states"] = z["layers"].astype(int)
    D["cal_margins"] = z["margins"].astype(float)
    cal_dh = z["dh_norm"].astype(float)                    # (nr, ndir, nstate, T)
    D["cal_dh"] = np.median(cal_dh[:, :, :, TWIN[0]:TWIN[1]], axis=(1, 3))
    D["cal_gain"] = D["cal_dh"] / D["cal_radii"][:, None]

    # ---------------- A1
    D["m"] = a["margins"].astype(float)
    D["top1"] = a["top1"].astype(int)
    D["top2"] = a["top2"].astype(int)
    D["ref"] = a["gen_ids"].astype(np.int64)
    D["dm"] = a["dm"][0].astype(float)                     # (K,T)
    D["dh"] = a["dh_norm"][0].astype(float)                # (K,32,T)
    D["dh_last"] = a["dh_last"][0].astype(float)           # (24,T,D)
    D["h0_last"] = a["h0_last"].astype(float)              # (T,D)
    D["norm_h0"] = a["norm_h0"].astype(float)              # (32,T)
    D["K"] = D["dm"].shape[0]
    D["T"] = D["dm"].shape[1]

    # signed threshold radius: dm<0 destroys the margin; dm>=0 cannot flip
    D["rc"] = np.where(D["dm"] < 0,
                       D["m"][None, :] * R_RUN / np.maximum(-D["dm"], 1e-300),
                       np.inf)
    D["tstar"] = np.argmin(D["rc"], axis=1)
    D["rstar"] = D["rc"][np.arange(D["K"]), D["tstar"]]
    D["flip_run"] = D["rstar"] <= R_RUN

    # layer profile: median over directions at each (state, t)
    D["prof"] = np.median(D["dh"], axis=0) / R_RUN        # (32,T)
    D["prof_med_t"] = np.median(D["prof"], axis=1)        # (32,)
    D["norm_med"] = np.median(D["norm_h0"], axis=1)       # (32,)

    # gain at the last saved state (block 30), all 1000 directions
    D["G"] = np.median(D["dh"][:, 31, :], axis=1) / R_RUN

    # ---------------- free run
    D["obs"] = f["obs"].astype(int)
    D["fradii"] = f["radii"].astype(float)
    D["gen"] = np.asarray(f["gen_ids"], dtype=np.int64)   # (3,1000,300)
    D["agreement_stored"] = f["agreement"].astype(float)
    D["summ"] = summ

    D["pred_signed"] = {}
    D["pred_abs"] = {}
    for rho in D["fradii"]:
        hit = D["rc"] <= rho
        anyh = hit.any(axis=1)
        p = np.full(D["K"], -1, dtype=int)
        p[anyh] = hit.argmax(axis=1)[anyh]
        D["pred_signed"][float(rho)] = p
        rcabs = D["m"][None, :] * R_RUN / np.maximum(np.abs(D["dm"]), 1e-300)
        h2 = rcabs <= rho
        a2 = h2.any(axis=1)
        p2 = np.full(D["K"], -1, dtype=int)
        p2[a2] = h2.argmax(axis=1)[a2]
        D["pred_abs"][float(rho)] = p2

    return D


# ------------------------------------------------------------------ quantities
def compute(D):
    m = D["m"]
    Q = {}
    Q["margins"] = qsummary(m) | {
        "min_step": int(np.argmin(m)),
        "n_below_0p1": int((m < 0.1).sum()),
        "n_below_0p01": int((m < 0.01).sum()),
        "smallest5": [{"t": int(t), "m": float(m[t]),
                       "top1": dec([D["top1"][t]]),
                       "top2": dec([D["top2"][t]])}
                      for t in np.argsort(m)[:5]],
    }
    Q["rstar"] = qsummary(D["rstar"]) | {
        "min_step": int(D["tstar"][int(np.argmin(D["rstar"]))]),
        "flip_frac_at_0p1": float(D["flip_run"].mean()),
        "flip_frac_at_0p3": float((D["rstar"] <= 0.3).mean()),
        "n_above_1": int((D["rstar"] > 1).sum()),
    }
    # controlling-step ladder
    allc = np.bincount(D["tstar"], minlength=D["T"])
    flc = np.bincount(D["tstar"][D["flip_run"]], minlength=D["T"])
    order = np.argsort(allc)[::-1]
    Q["ladder"] = [{"t": int(t), "n_all": int(allc[t]), "n_flip": int(flc[t]),
                    "m": float(m[t]),
                    "top1": dec([D["top1"][t]]),
                    "top2": dec([D["top2"][t]])}
                   for t in order if allc[t] > 0]
    Q["ladder_n_distinct"] = int((allc > 0).sum())
    Q["ladder_n_distinct_flip"] = int((flc > 0).sum())

    # ---------------- calibration
    cm = D["cal_margins"]
    Q["cal_margins"] = qsummary(cm) | {
        "min_step": int(np.argmin(cm)),
        "n_below_0p1": int((cm < 0.1).sum()),
        "n_below_0p01": int((cm < 0.01).sum()),
        "smallest6": [{"t": int(t), "m": float(cm[t])}
                      for t in np.argsort(cm)[:6]],
    }
    Q["cal"] = {
        "radii": D["cal_radii"].tolist(),
        "states": D["cal_states"].tolist(),
        "median_dh": D["cal_dh"].tolist(),
        "gain_dh_over_r": D["cal_gain"].tolist(),
        "floor_last_state": float(D["cal_dh"][0, -1]),
        "floor_last_state_3e4": float(D["cal_dh"][1, -1]),
        "clean_window": list(CLEAN),
        "signal_over_floor_at_r0p1": float(D["cal_dh"][6, -1] / D["cal_dh"][0, -1]),
        "gain_plateau_last_state": float(np.median(D["cal_gain"][4:8, -1])),
        "gain_at_r1_last_state": float(D["cal_gain"][-1, -1]),
    }

    # ---------------- layer profile
    prof = D["prof"]
    pm = D["prof_med_t"]
    growth = pm[1:] / np.maximum(pm[:-1], 1e-30)          # growth[b] = through block b
    Q["layer"] = {
        "blocks_saved": list(range(0, 31)),
        "states_saved": list(range(1, 32)),
        "full_attention_blocks": FULL_BLOCKS,
        "full_attention_blocks_saved": FULL_SAVED,
        "median_over_t": pm.tolist(),
        "median_over_t_blocks0_30": pm[1:].tolist(),
        "growth_ratio_blocks1_30": growth[1:].tolist(),
        "total_growth_block0_to_30": float(pm[31] / pm[1]),
        "monotone_blocks": int(np.sum(np.diff(pm[1:]) > 0)),
        "n_block_intervals": 30,
        "full_growth_mean": float(growth[FULL_BLOCKS[:7]].mean()),
        "linear_growth_mean_blocks1_30": float(
            growth[[b for b in range(1, 31) if b not in FULL_BLOCKS]].mean()),
        "full_over_linear_growth": float(
            growth[FULL_BLOCKS[:7]].mean() /
            growth[[b for b in range(1, 31) if b not in FULL_BLOCKS]].mean()),
        "relative_response": (pm[1:] / D["norm_med"][1:]).tolist(),
        "late_over_early": [float(np.median(prof[l, 250:]) /
                                  np.median(prof[l, :50]))
                            for l in [1, 4, 8, 12, 16, 20, 24, 28, 31]],
        "prof_early_med": float(np.median(prof[1:, :50])),
        "prof_late_med": float(np.median(prof[1:, 250:])),
    }
    # permutation test for the 7 saved full-attention blocks, blocks 1..30
    rng = np.random.default_rng(0)
    blocks = np.arange(1, 31)
    glog = np.log(growth[blocks])
    obs = glog[np.searchsorted(blocks, FULL_BLOCKS[:7])].mean()
    draws = np.array([glog[np.searchsorted(blocks, rng.choice(blocks, 7, replace=False))].mean()
                      for _ in range(20000)])
    Q["layer"]["perm_p_full_blocks"] = float((draws >= obs).mean())
    Q["layer"]["perm_z_full_blocks"] = float((obs - draws.mean()) / draws.std())
    # same test after removing a linear depth trend
    X = np.vstack([np.ones(30), blocks]).T
    beta = np.linalg.lstsq(X, glog, rcond=None)[0]
    res = glog - X @ beta
    obs_r = res[np.searchsorted(blocks, FULL_BLOCKS[:7])].mean()
    draws_r = np.array([res[np.searchsorted(blocks, rng.choice(blocks, 7, replace=False))].mean()
                        for _ in range(20000)])
    Q["layer"]["perm_p_full_blocks_detrended"] = float((draws_r >= obs_r).mean())

    # ---------------- gain / pairwise / angle
    Q["gain"] = qsummary(D["G"]) | {"max_over_min": float(D["G"].max() / D["G"].min())}

    dhl = D["dh_last"]
    nvec, T, _ = dhl.shape
    nrm = np.linalg.norm(dhl, axis=2)
    tmed = np.argmin(np.abs(nrm - np.median(nrm, axis=1, keepdims=True)), axis=1)
    V = np.stack([dhl[i, tmed[i]] for i in range(nvec)])
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    C = Vn @ Vn.T
    iu = np.triu_indices(nvec, 1)
    pair = C[iu]
    sv = np.linalg.svd(Vn, compute_uv=False)
    pca = sv ** 2 / np.sum(sv ** 2)
    Q["pairwise"] = {
        "n_vectors": int(nvec),
        "mean_cos": float(pair.mean()),
        "median_cos": float(np.median(pair)),
        "frac_abs_cos_gt_0p1": float((np.abs(pair) > 0.1).mean()),
        "pca_first": pca[:8].tolist(),
        "isotropic_first": float(1.0 / nvec),
    }
    lag_cos = {}
    for lag in (1, 10, 50):
        cs = []
        for i in range(nvec):
            a = dhl[i, :-lag]
            b = dhl[i, lag:]
            cs.append(np.sum(a * b, 1) /
                      np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30))
        lag_cos[str(lag)] = float(np.median(np.concatenate(cs)))
    Q["pairwise"]["temporal_median_cos"] = lag_cos

    h0 = D["h0_last"]
    nh = np.linalg.norm(h0, axis=1)
    c_ang = np.einsum("utd,td->ut", dhl, h0) / (np.maximum(nrm, 1e-30) * nh[None, :])
    ang = np.degrees(np.arccos(np.clip(c_ang, -1, 1)))
    Q["angle"] = {
        "n_pairs": int(ang.size),
        "cos_percentiles": {f"p{p}": float(np.percentile(c_ang, p))
                            for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        "angle_percentiles": {f"p{p}": float(np.percentile(ang, p))
                              for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        "mean_cos": float(c_ang.mean()),
        "mean_angle": float(ang.mean()),
        "median_angle": float(np.median(ang)),
        "frac_angle_gt_90": float((ang > 90).mean()),
        "frac_within_87_pm_5": float((np.abs(ang - 87) < 5).mean()),
        "per_direction_median_angle": np.median(ang, axis=1).tolist(),
        "per_step_median_angle": np.median(ang, axis=0).tolist(),
        "dh_over_h0_median": float(np.median(nrm / nh[None, :])),
        "cos_hp_h0_min": float((np.einsum("utd,td->ut", h0[None] + dhl, h0)
                                / (np.linalg.norm(h0[None] + dhl, axis=2) * nh[None, :])).min()),
    }

    # ---------------- free validation
    free = {}
    for ri, rho in enumerate(D["fradii"]):
        rho = float(rho)
        o = D["obs"][ri]
        pred = D["pred_signed"][rho]
        fl = o >= 0
        exact = int(np.sum(fl & (pred >= 0) & (o == pred)))
        wrong = int(np.sum(fl & (pred >= 0) & (o != pred)))
        fa = int(np.sum((~fl) & (pred >= 0)))
        miss = int(np.sum(fl & (pred < 0)))
        none = int(np.sum((~fl) & (pred < 0)))
        allg = D["gen"][ri]
        ref = D["ref"]
        a_after, a_late, longest, resync = [], [], [], 0
        for u in np.where(fl)[0]:
            oo = int(o[u])
            post = allg[u, oo + 1:] == ref[oo + 1:]
            a_after.append(float(post.mean()) if len(post) else 0.0)
            late = allg[u, oo + 46:] == ref[oo + 46:]
            a_late.append(float(late.mean()) if len(late) else np.nan)
            best = cur = 0
            for v in post:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            longest.append(best)
            resync += int(best >= 20)
        free[str(rho)] = {
            "observed_flips": int(fl.sum()),
            "observed_flip_fraction": float(fl.mean()),
            "predicted_flips_signed": int((pred >= 0).sum()),
            "predicted_flip_fraction_signed": float((pred >= 0).mean()),
            "predicted_flips_signblind": int((D["pred_abs"][rho] >= 0).sum()),
            "predicted_flip_fraction_signblind": float((D["pred_abs"][rho] >= 0).mean()),
            "exact": exact, "wrong_step": wrong, "false_alarm": fa,
            "miss": miss, "both_none": none,
            "accuracy": (exact + none) / D["K"],
            "mean_flip_step_observed": float(o[fl].mean()) if fl.any() else None,
            "median_flip_step_observed": float(np.median(o[fl])) if fl.any() else None,
            "mean_flip_step_predicted": float(pred[pred >= 0].mean()) if (pred >= 0).any() else None,
            "A_after_mean": float(np.mean(a_after)) if a_after else None,
            "A_late_mean": float(np.nanmean(a_late)) if a_late else None,
            "longest_identical_run": int(max(longest)) if longest else 0,
            "resynchronised_ge20": int(resync),
            "flip_words": {},
        }
        fw = {}
        for u in np.where(fl)[0]:
            oo = int(o[u])
            key = (oo, int(allg[u, oo]))
            fw[key] = fw.get(key, 0) + 1
        free[str(rho)]["flip_words"] = {
            f"t{oo}:{dec([ref[oo]])}->{dec([tok])}": int(n)
            for (oo, tok), n in sorted(fw.items(), key=lambda kv: -kv[1])}
        # survival
        INF = 10 ** 9
        og = np.where(o < 0, INF, o)
        pg = np.where(pred < 0, INF, pred)
        so = np.array([np.mean(og > t) for t in range(T_ALL + 1)])
        sp = np.array([np.mean(pg > t) for t in range(T_ALL + 1)])
        free[str(rho)]["survival_max_abs_diff"] = float(np.max(np.abs(so - sp)))
        free[str(rho)]["survival_t_max_diff"] = int(np.argmax(np.abs(so - sp)))
        free[str(rho)]["S69_obs"] = float(np.mean(og > 69))
        free[str(rho)]["S69_pred"] = float(np.mean(pg > 69))
        free[str(rho)]["S267_obs"] = float(np.mean(og > 267))
        free[str(rho)]["S267_pred"] = float(np.mean(pg > 267))
    Q["free"] = free
    return Q


# ------------------------------------------------------------------ figure 1
def fig_calib(D, Q):
    R = D["cal_radii"]
    L = D["cal_states"]
    G = D["cal_gain"]
    med = D["cal_dh"]
    m = D["cal_margins"]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    ax = axes[0, 0]
    t = np.arange(len(m))
    ax.semilogy(t, m, color=C_OBS, lw=1.0)
    ax.axvline(T_ALL, color=C_GREY, ls=":", lw=1.0)
    ax.annotate("A1 horizon (300 tokens)", xy=(T_ALL, 0.955),
                xycoords=("data", "axes fraction"),
                xytext=(T_ALL + 8, 0.985), textcoords=("data", "axes fraction"),
                fontsize=7.2, color=C_GREY, va="top")
    for t0 in np.argsort(m)[:5]:
        ax.plot(t0, m[t0], "o", ms=4.5, color=C_FULL, zorder=5)
        off = (t0 + 9, m[t0] * 1.9)
        if t0 == 322:
            off = (t0 - 62, m[t0] * 2.9)
        if t0 == 327:
            off = (t0 + 8, m[t0] * 1.7)
        ax.annotate(f"t={t0}\n{m[t0]:.3g}", xy=(t0, m[t0]),
                    xytext=off, fontsize=7.1,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C_FULL))
    ax.set_xlabel("generated step $t$ (calibration run, 400 tokens)")
    ax.set_ylabel("$m_t=\\ell_t^{(1)}-\\ell_t^{(2)}$  (nats, log scale)")
    ax.set_xlim(-2, len(m) + 1)
    ax.set_title("(a) No razor tie: min 0.0069 nats, 75x the 14B's")

    ax = axes[0, 1]
    xs, ys = ecdf(m)
    ax.plot(xs, ys, color=C_OBS, lw=1.6)
    ax.set_xscale("log")
    for val, lab, col in [(m.min(), f"this run {m.min():.1e}", C_FULL),
                          (9.727e-5, "14B min 9.7e-5", C_PRED),
                          (2.016e-2, "1.5B min 2.0e-2", C_LIN)]:
        ax.axvline(val, color=col, ls="--", lw=1.2)
        ax.text(val * 1.06, 0.35, lab, rotation=90, fontsize=7.2,
                color=col, va="bottom")
    ax.set_xlabel("baseline top-1/top-2 logit gap $m_t$ (nats, log scale)")
    ax.set_ylabel("fraction of steps with margin $\\leq m$")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) Margins sit between 1.5B and 14B, far from the 14B's knife edge")

    ax = axes[1, 0]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(L)))
    for si, (st, c) in enumerate(zip(L, cols)):
        if st == 0:
            continue
        ax.loglog(R, med[:, si], "o-", ms=3.2, lw=1.2, color=c,
                  label=f"state {st} (block {st - 1})")
        # slope-1 guide through the r=0.1 point
        i0 = list(R).index(0.1)
        xx = np.array([2e-3, 2e-1])
        ax.loglog(xx, med[i0, si] * xx / 0.1, ls=":", lw=0.9, color=c, alpha=0.75)
    ax.axhline(Q["cal"]["floor_last_state"], color=C_GREY, ls="--", lw=1.0)
    ax.text(1.1e-4, Q["cal"]["floor_last_state"] * 1.18,
            f"fp32 floor {Q['cal']['floor_last_state']:.2e}",
            fontsize=7.6, color=C_GREY)
    ax.axvspan(*CLEAN, color=C_LIN, alpha=0.07)
    ax.axvline(R_RUN, color=C_FULL, ls=":", lw=1.0)
    ax.set_xlabel("perturbation radius $r$ (log scale)")
    ax.set_ylabel("median $\\|\\delta h^{\\mathrm{last}}\\|$  (log scale)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("(c) Linear in $r$ from 0.01 to 0.3; floor-bound below 3e-4")

    ax = axes[1, 1]
    for st, c in zip(L, cols):
        if st == 0:
            continue
        ax.loglog(R, G[:, list(L).index(st)], "o-", ms=3.2, lw=1.2, color=c,
                  label=f"state {st}")
    ax.axvspan(*CLEAN, color=C_LIN, alpha=0.07)
    ax.axhline(Q["cal"]["gain_plateau_last_state"], color=C_GREY, ls="--", lw=0.9)
    ax.axvline(R_RUN, color=C_FULL, ls=":", lw=1.0)
    ax.text(0.012, Q["cal"]["gain_plateau_last_state"] * 0.72,
            "plateau 0.50 nats/unit\n(last saved state)", fontsize=7.4, color=C_GREY)
    ax.set_xlabel("perturbation radius $r$ (log scale)")
    ax.set_ylabel("$\\|\\delta h^{\\mathrm{last}}\\|/r$  (log scale)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("(d) Flat gain 0.50; inflates below the floor and above $r=0.3$")

    fig.suptitle("Qwen3.5-9B calibration: no near-tie, a linear window "
                 "$r\\in[0.01,0.3]$, and a floor 4.4x below the 14B's",
                 fontsize=11.8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "q35_calib.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_layer(D, Q):
    prof = D["prof"]
    pm = D["prof_med_t"]
    blocks = np.arange(31)
    growth = pm[1:] / np.maximum(pm[:-1], 1e-30)
    B = np.arange(1, 31)
    Z = prof[1:]                                     # rows = blocks 0..30

    fig = plt.figure(figsize=(13.6, 8.4))
    gs = GridSpec(2, 2, width_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(np.log10(np.maximum(Z, 1e-8)), aspect="auto", origin="lower",
                   extent=[-0.5, D["T"] - 0.5, -0.5, 30.5],
                   cmap="magma", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("$\\log_{10}(\\mathrm{median}_u\\,\\|\\delta h_t^b\\|/r)$")
    for b in FULL_SAVED:
        ax.axhline(b, color="#39d3ff", lw=0.9, alpha=0.85)
    ax.set_yticks(list(range(0, 31, 2)))
    for lab, y in zip(ax.get_yticklabels(), range(0, 31, 2)):
        if y in FULL_SAVED:
            lab.set_color("#1c9fc4")
            lab.set_fontweight("bold")
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("block index $b$  (saved as state $b+1$)")
    ax.set_title("(a) Response by block and step: depth growth, time decay,\n"
                 "and no separate full-attention band (cyan rows)")

    ax = fig.add_subplot(gs[0, 1])
    lo = np.percentile(Z, 10, axis=1)
    hi = np.percentile(Z, 90, axis=1)
    ax.fill_between(blocks, lo, hi, color=C_LIN, alpha=0.16,
                    label="10-90% over $t$")
    ax.semilogy(blocks, pm[1:], "-", color=C_OBS, lw=1.7, label="median over $t$")
    ax.semilogy(FULL_SAVED, pm[np.array(FULL_SAVED) + 1], "o", ms=6.0,
                color=C_FULL, label="full-attention block")
    ax.semilogy([b for b in blocks if b not in FULL_SAVED],
                pm[[b + 1 for b in blocks if b not in FULL_SAVED]], "o", ms=3.6,
                color=C_LIN, alpha=0.75, label="linear-attention block")
    ax.set_xlabel("block index $b$  (state $b+1$)")
    ax.set_ylabel("median $\\|\\delta h_t^b\\|/r$  (log scale)")
    ax.set_xlim(-0.5, 31.5)
    ax.legend(loc="lower right", ncol=2, fontsize=7.4)
    ax.set_title(f"(b) Monotone {Q['layer']['total_growth_block0_to_30']:.0f}x growth "
                 "with depth; no block contracts")

    ax = fig.add_subplot(gs[1, 1])
    colr = [C_FULL if b in FULL_SAVED else C_LIN for b in B]
    ax.bar(B, growth[1:], color=colr, width=0.72)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(growth[1:].mean(), color=C_GREY, ls="--", lw=1.0,
               label=f"mean {growth[1:].mean():.2f}x")
    ax.legend(loc="upper right", fontsize=7.4)
    ax.set_xlabel("block index $b$  (growth from state $b$ to state $b+1$)")
    ax.set_ylabel("response ratio across one block")
    ax.set_xlim(0.3, 31.0)
    ax.set_title(f"(c) Full-attention blocks grow {Q['layer']['full_over_linear_growth']:.2f}x "
                 f"faster; permutation p={Q['layer']['perm_p_full_blocks']:.2f}")

    fig.suptitle("Qwen3.5-9B layer profile: 159x monotone growth with depth, no compression, "
                 "and only a weak (~5%/block) full-attention excess",
                 fontsize=11.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "q35_layer_profile.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_margins(D, Q):
    m = D["m"]
    rs = D["rstar"]
    K = D["K"]
    allc = np.bincount(D["tstar"], minlength=D["T"])
    flc = np.bincount(D["tstar"][D["flip_run"]], minlength=D["T"])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    ax = axes[0, 0]
    ax.semilogy(np.arange(D["T"]), m, color=C_OBS, lw=1.0)
    for t0 in np.argsort(m)[:5]:
        ax.plot(t0, m[t0], "o", ms=4.8, color=C_FULL, zorder=5)
        ax.annotate(f"t={t0}", xy=(t0, m[t0]), xytext=(t0 + 7, m[t0] * 2.1),
                    fontsize=7.2, color=C_FULL,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C_FULL))
    words = "\n".join(f"t={q['t']:3d}  m={q['m']:.2e}  {q['top1']!r} vs {q['top2']!r}"
                      for q in Q["margins"]["smallest5"])
    ax.text(0.985, 0.96, words, transform=ax.transAxes, va="top", ha="right",
            fontsize=7.1, family="monospace",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", lw=0.6))
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("baseline margin $m_t$  (nats, log scale)")
    ax.set_xlim(-2, D["T"] + 1)
    ax.set_title("(a) The A1 run's own margins: one step below 0.01 nats "
                 "(t=267, 'Ex' vs 'Specific')")

    ax = axes[0, 1]
    xs, ys = ecdf(rs)
    ax.plot(xs, ys, color=C_OBS, lw=1.8)
    ax.axvline(R_RUN, color=C_FULL, ls="--", lw=1.2)
    ax.text(R_RUN * 1.07, 0.40, "run radius\n$r=0.1$", fontsize=7.6, color=C_FULL)
    ax.axvline(0.116, color=C_PRED, ls=":", lw=1.1)
    ax.text(0.121, 0.68, "14B p50 0.116", fontsize=7.2, color=C_PRED)
    ax.axvline(1.555, color=C_LIN, ls=":", lw=1.1)
    ax.text(1.62, 0.36, "1.5B p50 1.56", fontsize=7.2, color=C_LIN)
    ax.set_xscale("log")
    ax.set_xlabel("flux-tube radius $r_*(u)=\\min_t r_c^-(t,u)$  (log scale)")
    ax.set_ylabel("fraction of directions with $r_*\\leq r$")
    ax.set_ylim(0, 1.02)
    ax.text(0.035, 0.96,
            f"p50 = {np.median(rs):.3f}\n"
            f"only {100 * Q['rstar']['flip_frac_at_0p1']:.1f}% flip at $r=0.1$\n"
            f"{100 * (rs > 1).mean():.0f}% need $r>1$",
            transform=ax.transAxes, va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", lw=0.6))
    ax.set_title("(b) The flux tube is 6x wider: 12/1000 cross at $r=0.1$")

    ax = axes[1, 0]
    top = [q for q in Q["ladder"][:10]][::-1]
    y = np.arange(len(top))
    ax.barh(y, [q["n_all"] for q in top], color="#c9d6e3",
            label="all directions")
    ax.barh(y, [q["n_flip"] for q in top], color=C_FULL,
            label=f"flip at $r=0.1$ ({int(D['flip_run'].sum())})")
    for yi, q in zip(y, top):
        ax.text(q["n_all"] + 7, yi, f"{q['n_all']}", va="center", fontsize=7.3)
        if q["n_flip"] > 0:
            ax.text(max(q["n_flip"] + 7, 16), yi, f"{q['n_flip']} flip",
                    va="center", fontsize=7.0, color=C_FULL)
    ax.set_yticks(y)
    ax.set_yticklabels([f"t={q['t']}  m={q['m']:.1e}" for q in top], fontsize=7.6)
    ax.set_xlabel("number of directions whose smallest threshold is this step")
    ax.set_xlim(0, max(q["n_all"] for q in top) * 1.25)
    ax.legend(loc="lower right")
    ax.set_title("(c) The danger ladder: t=267 controls 315; only 267 and 69 flip")

    ax = axes[1, 1]
    rg = np.geomspace(5e-3, 3.0, 320)
    frac = np.array([(rs <= x).mean() for x in rg])
    ax.semilogx(rg, 100 * frac, color=C_OBS, lw=1.9,
                label="predicted from signed rule")
    obs = [Q["free"][str(r)]["observed_flip_fraction"] for r in RHOS]
    prd = [Q["free"][str(r)]["predicted_flip_fraction_signed"] for r in RHOS]
    ax.errorbar(RHOS, 100 * np.array(obs), yerr=100 * np.sqrt(
        np.array(obs) * (1 - np.array(obs)) / K), fmt="o", ms=6.0,
        color=C_PRED, capsize=2.5, label="free-run observed")
    ax.plot(RHOS, 100 * np.array(prd), "x", ms=7, color=C_OBS,
            label="signed prediction at $\\rho$")
    ax.axvline(R_RUN, color=C_FULL, ls="--", lw=1.0)
    ax.set_xlabel("radius (log scale)")
    ax.set_ylabel("directions that flip (% of 1000)")
    ax.set_xlim(5e-3, 3)
    ax.legend(loc="upper left")
    ax.text(0.035, 0.44,
            "at $\\rho=0.1$: Qwen 1.2%,\n14B 45.3%, 1.5B 0/20",
            transform=ax.transAxes, ha="left", fontsize=7.4, color=C_GREY)
    ax.set_title("(d) Predicted flip curve reproduces all three free-run radii")

    fig.suptitle("Qwen3.5-9B margins and thresholds: a 0.0073-nat minimum (75x the 14B's) "
                 "makes the model 38x harder to flip at $r=0.1$",
                 fontsize=12.2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "q35_margins_rstar.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_free(D, Q):
    fig = plt.figure(figsize=(13.4, 7.6))
    gs = GridSpec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

    # scorecard: the four informative classes only; both-none is written above
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(RHOS))
    keys = [("exact", "exact"), ("wrong_step", "wrong step"),
            ("false_alarm", "false alarm"), ("miss", "miss")]
    bottom = np.zeros(len(RHOS))
    for k, lab in keys:
        vals = np.array([Q["free"][str(r)][k] for r in RHOS], float)
        ax.bar(x, vals, bottom=bottom, width=0.56, color=COL[k],
               edgecolor="white", linewidth=0.8, label=lab)
        for xi, vi, bi in zip(x, vals, bottom):
            if vi >= 18:
                ax.text(xi, bi + vi / 2, f"{int(vi)}", ha="center", va="center",
                        fontsize=7.8, color="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"$\\rho={r:g}$\nobs {Q['free'][str(r)]['observed_flips']} / "
        f"pred {Q['free'][str(r)]['predicted_flips_signed']}\n"
        f"none {Q['free'][str(r)]['both_none']}" for r in RHOS], fontsize=8.2)
    ax.set_ylim(0, 190)
    ax.set_ylabel("directions (of 1000)")
    ax.set_xlabel("free-run radius $\\rho$  (A1 calibrated at $r=0.1$)")
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("(a) Every flip is exact at $\\rho=0.1$; $\\rho=0.3$ is 91.8% accurate")

    # fractions
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(RHOS))
    w = 0.34
    obs = [100 * Q["free"][str(r)]["observed_flip_fraction"] for r in RHOS]
    prd = [100 * Q["free"][str(r)]["predicted_flip_fraction_signed"] for r in RHOS]
    absb = [100 * Q["free"][str(r)]["predicted_flip_fraction_signblind"] for r in RHOS]
    ax.bar(x - w / 2, obs, w, color=C_PRED, label="observed")
    ax.bar(x + w / 2, prd, w, color=C_OBS, alpha=0.82, label="signed prediction")
    ax.plot(x, absb, "x", ms=8, color=C_ABS, label="sign-blind prediction")
    for xi, a, b, c in zip(x, obs, prd, absb):
        if xi == len(RHOS) - 1:
            ax.text(xi - w / 2, a + 0.7, f"{a:.1f}", ha="center", fontsize=7.0)
            ax.text(xi + w / 2, b + 0.7, f"{b:.1f}", ha="center", fontsize=7.0)
            ax.text(xi, c + 1.0, f"{c:.1f}", ha="center", fontsize=7.0, color=C_ABS)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\rho={r:g}$" for r in RHOS])
    ax.set_ylabel("directions that flip (%)")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left", fontsize=7.4)
    ax.set_title("(b) Signed rule matches all radii;\n"
                 "sign-blind overpredicts by 15 points")

    # survival
    for ri, rho in enumerate(RHOS):
        ax = fig.add_subplot(gs[1, ri])
        o = D["obs"][ri]
        pred = D["pred_signed"][rho]
        INF = 10 ** 9
        og = np.where(o < 0, INF, o)
        pg = np.where(pred < 0, INF, pred)
        tg = np.arange(T_ALL + 1)
        so = np.array([np.mean(og > t) for t in tg])
        sp = np.array([np.mean(pg > t) for t in tg])
        ax.plot(tg, so, color="black", lw=1.8, drawstyle="steps-post",
                label="observed")
        ax.plot(tg, sp, color=C_PRED, lw=1.6, ls="--", drawstyle="steps-post",
                label="signed prediction")
        for tt in (69, 267):
            ax.axvline(tt, color="#777777", lw=0.7, ls=":", zorder=0)
        ax.set_xlim(0, 300)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("generated step $t$")
        if ri == 0:
            ax.set_ylabel("$S(t)=P(\\mathrm{first\\ flip}>t)$")
            ax.legend(loc="upper right")
        ax.set_title(f"$\\rho={rho:g}$   {Q['free'][str(rho)]['observed_flips']} flips "
                     f"vs {Q['free'][str(rho)]['predicted_flips_signed']} predicted",
                     fontsize=9.2)
        ax.text(0.04, 0.60,
                f"$S(69)$ {Q['free'][str(rho)]['S69_obs']:.3f} / "
                f"{Q['free'][str(rho)]['S69_pred']:.3f}\n"
                f"$S(267)$ {Q['free'][str(rho)]['S267_obs']:.3f} / "
                f"{Q['free'][str(rho)]['S267_pred']:.3f}\n"
                f"max $|\\Delta S|$ {Q['free'][str(rho)]['survival_max_abs_diff']:.3f}",
                transform=ax.transAxes, fontsize=7.2, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))

    fig.suptitle("Qwen3.5-9B free-run validation: the signed top-2 rule predicts 12/12 flips "
                 "at $\\rho=0.1$ and 160 vs 157 at $\\rho=0.3$",
                 fontsize=12.0, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "q35_free_validation.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_gain(D, Q):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    G = D["G"]

    ax = axes[0]
    lg = np.log10(G)
    ax.hist(lg, bins=42, color=C_LIN, alpha=0.82)
    tickvals = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_xticks(np.log10(tickvals))
    ax.set_xticklabels(["0.4", "0.6", "0.8", "1", "1.5", "2", "3", "5"])
    ax.axvline(np.median(lg), color=C_FULL, lw=1.4)
    ax.axvspan(np.log10(np.percentile(G, 25)), np.log10(np.percentile(G, 75)),
               color=C_FULL, alpha=0.08)
    ax2 = ax.twinx()
    xs, ys = ecdf(G)
    ax2.plot(np.log10(xs), ys, color=C_PRED, lw=1.3)
    ax2.set_ylabel("ECDF", color=C_PRED)
    ax2.tick_params(axis="y", colors=C_PRED)
    ax2.grid(False)
    ax.set_xlabel("$G(u)=\\mathrm{median}_t\\,\\|\\delta h_t^{\\mathrm{last}}\\|/r$  "
                  "(per unit radius, log scale)")
    ax.set_ylabel("directions per bin")
    ax.text(0.03, 0.96,
            f"p5={np.percentile(G, 5):.3f}\n"
            f"median={np.median(G):.3f}\n"
            f"p95={np.percentile(G, 95):.3f}\n"
            f"max/min={G.max() / G.min():.1f}x",
            transform=ax.transAxes, va="top", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(a) Gain spans 15x around a median of 0.82")

    ax = axes[1]
    pair = Q["_pair"]
    ax.hist(pair, bins=np.linspace(-0.25, 0.25, 55), color=C_LIN, alpha=0.82)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.median(pair), color=C_FULL, lw=1.2)
    ax.set_xlabel("$\\cos(\\delta h_i,\\delta h_j)$ between two response vectors")
    ax.set_ylabel("pairs per bin")
    ax.text(0.03, 0.96,
            f"mean cos={pair.mean():+.3f}\n"
            f"|cos|>0.1: {100 * np.mean(np.abs(pair) > 0.1):.1f}%\n"
            f"PC1: {100 * Q['pairwise']['pca_first'][0]:.1f}% of variance\n"
            f"(isotropic: {100 / 24:.1f}%)",
            transform=ax.transAxes, va="top", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(b) Response vectors are nearly orthogonal; no shared mode")

    ax = axes[2]
    c_ang = Q["_cos"]
    ax.hist(c_ang.ravel(), bins=np.linspace(-0.28, 0.28, 61), color=C_ABS, alpha=0.80)
    ax.axvline(0, color="k", lw=0.9)
    ax.axvline(np.cos(np.radians(87)), color=C_PRED, ls="--", lw=1.2,
               label="1.5B claim: 87 deg")
    ax.axvline(np.median(c_ang), color=C_FULL, lw=1.2, label="this model: median")
    ax.set_xlabel("$\\cos(\\delta h,\\,h_0)$ at the final-norm state")
    ax.set_ylabel("direction $\\times$ step pairs")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.text(0.97, 0.96,
            f"median angle {np.median(Q['_ang']):.2f} deg\n"
            f"IQR {np.percentile(Q['_ang'], 25):.2f}-"
            f"{np.percentile(Q['_ang'], 75):.2f} deg\n"
            f"{100 * (Q['_ang'] > 90).mean():.1f}% beyond 90 deg\n"
            f"{100 * Q['angle']['frac_within_87_pm_5']:.1f}% within 87$\\pm$5 deg",
            transform=ax.transAxes, va="top", ha="right", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(c) Responses sit 90.1 deg from the baseline state")

    fig.suptitle("Qwen3.5-9B gain structure: a 15x gain spread, orthogonal response "
                 "directions, and a 90.1-deg angle to the state",
                 fontsize=11.6, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGDIR, "q35_gain_structure.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    D = load()
    Q = compute(D)

    # derived arrays used by the gain figure
    dhl = D["dh_last"]
    nrm = np.linalg.norm(dhl, axis=2)
    tmed = np.argmin(np.abs(nrm - np.median(nrm, axis=1, keepdims=True)), axis=1)
    V = np.stack([dhl[i, tmed[i]] for i in range(dhl.shape[0])])
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    Q["_pair"] = (Vn @ Vn.T)[np.triu_indices(Vn.shape[0], 1)]
    h0 = D["h0_last"]
    nh = np.linalg.norm(h0, axis=1)
    Q["_cos"] = np.einsum("utd,td->ut", dhl, h0) / (
        np.maximum(nrm, 1e-30) * nh[None, :])
    Q["_ang"] = np.degrees(np.arccos(np.clip(Q["_cos"], -1, 1)))

    fig_calib(D, Q)
    fig_layer(D, Q)
    fig_margins(D, Q)
    fig_free(D, Q)
    fig_gain(D, Q)

    out = {
        "protocol": {
            "model": "Qwen3.5-9B (Qwen3_5ForConditionalGeneration), FP32",
            "blocks": 32,
            "hidden": 4096,
            "attention_pattern": "period 4: three Gated-DeltaNet linear-attention blocks, "
                                 "then one full-attention block",
            "full_attention_blocks": FULL_BLOCKS,
            "sliding_window": None,
            "window_field": 0,
            "prompt": "interstellar_travel, P=64 tokens",
            "T": T_ALL,
            "K": D["K"],
            "r": R_RUN,
            "saved_states": "0..31, state l = output of block l-1; state 0 = embeddings",
            "final_norm_state": "state 32 (block 31 output); only dh_last/h0_last, 24 dirs",
            "files": {"calib": "results_big/qwen35/calib/calib.npz",
                      "a1": "results_big/qwen35/a1_inter_w0/a1.npz",
                      "free": "results_big/qwen35/free_inter_w0/free_run.npz"},
        },
        "margins": Q["margins"],
        "calibration": Q["cal"],
        "calibration_margins": Q["cal_margins"],
        "rstar": Q["rstar"],
        "ladder": Q["ladder"],
        "ladder_n_distinct": Q["ladder_n_distinct"],
        "ladder_n_distinct_flip": Q["ladder_n_distinct_flip"],
        "layer_profile": Q["layer"],
        "gain": Q["gain"],
        "pairwise": Q["pairwise"],
        "angle": Q["angle"],
        "free": Q["free"],
        "cross_model": {
            "1.5B": {
                "blocks": 28, "hidden": 1536, "K": 20, "r": 0.1,
                "margin_min": 2.015686e-2, "margin_min_step": 123,
                "margin_median": 4.778, "rstar_p50": 1.555,
                "rstar_p90": 4.124, "gain_p50": 1.333,
                "gain_range": "0.635-7.468 (11.8x)",
                "floor": 1.0e-3, "clean_window": [1e-3, 0.1],
                "flip_fraction_at_0p1": 0.0,
                "free_flip_fraction_0p05_0p1_0p3": [None, 0.0, None],
                "angle_deg": 87.0,
            },
            "DS-14B": {
                "blocks": 48, "hidden": 5120, "K": 1000, "r": 0.1,
                "margin_min": 9.727478e-5, "margin_min_step": 21,
                "margin_median": 3.096, "rstar_p50": 0.116,
                "rstar_p90": 2.593, "gain_p50": 0.583,
                "gain_range": "0.224-3.064 (13.7x)",
                "floor": 2.7e-3, "clean_window": [0.03, 0.3],
                "flip_fraction_at_0p1": 0.453,
                "free_flip_fraction_0p05_0p1_0p3": [0.321, 0.453, 0.643],
                "angle_deg": None,
            },
            "Qwen3.5-9B": {
                "blocks": 32, "hidden": 4096, "K": 1000, "r": 0.1,
                "margin_min": Q["margins"]["min"], "margin_min_step": Q["margins"]["min_step"],
                "margin_median": Q["margins"]["p50"], "rstar_p50": Q["rstar"]["p50"],
                "rstar_p90": Q["rstar"]["p90"], "gain_p50": Q["gain"]["p50"],
                "gain_range": f"{Q['gain']['min']:.3f}-{Q['gain']['max']:.3f} "
                              f"({Q['gain']['max_over_min']:.1f}x)",
                "floor": Q["cal"]["floor_last_state"],
                "clean_window": list(CLEAN),
                "flip_fraction_at_0p1": Q["rstar"]["flip_frac_at_0p1"],
                "free_flip_fraction_0p05_0p1_0p3": [
                    Q["free"][str(r)]["observed_flip_fraction"] for r in RHOS],
                "angle_deg": Q["angle"]["median_angle"],
            },
        },
        "caveats": [
            "state l is the output of block l-1; the saved 0..31 covers blocks 0..30. "
            "Block 31 (full attention) output (state 32) was not saved.",
            "the stored cos field is cos(h',h0), which is 1 at fp32 resolution; the angle "
            "is computed only where h0 vectors exist (h0_last, 24 directions).",
            "the brief's calibration median (~3.99) is not what calib.npz contains: "
            "median 2.634, mean 4.143 over 400 steps.",
            "the free validation at rho=0.05 has zero observed flips, so its classification "
            "is trivially 100% and carries no step information.",
        ],
    }
    json.dump(jf(out), open(OUTJSON, "w"), indent=2)

    print(json.dumps(jf({
        "margins": Q["margins"],
        "cal": Q["cal"],
        "rstar": Q["rstar"],
        "layer": {k: Q["layer"][k] for k in
                  ("total_growth_block0_to_30", "full_growth_mean",
                   "linear_growth_mean_blocks1_30", "full_over_linear_growth",
                   "perm_p_full_blocks", "perm_p_full_blocks_detrended")},
        "gain": Q["gain"],
        "angle": {k: Q["angle"][k] for k in
                  ("median_angle", "mean_angle", "frac_angle_gt_90",
                   "frac_within_87_pm_5")},
        "free": Q["free"],
    })))


if __name__ == "__main__":
    main()
