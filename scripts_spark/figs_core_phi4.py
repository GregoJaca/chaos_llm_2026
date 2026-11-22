#!/usr/bin/env python
"""phi-4 core analysis: calibration, margins/thresholds, free validation,
layer profile, and -- when the GPU finishes them -- depth, non-normality,
state kicks and multi-prompt generality.

Reads (local copies pulled from the Spark; every path is optional except a1):
    results_big/phi4/calib/calib.npz              8 dirs x 7 radii, 400 tokens
    results_big/phi4/a1_inter_w0/a1.npz           1000 dirs, r=0.1, T=300,
                                                  14 saved states
    results_big/phi4/free_inter_w0/{free_run.npz,summary.json}
    results_big/phi4/state_w0/state2.npz          kicks at 10 positions x 24 dirs
    results_big/phi4/depth_w0/depth.npz           8 tokens x 40 blocks x k=8
    results_big/phi4/nonnorm_w0/nonnorm.npz       pending (fused-MLP Jacobian fix)
    results_big/phi4/a1_childhood_w0/a1.npz       200-dir multiprompt run
    results_big/phi4/a1_cyber_w0/a1.npz           200-dir multiprompt run
    assets/tok_phi4/tokenizer.json                tie/flip words

Writes
    notebook_big/figures/phi4_calib.png
    notebook_big/figures/phi4_core.png
    notebook_big/figures/phi4_free.png
    notebook_big/figures/phi4_layer_profile.png
    notebook_big/figures/phi4_depth.png           only if depth_w0 exists
    notebook_big/figures/phi4_nonnorm.png         only if nonnorm_w0 exists
    notebook_big/figures/phi4_state.png           only if state_w0 exists
    notebook_big/figures/phi4_generality.png      only if a multiprompt run exists
    notebook_big/figures/phi4_numbers.json

Index convention (checked against src/chaos_llm/big_engine.py:Capture):
    state 0      = embedding output
    state l >= 1 = output of BLOCK l-1
    so block b is visible as state b+1.  phi-4 has 40 blocks but only
    states 0,1,2,3,5,8,12,16,20,24,28,32,36,39 are saved: blocks
    0,1,2,4,7,11,15,19,23,27,31,35,38.  Block 39 (-> state 40) is not in
    dh_norm; its post-final-norm response exists for 24 directions in
    dh_last.  State 0 is identically zero (the perturbation only touches
    prompt positions, so nothing reaches a generated position before the
    first attention mix).

Run from the repository root:
    python scripts_spark/figs_core_phi4.py
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
ROOT = os.environ.get("CHAOS_PHI_ROOT", os.path.join(HERE, "results_big", "phi4"))
FIGDIR = os.environ.get("CHAOS_FIGDIR", os.path.join(HERE, "notebook_big", "figures"))
OUTJSON = os.environ.get("CHAOS_OUTJSON", os.path.join(FIGDIR, "phi4_numbers.json"))

CALIB = os.path.join(ROOT, "calib", "calib.npz")
A1 = os.path.join(ROOT, "a1_inter_w0", "a1.npz")
FREE = os.path.join(ROOT, "free_inter_w0", "free_run.npz")
SUMMARY = os.path.join(ROOT, "free_inter_w0", "summary.json")
STATE = os.path.join(ROOT, "state_w0", "state2.npz")
STATE_META = os.path.join(ROOT, "state_w0", "meta.json")
DEPTH = os.path.join(ROOT, "depth_w0", "depth.npz")
NONNORM = os.path.join(ROOT, "nonnorm_w0", "nonnorm.npz")
NONNORM_META = os.path.join(ROOT, "nonnorm_w0", "meta.json")
CHILD = os.path.join(ROOT, "a1_childhood_w0", "a1.npz")
CYBER = os.path.join(ROOT, "a1_cyber_w0", "a1.npz")
TOK = os.path.join(HERE, "assets", "tok_phi4", "tokenizer.json")

# reference A1 files for the same-computation fragility decomposition
REF_A1 = {
    "Qwen3.5-9B": os.path.join(HERE, "results_big", "qwen35", "a1_inter_w0", "a1.npz"),
    "DS-R1-14B": os.path.join(HERE, "results_big", "ds14", "a1_inter_w100", "a1.npz"),
}

R_RUN = 0.1
T_ALL = 300
T_CAL = 400
TWIN = (100, 400)
CLEAN = (1e-2, 0.3)
RHOS = (0.1, 0.3)
N_BLOCKS = 40
HIDDEN = 5120
VOCAB = 100352

C_OBS = "#111111"
C_PRED = "#d95f02"
C_ROB = "#2c7fb8"
C_HOT = "#c0392b"
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
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8.3,
})


# ------------------------------------------------------------------ helpers
def exists(p):
    return os.path.exists(p)


def jf(o):
    if isinstance(o, dict):
        return {str(k): jf(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jf(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jf(o.tolist())
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
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


TOK_DECODE = None
if exists(TOK):
    try:
        from tokenizers import Tokenizer

        TOK_DECODE = Tokenizer.from_file(TOK).decode
    except Exception as exc:                       # pragma: no cover
        print("tokenizer unavailable:", exc)


def dec(ids):
    if TOK_DECODE is None:
        return " ".join(str(int(i)) for i in np.atleast_1d(ids))
    return TOK_DECODE([int(i) for i in np.atleast_1d(ids)]).replace("\n", "\\n")


def signed_rc(m, dm, r=R_RUN):
    """Signed threshold radius (dm>=0 -> +inf) and its decomposition."""
    dm = np.asarray(dm, float)
    rc = np.where(dm < 0, m[None, :] * r / np.maximum(-dm, 1e-300), np.inf)
    return rc


# ------------------------------------------------------------------ data
def load():
    D = {"present": {
        "calib": exists(CALIB), "a1": exists(A1), "free": exists(FREE),
        "state": exists(STATE), "depth": exists(DEPTH), "nonnorm": exists(NONNORM),
        "childhood": exists(CHILD), "cyber": exists(CYBER)}}

    if D["present"]["calib"]:
        z = np.load(CALIB)
        D["cal_radii"] = z["radii"].astype(float)
        D["cal_states"] = z["layers"].astype(int)
        D["cal_m"] = z["margins"].astype(float)
        D["cal_top1"] = z["top1"].astype(int)
        D["cal_top2"] = z["top2"].astype(int)
        cal_dh = z["dh_norm"].astype(float)                       # (nr,ndir,ns,T)
        D["cal_dh"] = np.median(cal_dh[:, :, :, TWIN[0]:TWIN[1]], axis=(1, 3))
        D["cal_gain"] = D["cal_dh"] / D["cal_radii"][:, None]

    if D["present"]["a1"]:
        a = np.load(A1)
        D["m"] = a["margins"].astype(float)
        D["top1"] = a["top1"].astype(int)
        D["top2"] = a["top2"].astype(int)
        D["ref"] = a["gen_ids"].astype(np.int64)
        D["prompt_ids"] = a["prompt_ids"].astype(np.int64)
        D["dm"] = a["dm"][0].astype(float)                        # (K,T)
        D["dh"] = a["dh_norm"][0].astype(float)                   # (K,L,T)
        D["states"] = a["layers"].astype(int)
        D["dh_last"] = a["dh_last"][0].astype(float)              # (24,T,d)
        D["h0_last"] = a["h0_last"].astype(float)                 # (T,d)
        D["norm_h0"] = a["norm_h0"].astype(float)                 # (L,T)
        D["K"] = D["dm"].shape[0]
        D["T"] = D["dm"].shape[1]
        D["theta"] = a["radii"].astype(float)[0] if a["radii"].size else R_RUN

        D["rc"] = signed_rc(D["m"], D["dm"])
        D["tstar"] = D["rc"].argmin(axis=1)
        D["rstar"] = D["rc"][np.arange(D["K"]), D["tstar"]]
        D["flip_run"] = D["rstar"] <= R_RUN
        # local sensitivity at the controlling step, per unit radius
        dmc = D["dm"][np.arange(D["K"]), D["tstar"]]
        D["sens"] = np.where(dmc < 0, -dmc / R_RUN, np.inf)
        D["mstar"] = D["m"][D["tstar"]]

        D["prof"] = np.median(D["dh"], axis=0) / R_RUN             # (L,T)
        D["prof_med_t"] = np.median(D["prof"], axis=1)
        D["norm_med"] = np.median(D["norm_h0"], axis=1)
        # last saved state (block 38): gain for all 1000 directions
        D["G"] = np.median(D["dh"][:, -1, :], axis=1) / R_RUN

        # free-run predictions from A1 (signed and sign-blind)
        D["pred_signed"], D["pred_abs"] = {}, {}
        rcabs = D["m"][None, :] * R_RUN / np.maximum(np.abs(D["dm"]), 1e-300)
        for rho in RHOS:
            hit = D["rc"] <= rho
            anyh = hit.any(axis=1)
            p = np.full(D["K"], -1, dtype=int)
            p[anyh] = hit.argmax(axis=1)[anyh]
            D["pred_signed"][float(rho)] = p
            h2 = rcabs <= rho
            a2 = h2.any(axis=1)
            p2 = np.full(D["K"], -1, dtype=int)
            p2[a2] = h2.argmax(axis=1)[a2]
            D["pred_abs"][float(rho)] = p2

    if D["present"]["free"]:
        f = np.load(FREE, allow_pickle=True)
        D["obs"] = f["obs"].astype(int)
        D["fradii"] = f["radii"].astype(float)
        D["gen"] = np.asarray(f["gen_ids"], dtype=np.int64)
        D["agreement"] = f["agreement"].astype(float)
        if exists(SUMMARY):
            D["summ"] = json.load(open(SUMMARY, encoding="utf-8"))

    return D


# ------------------------------------------------------------------ quantities
def compute(D):
    Q = {}
    P = D["present"]

    if P["calib"]:
        cm = D["cal_m"]
        Q["cal_margins"] = qsummary(cm) | {
            "min_step": int(np.argmin(cm)),
            "n_below_0p1": int((cm < 0.1).sum()),
            "n_below_0p01": int((cm < 0.01).sum()),
            "smallest6": [{"t": int(t), "m": float(cm[t]),
                           "top1": dec([D["cal_top1"][t]]),
                           "top2": dec([D["cal_top2"][t]])}
                          for t in np.argsort(cm)[:6]],
        }
        R = D["cal_radii"]
        L = D["cal_states"]
        ls = list(L).index(39) if 39 in L else -1
        Q["calibration"] = {
            "radii": R.tolist(), "states": L.tolist(),
            "median_dh": D["cal_dh"].tolist(),
            "gain_dh_over_r": D["cal_gain"].tolist(),
            "floor_last_state": float(D["cal_dh"][0, ls]),
            "floor_last_state_3e3": float(D["cal_dh"][1, ls]),
            "signal_over_floor_at_r0p1": float(D["cal_dh"][4, ls] / D["cal_dh"][0, ls]),
            "gain_plateau_last_state": float(np.median(D["cal_gain"][3:6, ls])),
            "gain_at_r0p01_last_state": float(D["cal_gain"][2, ls]),
            "gain_at_r1_last_state": float(D["cal_gain"][-1, ls]),
            "clean_window": list(CLEAN),
        }

    if P["a1"]:
        m = D["m"]
        Q["margins"] = qsummary(m) | {
            "min_step": int(np.argmin(m)),
            "n_below_0p1": int((m < 0.1).sum()),
            "n_below_0p01": int((m < 0.01).sum()),
            "smallest5": [{"t": int(t), "m": float(m[t]),
                           "top1": dec([D["top1"][t]]),
                           "top2": dec([D["top2"][t]])}
                          for t in np.argsort(m)[:5]],
        }
        rs = D["rstar"]
        Q["rstar"] = qsummary(rs) | {
            "flip_frac_at_0p1": float(D["flip_run"].mean()),
            "flip_frac_at_0p3": float((rs <= 0.3).mean()),
            "n_above_1": int((rs > 1).sum()),
            "n_above_10": int((rs > 10).sum()),
        }
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

        # fragility decomposition  r* = m_{t*} / s_{t*}
        ok = np.isfinite(D["rstar"]) & np.isfinite(D["sens"])
        Q["decomposition"] = {
            "median_mstar": float(np.median(D["mstar"][ok])),
            "median_log10_mstar": float(np.median(np.log10(D["mstar"][ok]))),
            "median_sensitivity": float(np.median(D["sens"][ok])),
            "median_log10_sensitivity": float(np.median(np.log10(D["sens"][ok]))),
            "median_rstar": float(np.median(rs[ok])),
            "median_log10_rstar": float(np.median(np.log10(rs[ok]))),
            "identity_log10_rstar": float(np.median(np.log10(D["mstar"][ok]))
                                          - np.median(np.log10(D["sens"][ok]))),
            "share_of_dirs_on_5_thinnest_steps": float(
                np.isin(D["tstar"], np.argsort(m)[:5]).mean()),
            "per_direction": {
                "mstar": D["mstar"][ok].tolist(),
                "sensitivity": D["sens"][ok].tolist(),
                "rstar": rs[ok].tolist(),
                "tstar": D["tstar"][ok].tolist()},
        }

        # layer profile
        L = D["states"]
        pm = D["prof_med_t"]
        blocks = np.array([s - 1 for s in L])
        sel = blocks >= 0
        bl = blocks[sel]
        pmv = pm[sel]
        dlt = np.diff(bl)
        growth = (pmv[1:] / pmv[:-1]) ** (1.0 / dlt)
        Q["layer"] = {
            "states_saved": L.tolist(),
            "blocks_saved": bl.tolist(),
            "states_are_zero": [int(s) for s in L if pm[list(L).index(s)] == 0],
            "median_over_t": pmv.tolist(),
            "median_over_t_all_states": pm.tolist(),
            "growth_per_block_by_interval": [
                {"from_block": int(bl[i]), "to_block": int(bl[i + 1]),
                 "blocks": int(dlt[i]), "rate_per_block": float(growth[i])}
                for i in range(len(growth))],
            "total_growth_block0_to_38": float(pmv[-1] / pmv[0]),
            "span_blocks": int(bl[-1] - bl[0]),
            "mean_rate_per_block": float((pmv[-1] / pmv[0]) ** (1.0 / (bl[-1] - bl[0]))),
            "monotone_intervals": int(np.sum(np.diff(pmv) > 0)),
            "n_intervals": int(len(pmv) - 1),
            "relative_response": (pmv / D["norm_med"][sel]).tolist(),
            "state_norm_median": D["norm_med"][sel].tolist(),
            "state_norm_growth_block0_to_38": float(
                D["norm_med"][sel][-1] / D["norm_med"][sel][0]),
            "late_over_early": [float(np.median(D["prof"][i, 250:]) /
                                      np.median(D["prof"][i, :50]))
                                for i in np.where(sel)[0]],
            "final_norm_state": "state 40 (block 39), only dh_last/h0_last, 24 dirs",
        }

        # gain / pairwise / angle (final-norm state, 24 directions)
        Q["gain"] = qsummary(D["G"]) | {
            "max_over_min": float(D["G"].max() / D["G"].min())}
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
            "n_vectors": int(nvec), "mean_cos": float(pair.mean()),
            "median_cos": float(np.median(pair)),
            "frac_abs_cos_gt_0p1": float((np.abs(pair) > 0.1).mean()),
            "pca_first": pca[:8].tolist(),
            "isotropic_first": float(1.0 / nvec)}
        lag_cos = {}
        for lag in (1, 10, 50):
            cs = []
            for i in range(nvec):
                a = dhl[i, :-lag]
                b = dhl[i, lag:]
                cs.append(np.sum(a * b, 1) / np.maximum(
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30))
            lag_cos[str(lag)] = float(np.median(np.concatenate(cs)))
        Q["pairwise"]["temporal_median_cos"] = lag_cos

        h0 = D["h0_last"]
        nh = np.linalg.norm(h0, axis=1)
        c_ang = np.einsum("utd,td->ut", dhl, h0) / (
            np.maximum(nrm, 1e-30) * nh[None, :])
        ang = np.degrees(np.arccos(np.clip(c_ang, -1, 1)))
        Q["angle"] = {
            "n_pairs": int(ang.size),
            "angle_percentiles": {f"p{p}": float(np.percentile(ang, p))
                                  for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
            "mean_cos": float(c_ang.mean()), "median_cos": float(np.median(c_ang)),
            "mean_angle": float(ang.mean()), "median_angle": float(np.median(ang)),
            "frac_angle_gt_90": float((ang > 90).mean()),
            "frac_within_87_pm_5": float((np.abs(ang - 87) < 5).mean()),
            "per_direction_median_angle": np.median(ang, axis=1).tolist(),
            "per_step_median_angle_range": [
                float(np.min(np.median(ang, axis=0))),
                float(np.max(np.median(ang, axis=0)))],
            "dh_over_h0_median": float(np.median(nrm / nh[None, :])),
            "cos_hp_h0_min": float((np.einsum("utd,td->ut", h0[None] + dhl, h0) /
                                    (np.linalg.norm(h0[None] + dhl, axis=2)
                                     * nh[None, :])).min()),
        }

    if P["free"] and P["a1"]:
        Q["free"] = {}
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
            a_after, a_late, longest = [], [], []
            n_later_diff = []
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
                n_later_diff.append(int(np.sum(allg[u, oo + 1:] != ref[oo + 1:])))
            entry = {
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
                "A_after_mean": float(np.mean(a_after)) if a_after else None,
                "A_late_mean": float(np.nanmean(a_late)) if a_late else None,
                "longest_identical_run": longest,
                "n_rows_resync_ge20": int(np.sum(np.asarray(longest) >= 20)),
                "n_rows_permanent": int(np.sum(np.asarray(longest) < 20)),
                "flip_words": {},
                "flip_rows": [],
            }
            fw = {}
            for u in np.where(fl)[0]:
                oo = int(o[u])
                key = (oo, int(allg[u, oo]))
                fw[key] = fw.get(key, 0) + 1
                entry["flip_rows"].append({
                    "dir": int(u), "t": oo,
                    "ref_token": dec([ref[oo]]),
                    "flip_token": dec([allg[u, oo]]),
                    "predicted_t": int(pred[u]),
                    "longest_identical_post_run": int(longest[len(entry["flip_rows"])]),
                    "n_later_differences": int(n_later_diff[len(entry["flip_rows"])]),
                })
            entry["flip_words"] = {
                f"t{oo}:{dec([ref[oo]])}->{dec([tok])}": int(n)
                for (oo, tok), n in sorted(fw.items(), key=lambda kv: -kv[1])}
            INF = 10 ** 9
            og = np.where(o < 0, INF, o)
            pg = np.where(pred < 0, INF, pred)
            tg = np.arange(T_ALL + 1)
            so = np.array([np.mean(og > t) for t in tg])
            sp = np.array([np.mean(pg > t) for t in tg])
            entry["survival_max_abs_diff"] = float(np.max(np.abs(so - sp)))
            entry["survival_t_max_diff"] = int(np.argmax(np.abs(so - sp)))
            entry["S_at_171_obs"] = float(np.mean(og > 171))
            entry["S_at_171_pred"] = float(np.mean(pg > 171))
            entry["S_at_219_obs"] = float(np.mean(og > 219))
            entry["S_at_219_pred"] = float(np.mean(pg > 219))
            entry["S_at_277_obs"] = float(np.mean(og > 277))
            entry["S_at_277_pred"] = float(np.mean(pg > 277))
            Q["free"][str(rho)] = entry

    # reference decomposition (same signed rule, same code path)
    Q["decomposition_ref"] = {}
    for name, path in REF_A1.items():
        if not exists(path):
            continue
        try:
            a = np.load(path)
            mm = a["margins"].astype(float)
            dd = a["dm"][0].astype(float)
            rr = float(a["radii"][0]) if a["radii"].size else R_RUN
            rc = np.where(dd < 0, mm[None, :] * rr / np.maximum(-dd, 1e-300), np.inf)
            ts = rc.argmin(axis=1)
            rs = rc[np.arange(rc.shape[0]), ts]
            ss = -dd[np.arange(rc.shape[0]), ts] / rr
            mst = mm[ts]
            ok = np.isfinite(rs) & np.isfinite(ss) & (ss > 0)
            Q["decomposition_ref"][name] = {
                "median_mstar": float(np.median(mst[ok])),
                "median_sensitivity": float(np.median(ss[ok])),
                "median_rstar": float(np.median(rs[ok])),
                "median_log10_mstar": float(np.median(np.log10(mst[ok]))),
                "median_log10_sensitivity": float(np.median(np.log10(ss[ok]))),
                "median_log10_rstar": float(np.median(np.log10(rs[ok]))),
                "r_mstar_over_s_geometric": float(10 ** (np.median(np.log10(mst[ok]))
                                                       - np.median(np.log10(ss[ok])))),
                "source": os.path.relpath(path, HERE).replace("\\", "/"),
            }
        except Exception as exc:                     # pragma: no cover
            print("reference decomposition failed for", name, exc)

    # ---------------- pending / optional blocks
    Q["pending"] = [k for k in ("state", "depth", "nonnorm", "childhood", "cyber")
                    if not D["present"][k]]

    if D["present"]["depth"]:
        Q["depth"] = compute_depth(D)
    if D["present"]["nonnorm"]:
        Q["nonnorm"] = compute_nonnorm(D)
    if D["present"]["state"]:
        Q["state"] = compute_state(D)
    if D["present"]["childhood"] or D["present"]["cyber"]:
        Q["generality"] = compute_generality(D)

    return Q


# ------------------------------------------------------------------ optional blocks
def _norm_blocks(D):
    """(block index, median baseline norm) pairs; state s -> block s-1."""
    L = D["states"]
    return np.array([s - 1 for s in L if s >= 1]), D["norm_med"][np.array(L) >= 1]


def compute_depth(D):
    z = np.load(DEPTH)
    lam = z["lam"].astype(float)                     # (ntok, nlayers, k)
    lg = z["local_gain"].astype(float)               # (ntok, nlayers)
    hd = z["h_depth"].astype(float)
    tokens = z["tokens"].astype(int)
    layers = z["layers"].astype(int)
    n = lam.shape[1]
    k = lam.shape[2]
    cum = (layers[None, :, None] + 1) * lam          # cumulative log growth
    ex = np.empty_like(cum)
    ex[:, 0, :] = cum[:, 0, :]
    ex[:, 1:, :] = cum[:, 1:, :] - cum[:, :-1, :]    # per-block exponent e_l
    vol_growth = np.exp(n * lam[:, -1, :].mean(axis=1))

    bl, nb = _norm_blocks(D)
    keep = (bl >= layers[0]) & (bl <= layers[-1])
    bl, nb = bl[keep], nb[keep]
    cumb = (bl[:, None] + 1) * lam[:, bl, :].mean(axis=2).T   # (ninterval+1, ntok)
    rel = {}
    for i in range(len(bl) - 1):
        d = bl[i + 1] - bl[i]
        tang = (cumb[i + 1] - cumb[i]) / d
        nrate = np.log(nb[i + 1] / nb[i]) / d
        rel[f"{bl[i]}->{bl[i + 1]}"] = (tang - nrate).tolist()
    rel_arr = np.array(list(rel.values()))           # (ninterval, ntok)
    nh_tok = np.asarray(D["norm_h0"][-1, tokens]
                        / D["norm_h0"][0, tokens], float)
    return {
        "tokens": tokens.tolist(), "layers": layers.tolist(), "k": int(k),
        "h_depth": hd.tolist(),
        "final_lambda_median": np.median(lam[:, -1, :], axis=1).tolist(),
        "final_lambda_min": lam[:, -1, :].min(axis=1).tolist(),
        "final_lambda_max": lam[:, -1, :].max(axis=1).tolist(),
        "final_spectrum": lam[:, -1, :].tolist(),
        "tangent_volume_growth": vol_growth.tolist(),
        "state_norm_growth_by_token": nh_tok.tolist(),
        "relative_endpoint_by_token": (
            np.log(vol_growth) - np.log(nh_tok)) / n,
        "block0_exponent_median": np.median(ex[:, 0, :], axis=1).tolist(),
        "late_exponent_median": np.median(ex[:, -5:, :].mean(axis=2), axis=1).tolist(),
        "local_gain_median_tokens": np.median(lg, axis=0).tolist(),
        "state_norm_growth_blocks": float(nb[-1] / nb[0]) if len(nb) else None,
        "relative_chunks": {kk: {"median": float(np.median(vv)),
                                 "min": float(np.min(vv)),
                                 "max": float(np.max(vv))}
                            for kk, vv in rel.items()},
        "relative_n_negative": int((rel_arr < 0).sum()),
        "relative_n_cells": int(rel_arr.size),
        "relative_overall_median": float(np.median(rel_arr)),
        "relative_overall_by_token": np.median(rel_arr, axis=0).tolist(),
        "relative_min_cell": float(rel_arr.min()),
        "relative_max_cell": float(rel_arr.max()),
        "norm_state_blocks": bl.tolist(),
        "norm_state_norms": nb.tolist(),
        "arrays": {"lambda_tokens": lam.mean(axis=2).tolist(),
                   "lam_med": np.median(lam.mean(axis=2), axis=0).tolist(),
                   "lambda_lo": np.percentile(lam.mean(axis=2), 25, axis=0).tolist(),
                   "lambda_hi": np.percentile(lam.mean(axis=2), 75, axis=0).tolist(),
                   "ex_med": np.median(ex.mean(axis=2), axis=0).tolist(),
                   "ex_lo": np.percentile(ex.mean(axis=2), 25, axis=0).tolist(),
                   "ex_hi": np.percentile(ex.mean(axis=2), 75, axis=0).tolist(),
                   "local_gain_med": np.median(lg, axis=0).tolist(),
                   "local_gain_lo": np.percentile(lg, 25, axis=0).tolist(),
                   "local_gain_hi": np.percentile(lg, 75, axis=0).tolist(),
                   "cum_rel": cumb.tolist()},
    }


def compute_nonnorm(D):
    z = np.load(NONNORM)
    meta = json.load(open(NONNORM_META, encoding="utf-8")) if exists(NONNORM_META) else {}
    sel = z["sel_layers"].astype(int)
    full = np.array(meta.get("full_layers", []), dtype=int)
    if full.size == 0:
        n = z["layers"].size
        full = np.unique(np.linspace(0, n - 1, 5).round().astype(int))
    full = full[np.isin(full, z["layers"].astype(int))]
    layers_all = z["layers"].astype(int)
    fi = np.array([list(layers_all).index(b) for b in full])
    angles = z["angles"].astype(float).ravel()
    ms = z["mlp_smax"][:, sel]
    return {
        "tokens": z["tokens"].astype(int).tolist(),
        "sel_layers": sel.tolist(), "full_layers": full.tolist(),
        "mlp_smax_med": np.median(ms, axis=0).tolist(),
        "mlp_smax_lo": np.min(ms, axis=0).tolist(),
        "mlp_smax_hi": np.max(ms, axis=0).tolist(),
        "mlp_rho_med": np.median(z["mlp_rho"][:, sel], axis=0).tolist(),
        "mlp_kappa_med": np.median(z["mlp_kappa"][:, sel], axis=0).tolist(),
        "full_smax_med": np.median(z["full_smax"][:, fi], axis=0).tolist(),
        "full_smax_lo": np.min(z["full_smax"][:, fi], axis=0).tolist(),
        "full_smax_hi": np.max(z["full_smax"][:, fi], axis=0).tolist(),
        "full_rho_med": np.median(z["full_rho"][:, fi], axis=0).tolist(),
        "full_rho_lo": np.min(z["full_rho"][:, fi], axis=0).tolist(),
        "full_rho_hi": np.max(z["full_rho"][:, fi], axis=0).tolist(),
        "full_kappa_med": np.median(z["full_kappa"][:, fi], axis=0).tolist(),
        "full_kappa_lo": np.min(z["full_kappa"][:, fi], axis=0).tolist(),
        "full_kappa_hi": np.max(z["full_kappa"][:, fi], axis=0).tolist(),
        "angles": angles.tolist(),
        "angle_median": float(np.median(angles)),
        "angle_per_token_median": np.median(z["angles"], axis=1).tolist(),
        "angle_layers": z["angle_layers"].astype(int).tolist(),
        "full_smax_at_full": z["full_smax"][:, fi].tolist(),
        "full_rho_at_full": z["full_rho"][:, fi].tolist(),
        "full_kappa_at_full": z["full_kappa"][:, fi].tolist(),
    }


def compute_state(D):
    z = np.load(STATE)
    dh = z["dh"].astype(float)                       # (nr,ns,nd,nl,T)
    dm = z["dm"].astype(float)                       # (nr,ns,nd,T)
    radii = z["radii"].astype(float)
    steps = z["steps"].astype(int)
    layers = z["layers"].astype(int)
    m = D["m"]
    nr, ns, nd, nl, T = dh.shape
    rip = 0 if abs(radii[0] - R_RUN) < abs(radii[-1] - R_RUN) else nr - 1
    r_used = radii[rip]
    res = {"radii": radii.tolist(), "steps": steps.tolist(),
           "layers": layers.tolist(), "inject_layer": int(z["inject_layer"]),
           "readout_state": int(layers[-1]),
           "immediate": [], "halflife": [], "late_ratio": [],
           "frac_decaying": [], "flip_frac": [], "earliest_median": [],
           "rstar_median": [], "rstar_min": [], "immediate_by_radius": {},
           "trajectories": {}}
    for ri_, rv in enumerate(radii):
        res["immediate_by_radius"][f"{rv:g}"] = [
            float(np.median(dh[ri_, si, :, -1, s] / rv))
            for si, s in enumerate(steps)]
    for si, s in enumerate(steps):
        v = dh[rip, si, :, -1, :] / r_used
        imm = np.median(v[:, s])
        res["immediate"].append(float(imm))
        hl, dec = [], 0
        later, ratio = [], []
        for u in range(nd):
            post = v[u, s + 1:]
            below = np.where(post < 0.5 * v[u, s])[0]
            if below.size:
                hl.append(int(below[0] + 1))
                dec += 1
            if post.size:
                later.append(float(post.max()))
                ratio.append(float(post.max() / max(v[u, s], 1e-30)))
        res["halflife"].append(float(np.median(hl)) if hl else None)
        res["frac_decaying"].append(dec / nd)
        res["late_ratio"].append(float(np.median(ratio)) if ratio else None)
        # flips at radius <= 1, first later step
        rc = np.where(dm[rip, si] < 0, m[None, :] * r_used / np.maximum(-dm[rip, si], 1e-300),
                      np.inf)
        first, nflip = [], 0
        for u in range(nd):
            hit = np.where((rc[u] <= 1.0) & (np.arange(T) > s))[0]
            if hit.size:
                nflip += 1
                first.append(int(hit[0]))
        res["flip_frac"].append(nflip / nd)
        res["earliest_median"].append(float(np.median(first)) if first else None)
        if s < T - 1:
            rs = rc.min(axis=1)
            res["rstar_median"].append(float(np.median(rs)))
            res["rstar_min"].append(float(rs.min()))
        else:
            res["rstar_median"].append(None)
            res["rstar_min"].append(None)
        res["trajectories"][str(int(s))] = np.median(v, axis=0).tolist()
    return res


def _rstar_block(path, D):
    a = np.load(path)
    m = a["margins"].astype(float)
    dm = a["dm"][0].astype(float)
    r = float(a["radii"][0]) if a["radii"].size else R_RUN
    rc = np.where(dm < 0, m[None, :] * r / np.maximum(-dm, 1e-300), np.inf)
    ts = rc.argmin(axis=1)
    rs = rc.min(axis=1)
    ss = -dm[np.arange(rc.shape[0]), ts] / r
    mst = m[ts]
    ok = np.isfinite(rs) & np.isfinite(ss) & (ss > 0)
    allc = np.bincount(ts, minlength=m.size)
    order = np.argsort(allc)[::-1]
    return {"m": m, "rstar": rs, "K": int(rc.shape[0]),
            "margins": qsummary(m), "rstar_q": qsummary(rs),
            "flip_frac_at_0p1": float((rs <= 0.1).mean()),
            "flip_frac_at_0p3": float((rs <= 0.3).mean()),
            "flip_frac_at_1": float((rs <= 1.0).mean()),
            "median_mstar": float(np.median(mst[ok])),
            "median_sensitivity": float(np.median(ss[ok])),
            "n_distinct_controlling_steps": int((allc > 0).sum()),
            "share_5_thinnest": float(np.isin(ts, np.argsort(m)[:5]).mean()),
            "ladder": [{"t": int(t), "n": int(allc[t]), "m": float(m[t])}
                       for t in order[:6] if allc[t] > 0]}


def _spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_generality(D):
    out = {}
    for name, path in (("interstellar", A1), ("childhood", CHILD), ("cyber", CYBER)):
        if name != "interstellar":
            if not exists(path):
                out[name] = "pending"
                continue
        b = _rstar_block(path, D)
        if "interstellar" in out and isinstance(out["interstellar"], dict):
            ref = out["interstellar"]
            n = min(ref["K"], b["K"])
            b["vs_interstellar"] = {
                "n": int(n),
                "spearman_log_rstar": _spearman(np.log(ref["rstar"][:n]),
                                               np.log(b["rstar"])),
                "shared_flip_le1": int(np.sum((ref["rstar"][:n] <= 1)
                                              & (b["rstar"] <= 1))),
                "ref_flip_le1": int(np.sum(ref["rstar"][:n] <= 1)),
                "flip_le1": int(np.sum(b["rstar"] <= 1)),
                "note": "same seed-69 direction generator; the first n directions "
                        "coincide"},
        out[name] = b
    return out


# ------------------------------------------------------------------ figures
def fig_calib(D, Q):
    R = D["cal_radii"]
    L = D["cal_states"]
    G = D["cal_gain"]
    med = D["cal_dh"]
    m = D["cal_m"]
    ls = list(L).index(39) if 39 in L else -1

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    ax = axes[0, 0]
    t = np.arange(len(m))
    ax.semilogy(t, m, color=C_OBS, lw=1.0)
    ax.axvline(T_ALL, color=C_GREY, ls=":", lw=1.0)
    ax.annotate("A1 horizon (300 tokens)", xy=(T_ALL, 0.965),
                xycoords=("data", "axes fraction"),
                xytext=(T_ALL + 8, 0.995), textcoords=("data", "axes fraction"),
                fontsize=7.2, color=C_GREY, va="top")
    offs = {277: (-48, 12), 219: (-34, -20), 53: (8, 12), 34: (-36, 14), 186: (8, -18)}
    for k, t0 in enumerate(np.argsort(m)[:5]):
        ax.plot(t0, m[t0], "o", ms=4.6, color=C_HOT, zorder=5)
        dx, dy = offs.get(int(t0), (8, 10))
        ax.annotate(f"t={t0}\n{m[t0]:.2g}", xy=(t0, m[t0]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.9, color=C_HOT,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HOT))
    ax.set_xlabel("generated step $t$ (calibration run, 400 tokens)")
    ax.set_ylabel("$m_t=\\ell_t^{(1)}-\\ell_t^{(2)}$  (nats, log scale)")
    ax.set_xlim(-2, len(m) + 1)
    ax.set_ylim(m.min() * 0.55, None)
    ax.set_title("(a) Thinnest gap 0.0092 nats at t=277; no razor tie")

    ax = axes[0, 1]
    xs, ys = ecdf(m)
    ax.plot(xs, ys, color=C_OBS, lw=1.6)
    ax.set_xscale("log")
    labs = [(m.min(), f"phi-4 {m.min():.1e}", C_HOT, 0.55, 1.07, "left"),
            (9.727e-5, "14B 9.7e-5", C_PRED, 0.08, 1.06, "left"),
            (7.27e-3, "Qwen3.5 7.3e-3", C_ABS, 0.30, 0.94, "right"),
            (2.016e-2, "1.5B 2.0e-2", C_ROB, 0.20, 1.06, "left")]
    for val, lab, col, yy, fx, ha in labs:
        ax.axvline(val, color=col, ls="--", lw=1.15)
        ax.text(val * fx, yy, lab, rotation=90, fontsize=6.9,
                color=col, va="bottom", ha=ha)
    ax.set_xlabel("baseline top-1/top-2 logit gap $m_t$ (nats, log scale)")
    ax.set_ylabel("fraction of steps with margin $\\leq m$")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) The minimum is 95x the 14B's, close to Qwen3.5's")

    ax = axes[1, 0]
    cols = plt.cm.viridis(np.linspace(0.08, 0.88, len(L)))
    i0 = 4  # r=0.1
    for si, (st, c) in enumerate(zip(L, cols)):
        if st == 0:
            continue
        ax.loglog(R, med[:, si], "o-", ms=3.2, lw=1.2, color=c,
                  label=f"state {st} (block {st - 1})")
        xx = np.array([3e-3, 2e-1])
        ax.loglog(xx, med[i0, si] * xx / R[i0], ls=":", lw=0.9, color=c, alpha=0.75)
    ax.axhline(Q["calibration"]["floor_last_state"], color=C_GREY, ls="--", lw=1.0)
    ax.text(1.1e-3, Q["calibration"]["floor_last_state"] * 1.17,
            f"fp32 floor {Q['calibration']['floor_last_state']:.2e}",
            fontsize=7.5, color=C_GREY)
    ax.axvspan(*CLEAN, color=C_ROB, alpha=0.07)
    ax.axvline(R_RUN, color=C_HOT, ls=":", lw=1.0)
    ax.set_xlabel("perturbation radius $r$ (log scale)")
    ax.set_ylabel("median $\\|\\delta h^{\\mathrm{last}}\\|$  (log scale)")
    ax.legend(loc="upper left", ncol=2, fontsize=7.4)
    ax.set_title("(c) Linear in $r$ from 0.01 to 0.3; floor at $r\\leq3\\times10^{-3}$")

    ax = axes[1, 1]
    for st, c in zip(L, cols):
        if st == 0:
            continue
        ax.loglog(R, G[:, list(L).index(st)], "o-", ms=3.2, lw=1.2, color=c,
                  label=f"state {st}")
    ax.axvspan(*CLEAN, color=C_ROB, alpha=0.07)
    ax.axhline(Q["calibration"]["gain_plateau_last_state"], color=C_GREY, ls="--", lw=0.9,
               label=f"plateau (state 39) {Q['calibration']['gain_plateau_last_state']:.2f}")
    ax.axvline(R_RUN, color=C_HOT, ls=":", lw=1.0)
    ax.set_xlabel("perturbation radius $r$ (log scale)")
    ax.set_ylabel("$\\|\\delta h^{\\mathrm{last}}\\|/r$  (log scale)")
    ax.legend(loc="upper right", ncol=2, fontsize=7.2)
    ax.set_title("(d) The weakest response gain of the four models")

    fig.suptitle("phi-4 calibration: a 0.0092-nat minimum (95x the 14B's) but a response "
                 "gain of only 0.38, the lowest of the four models",
                 fontsize=11.6, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_calib.png"), dpi=165)
    plt.close(fig)


def _anchor_points(Q):
    pts = [("phi-4", Q["decomposition"]["median_sensitivity"],
            Q["decomposition"]["median_mstar"], C_HOT)]
    for name, col in (("Qwen3.5-9B", C_ABS), ("DS-R1-14B", C_PRED)):
        if name in Q.get("decomposition_ref", {}):
            r = Q["decomposition_ref"][name]
            pts.append((name, r["median_sensitivity"], r["median_mstar"], col))
    return pts


def fig_core(D, Q):
    m = D["m"]
    rs = D["rstar"]
    K = D["K"]
    allc = np.bincount(D["tstar"], minlength=D["T"])
    flc = np.bincount(D["tstar"][D["flip_run"]], minlength=D["T"])
    G = D["G"]
    ang = Q["angle"]

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.7))
    ax = axes[0, 0]
    ax.semilogy(np.arange(D["T"]), m, color=C_OBS, lw=1.0)
    offs = {277: (-48, 12), 219: (-34, -20), 53: (8, 12), 34: (-36, 14), 186: (8, -18)}
    for k, t0 in enumerate(np.argsort(m)[:5]):
        ax.plot(t0, m[t0], "o", ms=4.8, color=C_HOT, zorder=5)
        dx, dy = offs.get(int(t0), (8, 10))
        ax.annotate(f"t={t0}", xy=(t0, m[t0]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.0, color=C_HOT,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HOT))
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("baseline margin $m_t$  (nats, log scale)")
    ax.set_xlim(-2, D["T"] + 1)
    ax.set_ylim(m.min() * 0.55, None)
    ax.set_title("(a) Only one step below 0.01 nats: t=277")

    ax = axes[0, 1]
    xs, ys = ecdf(rs)
    ax.plot(xs, ys, color=C_OBS, lw=1.8)
    ax.axvline(R_RUN, color=C_HOT, ls="--", lw=1.2)
    ax.text(R_RUN * 1.07, 0.86, "run radius\n$r=0.1$", fontsize=7.2, color=C_HOT)
    for val, lab, col, yy in [(0.116, "14B p50 0.116", C_PRED, 0.62),
                              (0.707, "Qwen3.5 p50 0.707", C_ABS, 0.43),
                              (1.555, "1.5B p50 1.56", C_ROB, 0.24)]:
        ax.axvline(val, color=col, ls=":", lw=1.1)
        ax.text(val * 1.05, yy, lab, fontsize=6.9, color=col)
    ax.set_xscale("log")
    ax.set_xlim(0.09, 60)
    ax.set_xlabel("flux-tube radius $r_*(u)=\\min_t r_c^-(t,u)$  (log scale)")
    ax.set_ylabel("fraction of directions with $r_*\\leq r$")
    ax.set_ylim(0, 1.02)
    ax.text(0.97, 0.04,
            f"p50 = {np.median(rs):.2f}\n"
            f"0/1000 cross at $r=0.1$\n"
            f"2/1000 at $r=0.3$, {100 * (rs > 1).mean():.0f}% need $r>1$",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", lw=0.6))
    ax.set_title("(b) The widest flux tube of the four models")

    ax = axes[1, 0]
    top = [q for q in Q["ladder"][:9]][::-1]
    y = np.arange(len(top))
    ax.barh(y, [q["n_all"] for q in top], color="#c9d6e3", label="all directions")
    ax.barh(y, [q["n_flip"] for q in top], color=C_HOT, label="flip at $r=0.1$ (0)")
    for yi, q in zip(y, top):
        ax.text(q["n_all"] + 5, yi, f"{q['n_all']}", va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"t={q['t']}  m={q['m']:.1e}" for q in top], fontsize=7.4)
    ax.set_xlabel("directions whose smallest threshold is this step")
    ax.set_xlim(0, max(q["n_all"] for q in top) * 1.22)
    ax.legend(loc="lower right")
    ax.set_title("(c) Two thin steps (219, 277) control 53% of directions")

    ax = axes[1, 1]
    lG = np.log10(G)
    ax.hist(lG, bins=40, color=C_ROB, alpha=0.82)
    ticks = np.array([0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0])
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.axvline(np.median(lG), color=C_HOT, lw=1.4)
    ax.axvspan(np.log10(np.percentile(G, 25)), np.log10(np.percentile(G, 75)),
               color=C_HOT, alpha=0.08)
    ax2 = ax.twinx()
    xs, ys = ecdf(G)
    ax2.plot(np.log10(xs), ys, color=C_PRED, lw=1.3)
    ax2.set_ylabel("ECDF", color=C_PRED)
    ax2.tick_params(axis="y", colors=C_PRED)
    ax2.grid(False)
    ax.set_xlabel("$G(u)=\\mathrm{median}_t\\,\\|\\delta h_t^{\\mathrm{last}}\\|/r$  (log scale)")
    ax.set_ylabel("directions per bin")
    ax.text(0.03, 0.96,
            f"p5={np.percentile(G, 5):.3f}\nmedian={np.median(G):.3f}\n"
            f"p95={np.percentile(G, 95):.3f}\nmax/min={G.max() / G.min():.1f}x",
            transform=ax.transAxes, va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(d) Gain spans 20x around a median of 0.38")

    ax = axes[0, 2]
    ax.hist(ang["_cos"].ravel() if "_cos" in ang else np.cos(np.radians(ang["_ang"])),
            bins=np.linspace(-0.35, 0.35, 61), color=C_ABS, alpha=0.8)
    ax.axvline(0, color="k", lw=0.9)
    ax.axvline(np.cos(np.radians(87)), color=C_ROB, ls="--", lw=1.1,
               label="1.5B claim: 87 deg")
    ax.set_xlabel("$\\cos(\\delta h,\\,h_0)$ at the final-norm state")
    ax.set_ylabel("direction $\\times$ step pairs")
    ax.legend(loc="upper left", fontsize=7.4)
    ax.text(0.97, 0.96,
            f"median angle {ang['median_angle']:.2f} deg\n"
            f"IQR {ang['angle_percentiles']['p25']:.1f}-"
            f"{ang['angle_percentiles']['p75']:.1f} deg\n"
            f"{100 * ang['frac_angle_gt_90']:.1f}% beyond 90 deg",
            transform=ax.transAxes, va="top", ha="right", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(e) Responses sit 89.9 deg from the state")

    ax = axes[1, 2]
    s = D["sens"]
    mm = D["mstar"]
    ok = np.isfinite(rs) & np.isfinite(s) & (s > 0)
    ax.scatter(s[ok], mm[ok], s=5, alpha=0.22, color=C_ROB, edgecolors="none")
    ss = np.geomspace(3e-4, 2.0, 50)
    iso = {0.1: (0.05, 0.0052), 1.0: (0.05, 0.052), 10.0: (0.05, 0.52)}
    for rr, dsh in ((0.1, ":"), (1.0, "--"), (10.0, "-.")):
        ax.plot(ss, rr * ss, dsh, color=C_GREY, lw=1.0)
        lx, ly = iso[rr]
        ax.text(lx, ly, f"$r_*={rr:g}$", fontsize=6.6, color="#555555",
                rotation=25, va="bottom")
    for name, sx, my, col in _anchor_points(Q):
        ax.plot(sx, my, "D", ms=8.5, color=col, markeredgecolor="white", zorder=6)
        off = {"phi-4": (-30, -3), "Qwen3.5-9B": (5, 4), "DS-R1-14B": (5, -2)}.get(
            name, (5, -11))
        ax.annotate(name, xy=(sx, my), xytext=off, textcoords="offset points",
                    fontsize=7.2, color=col, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e-4, 0.3)
    ax.set_ylim(4e-4, 3.0)
    ax.set_xlabel("local sensitivity $s=|\\delta m|/r$ at the controlling step")
    ax.set_ylabel("controlling margin $m_{t_*}$  (nats)")
    ax.set_title("(f) Why: thin margins, but $s$ is 4x below Qwen3.5's")
    ax.text(0.03, 0.96,
            f"median $r_*$ {Q['decomposition']['median_rstar']:.2f}\n"
            f"$m_*$ {Q['decomposition']['median_mstar']:.3f} nats\n"
            f"$s$ {Q['decomposition']['median_sensitivity']:.4f}",
            transform=ax.transAxes, va="top", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))

    fig.suptitle("phi-4 margins and thresholds: 0.0092-nat minimum and a 3.6-unit flux tube -- "
                 "the least flippable model of the four",
                 fontsize=12.0, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_core.png"), dpi=165)
    plt.close(fig)


def fig_free(D, Q):
    fig = plt.figure(figsize=(14.2, 8.0))
    gs = GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(RHOS))
    bottom = np.zeros(len(RHOS))
    for k, lab in (("exact", "exact"), ("wrong_step", "wrong step"),
                   ("false_alarm", "false alarm"), ("miss", "miss")):
        vals = np.array([Q["free"][str(r)][k] for r in RHOS], float)
        ax.bar(x, vals, bottom=bottom, width=0.55, color=COL[k],
               edgecolor="white", linewidth=0.8, label=lab)
        bottom += vals
    for xi, r in zip(x, RHOS):
        e = Q["free"][str(r)]
        total = e["exact"] + e["wrong_step"] + e["false_alarm"] + e["miss"]
        ax.text(xi, total + 0.25, f"obs {e['observed_flips']}\npred "
                                 f"{e['predicted_flips_signed']}\nboth none "
                                 f"{e['both_none']}", ha="center", va="bottom",
                fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\rho={r:g}$" for r in RHOS])
    ax.set_ylim(0, 8)
    ax.set_ylabel("directions in each class (of 1000)")
    ax.set_xlabel("free-run radius $\\rho$  (A1 calibrated at $r=0.1$)")
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("(a) No event at $\\rho=0.1$; 4 at $\\rho=0.3$")

    ax = fig.add_subplot(gs[0, 1])
    obs = [100 * Q["free"][str(r)]["observed_flip_fraction"] for r in RHOS]
    prd = [100 * Q["free"][str(r)]["predicted_flip_fraction_signed"] for r in RHOS]
    absb = [100 * Q["free"][str(r)]["predicted_flip_fraction_signblind"] for r in RHOS]
    w = 0.32
    ax.bar(x - w / 2, obs, w, color=C_PRED, label="observed")
    ax.bar(x + w / 2, prd, w, color=C_OBS, alpha=0.82, label="signed prediction")
    ax.plot(x + 0.30, absb, "x", ms=8, color=C_ABS, label="sign-blind prediction")
    for xi, a, b, c in zip(x, obs, prd, absb):
        if xi == 0:
            ax.text(xi, 0.12, "all 0.0", ha="center", fontsize=7.4)
            continue
        ax.text(xi - w / 2, a + 0.12, f"{a:.1f}", ha="center", fontsize=7.2,
                color=C_PRED)
        ax.text(xi + w / 2, b + 0.12, f"{b:.1f}", ha="center", fontsize=7.2)
        ax.text(xi + 0.34, c + 0.12, f"{c:.1f}", ha="left", fontsize=7.2, color=C_ABS)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\rho={r:g}$" for r in RHOS])
    ax.set_ylabel("directions that flip (%)")
    ax.set_ylim(0, 3.3)
    ax.legend(loc="upper left", fontsize=7.4)
    ax.set_title("(b) Signed rule: 0 vs 0, and 2 vs 4")

    ax = fig.add_subplot(gs[0, 2])
    e = Q["free"][str(0.3)]
    rows = e["flip_rows"]
    ax.barh(np.arange(len(rows)), [r["longest_identical_post_run"] for r in rows],
            color=[C_ROB if r["longest_identical_post_run"] >= 20 else C_HOT for r in rows])
    ax.axvline(20, color=C_GREY, ls="--", lw=1.0)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"dir {r['dir']}" for r in rows], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("longest identical post-flip run (tokens)")
    ax.set_xlim(0, 62)
    ax.text(0.98, 0.02, "\n".join(
        f"{r['dir']}: {r['ref_token']!r}$\\to${r['flip_token']!r} @ {r['t']}"
        for r in rows), transform=ax.transAxes, va="bottom", ha="right",
        fontsize=6.9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", lw=0.5))
    ax.set_title("(c) 2 of 4 flips re-sync: one-token substitutions")

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
        ax.plot(tg, so, color="black", lw=1.8, drawstyle="steps-post", label="observed")
        ax.plot(tg, sp, color=C_PRED, lw=1.6, ls="--", drawstyle="steps-post",
                label="signed prediction")
        for tt in (171, 219, 277):
            ax.axvline(tt, color="#777777", lw=0.7, ls=":", zorder=0)
        ax.set_xlim(0, 300)
        ylo = min(so.min(), sp.min())
        ax.set_ylim(max(0.980, ylo - 0.002), 1.0008)
        ax.set_xlabel("generated step $t$")
        if ri == 0:
            ax.set_ylabel("$S(t)=P(\\mathrm{first\\ flip}>t)$  (y zoomed)")
            ax.legend(loc="lower left")
        ax.set_title(f"$\\rho={rho:g}$: {Q['free'][str(rho)]['observed_flips']} observed "
                     f"vs {Q['free'][str(rho)]['predicted_flips_signed']} predicted",
                     fontsize=9.2)
        ax.text(0.03, 0.42,
                f"$S(171)$ {Q['free'][str(rho)]['S_at_171_obs']:.3f} / "
                f"{Q['free'][str(rho)]['S_at_171_pred']:.3f}\n"
                f"$S(277)$ {Q['free'][str(rho)]['S_at_277_obs']:.3f} / "
                f"{Q['free'][str(rho)]['S_at_277_pred']:.3f}\n"
                f"max $|\\Delta S|$ {Q['free'][str(rho)]['survival_max_abs_diff']:.4f}",
                transform=ax.transAxes, fontsize=7.0, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))

    ax = fig.add_subplot(gs[1, 2])
    ri03 = int(np.argmin(np.abs(D["fradii"] - 0.3)))
    o = D["obs"][ri03]
    fl = np.where(o >= 0)[0]
    img = (D["gen"][ri03][fl] != D["ref"][None, :]).astype(float)
    ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=1,
              extent=[0, T_ALL, len(fl) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(np.arange(len(fl)))
    ax.set_yticklabels([f"dir {u}" for u in fl], fontsize=7.4)
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("flipped direction ($\\rho=0.3$)")
    ax.set_title("(d) Where the free continuation differs (black)")
    ax.grid(False)

    fig.suptitle("phi-4 free-run validation: the model is stable at $\\rho=0.1$ (0/1000); "
                 "at $\\rho=0.3$ four one-token events are only partly predicted",
                 fontsize=11.8, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_free.png"), dpi=165)
    plt.close(fig)


def fig_layer(D, Q):
    L = np.array(Q["layer"]["states_saved"])
    blocks = np.array(Q["layer"]["blocks_saved"])
    pm = np.array(Q["layer"]["median_over_t"])
    prof = D["prof"]
    rows = np.where(np.array(L) >= 1)[0]

    fig = plt.figure(figsize=(14.6, 8.5))
    gs = GridSpec(2, 3, width_ratios=[1.15, 0.85, 0.95], hspace=0.44, wspace=0.32)

    ax = fig.add_subplot(gs[:, 0])
    Z = prof[rows]
    im = ax.imshow(np.log10(np.maximum(Z, 1e-6)), aspect="auto", origin="lower",
                   extent=[-0.5, D["T"] - 0.5, -0.5, len(rows) - 0.5],
                   cmap="magma", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.012)
    cb.set_label("$\\log_{10}(\\mathrm{median}_u\\,\\|\\delta h_t^b\\|/r)$")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"state {L[r]} (b{blocks[i]})" for i, r in enumerate(rows)],
                       fontsize=7.2)
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("saved state (block index)")
    ax.set_title("(a) Response by saved state and step:\n"
                 "growth with depth, decay with generated time")

    ax = fig.add_subplot(gs[0, 1])
    lo = np.percentile(Z, 10, axis=1)
    hi = np.percentile(Z, 90, axis=1)
    ax.fill_between(blocks, lo, hi, color=C_ROB, alpha=0.16, label="10-90% over $t$")
    ax.semilogy(blocks, pm, "-", color=C_OBS, lw=1.6, label="median over $t$")
    ax.semilogy(blocks, pm, "o", ms=4.6, color=C_ROB)
    ax.set_xlabel("block index $b$  (state $b+1$)")
    ax.set_ylabel("median $\\|\\delta h_t^b\\|/r$  (log scale)")
    ax.set_xlim(-2, 40)
    ax.legend(loc="lower right")
    ax.set_title(f"(b) {Q['layer']['total_growth_block0_to_38']:.0f}x growth over "
                 f"{Q['layer']['span_blocks']} blocks; every interval rises")

    ax = fig.add_subplot(gs[0, 2])
    gi = Q["layer"]["growth_per_block_by_interval"]
    xs = np.arange(len(gi))
    ax.bar(xs, [g["rate_per_block"] for g in gi], color=C_ROB, width=0.72)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(Q["layer"]["mean_rate_per_block"], color=C_GREY, ls="--", lw=1.0,
               label=f"mean {Q['layer']['mean_rate_per_block']:.2f}x per block")
    ax.legend(loc="upper right", fontsize=7.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g['from_block']}$\\to${g['to_block']}" for g in gi],
                       fontsize=6.9, rotation=45, ha="right")
    ax.set_xlabel("saved-state interval (block indices)")
    ax.set_ylabel("response ratio per block")
    ax.set_ylim(0.9, 2.7)
    ax.set_title("(c) Growth is fastest in the first two blocks")

    ax = fig.add_subplot(gs[1, 1:])
    rel = np.array(Q["layer"]["relative_response"])
    ax.plot(blocks, rel * 1e3, "s-", ms=4.2, color=C_HOT, lw=1.4,
            label="relative response")
    ax.set_xlabel("block index $b$  (state $b+1$)")
    ax.set_ylabel("$\\|\\delta h\\|/\\|h_0\\|$  ($\\times10^{-3}$)")
    ax.axvline(2, color=C_GREY, ls=":", lw=0.9)
    ax.legend(loc="upper right", fontsize=7.4)
    ax.set_title("(d) Normalized response peaks at block 2, then decays 2.4x: "
                 "the absolute growth is radial")

    fig.suptitle("phi-4 layer profile: 131x monotone growth over 38 blocks, but the "
                 "state-normalized response peaks at block 2 and then decays",
                 fontsize=11.4, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_layer_profile.png"), dpi=165)
    plt.close(fig)


def fig_depth(D, Q):
    q = Q["depth"]
    lam = np.array(q["arrays"]["lambda_tokens"])
    ex = np.array(q["arrays"]["ex_med"])
    exlo = np.array(q["arrays"]["ex_lo"])
    exhi = np.array(q["arrays"]["ex_hi"])
    lg = np.array(q["arrays"]["local_gain_med"])
    lglo = np.array(q["arrays"]["local_gain_lo"])
    lghi = np.array(q["arrays"]["local_gain_hi"])
    layers = np.array(q["layers"])
    n = layers.size
    fig = plt.figure(figsize=(14.4, 8.8))
    gs = GridSpec(2, 3, hspace=0.46, wspace=0.32)

    for j, toks in enumerate((q["tokens"][:4], q["tokens"][4:])):
        ax = fig.add_subplot(gs[0, j])
        for i, t in enumerate(toks):
            ax.plot(layers, lam[i], lw=1.1, label=f"t={t}")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("block index $l$")
        ax.set_ylabel("$\\lambda_i(l)$  (nats/block, symlog)")
        ax.legend(ncol=2, fontsize=7.2)
        ax.set_title(f"({'ab'[j]}) Running tangent exponent, tokens {toks[0]}-{toks[-1]}")

    ax = fig.add_subplot(gs[0, 2])
    hd = np.array(q["h_depth"])
    cols = [C_HOT if i == int(np.argmax(hd)) else C_ROB for i in range(len(hd))]
    ax.bar(np.arange(len(hd)), hd, color=cols)
    ax.set_xticks(np.arange(len(q["tokens"])))
    ax.set_xticklabels([str(t) for t in q["tokens"]], fontsize=8)
    ax.set_xlabel("probe token $t$")
    ax.set_ylabel("$h_{\\mathrm{depth}}=\\sum_i\\max(\\lambda_i,0)$  (nats)")
    ax.set_title("(c) Total tangent volume per token")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(layers, lg, "-", color=C_ROB, lw=1.4, label="tangent gain")
    ax.fill_between(layers, lglo, lghi, color=C_ROB, alpha=0.18, label="25-75%")
    bl = np.array(q["norm_state_blocks"])
    nb = np.array(q["norm_state_norms"])
    if len(bl) > 1:
        gm = (np.diff(bl) / 2 + bl[:-1]).astype(float)
        gr = (nb[1:] / nb[:-1]) ** (1.0 / np.diff(bl))
        ax.plot(gm, gr, "s--", color=C_PRED, lw=1.2, ms=4,
                label="state-norm growth / block")
    ax.set_yscale("log")
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("growth factor per block  (log scale)")
    ax.legend(loc="upper right", fontsize=7.1)
    ax.set_title("(d) Tangent gain vs state-norm growth")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(layers, ex, "-", color=C_GREY, lw=1.2, label="absolute $e_l$")
    ax.fill_between(layers, exlo, exhi, color=C_GREY, alpha=0.15)
    rk = list(q["relative_chunks"].keys())
    rv = [q["relative_chunks"][k]["median"] for k in rk]
    rx = [(int(k.split("->")[0]) + int(k.split("->")[1])) / 2 for k in rk]
    ax.bar(rx, rv, width=[int(k.split("->")[1]) - int(k.split("->")[0]) for k in rk],
           color=C_ROB, alpha=0.78, label="relative (tangent - norm)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("log gain per block (nats/block)")
    ax.set_ylim(-0.9, 1.2)
    ax.legend(loc="upper right", fontsize=7.2)
    ax.set_title(f"(e) Relative exponent: {q['relative_n_negative']}/{q['relative_n_cells']} "
                 "intervals contract")

    ax = fig.add_subplot(gs[1, 2])
    fs = np.array(q["final_spectrum"])
    for i, t in enumerate(q["tokens"]):
        ax.plot(np.arange(1, fs.shape[1] + 1), fs[i], "-o", ms=3.0, lw=1.0,
                color=C_HOT if i == int(np.argmax(hd)) else "#888888")
    ax.set_xlabel("tangent direction $i$ (1..$k$)")
    ax.set_ylabel("final $\\lambda_i$  (nats/block)")
    ax.set_title("(f) Final spectrum: flat across the $k$ directions")

    verdict = ("contracts" if q["relative_n_negative"] >= 0.75 * q["relative_n_cells"]
               else "mostly expands")
    vg = np.array(q["tangent_volume_growth"])
    fig.suptitle(f"phi-4 depth: the tangent volume grows {vg.min():.1f}-{vg.max():.0f}x over "
                 f"40 blocks, but the state-normalized tangent {verdict} in "
                 f"{q['relative_n_negative']}/{q['relative_n_cells']} intervals",
                 fontsize=11.2, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_depth.png"), dpi=165)
    plt.close(fig)


def fig_nonnorm(D, Q):
    q = Q["nonnorm"]
    sel = np.array(q["sel_layers"])
    full = np.array(q["full_layers"])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    ax = axes[0, 0]
    ax.semilogy(sel, q["mlp_smax_med"], "o-", color=C_ROB, lw=1.3, ms=4.2)
    ax.fill_between(sel, q["mlp_smax_lo"], q["mlp_smax_hi"], color=C_ROB, alpha=0.16)
    ax.semilogy(sel, q["mlp_rho_med"], "s--", color=C_PRED, lw=1.2, ms=3.8)
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("gain (log scale)")
    ax.set_title("(a) MLP Jacobian: $\\sigma_{\\max}$ and $\\rho$")

    ax = axes[0, 1]
    ax.plot(sel, q["mlp_kappa_med"], "o-", color=C_ROB, lw=1.3, ms=4.0, label="MLP")
    ax.plot(full, q["full_kappa_med"], "s--", color=C_HOT, lw=1.4, ms=5.0,
            label="full block")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("block index $l$")
    ax.set_ylabel("$\\kappa=\\sigma_{\\max}/\\rho$")
    ax.legend()
    ax.set_title("(b) Non-normality $\\kappa$ by block")

    ax = axes[1, 0]
    rr = np.array(q["full_rho_at_full"]).ravel()
    ss = np.array(q["full_smax_at_full"]).ravel()
    ax.scatter(rr, ss, s=16, color=C_ABS, alpha=0.7)
    gg = np.geomspace(rr.min() * 0.7, rr.max() * 1.4, 20)
    for kap in (1, 3, 10):
        ax.plot(gg, kap * gg, ls=":", color=C_GREY, lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("spectral radius $\\rho$  (log scale)")
    ax.set_ylabel("$\\sigma_{\\max}$  (log scale)")
    ax.set_title("(c) $\\sigma_{\\max}$ against $\\rho$ (lines $\\kappa=1,3,10$)")

    ax = axes[1, 1]
    ax.hist(q["angles"], bins=np.linspace(65, 91, 40), color=C_ROB, alpha=0.8)
    ax.axvline(q["angle_median"], color=C_HOT, lw=1.4, label="this run")
    ax.axvline(83.6, color=C_PRED, ls="--", lw=1.1, label="1.5B 83.6 deg")
    ax.axvline(86.96, color=C_ABS, ls=":", lw=1.1, label="14B 86.96 deg")
    ax.set_xlabel("angle between top singular directions, blocks 4 apart")
    ax.set_ylabel("block pairs")
    ax.legend(loc="upper left", fontsize=7.4)
    ax.set_title(f"(d) Principal-angle median {q['angle_median']:.1f} deg")

    fig.suptitle("phi-4 non-normality (pending run): U-shaped gain and near-orthogonal "
                 "amplifying directions", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_nonnorm.png"), dpi=165)
    plt.close(fig)


def fig_state(D, Q):
    q = Q["state"]
    steps = np.array(q["steps"])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    ax = axes[0, 0]
    cols = {0.1: C_ROB, 0.03: C_PRED}
    for rv, vals in q["immediate_by_radius"].items():
        rvf = float(rv)
        if rvf == 0.1:
            ax.semilogy(steps, vals, "o-", color=C_ROB, lw=1.3, ms=4.5,
                        label=f"$r={rvf:g}$")
        else:
            ax.semilogy(steps, vals, "^", color=C_PRED, ms=6.5, mfc="none",
                        mew=1.2, label=f"$r={rvf:g}$")
    ax.set_xlabel("kick position $s$")
    ax.set_ylabel("$\\|\\delta h^{\\mathrm{last}}_s\\|/r$  (log scale)")
    ax.legend(loc="upper right")
    ax.set_title("(a) Immediate response at the kick position")

    ax = axes[0, 1]
    hl = np.array([np.nan if v is None else v for v in q["halflife"]], float)
    ax.bar(np.arange(len(steps)), hl, color=C_ROB, alpha=0.8,
           label="half-life (steps)")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=7.5)
    ax.set_xlabel("kick position $s$")
    ax.set_ylabel("median half-life (steps)")
    ax2 = ax.twinx()
    lr = np.array([np.nan if v is None else v for v in q["late_ratio"]], float)
    ax2.semilogy(np.arange(len(steps)), lr, "^--", color=C_HOT, ms=5)
    ax2.axhline(1.0, color=C_HOT, ls=":", lw=0.9)
    ax2.set_ylabel("late / immediate", color=C_HOT)
    ax2.tick_params(axis="y", colors=C_HOT)
    ax2.grid(False)
    ax.set_title("(b) Initial half-life and late/immediate ratio")

    ax = axes[1, 0]
    ax.bar(np.arange(len(steps)), 100 * np.array(q["flip_frac"]), color=C_ABS, alpha=0.85,
           label="directions that flip at $r\\leq1$")
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=7.5)
    ax.set_xlabel("kick position $s$")
    ax.set_ylabel("% of 24 directions that flip at $r\\leq1$")
    ax.set_ylim(0, max(10, 110 * max(q["flip_frac"])))
    ax.set_ylim(0, 9.5)
    ax2 = ax.twinx()
    rsmed = np.array([np.nan if s >= 299 else v for s, v in
                      zip(steps, q["rstar_median"])], float)
    rsmin = np.array([np.nan if s >= 299 else v for s, v in
                      zip(steps, q["rstar_min"])], float)
    ax2.semilogy(np.arange(len(steps)), rsmed, "o--", color=C_HOT, ms=4.5, lw=1.1,
                 label="median $r_*$")
    ax2.plot(np.arange(len(steps)), rsmin, "x", color=C_HOT, ms=6.5,
             label="min $r_*$")
    ax2.set_ylim(0.5, 20)
    ax2.set_ylabel("median / min $r_*$ (red)", color=C_HOT)
    ax2.tick_params(axis="y", colors=C_HOT)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.2)
    ax.set_title("(c) Only $s=250$ reaches a gate: 2/24 flip at $t=277$")

    ax = axes[1, 1]
    for s in (8, 100, 250):
        tr = np.array(q["trajectories"][str(int(s))])
        ax.semilogy(np.arange(len(tr)), tr, lw=1.2, label=f"s={int(s)}")
        ax.axvline(s, color=C_GREY, ls=":", lw=0.8)
    for tt, lab in ((219, "t=219"), (277, "t=277")):
        ax.axvline(tt, color=C_ABS, ls="--", lw=0.9)
        ax.text(tt + 2, 0.012, lab, fontsize=7.0, color=C_ABS, rotation=90,
                va="bottom")
    ax.set_xlabel("generated step $t$")
    ax.set_ylabel("$\\|\\delta h^{\\mathrm{last}}_t\\|$  (log scale)")
    ax.set_ylim(7e-3, 1.5e1)
    ax.legend()
    ax.set_title("(d) Response trajectories")

    fig.suptitle("phi-4 state kicks: a one-position kick decays with a 1-4.5-step half-life; "
                 "only the s=250 kick survives to the t=277 tie (2/24 directions)",
                 fontsize=11.2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_state.png"), dpi=165)
    plt.close(fig)


def fig_generality(D, Q):
    g = Q["generality"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    cols = {"interstellar": C_OBS, "childhood": C_ROB, "cyber": C_PRED}
    ax = axes[0, 0]
    for name in ("interstellar", "childhood", "cyber"):
        if not isinstance(g.get(name), dict):
            continue
        xs, ys = ecdf(g[name]["m"])
        ax.plot(xs, ys, color=cols[name], lw=1.7,
                label=f"{name} (n={g[name]['K']})")
    ax.set_xscale("log")
    ax.set_xlabel("baseline margin $m_t$  (nats, log scale)")
    ax.set_ylabel("fraction of steps with $m_t\\leq m$")
    ax.legend()
    ax.set_title("(a) Margin spectra: do the gates move?")

    ax = axes[0, 1]
    for name in ("interstellar", "childhood", "cyber"):
        if not isinstance(g.get(name), dict):
            continue
        xs, ys = ecdf(g[name]["rstar"])
        ax.plot(xs, ys, color=cols[name], lw=1.7, label=f"{name} (n={g[name]['K']})")
    ax.axvline(R_RUN, color=C_GREY, ls="--", lw=1.0)
    ax.text(R_RUN * 1.07, 0.04, "run radius", fontsize=7.0, color=C_GREY)
    ax.set_xscale("log")
    ax.set_xlabel("flux-tube radius $r_*$  (log scale)")
    ax.set_ylabel("fraction of directions with $r_*\\leq r$")
    ax.legend()
    ax.set_title("(b) Threshold radii across prompts")

    ax = axes[1, 0]
    names = [n for n in ("interstellar", "childhood", "cyber")
             if isinstance(g.get(n), dict)]
    for i, n in enumerate(names):
        q10 = g[n]["rstar_q"]["p10"]
        q25 = g[n]["rstar_q"]["p25"]
        q50 = g[n]["rstar_q"]["p50"]
        q75 = g[n]["rstar_q"]["p75"]
        q90 = g[n]["rstar_q"]["p90"]
        ax.plot([q10, q90], [i, i], "-", color=cols[n], lw=2.0, alpha=0.55)
        ax.plot([q25, q75], [i, i], "-", color=cols[n], lw=6.0, alpha=0.8)
        ax.plot(q50, i, "o", color="white", markeredgecolor=cols[n], ms=7, zorder=5)
        ax.text(q90 * 1.08, i, f"p50 {q50:.2f}", va="center", fontsize=7.4,
                color=cols[n])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([f"{n}\n(n={g[n]['K']})" for n in names], fontsize=7.8)
    ax.set_xscale("log")
    ax.set_xlabel("$r_*$ (log scale); thick = 25-75%, thin = 10-90%")
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_title("(c) $r_*$ quantiles per prompt")

    ax = axes[1, 1]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [100 * g[n]["flip_frac_at_0p1"] for n in names], 0.34,
           color=[cols[n] for n in names], alpha=0.9, label="at $r=0.1$")
    ax.bar(x + 0.18, [100 * (g[n]["rstar"] <= 0.3).mean() for n in names], 0.34,
           color=[cols[n] for n in names], alpha=0.45, label="at $r=0.3$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={g[n]['K']})" for n in names], fontsize=8)
    ax.set_ylabel("directions that flip (%)")
    ax.set_ylim(0, 1.35)
    for xi, n in zip(x, names):
        ax.text(xi - 0.18, 0.03, "0", ha="center", fontsize=7.4)
        ax.text(xi + 0.18, 100 * (g[n]["rstar"] <= 0.3).mean() + 0.03,
                f"{100 * (g[n]['rstar'] <= 0.3).mean():.1f}", ha="center", fontsize=7.4)
    ax.legend(fontsize=7.4)
    ax.set_title("(d) 0 flips at $r=0.1$; 0.2% and 1% at $r=0.3$")

    fig.suptitle("phi-4 multi-prompt generality: the margin spectrum barely moves, but the "
                 "steps and directions that control the gates are entirely prompt-specific",
                 fontsize=11.2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIGDIR, "phi4_generality.png"), dpi=165)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    D = load()
    if not D["present"]["a1"]:
        raise SystemExit("a1_inter_w0/a1.npz is required")
    Q = compute(D)

    # arrays the figures need but the JSON should not carry
    dhl = D["dh_last"]
    nrm = np.linalg.norm(dhl, axis=2)
    h0 = D["h0_last"]
    nh = np.linalg.norm(h0, axis=1)
    c = np.einsum("utd,td->ut", dhl, h0) / (np.maximum(nrm, 1e-30) * nh[None, :])
    Q["angle"]["_cos"] = c
    Q["angle"]["_ang"] = np.degrees(np.arccos(np.clip(c, -1, 1)))

    if D["present"]["calib"]:
        fig_calib(D, Q)
    fig_core(D, Q)
    if D["present"]["free"]:
        fig_free(D, Q)
    fig_layer(D, Q)
    if D["present"]["depth"]:
        fig_depth(D, Q)
    if D["present"]["nonnorm"]:
        fig_nonnorm(D, Q)
    if D["present"]["state"]:
        fig_state(D, Q)
    if D["present"]["childhood"] or D["present"]["cyber"]:
        fig_generality(D, Q)

    out = {
        "protocol": {
            "model": "phi-4 (Phi3ForCausalLM), FP32",
            "blocks": N_BLOCKS, "hidden": HIDDEN, "vocab": VOCAB,
            "attention": "full attention; sliding window = None (documented null)",
            "sliding_window": None, "window_field": 0,
            "prompt": "interstellar_travel, P=64 tokens",
            "T": T_ALL, "K": D["K"], "r": R_RUN,
            "saved_states": D["states"].tolist(),
            "saved_blocks": [int(s - 1) for s in D["states"] if s >= 1],
            "final_norm_state": "state 40 (block 39); only dh_last/h0_last, 24 dirs",
            "files": {"calib": "results_big/phi4/calib/calib.npz",
                      "a1": "results_big/phi4/a1_inter_w0/a1.npz",
                      "free": "results_big/phi4/free_inter_w0/free_run.npz",
                      "state": "results_big/phi4/state_w0/state2.npz",
                      "depth": "results_big/phi4/depth_w0/depth.npz",
                      "childhood": "results_big/phi4/a1_childhood_w0/a1.npz",
                      "cyber": "results_big/phi4/a1_cyber_w0/a1.npz"},
            "depth_tokens": [0, 21, 50, 100, 150, 200, 250, 299],
            "depth_tokens_include_ties": False,
            "present": D["present"],
        },
        "pending": Q["pending"],
        "margins": Q["margins"],
        "calibration": Q["calibration"],
        "calibration_margins": Q["cal_margins"],
        "rstar": Q["rstar"],
        "ladder": Q["ladder"],
        "ladder_n_distinct": Q["ladder_n_distinct"],
        "ladder_n_distinct_flip": Q["ladder_n_distinct_flip"],
        "decomposition": Q["decomposition"],
        "decomposition_ref": Q["decomposition_ref"],
        "layer_profile": Q["layer"],
        "gain": Q["gain"],
        "pairwise": Q["pairwise"],
        "angle": {k: v for k, v in Q["angle"].items() if not k.startswith("_")},
        "free": Q["free"],
        "depth": Q.get("depth", "pending"),
        "nonnorm": Q.get("nonnorm", "pending"),
        "state": Q.get("state", "pending"),
        "generality": Q.get("generality", "pending"),
        "cross_model": {
            "1.5B": {
                "blocks": 28, "hidden": 1536, "K": 20, "r": 0.1,
                "margin_min": 2.015686e-2, "margin_min_step": 123,
                "margin_median": 4.778, "rstar_p50": 1.555,
                "rstar_p90": 4.124, "gain_p50": 1.333,
                "gain_range": "0.635-7.468 (11.8x)",
                "floor": 1.0e-3, "clean_window": [1e-3, 0.1],
                "flip_fraction_at_0p1": 0.0,
                "free_flip_fraction_0p1_0p3": [0.0, None],
                "angle_deg": 87.0,
            },
            "DS-R1-14B": {
                "blocks": 48, "hidden": 5120, "K": 1000, "r": 0.1,
                "margin_min": 9.727478e-5, "margin_min_step": 21,
                "margin_median": 3.096, "rstar_p50": 0.116,
                "rstar_p90": 2.593, "gain_p50": 0.583,
                "gain_range": "0.224-3.064 (13.7x)",
                "floor": 2.7e-3, "clean_window": [0.03, 0.3],
                "flip_fraction_at_0p1": 0.453,
                "free_flip_fraction_0p1_0p3": [0.453, 0.643],
                "angle_deg": None,
            },
            "Qwen3.5-9B": {
                "blocks": 32, "hidden": 4096, "K": 1000, "r": 0.1,
                "margin_min": 7.27e-3, "margin_min_step": 267,
                "margin_median": 2.461, "rstar_p50": 0.707,
                "rstar_p90": 1.956, "gain_p50": 0.822,
                "gain_range": "0.361-5.449 (15.1x)",
                "floor": 6.1e-4, "clean_window": [0.01, 0.3],
                "flip_fraction_at_0p1": 0.012,
                "free_flip_fraction_0p1_0p3": [0.012, 0.157],
                "angle_deg": 90.08,
            },
            "phi-4": {
                "blocks": N_BLOCKS, "hidden": HIDDEN, "K": D["K"], "r": R_RUN,
                "margin_min": Q["margins"]["min"],
                "margin_min_step": Q["margins"]["min_step"],
                "margin_median": Q["margins"]["p50"],
                "rstar_p50": Q["rstar"]["p50"], "rstar_p90": Q["rstar"]["p90"],
                "gain_p50": Q["gain"]["p50"],
                "gain_range": f"{Q['gain']['min']:.3f}-{Q['gain']['max']:.3f} "
                              f"({Q['gain']['max_over_min']:.1f}x)",
                "floor": Q["calibration"]["floor_last_state"],
                "clean_window": list(CLEAN),
                "flip_fraction_at_0p1": Q["rstar"]["flip_frac_at_0p1"],
                "free_flip_fraction_0p1_0p3": [
                    Q["free"][str(r)]["observed_flip_fraction"] for r in RHOS],
                "angle_deg": Q["angle"]["median_angle"],
            },
        },
        "caveats": [
            "states l is the output of block l-1; only 14 states are saved "
            "(0,1,2,3,5,8,12,16,20,24,28,32,36,39), so the layer profile and the "
            "relative depth exponent are measured on 12 intervals, not every block; "
            "block 39 (state 40) is only available as dh_last for 24 directions.",
            "the run log quoted a calibration median of ~1.2-1.5; that is the 25th "
            "percentile (1.36), not the median (3.78). The file is used throughout.",
            "the stored r_c/pred_T arrays use |dm| (sign-blind); the scorecard uses the "
            "signed rule recomputed from m and dm.",
            "only 4 free-run events exist at rho=0.3 and none at rho=0.1, so the "
            "scorecard's step-level accuracy is a small-sample statement.",
            "the response-vs-state angle uses 24 directions at the final-norm state and "
            "is noisier here (IQR ~8 deg) because ||dh||/||h0|| is only ~1e-4.",
            "phi-4 has no sliding window: window experiments are undefined (null by "
            "construction), so no window comparison is made.",
            "the depth tokens are 0,21,50,100,150,200,250,299; neither phi-4 tie "
            "(t=219, t=277) is in the list, so the 'is the tie token a tangent hotspot' "
            "question is unanswered by this run and needs a re-run with the tie tokens.",
            "the state-kick flip count uses r_c<=1 recomputed from the baseline m and "
            "the r=0.1 kick response; at the run radius r=0.1 itself no direction in "
            "any of the 10x24 kicks flips anything.",
            "the multiprompt runs use the same seed-69 direction generator as the "
            "1000-direction interstellar run, so their 200 directions are the first "
            "200 of that generator and are directly comparable.",
        ],
    }
    json.dump(jf(out), open(OUTJSON, "w", encoding="utf-8"), indent=2)

    print(json.dumps(jf({
        "present": D["present"], "pending": Q["pending"],
        "margins": Q["margins"], "cal": Q["calibration"],
        "rstar": Q["rstar"],
        "decomposition": {k: Q["decomposition"][k] for k in
                          ("median_mstar", "median_sensitivity", "median_rstar",
                           "median_log10_rstar")},
        "layer": {k: Q["layer"][k] for k in
                  ("total_growth_block0_to_38", "mean_rate_per_block",
                   "monotone_intervals", "n_intervals",
                   "state_norm_growth_block0_to_38")},
        "gain": Q["gain"],
        "angle": {k: Q["angle"][k] for k in
                  ("median_angle", "frac_angle_gt_90", "dh_over_h0_median")},
        "free": Q["free"],
    }), indent=1))


if __name__ == "__main__":
    main()
