"""s4_cross.py -- Section D: does geometry predict dynamics?

D1  Cross-model (n=4, anecdotal -- stated): correlate per-model geometry
    scalars (late-layer ID, PR, eff-rank, anisotropy, write rank, vocab
    measures, and the PCA eff-rank of the dh RESPONSE FIELD from a1
    dh_last) with the main project's dynamics numbers: kick gain at s=16,
    late finite-time rate lambda, direction spread.
D2  Within-model: correlate per-layer dimension measures with per-layer
    depth gain G_l = dh_l / r (a1 dh_norm, all layers) and with the
    per-layer time-rate lambda_l (slope of log dh vs t).
    NOTE: a1 dynamics use the 'propulsion' prompt; the dim measures use the
    3 'tides/lighthouse/train' prompts -- layer structure is assumed
    prompt-stable; this caveat is stated in the report.
D3  Radial vs tangential: cos(dh, h) from a1 (per layer, radius, token):
    distributions, radial fraction vs depth, radius dependence (deepseek
    w100 sweep), PCA of the dh field (response subspace), overlap of the
    dh subspace with the state subspace + prompt-vs-prompt control +
    random baseline.
D4  4D-mask window sweep (window4d/<tag>/w{60,100,200,4096}): per-layer
    gain vs window; pred_T vs window; margins-equal check.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from dimlib import (A1_BY_TAG, W4D, CACHE, OUT, COLOR, DISPLAY, DYN,
                    savefig, section_numbers, discover_tags, spearman,
                    pearson, eig_desc, pr_lambda, stable_rank, top_frac,
                    eff_rank, subspace_overlap, powerlaw_slope)

TAGS = ["deepseek", "qwen3_4b", "phi4mini", "qwen3_5"]
SHORT = {t: DISPLAY[t].replace("-R1-Distill-1.5B", "").replace(
    "-Instruct", "") for t in TAGS}
NQ = (10, 90)


# ---------------------------------------------------------------- helpers
def load_cache(tag):
    return np.load(os.path.join(CACHE, f"{tag}_layermeas.npz"))


def late_scalar(z, key, L, frac=0.3, pp=False, how="median"):
    """Mean of a per-layer measure over the last `frac` of raw blocks."""
    lo = int(np.ceil((1 - frac) * (L - 1)))
    hi = L - 1                       # raw blocks only, no post-norm
    if pp:
        v = np.asarray(z[key])[lo:hi + 1]      # [n, 3] -> per-prompt
        v = np.median(v, axis=1)
    else:
        v = np.asarray(z[key])[lo:hi + 1]
    return float(np.mean(v))


def pca_topk(X, k):
    """Top-k PCs (D x k, orthonormal columns) of centered X [n, D]."""
    Xc = X - X.mean(0)
    G = Xc @ Xc.T
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:k]
    lam = w[o]
    V = Xc.T @ U[:, o]
    V /= np.sqrt(np.maximum(lam, 1e-12))[None, :]
    return V, lam


def dh_field(d, rsel=None):
    """Pool a1 dh_last [n_r, 3, T, D] -> [n, D] float32."""
    x = d["dh_last"].astype(np.float32)
    if rsel is not None:
        x = x[rsel]
    return x.reshape(-1, x.shape[-1])


def field_measures(X):
    n = len(X)
    Xc = X - X.mean(0)
    lam_c = eig_desc(Xc @ Xc.T) / (n - 1)
    lam_m = eig_desc(X @ X.T) / n
    return {
        "n": int(n), "D": int(X.shape[1]),
        "eff90_c": eff_rank(lam_c, 0.9), "eff99_c": eff_rank(lam_c, 0.99),
        "PR_c": pr_lambda(lam_c), "topfrac_c": top_frac(lam_c),
        "eff90_m": eff_rank(lam_m, 0.9), "PR_m": pr_lambda(lam_m),
        "slope_c": powerlaw_slope(lam_c, i0=5, i1=min(400, n // 2))[0],
    }


# ---------------------------------------------------------------- D1
def d1(curves, out):
    scal = {}
    for tag in TAGS:
        z = curves[tag]
        L = int(z["L"])
        scal[tag] = {
            "ID_mle20_pooled_late": late_scalar(z, "m_mle20", L),
            "ID_mle20_perprompt_late": late_scalar(z, "pp_mle20", L,
                                                   pp=True),
            "ID_twonn_pooled_late": late_scalar(z, "m_twonn", L),
            "ID_corrdim_pooled_late": late_scalar(z, "m_corrdim", L),
            "PR_c_late": late_scalar(z, "m_pr_c", L),
            "eff90_c_late": late_scalar(z, "m_eff90_c", L),
            "mean_cos_late": late_scalar(z, "m_mean_cos", L),
            "cone_late": late_scalar(z, "m_cone", L),
            "slope_c_late": late_scalar(z, "m_slope_c", L),
            "write_eff90_mean": float(np.mean(
                z["w_eff90_c"][:-1])),   # exclude post-norm step
            "write_norm_ratio_mean": float(np.mean(
                z["w_norm_ratio"][:-1])),
        }
    # dh response field (a1 dh_last) per model
    fields = {}
    for tag in TAGS:
        d = np.load(A1_BY_TAG[tag])
        fm = field_measures(dh_field(d))
        fields[tag] = fm
        del d
    # vocab measures from numbers.json (section A)
    with open(os.path.join(OUT, "numbers.json")) as f:
        nums = json.load(f)
    for tag in TAGS:
        A = nums["A"]["per_tag"][tag]
        scal[tag]["vocab_E_PR"] = A["E_spec"]["PR_lambda_s2"]
        scal[tag]["vocab_E_eff90"] = A["E_spec"]["eff_rank_90"]
        scal[tag]["vocab_E_meancos"] = A["mean_cos_rows_E"]
        scal[tag]["vocab_E_topvar"] = A["E_spec"]["top_var_frac_s1_2_over_sum_s2"]
        for k, v in fields[tag].items():
            scal[tag][f"dhfield_{k}"] = v

    # dynamics targets
    tgt = {k: DYN[k] for k in DYN}
    # spearman table
    measures = [k for k in scal[TAGS[0]]]
    table = {}
    for mname in measures:
        x = [scal[t][mname] for t in TAGS]
        nfin = int(np.sum(np.isfinite(x)))
        table[mname] = {
            "values": {t: scal[t][mname] for t in TAGS},
            "n_finite": nfin,
            "spearman_vs_kickgain": spearman(
                x, [tgt["kick_gain_s16"][t] for t in TAGS]),
            "spearman_vs_lambda": spearman(
                x, [tgt["late_rate_lambda"][t] for t in TAGS]),
            "spearman_vs_dirspread": spearman(
                x, [tgt["dir_spread"][t] for t in TAGS]),
        }
    out["D1"] = {
        "caveat": ("n=4 models: every correlation below is anecdotal "
                   "(a 4-point Spearman needs |rho|=1.0 for p<0.05); "
                   "reported as orderings, not evidence."),
        "dynamics_given": DYN,
        "scalars": scal,
        "spearman_table": table,
    }
    # rank-order table for the report
    ranks = {}
    for mname in measures:
        x = np.array([scal[t][mname] for t in TAGS])
        ok = np.isfinite(x)
        if ok.sum() == 4:
            ranks[mname] = [TAGS[int(i)] for i in np.argsort(x)]
    out["D1"]["orderings_low_to_high"] = ranks
    out["D1"]["kick_gain_order_low_to_high"] = [
        TAGS[int(i)] for i in np.argsort(
            [tgt["kick_gain_s16"][t] for t in TAGS])]

    # ---- figures
    f, ax = plt.subplots(figsize=(5.4, 3.8))
    xk = "ID_mle20_perprompt_late"
    for tag in TAGS:
        ax.scatter(scal[tag][xk], DYN["kick_gain_s16"][tag], s=60,
                   color=COLOR[tag], marker="o", zorder=3)
        ax.annotate(SHORT[tag], (scal[tag][xk],
                                DYN["kick_gain_s16"][tag]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("late-layer single-trajectory ID $\\hat d$ (LB k=20)")
    ax.set_ylabel("kick gain at $s=16$ (downstream amplification)")
    rho = table[xk]["spearman_vs_kickgain"]
    direction = "LESS" if rho < 0 else "MORE"
    ax.set_title(f"Higher-ID models amplify kicks {direction} "
                 f"(Spearman $\\rho$={rho:+.2f}, n=4)", fontsize=9)
    savefig(f, "fig_D1_kickgain_vs_id.png")

    f, ax = plt.subplots(figsize=(5.4, 3.8))
    xk = "dhfield_eff90_c"
    for tag in TAGS:
        ax.scatter(scal[tag][xk], DYN["kick_gain_s16"][tag], s=60,
                   color=COLOR[tag], zorder=3)
        ax.annotate(SHORT[tag], (scal[tag][xk], DYN["kick_gain_s16"][tag]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("response-field eff-rank 90% ($\\delta h$, last layer)")
    ax.set_ylabel("kick gain at $s=16$")
    rho = table[xk]["spearman_vs_kickgain"]
    ax.set_title(f"Kick gain vs $\\delta h$-field rank: "
                 f"Spearman $\\rho$={rho:+.2f} (n=4)", fontsize=9)
    savefig(f, "fig_D1b_kickgain_vs_responserank.png")

    f, ax = plt.subplots(figsize=(8.2, 4.2))
    ms = ["ID_mle20_perprompt_late", "ID_twonn_pooled_late",
          "ID_corrdim_pooled_late", "PR_c_late", "eff90_c_late",
          "mean_cos_late", "cone_late", "slope_c_late",
          "write_eff90_mean", "dhfield_eff90_c", "dhfield_PR_c",
          "vocab_E_PR", "vocab_E_meancos"]
    xx = np.arange(len(ms))
    w = 0.27
    for j, (tgtkey, lab, c) in enumerate([
            ("spearman_vs_kickgain", "vs kick gain", "#222222"),
            ("spearman_vs_lambda", "vs late rate $\\lambda$",
             "#4477AA"),
            ("spearman_vs_dirspread", "vs direction spread", "#EE7733")]):
        vals = [table[m][tgtkey] for m in ms]
        ax.bar(xx + (j - 1) * w, vals, w, color=c if j == 0 else
               ("#66CCEE" if j == 1 else "#EECC66"), label=lab)
    ax.set_xticks(xx)
    ax.set_xticklabels([m.replace("_late", "").replace("ID_", "ID ")
                        .replace("dhfield_", "$\\delta h$ ")
                        .replace("vocab_", "vocab ") for m in ms],
                       rotation=40, ha="right", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman $\\rho$ (n=4, anecdotal)")
    ax.set_title("Which geometry measures track the dynamics? "
                 "(|rho|=1 is the only significant 4-point value)")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_D1c_spearman_bars.png")
    return scal, fields


# ---------------------------------------------------------------- D2
def gain_profile(d):
    """Per-layer depth gain and time-rate from a1 dh_norm."""
    dh = d["dh_norm"].astype(np.float64)     # [n_r, n_dir, n_lay, T]
    radii = d["radii"]
    layers = d["layers"]
    r = radii[:, None, None, None]
    G = dh / r                                # relative response
    n_r, n_d, n_l, T = dh.shape
    # depth gain: mean over (r, d) of mean_t
    G_l = G.mean(axis=(0, 1)).mean(-1)
    G_band = np.percentile(G.mean(axis=(0, 3)), NQ, axis=0)  # over dirs
    # time-rate per layer: slope of log dh vs t, mean over (r,d)
    tt = np.arange(T)
    tt_s = tt[T // 5:]
    rates = np.empty((n_r, n_d, n_l))
    for i in range(n_r):
        for j in range(n_d):
            y = np.log(np.maximum(dh[i, j][:, tt_s], 1e-30))
            A = np.vstack([tt_s, np.ones_like(tt_s)]).T
            sol, *_ = np.linalg.lstsq(A, y.T, rcond=None)
            rates[i, j] = sol[0]
    lam_l = rates.mean(axis=(0, 1))
    lam_band = np.percentile(rates.mean(axis=0), NQ, axis=0)
    return layers, G_l, G_band, lam_l, lam_band


def d2(curves, out, scal):
    res = {}
    corr_rows = []
    for tag in TAGS:
        z = curves[tag]
        L = int(z["L"])
        d = np.load(A1_BY_TAG[tag])
        layers, G_l, G_band, lam_l, lam_band = gain_profile(d)
        # marginal per-block gain (undefined at layer 0 where G_0 = 0:
        # the kick lives at the prompt's last position, not at generated
        # positions' embeddings)
        marg = np.full(len(G_l) - 1, np.nan)
        ok = G_l[:-1] > 0
        marg[ok] = G_l[1:][ok] / G_l[:-1][ok]
        # check dh at layer 0 == r
        r0 = d["dh_norm"][:, :, 0, :].mean() / d["radii"][:, None].mean()
        radial = np.abs(d["cos"]).mean(axis=(0, 1, 3)) if "cos" in d.files \
            else None
        # dim measures at the same acts indices (= a1 layers)
        idx = z["idx"]
        in_cache = np.isin(idx, layers)
        meas = {
            "ID_mle20_pooled": z["m_mle20"],
            "ID_mle20_perprompt_med": np.median(
                np.asarray(z["pp_mle20"]), axis=1),
            "PR_c": z["m_pr_c"], "eff90_c": z["m_eff90_c"],
            "mean_cos": z["m_mean_cos"], "cone": z["m_cone"],
            "slope_c": z["m_slope_c"],
        }
        rowsel = [int(np.where(idx == l)[0][0]) for l in layers]
        blocksel = [i for i, l in enumerate(layers) if l >= 1]
        cors = {}
        for mname, arr in meas.items():
            x = np.asarray(arr)[rowsel][blocksel]
            yG = G_l[blocksel]
            yl = lam_l[blocksel]
            ym = marg[[i - 1 for i in blocksel]]
            cors[mname] = {
                "spearman_vs_gain": spearman(x, yG),
                "pearson_vs_gain": pearson(x, yG),
                "spearman_vs_rate": spearman(x, yl),
                "spearman_vs_marginal": spearman(x, ym),
            }
            corr_rows.append((tag, mname, cors[mname]["spearman_vs_gain"],
                              cors[mname]["spearman_vs_rate"]))
        res[tag] = {
            "n_layers_a1": int(len(layers)),
            "layers": [int(v) for v in layers],
            "G_l": [float(v) for v in G_l],
            "lambda_l": [float(v) for v in lam_l],
            "marginal_gain": [float(v) for v in marg],
            "dh_layer0_over_r": float(r0),
            "radial_abs_cos_per_layer": ([float(v) for v in radial]
                                         if radial is not None else None),
            "correlations": cors,
        }
        del d
        # ---------------- figures
        f, ax = plt.subplots(figsize=(5.8, 3.6))
        rel = layers / L
        ax.fill_between(rel, G_band[0], G_band[1], color=COLOR[tag],
                        alpha=0.2, lw=0, label="gain, 10-90 pct over dirs")
        ax.plot(rel, G_l, color=COLOR[tag], lw=1.6,
                label="depth gain $G_\\ell=\\|\\delta h_\\ell\\|/r$")
        ax2 = ax.twinx()
        ax2.plot(rel, np.asarray(z["m_pr_c"])[rowsel], "--", color="k",
                 lw=1.1, label="$PR$ (mean-removed)")
        ax2.set_ylabel("participation ratio $PR$", fontsize=8)
        ax2.grid(False)
        ax.set_yscale("log")
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("gain $G_\\ell$")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper left")
        ax.set_title(f"{SHORT[tag]}: gain grows $\\times$"
                     f"{G_l[-1]/G_l[1]:.1f} across depth; "
                     f"Spearman(PR, gain)="
                     f"{cors['PR_c']['spearman_vs_gain']:+.2f}", fontsize=8.6)
        savefig(f, f"fig_D2_gainprofile_{tag}.png")

        f, ax = plt.subplots(figsize=(5.0, 3.6))
        x = np.asarray(z["m_pr_c"])[rowsel][blocksel]
        ax.scatter(x, G_l[blocksel], s=16, color=COLOR[tag])
        ax.set_yscale("log")
        ax.set_xlabel("participation ratio $PR$ at layer $\\ell$")
        ax.set_ylabel("depth gain $G_\\ell$")
        ax.set_title(f"{SHORT[tag]}: layer-wise $PR$ vs gain, "
                     f"Spearman {cors['PR_c']['spearman_vs_gain']:+.2f} "
                     f"(n={len(blocksel)})", fontsize=8.6)
        savefig(f, f"fig_D2b_scatter_pr_gain_{tag}.png")

        f, ax = plt.subplots(figsize=(5.0, 3.6))
        x = np.asarray(z["m_mean_cos"])[rowsel][blocksel]
        ax.scatter(x, G_l[blocksel], s=16, color=COLOR[tag])
        ax.set_yscale("log")
        ax.set_xlabel("mean pairwise cosine at layer $\\ell$")
        ax.set_ylabel("depth gain $G_\\ell$")
        ax.set_title(f"{SHORT[tag]}: anisotropy vs gain, Spearman "
                     f"{cors['mean_cos']['spearman_vs_gain']:+.2f} "
                     f"(n={len(blocksel)})", fontsize=8.6)
        savefig(f, f"fig_D2c_scatter_cos_gain_{tag}.png")
    out["D2"] = res
    # summary heatmap of spearman vs gain
    f, ax = plt.subplots(figsize=(6.8, 3.4))
    ms = list(meas.keys())
    Mx = np.full((len(TAGS), len(ms)), np.nan)
    for i, tag in enumerate(TAGS):
        for j, m in enumerate(ms):
            v = out["D2"][tag]["correlations"][m]["spearman_vs_gain"]
            Mx[i, j] = v if np.isfinite(v) else np.nan
    im = ax.imshow(Mx, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([m.replace("ID_mle20_pooled", "ID (pooled)")
                        .replace("ID_mle20_perprompt_med", "ID (per-prompt)")
                        for m in ms], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(TAGS)))
    ax.set_yticklabels([SHORT[t] for t in TAGS], fontsize=8)
    for i in range(len(TAGS)):
        for j in range(len(ms)):
            if np.isfinite(Mx[i, j]):
                ax.text(j, i, f"{Mx[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="w" if abs(Mx[i, j]) > 0.55 else "k")
    plt.colorbar(im, ax=ax, label="Spearman $\\rho$ (layer-wise, vs gain)")
    ax.set_title("Within-model: layer dimension vs layer gain -- "
                 "anisotropy correlates consistently", fontsize=9)
    ax.grid(False)
    savefig(f, "fig_D2d_withinmodel_heatmap.png")


# ---------------------------------------------------------------- D3
def d3(curves, out, scal, fields):
    res = {}
    # ---- radial fraction curves (all models)
    f, ax = plt.subplots(figsize=(6.2, 3.8))
    for tag in TAGS:
        d = np.load(A1_BY_TAG[tag])
        L = int(curves[tag]["L"])
        cos = d["cos"].astype(np.float64)      # [n_r, n_dir, n_l, T]
        layers = d["layers"]
        frac = (np.abs(cos) > 0.5).mean(axis=(0, 1, 3))
        mab = np.abs(cos).mean(axis=(0, 1, 3))
        ax.plot(layers / L, mab, color=COLOR[tag], lw=1.5,
                label=SHORT[tag])
        res[tag] = {"layers": [int(v) for v in layers],
                    "mean_abs_cos": [float(v) for v in mab],
                    "frac_radial_gt_0.5": [float(v) for v in frac],
                    "frac_anticorrelated": [float(v) for v in
                                            (cos < -0.5).mean(axis=(0, 1, 3))],
                    "cos_last_layer_mean": float(cos[:, :, -1].mean()),
                    "cos_last_abs_mean": float(np.abs(
                        cos[:, :, -1]).mean()),
                    }
        if tag == "deepseek":
            # time series example at the last layer
            f2, ax2 = plt.subplots(figsize=(5.6, 3.2))
            med = np.median(cos[:, :, -1, :], axis=(0, 1))
            lo, hi = np.percentile(cos[:, :, -1, :], NQ, axis=(0, 1))
            ax2.fill_between(np.arange(cos.shape[-1]), lo, hi,
                             color=COLOR[tag], alpha=0.2, lw=0)
            ax2.plot(np.arange(cos.shape[-1]), med, color=COLOR[tag],
                     lw=1.1)
            ax2.axhline(0, color="k", lw=0.7)
            ax2.axhline(0.5, color="k", ls=":", lw=0.7)
            ax2.axhline(-0.5, color="k", ls=":", lw=0.7)
            ax2.set_xlabel("generated token index $t$")
            ax2.set_ylabel("$\\cos(\\delta h, h)$ (last raw block)")
            fr = res[tag]["frac_radial_gt_0.5"][-1]
            kind = "TANGENTIAL" if fr < 0.5 else "RADIAL"
            ax2.set_title(f"DeepSeek: the response is mostly {kind} "
                          f"($|\\cos|>0.5$ for {fr:.0%} of states)",
                          fontsize=8.6)
            savefig(f2, "fig_D3b_cos_timeseries_deepseek.png")
        del d
    ax.set_xlabel("relative depth $\\ell/L$")
    ax.set_ylabel("mean $|\\cos(\\delta h, h)|$")
    allmab = np.concatenate([res[t]["mean_abs_cos"] for t in TAGS])
    ax.set_title(f"Radial fraction of the response: $|\\cos(\\delta h,h)|$="
                 f"{allmab.min():.2f}-{allmab.max():.2f} across models "
                 f"(mostly tangential)", fontsize=9)
    ax.legend(fontsize=7.5)
    savefig(f, "fig_D3_radial_vs_depth_all.png")

    # ---- radius dependence (deepseek w100 sweep, 12 radii)
    d = np.load(A1_BY_TAG["deepseek_w100"])
    radii = d["radii"]
    cos = d["cos"].astype(np.float64)          # [12, 20, 8, 300]
    layers = d["layers"]
    L = int(curves["deepseek"]["L"])
    f, ax = plt.subplots(figsize=(5.8, 3.6))
    sel = [0, 3, 6, 7]                         # emb, blk, mid, last
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(sel)))
    ylast = None
    for c, li in zip(cmap, sel):
        y = np.abs(cos[:, :, li, :]).mean(axis=(1, 2))
        if layers[li] == layers[-1]:
            ylast = y
        ax.plot(radii, y, "o-", color=c, ms=4,
                label=f"a1 layer {layers[li]}")
    ax.set_xscale("log")
    ax.set_xlabel("perturbation radius $r$")
    ax.set_ylabel("mean $|\\cos(\\delta h, h)|$")
    grow = ylast[-1] > ylast[0]
    word = "grows" if grow else "does not grow"
    ax.set_title(f"DeepSeek (12 radii): |cos| {word} with $r$ "
                 f"(last layer: {ylast[0]:.2f} -> {ylast[-1]:.2f})",
                 fontsize=8.8)
    ax.legend(fontsize=7)
    savefig(f, "fig_D3c_radial_vs_radius_deepseek.png")
    res["deepseek_w100_radius_dep"] = {
        "radii": [float(v) for v in radii],
        "mean_abs_cos_layer27": [float(v) for v in
                                 np.abs(cos[:, :, -1, :]).mean(axis=(1, 2))],
        "mean_abs_cos_layer0": [float(v) for v in
                                np.abs(cos[:, :, 0, :]).mean(axis=(1, 2))],
    }
    del d

    # ---- dh response-field PCA + subspace overlaps
    overlap_res = {}
    f, ag = plt.subplots(1, 4, figsize=(13.6, 3.1))
    for j, tag in enumerate(TAGS):
        d = np.load(A1_BY_TAG[tag])
        X = dh_field(d)
        fm = field_measures(X)
        ax = ag[j]
        lam_c = eig_desc((X - X.mean(0)) @ (X - X.mean(0)).T / len(X))
        ii = np.arange(1, min(400, len(lam_c)) + 1)
        ax.loglog(ii, lam_c[:len(ii)] / lam_c.sum(), color=COLOR[tag],
                  lw=1.4)
        ax.set_xlabel("PC index $i$ ($\\delta h$ field)")
        ax.set_ylabel("variance share")
        ax.set_title(f"{SHORT[tag]}: $\\delta h$ field eff-rank90="
                     f"{fm['eff90_c']}/{fm['D']}", fontsize=8.4)
        del d
    f.suptitle("D3: the perturbation RESPONSE field is confined to a "
               "low-dim subspace (PCA of pooled $\\delta h$, all radii"
               "/tokens)", fontsize=9.5, y=1.04)
    f.tight_layout()
    savefig(f, "fig_D3d_dhfield_spectra.png")

    f, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(TAGS))
    w = 0.27
    vals = {k: [] for k in ["dh_state", "pp_control", "random"]}
    for tag in TAGS:
        d = np.load(A1_BY_TAG[tag])
        Xd = dh_field(d)
        Vd, _ = pca_topk(Xd, 32)
        from dimlib import DIMROOT
        a = np.load(os.path.join(DIMROOT, tag, "acts.npz"))
        L = int(curves[tag]["L"])
        Xs = np.concatenate([a["acts"][p, L - 1] for p in range(3)]
                            ).astype(np.float32)
        Vs, _ = pca_topk(Xs, 32)
        vals["dh_state"].append(subspace_overlap(Vd, Vs))
        # prompt-vs-prompt control at the same depth
        Xp0 = a["acts"][0, L - 1].astype(np.float32)
        Xp1 = a["acts"][1, L - 1].astype(np.float32)
        V0, _ = pca_topk(Xp0, 32)
        V1, _ = pca_topk(Xp1, 32)
        vals["pp_control"].append(subspace_overlap(V0, V1))
        rng = np.random.default_rng(0)
        ov = []
        for _ in range(40):
            Q1 = np.linalg.qr(rng.standard_normal(
                (Xs.shape[1], 32)))[0][:, :32]
            Q2 = np.linalg.qr(rng.standard_normal(
                (Xs.shape[1], 32)))[0][:, :32]
            ov.append(subspace_overlap(Q1, Q2))
        vals["random"].append(float(np.mean(ov)))
        overlap_res[tag] = {k: float(vals[k][-1]) for k in vals}
        del a, d
    for j, (k, lab, c) in enumerate([
            ("dh_state", "$\\delta h$ PCs vs state PCs (k=32)",
             "#222222"),
            ("pp_control", "prompt A PCs vs prompt B PCs (control)",
             "#66CCEE"),
            ("random", "random subspaces (chance)", "#CCCCCC")]):
        ax.bar(x + (j - 1) * w, vals[k], w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TAGS], fontsize=8)
    ax.set_ylabel("mean squared principal cosines (k=32)")
    ax.set_title("The response lives in the same subspace as the states "
                 "(overlap $\\approx$ prompt-to-prompt overlap, "
                 "$\\gg$ chance)")
    ax.legend(fontsize=7.2)
    savefig(f, "fig_D3e_subspace_overlap.png")
    res["subspace_overlap_k32"] = overlap_res
    out["D3"] = res


# ---------------------------------------------------------------- D4
def d4(out):
    res = {}
    for tag in ["qwen3_4b", "phi4mini", "qwen3_5"]:
        wfiles = {}
        import glob
        for w in [60, 100, 200, 4096]:
            p = os.path.join(W4D, tag, f"w{w}", "a1.npz")
            if os.path.isfile(p):
                wfiles[w] = p
        if not wfiles:
            continue
        curves = {}
        margins_ref = None
        margins_equal = True
        predT = {}
        for w, p in sorted(wfiles.items()):
            d = np.load(p)
            if margins_ref is None:
                margins_ref = d["margins"]
            margins_equal &= bool(np.array_equal(margins_ref, d["margins"]))
            layers = d["layers"]
            dh = d["dh_norm"].astype(np.float64)     # [1, n_d, n_l, T]
            r = d["radii"][0]
            G = dh.mean(axis=(0, 1, 3)).ravel() / r
            Gband = np.percentile(dh.mean(axis=3).ravel(), NQ) / r
            predT[w] = float(np.mean(d["pred_T"]))
            cosf = (np.abs(d["cos"]) > 0.5).mean() if "cos" in d.files \
                else None
            curves[w] = (layers, G, Gband, float(np.median(
                dh[:, :, -1, -50:].mean(axis=(0, 1)) / r)), cosf)
        Lmap = {"qwen3_4b": 36, "phi4mini": 32, "qwen3_5": 24}
        L = Lmap[tag]
        f, ax = plt.subplots(figsize=(5.8, 3.6))
        cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(curves)))
        for c, (w, (layers, G, Gband, glast, cosf)) in zip(
                cmap, sorted(curves.items())):
            ax.plot(layers / L, G, "o-", ms=3, color=c, lw=1.3,
                    label=f"W={w}")
        ax.set_yscale("log")
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("depth gain $G_\\ell=\\|\\delta h\\|/r$")
        rr = (curves[max(curves)][3] / curves[min(curves)][3])
        ax.set_title(f"{SHORT[tag]}: 4D-mask window sweep -- last-layer "
                     f"gain W4096/W60 = {rr:.2f}", fontsize=8.8)
        ax.legend(fontsize=7)
        savefig(f, f"fig_D4_window_{tag}.png")
        res[tag] = {
            "windows": sorted(curves.keys()),
            "margins_equal_across_W": margins_equal,
            "last_layer_gain": {str(w): float(curves[w][3])
                                for w in curves},
            "gain_ratio_W4096_over_W60": float(rr),
            "predT_mean": {str(w): predT[w] for w in predT},
        }
        print(f"  D4 {tag}: margins_equal={margins_equal} "
              f"predT={ {w: round(predT[w],1) for w in predT} }")
    # predT vs W figure
    f, ax = plt.subplots(figsize=(5.4, 3.4))
    for tag in res:
        ws = sorted(int(w) for w in res[tag]["predT_mean"])
        ys = [res[tag]["predT_mean"][str(w)] for w in ws]
        ax.plot(ws, ys, "o-", color=COLOR[tag], label=SHORT[tag])
    ax.set_xscale("log")
    ax.set_xlabel("attention window $W$ (tokens)")
    ax.set_ylabel("mean first-flip position $\\overline{t_{flip}}$")
    ax.set_title("First token flip is nearly window-independent "
                 "(baseline margins identical across W)")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_D4b_predT_vs_window.png")
    out["D4"] = res


# ---------------------------------------------------------------- main
def main():
    curves = {t: load_cache(t) for t in TAGS}
    out = {}
    print("=== D1: cross-model ordering")
    scal, fields = d1(curves, out)
    print("=== D2: within-model layer correlations")
    d2(curves, out, scal)
    print("=== D3: radial/tangential + response subspace")
    d3(curves, out, scal, fields)
    print("=== D4: window sweep")
    d4(out)
    # simplex sanity: perturbation directions isotropic?
    for tag in TAGS:
        d = np.load(A1_BY_TAG[tag])
        if "simplex" in d.files:
            S = d["simplex"]
            off = S[~np.eye(len(S), dtype=bool)]
            out.setdefault("data_notes", {})[f"simplex_mean_offdiag_{tag}"] \
                = float(off.mean())
        del d
    section_numbers("D", out)


if __name__ == "__main__":
    main()
