"""s1_vocab.py -- Section A: vocabulary / embedding geometry.

IMPORTANT data-layout note (discovered during checks):
  GE/GW are D x D FEATURE Grams (E^T E with E in [V, D]), NOT V x V row Grams.
  Therefore pairwise cosines between vocabulary ROWS cannot be read off the
  saved Grams.  We instead compute them exactly on the uniform 1024-row
  sample (all C(1024,2)=523,776 pairs), which is an unbiased estimate of the
  full-V mean pairwise cosine (SE ~ 2e-4), and we cross-validate the
  row-cone via the exact identity s_1^2 ~ V ||mu||^2 (top singular value vs
  sampled mean row).  Pairwise cosines between DIMENSIONS (columns) are
  also computed exactly from the Gram (20k pairs) and reported separately.

A1  singular spectra of E and W_U: PR (lambda=s^2), stable rank, top-variance
    fraction s1^2/sum s^2, effective rank 90/99%, power-law tail fit,
    top-20 spectrum.
A2  anisotropy: mean pairwise cosine between random rows of E, W_U (exact
    sample pairs) and pooled hidden states at block 1 / last raw block
    (all 1200-point pairs); cosine distributions.
A3  cone structure: ||mean row|| / mean ||row|| (E, W_U from the exact
    sample with a noise-floor bound; hidden states exact) and the variance
    fraction of the first singular direction.
A4  row-norm distributions (log-histogram); row norm vs projection on the
    top singular direction(s), from the 1024 sampled rows.
A5  tied/untied: DeepSeek untied -> E-vs-W_U principal angles CANNOT be
    computed from separate Grams (no cross-Gram E^T W_U available): skipped.
    phi4mini / qwen3_4b / qwen3_5 tied -> GE == GW exactly: reported.
"""
import os
import numpy as np
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt

from dimlib import (load_tag, discover_tags, DISPLAY, COLOR, CE, CW,
                    section_numbers, savefig, allpairs_cos, mean_cos_rows,
                    cone_ratio_rows, pr_lambda, stable_rank, top_frac,
                    eff_rank, powerlaw_slope, eig_desc, pearson,
                    cos_pairs_from_gram)

NSMP = 1024


def top_evecs(G, k=3):
    try:
        w, u = eigsh(np.asarray(G, np.float64), k=k, which="LA")
        o = np.argsort(w)[::-1]
        return u[:, o], w[o]
    except Exception:
        w, u = np.linalg.eigh(np.asarray(G, np.float64))
        return u[:, ::-1][:, :k], w[::-1][:k]


def spec_measures(s):
    lam = np.asarray(s, float) ** 2
    a, r2 = powerlaw_slope(s, i0=10, i1=len(s) // 2)
    return {
        "PR_lambda_s2": pr_lambda(lam),
        "stable_rank_sum_s2_over_s1_2": stable_rank(s),
        "top_var_frac_s1_2_over_sum_s2": top_frac(s),
        "eff_rank_90": eff_rank(lam, 0.9),
        "eff_rank_99": eff_rank(lam, 0.99),
        "powerlaw_tail_alpha_i0_10_i1_halfD": a,
        "powerlaw_tail_R2": r2,
    }


def row_sample_cos(sample):
    """All-pairs cosine within the uniform row sample; exact pair values."""
    c = allpairs_cos(sample.astype(np.float32))
    return c


def cone_from_sample(sample, mean_norm_full):
    """||mu_hat|| / mean||row|| and its 2-sigma noise floor from n=1024."""
    mu = sample.mean(0)
    cone = float(np.linalg.norm(mu) / mean_norm_full)
    Xc = sample - mu
    tr = float((Xc**2).sum() / len(sample))       # ~ mean ||e-mu||^2
    floor = float(2 * np.sqrt(tr / len(sample)) / mean_norm_full)
    return cone, floor


def main():
    tags = discover_tags()
    out = {"n_sample_rows": NSMP,
           "note": ("row cosines for E/W_U use all C(1024,2) pairs of the "
                    "uniform row sample (SE~2e-4); GE/GW are DxD feature "
                    "grams, so row pairs are not directly readable from "
                    "them. Dimension-pair cosines from the grams are "
                    "reported separately."),
           "per_tag": {}}

    fg, ags = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_top, ags_top = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_cos, ags_cos = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_nh, ags_nh = plt.subplots(2, 2, figsize=(8.6, 6.0))
    ags, ags_top = ags.ravel(), ags_top.ravel()
    ags_cos, ags_nh = ags_cos.ravel(), ags_nh.ravel()

    bars = {k: [] for k in ["E", "W_U", "h1", "hL"]}
    cone_bars = {k: [] for k in ["E", "W_U", "h1", "hL"]}
    topvar_bars = {k: [] for k in ["E", "W_U", "h1", "hL"]}

    for i, tag in enumerate(tags):
        print(f"=== A: {tag}")
        meta, v, a, lg = load_tag(tag)
        D, V, L = int(v["D"][0]), int(v["V"][0]), int(v["L"][0])
        tied = bool(v["tied"][0])
        GE, GW = v["GE"], v["GW"]
        sE, sW = v["sE"].astype(np.float64), v["sW"].astype(np.float64)
        normE, normW = v["normE"], v["normW"]
        step = V // NSMP
        idx = np.arange(NSMP) * step

        rec = {"D": D, "V": V, "L": L, "tied": tied,
               "E_spec": spec_measures(sE), "W_spec": spec_measures(sW),
               "mean_row_norm_E": float(normE.mean()),
               "std_row_norm_E": float(normE.std()),
               "max_row_norm_E": float(normE.max()),
               "mean_row_norm_W": float(normW.mean()),
               "GE_eq_GW": bool(np.array_equal(GE, GW))}

        # ---------------- A1 spectra ------------------------------------
        ax = ags[i]
        ii = np.arange(1, D + 1)
        ax.loglog(ii, sE, color=CE, label="$E$: $s_i$")
        ax.loglog(ii, sW, color=CW, ls="--", label="$W_U$: $s_i$")
        i0, i1 = 10, D // 2
        fitx = np.arange(i0, i1 + 1).astype(float)
        for s, c in [(sE, CE), (sW, CW)]:
            ly = np.log10(s[i0 - 1:i1])
            cc = np.polyfit(np.log10(fitx), ly, 1)
            ax.loglog(fitx, 10 ** np.polyval(cc, np.log10(fitx)), color=c,
                      ls=":", lw=1.1)
        ax.set_xlabel("singular index $i$ (1-based)")
        ax.set_ylabel("singular value $s_i$")
        aE = rec["E_spec"]["powerlaw_tail_alpha_i0_10_i1_halfD"]
        aW = rec["W_spec"]["powerlaw_tail_alpha_i0_10_i1_halfD"]
        ax.set_title(f"{DISPLAY[tag]}: tail $s_i\\sim i^{{{aE:.2f}}}$ ($E$),"
                     f" $i^{{{aW:.2f}}}$ ($W_U$)", fontsize=8.6)
        ax.legend(fontsize=7)
        ax = ags_top[i]
        m = 20
        ax.loglog(ii[:m], sE[:m], "o-", color=CE, label="$E$ top-20", ms=3.5)
        ax.loglog(ii[:m], sW[:m], "s--", color=CW, label="$W_U$ top-20",
                  ms=3.5)
        gapE = sE[0] / sE[1]
        ax.set_xlabel("singular index $i$")
        ax.set_ylabel("singular value $s_i$")
        ax.set_title(f"{DISPLAY[tag]}: $s_1/s_2={gapE:.1f}$ "
                     "(mean direction dominates)", fontsize=8.6)
        ax.legend(fontsize=7)

        # ---------------- A2 anisotropy ---------------------------------
        smpE = v["sample_E"].astype(np.float32)
        smpW = v["sample_W"].astype(np.float32)
        cE_ = row_sample_cos(smpE)
        cW_ = row_sample_cos(smpW)
        cE, seE = float(cE_.mean()), float(cE_.std() / np.sqrt(len(cE_)))
        cW, seW = float(cW_.mean()), float(cW_.std() / np.sqrt(len(cW_)))
        rec["mean_cos_rows_E"] = cE
        rec["mean_cos_rows_E_se"] = seE
        rec["mean_cos_rows_W"] = cW
        rec["mean_cos_rows_W_se"] = seW
        rec["frac_abs_cos_gt_0p3_rows_E"] = float((np.abs(cE_) > 0.3).mean())
        # dimension pairs from the D x D gram (secondary, exact)
        dE = cos_pairs_from_gram(GE, 20000)
        dW = cos_pairs_from_gram(GW, 20000)
        rec["mean_cos_dims_from_GE_20k"] = float(dE.mean())
        rec["mean_cos_dims_from_GW_20k"] = float(dW.mean())

        acts = a["acts"]
        X1 = np.concatenate([acts[p, 1] for p in range(3)]).astype(np.float32)
        XL = np.concatenate([acts[p, L - 1] for p in range(3)]).astype(
            np.float32)
        c1, se1 = mean_cos_rows(X1)
        cL, seL = mean_cos_rows(XL)
        hH1, hHL = allpairs_cos(X1), allpairs_cos(XL)
        rec["mean_cos_h_block1"] = c1
        rec["mean_cos_h_block1_se"] = se1
        rec["mean_cos_h_layerLastRaw"] = cL
        rec["mean_cos_h_layerLastRaw_se"] = seL
        bars["E"].append(cE); bars["W_U"].append(cW)
        bars["h1"].append(c1); bars["hL"].append(cL)

        # ---------------- A3 cone + top-direction variance ---------------
        coneE, floorE = cone_from_sample(smpE, normE.mean())
        coneW, floorW = cone_from_sample(smpW, normW.mean())
        cone1, coneL = cone_ratio_rows(X1), cone_ratio_rows(XL)
        rec["cone_rows_E"] = coneE
        rec["cone_rows_E_noise_floor_2sigma"] = floorE
        rec["cone_rows_W"] = coneW
        rec["cone_rows_W_noise_floor_2sigma"] = floorW
        rec["cone_h_block1"] = cone1
        rec["cone_h_layerLastRaw"] = coneL
        # identity check: s1^2 should be ~ V ||mu||^2 if the top direction
        # is the shared mean
        muE = smpE.mean(0)
        rec["s1sq_over_V_mu2_E"] = float(sE[0]**2 / (V * (muE**2).sum()))
        tv1 = topvar_h(X1)
        tvL = topvar_h(XL)
        rec["top_var_frac_h_block1"] = tv1
        rec["top_var_frac_h_layerLastRaw"] = tvL
        cone_bars["E"].append(coneE); cone_bars["W_U"].append(coneW)
        cone_bars["h1"].append(cone1); cone_bars["hL"].append(coneL)
        topvar_bars["E"].append(
            rec["E_spec"]["top_var_frac_s1_2_over_sum_s2"])
        topvar_bars["W_U"].append(
            rec["W_spec"]["top_var_frac_s1_2_over_sum_s2"])
        topvar_bars["h1"].append(tv1); topvar_bars["hL"].append(tvL)

        # ---------------- A4 norms + projections ------------------------
        ax = ags_nh[i]
        be = np.logspace(np.log10(max(normE.min(), 1e-6)),
                         np.log10(normE.max()), 60)
        ax.hist(normE, bins=be, histtype="step", color=CE, density=True,
                label=f"$E$: $\\bar{{\\|e\\|}}$={normE.mean():.2f}")
        bw = np.logspace(np.log10(max(normW.min(), 1e-6)),
                         np.log10(normW.max()), 60)
        ax.hist(normW, bins=bw, histtype="step", color=CW, density=True,
                ls="--", label=f"$W_U$: $\\bar{{\\|w\\|}}$={normW.mean():.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("row norm")
        ax.set_ylabel("density per unit log-norm")
        ax.set_title(f"{DISPLAY[tag]}: row norms "
                     f"(cv={normE.std()/normE.mean():.2f}, "
                     f"max={normE.max():.1f})", fontsize=8.6)
        ax.legend(fontsize=7)

        uE, _ = top_evecs(GE, k=3)
        nrm = normE[idx]
        p1 = smpE @ uE[:, 0]
        p2 = smpE @ uE[:, 1]
        r1 = pearson(np.abs(p1), nrm)
        rec["pearson_abs_proj_u1_vs_norm_E"] = r1
        rec["frac_norm2_in_u1_E"] = float((p1**2).mean() / (nrm**2).mean())
        rec["frac_norm2_in_u2_E"] = float((p2**2).mean() / (nrm**2).mean())
        uW, _ = top_evecs(GW, k=3)
        rec["pearson_abs_proj_u1_vs_norm_W"] = pearson(
            np.abs(smpW @ uW[:, 0]), normW[idx])

        f, a2 = plt.subplots(1, 2, figsize=(8.2, 3.3))
        a2[0].scatter(nrm, np.abs(p1), s=5, color=CE, alpha=0.6)
        a2[0].set_xlabel("row norm $\\|e_v\\|$ (uniform sample, n=1024)")
        a2[0].set_ylabel("$|e_v\\cdot u_1|$")
        a2[0].set_title(f"$E$: top-dir projection vs norm, "
                        f"r={r1:.2f}; $u_1$ holds "
                        f"{rec['frac_norm2_in_u1_E']*100:.0f}% of "
                        f"$\\|e\\|^2$", fontsize=8.6)
        a2[1].scatter(nrm, np.abs(p2), s=5, color=CW, alpha=0.6)
        a2[1].set_xlabel("row norm $\\|e_v\\|$ (uniform sample, n=1024)")
        a2[1].set_ylabel("$|e_v\\cdot u_2|$")
        a2[1].set_title(f"$E$: 2nd-dir projection vs norm "
                        f"($u_2$: {rec['frac_norm2_in_u2_E']*100:.1f}% of "
                        f"$\\|e\\|^2$)", fontsize=8.6)
        f.suptitle(f"{DISPLAY[tag]}: the shared mean direction $u_1$ carries "
                   f"{rec['frac_norm2_in_u1_E']*100:.0f}% of vocab norm "
                   f"energy", fontsize=9.5, y=1.03)
        savefig(f, f"fig_A4b_proj_vs_norm_{tag}.png")

        # ---------------- A2 histogram panel ----------------------------
        ax = ags_cos[i]
        for c_, col, lab in [(cE_, CE, f"rows $E$: $\\bar c$={cE:+.3f}"),
                             (cW_, CW, f"rows $W_U$: $\\bar c$={cW:+.3f}"),
                             (hH1, "#888888", f"$h$@blk1: $\\bar c$={c1:+.3f}"),
                             (hHL, "#222222",
                              f"$h$@last: $\\bar c$={cL:+.3f}")]:
            ax.hist(c_, bins=80, range=(-1, 1), histtype="step", density=True,
                    color=col, lw=1.1, label=lab)
        ax.set_xlabel("pairwise cosine")
        ax.set_ylabel("pair density (1/unit cos)")
        ax.set_title(f"{DISPLAY[tag]}: $\\bar c$(h@last)/$\\bar c$(E) = "
                     f"{cL/cE:.1f}", fontsize=8.6)
        ax.legend(fontsize=6.0)

        # ---------------- A5 tied/untied ---------------------------------
        if tied:
            rec["A5_note"] = ("tied: W_U = E, verified GE == GW exactly; "
                              "E-vs-W_U principal angles are trivially 0.")
        else:
            rec["A5_note"] = ("untied: E-vs-W_U principal angles CANNOT be "
                              "computed from separate Grams (no E^T W_U); "
                              "skipped per protocol.")
            f, ax = plt.subplots(figsize=(4.6, 3.3))
            ax.loglog(ii, sE / sE[0], color=CE, label="$s_i(E)/s_1$")
            ax.loglog(ii, sW / sW[0], color=CW, ls="--",
                      label="$s_i(W_U)/s_1$")
            ax.set_xlabel("singular index $i$")
            ax.set_ylabel("normalized singular value")
            ax.set_title(f"{DISPLAY[tag]} (untied): $E$ and $W_U$ spectra "
                         "differ; principal angles unavailable", fontsize=8.6)
            ax.legend(fontsize=7.5)
            savefig(f, f"fig_A5_E_vs_W_{tag}.png")

        out["per_tag"][tag] = rec
        print(f"  E rows: meancos={cE:+.3f} cone={coneE:.2f} "
              f"(floor {floorE:.3f}) s1sq/Vmu2="
              f"{rec['s1sq_over_V_mu2_E']:.2f}")
        print(f"  h@1: meancos={c1:+.3f} cone={cone1:.2f} | "
              f"h@last: meancos={cL:+.3f} cone={coneL:.2f} topvar={tvL:.3f}")
        print(f"  E spectrum: PR={rec['E_spec']['PR_lambda_s2']:.1f} "
              f"eff90={rec['E_spec']['eff_rank_90']} "
              f"topvar={rec['E_spec']['top_var_frac_s1_2_over_sum_s2']:.3f} "
              f"alpha={aE:.2f}")
        rec["s1_over_s2_E"] = float(
            (np.asarray(v["sE"])[0] / np.asarray(v["sE"])[1]))
        del acts, X1, XL

    fg.suptitle("Vocabulary singular spectra: one dominant direction + smooth "
                "power-law bulk (4 models)", fontsize=10.5, y=1.01)
    fg.tight_layout()
    savefig(fg, "fig_A1_spectra_grid.png")
    fg_top.suptitle("Top-20 singular values (log-log): a single dominant "
                    "mode, then slow decay", fontsize=10.5, y=1.01)
    fg_top.tight_layout()
    savefig(fg_top, "fig_A1b_top20_grid.png")
    fg_cos.suptitle("Pairwise-cosine distributions: hidden-state anisotropy "
                    "grows with depth in 2 models, shrinks in 2",
                    fontsize=10.5, y=1.01)
    fg_cos.tight_layout()
    savefig(fg_cos, "fig_A2_cos_hist_grid.png")
    fg_nh.suptitle("Vocabulary row-norm distributions (log bins)",
                   fontsize=10.5, y=1.01)
    fg_nh.tight_layout()
    savefig(fg_nh, "fig_A4_norm_hist_grid.png")

    short = {t: DISPLAY[t].replace("-R1-Distill-1.5B", "")
             .replace("-Instruct", "").replace("-mini", " m")
             .replace("DeepSeek", "DS").replace("Qwen3-4B", "Qwen3-4B")
             .replace("Qwen3.5-2B", "Qwen3.5-2B").replace("Phi-4 m", "Phi-4m")
            for t in tags}
    x = np.arange(len(tags))
    wd = 0.2
    labs = [("E", "vocab $E$ rows", CE), ("W_U", "unemb $W_U$ rows", CW),
            ("h1", "hidden @ block 1", "#888888"),
            ("hL", "hidden @ last raw block", "#222222")]

    f, ax = plt.subplots(figsize=(6.6, 3.5))
    for j, (k, lab, c) in enumerate(labs):
        ax.bar(x + (j - 1.5) * wd, bars[k], wd, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([short[t] for t in tags], fontsize=8)
    ax.set_ylabel("mean pairwise cosine $\\bar c$")
    r_lo = min(bars["hL"][i] / max(bars["E"][i], 1e-9)
               for i in range(len(tags)))
    r_hi = max(bars["hL"][i] / max(bars["E"][i], 1e-9)
               for i in range(len(tags)))
    ax.set_title(f"Anisotropy: h@last is {r_lo:.1f}-{r_hi:.1f}x the vocab "
                 f"value (model-dependent, not universal)")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A2b_mean_cos_bars.png")

    f, ax = plt.subplots(figsize=(6.6, 3.5))
    for j, (k, lab, c) in enumerate(labs):
        ax.bar(x + (j - 1.5) * wd, cone_bars[k], wd, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([short[t] for t in tags], fontsize=8)
    ax.set_ylabel("cone ratio $\\|\\mu\\|/\\overline{\\|x\\|}$")
    ax.set_title(f"Cone: vocab rows {min(cone_bars['E']):.2f}-"
                 f"{max(cone_bars['E']):.2f}, hidden states "
                 f"{min(cone_bars['hL']):.2f}-{max(cone_bars['hL']):.2f} "
                 f"(E/W_U from n=1024 sample, floor $\\leq$0.08)")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A3_cone_bars.png")

    f, ax = plt.subplots(figsize=(6.6, 3.5))
    for j, (k, lab, c) in enumerate(labs):
        ax.bar(x + (j - 1.5) * wd, topvar_bars[k], wd, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([short[t] for t in tags], fontsize=8)
    ax.set_ylabel("$\\lambda_1/\\sum\\lambda$ (top-dir var. frac.)")
    ax.set_title(f"Top-direction variance: vocab "
                 f"{min(topvar_bars['E']):.2f}-{max(topvar_bars['E']):.2f}, "
                 f"hidden {min(topvar_bars['hL']):.2f}-"
                 f"{max(topvar_bars['hL']):.2f} after mean removal")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A3b_topvar_bars.png")

    section_numbers("A", out)


def topvar_h(X):
    Xc = X - X.mean(0)
    lam = eig_desc(Xc @ Xc.T)
    return float(lam[0] / lam.sum())


if __name__ == "__main__":
    main()
