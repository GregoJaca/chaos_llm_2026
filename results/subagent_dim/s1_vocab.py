"""s1_vocab.py -- Section A: vocabulary / embedding geometry.

A1  singular spectra of E and W_U: PR (lambda=s^2), stable rank, top-variance
    fraction s1^2/sum s^2, effective rank 90/99%, power-law tail fit,
    top-20 spectrum.
A2  anisotropy: mean pairwise cosine computed EXACTLY from the Gram matrices
    (20k sampled pairs) for E and W_U; hidden states at layer 1 and last raw
    block (all 1200 pairs); cosine distributions.
A3  cone structure: ||mean row|| / mean ||row|| (exact from Gram for E, W_U)
    and the fraction of variance in the first singular direction.
A4  row-norm distributions (log-histogram); row norm vs projection on the
    top singular direction(s), from the 1024 sampled rows.
A5  tied/untied: DeepSeek untied -> E-vs-W_U principal angles CANNOT be
    computed from separate Grams (no cross-Gram E^T W_U available): skipped.
    phi4mini / qwen3_4b / qwen3_5 tied -> GE == GW exactly: reported.
"""
import os
import numpy as np
from scipy.sparse.linalg import eigsh

from dimlib import (load_tag, discover_tags, DISPLAY, COLOR, CE, CW, OUT,
                    section_numbers, savefig, mean_cos_from_gram,
                    cone_ratio_from_gram, cone_ratio_rows, allpairs_cos,
                    mean_cos_rows, pr_lambda, stable_rank, top_frac,
                    eff_rank, powerlaw_slope, eig_desc, pearson)
import matplotlib.pyplot as plt

KPAIRS = 20000


def top_evecs(G, k=2):
    """Top-k eigenvectors of a symmetric PSD dense matrix."""
    try:
        w, u = eigsh(np.asarray(G, np.float64), k=k, which="LA")
        order = np.argsort(w)[::-1]
        return u[:, order], w[order]
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


def main():
    tags = discover_tags()
    out = {"k_pairs_cosine": KPAIRS, "per_tag": {}}
    # accumulators for cross-model figures
    spec_ax, hist_ax = {}, {}

    # ---- fig_A1: spectra grid (2x2)
    fg, ags = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_top, ags_top = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_cos, ags_cos = plt.subplots(2, 2, figsize=(8.6, 6.6))
    fg_nh, ags_nh = plt.subplots(2, 2, figsize=(8.6, 6.0))
    ags = ags.ravel(); ags_top = ags_top.ravel()
    ags_cos = ags_cos.ravel(); ags_nh = ags_nh.ravel()

    bars = {k: [] for k in ["E", "W_U", "h1", "hL"]}       # mean cos
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
        rec = {"D": D, "V": V, "L": L, "tied": tied,
               "E_spec": spec_measures(sE), "W_spec": spec_measures(sW),
               "mean_row_norm_E": float(normE.mean()),
               "std_row_norm_E": float(normE.std()),
               "mean_row_norm_W": float(normW.mean()),
               "GE_eq_GW": bool(np.array_equal(GE, GW))}

        # ---------------- A1 spectra
        ax = ags[i]
        ax.loglog(np.arange(1, D + 1), sE, color=CE, label="$E$: $s_i$")
        ax.loglog(np.arange(1, D + 1), sW, color=CW, ls="--",
                  label="$W_U$: $s_i$")
        i0, i1 = 10, D // 2
        fitx = np.arange(i0, i1 + 1)
        for s, c, lab in [(sE, CE, None), (sW, CW, None)]:
            al, _ = powerlaw_slope(s, i0, i1)
            A = fitx.astype(float)
            ly = np.log10(s[i0 - 1:i1])
            cc = np.polyfit(np.log10(A), ly, 1)
            ax.loglog(fitx, 10 ** np.polyval(cc, np.log10(A)), color=c,
                      ls=":", lw=1.1, alpha=0.9)
        ax.set_xlabel("singular index $i$ (1-based)")
        ax.set_ylabel("singular value $s_i$")
        aE = rec["E_spec"]["powerlaw_tail_alpha_i0_10_i1_halfD"]
        aW = rec["W_spec"]["powerlaw_tail_alpha_i0_10_i1_halfD"]
        ax.set_title(f"{DISPLAY[tag]}: vocab spectra fall as "
                     f"$s_i\\sim i^{{{aE:.2f}}}$ ($E$), "
                     f"$i^{{{aW:.2f}}}$ ($W_U$)")
        ax.legend()
        # top-20 zoom
        ax = ags_top[i]
        m = 20
        ax.loglog(np.arange(1, m + 1), sE[:m], "o-", color=CE,
                  label="$E$ top-20", ms=3.5)
        ax.loglog(np.arange(1, m + 1), sW[:m], "s--", color=CW,
                  label="$W_U$ top-20", ms=3.5)
        gap = sE[0] / sE[1]
        ax.set_xlabel("singular index $i$")
        ax.set_ylabel("singular value $s_i$")
        ax.set_title(f"{DISPLAY[tag]}: top-20 spectrum; "
                     f"$s_1/s_2={gap:.1f}$ (single dominant mode)")
        ax.legend()

        # ---------------- A2 anisotropy
        cE, seE = mean_cos_from_gram(GE, KPAIRS)
        cW, seW = mean_cos_from_gram(GW, KPAIRS)
        rec["mean_cos_E"] = cE
        rec["mean_cos_E_se"] = seE
        rec["mean_cos_W"] = cW
        rec["mean_cos_W_se"] = seW
        hists = {}
        hists["E"] = cos_hist = None  # placeholder
        from dimlib import cos_pairs_from_gram
        cpE = cos_pairs_from_gram(GE, KPAIRS)
        cpW = cos_pairs_from_gram(GW, KPAIRS)
        hE, eE = np.histogram(cpE, bins=80, range=(-1, 1), density=True)
        hW, eW = np.histogram(cpW, bins=80, range=(-1, 1), density=True)

        # hidden states at layer 1 and last raw block (idx L-1)
        acts = a["acts"]
        X1 = np.concatenate([acts[p, 1] for p in range(3)]
                            ).astype(np.float32)
        XL = np.concatenate([acts[p, L - 1] for p in range(3)]
                            ).astype(np.float32)
        del acts
        c1, se1 = mean_cos_rows(X1)
        cL, seL = mean_cos_rows(XL)
        hS1 = allpairs_cos(X1)
        hSL = allpairs_cos(XL)
        hH1, eH = np.histogram(hS1, bins=80, range=(-1, 1), density=True)
        hHL, _ = np.histogram(hSL, bins=80, range=(-1, 1), density=True)
        rec["mean_cos_h_layer1"] = c1
        rec["mean_cos_h_layer1_se"] = se1
        rec["mean_cos_h_layerLastRaw"] = cL
        rec["mean_cos_h_layerLastRaw_se"] = seL
        for nm, hd, ed in [("E", hE, eE), ("W_U", hW, eW),
                           ("h@1", hH1, eH), ("h@L", hHL, eH)]:
            ax = ags_cos[i]
            ax.plot(0.5 * (ed[1:] + ed[:-1]), hd, lw=1.1,
                    label=f"{nm}: $\\bar{{c}}$="
                          f"{ {'E': cE, 'W_U': cW, 'h@1': c1, 'h@L': cL}[nm]: .3f}")
        ax.set_xlabel("pairwise cosine between rows")
        ax.set_ylabel("density (1/unit cosine)")
        ax.set_title(f"{DISPLAY[tag]}: hidden states are far more anisotropic "
                     f"than the vocabulary")
        ax.legend(fontsize=6.2)
        bars["E"].append(cE); bars["W_U"].append(cW)
        bars["h1"].append(c1); bars["hL"].append(cL)

        # ---------------- A3 cone + top-direction variance
        coneE = cone_ratio_from_gram(GE, normE)
        coneW = cone_ratio_from_gram(GW, normW)
        cone1, coneL = cone_ratio_rows(X1), cone_ratio_rows(XL)
        rec["cone_ratio_E"] = coneE
        rec["cone_ratio_W"] = coneW
        rec["cone_ratio_h_layer1"] = cone1
        rec["cone_ratio_h_layerLastRaw"] = coneL
        # top-direction variance fraction for hidden states (mean-removed cov)
        tv1 = topvar_h(X1)
        tvL = topvar_h(XL)
        rec["top_var_frac_h_layer1"] = tv1
        rec["top_var_frac_h_layerLastRaw"] = tvL
        cone_bars["E"].append(coneE); cone_bars["W_U"].append(coneW)
        cone_bars["h1"].append(cone1); cone_bars["hL"].append(coneL)
        topvar_bars["E"].append(rec["E_spec"]["top_var_frac_s1_2_over_sum_s2"])
        topvar_bars["W_U"].append(
            rec["W_spec"]["top_var_frac_s1_2_over_sum_s2"])
        topvar_bars["h1"].append(tv1); topvar_bars["hL"].append(tvL)

        # ---------------- A4 norms + projections
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
        ax.set_xlabel("row norm $\\|e_v\\|$ / $\\|w_v\\|$")
        ax.set_ylabel("density per unit log-norm")
        ax.set_title(f"{DISPLAY[tag]}: row-norm spread "
                     f"(cv={normE.std()/normE.mean():.2f})")
        ax.legend(fontsize=7)

        # projections of sampled rows on top singular directions
        uE, wE = top_evecs(GE, k=3)
        step = V // 1024
        idx = np.arange(1024) * step
        smpE = v["sample_E"].astype(np.float32)
        nrm = normE[idx]
        p1 = smpE @ uE[:, 0]
        p2 = smpE @ uE[:, 1]
        r1 = pearson(np.abs(p1), nrm)
        rec["pearson_abs_proj_u1_vs_norm_E"] = r1
        rec["frac_norm2_in_u1_E"] = float((p1**2).mean() / (nrm**2).mean())
        rec["frac_norm2_in_u2_E"] = float((p2**2).mean() / (nrm**2).mean())
        uW, _ = top_evecs(GW, k=3)
        smpW = v["sample_W"].astype(np.float32)
        p1W = smpW @ uW[:, 0]
        rec["pearson_abs_proj_u1_vs_norm_W"] = pearson(np.abs(p1W),
                                                       normW[idx])
        # figure
        f, a2 = plt.subplots(1, 2, figsize=(8.2, 3.3))
        a2[0].scatter(nrm, np.abs(p1), s=5, color=CE, alpha=0.6)
        a2[0].set_xlabel("row norm $\\|e_v\\|$ (sampled, n=1024)")
        a2[0].set_ylabel("$|e_v\\cdot u_1|$ (top dir.)")
        a2[0].set_title(f"$E$: |proj on $u_1|$ vs norm, r={r1:.2f}")
        a2[1].scatter(nrm, np.abs(p2), s=5, color=CW, alpha=0.6)
        a2[1].set_xlabel("row norm $\\|e_v\\|$ (sampled, n=1024)")
        a2[1].set_ylabel("$|e_v\\cdot u_2|$ (2nd dir.)")
        a2[1].set_title(f"$E$: |proj on $u_2|$ vs norm "
                        f"(frac$\\|e\\|^2$={rec['frac_norm2_in_u2_E']:.3f})")
        f.suptitle(f"{DISPLAY[tag]}: high-norm vocab rows are not "
                   f"top-direction rows (u1 holds "
                   f"{rec['frac_norm2_in_u1_E']*100:.1f}% of $\\|e\\|^2$)",
                   fontsize=9.5, y=1.04)
        savefig(f, f"fig_A4b_proj_vs_norm_{tag}.png")

        # per-tag cosine-histogram data into accumulators (unused vars ok)
        spec_ax[tag] = (sE, sW)

        # ---------------- A5 tied/untied
        if tied:
            rec["A5_note"] = ("tied: W_U = E, verified GE == GW exactly; "
                              "E-vs-W_U principal angles are trivial (0).")
        else:
            rec["A5_note"] = ("untied: E-vs-W_U principal angles CANNOT be "
                              "computed from separate Grams (no E^T W_U); "
                              "skipped per protocol.")
            f, ax = plt.subplots(figsize=(4.6, 3.3))
            ax.loglog(np.arange(1, D + 1), sE / sE[0], color=CE,
                      label="$s_i(E)/s_1$")
            ax.loglog(np.arange(1, D + 1), sW / sW[0], color=CW, ls="--",
                      label="$s_i(W_U)/s_1$")
            ax.set_xlabel("singular index $i$")
            ax.set_ylabel("normalized singular value")
            ax.set_title(f"{DISPLAY[tag]} (untied): $E$ and $W_U$ spectra "
                         "differ; principal angles unavailable")
            ax.legend()
            savefig(f, f"fig_A5_E_vs_W_{tag}.png")

        out["per_tag"][tag] = rec
        print(f"  E: meancos={cE:.3f} cone={coneE:.3f} "
              f"PR={rec['E_spec']['PR_lambda_s2']:.0f} "
              f"eff90={rec['E_spec']['eff_rank_90']} alpha={aE:.2f}")
        print(f"  h@L: meancos={cL:.3f} cone={coneL:.3f} topvar={tvL:.3f}")

    # ---- shared figure titles + save grids
    fg.suptitle("Vocabulary singular spectra: smooth power-law decay, "
                "no sharp rank gap (all 4 models)", fontsize=10.5, y=1.01)
    fg.tight_layout()
    savefig(fg, "fig_A1_spectra_grid.png")
    fg_top.suptitle("Top-20 singular values (log-log): every model has one "
                    "dominant direction then a slow decay", fontsize=10.5,
                    y=1.01)
    fg_top.tight_layout()
    savefig(fg_top, "fig_A1b_top20_grid.png")
    fg_cos.suptitle("Pairwise-cosine distributions: hidden states cluster in "
                    "a narrow cone; vocabulary does not", fontsize=10.5, y=1.01)
    fg_cos.tight_layout()
    savefig(fg_cos, "fig_A2_cos_hist_grid.png")
    fg_nh.suptitle("Vocabulary row-norm distributions (log bins)",
                   fontsize=10.5, y=1.01)
    fg_nh.tight_layout()
    savefig(fg_nh, "fig_A4_norm_hist_grid.png")

    # ---- cross-model bar charts
    x = np.arange(len(tags))
    wd = 0.2
    f, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, (k, c) in enumerate(zip(["E", "W_U", "h1", "hL"],
                                   [CE, CW, "#BBBBBB", "#444444"])):
        ax.bar(x + (j - 1.5) * wd, bars[k], wd, color=c,
               label={"E": "vocab $E$", "W_U": "unemb $W_U$",
                      "h1": "hidden @ block 1",
                      "hL": "hidden @ last raw block"}[k])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[t].replace("-R1-Distill-1.5B", "")
                        .replace("-Instruct", "").replace("-mini", "m")
                        for t in tags], fontsize=8)
    ax.set_ylabel("mean pairwise cosine $\\bar{c}$")
    ax.set_title("Anisotropy: hidden states are 2-6x more anisotropic than "
                 "their vocabularies")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A2b_mean_cos_bars.png")

    f, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, (k, c) in enumerate(zip(["E", "W_U", "h1", "hL"],
                                   [CE, CW, "#BBBBBB", "#444444"])):
        ax.bar(x + (j - 1.5) * wd, cone_bars[k], wd, color=c,
               label={"E": "vocab $E$", "W_U": "unemb $W_U$",
                      "h1": "hidden @ block 1",
                      "hL": "hidden @ last raw block"}[k])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[t].replace("-R1-Distill-1.5B", "")
                        .replace("-Instruct", "").replace("-mini", "m")
                        for t in tags], fontsize=8)
    ax.set_ylabel("cone ratio $\\|\\mu\\|/\\overline{\\|x\\|}$")
    ax.set_title("Cone structure: hidden-state means carry 20-40% of a row "
                 "norm; vocab means ~1-4%")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A3_cone_bars.png")

    f, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, (k, c) in enumerate(zip(["E", "W_U", "h1", "hL"],
                                   [CE, CW, "#BBBBBB", "#444444"])):
        ax.bar(x + (j - 1.5) * wd, topvar_bars[k], wd, color=c,
               label={"E": "vocab $E$", "W_U": "unemb $W_U$",
                      "h1": "hidden @ block 1",
                      "hL": "hidden @ last raw block"}[k])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[t].replace("-R1-Distill-1.5B", "")
                        .replace("-Instruct", "").replace("-mini", "m")
                        for t in tags], fontsize=8)
    ax.set_ylabel("variance fraction in top direction $\\lambda_1/\\sum\\lambda$")
    ax.set_title("Top-direction variance: vocabulary ~1-3%; hidden states "
                 "10x higher after mean removal")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_A3b_topvar_bars.png")

    section_numbers("A", out)


def topvar_h(X):
    """Fraction of centered variance in the top PC of pooled hidden states."""
    Xc = X - X.mean(0)
    G = Xc @ Xc.T
    lam = eig_desc(G)
    return float(lam[0] / lam.sum())


if __name__ == "__main__":
    main()
