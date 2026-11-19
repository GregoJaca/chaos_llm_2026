"""s2b_figs.py -- Section B figures, built from cache/<tag>_layermeas.npz
produced by s2_hidden.py.  Run after s2_hidden.py.

Conventions: x-axis is the acts index (0 = embedding, 1..L-1 = raw blocks,
L = post-final-RMSNorm, marked with a star) or relative depth idx/L.
Bands: min/max across the 3 prompts where per-prompt values exist.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dimlib import (discover_tags, DISPLAY, COLOR, savefig, section_numbers,
                    CACHE)

STARK = "#000000"


def load_cache(tag):
    return np.load(os.path.join(CACHE, f"{tag}_layermeas.npz"))


def mark_ends(ax, x, L, color=STARK):
    ax.plot([x[0]], [ax.get_ylim()[0]], marker="o", mfc="none", mec=color,
            ms=6, ls="none")


def main():
    tags = discover_tags()
    curves = {}
    for tag in tags:
        z = load_cache(tag)
        L = int(z["L"])
        idx = z["idx"]
        rel = z["rel"]
        curves[tag] = z
        disp = DISPLAY[tag]
        # ---------------- B0 norm ----------------
        f, ax = plt.subplots(figsize=(5.0, 3.3))
        ax.plot(rel, z["m_mean_norm"], color=COLOR[tag], lw=1.5,
                label="mean $\\|h\\|$")
        ax.fill_between(rel, z["m_mean_norm"] - z["m_std_norm"],
                        z["m_mean_norm"] + z["m_std_norm"],
                        color=COLOR[tag], alpha=0.2, lw=0,
                        label="$\\pm$1 std over 1200 states")
        ax.plot(rel[0], z["m_mean_norm"][0], "o", mfc="none", mec="k",
                ms=7, ls="none")
        ax.plot(rel[-1], z["m_mean_norm"][-1], "*", mec="k", mfc="none",
                ms=11, ls="none")
        ax.set_yscale("log")
        ax.set_xlabel("relative depth $\\ell/L$ (0 = embedding, star = "
                      "post-norm)")
        ax.set_ylabel("state norm $\\|h\\|$")
        ax.set_title(f"{disp}: residual-stream norm grows ~exponentially "
                     f"with depth", fontsize=8.6)
        ax.legend(fontsize=7)
        savefig(f, f"fig_B0_norm_{tag}.png")

        # ---------------- B1 spectra ----------------
        f, ax = plt.subplots(figsize=(5.4, 3.6))
        keep = [int(k[6:]) for k in z.files if k.startswith("lamc_")]
        keep = sorted(set(keep))
        cmap = plt.cm.viridis(np.linspace(0, 0.92, len(keep)))
        for c, ix in zip(cmap, keep):
            lam = z[f"lamc_{ix}"]
            ax.loglog(np.arange(1, len(lam) + 1), lam / lam.sum(),
                      color=c, lw=1.2, label=f"idx {ix} "
                      f"({'emb' if ix == 0 else 'post-norm' if ix == L else 'blk'})")
        ax.axvline(1199, color="k", ls=":", lw=0.8)
        ax.annotate("sample limit $n-1$=1199", (1199, 3e-3), fontsize=6.5,
                    rotation=90, textcoords="offset points", xytext=(-3, 0))
        ax.set_xlabel("PC index $i$")
        ax.set_ylabel("eigenvalue share $\\lambda_i/\\sum\\lambda$")
        ax.set_title(f"{disp}: PCA spectra deepen then flatten; bulk is "
                     f"sample-limited past $i\\approx$1199", fontsize=8.4)
        ax.legend(fontsize=6.0)
        savefig(f, f"fig_B1_pca_spectrum_{tag}.png")

        f, ax = plt.subplots(figsize=(5.4, 3.6))
        for c, ix in zip(cmap, keep):
            lamc = z[f"lamc_{ix}"]
            lamm = z[f"lamm_{ix}"]
            ax.loglog(np.arange(1, len(lamc) + 1), lamc / lamc.sum(),
                      color=c, lw=1.2)
            ax.loglog(np.arange(1, len(lamm) + 1), lamm / lamm.sum(),
                      color=c, lw=1.0, ls="--")
        ax.loglog([], [], "k-", label="centered (mean removed)")
        ax.loglog([], [], "k--", label="with mean")
        ax.set_xlabel("PC index $i$")
        ax.set_ylabel("eigenvalue share")
        ax.set_title(f"{disp}: the common mean adds one huge top mode "
                     f"(dashed $>$ solid at $i=1$)", fontsize=8.6)
        ax.legend(fontsize=7)
        savefig(f, f"fig_B1b_mean_vs_centered_{tag}.png")

        # ---------------- B1c measures ----------------
        f, ag = plt.subplots(2, 2, figsize=(8.6, 6.0))
        panels = [("m_pr_c", "participation ratio $PR$ (centered)"),
                  ("m_eff90_c", "eff-rank 90% (centered)"),
                  ("m_topfrac_c", "$\\lambda_1/\\sum\\lambda$ (centered)"),
                  ("m_slope_c", "bulk slope $\\beta$ ($\\lambda\\sim i^{{-\\beta}}$)")]
        for ax, (k, lab) in zip(ag.ravel(), panels):
            ax.plot(rel, z[k], color=COLOR[tag], lw=1.5)
            ax.plot(rel[0], z[k][0], "o", mfc="none", mec="k", ms=7,
                    ls="none")
            ax.plot(rel[-1], z[k][-1], "*", mec="k", mfc="none", ms=11,
                    ls="none")
            ax.set_xlabel("relative depth $\\ell/L$")
            ax.set_ylabel(lab)
        f.suptitle(f"{disp}: PCA dimensionality measures vs depth "
                   "(circle = embedding, star = post-norm)", fontsize=9.5,
                   y=1.01)
        f.tight_layout()
        savefig(f, f"fig_B1c_measures_{tag}.png")

        # ---------------- B2 MLE vs k ----------------
        f, ax = plt.subplots(figsize=(5.2, 3.6))
        for c, ix in zip(cmap, keep):
            ys = [z[f"m_mle{k}"][ix] for k in (5, 10, 20, 50)]
            ax.plot((5, 10, 20, 50), ys, "o-", color=c, ms=3.5,
                    label=f"idx {ix}")
        ax.set_xscale("log")
        ax.set_xlabel("neighbors $k$")
        ax.set_ylabel("Levina-Bickel $\\hat d(k)$")
        ax.set_title(f"{disp}: MLE dimension vs $k$ -- plateau at "
                     f"$\\hat d\\approx$"
                     f"{np.mean([z['m_mle20'][ix] for ix in keep[1:]]):.0f}",
                     fontsize=8.6)
        ax.legend(fontsize=6.0)
        savefig(f, f"fig_B2_mle_vs_k_{tag}.png")

        f, ax = plt.subplots(figsize=(5.2, 3.6))
        for k, st in [(10, "-"), (20, "-"), (50, ":")]:
            ax.plot(rel, z[f"m_mle{k}"], st, color=COLOR[tag], lw=1.4,
                    label=f"LB-MLE $k$={k}")
        ax.plot(rel, z["m_twonn"], "--", color="#EE7733", lw=1.4,
                label="TWO-NN")
        ax.plot(rel, z["m_corrdim"], "-.", color="#228833", lw=1.4,
                label="corr. dim.")
        ax.plot(rel[0], z["m_mle20"][0], "o", mfc="none", mec="k", ms=7,
                ls="none")
        ax.plot(rel[-1], z["m_mle20"][-1], "*", mec="k", mfc="none", ms=11,
                ls="none")
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("intrinsic dimension estimate $\\hat d$")
        ax.set_title(f"{disp}: ID estimators agree to a factor ~2 and "
                     f"both drop with depth", fontsize=8.6)
        ax.legend(fontsize=7)
        savefig(f, f"fig_B2b_id_vs_depth_{tag}.png")

        # ---------------- B3 local dim ----------------
        ldt = z["localdim_t"]                # [L+1, 3, T]
        f, ag = plt.subplots(1, 3, figsize=(10.4, 3.0))
        for ax, ix in zip(ag, [1, L // 2, L - 1]):
            vals = ldt[ix].ravel()
            ax.hist(vals[np.isfinite(vals)], bins=40, color=COLOR[tag],
                    alpha=0.8)
            ax.axvline(np.nanmedian(vals), color="k", ls="--", lw=0.9)
            ax.set_xlabel("local $\\hat d_i$ ($k$=20, within-prompt)")
            ax.set_ylabel("states")
            ax.set_title(f"idx {ix}: med={np.nanmedian(vals):.1f} "
                         f"iqr=[{np.nanpercentile(vals,25):.0f},"
                         f"{np.nanpercentile(vals,75):.0f}]", fontsize=8.2)
        f.suptitle(f"{disp}: per-point local dimension distribution "
                   "(k=20 MLE, 1200 generated states)", fontsize=9.5, y=1.03)
        savefig(f, f"fig_B3_localdim_hist_{tag}.png")

        f, ag = plt.subplots(1, 3, figsize=(10.4, 3.0), sharey=True)
        for ax, ix in zip(ag, [1, L // 2, L - 1]):
            med = np.nanmedian(ldt[ix], axis=0)
            lo, hi = np.nanmin(ldt[ix], axis=0), np.nanmax(ldt[ix], axis=0)
            ax.fill_between(np.arange(ldt.shape[2]), lo, hi,
                            color=COLOR[tag], alpha=0.2, lw=0)
            ax.plot(np.arange(ldt.shape[2]), med, color=COLOR[tag], lw=1.2)
            ax.set_xlabel("position $t$ in trajectory (400 tokens)")
            ax.set_title(f"idx {ix}", fontsize=8.2)
        ag[0].set_ylabel("local $\\hat d_i$ (median over 3 prompts; band "
                         "min/max)")
        f.suptitle(f"{disp}: local dimension is NOT constant along the "
                    f"trajectory; bursts of high-$\\hat d$ states appear",
                   fontsize=9.5, y=1.03)
        savefig(f, f"fig_B3b_localdim_time_{tag}.png")

        f, ax = plt.subplots(figsize=(6.0, 3.4))
        im = ax.imshow(np.nanmedian(ldt, axis=1), aspect="auto",
                       cmap="magma", origin="lower",
                       extent=[0, ldt.shape[2], 0, 1.0])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{int(v*L)}" for v in np.linspace(0, 1, 6)])
        ax.set_xlabel("position $t$ (tokens)")
        ax.set_ylabel("acts index $\\ell$ (0=emb, L=post-norm)")
        plt.colorbar(im, ax=ax, label="local $\\hat d_i$ (median, $k$=20)")
        ax.set_title(f"{disp}: local-dimension map over depth and time",
                     fontsize=8.6)
        savefig(f, f"fig_B3c_localdim_heat_{tag}.png")

        # ---------------- B4 anisotropy ----------------
        f, ag = plt.subplots(1, 2, figsize=(9.0, 3.2))
        ax = ag[0]
        pp = z["pp_mean_cos"]
        ax.fill_between(rel, np.min(pp, axis=1), np.max(pp, axis=1),
                        color=COLOR[tag], alpha=0.2, lw=0)
        ax.plot(rel, z["m_mean_cos"], color=COLOR[tag], lw=1.5,
                label="mean pairwise cos")
        ax.plot(rel, z["m_cone"], "--", color=COLOR[tag], lw=1.4,
                label="cone $\\|\\mu\\|/\\overline{\\|h\\|}$")
        ax.plot(rel[0], z["m_mean_cos"][0], "o", mfc="none", mec="k", ms=7,
                ls="none")
        ax.plot(rel[-1], z["m_mean_cos"][-1], "*", mec="k", mfc="none",
                ms=11, ls="none")
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("anisotropy")
        ax.legend(fontsize=7)
        ax.set_title("mean cosine & cone vs depth", fontsize=8.6)
        ax = ag[1]
        ax.plot(rel, z["m_pr_m"], "--", color=COLOR[tag], lw=1.4,
                label="$PR$ with mean")
        ax.plot(rel, z["m_pr_c"], "-", color=COLOR[tag], lw=1.5,
                label="$PR$ mean-removed")
        ax.plot(rel, z["m_topfrac_m"], ":", color="#EE7733", lw=1.4,
                label="$\\lambda_1/\\sum\\lambda$ with mean")
        ax.plot(rel, z["m_topfrac_c"], "-.", color="#EE7733", lw=1.4,
                label="$\\lambda_1/\\sum\\lambda$ mean-removed")
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("participation / top fraction")
        ax.legend(fontsize=6.6)
        ax.set_title("with-mean vs mean-removed", fontsize=8.6)
        f.suptitle(f"{disp}: anisotropy profile (bands = min/max over 3 "
                   "prompts)", fontsize=9.5, y=1.03)
        savefig(f, f"fig_B4_aniso_{tag}.png")

        # ---------------- B5 write ----------------
        f, ag = plt.subplots(1, 3, figsize=(10.4, 3.1))
        wrel = rel[:-1] + 0.5 / L
        ax = ag[0]
        ax.plot(wrel, z["w_norm_ratio"], "o-", color=COLOR[tag], ms=3,
                lw=1.2)
        ax.plot(wrel[-1], z["w_norm_ratio"][-1], "x", color="k", ms=8,
                ls="none", label="final block + RMSNorm (mixed)")
        ax.set_xlabel("relative depth (write $\\ell\\to\\ell+1$)")
        ax.set_ylabel("$\\overline{\\|\\Delta\\|}/\\overline{\\|h\\|}$")
        ax.set_title("relative write magnitude", fontsize=8.6)
        ax.legend(fontsize=6.6)
        ax = ag[1]
        ax.plot(wrel, z["w_eff90_c"], "o-", color=COLOR[tag], ms=3, lw=1.2,
                label="write eff-rank 90%")
        ax.plot(wrel[:-1], z["m_eff90_c"][:-1], "s--", color="#888888",
                ms=3, lw=1.1, label="state eff-rank 90%")
        ax.plot(wrel[-1], z["w_eff90_c"][-1], "x", color="k", ms=8,
                ls="none")
        ax.set_xlabel("relative depth (write $\\ell\\to\\ell+1$)")
        ax.set_ylabel("eff-rank 90%")
        ax.set_title("write rank vs state rank", fontsize=8.6)
        ax.legend(fontsize=6.6)
        ax = ag[2]
        ax.plot(wrel, z["w_cos_h"], "o-", color=COLOR[tag], ms=3, lw=1.2)
        ax.plot(wrel[-1], z["w_cos_h"][-1], "x", color="k", ms=8, ls="none")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("relative depth (write $\\ell\\to\\ell+1$)")
        ax.set_ylabel("$\\overline{\\cos(\\Delta, h)}$")
        ax.set_title("radial fraction of the write", fontsize=8.6)
        f.suptitle(f"{disp}: per-layer writes are low-rank and mostly "
                   f"tangential (x = post-norm step, not a pure write)",
                   fontsize=9.5, y=1.04)
        savefig(f, f"fig_B5_write_{tag}.png")

    # ================= cross-model depth figures =================
    def overlay(ax, key, ylabel, title, scale=1.0, ppkey=None, ls="-"):
        for tag in tags:
            z = curves[tag]
            rel = z["rel"]
            ax.plot(rel, z[key] / scale, ls, color=COLOR[tag], lw=1.4,
                    label=DISPLAY[tag].replace("-R1-Distill-1.5B", "")
                    .replace("-Instruct", ""))
            if ppkey is not None:
                pp = z[ppkey]                    # [L+1, 3] (idx, prompt)
                ax.fill_between(rel, np.min(pp, axis=1) / scale,
                                np.max(pp, axis=1) / scale,
                                color=COLOR[tag], alpha=0.12, lw=0)
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.0)
        ax.legend(fontsize=6.6)

    f, ag = plt.subplots(2, 3, figsize=(12.4, 6.8))
    overlay(ag[0, 0], "m_pr_c", "$PR$ (mean-removed)", "Participation "
             "ratio drops with depth in 3 of 4 models")
    overlay(ag[0, 1], "m_eff90_c", "eff-rank 90%", "Eff-rank 90% of state "
             "covariance")
    overlay(ag[0, 2], "m_mle20", "LB-MLE $\\hat d$ ($k$=20)", "POOLED "
             "intrinsic dim (3 prompts mixed): union of 3 thin trajectories",
             ppkey="pp_mle20")
    overlay(ag[1, 0], "m_mean_cos", "mean pairwise cos", "State anisotropy "
             "vs depth", ppkey="pp_mean_cos")
    overlay(ag[1, 1], "m_cone", "cone ratio", "Cone ratio vs depth")
    overlay(ag[1, 2], "m_twonn", "TWO-NN $\\hat d$", "TWO-NN dimension vs "
             "depth")
    f.suptitle("Section B6: dimensionality vs relative depth -- all models, "
               "same axes", fontsize=10, y=1.01)
    f.tight_layout()
    savefig(f, "fig_B6_measures_all.png")

    # per-prompt (single-trajectory) ID -- the interpretable version
    f, ag = plt.subplots(1, 3, figsize=(11.6, 3.2))
    for j, (pk, lab) in enumerate([("pp_mle20", "LB-MLE $k$=20"),
                                   ("pp_twonn", "TWO-NN"),
                                   ("pp_corrdim", "corr. dim.")]):
        ax = ag[j]
        for tag in tags:
            z = curves[tag]
            pp = z[pk]                        # [L+1, 3]
            ax.plot(z["rel"], np.median(pp, axis=1), color=COLOR[tag],
                    lw=1.4, label=DISPLAY[tag].replace(
                        "-R1-Distill-1.5B", "").replace("-Instruct", ""))
            ax.fill_between(z["rel"], np.min(pp, axis=1), np.max(pp, axis=1),
                            color=COLOR[tag], alpha=0.12, lw=0)
        ax.set_xlabel("relative depth $\\ell/L$")
        ax.set_ylabel("$\\hat d$ (single prompt, n=400)")
        ax.set_title(f"{lab}: per-trajectory ID", fontsize=9.0)
        ax.legend(fontsize=6.4)
    f.suptitle("Single-trajectory intrinsic dimension (bands over 3 "
               "prompts): 5-15, far above the pooled estimate",
               fontsize=9.5, y=1.03)
    f.tight_layout()
    savefig(f, "fig_B6e_perprompt_id_all.png")

    f, ag = plt.subplots(1, 3, figsize=(11.6, 3.2))
    overlay(ag[0], "m_slope_c", "bulk slope $\\beta$",
            "Spectrum slope: bulk flattens with depth in all models")
    overlay(ag[1], "m_corrdim", "correlation dim $\\nu$",
            "Correlation dimension vs depth")
    overlay(ag[2], "m_topfrac_c", "$\\lambda_1/\\sum\\lambda$ (centered)",
            "Top-PC variance share vs depth")
    f.suptitle("B6 (cont.): spectrum shape and scaling dimension",
               fontsize=10, y=1.03)
    f.tight_layout()
    savefig(f, "fig_B6b_measures_all2.png")

    f, ax = plt.subplots(figsize=(6.4, 3.8))
    for tag in tags:
        z = curves[tag]
        ax.plot(z["m_pr_m"], z["m_pr_c"], "o", ms=4, color=COLOR[tag],
                label=DISPLAY[tag].replace("-R1-Distill-1.5B", "")
                .replace("-Instruct", ""))
    ax.plot([0, 160], [0, 160], "k:", lw=0.8, label="$y=x$")
    ax.set_xlabel("$PR$ WITH mean (uncentered 2nd moment)")
    ax.set_ylabel("$PR$ mean-removed (centered)")
    ax.set_title("The common mean inflates apparent dimensionality "
                 "2-10x (all layers, all models)")
    ax.legend(fontsize=7)
    savefig(f, "fig_B6c_pr_mean_effect.png")

    f, ax = plt.subplots(figsize=(6.4, 3.8))
    for tag in tags:
        z = curves[tag]
        wrel = z["rel"][:-1] + 0.5 / z["L"]
        ax.plot(wrel[:-1] * z["L"], z["w_eff90_c"][:-1], "o-", ms=3,
                color=COLOR[tag], lw=1.2,
                label=DISPLAY[tag].replace("-R1-Distill-1.5B", "")
                .replace("-Instruct", ""))
    ax.set_xlabel("write index $\\ell$ (block writes only)")
    ax.set_ylabel("write eff-rank 90%")
    ax.set_title("Per-layer writes are 2-6x lower-rank than the states they "
                 "update")
    ax.legend(fontsize=7)
    savefig(f, "fig_B5b_write_rank_all.png")

    section_numbers("B_figs_note", {
        "note": "all B figures are built from cache/<tag>_layermeas.npz",
        "tags": tags})


if __name__ == "__main__":
    main()
