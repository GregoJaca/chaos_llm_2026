"""s3_logits.py -- Section C: logit space.

C1  SVD of the 128 x V logit matrix (prompt 1, generated positions):
    spectrum (via the 128x128 Gram), PR, stable rank, eff-rank 90/99,
    power-law fit; comparison with the vocabulary spectrum and with the
    PCA of the post-norm states at the SAME 128 positions (how much
    dimensionality does the unembedding head preserve?).
C2  decision subspace: top-2 logit gap distribution per position; the rank
    needed to explain 90% (and 50/70/80/90/95/99%) of the variance of the
    centered logits (logits - mean logit over positions).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dimlib import (load_tag, discover_tags, DISPLAY, COLOR, CE, CW,
                    savefig, section_numbers, eig_desc, pr_lambda,
                    stable_rank, top_frac, eff_rank, powerlaw_slope,
                    pearson)

RANKS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def main():
    tags = discover_tags()
    out = {"per_tag": {}}
    gaps_all, cum_all, rank90_all = {}, {}, {}
    for tag in tags:
        print(f"=== C: {tag}")
        meta, v, a, lg = load_tag(tag)
        D, V, L = int(v["D"][0]), int(v["V"][0]), int(v["L"][0])
        Lg = lg["logits"].astype(np.float32)          # [128, V]
        npos = Lg.shape[0]
        ids = lg["ids"]

        # ---- C1 spectrum of the logit matrix
        G = Lg @ Lg.T / V                            # [128,128], scale-free
        lam = eig_desc(G)                             # descending, clipped
        s = np.sqrt(np.maximum(lam, 0))
        spec = {
            "PR_lambda": pr_lambda(lam),
            "stable_rank": stable_rank(s),
            "top_var_frac": top_frac(s),
            "eff_rank_90": eff_rank(lam, 0.9),
            "eff_rank_99": eff_rank(lam, 0.99),
            "powerlaw_alpha_i3_64": powerlaw_slope(s, i0=3, i1=64)[0],
            "n_positions": int(npos), "V": int(V),
        }
        # vocabulary spectrum for comparison (normalized)
        sE = v["sE"].astype(np.float64)
        # post-norm states at the same positions (acts idx L, prompt 0)
        acts = a["acts"]
        H = acts[0, L, :npos].astype(np.float32)      # [128, D]
        Hc = H - H.mean(0)
        lamH = eig_desc(Hc @ Hc.T / D)
        spec_state = {
            "PR_lambda": pr_lambda(lamH),
            "stable_rank": stable_rank(np.sqrt(np.maximum(lamH, 0))),
            "eff_rank_90": eff_rank(lamH, 0.9),
            "eff_rank_99": eff_rank(lamH, 0.99),
            "mean_norm": float(np.linalg.norm(H, axis=1).mean()),
        }
        del acts

        # ---- C2 gaps + decision subspace
        top2 = np.partition(Lg, -2, axis=1)[:, -2:]
        top1 = top2.max(axis=1)
        gap = top1 - top2.min(axis=1)
        gap_stats = {
            "mean": float(gap.mean()), "std": float(gap.std()),
            "median": float(np.median(gap)),
            "q05": float(np.percentile(gap, 5)),
            "q95": float(np.percentile(gap, 95)),
            "min": float(gap.min()), "max": float(gap.max()),
        }
        Y = Lg - Lg.mean(0, keepdims=True)           # centered over positions
        lamY = eig_desc(Y @ Y.T / V)
        cum = np.cumsum(lamY) / lamY.sum()
        rk = {f"rank_{int(q*100)}": eff_rank(lamY, q) for q in RANKS}
        # gap vs position
        gaps_all[tag] = gap
        cum_all[tag] = cum
        rank90_all[tag] = rk["rank_90"]

        out["per_tag"][tag] = {"logit_spectrum": spec,
                               "postnorm_state_same_positions": spec_state,
                               "top2_gap": gap_stats,
                               "centered_logit_ranks": rk,
                               "rank_eff90_logit_over_state":
                                   rk["rank_90"] / spec_state["eff_rank_90"]}
        print(f"  logit: PR={spec['PR_lambda']:.1f} "
              f"eff90={spec['eff_rank_90']}/{npos} "
              f"alpha={spec['powerlaw_alpha_i3_64']:.2f}")
        print(f"  state(same pos): PR={spec_state['PR_lambda']:.1f} "
              f"eff90={spec_state['eff_rank_90']}")
        print(f"  gap: mean={gap_stats['mean']:.2f} "
              f"q05={gap_stats['q05']:.2f} q95={gap_stats['q95']:.2f}")
        print(f"  centered ranks: {rk}")

        # ---------------- figures ----------------
        f, ax = plt.subplots(figsize=(5.0, 3.4))
        ii = np.arange(1, npos + 1)
        ax.loglog(ii, s / s[0], "o-", ms=3, color=COLOR[tag],
                  label=f"logits $s_i/s_1$ (n={npos})")
        ax.loglog(ii, (sE[:npos] / sE[0]), color=CE, lw=1.2,
                  label="vocab $E$: $s_i/s_1$")
        ax.set_xlabel("singular index $i$")
        ax.set_ylabel("normalized singular value")
        ax.set_title(f"{DISPLAY[tag]}: logits use "
                     f"{spec['eff_rank_90']}/{npos} dims for 90% var.; "
                     f"spectrum flatter than vocab", fontsize=8.6)
        ax.legend(fontsize=7.5)
        savefig(f, f"fig_C1_logit_spectrum_{tag}.png")

        f, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.plot(np.arange(npos), gap, lw=1.0, color=COLOR[tag])
        ax.axhline(gap.mean(), ls="--", lw=0.8, color="k",
                   label=f"mean={gap.mean():.1f}")
        ax.axhline(np.median(gap), ls=":", lw=0.8, color="k",
                   label=f"median={np.median(gap):.1f}")
        ax.set_xlabel("generated position $t$ (tokens)")
        ax.set_ylabel("top-2 logit gap")
        ax.set_title(f"{DISPLAY[tag]}: decision margin fluctuates "
                     f"${gap_stats['q05']:.0f}-{gap_stats['q95']:.0f}$ "
                     f"(q05-q95)", fontsize=8.6)
        ax.legend(fontsize=7.5)
        savefig(f, f"fig_C2_gap_timeseries_{tag}.png")

        f, ax = plt.subplots(figsize=(4.6, 3.2))
        bins = np.linspace(0, np.percentile(gap, 99), 60)
        ax.hist(gap, bins=bins, color=COLOR[tag], alpha=0.75)
        ax.set_xlabel("top-2 logit gap")
        ax.set_ylabel("positions per bin")
        ax.set_title(f"{DISPLAY[tag]}: gap distribution "
                     f"(median={np.median(gap):.1f})", fontsize=8.6)
        savefig(f, f"fig_C2b_gap_hist_{tag}.png")

        f, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.plot(np.arange(1, npos + 1), cum, color=COLOR[tag], lw=1.4)
        for q, c in [(0.9, "k")]:
            k = rk["rank_90"]
            ax.axhline(q, ls=":", lw=0.8, color=c)
            ax.axvline(k, ls=":", lw=0.8, color=c)
            ax.annotate(f"90% at rank {k}/{npos}",
                        (k, q), textcoords="offset points",
                        xytext=(6, -12), fontsize=7.5)
        ax.set_xlabel("rank $k$ of centered logits")
        ax.set_ylabel("cumulative variance fraction")
        ax.set_title(f"{DISPLAY[tag]}: decision subspace rank "
                     f"{rk['rank_90']}/{npos}", fontsize=8.6)
        savefig(f, f"fig_C2c_cumvar_{tag}.png")

    # ---- cross-model logit measures
    f, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(tags))
    for j, (key, lab, c) in enumerate([
            ("eff_rank_90", "logits eff-rank 90%", COLOR["deepseek"]),
            ("eff_rank_99", "logits eff-rank 99%", "#999999")]):
        vals = [out["per_tag"][t]["logit_spectrum"][key] / 128
                for t in tags]
        ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, color=c if j == 0 else
               "#bbbbbb", label=lab)
    sv = [out["per_tag"][t]["postnorm_state_same_positions"]
          ["eff_rank_90"] / 128 for t in tags]
    ax.bar(x + 0.7, sv, 0.35, color=CE, label="post-norm states eff-rank 90%")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[t].replace("-R1-Distill-1.5B", "")
                        .replace("-Instruct", "").replace("-mini", " m")
                        for t in tags], fontsize=8)
    ax.set_ylabel("eff-rank / n positions (128)")
    ax.set_title("The unembedding collapses state dimensionality: logit "
                 "eff-rank < state eff-rank at the same positions")
    ax.legend(fontsize=7.5)
    savefig(f, "fig_C1b_rank_bars.png")

    f, ax = plt.subplots(figsize=(5.6, 3.4))
    for tag in tags:
        ax.plot(np.arange(1, 129), cum_all[tag][:128], color=COLOR[tag],
                lw=1.3, label=DISPLAY[tag].replace("-R1-Distill-1.5B", "")
                .replace("-Instruct", ""))
    ax.axhline(0.9, ls=":", lw=0.8, color="k")
    ax.set_xlabel("rank $k$ of centered logits")
    ax.set_ylabel("cumulative variance fraction")
    ax.set_title(f"Decision subspace: 90% of logit variance in "
                 f"{min(rank90_all.values())}-{max(rank90_all.values())} "
                 f"of 128 dims (all models)")
    ax.legend(fontsize=7)
    savefig(f, "fig_C2d_cumvar_all.png")

    section_numbers("C", out)


if __name__ == "__main__":
    main()
