"""s2_hidden.py -- Section B: hidden-state geometry, per layer, 4 models.

For every acts index idx in 0..L (0 = embedding, 1..L-1 = raw block outputs,
L = post-final-RMSNorm state -- verified) and pooled over 3 prompts x 400
generated tokens (n=1200):
  B1 PCA spectrum of the centered covariance (via the n x n sample Gram, exact
     nonzero spectrum), PR, stable rank, eff-rank 90/99, bulk log-log slope;
     the same for the UNcentered second moment (with mean) -- both reported.
  B2 intrinsic dimension: Levina-Bickel MLE (k=5,10,20,50), TWO-NN,
     correlation dimension (scaling region 2e-3 <= C(r) <= 0.2).
  B3 local per-point MLE (k=20): distribution (quantiles per layer) and
     per-position time series (neighbors within each prompt's trajectory).
  B4 anisotropy: mean pairwise cosine, cone ratio, PR/top-frac with and
     without the mean.
  B5 per-layer write: Delta_l = h_{l+1} - h_l (block writes for l<=L-2; the
     last one mixes the final block with the final RMSNorm and is flagged).
  B6 all dimensionality measures vs relative depth, all models, one axes.

Results are cached in cache/<tag>_layermeas.npz for reuse in section D.
Layer bands: min/max across the 3 prompts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from dimlib import (load_tag, discover_tags, DISPLAY, COLOR, savefig,
                    section_numbers, eig_desc, pr_lambda, stable_rank,
                    top_frac, eff_rank, powerlaw_slope, pairwise_dists,
                    knn_dists, lb_mle_orig, lb_mle_perpoint, two_nn,
                    corr_dim_from_dists, allpairs_cos, cone_ratio_rows, CACHE)

KLIST = (5, 10, 20, 50)
KMAX = 50
NQ = (5, 10, 25, 50, 75, 90, 95)


def pooled_spectrum(X):
    """Centered + uncentered spectra of X [n,D] via the n x n sample Gram."""
    n = len(X)
    mu = X.mean(0)
    Xc = X - mu
    lam_c = eig_desc(Xc @ Xc.T) / (n - 1)
    lam_m = eig_desc(X @ X.T) / n
    return lam_c, lam_m, mu


def analyze_layer(X, Xps):
    """All pooled + per-prompt measures for one layer."""
    n = len(X)
    lam_c, lam_m, mu = pooled_spectrum(X)
    meas = {
        "pr_c": pr_lambda(lam_c), "srank_c": stable_rank(lam_c),
        "topfrac_c": top_frac(lam_c), "eff90_c": eff_rank(lam_c, 0.9),
        "eff99_c": eff_rank(lam_c, 0.99),
        "slope_c": powerlaw_slope(lam_c, i0=10, i1=600)[0],
        "pr_m": pr_lambda(lam_m), "topfrac_m": top_frac(lam_m),
        "eff90_m": eff_rank(lam_m, 0.9),
        "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
        "std_norm": float(np.linalg.norm(X, axis=1).std()),
        "cone": cone_ratio_rows(X),
        "mu_norm_over_meannorm": float(np.linalg.norm(mu)
                                       / np.linalg.norm(X, axis=1).mean()),
    }
    c = allpairs_cos(X)
    meas["mean_cos"] = float(c.mean())
    # intrinsic dimension (pooled)
    Dm = pairwise_dists(X)
    nn = knn_dists(Dm, KMAX)
    for k in KLIST:
        meas[f"mle{k}"], meas[f"mle{k}_nexcl"] = lb_mle_orig(nn, k,
                                                            True)
    d_i = lb_mle_perpoint(nn, 20)
    meas["mle20_mean_of_perpoint"] = float(np.nanmean(d_i))
    meas["localdim_q"] = np.nanpercentile(d_i, NQ)
    d2, r2_2nn, _ = two_nn(nn)
    meas["twonn"], meas["twonn_r2"] = d2, r2_2nn
    iu = np.triu_indices(n, 1)
    cd, cd_r2, cd_rng = corr_dim_from_dists(Dm[iu])
    meas["corrdim"], meas["corrdim_r2"] = cd, cd_r2
    # per-prompt variants (bands)
    pp = {"mean_cos": [], "pr_c": [], "eff90_c": [], "mle20": [],
          "twonn": [], "corrdim": [], "localdim_t": []}
    for Xp in Xps:
        cc = allpairs_cos(Xp)
        pp["mean_cos"].append(cc.mean())
        lam_cp = eig_desc((Xp - Xp.mean(0)) @ (Xp - Xp.mean(0)).T)
        pp["pr_c"].append(pr_lambda(lam_cp))
        pp["eff90_c"].append(eff_rank(lam_cp, 0.9))
        nn_p_d = pairwise_dists(Xp)
        nn_p = knn_dists(nn_p_d, KMAX)
        pp["mle20"].append(lb_mle_orig(nn_p, 20))
        pp["twonn"].append(two_nn(nn_p)[0])
        iu_p = np.triu_indices(len(Xp), 1)
        pp["corrdim"].append(corr_dim_from_dists(nn_p_d[iu_p])[0])
        d_ip = lb_mle_perpoint(nn_p, 20)
        pp["localdim_t"].append(d_ip)
    for k in pp:
        pp[k] = np.asarray(pp[k])
    return meas, pp, lam_c, lam_m


def analyze_write(Xa, Xb):
    """Measures of Delta = h_{l+1} - h_l for the pooled states."""
    dX = Xb - Xa
    lam_c, _, _ = pooled_spectrum(dX)
    r = np.linalg.norm(dX, axis=1)
    r0 = np.linalg.norm(Xa, axis=1)
    cos = (dX * Xa).sum(1) / np.maximum(r * r0, 1e-9)
    return {
        "norm_ratio": float(r.mean() / r0.mean()),
        "cos_h": float(cos.mean()),
        "frac_radial_gt_0p5": float((np.abs(cos) > 0.5).mean()),
        "cone": cone_ratio_rows(dX),
        "pr_c": pr_lambda(lam_c), "eff90_c": eff_rank(lam_c, 0.9),
        "srank_c": stable_rank(lam_c), "topfrac_c": top_frac(lam_c),
    }


def main():
    tags = discover_tags()
    out = {"per_tag": {}}
    for tag in tags:
        print(f"=== B: {tag}")
        meta, v, a, lg = load_tag(tag)
        D, V, L = int(v["D"][0]), int(v["V"][0]), int(v["L"][0])
        acts = a["acts"]                      # [3, L+1, 400, D] float16
        T = acts.shape[2]
        idxs = np.arange(L + 1)
        M = {k: [] for k in ["pr_c", "srank_c", "topfrac_c", "eff90_c",
                             "eff99_c", "slope_c", "pr_m", "topfrac_m",
                             "eff90_m", "mean_norm", "std_norm", "cone",
                             "mean_cos"] + [f"mle{k}" for k in KLIST]
                 + [f"mle{k}_nexcl" for k in KLIST]
                 + ["mle20_mean_of_perpoint", "twonn", "twonn_r2",
                    "corrdim", "corrdim_r2"]}
        PP = {k: [] for k in ["mean_cos", "pr_c", "eff90_c", "mle20",
                              "twonn", "corrdim"]}
        localdim_t = []
        localdim_q = []
        spec_store = {}
        for ix in idxs:
            X = np.concatenate([acts[p, ix] for p in range(3)]
                               ).astype(np.float32)
            Xps = [acts[p, ix].astype(np.float32) for p in range(3)]
            meas, pp, lam_c, lam_m = analyze_layer(X, Xps)
            for k in M:
                M[k].append(meas[k])
            for k in PP:
                PP[k].append(pp[k])
            localdim_t.append(pp["localdim_t"])     # [3, 400]
            localdim_q.append(meas["localdim_q"])
            spec_store[ix] = (lam_c[:1200].astype(np.float32),
                              lam_m[:1200].astype(np.float32))
            print(f"  idx {ix:2d}/{L}: PR={meas['pr_c']:.1f} "
                  f"eff90={meas['eff90_c']} mle20={meas['mle20']:.1f} "
                  f"2nn={meas['twonn']:.1f} cd={meas['corrdim']:.1f} "
                  f"cos={meas['mean_cos']:.3f}")        # writes
        W = {k: [] for k in ["norm_ratio", "cos_h", "frac_radial_gt_0p5",
                             "cone", "pr_c", "eff90_c", "srank_c",
                             "topfrac_c"]}
        for ix in range(L):
            Xa = np.concatenate([acts[p, ix] for p in range(3)]
                                ).astype(np.float32)
            Xb = np.concatenate([acts[p, ix + 1] for p in range(3)]
                                ).astype(np.float32)
            w = analyze_write(Xa, Xb)
            for k in W:
                W[k].append(w[k])
            print(f"  write {ix}->{ix+1}: ||d||/||h||={w['norm_ratio']:.3f} "
                  f"eff90={w['eff90_c']} cos(d,h)={w['cos_h']:.3f}")
        del acts

        # -------- cache
        keep_spec = sorted({0, 1, L // 4, L // 2, 3 * L // 4, L - 1, L})
        np.savez_compressed(
            os.path.join(CACHE, f"{tag}_layermeas.npz"),
            idx=idxs, rel=idxs / L, D=D, L=L, T=T,
            **{f"m_{k}": np.asarray(M[k], float) for k in M},
            **{f"pp_{k}": np.asarray(PP[k], float) for k in PP},
            localdim_t=np.asarray(localdim_t, dtype=np.float32),
            localdim_q=np.asarray(localdim_q, dtype=np.float32),
            **{f"w_{k}": np.asarray(W[k], float) for k in W},
            **{f"lamc_{ix}": spec_store[ix][0] for ix in keep_spec},
            **{f"lamm_{ix}": spec_store[ix][1] for ix in keep_spec},
        )
        out["per_tag"][tag] = summarize(tag, idxs, M, PP, W, L, D)

    section_numbers("B", out)


def summarize(tag, idxs, M, PP, W, L, D):
    """Compact per-model summary for numbers.json + full curves for figs."""
    def g(k, i):
        return float(M[k][i])
    rec = {
        "D": int(D), "L": int(L),
        "embedding_idx0": {"pr_c": g("pr_c", 0), "eff90_c": g("eff90_c", 0),
                           "mle20": g("mle20", 0), "mean_cos":
                               g("mean_cos", 0)},
        "block1_idx1": {"pr_c": g("pr_c", 1), "eff90_c": g("eff90_c", 1),
                        "mle20": g("mle20", 1), "twonn": g("twonn", 1),
                        "corrdim": g("corrdim", 1),
                        "mean_cos": g("mean_cos", 1), "cone": g("cone", 1),
                        "slope": g("slope_c", 1)},
        "last_raw_block": {"pr_c": g("pr_c", L - 1),
                           "eff90_c": g("eff90_c", L - 1),
                           "eff99_c": g("eff99_c", L - 1),
                           "mle20": g("mle20", L - 1),
                           "twonn": g("twonn", L - 1),
                           "corrdim": g("corrdim", L - 1),
                           "mean_cos": g("mean_cos", L - 1),
                           "cone": g("cone", L - 1),
                           "topfrac_c": g("topfrac_c", L - 1),
                           "slope": g("slope_c", L - 1)},
        "postnorm_idxL": {"pr_c": g("pr_c", L), "eff90_c": g("eff90_c", L),
                          "mle20": g("mle20", L),
                          "mean_cos": g("mean_cos", L),
                          "cone": g("cone", L)},
        "write_mean_over_blocks": {
            k: float(np.mean([W[k][i] for i in range(L - 1)]))
            for k in W},
        "write_first_block": {k: float(W[k][0]) for k in W},
        "write_last_blockish": {k: float(W[k][L - 2]) for k in W},
    }
    return rec


if __name__ == "__main__":
    main()
