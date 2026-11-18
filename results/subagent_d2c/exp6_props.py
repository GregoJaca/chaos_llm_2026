"""Exp 6: text/state properties vs predictability.

Per trajectory: token-level stats from gen_ids (unique fraction, unigram
entropy, top-token share, repeated-bigram fraction, longest repeated n-gram),
state-level stats (norm CV, PCA participation ratio, #PCs for 90% var, step /
median-distance, loop-screen distance min_P median||x_t - x_{t+P}||/med),
correlated (Spearman, n=16) against the exp3 predictability gaps.

Output: props_results.json, fig6_properties.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common_d2c import (PROMPTS, REP, OUT, load, distm, pair_values, save_json)

plt.rcParams.update({"figure.dpi": 150, "font.size": 8.5,
                     "axes.titlesize": 9.0, "axes.titleweight": "bold",
                     "figure.constrained_layout.use": True})


def token_props(ids):
    n = ids.size
    vals, cnts = np.unique(ids, return_counts=True)
    p = cnts / n
    H = float(-(p * np.log2(p)).sum())
    bi = {}
    for t in range(n - 1):
        k = (int(ids[t]), int(ids[t + 1]))
        bi[k] = bi.get(k, 0) + 1
    bi_rep = sum(v for v in bi.values() if v > 1) / (n - 1)
    # longest repeated k-gram
    longest = 0
    for k in range(2, 60):
        seen = set()
        found = False
        for t in range(n - k + 1):
            key = tuple(int(x) for x in ids[t:t + k])
            if key in seen:
                found = True
                break
            seen.add(key)
        if not found:
            break
        longest = k
    return dict(unique_frac=len(vals) / n, entropy_bits=round(H, 3),
                top_share=float(cnts.max() / n), bigram_rep_frac=round(bi_rep, 3),
                longest_rep_ngram=longest)


def state_props(X):
    Xc = X - X.mean(0)
    D = distm(Xc)
    iu = np.triu_indices(X.shape[0], 1)
    dv = D[iu]
    med = float(np.median(dv))
    lags = np.abs(iu[0] - iu[1])
    steps = np.linalg.norm(np.diff(Xc, axis=0), axis=1)
    lam = np.power(D, 2) / med ** 2
    best = (np.inf, 0)
    for P in range(2, 251):
        m = lags == P
        if m.sum() < 50:
            continue
        v = float(np.median(dv[m]) / med)
        if v < best[0]:
            best = (v, P)
    # PCA stats
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    lam2 = sv ** 2
    pr = float(lam2.sum() ** 2 / (lam2 ** 2).sum())
    n90 = int(np.searchsorted(np.cumsum(lam2) / lam2.sum(), 0.90) + 1)
    nrm = np.linalg.norm(Xc, axis=1)
    return dict(med_dist=round(med, 2), step_over_med=round(float(np.median(steps) / med), 3),
                loop_dist=round(best[0], 3), loop_P=best[1],
                norm_cv=round(float(nrm.std() / nrm.mean()), 4),
                pc_participation=round(pr, 2), n90=n90)


def main():
    import json
    with open(OUT / "pred_results.json") as f:
        pred = json.load(f)
    res = {}
    for name in PROMPTS:
        X, ids = load(name)
        res[name] = dict(**token_props(ids), **state_props(X))
    # correlate with predictability
    keys = list(res[PROMPTS[0]].keys())
    gap_s = np.array([pred[n]["gap_best_surr_h8"] for n in PROMPTS])
    gap_i = np.array([pred[n]["gap_iaaft_h8"] for n in PROMPTS])
    gap_r = np.array([pred[n]["gap_ridge_h8"] for n in PROMPTS])
    knn1 = np.array([pred[n]["nse_knn"][0] for n in PROMPTS])
    corr = {}
    for k in keys:
        v = np.array([res[n][k] for n in PROMPTS], dtype=float)
        if np.std(v) == 0:
            continue
        corr[k] = {m: (round(float(stats.spearmanr(v, g).statistic), 3),
                        round(float(stats.spearmanr(v, g).pvalue), 4))
                   for m, g in (("gap_surr", gap_s), ("gap_iaaft", gap_i),
                                ("gap_ridge", gap_r), ("knn_h1", knn1))}
    save_json(dict(props=res, spearman_vs=corr), "props_results.json")
    for k, v in corr.items():
        print(f"{k:22s} gap_surr rho={v['gap_surr'][0]:+.2f} p={v['gap_surr'][1]:.3f}", flush=True)
    make_fig(res, corr, pred)


def make_fig(res, corr, pred):
    gap_i = np.array([pred[n]["gap_iaaft_h8"] for n in PROMPTS])
    gap_s = np.array([pred[n]["gap_best_surr_h8"] for n in PROMPTS])
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    panels = [("loop_dist", "loop-screen distance min$_P$ median$\\|x_t-x_{t+P}\\|$/med"),
              ("entropy_bits", "token unigram entropy (bits)"),
              ("unique_frac", "fraction of unique tokens"),
              ("pc_participation", "PC participation ratio")]
    for ax, (k, lab) in zip(axes.ravel(), panels):
        v = np.array([res[n][k] for n in PROMPTS], dtype=float)
        ax.plot(v, gap_s, "o", ms=4, color="tab:blue")
        ri = PROMPTS.index(REP)
        ax.plot([v[ri]], [gap_s[ri]], "o", ms=5.5, color="tab:red")
        ax.annotate("renewable", (v[ri], gap_s[ri]), fontsize=6, color="tab:red",
                    xytext=(4, 4), textcoords="offset points")
        rho, p = corr[k]["gap_surr"]
        ax.set_xlabel(lab)
        ax.set_ylabel("NSE gap vs best surrogate, h=8")
        concl = "no association" if p > 0.05 else ("positive" if rho > 0 else "negative")
        ax.set_title(f"{k}: Spearman rho={rho:+.2f} (p={p:.3f}, n=16) -> {concl}", fontsize=7.5)
    fig.suptitle("Fig 6 - What tracks the (weak) predictability gap: text repetition, not state geometry", fontsize=9.5)
    fig.savefig(OUT / "fig6_properties.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
