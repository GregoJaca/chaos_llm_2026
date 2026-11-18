"""TEST 3: clustering done right.

Features: z-scored log||dh_t|| curves at layer 27 (gain removed by z-scoring).
- hierarchical clustering on correlation distance (1 - Pearson), average + complete;
- silhouette for k=2..8 with an EXPLICIT null: common shape + replicate noise;
- stability of cluster labels across radii (0.1/0.3/1/2), layers (14/21/27) and
  the other two baseline rollouts (w200, w500) via adjusted Rand index;
- Gaussian mixture (diagonal EM, numpy) + BIC on curve-PCs and on a compact
  per-direction fingerprint [gain, shape-PCs, large-r exponent, dm sign bias].
Honest reporting of whether k=1 wins.
"""
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import norm

from common import (L14, L21, L27, LIN_R, load_rollouts, load_w100, logd,
                    pearson, save_json, savefig)

z = load_w100()
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
D, T = dh.shape[1], dh.shape[-1]
EX = [10, 9, 13]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
np.random.seed(0)


def zcurves(mat):
    """[D,T] -> z-scored rows (gain removed)."""
    X = logd(mat)
    X = X - X.mean(axis=1, keepdims=True)
    return X / X.std(axis=1, keepdims=True)


def dist_corr(X):
    C = np.corrcoef(X)
    np.fill_diagonal(C, 1.0)
    return 1.0 - C


def silhouette(Dm, labels):
    n = len(labels); s = np.zeros(n)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    for i in range(n):
        same = labels == labels[i]
        a = Dm[i, same].sum() / max(1, same.sum() - 1) if same.sum() > 1 else 0.0
        b = np.inf
        for u in uniq:
            if u == labels[i]:
                continue
            oth = labels == u
            b = min(b, Dm[i, oth].mean())
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def ari(l1, l2):
    from math import comb
    l1 = np.asarray(l1); l2 = np.asarray(l2)
    n = len(l1)
    ct = np.zeros((len(np.unique(l1)), len(np.unique(l2))))
    u1, u2 = np.unique(l1), np.unique(l2)
    for i, a in enumerate(u1):
        for j, b in enumerate(u2):
            ct[i, j] = ((l1 == a) & (l2 == b)).sum()
    sumij = sum(comb(int(v), 2) for v in ct.ravel())
    ai = ct.sum(1); bj = ct.sum(0)
    sumi = sum(comb(int(v), 2) for v in ai)
    sumj = sum(comb(int(v), 2) for v in bj)
    exp = sumi * sumj / comb(n, 2)
    maxi = 0.5 * (sumi + sumj)
    return float((sumij - exp) / (maxi - exp)) if maxi > exp else 1.0


def gmm_bic(F, kmax=4, nrestart=8, maxiter=100):
    """Diagonal-covariance GMM via EM; returns BIC and best labels per k."""
    n, d = F.shape
    out = {}
    for k in range(1, kmax + 1):
        best = None
        for rep in range(nrestart):
            rng = np.random.default_rng(100 * k + rep)
            mu = F[rng.choice(n, k, replace=False)].copy()
            var = np.ones((k, d)) * F.var(0)
            w = np.full(k, 1.0 / k)
            prev = -np.inf
            for _ in range(maxiter):
                # E step (log domain)
                logp = (-0.5 * ((F[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
                        - 0.5 * np.log(var[None]).sum(-1) + np.log(w[None]))
                m = logp.max(1, keepdims=True)
                r = np.exp(logp - m); r /= r.sum(1, keepdims=True)
                # M step
                nk = r.sum(0) + 1e-9
                w = nk / n
                mu = (r[:, :, None] * F[:, None, :]).sum(0) / nk[:, None]
                var = (r[:, :, None] * (F[:, None, :] - mu[None]) ** 2).sum(0) / nk[:, None]
                var = np.maximum(var, 1e-3)
                logL = float(np.log(np.exp(
                    -0.5 * ((F[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
                    - 0.5 * np.log(var[None]).sum(-1) + np.log(w[None])).sum(1) + 1e-300).sum())
                if abs(logL - prev) < 1e-6:
                    break
                prev = logL
            if best is None or logL > best[0]:
                best = (logL, np.argmax(r, 1))
        p = (k - 1) + k * d + k * d
        out[k] = {"bic": float(-2 * best[0] + p * np.log(n)), "labels": best[1].tolist()}
    return out


# ---------------- primary clustering: r=0.3, layer 27 ----------------
r_idx = 6  # r=0.3
X = zcurves(dh[r_idx, :, L27])
Dm = dist_corr(X)
cond = Dm[np.triu_indices(D, 1)]
link_avg = linkage(cond, method="average")
link_cmp = linkage(cond, method="complete")

sil = {}
for name, L in [("average", link_avg), ("complete", link_cmp)]:
    sv = []
    for k in range(2, 9):
        lab = fcluster(L, k, criterion="maxclust")
        sv.append(silhouette(Dm, lab))
    sil[name] = sv
sil_avg = sil["average"]
best_k = int(np.argmax(sil_avg)) + 2

# ---------------- explicit null: common shape + replicate noise ----------------
resid = logd(dh[np.ix_(LIN_R, range(D), [L27])])[:, :, 0]
resid = resid - resid.mean(axis=-1, keepdims=True)     # gain+scale removed per (r,d)
resid = resid - resid.mean(axis=0, keepdims=True)      # replicate residual around per-dir mean shape
sig1 = float(np.sqrt(resid.var(axis=0, ddof=1).mean()))  # noise of ONE radius
common = X.mean(axis=0)

def null_sil(nsim=400, scale=1.0):
    rng = np.random.default_rng(7)
    vals = []
    for _ in range(nsim):
        Xn = common[None, :] + scale * sig1 * rng.standard_normal((D, T))
        Xn = (Xn - Xn.mean(1, keepdims=True)) / Xn.std(1, keepdims=True)
        Dn = dist_corr(Xn)
        Ln = linkage(Dn[np.triu_indices(D, 1)], method="average")
        lab = fcluster(Ln, 2, criterion="maxclust")
        vals.append(silhouette(Dn, lab))
    return np.array(vals)

null1 = null_sil(400, 1.0)
null3 = null_sil(200, 3.0)
pct1 = float((null1 < sil_avg[0]).mean())
pct3 = float((null3 < sil_avg[0]).mean())
NNULL = {"median_null_sil_k2_scale1": float(np.median(null1)),
         "median_null_sil_k2_scale3": float(np.median(null3))}

# ---------------- stability of k=2 / k=3 labels across conditions ----------------
def labels_for(mat):
    Xc = zcurves(mat)
    Dc = dist_corr(Xc)
    Lc = linkage(Dc[np.triu_indices(D, 1)], method="average")
    return [fcluster(Lc, k, criterion="maxclust") for k in (2, 3, 4)]

conds = {
    "r=0.1": labels_for(dh[4, :, L27]),
    "r=0.3": labels_for(dh[6, :, L27]),
    "r=1": labels_for(dh[9, :, L27]),
    "r=2": labels_for(dh[11, :, L27]),
    "L14@0.3": labels_for(dh[6, :, L14]),
    "L21@0.3": labels_for(dh[6, :, L21]),
}
ro = load_rollouts()
rmap = {n: {float(rr): i for i, rr in enumerate(d["radii"])} for n, d in ro.items()}
for nm, ruse in [("w200@0.1", 0.1), ("w500@0.1", 0.1), ("w200@0.3", 0.3), ("w500@0.3", 0.3)]:
    roll, rr = nm.split("@")
    conds[nm] = labels_for(ro[roll]["dh_norm"][rmap[roll][float(rr)], :, -1])

ari_mat = {}
for (na, la), (nb, lb) in itertools.combinations(conds.items(), 2):
    for k in (2, 3):
        ari_mat[(na, nb, k)] = ari(la[k - 2], lb[k - 2])
ari_pairs_k2 = {f"{a} vs {b}": v for (a, b, k), v in ari_mat.items() if k == 2}
ari_pairs_k3 = {f"{a} vs {b}": v for (a, b, k), v in ari_mat.items() if k == 3}
med_ari_k2 = float(np.median(list(ari_pairs_k2.values())))
med_ari_k3 = float(np.median(list(ari_pairs_k3.values())))
frac_ari1_k2 = float(np.mean([v == 1.0 for v in ari_pairs_k2.values()]))

# identify the repeated k=2 partitions
partitions = {}
for nm, labs in conds.items():
    lab = labs[0]
    groups = tuple(sorted([tuple(np.where(lab == u)[0]) for u in np.unique(lab)],
                          key=len))
    partitions.setdefault(groups, []).append(nm)
print("\nk=2 partitions and the conditions producing them:")
for g, nms in partitions.items():
    print("  ", [list(x) for x in g], "<-", nms)
big_part = max(partitions, key=lambda g: len(partitions[g]))
STABLE_K2 = {
    "groups": [list(map(int, x)) for x in big_part],
    "conditions": partitions[big_part],
    "frac_condition_pairs_ARI1_k2": frac_ari1_k2,
}

# ---------------- GMM + BIC ----------------
# (a) on curve PCs (z-scored curves at r=0.3, first 4 PCs)
Xp = X - X.mean(0)
U, s, Vt = np.linalg.svd(Xp, full_matrices=False)
Fp = U[:, :4] * s[:4]
Fpz = (Fp - Fp.mean(0)) / Fp.std(0)
gmm_curves = gmm_bic(Fpz, kmax=4)
best_k_curves = min(gmm_curves, key=lambda k: gmm_curves[k]["bic"])

# (b) on the direction fingerprint: gain, 3 shape PCs, exponent, sign bias
alpha = []
for d in range(D):
    gmed = np.median(dh[6:12, d, L27], axis=-1)
    A = np.polyfit(np.log(radii[6:12]), np.log(gmed), 1)
    alpha.append(float(A[0]))
alpha = np.array(alpha)
dm1 = z["dm"][9]  # r=1
sign_bias = (dm1 < 0).mean(axis=1)
gain01 = np.median(dh[4, :, L27], axis=-1)
Xdev = X - X.mean(0)
Ud = np.linalg.svd(Xdev - Xdev.mean(0), full_matrices=False)[0]
F = np.stack([np.log(gain01), Ud[:, 0], Ud[:, 1], Ud[:, 2], alpha, sign_bias], axis=1)
Fz = (F - F.mean(0)) / F.std(0)
gmm_fp = gmm_bic(Fz, kmax=4)
best_k_fp = min(gmm_fp, key=lambda k: gmm_fp[k]["bic"])
fp_labels2 = np.array(gmm_fp[best_k_fp]["labels"]) if best_k_fp > 1 else np.zeros(D, int)


def param_bootstrap_bestk(F, kmax=4, nsim=100, seed=11):
    """Under H0 'one Gaussian', how often does BIC pick k>1?"""
    rng = np.random.default_rng(seed)
    mu, sd = F.mean(0), F.std(0, ddof=1)
    cnt = {k: 0 for k in range(1, kmax + 1)}
    for _ in range(nsim):
        Fs = mu + sd * rng.standard_normal(F.shape)
        out = gmm_bic(Fs, kmax=kmax, nrestart=4)
        cnt[min(out, key=lambda k: out[k]["bic"])] += 1
    return cnt


boot_curves = param_bootstrap_bestk(Fpz, nsim=100)
boot_fp = param_bootstrap_bestk(Fz, nsim=100)

# 1-D mixture on log gain alone (is the gain distribution 'hot tail vs bulk'?)
gainlog = np.log(gain01)[:, None]
gmm_gain = gmm_bic(gainlog, kmax=3)
best_k_gain = min(gmm_gain, key=lambda k: gmm_gain[k]["bic"])
boot_gain = param_bootstrap_bestk(gainlog, kmax=3, nsim=200)

N = {
    "silhouette_avg_linkage": {f"k={k}": float(v) for k, v in zip(range(2, 9), sil_avg)},
    "silhouette_complete": {f"k={k}": float(v) for k, v in zip(range(2, 9), sil["complete"])},
    "best_k_silhouette": best_k,
    "null_noise_sigma_log_units": sig1,
    "observed_silhouette_k2": float(sil_avg[0]),
    "null_sil_k2_percentile_scale1": pct1,
    "null_sil_k2_percentile_scale3": pct3,
    "ari_stability_median_k2": med_ari_k2,
    "ari_stability_median_k3": med_ari_k3,
    "ari_pairs_k2": ari_pairs_k2,
    "gmm_curves_BIC": {k: v["bic"] for k, v in gmm_curves.items()},
    "gmm_curves_best_k": best_k_curves,
    "gmm_curves_bootstrap_p_gt1": float(sum(v for k, v in boot_curves.items() if k > 1) / 100),
    "gmm_fingerprint_BIC": {k: v["bic"] for k, v in gmm_fp.items()},
    "gmm_fingerprint_best_k": best_k_fp,
    "gmm_fingerprint_bootstrap_p_gt1": float(sum(v for k, v in boot_fp.items() if k > 1) / 100),
    "gmm_fingerprint_labels": {str(k): gmm_fp[k]["labels"] for k in gmm_fp},
    "gmm_1d_gain_BIC": {k: v["bic"] for k, v in gmm_gain.items()},
    "gmm_1d_gain_best_k": best_k_gain,
    "gmm_1d_gain_bootstrap_p_gt1": float(sum(v for k, v in boot_gain.items() if k > 1) / 200),
    "gmm_1d_gain_labels_bestk": gmm_gain[best_k_gain]["labels"] if best_k_gain > 1 else [0] * D,
    "fingerprint_labels_bestk": fp_labels2.tolist(),
    "alpha_spread": {"min": float(alpha.min()), "max": float(alpha.max()),
                     "std": float(alpha.std())},
    "gain_top3_dirs": np.argsort(gain01)[-3:].tolist(),
    "stable_partition_k2": STABLE_K2,
    **NNULL,
}

# hot-direction check across rollouts
hot = {}
for nm, d in ro.items():
    gg = np.median(d["dh_norm"][rmap[nm][0.1], :, -1], axis=-1)
    hot[nm] = np.argsort(gg)[-3:].tolist()
N["gain_top3_dirs_by_rollout"] = hot

print("=== TEST 3 ===")
print("silhouette (avg linkage):", {k: round(v, 3) for k, v in zip(range(2, 9), sil_avg)})
print("silhouette (complete):   ", {k: round(v, 3) for k, v in zip(range(2, 9), sil["complete"])})
print(f"null: sigma1={sig1:.4f} log-units; observed k=2 silhouette {sil_avg[0]:.3f}; "
      f"percentile in null: scale1 {pct1:.2f}, scale3 {pct3:.2f}")
print("median ARI stability: k=2", round(med_ari_k2, 3), " k=3", round(med_ari_k3, 3))
print("worst ARI pairs (k=2):", sorted(ari_pairs_k2.items(), key=lambda kv: kv[1])[:3])
print("GMM BIC on curve PCs:", {k: round(v["bic"], 1) for k, v in gmm_curves.items()},
      "-> best k =", best_k_curves,
      "| bootstrap P(k>1 under 1-Gaussian null) =", sum(v for k, v in boot_curves.items() if k > 1) / 100)
print("GMM BIC on fingerprint:", {k: round(v["bic"], 1) for k, v in gmm_fp.items()},
      "-> best k =", best_k_fp,
      "| bootstrap P(k>1) =", sum(v for k, v in boot_fp.items() if k > 1) / 100)
print("GMM on log gain (1-D):", {k: round(v["bic"], 1) for k, v in gmm_gain.items()},
      "-> best k =", best_k_gain,
      "| bootstrap P(k>1) =", sum(v for k, v in boot_gain.items() if k > 1) / 200,
      "| labels:", gmm_gain[best_k_gain]["labels"] if best_k_gain > 1 else "k=1")
print("fingerprint labels (best k):", fp_labels2)
print("hot dirs per rollout:", hot)

# ---------------- FIGURE ----------------
fig = plt.figure(figsize=(14.5, 9))
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
dendrogram(link_avg, labels=[f"{i}" for i in range(D)], ax=ax, color_threshold=0.55)
ax.set_ylabel("correlation distance (1 - Pearson)")
ax.set_title(f"(a) Dendrogram, z-scored log||dh|| curves, r=0.3, layer 27\n"
             f"silhouette: k=2 {sil_avg[0]:.2f}, k=3 {sil_avg[1]:.2f}, "
             f"best k={best_k} (sil {max(sil_avg):.2f})")
ax.tick_params(axis="x", labelsize=7)

ax = fig.add_subplot(gs[0, 1])
ks = list(range(2, 9))
ax.plot(ks, sil_avg, "o-", label="average linkage")
ax.plot(ks, sil["complete"], "s--", color="C1", label="complete linkage")
ax.axhline(0.25, color="gray", ls=":", label="weak structure rule-of-thumb 0.25")
ax.set_xlabel("number of clusters k")
ax.set_ylabel("mean silhouette (correlation distance)")
ax.set_title(f"(b) Silhouette vs k\n"
             f"null (common shape + noise): observed k=2 silhouette is "
             f"pct {pct1*100:.0f} of null (sigma x1), pct {pct3*100:.0f} (x3)")
ax.legend()

ax = fig.add_subplot(gs[0, 2])
ax.hist(null1, bins=30, alpha=0.6, label=f"null, noise sigma x1 (median {np.median(null1):.2f})")
ax.hist(null3, bins=30, alpha=0.6, label=f"null, noise sigma x3 (median {np.median(null3):.2f})")
ax.axvline(sil_avg[0], color="k", lw=2, label=f"observed k=2 = {sil_avg[0]:.2f}")
ax.set_xlabel("null silhouette at k=2")
ax.set_ylabel("count (simulated datasets)")
ax.set_title("(c) Explicit null: common shape + measured replicate noise\n"
             "400/200 simulated 20-direction datasets")
ax.legend()

ax = fig.add_subplot(gs[1, 0])
names = list(conds.keys())
K2 = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i != j:
            K2[i, j] = ari_mat[(a, b, 2)] if (a, b, 2) in ari_mat else ari_mat[(b, a, 2)]
from common import annot_heat
annot_heat(ax, K2, names, names,
           f"(d) Cluster stability (adjusted Rand index, k=2)\n"
           f"median = {med_ari_k2:.2f}  (1 = same partition, 0 = chance)",
           cmap="RdYlGn", vmin=0, vmax=1, fs=5.5, cbar=True)

ax = fig.add_subplot(gs[1, 1])
ks = sorted(gmm_curves)
ax.bar([k - 0.18 for k in ks], [gmm_curves[k]["bic"] for k in ks], width=0.36,
       label="GMM on curve PCs (shape only)")
ax.bar([k + 0.18 for k in ks], [gmm_fp[k]["bic"] for k in ks], width=0.36,
       color="C1", label="GMM on 6-feature fingerprint (incl. gain, exponent, sign)")
ax.set_xticks(ks)
ax.set_xlabel("number of mixture components k")
ax.set_ylabel("BIC (lower = better)")
ax.set_title(f"(e) BIC: shape-only clusters -> best k={best_k_curves}; "
             f"fingerprint -> best k={best_k_fp}\n"
             f"fingerprint best-k labels: {fp_labels2.tolist()}")
ax.legend()

ax = fig.add_subplot(gs[1, 2])
gp = np.log(np.median(ro['w100']['dh_norm'][rmap['w100'][0.1], :, -1], axis=-1))
ax.scatter(F[:, 0], F[:, 4], s=40, c=fp_labels2, cmap="tab10")
for d in EX:
    ax.annotate(f"dir {d}", (F[d, 0], F[d, 4]), fontsize=8)
ax.set_xlabel("log gain at r=0.1, layer 27 (log-units)")
ax.set_ylabel("large-r exponent alpha (r=0.3..2)")
ax.set_title(f"(f) Fingerprint space (colors = best-k={best_k_fp} GMM clusters)\n"
             f"alpha spread: min {alpha.min():.2f} max {alpha.max():.2f} std {alpha.std():.2f}")

fig.suptitle("TEST 3: are there direction clusters? hierarchical + silhouette + "
             "null model + BIC, layer 27", y=1.005)
savefig(fig, "fig3_cluster.png")
save_json("numbers_t3.json", N)
