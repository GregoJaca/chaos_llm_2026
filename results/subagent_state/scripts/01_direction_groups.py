"""01: Direction groups — are the 32 injection directions equivalent?

Per injection time s:
  - log-response curves per direction, t in [s, 300)
  - pairwise Pearson correlation (full window, and excluding lag<=2 which is
    dominated by the shared immediate transient)
  - hierarchical clustering (average linkage on 1-r), sorted heatmaps
  - stats: median/p10/p90/frac>0.9/min; PC1 share of variance; cluster counts
  - "special" directions: median correlation to all others, flagged if < 0.85
  - cross-s consistency of direction effects (is a direction's deviation from
    the median curve stable across s?)

Figures:
  figures/fig01_dir_corr_heatmaps.png   (2x2, one sorted heatmap per s)
  figures/fig01_dir_curves.png          (1x4, 3 individual + median/IQR per s)
  figures/fig01_pc1.png                 (PC1 shape per s + PC1 share)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from common import load_state, log10safe, save_json, savefig

st = load_state()
dh, steps = st["dh"], st["steps"]
K = dh.shape[1]
stats = {}

# ---------------------------------------------------------------- correlations
def corr_stats(X):
    C = np.corrcoef(X)
    iu = np.triu_indices(K, 1)
    v = C[iu]
    return C, dict(median=float(np.median(v)), p10=float(np.quantile(v, .1)),
                   p90=float(np.quantile(v, .9)), frac_gt_0_9=float(np.mean(v > .9)),
                   min=float(v.min()), n_pairs=int(v.size))

orders = {}
for si, s in enumerate(steps):
    Xfull = log10safe(dh[si, :, s:])                       # full window incl. lag 0
    Xlate = log10safe(dh[si, :, s + 3:])                   # exclude lag<=2
    Cf, sf = corr_stats(Xfull - Xfull.mean(1, keepdims=True))
    Cl, sl = corr_stats(Xlate - Xlate.mean(1, keepdims=True))
    # PC1 share of variance (row-centered SVD)
    U, Sv, Vt = np.linalg.svd(Xfull - Xfull.mean(1, keepdims=True), full_matrices=False)
    pc1_share = float(Sv[0] ** 2 / np.sum(Sv ** 2))
    # clustering on the full-window correlation
    D = np.clip(1 - Cf, 0, 2)
    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)
    orders[int(s)] = order
    nclust_90 = len(np.unique(fcluster(Z, t=0.1, criterion="distance")))       # r<0.9 merges
    nclust_80 = len(np.unique(fcluster(Z, t=0.2, criterion="distance")))       # r<0.8 merges
    # per-direction median correlation to the others
    Cf0 = Cf.copy()
    np.fill_diagonal(Cf0, np.nan)
    medcorr = np.nanmedian(Cf0, axis=1)
    special = [int(i) for i in np.where(medcorr < 0.85)[0]]
    stats[int(s)] = dict(full=sf, excl_lag0_2=sl, pc1_share=pc1_share,
                         pc1_curve=Vt[0].tolist(),
                         n_clusters_r_gt_0_9=nclust_90, n_clusters_r_gt_0_8=nclust_80,
                         per_dir_median_corr=[float(x) for x in medcorr],
                         special_dirs_lt_0_85=special)
    # residual size after removing the median curve (direction effect magnitude)
    med_curve = np.median(dh[si], axis=0)[s:]
    resid = log10safe(dh[si, :, s:]) - log10safe(med_curve)[None, :]
    stats[int(s)]["resid_std_dex"] = float(resid.std())
    stats[int(s)]["resid_p90_abs_dex"] = float(np.quantile(np.abs(resid), .9))

# worst pair (full window)
    iu = np.triu_indices(K, 1)
    k = np.argmin(Cf[iu])
    stats[int(s)]["worst_pair"] = [int(iu[0][k]), int(iu[1][k])]

# cross-s consistency of direction effects: correlate each direction's deviation
# from the per-s median over the common t range, for each s pair.
cross = {}
for i in range(len(steps)):
    for j in range(i + 1, len(steps)):
        t0 = int(max(steps[i], steps[j]))
        zi = log10safe(dh[i, :, t0:]) - log10safe(np.median(dh[i], axis=0)[t0:])[None, :]
        zj = log10safe(dh[j, :, t0:]) - log10safe(np.median(dh[j], axis=0)[t0:])[None, :]
        cc = [float(np.corrcoef(zi[k], zj[k])[0, 1]) for k in range(K)]
        cross[f"s{steps[i]}_vs_s{steps[j]}"] = dict(
            per_dir_corr_median=float(np.median(cc)), per_dir_corr_p10=float(np.quantile(cc, .1)))
stats["cross_s_dir_effect_persistence"] = cross

# flagged directions appearing in >=2 s values
flag_counts = {}
for s in steps:
    for d in stats[int(s)]["special_dirs_lt_0_85"]:
        flag_counts[d] = flag_counts.get(d, 0) + 1
stats["special_dirs_in_ge2_s"] = {int(k): v for k, v in flag_counts.items() if v >= 2}
save_json(stats, "direction_groups.json")

# ---------------------------------------------------------------- figure: heatmaps
fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.5))
for ax, (si, s) in zip(axs.flat, enumerate(steps)):
    X = log10safe(dh[si, :, s:])
    Xc = X - X.mean(1, keepdims=True)
    C = np.corrcoef(Xc)
    o = orders[int(s)]
    im = ax.imshow(C[np.ix_(o, o)], cmap="viridis", vmin=0.3, vmax=1)
    f = stats[int(s)]["full"]
    ax.set_title(f"s={s}: median r={f['median']:.2f}, frac>0.9={f['frac_gt_0_9']:.2f}, "
                 f"min={f['min']:.2f}\nPC1 share={stats[int(s)]['pc1_share']:.2f}", fontsize=9)
    ax.set_xlabel("direction (cluster-sorted)")
    ax.set_ylabel("direction (cluster-sorted)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r (log curves)")
fig.suptitle("Pairwise correlation of log response curves across 32 directions", fontsize=11)
fig.tight_layout()
savefig(fig, "fig01_dir_corr_heatmaps.png")

# ---------------------------------------------------------------- figure: curves
fig, axs = plt.subplots(1, len(steps), figsize=(3.6 * len(steps), 3.6), sharey=True)
for ax, (si, s) in zip(axs, enumerate(steps)):
    x = dh[si][:, s:]
    tt = np.arange(s, dh.shape[-1])
    # representative directions: smallest / median / largest total response
    tot = np.mean(log10safe(x), axis=1)
    d_lo, d_md, d_hi = int(np.argmin(tot)), int(np.argsort(tot)[K // 2]), int(np.argmax(tot))
    for d, lab in [(d_hi, f"d{d_hi} (largest)"), (d_md, f"d{d_md} (median)"), (d_lo, f"d{d_lo} (smallest)")]:
        ax.plot(tt, np.maximum(x[d], 1e-12), lw=0.9, alpha=0.85, label=lab)
    med = np.median(x, axis=0)
    q1, q3 = np.quantile(x, .25, axis=0), np.quantile(x, .75, axis=0)
    ax.plot(tt, med, lw=2, color="k", label="median")
    ax.fill_between(tt, q1, q3, color="k", alpha=0.15, label="IQR (32 dirs)")
    ax.set_yscale("log")
    ax.set_xlabel("generated token t")
    ax.set_title(f"s={s}", fontsize=10)
    ax.grid(alpha=0.25)
axs[0].set_ylabel(r"$\|\delta h_t\|$ (layer 27)")
axs[0].legend(fontsize=7)
fig.suptitle("Response per injection time: 3 individual directions + aggregate", fontsize=11)
fig.tight_layout()
savefig(fig, "fig01_dir_curves.png")

# ---------------------------------------------------------------- figure: PC1
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for si, s in enumerate(steps):
    lag = np.arange(len(stats[int(s)]["pc1_curve"]))
    axs[0].plot(lag, stats[int(s)]["pc1_curve"], lw=1.5, label=f"s={s} (PC1 share {stats[int(s)]['pc1_share']:.2f})")
axs[0].set_xlabel(r"lag $\tau=t-s$")
axs[0].set_ylabel("PC1 of log response (a.u.)")
axs[0].legend(fontsize=8)
axs[0].grid(alpha=0.25)
axs[0].set_title("First principal component of log response curves")
axs[1].bar([str(int(s)) for s in steps], [stats[int(s)]["pc1_share"] for s in steps],
           color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
axs[1].set_ylim(0, 1)
axs[1].set_xlabel("injection time s")
axs[1].set_ylabel("PC1 explained variance share")
axs[1].grid(alpha=0.25, axis="y")
axs[1].set_title("Directions are dominated by one shared response shape")
fig.tight_layout()
savefig(fig, "fig01_pc1.png")

print("\nsummary per s:")
for s in steps:
    f = stats[int(s)]
    print(f"s={s}: median r={f['full']['median']:.3f} p10={f['full']['p10']:.3f} "
          f"frac>0.9={f['full']['frac_gt_0_9']:.2f} min={f['full']['min']:.3f} "
          f"PC1={f['pc1_share']:.3f} resid_std={f['resid_std_dex']:.3f} dex "
          f"special(<0.85)={f['special_dirs_lt_0_85']}")
print("cross-s direction-effect persistence:", cross)
print("special dirs in >=2 s:", stats["special_dirs_in_ge2_s"])
