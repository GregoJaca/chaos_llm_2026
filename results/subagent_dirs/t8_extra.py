"""TEST 8: further falsification checks.

(a) Equivalence score vs radius: median pairwise correlation and PC1 share of
    the per-direction-centered curves -- for the NORM response (log||dh||) and
    the MARGIN response (dm). Is the margin response less 'rank-1' than the
    norm response? Where does the previous 94-97% PC1 number come from?
(b) burstiness: per-direction coefficient of variation across tokens.
(c) does the norm response predict the margin response? (per-direction gains)
(d) the propagation front: identical for all directions (equivalence in TIMING).
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (L27, LIN_R, load_w100, logd, save_json,
                    savefig, spear)

z = load_w100()
dh, dm, radii = z["dh_norm"], z["dm"], z["radii"]
D, T = dh.shape[1], dh.shape[-1]
EX = [10, 9, 13]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
N = {}

def pcstats(M):
    """M [D, T] already row-centered. returns (median pairwise pearson, pc1 share)."""
    C = np.corrcoef(M)
    off = C[np.triu_indices(len(M), 1)]
    Xc = M - M.mean(0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return float(np.median(off)), float(s[0] ** 2 / (s ** 2).sum())

rows = []
for r in range(len(radii)):
    Xd = logd(dh[r, :, L27])
    Xd = Xd - Xd.mean(axis=1, keepdims=True)          # gain removed
    Xdw = logd(dh[r, :, L27]) - logd(dh[r, :, L27]).mean()   # gain kept
    Xm = dm[r].copy()
    Xm = Xm - Xm.mean(axis=1, keepdims=True)
    Xa = np.log(np.abs(dm[r]) + 1e-12)
    Xa = Xa - Xa.mean(axis=1, keepdims=True)
    pw_dh, pc_dh = pcstats(Xd)
    pw_dm, pc_dm = pcstats(Xm)
    pw_adm, pc_adm = pcstats(Xa)
    _, pc_dh_w = pcstats(Xdw)
    rows.append(dict(r=float(radii[r]), pw_dh=pw_dh, pc_dh=pc_dh,
                     pw_dm=pw_dm, pc_dm=pc_dm, pw_adm=pw_adm, pc_adm=pc_adm,
                     pc_dh_gainkept=pc_dh_w))
N["equiv_vs_radius"] = rows
i01 = 4
N["reproduction_of_previous_claim_r0.1"] = {
    "pc1_share_gain_kept": rows[i01]["pc_dh_gainkept"],
    "pc1_share_gain_removed": rows[i01]["pc_dh"],
    "median_pairwise_corr_dh": rows[i01]["pw_dh"],
    "median_pairwise_corr_dm": rows[i01]["pw_dm"],
}
print("=== TEST 8 ===")
for x in rows:
    print("  r=%-6g pairwise dh %.3f |dm| %.3f dm %.3f | PC1 dh %.3f (kept %.3f) |dm| %.3f"
          % (x["r"], x["pw_dh"], x["pw_adm"], x["pw_dm"], x["pc_dh"],
             x["pc_dh_gainkept"], x["pc_adm"]))
print("spearman pairwise dh vs dm over radii:",
      round(spear(np.array([x['pw_dh'] for x in rows]), np.array([x['pw_dm'] for x in rows])), 3))

# (b) burstiness at r=0.1
Xd = dh[4, :, L27]
cv = Xd.std(axis=1) / Xd.mean(axis=1)
N["cv_across_tokens_r0.1"] = {"min": float(cv.min()), "max": float(cv.max()),
                              "median": float(np.median(cv))}
# (c) norm gain vs margin gain
g_dh = np.median(dh[4, :, L27], axis=-1)
g_dm = np.median(np.abs(dm[4]), axis=-1)
N["spearman_dhgain_vs_dmgain_r0.1"] = spear(np.log(g_dh), np.log(g_dm))
N["spearman_dhgain_vs_dmgain_r1"] = spear(np.log(np.median(dh[9, :, L27], axis=-1)),
                                          np.log(np.median(np.abs(dm[9]), axis=-1)))
print("burstiness CV: median %.2f range %.2f..%.2f" % (np.median(cv), cv.min(), cv.max()))
print("spearman(norm gain, |dm| gain): r=0.1 %.3f  r=1 %.3f" %
      (N["spearman_dhgain_vs_dmgain_r0.1"], N["spearman_dhgain_vs_dmgain_r1"]))

# (d) propagation front (identical for all directions)
resp = (dh[4, :, 1:4] > 0)          # [D, 3, T] layer indices 1,2,3
frac_dirs = resp.mean(axis=0)       # [3, T]
fronts = [int((frac_dirs[l] > 0).sum()) for l in range(3)]
N["front_tokens_layers_1_2_3"] = fronts

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

ax = axs[0, 0]
rr = [x["r"] for x in rows]
ax.plot(rr, [x["pw_dh"] for x in rows], "o-", color="C0",
        label="norm response log||dh||: median pairwise corr")
ax.plot(rr, [x["pw_adm"] for x in rows], "^-", color="C3",
        label="margin magnitude log|dm|: median pairwise corr")
ax.plot(rr, [x["pw_dm"] for x in rows], "s-", color="C1",
        label="margin response dm (SIGNED): median pairwise corr")
ax.plot(rr, [x["pc_dh"] for x in rows], "o--", color="C0", alpha=0.6,
        label="norm: PC1 share (gain removed)")
ax.plot(rr, [x["pc_dh_gainkept"] for x in rows], ":", color="C2", lw=2,
        label="norm: PC1 share (gain kept) ~ the previous 94-97% type of claim")
ax.set_xscale("log")
ax.set_xlabel("perturbation radius r")
ax.set_ylabel("correlation / PC1 variance share of per-direction curves (layer 27)")
ax.set_title(f"(a) How 'equivalent' are the 20 responses? (all 300 tokens)\n"
             f"signed margin responses are ~UNCORRELATED across directions "
             f"(median r = {rows[i01]['pw_dm']:.2f}); magnitudes share the token profile")
ax.legend(fontsize=6.5)
ax.set_ylim(-0.1, 1.05)

ax = axs[0, 1]
o = np.argsort(cv)
ax.bar(range(D), cv[o], color="gray")
for k, d in enumerate(EX):
    ax.bar(np.where(o == d)[0], cv[d], color=cols[k], label=f"dir {d}")
ax.set_xticks(range(D)); ax.set_xticklabels([f"{d}" for d in o], fontsize=6)
ax.set_xlabel("direction (sorted by burstiness)")
ax.set_ylabel("std_t ||dh_t|| / mean_t ||dh_t||  at r=0.1, layer 27")
ax.set_title(f"(b) Burstiness across tokens: median CV {np.median(cv):.2f}, "
             f"range {cv.min():.2f}..{cv.max():.2f}\n"
             f"high-CV directions respond in spikes at particular tokens")
ax.legend(fontsize=7)

ax = axs[1, 0]
ax.scatter(np.log(g_dh), np.log(g_dm), s=40, color="gray")
for k, d in enumerate(EX):
    ax.scatter(np.log(g_dh[d]), np.log(g_dm[d]), s=70, color=cols[k], zorder=3,
               label=f"dir {d}")
ax.set_xlabel("log norm-gain (median ||dh||, r=0.1)")
ax.set_ylabel("log margin-gain (median |dm|, r=0.1)")
ax.set_title(f"(c) Norm response only partially predicts margin response\n"
             f"spearman = {N['spearman_dhgain_vs_dmgain_r0.1']:.2f} (r=0.1), "
             f"{N['spearman_dhgain_vs_dmgain_r1']:.2f} (r=1): a direction can be big in "
             f"||dh|| yet mild in margin impact")
ax.legend(fontsize=7)

ax = axs[1, 1]
t = np.arange(T)
for l, c in zip(range(3), ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    ax.step(t, frac_dirs[l], where="post", color=c, lw=1.4,
            label=f"layer {l+1}: responds on tokens 0..{fronts[l]-1}")
ax.set_xlabel("generated-token position t")
ax.set_ylabel("fraction of the 20 directions with dh > 0 at r=0.1")
ax.set_title(f"(d) Propagation front: identical for ALL directions and radii\n"
             f"fronts at t = {fronts} (layers 1,2,3) - timing is a model property, "
             f"not a direction property")
ax.legend(fontsize=7)
ax.set_ylim(-0.05, 1.1)

fig.suptitle("TEST 8: additional equivalence checks "
             "(rank-1-ness vs radius, burstiness, norm-vs-margin coupling, timing)", y=1.005)
savefig(fig, "fig8_extra.png")
save_json("numbers_t8.json", N)
