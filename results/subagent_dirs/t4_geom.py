"""TEST 4: is a direction's gain predictable from its geometry?

Facts about the direction set (verified):
- The 20 perturbation directions form a REGULAR SIMPLEX in a 20-dim subspace of
  token-embedding space: row i of `simplex` has 0.975 on coordinate i and
  -0.0513 on all others; mutual dot products are all -1/19; every row has the
  same L1/L2 (1.95), same max coefficient (0.975), same sparsity.
  => The ONLY geometric degree of freedom that varies across directions is
  WHICH token axis (subspace[i]) a direction occupies. All simple scalar
  predictors (L1/L2, largest coordinate) are constant by construction.

What we CAN test:
(a) cross-rollout persistence of per-direction gain (already: rho 0.80-0.98)
    => gain is a property of (prompt, token axis), i.e. geometry, not of the
    sampled continuation;
(b) hidden-space geometry of the response: dh_last stores the actual
    delta-h vectors at layer 27 for directions 0,1,2 -> are the responses of
    different directions parallel (common error mode) or independent?
(c) radial vs transverse: decompose dh relative to the baseline hidden state.
"""
import numpy as np
import matplotlib.pyplot as plt

from common import (L27, annot_heat, load_rollouts, load_w100, logd,
                    save_json, savefig, spear)

z = load_w100()
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
cos_h, norm_h0 = z["cos"], z["norm_h0"]
dh_last = z["dh_last"]
simplex, subspace = z["simplex"].astype(np.float64), z["subspace"]
D, T = dh.shape[1], dh.shape[-1]
EX = [10, 9, 13]
cols = ["#d62728", "#1f77b4", "#2ca02c"]

N = {}
G = simplex @ simplex.T
offd = G[np.triu_indices(D, 1)]
N["simplex_facts"] = {
    "diag": float(np.diag(G)[0]), "offdiag_min": float(offd.min()),
    "offdiag_max": float(offd.max()),
    "row_L1_over_L2": float(np.abs(simplex).sum(1).max() / np.linalg.norm(simplex, axis=1).max()),
    "row_max_coeff": [float(x) for x in np.abs(simplex).max(1)[:3]],
    "note": "all rows identical up to permutation; regular simplex",
}

# (a) hot directions and their token axes, across rollouts
ro = load_rollouts()
rmap = {n: {float(rr): i for i, rr in enumerate(d["radii"])} for n, d in ro.items()}
gains_ro = {}
for n, d in ro.items():
    gains_ro[n] = np.median(d["dh_norm"][rmap[n][0.1], :, -1], axis=-1)
hot_tokens = {}
for n in ro:
    top = np.argsort(gains_ro[n])[-3:]
    hot_tokens[n] = [(int(i), int(subspace[i]), float(gains_ro[n][i])) for i in top]
N["hot3_tokenaxes_by_rollout"] = hot_tokens
N["gain_spread_by_rollout"] = {n: float(gains_ro[n].max() / gains_ro[n].min()) for n in ro}

# (b) hidden-space response geometry: dh vectors for dirs 0,1,2 at layer 27
def pair_cos(ridx):
    """median over t of cos(dh_i, dh_j) from dh_last, + cos(dh, h0)."""
    V = dh_last[ridx]                       # [3, T, 1536]
    nrm = np.linalg.norm(V, axis=-1)        # [3, T]
    out = {}
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        cs = (V[i] * V[j]).sum(-1) / np.maximum(nrm[i] * nrm[j], 1e-30)
        out[f"cos({i},{j})"] = float(np.median(cs))
    return out

cos_pairs = {f"r={radii[r]:g}": pair_cos(r) for r in [4, 9, 11]}
N["response_vector_cosines_layer27"] = cos_pairs
print("response vector cosines:", cos_pairs)

# per-token distributions for the figure at r=0.1 and r=2
def pair_cos_t(ridx):
    V = dh_last[ridx]
    nrm = np.linalg.norm(V, axis=-1)
    return [(V[0] * V[1]).sum(-1) / np.maximum(nrm[0] * nrm[1], 1e-30),
            (V[0] * V[2]).sum(-1) / np.maximum(nrm[0] * nrm[2], 1e-30),
            (V[1] * V[2]).sum(-1) / np.maximum(nrm[1] * nrm[2], 1e-30)]
P01 = pair_cos_t(4)
P2 = pair_cos_t(11)
N["cos12_median_r0.1"] = float(np.median(P01[0]))
N["cos12_median_r2"] = float(np.median(P2[0]))

# input-space off-diagonal cosine for reference
N["input_offdiag_cos"] = float(offd.mean())

# (c) angle between the response and the baseline state.
# The stored cos(h',h) equals 1/sqrt(1+(|dh|/|h|)^2) to within 1e-4 (verified),
# i.e. the response is essentially transverse; solving exactly for the radial
# component:  <dh,h> = c|h||h'| - |h|^2, |h'| = c|h| + sqrt(c^2|h|^2 - |h|^2 + |dh|^2)
def cos_dh_h(ridx):
    c = cos_h[ridx, :, L27]
    d = dh[ridx, :, L27]
    h = norm_h0[L27][None, :]
    disc = np.maximum(c * c * h * h - h * h + d * d, 0)
    y = c * h + np.sqrt(disc)
    return (c * y - h) / np.maximum(d, 1e-30)

a2 = cos_dh_h(11)
med_ang = np.median(a2, axis=1)                # per-direction
N["cos_dh_vs_h0_r2"] = {
    "median": float(np.median(a2)), "median_abs": float(np.median(np.abs(a2))),
    "p90_abs": float(np.percentile(np.abs(a2), 90)),
    "genericity_scale": float(1 / np.sqrt(1536)),
    "frac_abs_gt_0.1": float((np.abs(a2) > 0.1).mean()),
    "per_direction": [float(x) for x in med_ang],
}
g01 = np.median(dh[4, :, L27], axis=-1)
N["spearman_absradial_vs_gain"] = spear(np.abs(med_ang), g01)
print("cos(dh,h0) at r=2, L27: median %.3f, |.| median %.3f (genericity %.3f), "
      "frac>0.1: %.2f, spearman(|radial|,gain) = %.2f" %
      (np.median(a2), np.median(np.abs(a2)), 1 / np.sqrt(1536),
       (np.abs(a2) > 0.1).mean(), N["spearman_absradial_vs_gain"]))

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))

ax = axs[0, 0]
S6 = np.abs(simplex[:6, :6])
annot_heat(ax, S6, [f"tok{subspace[i]}" for i in range(6)],
           [f"dir {i}" for i in range(6)],
           "(a) Direction coordinates in the 20-token subspace (6x6 corner)\n"
           f"ALL 20 rows: max coeff 0.975 on own axis, {abs(simplex[0,1]):.3f} on the other 19;\n"
           f"L1/L2 = 1.95 for every direction (no sparsity variation by design)",
           cmap="Blues", vmin=0, vmax=1.0, fmt="{:.2f}", fs=7)

ax = axs[0, 1]
data = [P01[k] for k in range(3)] + [P2[k] for k in range(3)]
lbls = ["dirs 0-1, r=0.1", "dirs 0-2, r=0.1", "dirs 1-2, r=0.1",
        "dirs 0-1, r=2", "dirs 0-2, r=2", "dirs 1-2, r=2"]
bp = ax.boxplot(data, labels=lbls, showfliers=False, patch_artist=True)
for p, c in zip(bp["boxes"], ["#aec7e8"] * 3 + ["#ffbb78"] * 3):
    p.set_facecolor(c)
ax.axhline(0, color="k", lw=0.6, label="cos = 0 (independent responses)")
ax.axhline(offd.mean(), color="r", ls="--", lw=1,
           label=f"input-space cosine between directions = {offd.mean():.3f}")
ax.set_ylabel("cosine between delta-h vectors of two directions (layer 27, per token)")
ax.tick_params(axis="x", rotation=20, labelsize=7)
ax.set_title(f"(b) Do different directions push the hidden state the same way?\n"
             f"median cos(dirs 0,1): {N['cos12_median_r0.1']:.2f} (r=0.1) -> "
             f"{N['cos12_median_r2']:.2f} (r=2)")
ax.legend(fontsize=7)

ax = axs[1, 0]
o = np.argsort(med_ang)
ax.bar(range(D), med_ang[o], color="gray")
for k, d in enumerate(EX):
    ax.bar(np.where(o == d)[0], med_ang[d], color=cols[k], label=f"dir {d}")
ax.set_xticks(range(D)); ax.set_xticklabels([f"{i}" for i in o], fontsize=6)
ax.set_xlabel("direction (sorted by radialness)")
ax.set_ylabel("median over t of cos(delta h, baseline h)\n1 = pure inflation, 0 = pure rotation/deflection")
ax.set_title(f"(c) Radial vs transverse response, r=2, layer 27\n"
             f"median {np.median(a):.2f}; spread 10-90% = "
             f"[{np.percentile(a,10):.2f}, {np.percentile(a,90):.2f}]; "
             f"spearman vs gain = {N['spearman_radialness_vs_gain_r2']:.2f}")
ax.legend(fontsize=7)

ax = axs[1, 1]
o = np.argsort(gains_ro["w100"])
w = 0.27
ax.bar(np.arange(D) - w, gains_ro["w100"][o], width=w, label="w100 continuation")
ax.bar(np.arange(D), gains_ro["w200"][o], width=w, label="w200 continuation")
ax.bar(np.arange(D) + w, gains_ro["w500"][o], width=w, label="w500 continuation")
ax.set_yscale("log")
ax.set_xticks(range(D)); ax.set_xticklabels([f"{i} (tok {subspace[i]})" for i in o],
                                            fontsize=5.5, rotation=90)
ax.set_ylabel("gain at r=0.1, layer 27 (median ||dh_t||, log scale)")
ax.set_xlabel("direction, sorted by w100 gain (token axis id in label)")
ax.set_title(f"(d) SAME direction set, 3 different sampled continuations\n"
             f"gain is a stable property of the token axis: hot trio = dirs "
             f"{np.argsort(gains_ro['w100'])[-3:].tolist()} (tokens "
             f"{[int(subspace[i]) for i in np.argsort(gains_ro['w100'])[-3:]]})")
ax.legend()

fig.suptitle("TEST 4: gain vs geometry - directions differ only by which token axis "
             "they occupy, and that DOES predict gain", y=1.005)
savefig(fig, "fig4_geometry.png")
save_json("numbers_t4.json", N)
print("hot token axes by rollout:", hot_tokens)
print("gain spread by rollout:", N["gain_spread_by_rollout"])
print("saved fig4 + numbers_t4.json")
