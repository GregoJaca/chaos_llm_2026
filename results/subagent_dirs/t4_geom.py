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

from common import (BASE, L27, annot_heat, load_rollouts, load_w100, logd,
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

# (c) angle between the response and the baseline state -- DIRECT computation
# using the stored baseline hidden states (margins/w100.npz, h27; token axis
# verified equal via gen_ids/top1). Only directions 0,1,2 have stored dh
# vectors (dh_last). The stored `cos` array equals 1/sqrt(1+(|dh|/|h|)^2) to
# ~1e-4 (verified), i.e. it is a derived near-transversality diagnostic; the
# direct vectors are authoritative for the radial component.
zm = np.load(BASE + r"\margins\w100.npz")
h27 = zm["h27"].astype(np.float64)[:300]
assert np.array_equal(zm["top1"][:300], z["top1"]), "token alignment failed"
cos_direct = {}
for ridx, rname in [(4, "r=0.1"), (11, "r=2")]:
    per_dir = []
    for d in range(3):
        v = dh_last[ridx, d]
        cs = (v * h27).sum(-1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(h27, axis=-1))
        per_dir.append(cs)
    C = np.stack(per_dir)                       # [3, T]
    cos_direct[rname] = C
    N[f"cos_dh_h27_direct_{rname}"] = {
        "median_abs": float(np.median(np.abs(C))),
        "frac_abs_gt_0.1": float((np.abs(C) > 0.1).mean()),
        "per_dir_median": [float(np.median(C[d])) for d in range(3)],
    }
print("direct cos(dh,h27) r=2: per-dir medians",
      N["cos_dh_h27_direct_r=2"]["per_dir_median"],
      "|median|", round(N["cos_dh_h27_direct_r=2"]["median_abs"], 4))
# transversality check on the stored cos array (for the record)
c = cos_h[11, :, L27]; dnn = dh[11, :, L27]; h = norm_h0[L27][None, :]
pred = 1.0 / np.sqrt(1.0 + (dnn / h) ** 2)
N["stored_cos_vs_transverse_formula"] = {
    "corr": float(np.corrcoef(c.ravel(), pred.ravel())[0, 1]),
    "mean_abs_diff": float(np.abs(c - pred).mean()),
    "note": "stored cos is a derived quantity; do not use it as an independent measurement",
}

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
bp = ax.boxplot(data, tick_labels=lbls, showfliers=False, patch_artist=True)
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
C2 = cos_direct["r=2"]
bp = ax.boxplot([C2[d] for d in range(3)],
                tick_labels=["dir 0", "dir 1", "dir 2"],
                showfliers=False, patch_artist=True)
for p, c in zip(bp["boxes"], cols):
    p.set_facecolor(c); p.set_alpha(0.6)
ax.axhline(0, color="k", lw=0.6)
ax.axhline(1 / np.sqrt(1536), color="r", ls="--", lw=1,
          label="genericity scale 1/sqrt(1536) = 0.026")
ax.axhline(-1 / np.sqrt(1536), color="r", ls="--", lw=1)
ax.set_xlabel("direction (vectors stored only for dirs 0,1,2)")
ax.set_ylabel("cos(delta h_t, baseline h_t) per token, r=2, layer 27\n0 = pure deflection of the state")
ax.set_title(f"(c) Radial component of the response (direct vectors)\n"
             f"|cos| median {N['cos_dh_h27_direct_r=2']['median_abs']:.3f}; "
             f"{N['cos_dh_h27_direct_r=2']['frac_abs_gt_0.1']*100:.0f}% of tokens |cos|>0.1; "
             f"per-dir medians {[round(x,3) for x in N['cos_dh_h27_direct_r=2']['per_dir_median']]}")
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
