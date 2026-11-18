"""TEST 1: Is a direction's gain consistent across layers / radii / rollouts?

Gain g[r,d,l] = median over 300 token positions of ||delta h_t||.
A single-scalar-gain model of direction differences predicts Spearman rank
correlation = 1 between the 20-dimensional gain vectors measured in any two
conditions. We measure it and show where it breaks.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from common import (L27, LIN_R, SAT_R, LAYS, annot_heat, load_rollouts,
                    load_w100, pick_examples, save_json, savefig, spear)

z = load_w100()
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
D = dh.shape[1]

# Responsive mask per layer: early layers respond only on a token prefix
# (the perturbation reaches generated positions through a per-layer causal
# front that is identical for every direction and radius).
resp = dh.max(axis=(0, 1)) > 0                # [L, T] union over radii/dirs
front = [int(resp[l].sum()) for l in range(len(layers))]
maskT = np.zeros(dh.shape, dtype=bool)        # [R, D, L, T]
for l in range(len(layers)):
    maskT[:, :, l, :front[l]] = True
g = np.ma.median(np.ma.masked_array(dh, ~maskT), axis=-1).filled(np.nan)  # [R,D,L]
print("responsive token count per layer:", dict(zip(layers.tolist(), front)))
gl = np.log(g)                       # log gain

# example directions by linear gain at layer 27, r=0.1 (index 4)
exv = g[4, :, L27]
hi, mid, lo = pick_examples(exv, None)
EX = [hi, mid, lo]
print("example dirs (hi, mid, lo):", EX,
      "gains r=0.1 L27:", np.array2string(exv[EX], precision=3))

N = {}
N["gain_ratio_maxmin_r0.1_L27"] = float(exv.max() / exv.min())
N["gain_cv_r0.1_L27"] = float(exv.std() / exv.mean())
N["example_dirs"] = EX
N["front_tokens_per_layer"] = dict(zip([int(x) for x in layers], front))
g_r001 = g[2, :, L27]; g_r0001 = g[0, :, L27]
N["gain_ratio_maxmin_r0.01_L27"] = float(g_r001.max() / g_r001.min())
N["gain_ratio_maxmin_r0.001_L27"] = float(g_r0001.max() / g_r0001.min())
N["gain_ratio_maxmin_r2_L27"] = float(g[11, :, L27].max() / g[11, :, L27].min())

# ---------------- rank correlation between layers ----------------
lay_lab = [f"L{l}" for l in layers[LAYS]]
def layer_rho(ridx):
    M = np.eye(len(LAYS))
    for a, la in enumerate(LAYS):
        for b, lb in enumerate(LAYS):
            if a < b:
                ga, gb = g[ridx, :, la], g[ridx, :, lb]
                r = spearmanr(ga, gb).statistic
                M[a, b] = M[b, a] = r
    return M

Mlin = layer_rho(4)   # r = 0.1 (linear regime)
Msat = layer_rho(11)  # r = 2   (saturated regime)
iu = np.triu_indices(len(LAYS), 1)
N["layer_rho_r0.1"] = {"min": float(Mlin[iu].min()), "median": float(np.median(Mlin[iu]))}
N["layer_rho_r2"] = {"min": float(Msat[iu].min()), "median": float(np.median(Msat[iu]))}

# ---------------- rank correlation between radii (layer 27) ----------------
R12 = len(radii)
Mrad = np.eye(R12)
for i in range(R12):
    for j in range(R12):
        if i < j:
            Mrad[i, j] = Mrad[j, i] = spearmanr(g[i, :, L27], g[j, :, L27]).statistic
N["radius_rho_L27"] = {
    "within_linear(r<=0.1)": float(np.median(Mrad[:5, :5][np.triu_indices(5, 1)])),
    "within_sat(r>=0.7)": float(np.median(Mrad[8:, 8:][np.triu_indices(4, 1)])),
    "lin_r0.01_vs_sat_r2": float(Mrad[2, 11]),
    "min_cross": float(Mrad[:5, 8:][~np.eye(5, 4, dtype=bool)].min()) if False else float(np.min(Mrad[:5, 8:])),
}

# ---------------- cross-rollout stability (same prompt+dirs, different continuation) ----
ro = load_rollouts()
prompt_ok = all(np.array_equal(ro[n]["prompt_ids"], ro["w100"]["prompt_ids"]) for n in ro)
simplex_ok = all(np.array_equal(ro[n]["simplex"], ro["w100"]["simplex"]) for n in ro)
gen_ok = not np.array_equal(ro["w200"]["gen_ids"], ro["w100"]["gen_ids"])
print("same prompt across rollouts:", prompt_ok, "| same directions:", simplex_ok,
      "| different continuations:", gen_ok,
      "| w500 gen differs:", not np.array_equal(ro["w500"]["gen_ids"], ro["w200"]["gen_ids"]))

g_ro = {}
for n, d in ro.items():
    gg = np.median(d["dh_norm"], axis=-1)  # [R,D,L]
    g_ro[n] = gg
rmap = {n: {float(rr): i for i, rr in enumerate(d["radii"])} for n, d in ro.items()}
common_r = [0.01, 0.03, 0.1, 0.3]
Mroro = {}
for rr in common_r:
    A = np.eye(3)
    names = ["w100", "w200", "w500"]
    for a in range(3):
        for b in range(3):
            if a < b:
                A[a, b] = A[b, a] = spearmanr(
                    g_ro[names[a]][rmap[names[a]][rr], :, -1],
                    g_ro[names[b]][rmap[names[b]][rr], :, -1]).statistic
    Mroro[rr] = A
N["rollout_rho_L27"] = {f"r={rr}": {"min": float(Mroro[rr][np.triu_indices(3, 1)].min()),
                                     "median": float(np.median(Mroro[rr][np.triu_indices(3, 1)]))}
                        for rr in common_r}

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))

# (a) log-gain vs radius, layer 27, 20 dirs grey + 3 examples
ax = axs[0, 0]
cols = ["#d62728", "#1f77b4", "#2ca02c"]
for d in range(D):
    ax.plot(radii, gl[:, d, L27], color="gray", alpha=0.35, lw=0.8)
lbl = ["highest gain", "median gain", "lowest gain"]
for k, d in enumerate(EX):
    ax.plot(radii, gl[:, d, L27], color=cols[k], lw=2.2,
            label=f"dir {d} ({lbl[k]})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("perturbation radius r (norm units, log scale)")
ax.set_ylabel("gain = median_t ||delta h_t|| at layer 27 (log)")
ax.set_title(f"(a) Gain vs radius, layer 27. Spread at r=0.1: "
             f"max/min = {N['gain_ratio_maxmin_r0.1_L27']:.1f}x, CV = {N['gain_cv_r0.1_L27']:.2f}\n"
             f"rank(Spearman) r=0.01 vs r=2: rho = {Mrad[2, 11]:.2f}  <-- rank reshuffles at large r")
ax.legend(loc="lower right")

# (b) layer x layer rank corr, r=0.1 and r=2
ax = axs[0, 1]
annot_heat(ax, Mlin, lay_lab, lay_lab,
           f"(b) Rank corr of gains between layers at r=0.1 (linear)\n"
           f"median rho = {np.median(Mlin[iu]):.2f}, min = {Mlin[iu].min():.2f}  "
           f"(1.0 = single-scalar-gain model)", vmin=0.0, vmax=1.0)

# (c) radius x radius rank corr at layer 27
ax = axs[1, 0]
annot_heat(ax, Mrad, [f"{r:g}" for r in radii], [f"{r:g}" for r in radii],
           f"(c) Rank corr of layer-27 gains between radii\n"
           f"within r<=0.1: median rho={N['radius_rho_L27']['within_linear(r<=0.1)']:.2f}; "
           f"within r>=0.7: {N['radius_rho_L27']['within_sat(r>=0.7)']:.2f}; "
           f"r=0.01 vs r=2: {Mrad[2, 11]:.2f}",
           vmin=0.0, vmax=1.0, fs=5.2)

# (d) cross-rollout scatter
ax = axs[1, 1]
rr = 0.1
x = g_ro["w100"][rmap["w100"][rr], :, -1]
y = g_ro["w200"][rmap["w200"][rr], :, -1]
ax.scatter(x, y, s=26, color="gray", alpha=0.8)
for k, d in enumerate(EX):
    ax.scatter(x[d], y[d], s=60, color=cols[k], zorder=3, label=f"dir {d}")
lim = [0.8 * min(x.min(), y.min()), 1.25 * max(x.max(), y.max())]
ax.plot(lim, lim, "k--", lw=0.8, label="y=x")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("gain, w100 continuation (median ||dh|| at layer 27, r=0.1)")
ax.set_ylabel("gain, w200 continuation")
rho_all = np.median([Mroro[rr][i, j] for i in range(3) for j in range(3) if i < j])
ax.set_title(f"(d) Same prompt+directions, different sampled continuation\n"
             f"rank corr across the 3 continuations at r=0.1: median rho = {rho_all:.2f}, "
             f"min = {N['rollout_rho_L27']['r=0.1']['min']:.2f}")
ax.legend(loc="upper left")

fig.suptitle("TEST 1: is a direction's gain a consistent scalar? "
             "(20 simplex directions, teacher-forced, w100 prompt)", y=1.005)
savefig(fig, "fig1_gain_consistency.png")
save_json("numbers_t1.json", N)

# extra printout for the report
print("\nlayer rho at r=0.1 (min/median):", Mlin[iu].min(), np.median(Mlin[iu]))
print("layer rho at r=2   (min/median):", Msat[iu].min(), np.median(Msat[iu]))
print("radius rho lin-vs-sat extreme:", Mrad[2, 11])
print("rollout rho per radius:", {rr: Mroro[rr][np.triu_indices(3, 1)] for rr in common_r})
print("gain values at r=0.1 L27 sorted:", np.array2string(np.sort(exv), precision=3))
print("top-3 dirs:", np.argsort(exv)[-3:], "bottom-3:", np.argsort(exv)[:3])
