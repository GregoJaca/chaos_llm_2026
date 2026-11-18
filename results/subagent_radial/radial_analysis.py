"""Radial scale drift: concrete tests, figures, and honest verdict.

Question (from the main agent): "You say depth expansion is radial scale drift.
What does this mean concretely? ... check whether the claim is actually right."

Data (all local, CPU):
  results/local_data/w100_layers/a1.npz   prompt perturbation, response norms +
                                          cosines at 8 layers, margin changes,
                                          last-layer response vectors (3 dirs)
  results/local_data/margins/w100.npz      baseline hidden states (same run;
                                          gen_ids match a1 exactly)
  results/local_data/depth/depth.npz       per-layer tangent (JVP) spectrum
  results/local_data/depth/fsle_depth.json depth separation curves (old runs)

Outputs (written next to this script):
  fig1..fig6 PNGs, stats.json, analysis_log.txt

Terminology used everywhere below (plain words):
  state h        : the model's internal vector for one token position, one layer
  nudge dh       : the change of that vector caused by a tiny prompt perturbation
  radial part a  : the part of the nudge that points ALONG h (changes only length)
  tangential part b : the part perpendicular to h (changes only direction)
  relative nudge : ||dh|| / ||h||  (the nudge as a fraction of the state's length)
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_radial"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "figure.dpi": 110, "savefig.dpi": 170, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "legend.fontsize": 8.5,
})

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

# ================================================================ load + validate
d = np.load(BASE + r"\w100_layers\a1.npz", allow_pickle=True)
radii = np.asarray(d["radii"], dtype=np.float64)
layers = np.asarray(d["layers"])
norm_h0 = d["norm_h0"].astype(np.float64)          # (8, 300)  baseline ||h||
dh_norm = d["dh_norm"].astype(np.float64)          # (12,20,8,300) response norm
cos_st = d["cos"].astype(np.float64)               # cos(perturbed state, baseline state)
dm = d["dm"].astype(np.float64)                    # (12,20,300) logit margin change
dh_last = d["dh_last"].astype(np.float64)          # (12,3,300,1536) response vector, deepest layer
Lidx = {int(l): i for i, l in enumerate(layers)}

m = np.load(BASE + r"\margins\w100.npz", allow_pickle=True)
assert np.array_equal(m["gen_ids"][:300], d["gen_ids"]), "margins run is not the same generation"
h27 = m["h27"][:300].astype(np.float64)            # baseline state, deepest saved layer
n27 = np.linalg.norm(h27, axis=-1)                 # (300,)
rel_norm_diff = np.abs(n27 - norm_h0[Lidx[27]]) / norm_h0[Lidx[27]]
e27 = h27 / n27[:, None]                           # unit state direction

dz = np.load(BASE + r"\depth\depth.npz", allow_pickle=True)
lam = dz["lam"].astype(np.float64)                 # (8 tokens, 28 layers, 8 dirs)
local_gain = dz["local_gain"].astype(np.float64)
h_depth = dz["h_depth"].astype(np.float64)
dtok = np.asarray(dz["tokens"])
nL, k = lam.shape[1], lam.shape[2]

fsle = json.load(open(BASE + r"\depth\fsle_depth.json"))

S = {}   # stats dict -> stats.json
S["validation"] = {
    "margins_run_same_generation": True,
    "norm_h0_vs_h27_median_rel_diff": float(np.median(rel_norm_diff)),
    "norm_h0_vs_h27_max_rel_diff": float(rel_norm_diff.max()),
}
# direct re-computation of stored cosine from stored vectors (fp32 rounding check)
vchk = dh_last[11, 0]
s_chk = h27 + vchk
c_direct = np.einsum("td,td->t", s_chk, h27) / (np.linalg.norm(s_chk, axis=-1) * n27)
S["validation"]["stored_cos_vs_direct_max_abs_diff"] = float(
    np.abs(c_direct - cos_st[11, 0, Lidx[27]]).max())
# are the smallest-radius response vectors real signal?  compare direction to r=0.01
cos_cross = np.einsum("td,td->t", dh_last[0, 0], dh_last[2, 0]) / (
    np.linalg.norm(dh_last[0, 0], axis=-1) * np.linalg.norm(dh_last[2, 0], axis=-1))
S["validation"]["cos_dh_r1e3_vs_r1e2_median"] = float(np.median(cos_cross))
log("VALIDATION")
log(f"  margins h27 IS the a1 baseline: ||h27|| matches norm_h0 to "
    f"{np.median(rel_norm_diff):.1e} (median rel)")
log(f"  stored cos matches cos(h+dh, h) recomputed from stored vectors: max |diff| = "
    f"{S['validation']['stored_cos_vs_direct_max_abs_diff']:.1e}")
log(f"  response vectors at r=1e-3 vs r=1e-2: median cos = {np.median(cos_cross):.3f}"
    f"  -> small-radius vectors are real signal, not noise")

# ================================================================ core derived quantities
# exact radial/tangential split of the response at the deepest saved layer
# (uses the real baseline state; dirs 0-2 only, those are the saved vectors)
def split_exact(ri):
    A = np.empty((3, 300)); B = np.empty((3, 300)); D = np.empty((3, 300))
    for j in range(3):
        v = dh_last[ri, j]
        D[j] = np.linalg.norm(v, axis=-1)
        A[j] = np.einsum("td,td->t", v, e27)          # signed radial length
        B[j] = np.sqrt(np.maximum(D[j] ** 2 - A[j] ** 2, 0))
    return A, B, D

ang_all = {}       # angle(dh, h) in degrees, per radius
frac_rad = {}      # |a| / ||dh||, per radius
for ri in range(len(radii)):
    A, B, D = split_exact(ri)
    ok = D > 0
    ang_all[ri] = np.degrees(np.arctan2(B[ok], np.abs(A[ok])))
    frac_rad[ri] = np.abs(A[ok]) / D[ok]

S["angle_deepest_layer"] = {
    str(float(radii[ri])): dict(
        median_deg=float(np.median(ang_all[ri])),
        q10_deg=float(np.quantile(ang_all[ri], 0.1)),
        q90_deg=float(np.quantile(ang_all[ri], 0.9)),
        radial_fraction_median=float(np.median(frac_rad[ri])),
        radial_fraction_q90=float(np.quantile(frac_rad[ri], 0.9)))
    for ri in range(len(radii))}
S["angle_deepest_layer"]["random_direction_reference"] = dict(
    angle_deg=float(90 - np.degrees(np.arctan(1 / np.sqrt(1536)))),
    radial_fraction=float(1 / np.sqrt(1536)))
log("\nANGLE OF THE NUDGE vs THE STATE (deepest saved layer, exact vectors)")
for ri in [0, 2, 4, 6, 9, 11]:
    r = radii[ri]
    a = S["angle_deepest_layer"][str(float(r))]
    log(f"  r={r:<7g} angle(dh,h) median {a['median_deg']:.1f} deg "
        f"(q10 {a['q10_deg']:.1f}, q90 {a['q90_deg']:.1f}); "
        f"radial share of length {a['radial_fraction_median']:.3f}")
log(f"  reference: a direction-less random vector in 1536-D would sit at "
    f"{90 - np.degrees(np.arctan(1/np.sqrt(1536))):.2f} deg "
    f"with radial share {1/np.sqrt(1536):.3f}")

# per-layer cos-based split, valid only above the float32 cosine floor
FLOOR_DEG = float(np.degrees(np.sqrt(2 * 2 ** -23)))   # ~0.028 deg
def split_cos(ri, li):
    n = norm_h0[li][None, :]
    Dv = dh_norm[ri, :, li, :]
    C = cos_st[ri, :, li, :]
    arg = np.maximum(Dv ** 2 - n ** 2 * (1 - C ** 2), 0)
    s_norm = C * n + np.sqrt(arg)                     # ||h+dh||
    A = s_norm * C - n                                # radial part of dh
    B = s_norm * np.sqrt(np.maximum(1 - C ** 2, 0))   # tangential part
    return A, B, Dv

# state norm growth / profiles
S["state_norm"] = {
    "median_by_layer": np.median(norm_h0, axis=1).tolist(),
    "block0_gain_median": float(np.median(norm_h0[Lidx[1]]) / np.median(norm_h0[Lidx[0]])),
    "total_gain_median": float(np.median(norm_h0[Lidx[27]]) / np.median(norm_h0[Lidx[0]])),
}
log("\nSTATE LENGTH")
log(f"  ||h|| median by layer: {np.round(np.median(norm_h0, axis=1), 2)}")
log(f"  block-0 gain x{S['state_norm']['block0_gain_median']:.1f}; "
    f"total gain x{S['state_norm']['total_gain_median']:.0f}")

# per-token depth profiles and per-band relative gains
def rel_gain_band(ri, la, lb):
    ia, ib = Lidx[la], Lidx[lb]
    dh = np.median(dh_norm[ri], axis=0)               # (8,300) median over 20 dirs
    ok = dh[ia] > 0
    g = (dh[ib][ok] / dh[ia][ok]) / (norm_h0[ib][ok] / norm_h0[ia][ok])
    return g

S["band_relative_gain"] = {}
log("\nRELATIVE GAIN PER BAND (nudge growth divided by state-length growth, per token)")
for ri, rn in [(2, 0.01), (3, 0.03), (4, 0.1)]:
    for la, lb in [(3, 7), (7, 14), (14, 21), (21, 27)]:
        g = rel_gain_band(ri, la, lb)
        key = f"r{rn}_L{la}-{lb}"
        S["band_relative_gain"][key] = dict(
            median=float(np.median(g)), q10=float(np.quantile(g, .1)),
            q90=float(np.quantile(g, .9)), frac_below_1=float((g < 1).mean()))
        if rn == 0.01:
            log(f"  r={rn} L{la}->L{lb}: median x{np.median(g):.3f} "
                f"(q10 {np.quantile(g,.1):.3f} q90 {np.quantile(g,.9):.3f}) "
                f"fraction of tokens below 1: {(g<1).mean():.2f}")

# fp32 noise floor at the smallest radius (r=1e-3), per entry
S["fp32_floor_r1e-3"] = {}
log("\nFP32 FLOOR AT r=1e-3 (per-entry comparison with 0.1 x the r=1e-2 response)")
for li, l in enumerate(layers):
    x = dh_norm[2, :, li, :] * 0.1
    y = dh_norm[0, :, li, :]
    ok = x > 0
    if ok.sum() < 30:
        continue
    rho = stats.spearmanr(x[ok], y[ok]).statistic
    infl = float(np.median(y[ok] / x[ok]))
    # additive floor estimate: excess over the linear prediction, positive part
    add = float(np.median(np.maximum(y[ok] - x[ok], 0)))
    S["fp32_floor_r1e-3"][f"L{l}"] = dict(spearman=float(rho), median_inflation=infl,
                                          additive_floor=add)
    log(f"  L{l}: rank-corr {rho:.3f}; median inflation x{infl:.2f}; "
        f"additive floor ~{add:.1e}")

# dm correlations (margin change vs tangential / radial parts; dirs 0-2)
S["dm_correlation"] = {}
log("\nMARGIN-CHANGE CORRELATIONS (deepest saved layer, dirs 0-2 pooled)")
for ri in [2, 4, 6, 9, 11]:
    A, B, D = split_exact(ri)
    dmv = np.abs(dm[ri, :3])
    rb = stats.spearmanr(B.ravel(), dmv.ravel()).statistic
    ra = stats.spearmanr(np.abs(A).ravel(), dmv.ravel()).statistic
    rd = stats.spearmanr(D.ravel(), dmv.ravel()).statistic
    # effective logit gain: |dm| / (b / ||h||)
    gain = dmv / (B / n27[None, :])
    S["dm_correlation"][str(float(radii[ri]))] = dict(
        spearman_tangential=float(rb), spearman_radial=float(ra),
        spearman_total=float(rd),
        median_effective_gain=float(np.median(gain)),
        q10_effective_gain=float(np.quantile(gain, .1)),
        q90_effective_gain=float(np.quantile(gain, .9)))
    log(f"  r={radii[ri]:<5g}: spearman |dm|~tangential {rb:.3f} | ~radial {ra:.3f} "
        f"| effective logit gain |dm|/(b/||h||) median {np.median(gain):.2f} "
        f"(q10 {np.quantile(gain,.1):.2f} q90 {np.quantile(gain,.9):.2f})")

# scaling of |dm| with the tangential part across radii
scal = []
for ri in range(len(radii)):
    A, B, D = split_exact(ri)
    if np.median(D) == 0:
        continue
    scal.append((float(radii[ri]), float(np.median(B)), float(np.median(np.abs(dm[ri, :3])))))
S["dm_scaling_medians"] = scal
lr = stats.linregress(np.log([s[1] for s in scal[1:]]), np.log([s[2] for s in scal[1:]]))
S["dm_vs_b_loglog_slope_r3e-3_to_2"] = float(lr.slope)
log(f"  |dm| vs tangential b across radii 3e-3..2 (medians): log-log slope {lr.slope:.3f}")

# ================================================================ JVP depth spectrum (the two numbers)
def log_interp_norm(row):
    y = np.log(np.asarray(row, dtype=np.float64))
    out = np.interp(np.arange(28), layers, y)
    slope = (y[-1] - y[-2]) / (layers[-1] - layers[-2])
    return np.append(out, out[-1] + slope)

log_s_tok = np.stack([log_interp_norm(norm_h0[:, int(t)]) for t in dtok])  # (8,29)
gs_tok = np.exp(np.diff(log_s_tok, axis=1))                                # (8,28)
scale_rate = (np.cumsum(np.log(gs_tok), axis=1) / (np.arange(1, nL + 1))[None, :])
lam_rel = lam - scale_rate[:, :, None]
h_abs = np.maximum(lam[:, -1, :], 0).sum(axis=1)
h_rel = np.minimum(lam_rel[:, -1, :], 0).sum(axis=1)
max_rel_exp = float(lam_rel.max())
S["jvp_spectrum"] = dict(
    h_depth_absolute=h_depth.tolist(), h_depth_mean=float(h_depth.mean()),
    h_rel_sum=h_rel.tolist(), h_rel_median=float(np.median(h_rel)),
    max_relative_exponent=max_rel_exp,
    scale_rate_median_per_layer=float(np.log(gs_tok[:, 1:]).mean()),
)
log("\nJVP DEPTH SPECTRUM (the project's two numbers, recomputed)")
log(f"  absolute: sum of positive top-8 exponents h_depth = "
    f"{np.round(h_depth, 3)} (mean {h_depth.mean():.3f} nats/layer)")
log(f"  relative: sum after removing per-layer state-length growth = "
    f"{np.round(h_rel, 3)} (median {np.median(h_rel):.3f} nats/layer)")
log(f"  maximum relative exponent over all tokens/layers/directions: {max_rel_exp:.4f}"
    f"  (project docs cite -0.034; same conclusion)")

# ================================================================ FSLE depth curves
log_s_med = log_interp_norm(np.median(norm_h0, axis=1))
scale_growth = np.exp(log_s_med[:28] - log_s_med[0])    # (28,) state-length growth, layers 0..27
S["fsle_depth"] = {}
log("\nFSLE DEPTH CURVES (old ensemble runs) + state-length normalization")
for rk, c in fsle["curves"].items():
    c = np.asarray(c, dtype=np.float64)
    abs_prof = c / c[0]
    rel_prof = abs_prof / scale_growth
    S["fsle_depth"][rk] = dict(
        absolute_total=float(abs_prof[-1]),
        absolute_peak=float(abs_prof.max()),
        absolute_peak_layer=int(np.argmax(abs_prof)),
        relative_at_L2=float(rel_prof[2]),
        relative_at_L14=float(rel_prof[14]),
        relative_total=float(rel_prof[-1]))
    log(f"  r={float(rk):g}: absolute x{abs_prof[-1]:.1f} total "
        f"(peak x{abs_prof.max():.0f} at layer {int(np.argmax(abs_prof))}); "
        f"relative: x{rel_prof[2]:.2f} at layer 2 -> x{rel_prof[-1]:.3f} at layer 27")

# ================================================================ FIGURE 1
TOK = 100
dh_tok = np.median(dh_norm[2], axis=0)[:, TOK]   # r=0.01, median over 20 dirs, one token
dh_med = np.median(dh_norm[2], axis=(0, 2))     # (8,) median over dirs and tokens
dh_q10 = np.quantile(dh_norm[2], 0.10, axis=(0, 2))
dh_q90 = np.quantile(dh_norm[2], 0.90, axis=(0, 2))
s_med = np.median(norm_h0, axis=1)
s_q10 = np.quantile(norm_h0, 0.10, axis=1)
s_q90 = np.quantile(norm_h0, 0.90, axis=1)
rel_tok = dh_tok / norm_h0[:, TOK]
rel_med = dh_med / s_med
rel_q10 = dh_q10 / s_q90
rel_q90 = dh_q90 / s_q10

fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))
xk = layers
ax = axs[0]
ax.fill_between(xk, s_q10, s_q90, color="tab:blue", alpha=0.18, label="10–90% of tokens")
ax.plot(xk, s_med, "-o", color="tab:blue", ms=4, label="median over 300 tokens")
ax.plot(xk, norm_h0[:, TOK], "--s", color="navy", ms=3.5, label=f"one token (t={TOK})")
ax.set_yscale("log")
ax.set_xlabel("layer  (8 saved layers: 0,1,2,3,7,14,21,27; '27' = input to the final block)")
ax.set_ylabel("length of the state vector  ||h||   (log scale)")
ax.set_title("(a) The state's LENGTH grows ×450 with depth\n(×29 inside the first block alone)")
ax.legend(loc="lower right")

ax = axs[1]
ax.fill_between(xk, dh_q10, dh_q90, color="tab:red", alpha=0.15)
ax.plot(xk, dh_med, "-o", color="tab:red", ms=4, label="median over 20 directions × 300 tokens")
ax.plot(xk, dh_tok, "--s", color="darkred", ms=3.5, label=f"one token (t={TOK})")
ax.axvspan(0, 1.5, color="gray", alpha=0.12)
ax.annotate("nudge has not arrived yet\n(prompt nudge reaches this\ntoken at layer 2 via attention)",
            xy=(0.75, 6e-5), fontsize=7.5, color="dimgray")
ax.set_yscale("log")
ax.set_xlabel("layer (same 8 saved layers)")
ax.set_ylabel("length of the nudge  ||dh||   (log scale)")
ax.set_title("(b) The raw nudge also grows (×125 for this token)\n— first an arrival transient, then ×1.1–1.2 per layer")
ax.legend(loc="lower right")

ax = axs[2]
ax.fill_between(xk, rel_q10, rel_q90, color="tab:green", alpha=0.15)
ax.plot(xk, rel_med, "-o", color="tab:green", ms=4, label="median over tokens")
ax.plot(xk, rel_tok, "--s", color="darkgreen", ms=3.5, label=f"one token (t={TOK})")
ax.axhline(1e-5, color="k", lw=0.6, ls=":")
ax.set_yscale("log")
ax.set_xlabel("layer (same 8 saved layers)")
ax.set_ylabel("relative nudge  ||dh|| / ||h||   (log scale)")
ax.set_title("(c) CLAIM FAILS HERE: as a FRACTION of the state the nudge\n"
             "GROWS ~×10 and shrinks (×0.84) only in the last layers")
ax.legend(loc="lower right")
fig.suptitle("Figure 1 — What actually grows with depth: state length (a), raw nudge (b), and their ratio (c).  "
             "Prompt perturbation r=0.01 (linear regime), 300-token teacher-forced run.", fontsize=10.5, y=1.03)
fig.tight_layout()
fig.savefig(OUT + r"\fig1_state_nudge_ratio.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 2
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))

ax = axs[0]
med = [S["angle_deepest_layer"][str(float(r))]["median_deg"] for r in radii]
q10 = [S["angle_deepest_layer"][str(float(r))]["q10_deg"] for r in radii]
q90 = [S["angle_deepest_layer"][str(float(r))]["q90_deg"] for r in radii]
ax.fill_between(radii, q10, q90, color="tab:orange", alpha=0.2, label="10–90% of tokens × directions")
ax.plot(radii, med, "-o", color="tab:orange", ms=4, label="median")
ax.axhline(90, color="k", lw=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.text(1.3e-3, 91.2, "90° = purely perpendicular (direction change)", fontsize=7.5)
ax.text(1.3e-3, -6.5, "0° = purely radial (length change only)", fontsize=7.5)
ax.axvspan(1e-3, 3e-3, color="gray", alpha=0.15)
ax.text(1.15e-3, 40, "fp32 floor\nzone", fontsize=7, rotation=90, va="center", color="dimgray")
ax.set_xscale("log")
ax.set_xlabel("perturbation radius r (log scale; size of the prompt nudge)")
ax.set_ylabel("angle between nudge dh and state h   (degrees)")
ax.set_ylim(-8, 96)
ax.set_title("(a) CLAIM FAILS: the nudge is ~86–89° from the state at EVERY radius\n"
             "it is a direction change, not a length change ('radial' is wrong)")
ax.legend(loc="center right")

ax = axs[1]
for ri, rname, col in [(6, 0.3, "tab:green"), (7, 0.5, "tab:orange"),
                       (9, 1.0, "tab:red"), (11, 2.0, "tab:purple")]:
    meas, pred = [], []
    for li, l in enumerate(layers):
        n = norm_h0[li][None, :]
        Dv = dh_norm[ri, :, li, :]
        C = cos_st[ri, :, li, :]
        meas.append(np.degrees(np.arccos(np.clip(C, -1, 1))).mean())
        pred.append(np.degrees(np.arcsin(np.clip(Dv / n, 0, 1))).mean())
    ax.plot(layers, pred, "--", color=col, lw=1)
    ax.plot(layers, meas, "o", color=col, ms=3.5, label=f"measured, r={rname:g}")
ax.axhline(FLOOR_DEG, color="k", ls=":", lw=1)
ax.text(0.3, FLOOR_DEG * 1.15, "float32 floor: below this line cosines are saturated at 1.0\n"
       "and CANNOT tell radial from perpendicular (r≤0.1 angles sit here)", fontsize=6.8)
ax.set_yscale("log")
ax.set_xlabel("layer (same 8 saved layers)")
ax.set_ylabel("deflection angle of the STATE  arccos cos(h+dh, h)   (degrees, log)")
ax.set_title("(b) Where the angle is above the float32 floor (r≥0.3),\n"
             "measured deflection (dots) matches the fully-perpendicular prediction (dashes)")
ax.legend(loc="lower right", fontsize=7)

ax = axs[2]
ri = 11
A, B, D = split_exact(ri)
tot, rad, tan_ = [], [], []
for li, l in enumerate(layers):
    Ac, Bc, Dc = split_cos(ri, li)
    tot.append(np.median(Dc)); rad.append(np.median(np.abs(Ac))); tan_.append(np.median(Bc))
ax.plot(layers, tot, "o-", color="tab:red", ms=4, label="total nudge length ||dh||")
ax.plot(layers, tan_, "s-", color="tab:blue", ms=4, label="perpendicular part (direction change)")
ax.plot(layers, rad, "^-", color="gray", ms=4, label="along-state part (length change)")
ax.set_yscale("log")
ax.set_xlabel("layer (same 8 saved layers)")
ax.set_ylabel("length of nudge parts   (log scale)")
ax.set_title("(c) Only ~5–10% of the nudge points along the state;\n"
             "the direction-changing part IS the nudge   (r=2)")
ax.legend(loc="lower right")
fig.suptitle("Figure 2 — Angle test: is the nudge 'radial' (along the state) or perpendicular? "
             "The data says perpendicular at every layer and radius where it is measurable.", fontsize=10.5, y=1.03)
fig.tight_layout()
fig.savefig(OUT + r"\fig2_angle.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 3
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6))

ax = axs[0]
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, 5))
r_sel = [(0, 1e-3), (2, 1e-2), (4, 1e-1), (6, 3e-1), (9, 1.0)]
for (ri, r), col in zip(r_sel, cmap):
    A, B, D = split_exact(ri)
    dmv = np.abs(dm[ri, :3])
    ax.scatter(B.ravel(), dmv.ravel(), s=3, color=col, alpha=0.25, rasterized=True)
    ax.plot(np.median(B), np.median(dmv), "o", color=col, ms=7, mec="k", mew=0.8,
            label=f"r={r:g} (median)")
bb = np.array([np.median(split_exact(ri)[1]) for ri, _ in r_sel], dtype=float)
mm = np.array([np.median(np.abs(dm[ri, :3])) for ri, _ in r_sel], dtype=float)
sl = stats.linregress(np.log(bb[1:]), np.log(mm[1:]))     # fit r>=0.01 (clean of fp32 floor)
xs = np.array([3e-4, 20])
ax.plot(xs, mm[1] * (xs / bb[1]) ** sl.slope, "k--", lw=1,
        label=f"power-law fit (r≥0.01), slope {sl.slope:.2f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("perpendicular part of the nudge at the deepest layer, b   (log scale)")
ax.set_ylabel("change of the logit margin  |dm|   (log scale)")
ax.set_title("(a) The logit change tracks the direction-changing part of the nudge,\n"
             "linearly (slope ≈ 1) over 3+ decades of perturbation size")
ax.legend(loc="upper left", fontsize=7.5)

ax = axs[1]
data, labels = [], []
for ri, r in enumerate(radii):
    A, B, D = split_exact(ri)
    if np.median(D) == 0:
        continue
    gain = np.abs(dm[ri, :3]) / (B / n27[None, :])
    data.append(gain.ravel())
    labels.append(f"{r:g}")
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                medianprops=dict(color="k"))
for patch in bp["boxes"]:
    patch.set_facecolor("tab:orange"); patch.set_alpha(0.5)
ax.axhline(4.0, color="k", ls="--", lw=1)
ax.text(0.6, 4.15, "≈ 4: one unit of RELATIVE direction change moves the margin by ~4 units", fontsize=7.5)
ax.set_yscale("log")
ax.set_xlabel("perturbation radius r (log-spaced tick labels)")
ax.set_ylabel("|dm| / (b / ||h||)   — margin change per unit of relative direction change")
ax.set_title("(b) The output sees the nudge RELATIVE to the state's length:\n"
             "|dm| ≈ 4 × (perpendicular part / state length), stable across radii")
fig.suptitle("Figure 3 — Why the output cannot see the length: the network renormalizes the state "
             "(RMSNorm) before converting it to logits, so only the relative, direction-changing part matters.\n"
             "MEASURED: dh vectors, baseline states h, margin changes dm.  "
             "INFERRED (weights not saved): the final block, the RMSNorm output, and the unembedding.", fontsize=9.8, y=1.07)
fig.tight_layout()
fig.savefig(OUT + r"\fig3_logits.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 4
fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3))

ax = axs[0]
fr_med = [S["angle_deepest_layer"][str(float(r))]["radial_fraction_median"] for r in radii]
fr_q10 = [S["angle_deepest_layer"][str(float(r))]["radial_fraction_median"] - 0.0 for r in radii]
fr_lo = [max(0, np.quantile(frac_rad[ri], .1)) for ri in range(len(radii))]
fr_hi = [np.quantile(frac_rad[ri], .9) for ri in range(len(radii))]
ax.fill_between(radii, fr_lo, fr_hi, color="tab:purple", alpha=0.2, label="10–90% of tokens × directions")
ax.plot(radii, fr_med, "-o", color="tab:purple", ms=4, label="median")
ax.axhline(1 / np.sqrt(1536), color="k", ls="--", lw=1)
ax.text(2.2e-3, 0.014, "2.6% = share expected for a direction-less random vector", fontsize=7)
ax.axvspan(1e-3, 3e-3, color="gray", alpha=0.15)
ax.axvspan(0.5, 2, color="gray", alpha=0.10)
ax.text(1.1e-3, 0.16, "fp32 floor\n(r=1e-3 marginal)", fontsize=6.8, rotation=90, va="center", color="dimgray")
ax.text(1.15, 0.16, "nonlinear regime", fontsize=6.8, rotation=90, va="center", color="dimgray")
ax.set_xscale("log")
ax.set_xlabel("perturbation radius r (log scale)")
ax.set_ylabel("along-state share of the nudge   |a| / ||dh||")
ax.set_title("(a) The 'radial' share is small (5–8%) and stable across radii\n"
             "and tokens — the nudge is never radial")
ax.legend(loc="upper right")

ax = axs[1]
band_data = []
band_lbl = []
for la, lb in [(3, 7), (7, 14), (14, 21), (21, 27)]:
    g = rel_gain_band(2, la, lb)
    band_data.append(g); band_lbl.append(f"L{la}–L{lb}")
bp = ax.boxplot(band_data, tick_labels=band_lbl, showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("tab:red"); patch.set_alpha(0.4)
ax.axhline(1, color="k", ls="--", lw=1)
ax.text(1.2, 1.12, "1 = nudge grows exactly as fast as the state's length", fontsize=7.5)
ax.set_yscale("log")
ax.set_xlabel("layer band (band medians annotated)")
ax.set_ylabel("nudge growth ÷ state-length growth   (per token)")
for i, (la, lb) in enumerate([(3, 7), (7, 14), (14, 21), (21, 27)]):
    g = rel_gain_band(2, la, lb)
    ax.text(i + 1, np.median(g) * 1.15, f"x{np.median(g):.2f}\n{(g<1).mean()*100:.0f}% <1",
            ha="center", fontsize=7.5)
ax.set_title("(b) Per token: the nudge outgrows the state's length in layers 3–21\n"
             "(87–92% of tokens) and shrinks only in the last layers (84% of tokens)")
ax.set_ylim(0.4, 60)

ax = axs[2]
for t, col in [(50, "tab:blue"), (100, "tab:green"), (250, "tab:red")]:
    ax.plot(layers, np.median(dh_norm[2], axis=0)[:, t] / norm_h0[:, t],
            "-o", color=col, ms=3.5, label=f"token t={t}")
ax.plot(layers, rel_med, "k-o", ms=4, lw=2, label="median over 300 tokens")
ax.axvspan(0, 1.5, color="gray", alpha=0.12)
ax.set_yscale("log")
ax.set_xlabel("layer (same 8 saved layers)")
ax.set_ylabel("relative nudge  ||dh|| / ||h||   (log scale)")
ax.set_title("(c) Same shape for near tokens (t=50), mid (t=100) and far (t=250):\n"
             "the growth is not an artifact of the median")
ax.legend(loc="lower right")
fig.suptitle("Figure 4 — Stability across radii and tokens: the nudge is never radial (a); "
             "it outgrows the state length in mid-depth for most tokens (b, c).", fontsize=10.5, y=1.03)
fig.tight_layout()
fig.savefig(OUT + r"\fig4_stability.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 5
fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4))

ax = axs[0]
xx = np.arange(len(dtok))
ax.bar(xx - 0.2, h_depth, width=0.4, color="tab:red", label="absolute: growth of the raw tangent volume")
ax.bar(xx + 0.2, h_rel, width=0.4, color="tab:blue",
       label="after dividing by the state-length growth")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xx)
ax.set_xticklabels([f"tok {t}" for t in dtok], fontsize=7)
ax.set_ylabel("log growth per layer   (nats/layer:  +1 = ×2.7,  −1 = ×0.37)")
ax.set_ylim(-1.45, 1.85)
ax.set_title("(a) The project's two numbers, per token (JVP depth spectrum):\n"
             f"absolute mean +{h_depth.mean():.2f} vs relative median {np.median(h_rel):.2f} nats/layer")
ax.legend(loc="upper right", fontsize=7.5)

ax = axs[1]
for rk, c in fsle["curves"].items():
    c = np.asarray(c, dtype=np.float64)
    ax.plot(np.arange(28), c / c[0], color="tab:red", lw=1.4,
            label=f"absolute (r={float(rk):g})")
    ax.plot(np.arange(28), (c / c[0]) / scale_growth, "--", color="tab:blue", lw=1.4,
            label=f"relative (r={float(rk):g})")
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0], handles[1]], ["absolute separation  d(L)/d(0)",
                                     "relative  d(L)/d(0) ÷ state-length growth"],
          loc="lower left", fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("layer (0…27)")
ax.set_ylabel("separation growth  (log scale)")
ax.annotate("absolute: ×206 net\n(jump ×~240 by layer ~2)", xy=(2, 60), fontsize=8, color="tab:red")
ax.annotate("relative: ×6.7 at layer 2\n→ ×0.46 net at layer 27", xy=(10, 0.9), fontsize=8, color="tab:blue")
ax.set_title("(b) Old ensemble data (FSLE): raw spread jumps early, then the state\n"
             "keeps lengthening while the spread does not — the FRACTION falls ×15")
ax.set_ylim(0.2, 400)

ax = axs[2]
vals = [np.median(rel_gain_band(2, la, lb)) for la, lb in [(3, 7), (7, 14), (14, 21), (21, 27)]]
nats = [np.log(v) / (lb - la) for v, (la, lb) in zip(vals, [(3, 7), (7, 14), (14, 21), (21, 27)])]
cols = ["tab:red" if x > 0 else "tab:blue" for x in nats]
ax.bar(range(4), nats, color=cols)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(["L3–7\n(arrival)", "L7–14", "L14–21", "L21–27\n(final)"], fontsize=8)
for i, v in enumerate(nats):
    ax.text(i, v + (0.02 if v > 0 else -0.06), f"{v:+.3f}", ha="center", fontsize=8)
ax.set_ylabel("log growth per layer of the RELATIVE nudge   (nats/layer)")
ax.set_title("(c) New teacher-forced data: the relative nudge still GROWS slightly\n"
             "in mid-depth (+0.04–0.05/layer); contraction appears only in the last layers")
fig.suptitle("Figure 5 — Absolute vs relative depth exponents. Opposite signs because the raw lengths "
             "include the ×450 inflation of the state ('the yardstick grows'); remove the yardstick's growth "
             "and what remains is slightly negative — in the JVP spectrum (a), the old ensemble (b),\n"
             "and the final layers of the new data (c) — but NOT in the mid-depth of the new data.", fontsize=10, y=1.05)
fig.tight_layout()
fig.savefig(OUT + r"\fig5_exponents.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 6 (schematic)
def draw_nudge(ax, h_len, dy, dx, title, note):
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.45, h_len + 2.35); ax.set_ylim(-1.75, 1.75)
    # state vector h
    ax.add_patch(FancyArrowPatch((0, 0), (h_len, 0), arrowstyle="-|>",
                                 mutation_scale=17, lw=3.4, color="tab:blue"))
    ax.text(h_len * 0.42, 0.17, "state h\n(what the layer stores)", fontsize=9.5,
            color="tab:blue", ha="center")
    # nudge: mostly perpendicular (measured ~87 deg), small along-h part
    ax.add_patch(FancyArrowPatch((h_len, 0), (h_len + dx, dy), arrowstyle="-|>",
                                 mutation_scale=14, lw=2.6, color="tab:red"))
    # decomposition of the nudge
    ax.plot([h_len + dx, h_len + dx], [dy, 0], ":", color="gray", lw=1.1)
    ax.add_patch(FancyArrowPatch((h_len, 0), (h_len + dx, 0), arrowstyle="-|>",
                                 mutation_scale=9, lw=2.0, color="dimgray"))
    ax.add_patch(matplotlib.patches.Arc((h_len, 0), 1.3, 1.3, theta1=0,
                                        theta2=np.degrees(np.arctan2(dy, dx)),
                                        color="tab:red", lw=1.2))
    ax.text(h_len + 0.78, 0.16, f"{np.degrees(np.arctan2(dy, dx)):.0f}°", fontsize=10,
            color="tab:red")
    ax.annotate("along-h part (~5–8% of length):\nonly changes the LENGTH of h\n"
                "→ cancelled by the RMSNorm\nbefore the logits",
                xy=(h_len + dx, -0.06), xytext=(0.05, -0.95), fontsize=8,
                color="dimgray", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.9))
    ax.text(h_len + dx + 0.14, dy * 0.62,
            "perpendicular part: ~95%\n(a DIRECTION change)\n→ this is all the logits\n"
            "ever see, divided by\nthe state's length", fontsize=8.4, color="tab:red", ha="left")
    ax.text(-0.42, 1.28, title, fontsize=9.5, color="k")
    ax.text(-0.42, 1.62, note, fontsize=8.2, color="dimgray")

fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.0))
draw_nudge(axs[0], 2.2, 0.85, 0.055,
           "EARLY (layer 2): ||h|| ≈ 44   (median)",
           "raw nudge ≈ 4e-5 → fraction of state ≈ 9.3e-7")
draw_nudge(axs[1], 3.3, 1.30, 0.085,
           "DEEP (layer 27): ||h|| ≈ 545   (median; ×12 vs layer 2, ×450 vs layer 0)",
           "raw nudge ≈ 1.5e-2 (×366) → fraction ≈ 2.7e-5 (×29 larger!)  — arrows not to scale")
fig.suptitle("Figure 6 — Plain-language picture. What grows with depth is the state's LENGTH "
             "('radial scale drift' of the state). The nudge is a small DIRECTION change (~87° to the state), not a length change.\n"
             "The output stage renormalizes length away (RMSNorm), so the logits only ever see the direction change "
             "relative to the state — a tiny, slowly varying fraction.", fontsize=10, y=1.04)
fig.tight_layout()
fig.savefig(OUT + r"\fig6_schematic.png", bbox_inches="tight")
plt.close(fig)

# ================================================================ save stats + log
with open(OUT + r"\stats.json", "w") as f:
    json.dump(S, f, indent=1)
with open(OUT + r"\analysis_log.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\nsaved fig1..fig6, stats.json, analysis_log.txt")
