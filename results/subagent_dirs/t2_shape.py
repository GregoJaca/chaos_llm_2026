"""TEST 2: after dividing out each direction's own gain, do the response
SHAPES (token-profiles of log ||delta h_t||) still differ beyond noise?

Noise model: the 4 smallest radii (r = 1e-3, 3e-3, 1e-2, 3e-2) are replicates of
the same shape under the gain model, so scatter across them is a conservative
noise estimate (it includes float32 rounding AND mild nonlinearity, both of
which inflate the noise and hence favour the equivalence null).

Reported:
- shape-variance fraction: share of the between-direction shape variance that
  survives the noise floor, at layer 27, linear and saturated regime.
- per-direction shape z-scores with autocorrelation-corrected dof.
- split-half (radius split) reliability of direction shapes.
- HOW shapes differ: per-direction correlations with token position,
  baseline margin, baseline hidden norm.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, spearmanr

from common import (L14, L21, L27, LIN_R, SAT_R, load_w100, logd,
                    pearson, save_json, savefig)

z = load_w100()
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
margins, norm_h0 = z["margins"], z["norm_h0"]
D, T = dh.shape[1], dh.shape[-1]
EX = [10, 9, 13]          # from t1 (hi, mid, lo gain at r=0.1, layer 27)
cols = ["#d62728", "#1f77b4", "#2ca02c"]


def shape_analysis(lidx, ridx_set, tag):
    """log-shape curves s[r,d,t]; returns dict of stats."""
    S = logd(dh[np.ix_(ridx_set, range(D), [lidx])][:, :, 0])   # [K, D, T]
    S = S - S.mean(axis=-1, keepdims=True)                        # gain removed
    K = len(ridx_set)
    sbar = S.mean(axis=0)                                        # [D, T] per-direction mean shape
    grand = sbar.mean(axis=0)                                    # [T] common shape
    dev = sbar - grand                                            # [D, T] direction deviations
    resid = S - sbar                                              # [K, D, T] replicate scatter
    sig2 = resid.var(axis=0, ddof=1).mean()                       # mean noise var of ONE radius
    Vb = dev.var(axis=0, ddof=1)                                  # [T] between-direction variance
    noise_of_mean = sig2 / K
    frac = max(0.0, 1.0 - noise_of_mean / Vb.mean())               # reliable shape fraction
    # per-token F test (F(19, K*D-D... approx 60) at K=4)
    F = Vb / noise_of_mean
    from scipy.stats import f as fdist
    Fcrit = fdist.ppf(0.95, D - 1, K * (D - 1))
    # per-direction z with autocorrelation-corrected effective length
    # residual autocorrelation over t (averaged over replicates+dirs)
    rc = np.zeros(40)
    for d in range(D):
        for k in range(K):
            x = resid[k, d] - resid[k, d].mean()
            v = (x * x).sum()
            for lag in range(1, 41):
                rc[lag - 1] += (x[:-lag] * x[lag:]).sum() / v
    rc /= D * K
    tau = 1.0 + 2.0 * np.clip(rc, -0.5, 1.0).sum()
    Teff = np.clip(T / tau, 1.0, T)
    zsc = np.sqrt((dev ** 2).sum(axis=1) / (T * noise_of_mean))
    # E[z^2] under H0 with Teff effective dof: z^2 ~ chi2(Teff)/Teff... scale
    pvals = chi2.sf((zsc ** 2) * Teff, Teff)
    # split-half reliability of direction deviations
    h1 = S[0::2].mean(axis=0) - S[0::2].mean(axis=0).mean(axis=0)
    h2 = S[1::2].mean(axis=0) - S[1::2].mean(axis=0).mean(axis=0)
    r_split = pearson((h1 - grand).ravel(), (h2 - grand).ravel())
    return dict(tag=tag, S=S, dev=dev, grand=grand, noise=np.sqrt(noise_of_mean),
                frac=frac, F=F, Fcrit=float(Fcrit), fsig=float((F > Fcrit).mean()),
                zsc=zsc, pvals=pvals, Teff=float(Teff), r_split=r_split,
                Vb=Vb, noise_of_mean=noise_of_mean, ridx_set=ridx_set)


A_lin = shape_analysis(L27, LIN_R, "linear")
A_sat = shape_analysis(L27, SAT_R, "saturated")
A_l14 = shape_analysis(L14, LIN_R, "linear_l14")
A_l21 = shape_analysis(L21, LIN_R, "linear_l21")

N = {
    "layer27_linear": {"shape_var_fraction": A_lin["frac"],
                       "split_half_r": A_lin["r_split"],
                       "frac_tokens_Fsig_at_5pct": A_lin["fsig"],
                       "Teff": A_lin["Teff"],
                       "n_dirs_p_lt_0.05": int((A_lin["pvals"] < 0.05).sum()),
                       "n_dirs_p_lt_0.01": int((A_lin["pvals"] < 0.01).sum())},
    "layer27_saturated": {"shape_var_fraction": A_sat["frac"],
                          "split_half_r": A_sat["r_split"],
                          "frac_tokens_Fsig_at_5pct": A_sat["fsig"],
                          "n_dirs_p_lt_0.05": int((A_sat["pvals"] < 0.05).sum())},
    "layer14_linear": {"shape_var_fraction": A_l14["frac"], "split_half_r": A_l14["r_split"]},
    "layer21_linear": {"shape_var_fraction": A_l21["frac"], "split_half_r": A_l21["r_split"]},
}

# raw pairwise correlation of log curves (gain kept) vs gain-removed, r=0.1 (idx 4)
X = logd(dh[4, :, L27])                      # [D, T]
Xc = X - X.mean(axis=1, keepdims=True)
r_raw = np.median([pearson(Xc[a], Xc[b]) for a in range(D) for b in range(a + 1, D)])
Xd = Xc - Xc.mean(axis=1, keepdims=True)     # additionally remove each dir's level = gain
r_gainfree = np.median([pearson(Xd[a], Xd[b]) for a in range(D) for b in range(a + 1, D)])
N["pairwise_corr_logcurves_r0.1_L27"] = {"with_gain": r_raw, "gain_removed": r_gainfree}

# ---- two-way variance decomposition of log-response (layer 27, r=0.1) ----
X = logd(dh[4, :, L27])                                  # [D, T]
glob = X.mean()
level = X.mean(axis=1) - glob                             # per-direction log gain part
common = X.mean(axis=0) - glob                           # common token profile
inter = X - glob - level[:, None] - common[None, :]      # direction x token interaction
decomp = {
    "gain(level) share": float((level ** 2).mean()),
    "common token-profile share": float((common ** 2).mean()),
    "direction-specific shape share": float((inter ** 2).mean()),
    "total": float(((X - glob) ** 2).mean()),
}
decomp = {k: v / decomp["total"] for k, v in decomp.items() if k != "total"}
decomp["rms_gain_log_units"] = float(np.sqrt((level ** 2).mean()))
decomp["rms_common_log_units"] = float(np.sqrt((common ** 2).mean()))
decomp["rms_dirshape_log_units"] = float(np.sqrt((inter ** 2).mean()))
N["log_response_decomposition_r0.1_L27"] = decomp
print("\nlog-response decomposition (fractions of variance):",
      {k: round(v, 4) for k, v in decomp.items() if "share" in k})
print("RMS (log-units): gain %.3f, common %.3f, dir-shape %.3f" %
      (decomp["rms_gain_log_units"], decomp["rms_common_log_units"],
       decomp["rms_dirshape_log_units"]))

# HOW shapes differ
post = slice(95, 300)   # fully-responsive token range (front passes 92 by layer 27 anyway full)
# interaction over the fully-responsive range only (t>=95): robustness
inter_post = inter[:, post]
N["dirshape_share_post95"] = float((inter_post ** 2).mean() / ((X[:, post] - glob) ** 2).mean())
devL = A_lin["dev"][:, post]
c_t = np.array([spearmanr(devL[d], np.arange(devL.shape[1])).statistic for d in range(D)])
c_m = np.array([spearmanr(devL[d], np.log(margins[post])).statistic for d in range(D)])
c_h = np.array([spearmanr(devL[d], np.log(norm_h0[L27, post])).statistic for d in range(D)])
N["shape_corr_with"] = {
    "token_position": {"median": float(np.median(c_t)), "n_sig": int((np.abs(c_t) > 0.113).sum())},
    "log_margin": {"median": float(np.median(c_m)), "n_sig": int((np.abs(c_m) > 0.113).sum())},
    "log_baseline_norm": {"median": float(np.median(c_h)), "n_sig": int((np.abs(c_h) > 0.113).sum())},
}
g = np.median(dh[4, :, L27], axis=-1)
U, s, Vt = np.linalg.svd(devL - devL.mean(0), full_matrices=False)
N["dev_PCs"] = {"pc1_frac": float(s[0] ** 2 / (s ** 2).sum()),
                "pc2_frac": float(s[1] ** 2 / (s ** 2).sum()),
                "pc3_frac": float(s[2] ** 2 / (s ** 2).sum())}
for i in range(3):
    N[f"dev_pc{i+1}_vs_gain_spearman"] = float(spearmanr(U[:, i] * s[i], g).statistic)

# ---------------- FIGURE ----------------
fig, axs = plt.subplots(2, 2, figsize=(12.5, 9))
tt = np.arange(T)

# (a) 3 example deviation curves + noise band
ax = axs[0, 0]
band = 2 * A_lin["noise"] * np.sqrt(A_lin["Vb"].mean() / np.maximum(A_lin["Vb"], 1e-12))
ax.fill_between(tt, -2 * A_lin["noise"], 2 * A_lin["noise"], color="gray", alpha=0.35,
                label="noise floor +/-2 sigma (from 4-radii replicates)")
lbl = ["highest gain", "median gain", "lowest gain"]
for k, d in enumerate(EX):
    ax.plot(tt, A_lin["dev"][d], color=cols[k], lw=1.0, alpha=0.55)
    ax.plot(tt, np.convolve(A_lin["dev"][d], np.ones(9) / 9, mode="same"),
            color=cols[k], lw=1.8, label=f"dir {d} ({lbl[k]}), 9-token smooth")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("generated-token position t (teacher-forced on baseline text)")
ax.set_ylabel("log-shape deviation (log-units: log||dh_t|| minus own mean,\n minus population mean shape)")
ax.set_title(f"(a) Direction shapes at layer 27, linear radii (r=0.001..0.03)\n"
             f"shape-variance fraction = {A_lin['frac']*100:.0f}%  "
             f"split-half reliability r = {A_lin['r_split']:.2f}")
ax.legend(loc="upper right")

# (b) per-direction shape z, sorted
ax = axs[0, 1]
o = np.argsort(A_lin["zsc"])
ax.bar(range(D), A_lin["zsc"][o], color="gray")
thr = np.sqrt(chi2.ppf(0.95, A_lin["Teff"]) / A_lin["Teff"])
ax.axhline(thr, color="k", ls="--", lw=1,
           label=f"95% chi2 threshold (T_eff={A_lin['Teff']:.0f})")
for k, d in enumerate(EX):
    ax.bar(np.where(o == d)[0], A_lin["zsc"][d], color=cols[k], label=f"dir {d}")
ax.set_xlabel("direction rank (sorted by shape distinctiveness)")
ax.set_ylabel("shape z-score: ||shape deviation|| / noise scale")
ax.set_title(f"(b) How many directions have shapes beyond noise?\n"
             f"significant at p<0.05: {(A_lin['pvals']<0.05).sum()}/20; "
             f"at p<0.01: {(A_lin['pvals']<0.01).sum()}/20  "
             f"(saturated regime: {(A_sat['pvals']<0.05).sum()}/20)")
ax.legend(loc="upper left")

# (c) split-half reliability
ax = axs[1, 0]
S = A_lin["S"]
h1 = S[0::2].mean(axis=0); h2 = S[1::2].mean(axis=0)
d1 = (h1 - h1.mean(axis=0)).ravel(); d2 = (h2 - h2.mean(axis=0)).ravel()
ax.scatter(d1, d2, s=4, alpha=0.25, color="gray", rasterized=True)
ax.set_xlabel("shape deviation, radii {0.001, 0.01} (log-units)")
ax.set_ylabel("shape deviation, radii {0.003, 0.03} (log-units)")
ax.set_title(f"(c) Split-half reliability of direction shapes (all 20 dirs x 300 t)\n"
             f"Pearson r = {A_lin['r_split']:.2f}  (0 = pure noise = equivalence null; "
             f"saturated regime r = {A_sat['r_split']:.2f})")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)

# (d) how shapes differ: early/late vs margin-seeking
ax = axs[1, 1]
sz = 14 + 70 * (np.log(g) - np.log(g.min())) / (np.log(g.max()) - np.log(g.min()))
ax.scatter(c_t, c_m, s=sz, color="gray", alpha=0.8)
for k, d in enumerate(EX):
    ax.scatter(c_t[d], c_m[d], s=sz[d] * 1.6, color=cols[k], zorder=3, label=f"dir {d} (marker size = gain)")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Spearman(shape, token position t)  >0 = responds late")
ax.set_ylabel("Spearman(shape, baseline logit margin)  >0 = responds at high-margin tokens")
ax.set_title(f"(d) HOW shapes differ (layer 27, linear)\n"
             f"late-biased dirs: {(c_t>0.113).sum()}/20, early: {(c_t<-0.113).sum()}/20; "
             f"margin-seeking: {(c_m>0.113).sum()}/20, margin-avoiding: {(c_m<-0.113).sum()}/20")
ax.legend(loc="lower right", handletextpad=0.1)

fig.suptitle("TEST 2: shape differences beyond per-direction gain "
             "(deviations from the population-mean log||dh_t|| profile)", y=1.005)
savefig(fig, "fig2_shape.png")
save_json("numbers_t2.json", N)

print("\n=== TEST 2 summary ===")
print("layer 27 linear: shape fraction", round(A_lin["frac"], 3),
      " split-half r", round(A_lin["r_split"], 3),
      " tokens F-sig", round(A_lin["fsig"], 3),
      " dirs p<0.05:", (A_lin["pvals"] < 0.05).sum())
print("layer 27 saturated: shape fraction", round(A_sat["frac"], 3),
      " split-half r", round(A_sat["r_split"], 3),
      " dirs p<0.05:", (A_sat["pvals"] < 0.05).sum())
print("layer 14 / 21 linear: fraction", round(A_l14["frac"], 3), "/", round(A_l21["frac"], 3))
print("pairwise log-curve corr r=0.1 L27: with gain", round(r_raw, 3),
      " gain-removed", round(r_gainfree, 3))
print("shape corr with position/margin/hnorm (median):",
      round(float(np.median(c_t)), 3), round(float(np.median(c_m)), 3), round(float(np.median(c_h)), 3))
print("dev PCs:", [round(x, 3) for x in
                  [N["dev_PCs"]["pc1_frac"], N["dev_PCs"]["pc2_frac"], N["dev_PCs"]["pc3_frac"]]])
for i in range(3):
    print(f"PC{i+1} score vs gain spearman:", round(N[f"dev_pc{i+1}_vs_gain_spearman"], 3))
