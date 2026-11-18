"""Figures 1-3: lambda_k(t) by layer, growth/decay-phase fits, radius dependence."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa

np.seterr(divide="ignore", invalid="ignore")
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8})

z = load("w100_layers/a1.npz")
dh, radii, layers = z["dh_norm"], z["radii"], z["layers"]
t = tvec(dh)
FLUSH0, FLUSH1 = 36, 100  # prompt-KV eviction epoch for W=100, 65-token prompt
NUM = {}

# ------------------------------------------------------------------ FIG 01
d = dh[4]  # r = 0.1
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
groups = [(0, [1, 2, 3, 4], "L1 L2 L3 L7"), (1, [5, 6, 7], "L14 L21 L27")]
for ax, (gidx, jdx, name) in zip(axes, groups):
    for j in jdx:
        A = d[:, j, :]
        la = lam_k_2d(A, 5)
        m, lo, hi = med_band(la)
        ax.plot(t, m, lw=1.4, label=f"L{layers[j]}")
        ax.fill_between(t, lo, hi, alpha=0.22, lw=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(FLUSH0, FLUSH1, color="orange", alpha=0.12, label="prompt-KV eviction (t=36..100)")
    lo_all, hi_all = [], []
    for j in jdx:
        la = lam_k_2d(d[:, j, :], 5)
        lo_all.append(np.nanpercentile(la, 1))
        hi_all.append(np.nanpercentile(la, 99))
    ax.set_ylim(min(lo_all), max(hi_all))
    ax.set_ylabel("λ₅(t) = (1/10)·ln[a(t+5)/a(t−5)]   [per token]")
    ttl = ("shallow: response dies to exact float32 zero after eviction epoch"
           if gidx == 0 else
           "deep: λ₅ hovers at ≈ −0.01; band touches 0 only transiently")
    ax.set_title(ttl)
    ax.legend(ncol=5)
axes[1].set_xlabel("generated token t (1-based, teacher-forced run)")
fig.suptitle("FIG 1  λ₅(t) per layer — W=100, r=0.1, prompt65, median over 20 perturbation dirs, 25–75% band.\n"
             "NO sustained positive phase: median λ₅ stays within ±0.07 (p2.5–p97.5) at deep layers; L0 ≡ 0 (embedding,\n"
             "unperturbed); L1 dead (a≡0 in float32) from t=93, L2 from t=192, L3 from t=291. Deep-layer tail ≈ −0.01/token.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(os.path.join(OUT, "fig01_lambda_layers.png"))
plt.close(fig)
print("fig01 done")

# per-layer one-step ratio and lam1 spike values
NUM["step1_ratio"] = {}
for j, L in enumerate(layers):
    A = d[:, j, :]
    if not np.all(A[:, :2] > 0):
        continue
    r10 = np.log(A[:, 1]) - np.log(A[:, 0])
    m, lo, hi = boot_dirs_stat(r10)
    NUM["step1_ratio"][int(L)] = dict(mean=m, lo=lo, hi=hi)

# ------------------------------------------------------------------ FIG 02
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, (j, L) in zip([axes[0, 0], axes[0, 1]], [(7, 27), (4, 14)]):
    A = d[:, j, :]
    med_loga = np.nanmedian(np.log(A), axis=0)
    tt = t[t > 110]
    yy = med_loga[t > 110]
    sl, s_lo, s_hi, r2e = ols_slope(tt, yy)
    M = np.column_stack([np.ones_like(tt), np.log(tt)])
    coef, *_ = np.linalg.lstsq(M, yy, rcond=None)
    ypow = M @ coef
    ax.plot(t, med_loga, color="0.3", lw=1.2, label="median ln a(t), 20 dirs")
    ax.plot(tt, coef[0] + coef[1] * np.log(tt), "b--", lw=1.2,
            label=f"power law  a∝t^{coef[1]:+.2f}   R²={r2(yy, ypow):.3f}")
    ax.plot(tt, sl * tt + (np.nanmean(yy) - sl * np.nanmean(tt)),
            "r-", lw=1.2,
            label=f"exponential  rate={sl:+.4f} [{s_lo:+.4f},{s_hi:+.4f}]  R²={r2e:.3f}")
    ax.axvspan(FLUSH0, FLUSH1, color="orange", alpha=0.12)
    aic_e = aic_bic(yy, sl * tt + (np.nanmean(yy) - sl * np.nanmean(tt)), 2)
    aic_p = aic_bic(yy, ypow, 2)
    ax.set_title(f"L{L}, t>110: AIC exp/pow = {aic_e[0]:.0f}/{aic_p[0]:.0f}, "
                 f"BIC = {aic_e[1]:.0f}/{aic_p[1]:.0f}")
    ax.set_xlabel("token t")
    ax.set_ylabel("ln a(t)  [a = ||δh||]")
    ax.legend(loc="lower left")
    NUM.setdefault("decay_fits", {})[f"L{L}"] = dict(
        exp=dict(rate=sl, lo=s_lo, hi=s_hi, r2=r2e, aic=aic_e[0], bic=aic_e[1]),
        pow=dict(p=float(coef[1]), r2=r2(yy, ypow), aic=aic_p[0], bic=aic_p[1]))

# panel C: lambda_1 early (the 'growth phase' is one token)
ax = axes[1, 0]
for j, L in [(4, 7), (5, 14), (7, 27)]:
    A = d[:, j, :]
    lam1 = np.diff(np.log(A), axis=1)
    m = np.nanmedian(lam1[:, :20], axis=0)
    ax.plot(np.arange(1, 21), m, marker="o", ms=3, lw=1.2, label=f"L{L}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("token t (λ₁(t) uses a(t+1)/a(t))")
ax.set_ylabel("median λ₁(t)  [per token]")
ax.set_title("'growth phase' is ONE token: +0.26 at L27 at t=1, negative from t=2 on\n"
             "(initial simplex direction snaps onto the fast direction, then decays)")
ax.legend()
NUM["lam1_t1"] = {f"L{L}": float(np.nanmedian(np.diff(np.log(d[:, j, :]), axis=1)[:, 0]))
                 for j, L in [(1, 1), (2, 2), (3, 3), (4, 7), (5, 14), (6, 21), (7, 27)]}

# panel D: one-step log-ratio per layer, bootstrap CI
ax = axes[1, 1]
Ls = sorted(int(k) for k in NUM["step1_ratio"])
vals = [NUM["step1_ratio"][L]["mean"] for L in Ls]
los = [NUM["step1_ratio"][L]["lo"] for L in Ls]
his = [NUM["step1_ratio"][L]["hi"] for L in Ls]
ax.errorbar(range(len(Ls)), vals, yerr=[np.array(vals) - np.array(los),
                                        np.array(his) - np.array(vals)],
            fmt="o", color="tab:red", capsize=3, lw=1)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(Ls)))
ax.set_xticklabels([f"L{L}" for L in Ls])
ax.set_ylabel("ln[a(2)/a(1)]  (median over dirs, 95% CI)")
ax.set_title("one-step response at t=1: positive only at L14–L27 (≤+0.3);\n"
             "shallow layers contract immediately (L1..L3 negative)")
fig.suptitle("FIG 2  growth phase & post-flush decay — W=100, r=0.1, prompt65.\n"
             "No extended growth phase exists (1-token spike at L21–L27 only). Post-eviction decay of ln a(t):\n"
             "L14 → exponential (fixed rate −0.0132 CI[−0.0135,−0.0128] wins AIC); L27 → power law a∝t^−2.2 wins AIC;\n"
             "both are decays — the fitted sustained rate is significantly NEGATIVE at every layer (Fig 8).",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(OUT, "fig02_growth_decay_fits.png"))
plt.close(fig)
print("fig02 done")

# ------------------------------------------------------------------ FIG 03
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
# panel A: collapse a(t)/r
ax = axes[0]
for i, r, c in [(1, 0.003, "tab:blue"), (3, 0.03, "tab:green"), (4, 0.1, "tab:red"),
                (6, 0.3, "tab:purple"), (11, 2.0, "0.4")]:
    A = dh[i][:, 7, :]
    med = np.nanmedian(A, axis=0) / r
    ax.plot(t, med, color=c, lw=1.2, ls="--" if r == 2 else "-",
            label=f"r={r:g}" + ("  (19/20 flips)" if r == 2 else ""))
ax.set_yscale("log")
ax.set_xlabel("token t")
ax.set_ylabel("median a(t) / r   [normalised response]")
ax.set_title("radius-normalised response: COLLAPSE for r≤0.3\n"
             "(r=2 lifted ×1.2 → weakly nonlinear regime)")
ax.legend()

# panel B: lambda_5 for three radii
ax = axes[1]
for i, r, c in [(2, 0.01, "tab:blue"), (4, 0.1, "tab:red"), (6, 0.3, "tab:purple")]:
    A = dh[i][:, 7, :]
    la = lam_k_2d(A, 5)
    m, lo, hi = med_band(la)
    ax.plot(t, m, color=c, lw=1.3, label=f"r={r:g}")
    ax.fill_between(t, lo, hi, color=c, alpha=0.2, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(-0.25, 0.25)
ax.set_xlabel("token t")
ax.set_ylabel("λ₅(t)  [per token]")
ax.set_title("λ₅(t) at L27 for 3 radii: statistically indistinguishable\n"
             "(20 dirs each; linear-response regime)")
ax.legend()

# panel C: rate summaries vs r
ax = axes[2]
tail_mask = t > TAIL_T0
rs, tail_m, tail_lo, tail_hi = [], [], [], []
er_m, er_lo, er_hi = [], [], []
for i, r in enumerate(radii):
    A = dh[i][:, 7, :]
    la10 = lam_k_2d(A, 10)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(tr)
    rs.append(r)
    tail_m.append(m); tail_lo.append(lo); tail_hi.append(hi)
    sl = np.array([ols_slope(t[1:30], np.log(A[u, 1:30]))[0] for u in range(20)])
    m2, lo2, hi2 = boot_dirs_stat(sl)
    er_m.append(m2); er_lo.append(lo2); er_hi.append(hi2)
ax.errorbar(rs, tail_m, yerr=[np.array(tail_m) - np.array(tail_lo),
                              np.array(tail_hi) - np.array(tail_m)],
            fmt="o", color="tab:red", capsize=3, label="tail rate  mean λ₁₀ (t>150)")
ax.errorbar(rs, er_m, yerr=[np.array(er_m) - np.array(er_lo),
                            np.array(er_hi) - np.array(er_m)],
            fmt="s", color="tab:blue", capsize=3, ms=4, label="early rate  OLS ln a (t=2..30)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("initial perturbation radius r")
ax.set_ylabel("rate  [per token]")
ax.set_title("tail rate flat at −0.010 (CIs ±0.001) for ALL r;\n"
             "early rate ≈ 0 (CI crosses 0) for r≤0.3, negative for r≥0.7")
ax.legend(loc="lower left")
NUM["radius_tail"] = {f"{r:g}": dict(m=a, lo=b, hi=c) for r, a, b, c in
                      zip(rs, tail_m, tail_lo, tail_hi)}
NUM["radius_early"] = {f"{r:g}": dict(m=a, lo=b, hi=c) for r, a, b, c in
                       zip(rs, er_m, er_lo, er_hi)}
fig.suptitle("FIG 3  radius dependence — w100_layers, L27, 20 dirs, 95% bootstrap CIs over directions.\n"
             "Finite-time rates are r-independent in the linear-response regime (r ≤ 0.3, prediction flips ≤ 2/20):\n"
             "stretching is a property of the trajectory, not the perturbation size.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig03_radius.png"))
plt.close(fig)
print("fig03 done")

save_json(NUM, "numbers_figs_1_3.json")
