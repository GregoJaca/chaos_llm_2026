"""Figures 4-6: window regimes, Jensen gap, layer profile."""
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

t = tvec(np.zeros((1, 1, T)))
tail_mask = t > TAIL_T0
early_mask = (t >= 6) & (t <= 30)
NUM = {}
COL = {50: "tab:blue", 100: "tab:red", 200: "tab:green", 400: "tab:orange",
       800: "tab:purple", 4096: "0.35"}


def getW(W, cyber=False):
    zz = load(("window_sweep_cyber" if cyber else "window_sweep") + f"/w{W}/a1.npz")
    return zz["dh_norm"][1]  # r=0.1, [10 dirs, 4 layers (7,14,21,27), 300]


# ------------------------------------------------------------------ FIG 04
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
WS = [50, 100, 200, 4096]
# A: lambda_5(t) per window
ax = axes[0, 0]
for W in WS:
    A = getW(W)[:, 3, :]  # L27
    la = lam_k_2d(A, 5)
    m, lo, hi = med_band(la)
    ax.plot(t, m, color=COL[W], lw=1.3, label=f"W={W}")
    ax.fill_between(t, lo, hi, color=COL[W], alpha=0.2, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("token t")
ax.set_ylabel("λ₅(t)  [per token]")
ax.set_ylim(-0.3, 0.3)
ax.set_title("λ₅(t), L27: W=50 contracts from t=1 (prompt truncated, a(1)≈0.9);\n"
             "W≥100 identical until first eviction (t≈38) then W=100 dips more")
ax.legend(ncol=4)

# B: cumulative Lambda(t) per window
ax = axes[0, 1]
cross_info = {}
for W in WS:
    A = getW(W)[:, 3, :]
    cum = cum_rate_2d(A)
    m, lo, hi = med_band(cum)
    ax.plot(t, m, color=COL[W], lw=1.3, label=f"W={W}")
    ax.fill_between(t, lo, hi, color=COL[W], alpha=0.2, lw=0)
    pos = np.where(m > 0)[0]
    lastpos_dir = []
    for u in range(cum.shape[0]):
        c = cum[u]
        f = np.isfinite(c)
        p = np.where(c[f] > 0)[0]
        lastpos_dir.append(int(p[-1]) + 1 if len(p) else 0)
    cross_info[W] = dict(max=float(np.nanmax(m)), tmax=int(np.nanargmax(m)) + 1,
                         n_pos=int(len(pos)), last_pos=int(pos[-1]) + 1 if len(pos) else 0,
                         lastpos_dir=sorted(int(v) for v in lastpos_dir))
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("token t")
ax.set_ylabel("Λ(t) = ln[a(t)/a(1)]/t   [per token]")
ax.set_title("cumulative Λ(t): ALL windows share the early burst (max +0.059 @ t≈13);\n"
             "W=50 negative from t≈7; W=100 stays positive until t≈%d; W≥200 until t≈245; all end negative"
             % cross_info[100]["last_pos"])
ax.legend(ncol=4)
NUM["cross_info"] = cross_info

# C: early rate vs W with two CI flavours
ax = axes[1, 0]
WALL = [50, 100, 200, 400, 800, 4096]
for k, (Lidx, lab, col) in enumerate([(3, "L27", "tab:red"), (1, "L14", "tab:blue")]):
    es, e_dir, e_2w = [], [], []
    for W in WALL:
        A = getW(W)[:, Lidx, :]
        la5 = lam_k_2d(A, 5)
        e = np.nanmean(la5[:, early_mask], axis=1)
        m, lo, hi = boot_dirs_stat(e)
        es.append(m)
        e_dir.append((lo, hi))
        m2, lo2, hi2 = two_way_boot(la5, early_mask, block=10, n_boot=800)
        e_2w.append((lo2, hi2))
    x = np.arange(len(WALL)) + k * 0.18 - 0.09
    es = np.array(es)
    ax.errorbar(x, es, yerr=[es - np.array([a for a, b in e_dir]),
                             np.array([b for a, b in e_dir]) - es],
                fmt="o", color=col, capsize=3, label=f"{lab}: dir bootstrap")
    ax.errorbar(x + 0.05, es, yerr=[es - np.array([a for a, b in e_2w]),
                                    np.array([b for a, b in e_2w]) - es],
                fmt="_", color=col, capsize=6, lw=1.6, alpha=0.6,
                label=f"{lab}: dir+token-block (block=10)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(WALL)))
ax.set_xticklabels([str(w) for w in WALL])
ax.set_xlabel("context window W (prompt 63 tokens)")
ax.set_ylabel("early rate: mean λ₅ over t∈[6,30]")
ax.set_title("early rate vs W: W=50 strongly negative; W=100 ≈ +0.004 [−0.0015,+0.011] (CI crosses 0);\n"
             "W≥100 identical (no eviction before t=38). Block bootstrap widens CIs ~3×")
ax.legend(fontsize=7, loc="lower right")
NUM["windows_early"] = {W: dict(m=np.nanmean(np.nanmean(lam_k_2d(getW(W)[:, 3, :], 5)[:, early_mask], axis=1)))
                        for W in WALL}

# D: tail rate vs W with the same two CIs
ax = axes[1, 1]
for k, (Lidx, lab, col) in enumerate([(3, "L27", "tab:red"), (1, "L14", "tab:blue")]):
    es, e_dir, e_2w = [], [], []
    for W in WALL:
        A = getW(W)[:, Lidx, :]
        la10 = lam_k_2d(A, 10)
        e = np.nanmean(la10[:, tail_mask], axis=1)
        m, lo, hi = boot_dirs_stat(e)
        es.append(m)
        e_dir.append((lo, hi))
        m2, lo2, hi2 = two_way_boot(la10, tail_mask, block=10, n_boot=800)
        e_2w.append((lo2, hi2))
    x = np.arange(len(WALL)) + k * 0.18 - 0.09
    es = np.array(es)
    ax.errorbar(x, es, yerr=[es - np.array([a for a, b in e_dir]),
                              np.array([b for a, b in e_dir]) - es],
                fmt="o", color=col, capsize=3, label=f"{lab}: dir bootstrap")
    ax.errorbar(x + 0.05, es, yerr=[es - np.array([a for a, b in e_2w]),
                                    np.array([b for a, b in e_2w]) - es],
                fmt="_", color=col, capsize=6, lw=1.6, alpha=0.6,
                label=f"{lab}: dir+token-block (block=10)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(WALL)))
ax.set_xticklabels([str(w) for w in WALL])
ax.set_xlabel("context window W (prompt 63 tokens)")
ax.set_ylabel("sustained rate: mean λ₁₀ over t>150")
ax.set_title("sustained rate vs W — ALL significantly negative or ≈0:\n"
             "W=50: −0.0175 [−0.0182,−0.0165]; W=100: −0.0065; W≥200: −0.005 (2-way CI upper bound −0.0006)")
ax.legend(fontsize=7, loc="upper right")
fig.suptitle("FIG 4  window regime (same prompt 63, r=0.1, 10 dirs, L27/L14). Small windows CONTRACT harder\n"
             "(forgetting rate ∝ eviction churn: tail −0.018 → −0.005 as W grows). The positive Λ excursion seen\n"
             "in every window is the shared t≈13 transient burst, not a window effect; sustained rates are ≤ 0 everywhere.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "fig04_windows.png"))
plt.close(fig)
print("fig04 done", cross_info)

# ------------------------------------------------------------------ FIG 05
z = load("w100_layers/a1.npz")
d = z["dh_norm"][4]  # r=0.1
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
# A: mean of lambda vs lambda of mean
ax = axes[0]
A = d[:, 7, :]
la = lam_k_2d(A, 5)
mean_lam = np.nanmean(la, axis=0)
lo, hi = boot_dirs_curve(la, 5, stat="mean")
lam_mean_a = lam_k(np.nanmean(A, axis=0), 5)
ax.plot(t, mean_lam, color="tab:red", lw=1.3, label="mean over 20 dirs of λ₅")
ax.fill_between(t, lo, hi, color="tab:red", alpha=0.2, lw=0)
ax.plot(t, lam_mean_a, color="k", lw=1.2, ls="--", label="λ₅ of the direction-averaged a(t)")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(-0.25, 0.25)
ax.set_xlabel("token t")
ax.set_ylabel("λ₅(t)  [per token]")
ax.set_title("'mean of λ' = 'λ of the mean' within the bootstrap band (L27)\n"
             "→ no Jensen artefact; direction averaging is legitimate")
ax.legend(loc="upper right")

# B: gap
ax = axes[1]
gap = lam_mean_a - mean_lam
m, lo, hi = med_band(gap[None, :])
ax.plot(t, gap, color="tab:purple", lw=1.1)
ax.axhline(0, color="k", lw=0.8)
mgap = np.nanmean(gap[(t > 20)])
ax.axhline(mgap, color="tab:purple", ls=":", lw=1.2,
           label=f"mean gap over t>20: {mgap:+.4f}")
ax.set_ylim(-0.12, 0.12)
ax.set_xlabel("token t")
ax.set_ylabel("Jensen gap: λ(mean a) − mean(λ)   [per token]")
ax.set_title(f"Jensen gap L27: mean {mgap:+.4f} per token — 6× smaller than the\n"
             "per-token noise (σ≈0.03) and ≈15× smaller than the tail rate magnitude")
ax.legend()

# C: per-direction histograms of the sustained rate
ax = axes[2]
for Lidx, col, lab in [(7, "tab:red", "L27"), (4, "tab:blue", "L14")]:
    A = d[:, Lidx, :]
    la10 = lam_k_2d(A, 10)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    ax.hist(tr, bins=8, alpha=0.5, color=col, label=f"{lab} (n=20 dirs)")
    m, lo, hi = boot_dirs_stat(tr)
    NUM[f"dir_tail_{lab}"] = dict(m=m, lo=lo, hi=hi,
                                  npos=int(np.sum(tr > 0)), max=float(np.max(tr)))
    print(lab, "dir tail:", m, lo, hi, "npos", np.sum(tr > 0), "max", np.max(tr))
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("per-direction sustained rate  mean λ₁₀ (t>150)")
ax.set_ylabel("count of directions")
ax.set_title("per-direction sustained rates: 0/20 directions positive at L27 or L14\n"
             "(all individual directions contract)")
ax.legend()
fig.suptitle("FIG 5  direction averaging (Jensen) — w100_layers, W=100, r=0.1, 20 simplex directions.\n"
             "mean-of-λ ≈ λ-of-mean (mean gap −0.0005/token); the negative sustained rate is not a\n"
             "mean-field artefact: every single direction has tail rate < 0.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig05_jensen.png"))
plt.close(fig)
print("fig05 done")

# ------------------------------------------------------------------ FIG 06
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
lay = [1, 2, 3, 7, 14, 21, 27]   # L0 excluded (a≡0)
jdx = {int(L): i for i, L in enumerate(z["layers"])}
pk, pk_t, pk_lo, pk_hi = [], [], [], []
tr_m, tr_lo, tr_hi = [], [], []
cu_m, cu_lo, cu_hi = [], [], []
rng = np.random.default_rng(0)
for L in lay:
    A = d[:, jdx[L], :]
    la = lam_k_2d(A, 5)
    sm = smooth(np.nanmedian(la, axis=0), 9)
    if np.all(~np.isfinite(sm)):
        continue
    pk.append(np.nanmax(sm))
    pk_t.append(int(np.nanargmax(sm)) + 1)
    reps = []
    for b in range(500):
        idx = rng.integers(0, 20, 20)
        s2 = smooth(np.nanmedian(la[idx], axis=0), 9)
        if np.any(np.isfinite(s2)):
            reps.append(np.nanmax(s2))
    pk_lo.append(np.percentile(reps, 2.5)); pk_hi.append(np.percentile(reps, 97.5))
    la10 = lam_k_2d(A, 10)
    tr = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(tr)
    tr_m.append(m); tr_lo.append(lo); tr_hi.append(hi)
    cum = cum_rate_2d(A)
    c299 = cum[:, 299]
    m, lo, hi = boot_dirs_stat(c299)
    cu_m.append(m); cu_lo.append(lo); cu_hi.append(hi)
Ls = lay[:len(pk)]
x = np.arange(len(Ls))

ax = axes[0]
pk, pk_lo, pk_hi = np.array(pk), np.array(pk_lo), np.array(pk_hi)
ax.errorbar(x, pk, yerr=[pk - pk_lo, pk_hi - pk], fmt="o-", color="tab:red", capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"L{s}" for s in Ls])
ax.set_ylabel("peak smoothed median λ₅  [per token]")
ax.set_title("peak rate vs layer: ≤ +0.11 (L1 death blip), ≤ +0.07 deep;\n"
             "peaks are transient flush-epoch blips, NOT a growth front")

ax = axes[1]
ax.plot(x, pk_t, "o-", color="tab:green")
ax.set_xticks(x); ax.set_xticklabels([f"L{s}" for s in Ls])
ax.set_ylabel("t of peak (argmax of smoothed curve)")
ax.set_title("peak times scattered (t=49..90), inside the eviction epoch t∈[36,100]\n"
             "→ no coherent depth-ordered growth front")

ax = axes[2]
trm, trlo, trhi = np.array(tr_m), np.array(tr_lo), np.array(tr_hi)
ax.errorbar(x, trm, yerr=[trm - trlo, trhi - trm], fmt="o-", color="tab:blue", capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"L{s}" for s in Ls])
ax.set_ylabel("sustained rate  mean λ₁₀ (t>150)")
ax.set_title("sustained rate vs layer: negative at EVERY layer (CIs exclude 0);\n"
             "most negative shallow (L2 −0.064, L3 −0.036), ≈ −0.01 deep")

ax = axes[3]
cum_, culo, cuhi = np.array(cu_m), np.array(cu_lo), np.array(cu_hi)
ax.errorbar(x, cum_, yerr=[cum_ - culo, cuhi - cum_], fmt="o-", color="tab:purple", capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"L{s}" for s in Ls])
ax.set_ylabel("Λ(299) = ln[a(300)/a(1)]/299")
ax.set_title("cumulative rate at the last token: negative at every layer L7–L27\n"
             "(L1 dead t≥93, L2 t≥192, L3 t≥291 → no Λ(299) exists)")
NUM["layer_profile"] = {f"L{L}": dict(peak=float(p), peak_t=int(pt), tail=(tm, tl, th), cum=(cm, cl, ch))
                        for L, p, pt, tm, tl, th, cm, cl, ch in
                        zip(Ls, pk, pk_t, tr_m, tr_lo, tr_hi, cu_m, cu_lo, cu_hi)}
fig.suptitle("FIG 6  layer dependence — w100_layers, W=100, r=0.1, 20 dirs, 95% CIs over directions.\n"
             "The response is a memory that DRAINS: contraction grows toward the surface layers and dies to exact\n"
             "float32 zero there; the deepest layers hold the perturbation longest but still at −0.008..−0.010/token.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(os.path.join(OUT, "fig06_layer_profile.png"))
plt.close(fig)
print("fig06 done")
save_json(NUM, "numbers_figs_4_6.json")
