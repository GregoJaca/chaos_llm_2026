"""Figures 10-12: synthetic sanity models, estimator/prompt robustness, flip & margin context."""
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

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.0, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8})

t = tvec(np.zeros((1, 1, T)))
tail_mask = t > TAIL_T0
NUM = {}

# real reference: prompt63 W=4096 (no eviction within 300 tokens), L27, r=0.1
zz = load("window_sweep/w4096/a1.npz")
REAL = zz["dh_norm"][1][:, 3, :]
la5_real = lam_k_2d(REAL, 5)
iqr_real = np.nanpercentile(la5_real, 75) - np.nanpercentile(la5_real, 25)
sigma_step = float(iqr_real * np.sqrt(10) / 1.349)
print("real lambda5 IQR (L27 W4096):", round(iqr_real, 4), "-> synth sigma_step:", round(sigma_step, 3))

# ------------------------------------------------------------------ FIG 10
SY = {
    "chaos (+0.035)": (synth_chaos(lam_true=0.035, sigma=sigma_step), "tab:green", "--"),
    "stable+reset (−0.012)": (synth_stable_reset(lam_true=-0.012, sigma=sigma_step, w=65, reset=0.55), "tab:purple", "-."),
    "transient then −0.012": (synth_transient(lam_asym=-0.012, amp=0.25, tau=18.0, sigma=sigma_step), "tab:orange", ":"),
}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
# A: lambda_5
ax = axes[0, 0]
m, lo, hi = med_band(la5_real)
ax.plot(t, m, color="k", lw=1.8, label="REAL  L27, W=4096, r=0.1")
ax.fill_between(t, lo, hi, color="k", alpha=0.2, lw=0)
for name, (A, c, ls) in SY.items():
    la = lam_k_2d(A, 5)
    m, lo, hi = med_band(la)
    ax.plot(t, m, color=c, ls=ls, lw=1.3, label=name)
    ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(-0.35, 0.35)
ax.set_xlabel("token t")
ax.set_ylabel("λ₅(t)  [per token]")
ax.set_title("λ₅(t): a true positive rate (+0.035, green) is a flat noisy band ABOVE zero —\n"
             "nothing in the real data (black) resembles it")
ax.legend(fontsize=7.5)

# B: Lambda(t)
ax = axes[0, 1]
cum = cum_rate_2d(REAL)
m, lo, hi = med_band(cum)
ax.plot(t, m, color="k", lw=1.8, label="REAL  L27, W=4096")
ax.fill_between(t, lo, hi, color="k", alpha=0.2, lw=0)
for name, (A, c, ls) in SY.items():
    cm = cum_rate_2d(A)
    m, lo, hi = med_band(cm)
    ax.plot(t, m, color=c, ls=ls, lw=1.3, label=name)
    ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("token t")
ax.set_ylabel("Λ(t) = ln[a(t)/a(1)]/t  [per token]")
ax.set_title("Λ(t): chaos → Λ locks at +0.035 (every dir > 0 by t≈60);\n"
             "real data decays through 0 into the stable band within t≈250")
ax.legend(fontsize=7.5)

# C: histogram of Lambda(299)
ax = axes[1, 0]
ax.hist(cum[:, 299], bins=10, color="k", alpha=0.55, label="REAL W=4096 (10 dirs)")
for name, (A, c, ls) in SY.items():
    cm = cum_rate_2d(A)
    ax.hist(cm[:, 299], bins=10, color=c, alpha=0.45, label=name)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Λ(299)  [per token]")
ax.set_ylabel("count of directions")
ax.set_title("final cumulative rate: chaos sits at +0.035 (disjoint from 0);\n"
             "real data overlaps the stable/transient models, not the chaotic one")
ax.legend(fontsize=7.5)

# D: sustained-rate mini-forest with 2-way bootstrap CIs
ax = axes[1, 1]
labels, ms, los, his = [], [], [], []
for name, (A, c, ls) in list(SY.items()) + [("REAL L27 W=4096", (REAL, "k", "-"))]:
    la10 = lam_k_2d(A, 10)
    e = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(e)
    m2, lo2, hi2 = two_way_boot(la10, tail_mask, block=10, n_boot=600)
    labels.append(name); ms.append(m); los.append(lo2); his.append(hi2)
y = np.arange(len(labels))
ax.errorbar(ms, y, xerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)],
            fmt="o", capsize=4, color="tab:blue")
for i, (m, lo, hi) in enumerate(zip(ms, los, his)):
    ax.text(hi + 0.002, i, f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]", fontsize=7, va="center")
ax.axvline(0, color="k", lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("sustained rate: mean λ₁₀ (t>150), 95% dir+token-block CI  [per token]")
ax.set_title("sustained-rate CIs: chaos [+0.031,+0.039] excludes 0 by a wide margin;\n"
             "real [−0.0055,−0.0006] hugs zero from below — indistinguishable from stable+reset")
NUM["synth"] = dict(sigma_step=sigma_step,
                    chaos=dict(m=ms[0], lo=los[0], hi=his[0]),
                    stable=dict(m=ms[1], lo=los[1], hi=his[1]),
                    transient=dict(m=ms[2], lo=los[2], hi=his[2]),
                    real=dict(m=ms[3], lo=los[3], hi=his[3]))
fig.suptitle("FIG 10  synthetic sanity check — same estimators, T=300, 20 dirs, noise σ matched to the real λ₅ IQR.\n"
             "What a REAL positive rate looks like: λ₅ flat above 0, Λ rising to the exponent, all directions positive.\n"
             "The teacher-forced data sit unambiguously on the contracting side (λ ≤ 0).",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(OUT, "fig10_synthetic.png"))
plt.close(fig)
print("fig10 done")

# ------------------------------------------------------------------ FIG 11
z = load("w100_layers/a1.npz")
d = z["dh_norm"][4]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
# A: k sensitivity
ax = axes[0]
for k, c in [(5, "tab:red"), (10, "tab:blue")]:
    la = lam_k_2d(d[:, 7, :], k)
    m, lo, hi = med_band(la)
    ax.plot(t, m, color=c, lw=1.3, label=f"λ{k}(t)")
    ax.fill_between(t, lo, hi, color=c, alpha=0.2, lw=0)
ax.axhline(0, color="k", lw=0.8)
diff = np.nanmedian(np.abs(lam_k_2d(d[:, 7, :], 5) - lam_k_2d(d[:, 7, :], 10)))
ax.set_xlabel("token t")
ax.set_ylabel("λₖ(t)  [per token]")
ax.set_title(f"half-width k=5 vs k=10 (L27, W=100, r=0.1): |λ₅−λ₁₀| median = {diff:.4f}\n"
             "— estimator choice does not affect any conclusion")
ax.legend()
NUM["k_median_absdiff"] = float(diff)

# B: per-prompt sustained rates
ax = axes[1]
names, vals, lo_, hi_ = [], [], [], []
for p in ["p1", "p2", "p3", "p4", "p5"]:
    A = load(f"rate_prompts/{p}/a1.npz")["dh_norm"][0][:, 3, :]
    la10 = lam_k_2d(A, 10)
    e = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(e)
    names.append(f"W100\n{p}"); vals.append(m); lo_.append(lo); hi_.append(hi)
for p in ["p1", "p2", "p3"]:
    A = load(f"rate_w50/{p}/a1.npz")["dh_norm"][0][:, 3, :]
    la10 = lam_k_2d(A, 10)
    e = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(e)
    names.append(f"W50\n{p}"); vals.append(m); lo_.append(lo); hi_.append(hi)
for name in ["prompts_long_w100", "prompts_long2_w100", "prompts_long_w4096", "prompts_long2_w4096"]:
    A = load(f"rate_stats/{name}/a1.npz")["dh_norm"][1][:, 3, :]
    la10 = lam_k_2d(A, 10)
    e = np.nanmean(la10[:, tail_mask], axis=1)
    m, lo, hi = boot_dirs_stat(e)
    nm = ("longP65 " if "long2" not in name else "longP63 ") + ("W100" if "w100" in name else "W4096")
    names.append(nm); vals.append(m); lo_.append(lo); hi_.append(hi)
x = np.arange(len(names))
ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo_), np.array(hi_) - np.array(vals)],
            fmt="o", capsize=3, color="tab:green")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel("sustained rate mean λ₁₀ (t>150)  [per token]")
ax.set_title("12 independent prompt runs (8–10 dirs each, 95% CI): 0/12 positive;\n"
             "spread −0.002…−0.039, W=50 runs contract hardest")
NUM["prompt_runs"] = {n: dict(m=v, lo=a, hi=b) for n, v, a, b in zip(names, vals, lo_, hi_)}

# C: Lambda(t) per prompt
ax = axes[2]
cums = []
for p in ["p1", "p2", "p3", "p4", "p5"]:
    A = load(f"rate_prompts/{p}/a1.npz")["dh_norm"][0][:, 3, :]
    cums.append(cum_rate_2d(A))
    ax.plot(t, np.nanmedian(cums[-1], axis=0), lw=0.9, alpha=0.7, label=f"prompt {p}")
allc = np.vstack(cums)
m, lo, hi = med_band(allc)
ax.plot(t, m, color="k", lw=2.0, label="median over 5 prompts × 8 dirs")
ax.fill_between(t, lo, hi, color="k", alpha=0.2, lw=0)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("token t")
ax.set_ylabel("Λ(t)  [per token]")
ax.set_title("cumulative rate, 5 prompts (W=100, r=0.1, L27): every prompt is\n"
             "monotonically ≈ flat-to-decaying; none shows a rising Λ")
ax.legend(fontsize=7)
fig.suptitle("FIG 11  robustness — estimator half-width k, prompt identity, and the rate_prompts / rate_stats grids.\n"
             "Conclusions are unchanged: no prompt, no k, no run produces a positive sustained rate.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig11_robustness.png"))
plt.close(fig)
print("fig11 done")

# ------------------------------------------------------------------ FIG 12
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
# A: teacher-forced flips vs radius
ax = axes[0]
z = load("w100_layers/a1.npz")
frac = (z["pred_T"] >= 0).mean(axis=1)
ax.plot(z["radii"], frac, "o-", color="tab:red")
ax.set_xscale("log")
ax.set_xlabel("perturbation radius r")
ax.set_ylabel("fraction of 20 dirs whose top-1 token flips")
ax.set_title("teacher-forced flips (W=100, 300 tokens): NONE for r ≤ 0.1;\n"
             "1/20 at r=0.2, 19/20 at r=2 — rates computed at r ≤ 0.3 are flip-free")
NUM["flip_frac"] = {f"{r:g}": float(f) for r, f in zip(z["radii"], frac)}

# B: free-run divergence horizons
ax = axes[1]
spec = [
    ("ds W=100\nr=1e-4..1e-2", os.path.join(BIG, "a1_ds_w100", "free_run", "free_run.npz"), [0, 1, 2]),
    ("ds W=100\nr=0.03..0.3", os.path.join(BIG, "a1_ds_w100", "free_run_large", "free_run.npz"), [0, 1, 2]),
    ("cyber W=100\nr=0.01, 0.1", os.path.join(BIG, "a1_cyber_w100", "free_run", "free_run.npz"), [0, 1]),
    ("cyber W=100\nr=0.3", os.path.join(BIG, "a1_cyber_w100", "free_run", "free_run.npz"), [2]),
    ("ds W=200\nr=0.1, 0.3", os.path.join(BIG, "a1_ds_w200_large", "free_run", "free_run.npz"), [0, 1]),
    ("ds W=500\nr=0.1, 0.3", os.path.join(BIG, "a1_ds_w500_large", "free_run", "free_run.npz"), [0, 1]),
]
NUM["freerun"] = {}
for gi, (tag, f, rows) in enumerate(spec):
    zzd = np.load(f, allow_pickle=True)
    pred, rad = zzd["pred"], zzd["radii"]
    allv, nodiv = [], 0
    for ri in rows:
        allv.extend(int(v) for v in pred[ri])
        nodiv += int(np.sum(pred[ri] < 0))
        NUM["freerun"][f"{tag.replace(chr(10),' ')} r={rad[ri]:g}"] = dict(
            n_no_div=int(np.sum(pred[ri] < 0)), n=int(len(pred[ri])),
            mean_horizon=float(np.mean(pred[ri][pred[ri] >= 0])) if np.any(pred[ri] >= 0) else None,
            max_horizon=int(np.max(pred[ri])) if np.any(pred[ri] >= 0) else None)
    div = [v for v in allv if v >= 0]
    rng = np.random.default_rng(10 + gi)
    xs = [gi + rng.uniform(-0.2, 0.2) for _ in div]
    ax.scatter(xs, div, s=15, color="tab:blue", alpha=0.8)
    ax.text(gi, 288, f"{nodiv}/{len(allv)}\nno div.", fontsize=6.5, ha="center")
ax.axhline(300, color="k", lw=0.8, ls=":")
ax.set_ylim(-5, 310)
ax.set_xticks(range(len(spec)))
ax.set_xticklabels([s[0] for s in spec], fontsize=7)
ax.set_ylabel("tokens until first divergence (cap 300)")
ax.set_title("free-running at r ≤ 0.1: 174/180 runs NEVER diverge in 300 tokens; the 6 exceptions all\n"
             "flip at the SAME token t=44 (one vulnerable near-tie), not by progressive separation")

# C: lambda_5 vs margin
ax = axes[2]
margins = z["margins"]
la5 = lam_k_2d(d[:, 7, :], 5)
sl = slice(5, 295)
xs_ = np.tile(np.log(np.maximum(margins[sl], 1e-6)), (20, 1))
ys_ = la5[:, sl]
mask = np.isfinite(ys_)
r = np.corrcoef(xs_[mask], ys_[mask])[0, 1]
ax.scatter(xs_[mask], ys_[mask], s=5, alpha=0.25, color="tab:gray")
# binned median
xb = np.linspace(np.nanmin(xs_[mask]), np.nanmax(xs_[mask]), 12)
bmv, bmm = [], []
for i in range(len(xb) - 1):
    sel = (xs_[mask] >= xb[i]) & (xs_[mask] < xb[i + 1])
    if sel.sum() > 30:
        bmv.append(np.nanmedian(ys_[mask][sel]))
        bmm.append((xb[i] + xb[i + 1]) / 2)
ax.plot(bmm, bmv, "o-", color="tab:red", lw=1.5, label="binned median")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("ln(top1 − top2 logit margin) at token t")
ax.set_ylabel("λ₅(t) per direction")
ax.set_title(f"λ₅ vs decision margin (L27, 20 dirs pooled): r = {r:+.3f}\n"
             "— no correlation: near-tie tokens do NOT produce positive stretching")
ax.legend()
NUM["margin_corr"] = float(r)
fig.suptitle("FIG 12  context — when do trajectories actually diverge? Teacher-forced flips need r ≥ 0.2;\n"
             "free-running divergence at r ≤ 0.1 is a 6/180 event concentrated at a single vulnerable token (t=44)\n"
             "— a tipping point, not exponential separation. Local λ is uncorrelated with the margin (r=+0.09).",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig12_context.png"))
plt.close(fig)
print("fig12 done")
save_json(NUM, "numbers_figs_10_12.json")
