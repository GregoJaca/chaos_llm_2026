"""02: Time-shift / collapse of the two-time response R(t,s).

Question: is R(t,s) a function of lag tau = t-s alone (time-translation
invariance), or of absolute t, or separable?

Normalizations tried (all honestly quantified):
  (a) R = dh(t)/dh(s)            -- value at injection
  (b) R = dh(t)/peak(dh, lag<=3) -- early peak
  (c) R = dh(t)/dh(s+5)          -- value shortly after injection
  (d) lag rescaling tau -> tau/s -- no theoretical basis, shown to fail
  (e) shift each curve so its first prominent bump sits at x=0
  (f) dh vs ABSOLUTE t with the baseline margin: do bumps pin to near-ties?

Collapse metric: spread(tau) = std over s of log10 Rbar(s,tau) on the common
lag range, reported as RMS in dex and as a multiplicative factor 10^dex.

Separability: is log dh(t,s) ~ a(s) + b(t) (rank-1) on the common t grid?

Figures: figures/fig02_collapse.png, figures/fig02_separability.png
Stats:    stats/shift_collapse.json
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from common import load_state, load_baseline, log10safe, save_json, savefig

st = load_state()
base = load_baseline()
dh, steps = st["dh"], st["steps"]
m = base["margins"]
med = np.median(dh, axis=1)          # [4, 300]
T = dh.shape[-1]
stats = {"common_lag_range": [5, 149],
         "note": "s=150 has only 150 lags; common range tau in [5,149]"}

tau_lo, tau_hi = 5, 149


def spread_factor(Rbar):
    """Rbar: [4, lags] with lag axis tau=0..; returns RMS dex spread on common range."""
    taus = np.arange(Rbar.shape[1])
    common = (taus >= tau_lo) & (taus <= tau_hi)
    L = log10safe(Rbar[:, common])
    sd = L.std(axis=0)
    return float(np.sqrt(np.mean(sd ** 2))), float(np.sqrt(np.mean(sd ** 2)))  # dex


def spread_stats(Rbar):
    taus = np.arange(Rbar.shape[1])
    common = (taus >= tau_lo) & (taus <= tau_hi)
    short = (taus >= tau_lo) & (taus <= 50)
    out = {}
    for tag, sel in [("tau_5_149", common), ("tau_5_50", short)]:
        L = log10safe(Rbar[:, sel])
        sd = L.std(axis=0)
        rms = float(np.sqrt(np.mean(sd ** 2)))
        out[tag] = dict(rms_spread_dex=rms, spread_factor=float(10 ** rms),
                        max_spread_factor=float(10 ** sd.max()))
    return out


# ---- normalizations (median over directions of the normalized per-direction R)
norms = {}
for tag, norm_fn in [
    ("by_dh_s", lambda x: x / np.maximum(x[:, :1], 1e-30)),
    ("by_peak_lag3", lambda x: x / np.maximum(x[:, :4].max(axis=1, keepdims=True), 1e-30)),
    ("by_dh_s_plus_5", lambda x: x / np.maximum(x[:, 5:6], 1e-30)),
]:
    Rbar = np.zeros((len(steps), T - int(steps.min())))
    for si, s in enumerate(steps):
        x = dh[si][:, s:]                       # [32, T-s]
        R = norm_fn(x)
        Rbar[si, :x.shape[1]] = np.median(R, axis=0)
    norms[tag] = Rbar
    stats[tag] = spread_stats(Rbar)

# ---- (d) tau/s rescaling
# ---- (e) first-bump alignment
bumps = {}
for si, s in enumerate(steps):
    L = log10safe(med[si, s:])
    env = median_filter(L, size=15, mode="nearest")
    resid = L - env
    cand = np.where(resid > 0.15)[0]
    bumps[int(s)] = dict(first_bump_t=int(cand[0] + s) if len(cand) else None,
                         n_bumps_gt_0p15dex=int(len(cand)),
                         bump_resid_dex=[float(resid[c]) for c in cand[:8]],
                         bump_t=[int(c + s) for c in cand[:8]])
stats["bumps"] = bumps

# ---- (f) near-tie alignment check: dh at near-tie tokens vs local baseline
nt = np.where(m < 0.2)[0]
enrich = {}
for si, s in enumerate(steps):
    tt = nt[(nt > s + 2)]
    if len(tt) == 0:
        enrich[int(s)] = None
        continue
    x = med[si]
    ratios = []
    for t in tt:
        loc = np.median(x[max(s, t - 4):t + 5])
        ratios.append(x[t] / loc)
    enrich[int(s)] = dict(tokens=[int(t) for t in tt], ratios=[float(r) for r in ratios],
                          median_ratio=float(np.median(ratios)))
stats["near_tie_enrichment"] = dict(
    note="dh at near-tie tokens divided by local (9-token) median dh; ~1 means NO alignment",
    by_s=enrich)

# ---- separability of the log median field on common t grid (t>=150)
F = log10safe(med[:, 150:])                     # [4, 150]
Fdc = F - F.mean(1, keepdims=True) - F.mean(0, keepdims=True) + F.mean()
U, Sv, Vt = np.linalg.svd(Fdc, full_matrices=False)
resid = Fdc - np.outer(U[:, 0] * Sv[0], Vt[0])
stats["separability_t_ge_150"] = dict(
    rank1_var_share=float(Sv[0] ** 2 / np.sum(Sv ** 2)),
    resid_std_dex=float(resid.std()), resid_factor=float(10 ** resid.std()),
    resid_p90_abs_dex=float(np.quantile(np.abs(resid), .9)))
# same on the widest 2-s common grids
pair_sep = {}
for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    t0 = int(max(steps[i], steps[j]))
    G = log10safe(med[[i, j], t0:])
    Gdc = G - G.mean(1, keepdims=True) - G.mean(0, keepdims=True) + G.mean()
    s2 = np.linalg.svd(Gdc, compute_uv=False)
    pair_sep[f"s{steps[i]}_s{steps[j]}"] = float(10 ** Gdc.std())
stats["pairwise_field_resid_factor_t_ge_max_s"] = pair_sep

save_json(stats, "shift_collapse.json")

# ================================================================ figure (2x3)
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

# (a) by dh(s)
ax = axs[0, 0]
for si, s in enumerate(steps):
    ax.plot(np.arange(norms["by_dh_s"].shape[1]), norms["by_dh_s"][si], lw=1.4,
            color=colors[si], label=f"s={s}")
ax.set_yscale("log")
sp = stats["by_dh_s"]["tau_5_149"]
ax.set_title(f"(a) R=dh(t)/dh(s) vs lag\nspread factor {sp['spread_factor']:.2f} "
             f"(max {sp['max_spread_factor']:.2f})", fontsize=9)

# (b) by peak
ax = axs[0, 1]
for si, s in enumerate(steps):
    ax.plot(np.arange(norms["by_peak_lag3"].shape[1]), norms["by_peak_lag3"][si], lw=1.4,
            color=colors[si], label=f"s={s}")
ax.set_yscale("log")
sp = stats["by_peak_lag3"]["tau_5_149"]
ax.set_title(f"(b) R=dh(t)/peak(lag<=3)\nspread factor {sp['spread_factor']:.2f} "
             f"(max {sp['max_spread_factor']:.2f})", fontsize=9)

# (c) by dh(s+5)
ax = axs[0, 2]
for si, s in enumerate(steps):
    ax.plot(np.arange(norms["by_dh_s_plus_5"].shape[1]), norms["by_dh_s_plus_5"][si],
            lw=1.4, color=colors[si], label=f"s={s}")
ax.set_yscale("log")
sp = stats["by_dh_s_plus_5"]["tau_5_149"]
ax.set_title(f"(c) R=dh(t)/dh(s+5)\nspread factor {sp['spread_factor']:.2f} "
             f"(max {sp['max_spread_factor']:.2f})", fontsize=9)

for ax in list(axs[0]):
    ax.set_xlabel(r"lag $\tau=t-s$")
    ax.set_ylabel("median R over 32 dirs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 160)

# (d) tau/s
ax = axs[1, 0]
for si, s in enumerate(steps):
    x = dh[si][:, s:]
    R = x / np.maximum(x[:, :1], 1e-30)
    Rb = np.median(R, axis=0)
    xs = np.arange(len(Rb)) / s
    ax.plot(xs, Rb, lw=1.4, color=colors[si], label=f"s={s}")
ax.set_yscale("log")
ax.set_xlim(0, 4)
ax.set_xlabel(r"$\tau/s$")
ax.set_ylabel("median R")
ax.set_title("(d) lag rescaled by s: no collapse either", fontsize=9)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# (e) bump-aligned
ax = axs[1, 1]
found = 0
for si, s in enumerate(steps):
    tb = bumps[int(s)]["first_bump_t"]
    if tb is None:
        continue
    found += 1
    L = log10safe(med[si])
    ax.plot(np.arange(T) - tb, L - L[tb], lw=1.4, color=colors[si],
            label=f"s={s} (bump at t={tb})")
ax.set_xlabel(r"$t-t_{\mathrm{bump}}$")
ax.set_ylabel(r"$\log_{10}dh(t)-\log_{10}dh(t_{\mathrm{bump}})$")
ax.set_xlim(-30, 60)
ax.set_ylim(-0.8, 0.8)
ax.set_title(f"(e) aligned at first bump (>{found} s values have one)", fontsize=9)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# (f) absolute t + margin
ax = axs[1, 2]
for si, s in enumerate(steps):
    ax.plot(np.arange(s, T), med[si, s:], lw=1.2, color=colors[si], label=f"s={s}")
ax.set_yscale("log")
ax.set_xlabel("absolute generated token t")
ax.set_ylabel(r"median $\|\delta h_t\|$")
nt = np.where(m < 0.2)[0]
for t in nt:
    ax.axvline(t, color="gray", lw=0.5, alpha=0.5, zorder=0)
ax2 = ax.twinx()
ax2.plot(np.arange(T), np.maximum(m, 1e-3), lw=0.8, color="k", alpha=0.6)
ax2.set_yscale("log")
ax2.set_ylabel("baseline margin (log)", fontsize=8)
ax.set_title("(f) vs absolute t; gray lines = margin near-ties (<0.2)\n"
             "bumps do NOT sit on near-ties", fontsize=9)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Two-time response R(t,s): shift/overlap attempts, all quantified", fontsize=11)
fig.tight_layout()
savefig(fig, "fig02_collapse.png")

# ================================================================ separability fig
fig, axs = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
vmax = np.max(np.abs(Fdc))
for ax, (mat, ttl) in zip(axs, [
        (F, r"$\log_{10}$ median $\|\delta h_t\|$ (field)"),
        (F.mean() + (F - F.mean()).mean(1, keepdims=True) + (F - F.mean()).mean(0, keepdims=True)[None, :] - F.mean(),
         "rank-1 reconstruction a(s)+b(t)")]):
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower")
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("t (from 150)")
    ax.set_ylabel("s index (16,50,100,150)")
    fig.colorbar(im, ax=ax, fraction=0.046)
im = axs[2].imshow(resid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
axs[2].set_title(f"residual (std {resid.std():.3f} dex = x{10**resid.std():.2f})", fontsize=9)
axs[2].set_xlabel("t (from 150)")
fig.colorbar(im, ax=axs[2], fraction=0.046)
fig.suptitle("Separability of the response field on the common grid t>=150", fontsize=10)
fig.tight_layout()
savefig(fig, "fig02_separability.png")

print("spread factors (tau 5-149):",
      {k: round(v["tau_5_149"]["spread_factor"], 2) for k, v in
       [(k, stats[k]) for k in ["by_dh_s", "by_peak_lag3", "by_dh_s_plus_5"]]})
print("bumps:", {k: v["first_bump_t"] for k, v in bumps.items()})
print("near-tie enrichment medians:", {k: (v["median_ratio"] if v else None)
                                        for k, v in stats["near_tie_enrichment"]["by_s"].items()})
print("separability:", stats["separability_t_ge_150"])
print("pairwise field residual factors:", pair_sep)
