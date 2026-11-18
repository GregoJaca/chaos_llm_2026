"""A1 two-regime analysis (w100): rise, plateau + pre-flush surge, flush, tail.

Uses the floor-free universal curve G(t,l) = median over safe radii (0.03-0.3)
of dh/r, and its per-direction versions.

Outputs:
  fig3_universal_curve.png   G(t) at L27: aggregate + 3 individual dirs
  fig4_regime_fits.png       regime segmentation, rates vs layer, exp vs power fits
  regime_rates.csv          the numbers
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

D = C.load("w100")
dh, dhm, radii, layers = D["dh"], D["dhm"], D["radii"], D["layers"]
G = D["G"]
li27 = int(np.where(layers == 27)[0][0])
small_idx = np.where(D["small"])[0]
safe_idx = np.where(D["safe"])[0]
T = 300


def smooth_log(y, w=5):
    """moving median of log-values, w tokens, NaN-aware."""
    x = np.log(np.where(y > 0, y, np.nan))
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        lo, hi = max(0, t - w // 2), min(len(x), t + w // 2 + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        out[t] = np.median(seg) if len(seg) else np.nan
    return out


def fit_rate(y, t0, t1, logx=False):
    """log-slope of log G over [t0,t1]; per-token if logx else power exponent."""
    t1 = min(t1, len(y) - 1)
    m = np.arange(t0, t1 + 1)
    x = np.log(m) if logx else m.astype(float)
    with np.errstate(divide="ignore"):
        yl = np.log(np.asarray(y)[m])
    ok = np.isfinite(yl)
    if ok.sum() < 5:
        return np.nan, np.nan, ok.sum()
    b, a = np.polyfit(x[ok], yl[ok], 1)
    r2 = 1 - np.sum((yl[ok] - (a + b * x[ok])) ** 2) / np.sum((yl[ok] - yl[ok].mean()) ** 2)
    return b, r2, ok.sum()


rows = ["layer,t_peak,rise_rate,plateau_decay_rate,tail_exp_rate,tail_exp_R2,"
        "tail_pow_exp,tail_pow_R2,surge_G80_over_G50\n"]

LG = [li for li, l in enumerate(layers) if l > 0]
for li in LG:
    l = int(layers[li])
    gs = smooth_log(G[li])
    tpeak = int(np.nanargmax(gs[:30]))
    fl = D["flush"].get(l, 10_000)
    rise, _, _ = fit_rate(G[li], 0, tpeak)
    plat, _, _ = fit_rate(G[li], tpeak + 5, min(70, fl - 12))
    t0, t1 = (110, 280) if l >= 7 else (fl + 8, fl + 60)
    lam, r2e, n = fit_rate(G[li], t0, t1)
    beta, r2p, _ = fit_rate(G[li], t0, t1, logx=True)
    surge = G[li, 80] / G[li, 50]
    rows.append(f"{l},{tpeak},{rise:.4f},{plat:.4f},{lam:.4f},{r2e:.3f},"
                f"{beta:.3f},{r2p:.3f},{surge:.2f}\n")
C.txt("regime_rates.csv", "".join(rows))
print("".join(rows))

# per-radius tail rate at L27 (does decay rate depend on r?)
lam_by_r = []
for ir in [4, 6, 7, 9, 11]:  # r = 0.1, 0.3, 0.5, 1, 2
    lam, r2, _ = fit_rate(dhm[ir, li27, :], 110, 280)
    lam_by_r.append((radii[ir], lam, r2))
print("L27 tail exp rate by radius:", [(r, round(l, 4), round(q, 3)) for r, l, q in lam_by_r])

# ------------------------------------------------------------------ fig 3
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharex=True, sharey=True)
panels = [("aggregate (median of 20 dirs, 7 radii)", None),
          ("direction 0 (typical gain)", 0),
          ("direction 6 (weakest gain)", 6),
          ("direction 10 (strongest gain)", 10)]
for k, (title, idir) in enumerate(panels):
    ax = axes[k // 2][k % 2]
    for ir in small_idx:
        y = dhm[ir, li27, :] / radii[ir] if idir is None else dh[ir, idir, li27, :] / radii[ir]
        ax.plot(y, lw=0.8, alpha=0.55, color=C.rcolor(radii[ir], radii))
    if idir is None:
        ymed = G[li27]
    else:
        ymed = np.array([np.median(dh[safe_idx, idir, li27, t] / radii[safe_idx])
                        for t in range(T)])
    ax.plot(ymed, "k-", lw=2.0, label="G(t) = median $\\delta h/r$")
    ax.plot(np.exp(smooth_log(ymed)), "k--", lw=1.1, label="smoothed")
    ax.set_yscale("log")
    ax.set_ylim(3e-2, 200)
    ax.set_title(title, loc="left")
    ax.set_ylabel(r"$\delta h(t,L27)/r$" if k % 2 == 0 else "")
    if k // 2 == 1:
        ax.set_xlabel("generated token t")
    ax.legend(loc="upper right")
a = axes[0][0]
a.annotate("rise\n(t=0$\\to$$\\sim$17)", xy=(6, 5.5), fontsize=10, color="0.15",
            ha="center")
a.annotate("growth + pre-flush surge", xy=(45, 9.0), fontsize=10, color="0.15")
a.annotate("surge peak t$\\approx$80:\ninjected position near\nwindow edge",
            xy=(80, 13), fontsize=9, color="0.15", ha="center")
a.annotate("tail decay (starts at flush t$\\approx$92)", xy=(150, 1.6), fontsize=10,
            color="0.15")
a.annotate("all 7 radii $r\\leq0.3$ lie on one curve (CV$\\approx$3%);\n"
            "r=1e-3 departs at very late t (float32 floor)", xy=(120, 0.09),
            fontsize=9.5, color="0.25")
fig.suptitle("A1 universal response curve G(t) = $\\delta h(t,L27)/r$ (w100): "
             "rise $\\to$ growth+surge $\\to$ tail decay;\nall 7 linear-regime radii "
             "collapse onto it per direction; directions share the envelope but "
             "differ in token-level oscillations", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.955))
C.save(fig, "fig3_universal_curve.png")

# ------------------------------------------------------------------ fig 4
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

# (a) smoothed G(t) for all layers
ax = axes[0, 0]
for li in LG:
    l = int(layers[li])
    ax.plot(np.exp(smooth_log(G[li])), lw=1.7, color=C.lcolor(l, layers),
            label=f"L{l}")
    fl = D["flush"].get(l)
    if fl:
        ax.axvline(fl, color=C.lcolor(l, layers), ls=":", lw=1.1)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("smoothed G(t,l) = $\\delta h/r$")
ax.set_title("(a) G(t) by layer: flushes at t=92,191,290 (dotted)", loc="left")
ax.legend(ncol=2, fontsize=8)
ax.annotate("spacing $\\approx$ 99 $\\approx$ window $w$", xy=(135, 4e-3),
            fontsize=10, color="0.2")
ax.annotate("L1..L3 hit zero bitwise:\nperturbed KV leaves the window", xy=(194, 3e-3),
            fontsize=9, color="0.35")

# (b) rates vs layer
ax = axes[0, 1]
ll = [int(layers[li]) for li in LG]
rise = [fit_rate(G[li], 0, int(np.nanargmax(smooth_log(G[li])[:30])))[0] for li in LG]
plat = [fit_rate(G[li], int(np.nanargmax(smooth_log(G[li])[:30])) + 5, 70)[0] for li in LG]
tail = [fit_rate(G[li], 110 if int(layers[li]) >= 7 else D["flush"][int(layers[li])] + 8,
                 280 if int(layers[li]) >= 7 else D["flush"][int(layers[li])] + 60)[0]
        for li in LG]
ax.plot(ll, rise, "o-", label="rise rate [0, $t_{peak}$]  (l$\\geq$21 only)")
ax.plot(ll, plat, "s--", label="drift after peak [$t_{peak}$+5, 70]")
ax.plot(ll, tail, "^-", label="tail decay rate [110, 280]")
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xticks(ll); ax.set_xticklabels([str(x) for x in ll])
ax.set_xlabel("layer"); ax.set_ylabel("log-slope of G per token")
ax.set_title("(b) shallow: max at t=0 then decay; deep: grow, then decay", loc="left")
ax.legend(fontsize=8.5)
ax.annotate("no rise for l$\\leq$14:\npeak at t=0", xy=(1.3, -0.004), fontsize=9.5,
            color="0.2")
ax.annotate("|rise| $\\approx$ 2x |tail decay|;\nrise lasts $\\sim$17 tok, "
            "decay $>$200 tok\n$\\Rightarrow$ decay is NOT the mirror of rise",
            xy=(3.2, 0.043), fontsize=9, color="0.2")

# (c) L27 tail: exponential view (aggregate + 3 dirs)
ax = axes[1, 0]
tt = np.arange(110, 281)
for idir, alpha, lw in [(None, 1.0, 2.2), (0, 0.45, 1.1), (6, 0.45, 1.1), (10, 0.45, 1.1)]:
    y = G[li27] if idir is None else np.array(
        [np.median(dh[safe_idx, idir, li27, t] / radii[safe_idx]) for t in range(T)])
    lab = "aggregate" if idir is None else f"dir {idir}"
    ax.plot(tt, y[tt], lw=lw, color="k" if idir is None else "0.55", label=lab)
lam, r2e, _ = fit_rate(G[li27], 110, 280)
ax.plot(tt, np.exp(np.log(G[li27][110]) + lam * (tt - 110)), "r-", lw=1.4,
        label=f"exp fit: $\\lambda$={lam:.4f}/tok, $R^2$={r2e:.3f}")
ax.set_yscale("log"); ax.set_xlabel("generated token t")
ax.set_ylabel("G(t, L27)")
ax.set_title("(c) tail at L27 on semilog: exponential comparison", loc="left")
ax.legend(fontsize=8.5)
ax.annotate("3 individual dirs shown grey:\nrates agree across dirs", xy=(190, 0.9),
            fontsize=9, color="0.3")

# (d) same tail, log-log: power-law view
ax = axes[1, 1]
for idir, alpha, lw in [(None, 1.0, 2.2), (0, 0.45, 1.1), (6, 0.45, 1.1), (10, 0.45, 1.1)]:
    y = G[li27] if idir is None else np.array(
        [np.median(dh[safe_idx, idir, li27, t] / radii[safe_idx]) for t in range(T)])
    ax.plot(tt, y[tt], lw=lw, color="k" if idir is None else "0.55",
            label="aggregate" if idir is None else f"dir {idir}")
beta, r2p, _ = fit_rate(G[li27], 110, 280, logx=True)
ax.plot(tt, np.exp(np.log(G[li27][110]) + beta * (np.log(tt) - np.log(110))), "b-",
        lw=1.4, label=f"power fit: $\\beta$={beta:.2f}, $R^2$={r2p:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t, L27)")
ax.set_title("(d) tail at L27 on log-log: power-law comparison", loc="left")
ax.legend(fontsize=8.5)

fig.suptitle("A1 regimes (w100, G=$\\delta h/r$): shallow layers peak at t=0; deep "
             "layers grow $\\to$ surge at t$\\approx$80 $\\to$ decay from the "
             "window flush;\nneither pure exponential nor power law wins the tail "
             "(both $R^2\\geq0.83$; power slightly better at $l\\geq$14)",
             y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
C.save(fig, "fig4_regime_fits.png")
