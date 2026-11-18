"""A1 layer (depth) dependence (w100). What separates curves across depth?

Outputs:
  fig5_depth.png      G(t) by layer, heatmap, amplification A(l), flush times,
                      marginal per-layer gain
  depth_stats.csv
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

D = C.load("w100")
dh, dhm, radii, layers = D["dh"], D["dhm"], D["radii"], D["layers"]
G = D["G"]
T, LG = 300, [li for li, l in enumerate(layers) if l > 0]
ll = [int(layers[li]) for li in LG]

# amplification A(l) = dh(t=0)/r, per direction (median over safe radii)
safe_idx = np.where(D["safe"])[0]
A = np.zeros((len(LG), 20))
for i, li in enumerate(LG):
    for idir in range(20):
        A[i, idir] = np.median(dh[safe_idx, idir, li, 0] / radii[safe_idx])
Amed = np.median(A, axis=1)

# fits: exponential vs power law in l
lexp = np.polyfit(ll, np.log(Amed), 1)
res_e = np.log(Amed) - np.polyval(lexp, ll)
r2_e = 1 - res_e.var() / np.log(Amed).var()
lpow = np.polyfit(np.log(ll), np.log(Amed), 1)
res_p = np.log(Amed) - np.polyval(lpow, np.log(ll))
r2_p = 1 - res_p.var() / np.log(Amed).var()
print(f"A(l) fits: exp  A~exp({lexp[0]:.4f} l): R2={r2_e:.3f}, max|res|={np.abs(res_e).max():.2f}")
print(f"           pow  A~l^{lpow[0]:.3f}:    R2={r2_p:.3f}, max|res|={np.abs(res_p).max():.2f}")
print("A(l) median:", dict(zip(ll, np.round(Amed, 3).tolist())))
marg = ["layer,marginal_gain_per_layer"]
for i in range(len(ll) - 1):
    marg.append(f"{ll[i]},{(Amed[i+1]/Amed[i])**(1.0/(ll[i+1]-ll[i])):.3f}")
C.txt("depth_stats.csv", "\n".join(marg) + "\n")
print("\n".join(marg))

# ---------------------------------------------------------------- fig 5
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))

# (a) G(t) by layer
ax = axes[0, 0]
for i, li in enumerate(LG):
    l = ll[i]
    ax.plot(G[li], lw=1.6, color=C.lcolor(l, layers), label=f"L{l}")
    fl = D["flush"].get(l)
    if fl:
        ax.axvline(fl, color=C.lcolor(l, layers), ls=":", lw=1.0)
ax.set_yscale("log")
ax.set_xlabel("generated token t"); ax.set_ylabel("G(t,l) = $\\delta h/r$")
ax.set_title("(a) depth separates the curves (r fixed):\ngain A(l) grows with depth",
             loc="left")
ax.legend(ncol=2, fontsize=8)
ax.annotate("bitwise flush:\nL1 t=92, L2 191, L3 290", xy=(120, 1.2e-3), fontsize=9,
            color="0.3")

# (b) heatmap of log10 G (white = bitwise-flushed, exactly zero)
ax = axes[0, 1]
H = np.full((len(LG), T), np.nan)
for i, li in enumerate(LG):
    with np.errstate(divide="ignore"):
        h = np.log10(np.where(G[li] > 0, G[li], np.nan))
    H[i] = h
im = ax.imshow(H, origin="lower", aspect="auto", extent=[0, T, -0.5, len(LG) - 0.5],
               cmap="magma", vmin=-3.2, vmax=1.0)
ax.set_yticks(range(len(LG))); ax.set_yticklabels([str(x) for x in ll])
ax.set_xlabel("generated token t"); ax.set_ylabel("layer")
ax.set_title("(b) map of $\\log_{10}G$: amplification grows downward,\n"
             "white = bitwise zero (flushed out of window)", loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.015); cb.set_label("$\\log_{10}G$")
for i, l in enumerate(ll):
    fl = D["flush"].get(l)
    if fl:
        ax.axvline(fl, color="w", ls="--", lw=1.3)

# (c) A(l) with fits
ax = axes[0, 2]
ax.errorbar(ll, Amed, yerr=np.vstack([Amed - np.percentile(A, 25, axis=1),
                                      np.percentile(A, 75, axis=1) - Amed]),
            fmt="o", color="k", ms=5, capsize=3, lw=1, label="median over 20 dirs (IQR)")
xs = np.linspace(1, 27, 100)
ax.plot(xs, np.exp(np.polyval(lexp, xs)), "r--", lw=1.2,
        label=f"exp: $e^{{{lexp[0]:.3f}l}}$, $R^2$={r2_e:.2f}")
ax.plot(xs, np.exp(np.polyval(lpow, np.log(xs))), "b:", lw=1.6,
        label=f"pow: $l^{{{lpow[0]:.2f}}}$, $R^2$={r2_p:.2f}")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xticks(ll); ax.set_xticklabels([str(x) for x in ll])
ax.set_xlabel("layer"); ax.set_ylabel("A(l) = $\\delta h(t{=}0,l)/r$")
ax.set_title("(c) depth amplification: front-loaded\n(first 3 layers do most of it)",
             loc="left")
ax.legend(fontsize=8)

# (d) marginal gain per layer
ax = axes[1, 0]
mid = [(ll[i] + ll[i + 1]) / 2 for i in range(len(ll) - 1)]
mg = [(Amed[i + 1] / Amed[i]) ** (1.0 / (ll[i + 1] - ll[i])) for i in range(len(ll) - 1)]
ax.plot(mid, mg, "o-")
ax.axhline(1.0, color="k", ls=":", lw=0.9)
for x, y in zip(mid, mg):
    ax.annotate(f"{y:.2f}", xy=(x, y * 1.03), fontsize=8.5, ha="center")
ax.set_xscale("log"); ax.set_xticks(mid); ax.set_xticklabels([str(x) for x in mid])
ax.set_xlabel("between layers"); ax.set_ylabel("geometric mean gain per layer")
ax.set_title("(d) per-layer gain: 1.9x at L1$\\to$2, ~1.04 mid-network,\n"
             "1.10x again at L21$\\to$27", loc="left")

# (e) flush times
ax = axes[1, 1]
fl = D["flush"]
ax.plot(list(fl.keys()), list(fl.values()), "o-")
for l, t in fl.items():
    ax.annotate(f"t*={t}", xy=(l, t + 12), fontsize=9, ha="center")
d = [fl[2] - fl[1], fl[3] - fl[2]]
ax.annotate(f"spacing = {d[0]}, {d[1]} $\\approx$ w$-$1 (w=100):\none attention "
            "window per extra layer", xy=(1.6, 120), fontsize=9.5, color="0.2")
ax.set_xlabel("layer"); ax.set_ylabel("bitwise flush time t*")
ax.set_title("(e) the perturbation leaves the sliding window\none window-length "
             "later per layer", loc="left")
ax.set_ylim(60, 340)

# (f) where behavior changes: rise vs no-rise
ax = axes[1, 2]
tp = []
for i, li in enumerate(LG):
    gs = G[li].copy()
    sm = np.array([np.median(gs[max(0, t - 2):t + 3]) for t in range(30)])
    tp.append(int(np.argmax(sm)))
bar = ax.bar([str(x) for x in ll], tp, color=[C.lcolor(x, layers) for x in ll])
for i, v in enumerate(tp):
    ax.annotate(str(v), xy=(i, v + 0.4), ha="center", fontsize=9)
ax.set_xlabel("layer"); ax.set_ylabel("$t_{peak}$ (tokens)")
ax.set_title("(f) rise appears only at depth:\n$t_{peak}$=0 for l$\\leq$14, "
             "then 14, 17", loc="left")

fig.suptitle("A1 layer dependence (w100): depth sets the gain A(l) "
             "(front-loaded, $\\times$12 over 27 layers), the rise time, and the "
             "memory (flush t* $\\approx$ 92 + 99(l$-$1));\nlayer 0 is identically "
             "zero (perturbation enters after the embedding output)", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
C.save(fig, "fig5_depth.png")
