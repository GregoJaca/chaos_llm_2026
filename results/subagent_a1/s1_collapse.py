"""A1 collapse analysis (w100, 8 measured layers).

Q: do curves for different r collapse after normalization, and with which exponent?
Answers produced here:
  fig1_collapse_L27.png  raw vs dh/r at the final layer (aggregate + 3 dirs)
  fig2_collapse_quality.png  collapse error per layer; exponent analysis
  collapse_stats.csv     numbers behind the figures
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import matplotlib.pyplot as plt

D = C.load("w100")
dh, radii, layers = D["dh"], D["radii"], D["layers"]
dhm, G, floor = D["dhm"], D["G"], D["floor"]
li27 = int(np.where(layers == 27)[0][0])
small_idx = np.where(D["small"])[0]
large_idx = np.where(D["large"])[0]
safe_idx = np.where(D["safe"])[0]

# ---------------------------------------------------------------- fig 1
rows = [("aggregate (median of 20 dirs)", None),
        ("direction 0 (typical gain)", 0),
        ("direction 6 (weakest gain)", 6),
        ("direction 10 (strongest gain, 10x)", 10)]
fig, axes = plt.subplots(4, 2, figsize=(11.5, 12.2), sharex=True)

for row, (title, idir) in enumerate(rows):
    ax_raw, ax_n = axes[row]
    for ir, r in enumerate(radii):
        y = dhm[ir, li27, :] if idir is None else dh[ir, idir, li27, :]
        lw = 1.9 if r in (0.1, 2.0) else 0.85
        ax_raw.plot(y, color=C.rcolor(r, radii), lw=lw)
        ax_n.plot(y / r, color=C.rcolor(r, radii), lw=lw)
    if idir is None:
        ir = int(np.where(radii == 0.1)[0][0])
        m = dhm[ir, li27, :]
        ax_raw.fill_between(range(300), m - D["iqr"][ir, li27, :] / 2,
                            m + D["iqr"][ir, li27, :] / 2, color="0.5", alpha=0.25,
                            lw=0, label="IQR, r=0.1")
        ax_raw.legend(loc="upper right")
    ax_raw.set_yscale("log"); ax_n.set_yscale("log")
    ax_raw.set_ylim(3e-4, 60); ax_n.set_ylim(3e-4 / 0.3, 60 / 0.001)
    ax_n.set_ylim(1e-1, 2e4)
    ax_raw.set_ylabel(r"$\delta h(t)$  [emb.-norm units]")
    ax_n.set_ylabel(r"$\delta h(t)\,/\,r$")
    ax_raw.set_title(f"{title} — raw", loc="left")
    ax_n.set_title(f"{title} — divided by r", loc="left")
    ax_raw.axhline(floor[li27], color="k", ls=":", lw=0.8)
    if row == 3:
        ax_raw.set_xlabel("generated token t"); ax_n.set_xlabel("generated token t")

axes[0, 0].annotate("curves for different r =\nvertical shifts on log scale",
                    xy=(150, 1.5), fontsize=10, color="0.25")
axes[0, 0].annotate("separating parameter is r (pure scale):\n"
                    r"$\delta h(t)\approx r\,G(t)$ for $r\leq0.3$",
                    xy=(140, 2e-3), fontsize=10, color="0.25")
axes[0, 1].annotate(r"$r\leq0.3$ (7 curves) collapse to ONE curve, CV$\approx$3%"
                    "\n" r"$r\geq0.5$ peel up: nonlinear at depth",
                    xy=(120, 3e3), fontsize=10, color="0.25")
axes[0, 0].annotate("float32 floor", xy=(205, floor[li27] * 1.2),
                    fontsize=8, color="k")
fig.suptitle("A1 response at final layer (w100, L27): r enters as a scale factor "
             "in the linear regime\n(raw left, /r right; colors = r from 1e-3 to 2.0)",
             y=0.995, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
C.save(fig, "fig1_collapse_L27.png")

# ---------------------------------------------------------------- stats
lines = ["layer,CV_all7_preflush_med_pct,CV_all7_preflush_p90_pct,"
         "CV_safe4_allt_med_pct,floorhit_frac_r1em3,floorhit_first_t\n"]
for li, l in enumerate(layers):
    if l == 0:
        continue
    cv7 = []
    for t in range(80):
        v = np.array([np.median(dh[ir, :, li, t]) / radii[ir] for ir in small_idx])
        cv7.append(v.std() / v.mean())
    cv7 = np.array(cv7)
    cvs = []
    for t in range(300):
        v = np.array([np.median(dh[ir, :, li, t]) / radii[ir] for ir in safe_idx])
        if v.mean() > 0:
            cvs.append(v.std() / v.mean())
    cvs = np.array(cvs)
    pred = 0.001 * G[li]
    ok = np.isfinite(pred) & (pred > 0)
    ratio = dhm[0, li][ok] / pred[ok]
    bad = ratio > 1.5
    tb = np.where(ok)[0][bad]
    lines.append(f"{l},{np.median(cv7)*100:.2f},{np.percentile(cv7,90)*100:.2f},"
                 f"{np.median(cvs)*100:.2f},{bad.mean():.3f},"
                 f"{tb.min() if len(tb) else -1}\n")
C.txt("collapse_stats.csv", "".join(lines))
print("".join(lines))

# ---------------------------------------------------------------- fig 2
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

# (a) collapse CV per layer
ax = axes[0, 0]
ll = [l for l in layers if l > 0]
cv7m = [float(np.median([np.array([np.median(dh[ir, :, li, t]) / radii[ir]
                                   for ir in small_idx]).std() /
                         np.array([np.median(dh[ir, :, li, t]) / radii[ir]
                                   for ir in small_idx]).mean()
                        for t in range(80)])) * 100
       for li, l in enumerate(layers) if l > 0]
cvsm = [float(np.median([np.array([np.median(dh[ir, :, li, t]) / radii[ir]
                                   for ir in safe_idx]).std() /
                         max(np.array([np.median(dh[ir, :, li, t]) / radii[ir]
                                       for ir in safe_idx]).mean(), 1e-12)
                        for t in range(300)])) * 100
       for li, l in enumerate(layers) if l > 0]
x = np.arange(len(ll))
ax.bar(x - 0.18, cv7m, 0.34, label="7 radii $\\leq$0.3, t$\\leq$80")
ax.bar(x + 0.18, cvsm, 0.34, label="4 radii 0.03-0.3, all t")
ax.set_xticks(x); ax.set_xticklabels([str(l) for l in ll])
ax.set_xlabel("layer"); ax.set_ylabel("collapse CV of $\\delta h/r$  [%]")
ax.set_title("(a) collapse error after $\\delta h\\to\\delta h/r$", loc="left")
ax.legend(); ax.set_ylim(0, max(cv7m + cvsm) * 1.35)
ax.axhline(5, color="r", ls="--", lw=0.8)
ax.annotate("5%", xy=(len(ll) - 0.6, 5.4), color="r", fontsize=9)

# (b) dh(t=0)/r vs r per layer  (log-log)
ax = axes[0, 1]
for li, l in enumerate(layers):
    if l == 0:
        continue
    y = dhm[:, li, 0] / radii
    ax.plot(radii, y, "o-", ms=3.4, color=C.lcolor(l, layers),
            label=f"L{l}" if l in (1, 7, 27) else None)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("perturbation radius r"); ax.set_ylabel("$\\delta h(t{=}0,l)\\,/\\,r$")
a27 = np.polyfit(np.log(radii[large_idx]),
                 np.log(dhm[large_idx, li27, 0] / radii[large_idx]), 1)[0]
a1 = np.polyfit(np.log(radii[large_idx]),
                np.log(dhm[large_idx, 1, 0] / radii[large_idx]), 1)[0]
ax.annotate(f"slope L27 = {a27:.2f}\n(superlinear)", xy=(0.55, 2.4),
            color=C.lcolor(27, layers), fontsize=10)
ax.annotate(f"slope L1 = {a1:.2f}\n(linear)", xy=(0.55, 0.19),
            color=C.lcolor(1, layers), fontsize=10)
ax.set_title("(b) response per unit $r$ vs $r$: flat $\\leq$0.3,\n"
             "superlinear at depth for $r\\geq$0.5", loc="left")
ax.legend(loc="lower left", ncol=1)

# (c) exponent of the large-r regime vs layer, at two times
ax = axes[1, 0]
lr = np.log(radii[large_idx])
for tmark, ls in [(0, "-"), (50, "--")]:
    alphas = []
    for li, l in enumerate(layers):
        if l == 0:
            continue
        y = np.log(dhm[large_idx, li, tmark])
        alphas.append(np.polyfit(lr, y, 1)[0])
    ax.plot([l for l in layers if l > 0], alphas, ls, "o-", ms=4,
            label=f"t = {tmark}")
ax.axhline(1.0, color="k", ls=":", lw=0.9)
ax.annotate("linear response\n(exponent 1)", xy=(2.5, 1.02), fontsize=9)
ax.set_xlabel("layer"); ax.set_ylabel(r"fit exponent $\alpha(l,t)$ over $r\in[0.5,2]$")
ax.set_title("(c) large-r exponent grows with depth (and t)", loc="left")
ax.legend()

# (d) alpha(l,t) heatmap, large-r set
ax = axes[1, 1]
LG = [li for li, l in enumerate(layers) if l > 0]
A = np.full((len(LG), 300), np.nan)
for i, li in enumerate(LG):
    fl = D["flush"].get(int(layers[li]))
    for t in range(300):
        if fl is not None and t >= fl - 2:
            continue
        y = np.log(dhm[large_idx, li, t])
        A[i, t] = np.polyfit(lr, y, 1)[0]
im = ax.imshow(A, origin="lower", aspect="auto", cmap="magma",
               extent=[0, 300, -0.5, len(LG) - 0.5], vmin=0.95, vmax=2.0)
ax.set_yticks(range(len(LG))); ax.set_yticklabels([str(layers[li]) for li in LG])
ax.set_xlabel("generated token t"); ax.set_ylabel("layer")
ax.set_title(r"(d) $\alpha(l,t)$ in the nonlinear regime: grows with depth AND time",
             loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.015)
cb.set_label(r"$\alpha$")
for i, li in enumerate(LG):
    fl = D["flush"].get(int(layers[li]))
    if fl is not None:
        ax.axvline(fl, color="w", ls="--", lw=1.2)

fig.suptitle("Collapse quality (w100): one exponent ($\\alpha{=}1$) works for "
             "$r\\leq0.3$ at every layer;\nlarge $r$ needs a depth- and "
             "time-dependent exponent — no single global power law", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.955))
C.save(fig, "fig2_collapse_quality.png")

# extra numbers for the report
print("alpha(l, t=0, large-r set):",
      {int(layers[li]): round(np.polyfit(lr, np.log(dhm[large_idx, li, 0]), 1)[0], 3)
       for li, l in enumerate(layers) if l > 0})
print("alpha(l, t=50, large-r set):",
      {int(layers[li]): round(np.polyfit(lr, np.log(dhm[large_idx, li, 50]), 1)[0], 3)
       for li, l in enumerate(layers) if l > 0})
# global single-exponent fit over ALL r at L27 (does one exponent work everywhere?)
y = np.log(dhm[:, li27, 0])
ag = np.polyfit(np.log(radii), y, 1)[0]
resid = y - np.polyval([ag, np.polyfit(np.log(radii), y, 1)[1]], np.log(radii))
print(f"single global exponent L27 all-r: alpha={ag:.3f}, "
      f"max |resid|={np.abs(resid).max():.2f} (spread over 3.6 decades of r)")
