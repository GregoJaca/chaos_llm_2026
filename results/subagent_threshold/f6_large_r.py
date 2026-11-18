"""Fig 6: large radii and the failure of the linear threshold.

Panel A: flip fraction vs radius, observed (free runs, solid markers) vs
         predicted by the linear |dm| rule (dashed) and signed rule (dotted).
Panel B: linearity test -- distribution of r_c(r)/r_c(0.01) over (d,t):
         quantile bands vs r. Linear regime == band hugging 1.
Panel C: median gain |dm(r)|/r vs r (superlinear growth breaks the
         small-r extrapolation from above and below).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FREE, Case, OUT

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# ---------- Panel A: flip fractions ----------
ax = axes[0]
c100 = Case("w100")
c200 = Case("w200")
c500 = Case("w500")


def obs_radii(case_name, free_key):
    import json
    with open(FREE[free_key]) as f:
        j = json.load(f)
    return [(r, np.array(j["obs"][i])) for i, r in enumerate(j["radii"])]


def pred_frac(case, signed=False):
    rs, fr = [], []
    rc = c100.rc if case is c100 else None
    for ir, r in enumerate(case.radii):
        if signed:
            rcn = np.where(case.dm[ir] < 0, case.rc[ir], np.inf)
        else:
            rcn = case.rc[ir]
        first = np.where((rcn <= r).any(axis=1), (rcn <= r).argmax(axis=1), -1)
        rs.append(r)
        fr.append((first >= 0).mean())
    return np.array(rs), np.array(fr)


for case, col, lab in [(c100, "tab:blue", "w100"), (c200, "tab:orange", "w200"),
                       (c500, "tab:green", "w500")]:
    rp, fp = pred_frac(case)
    ax.semilogx(rp, fp * 100, "--", color=col, lw=1.4, label=f"{lab} pred $|\\delta m|$")
obs_sets = {
    "w100": obs_radii("w100", "w100") + obs_radii("w100", "w100L"),
    "w200": obs_radii("w200", "w200"),
    "w500": obs_radii("w500", "w500"),
}
for lab, col in [("w100", "tab:blue"), ("w200", "tab:orange"), ("w500", "tab:green")]:
    pts = obs_sets[lab]
    rs = [r for r, o in pts]
    fs = [ (o >= 0).mean() * 100 for r, o in pts]
    ax.semilogx(rs, fs, "o", color=col, ms=9, mec="k", mew=0.6, label=f"{lab} observed",
                zorder=5)
ax.set_xlabel("kick radius r")
ax.set_ylabel("directions flipped (%)")
ax.set_title("Flip fraction vs radius", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.axvspan(0.5, 2.2, color="tab:red", alpha=0.08)
ax.annotate("nonlinear regime", xy=(0.9, 8), fontsize=8, color="tab:red", rotation=90)

# ---------- Panel B: threshold ratio bands ----------
ax = axes[1]
base = c100.rc[2]           # r_ref = 0.01 slice
radii = c100.radii
xs, bands = [], []
for ir, r in enumerate(radii):
    if r < 0.01:            # below float32 resolution of hidden norms ~ skip
        continue
    ok = np.isfinite(c100.rc[ir]) & np.isfinite(base)
    ratio = c100.rc[ir][ok] / base[ok]
    q = np.quantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95])
    xs.append(r)
    bands.append(q)
bands = np.array(bands)
ax.fill_between(xs, bands[:, 0], bands[:, 4], alpha=0.25, color="tab:blue", label="5-95%")
ax.fill_between(xs, bands[:, 1], bands[:, 3], alpha=0.45, color="tab:blue", label="25-75%")
ax.loglog(xs, bands[:, 2], "o-", color="k", ms=4, lw=1.4, label="median")
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.axhline(2.0, color="0.5", lw=0.6, ls=":")
ax.axhline(0.5, color="0.5", lw=0.6, ls=":")
for ir, r in enumerate(radii):
    if r >= 0.1:
        ok = np.isfinite(c100.rc[ir]) & np.isfinite(base)
        ratio = c100.rc[ir][ok] / base[ok]
        out = ((ratio > 2) | (ratio < 0.5)).mean()
        ax.annotate(f"{out*100:.0f}% out", xy=(r, np.quantile(ratio, 0.95)), xytext=(0, 4),
                    textcoords="offset points", fontsize=7, ha="center")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("kick radius r")
ax.set_ylabel("$r_c(r)\\,/\\,r_c(0.01)$   (all dir-step pairs)")
ax.set_title("Linearity of the critical radius (w100)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3, which="both")

# ---------- Panel C: median gain vs r ----------
ax = axes[2]
sel = [ir for ir, r in enumerate(radii) if r >= 0.01]
gm = []
for ir in sel:
    g = c100.gain[ir]
    ok = np.isfinite(g) & (g > 0)
    gm.append(np.median(g[ok]))
rs_sel = np.array(radii)[sel]
ax.loglog(rs_sel, gm, "o-", color="tab:purple", lw=1.5)
ax.loglog(rs_sel, np.full(len(rs_sel), gm[0]), "--", color="0.4",
          lw=1.0, label="linear extrapolation from r=0.01")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("kick radius r")
ax.set_ylabel("median gain  $|\\delta m|/r$")
ax.set_title("Gain grows superlinearly for r>0.3", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.annotate(f"r=2: median gain {gm[-1]:.3f}\n(= {gm[-1]/gm[0]:.2f}x the r=0.01 gain)",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8)

fig.suptitle("Small-r thresholds do not extrapolate: superlinear gain makes r_c(r) "
             "drift out of [0.5,2]x for >30% of pairs at r>=0.5", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT + r"\fig6_large_r_linearity.png", dpi=160)
print("saved fig6")
print("median gain by r:", dict(zip([f"{r:g}" for r in rs_sel], np.round(gm, 4))))
