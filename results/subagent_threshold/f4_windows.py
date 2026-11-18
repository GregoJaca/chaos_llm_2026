"""Fig 4: how the attention window reshapes the margin spectrum.

Small multiples: baseline margin traces m_t for windows 50..4096 (T=600, log y),
minima annotated. Summary row: min margin, 1%/5% quantiles, and near-tie counts
vs window. Also annotate where each trace departs from the full-context (w4096)
run -- the window only 'bites' after that step.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from common import OUT, load_margin_traces

traces, summary = load_margin_traces()
wins = [50, 100, 200, 500, 1000, 4096]
ref = traces[4096]

fig = plt.figure(figsize=(13, 8.5))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.25, 1.25, 1.0], hspace=0.55, wspace=0.3)

for i, w in enumerate(wins):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    m = traces[w]
    t = np.arange(len(m))
    ax.semilogy(t, np.maximum(m, 3e-4), lw=0.7, color="tab:blue")
    s = summary[str(w)]
    tmin = s["min_step"]
    ax.plot(tmin, max(m[tmin], 3e-4), "o", color="tab:red", ms=6, mec="k", mew=0.5)
    ax.annotate(f"min {m[tmin]:.4g} @ t={tmin}", xy=(tmin, max(m[tmin], 3e-4)),
                xytext=(8, -2), textcoords="offset points", fontsize=8, color="tab:red")
    if w != 4096:
        neq = np.where(m != ref)[0]
        div = int(neq[0]) if len(neq) else None
    else:
        div = None
    if div is not None:
        ax.axvline(div, color="tab:orange", lw=1.0, ls="--")
        ax.annotate(f"departs from full-context at t={div}", xy=(div, 20), fontsize=7.5,
                    color="tab:orange", xytext=(4, 2), textcoords="offset points")
    else:
        ax.annotate("identical to full context", xy=(0.03, 0.9), xycoords="axes fraction",
                    fontsize=7.5, color="tab:orange")
    ax.axhline(0.05, color="0.5", lw=0.6, ls=":")
    ax.set_ylim(3e-4, 60)
    ax.set_xlim(0, 600)
    ax.set_title(f"window {w}", fontsize=9.5)
    ax.grid(alpha=0.25, which="both")
    if i % 3 == 0:
        ax.set_ylabel("$m_t$")
    if i // 3 == 1:
        ax.set_xlabel("generation step $t$")

# summary panel: min margin and p01 / p05 vs window
ax = fig.add_subplot(gs[2, 0])
xw = np.array(wins, dtype=float)
mins = [summary[str(w)]["min_margin"] for w in wins]
p01 = [summary[str(w)]["p01"] for w in wins]
p05 = [summary[str(w)]["p05"] for w in wins]
ax.loglog(xw, mins, "o-", color="tab:red", label="min $m$")
ax.loglog(xw, p01, "s--", color="0.2", label="1% quantile")
ax.loglog(xw, p05, "^:", color="0.45", label="5% quantile")
ax.set_xscale("log")
ax.set_xlabel("attention window")
ax.set_ylabel("margin")
ax.set_title("margin floor vs window", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, which="both")

ax = fig.add_subplot(gs[2, 1])
n005 = [summary[str(w)]["n_below_0_05"] for w in wins]
n001 = [summary[str(w)]["n_below_0_01"] for w in wins]
med = [summary[str(w)]["median"] for w in wins]
ax.plot(xw, n005, "o-", color="tab:purple", label="count $m<0.05$")
ax.plot(xw, n001, "s--", color="tab:cyan", label="count $m<0.01$")
ax.set_xscale("log")
ax.set_xlabel("attention window")
ax.set_ylabel("near-tie count (of 600)")
ax.set_title("near-ties vs window", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, which="both")

ax = fig.add_subplot(gs[2, 2])
ax.loglog(xw, med, "D-", color="tab:green", label="median $m$")
ax.set_xscale("log")
ax.set_xlabel("attention window")
ax.set_ylabel("median margin")
ax.set_title("typical margin vs window", fontsize=9.5)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, which="both")

fig.suptitle("Attention window reshapes the margin spectrum: truncation departs late "
             "(t=36/136/436), but the margin floor collapses by ~50x from w50 to w200+",
             fontsize=11, y=0.99)
fig.savefig(OUT + r"\fig4_window_spectrum.png", dpi=160, bbox_inches="tight")
print("saved fig4")
for w in wins:
    s = summary[str(w)]
    print(f"w{w}: min {s['min_margin']:.4f}@{s['min_step']}  p01 {s['p01']:.4f}  "
          f"p05 {s['p05']:.4f}  n<0.05 {s['n_below_0_05']}")
