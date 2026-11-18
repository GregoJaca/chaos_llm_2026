"""Fig 7: three individual directions (window 200) -- why each flips where it does.

For each direction: the signed local threshold r_c^-(d,t) (bold) vs the baseline
margin m_t (thin). A sustained kick of radius r flips the argmax at the first t
where r_c^- dips below r. Stars mark observed flips in free runs; white dots the
linear prediction. Cases: d10 (most fragile, flips at 0.1 and 0.3), d1 (typical,
tie-controlled), d13 (resistant: stabilizing at the ties, never flips <=0.3).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Case, OUT

c = Case("w200")
rc_neg = np.where(c.dm[c.ir_ref] < 0, c.rc_ref(), np.inf)
obs01 = np.array(c.free()["obs"][0])   # r=0.1
obs03 = np.array(c.free()["obs"][1])   # r=0.3

CASES = [(10, "most fragile: flips at both radii, earliest at r=0.3 (t=16)"),
         (1, "typical: controlled by the t=44 tie, flips at both radii"),
         (13, "resistant: stabilizing response at the ties, no flip up to r=0.3")]

fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True, sharey=True)
t = np.arange(c.T)
for ax, (d, blurb) in zip(axes, CASES):
    rcd = rc_neg[d]
    ax.semilogy(t, np.maximum(rcd, 1e-4), lw=1.2, color="tab:red",
                label=f"$r_c^-$(d{d},t)  local signed threshold")
    ax.semilogy(t, np.maximum(c.m, 1e-4), lw=0.6, color="0.55", label="$m_t$ margin")
    for r, col in [(0.1, "tab:blue"), (0.3, "tab:green")]:
        ax.axhline(r, color=col, lw=1.0, ls="--")
        ax.annotate(f"r={r}", xy=(597, r), fontsize=8, color=col, ha="right",
                    va="bottom", xytext=(0, 2), textcoords="offset points")
    tstar = int(np.argmin(rcd))
    ax.plot(tstar, max(rcd[tstar], 1e-4), "o", ms=9, mfc="white", mec="k", mew=1.2, zorder=6)
    ax.annotate(f"weakest tie t={tstar}: $r_*^-$={rcd[tstar]:.3g}", xy=(tstar, rcd[tstar]),
                xytext=(8, 10), textcoords="offset points", fontsize=8.5)
    for r, obs, col in [(0.1, obs01, "tab:blue"), (0.3, obs03, "tab:green")]:
        if obs[d] >= 0:
            ax.plot(obs[d], r, "*", ms=15, color=col, mec="k", mew=0.6, zorder=7)
            ax.annotate(f"observed flip @{r}", xy=(obs[d], r), xytext=(6, 6),
                        textcoords="offset points", fontsize=8.5, color=col)
    ax.set_ylim(1e-3, 60)
    ax.set_ylabel("radius / margin")
    ax.set_title(f"d{d}: {blurb}", fontsize=10, loc="left")
    ax.grid(alpha=0.3, which="both")
    if d == 10:
        ax.legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("generation step $t$")
axes[-1].set_xlim(0, 600 if c.T == 600 else 300)
fig.suptitle("Threshold tubes for three directions (window 200): a kick of radius r flips "
             "at the first t where $r_c^-$ falls below r", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT + r"\fig7_cases.png", dpi=160)
print("saved fig7")
for d, _ in CASES:
    tstar = int(np.argmin(rc_neg[d]))
    print(f"d{d}: r_*^- = {rc_neg[d, tstar]:.4g} @ t={tstar}; "
          f"obs@0.1={obs01[d]}, obs@0.3={obs03[d]}")
