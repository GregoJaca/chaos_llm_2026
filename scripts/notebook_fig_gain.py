"""Clean version of the 'when the linear rule breaks' figure."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"

d = np.load(os.path.join(B, "w100_layers", "a1.npz"), allow_pickle=True)
radii = d["radii"]; dm = d["dm"]; m = d["margins"]

g = np.array([np.median(np.abs(dm[i]) / radii[i]) for i in range(len(radii))])
ref = g[int(np.argmin(np.abs(radii - 0.1)))]
sel = radii >= 0.3
beta, acoef = np.polyfit(np.log(radii[sel]), np.log(g[sel]), 1)

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})

# (a) pooled gain vs r
ax = axs[0]
ax.loglog(radii, g, "o-", color="tab:blue", label="median over 20 directions and 300 steps")
ax.axvspan(0.01, 0.3, color="tab:green", alpha=0.08)
rr = np.logspace(np.log10(0.3), np.log10(2.0), 20)
ax.loglog(rr, np.exp(acoef) * rr ** beta, "k--",
          label=rf"fit for $r\geq0.3$: $g\propto r^{{{beta:.2f}}}$")
ax.set_ylim(0.006, 0.020)
ax.text(0.03, 0.93, "flat: linear response", transform=ax.transAxes, fontsize=8, color="tab:green")
ax.text(0.58, 0.72, "power-law growth", transform=ax.transAxes, fontsize=8)
ax.text(0.05, 0.18, "float32 floor\n(noise$/r$)", transform=ax.transAxes, fontsize=8, color="0.35")
ax.set_xlabel(r"perturbation radius $r$")
ax.set_ylabel(r"$g(r)=\mathrm{median}\,|\delta m_t|/r$")
ax.set_title("(a) Gain is flat for $r\\in[0.01,0.3]$, then grows as $r^{0.19}$")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25, which="both")

# (b) individual near-ties
ax = axs[1]
for t, c in [(2, "tab:red"), (123, "tab:blue"), (16, "tab:green")]:
    v = np.abs(dm[:, :, t]) / radii[:, None]
    ax.loglog(radii, np.median(v, axis=1), "o-", color=c,
              label=f"$t={t}$, gap $m_t={m[t]:.3f}$")
ax.set_ylim(0.008, 0.06)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$|\delta m_t|/r$ at one close call")
ax.set_title("(b) The extra growth comes from a few close calls\n(most are flat)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "threshold_gain_scaling.png"), dpi=190)
print("saved; beta", round(float(beta), 3))
