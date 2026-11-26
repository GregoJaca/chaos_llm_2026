import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

d = json.load(open(os.path.join(HERE, "analysis_update", "rstar_numbers.json")))
order = ["Phi-4-mini", "Qwen3-4B", "Qwen3.5-2B", "Qwen2.5-3B", "DeepSeek-1.5B"]

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
ax = axes[0]
bins = np.linspace(-2.2, 0.9, 45)
for n in order:
    r = np.array(d[n]["r"])
    r = r[np.isfinite(r) & (r > 0)]
    ax.hist(np.log10(r), bins=bins, histtype="step", lw=1.8, density=True,
            color=COL[n], label=f"{n} (median {np.median(r):.2f})")
ax.set_xlabel(r"$\log_{10} r_*$  (critical radius)")
ax.set_ylabel("probability density")
ax.set_title("(a) Distribution of the critical radius, 2,560 directions per model", loc="left")
ax.legend(fontsize=7.2)
ax.grid(alpha=0.3)

ax = axes[1]
data = [np.log10(np.array(d[n]["r"])[np.isfinite(d[n]["r"])]) for n in order]
parts = ax.violinplot(data, showextrema=False, widths=0.8)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(COL[order[i]]); pc.set_alpha(0.55)
for i, n in enumerate(order):
    med = np.median(data[i])
    ax.plot(i + 1, med, "o", color="white", mec=COL[n], mew=2, ms=7)
    ax.plot([i + 0.8, i + 1.2], [med, med], color=COL[n], lw=2)
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
ax.set_ylabel(r"$\log_{10} r_*$")
ax.set_title("(b) The distributions are broad and overlapping, not a single law", loc="left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Critical radius: the probability density, not only the cumulative curve", fontsize=11.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig_rstar_pdf.png")
