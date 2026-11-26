"""Summary of the closed-loop saturation: L_inf (total log amplification vs the
replay control) and tau (rise time), per model, from pack_loop_loglog_models.json."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

D = json.load(open(os.path.join(HERE, "analysis_update", "pack_loop_loglog_models.json")))
names = [n for n in all_names() + ["Mamba-130m", "Pythia-410m", "randQwen"] if n in D]
rows = [(n, D[n]) for n in names
        if "L_end" in D[n] and "growth_closed" in D[n]]
rows.sort(key=lambda r: r[1]["L_end"])

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
ax = axes[0]
y = np.arange(len(rows))
for i, (n, d) in enumerate(rows):
    ax.barh(i, d["L_end"], color=COL[n], alpha=0.85)
    ax.text(d["L_end"] + 0.1, i, f"\u00d7{np.exp(d['L_end']):,.0f}",
            va="center", fontsize=6.6)
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=7.6)
ax.set_xlabel(r"$L_{\rm end}=\log(a_{\rm closed}/a_{\rm replay})$ at the end of the run (nats)")
ax.set_title("(a) Total amplification the loop adds over the network alone", loc="left")
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
for n, d in rows:
    if "tau" not in d or "sat_r2" not in d:
        continue
    good = d["sat_r2"] > 0.6
    ax.scatter(d["tau"], d["L_end"], s=70, color=COL[n], marker="o" if good else "x",
               alpha=0.9 if good else 0.6, edgecolor="k", linewidth=0.6)
    ax.annotate(n, (d["tau"], d["L_end"]), fontsize=6.6, xytext=(4, 3),
                textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel(r"rise time $\tau$ (steps) of the saturating fit")
ax.set_ylabel(r"$L_{\rm end}$ (nats)")
ax.set_title(r"(b) Where the loop settles vs how fast ($\times$ = poor fit, $R^2<0.6$)", loc="left")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_saturation.png")
print("done", len(rows))
