"""Flip probability vs radius, 5 models, 8 prompts x 192 directions, tie-corrected."""
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})
COL = {"deepseek": "tab:blue", "qwen3_4b": "tab:orange", "qwen25_3b": "tab:red",
       "phi4mini": "tab:green", "qwen3_5": "tab:purple"}
NAME = {"deepseek": "DeepSeek-1.5B", "qwen3_4b": "Qwen3-4B", "qwen25_3b": "Qwen2.5-3B",
        "phi4mini": "Phi-4-mini", "qwen3_5": "Qwen3.5-2B"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0, (c - s) / d), min(1, (c + s) / d)


rows = json.load(open(os.path.join(ROOT, "results", "flip_table.json")))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for x in rows:
    by[x["model"]][x["r"]].append(x)

fig, axs = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=True)
for ax, key, ttl in [(axs[0], "flips", "(a) All flips"),
                     (axs[1], "nontrivial", "(b) Flips at a resolvable margin only")]:
    for tag in COL:
        rs = sorted(by[tag])
        if not rs:
            continue
        med, lo, hi, pl, ph = [], [], [], [], []
        for r in rs:
            cells = by[tag][r]
            k = sum(c[key] for c in cells); n = sum(c["n"] for c in cells)
            a, b = wilson(k, n)
            med.append(k / n); lo.append(a); hi.append(b)
            fr = [c[key] / c["n"] for c in cells]
            pl.append(np.percentile(fr, 25)); ph.append(np.percentile(fr, 75))
        ax.errorbar(rs, med, yerr=[np.array(med) - lo, np.array(hi) - med], fmt="o-",
                    ms=5, lw=1.3, capsize=3, color=COL[tag], label=NAME[tag])
        ax.fill_between(rs, pl, ph, color=COL[tag], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("perturbation radius $r$")
    ax.set_title(ttl + "\nline = pooled fraction (95% CI), band = 25–75% across prompts")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(-0.02, 1.02)
axs[0].set_ylabel("fraction of directions with at least one flip in 300 tokens")
axs[0].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flip_fraction_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

summary = {}
for tag in COL:
    summary[tag] = {}
    for r in sorted(by[tag]):
        cells = by[tag][r]
        k = sum(c["nontrivial"] for c in cells); n = sum(c["n"] for c in cells)
        k2 = sum(c["flips"] for c in cells)
        summary[tag][r] = dict(raw=k2 / n, corrected=k / n, n=n)
json.dump(summary, open(os.path.join(ROOT, "results", "flip_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
