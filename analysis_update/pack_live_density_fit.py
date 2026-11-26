"""Live-gate density and the margin distribution.

P(m_t < T) is a monotone transform of the margin distribution, so the two
panels show the same data two ways.  The density is NOT a smooth function of T:
each temperature produces its own orbit, so the curve also encodes orbit-to-orbit
variation (and the loop's tendency to lock into confident repetition at some T).
A Hill fit is therefore not informative (the exponent pins at the bound); we
report the margin distribution instead, which is the robust cross-model number.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "Qwen3.5-9B", "randQwen"]


def runs(n):
    out = []
    for T, z, m in soft_runs(n):
        if T < 1e-3:
            continue
        out.append((T, z["m_ref"].astype(float)))
    return sorted(out, key=lambda r: r[0])


fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8))

# (a) density vs T
ax = axes[0]
for n in MODELS:
    rs = runs(n)
    if not rs:
        continue
    T = np.array([r[0] for r in rs])
    d = np.array([float(np.mean(r[1] < r[0])) for r in rs])
    ax.plot(T, d, "o-", ms=4, color=COL[n], label=n, lw=1.1,
            ls="--" if n == "randQwen" else "-")
ax.axhline(0.5, color="k", lw=0.6, ls="--")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"live-gate density $P(m_t<T)$")
ax.set_title("(a) Density vs $T$ (one orbit per $T$)", loc="left")
ax.set_xlim(0, 2.15); ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=6.2, ncol=2); ax.grid(alpha=0.3)

# (b) median margin vs T
ax = axes[1]
for n in MODELS:
    rs = runs(n)
    if not rs:
        continue
    T = np.array([r[0] for r in rs])
    med = np.array([np.median(r[1]) for r in rs])
    ax.plot(T, med, "o-", ms=4, color=COL[n], label=n, lw=1.1,
            ls="--" if n == "randQwen" else "-")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"median margin $\bar m$ (nats)")
ax.set_title(r"(b) The orbit's confidence vs $T$", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

# (c) pooled margin distribution per model
ax = axes[2]
names, data = [], []
for n in MODELS:
    rs = runs(n)
    if not rs:
        continue
    v = np.concatenate([r[1] for r in rs])
    names.append(n); data.append(v)
y = np.arange(len(names))
for i, (n, v) in enumerate(zip(names, data)):
    lo, med, hi = np.percentile(v, [25, 50, 75])
    p5, p95 = np.percentile(v, [5, 95])
    ax.plot([lo, hi], [i, i], color=COL[n], lw=6, alpha=0.45, solid_capstyle="butt")
    ax.plot([p5, p95], [i, i], color=COL[n], lw=1.2, alpha=0.7)
    ax.plot(med, i, "o", color=COL[n], ms=8, mec="k", mew=0.8)
ax.axvline(1.0, color="k", lw=0.8, ls=":")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.6)
ax.set_xscale("log")
ax.set_xlabel(r"margin $m_t$ (nats): bar = IQR, thin = 5--95%, dot = median")
ax.set_title("(c) Margin distribution, all temperatures pooled", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_live_density_fit.png")

rows = {n: dict(median_margin=float(np.median(v)),
                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                density_at_1=float(np.mean(v < 1.0)), n=int(len(v)))
        for n, v in zip(names, data)}
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_live_density_fit.json"), "w"), indent=1)
for n, v in rows.items():
    print(n, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
