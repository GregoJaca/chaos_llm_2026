"""Tie-corrected flip fractions: Mamba-130m vs the five LLMs (master.md section 8).

Reads the 8 prompt files from run_flipfrac.py, applies master's tie rule
(exclude flips at steps whose unperturbed margin <= 0.02), and plots the
fraction vs radius with Wilson 95% intervals, pooled over prompts, against
the LLM table from master.md.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.path.join(ROOT, "results", "ctrl_local", "mamba130", "flipfrac")
OUT = os.path.join(ROOT, "notebook", "figures_controls")
RADII = [0.1, 0.3, 0.6, 1.0]

master = {
    "DeepSeek-1.5B": [0.002, 0.034, 0.105, 0.237],
    "Qwen3.5-2B": [0.036, 0.201, 0.458, 0.719],
    "Qwen3-4B": [0.168, 0.199, 0.247, 0.383],
    "Qwen2.5-3B": [0.304, 0.337, 0.426, 0.530],
    "Phi-4-mini": [0.560, 0.656, 0.655, 0.666],
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


rows = []
files = sorted(glob.glob(os.path.join(FF, "flipfrac_*.npz")))
if not files:
    raise SystemExit("no flipfrac data yet")
for r in RADII:
    k_raw = n_raw = k = n = 0
    per_prompt = []
    for f in files:
        z = np.load(f)
        margins = z["margins"]
        flags = z["flags_%g" % r]
        steps = z["steps_%g" % r]
        n_raw += len(flags)
        k_raw += int(flags.sum())
        keep = []
        for fl, st in zip(flags, steps):
            if fl and margins[st] <= 0.02:
                continue  # tie-excluded flip
            keep.append(fl)
        k += int(sum(keep))
        n += len(keep)
        per_prompt.append(float(np.mean(keep)) if keep else 0.0)
    rows.append((r, k, n, per_prompt))
    print("r=%g: raw %d/%d (%.1f%%); tie-corrected %d/%d (%.1f%%); per-prompt %.3f-%.3f" % (
        r, k_raw, n_raw, 100 * k_raw / n_raw, k, n, 100 * k / n,
        min(per_prompt), max(per_prompt)))

fig, ax = plt.subplots(figsize=(9.5, 5.4))
for name, vals in master.items():
    ax.plot(RADII, vals, "o--", ms=4, lw=1.0, alpha=0.7,
            color={"DeepSeek-1.5B": "#D55E00", "Qwen3.5-2B": "#888", "Qwen3-4B": "#888",
                   "Qwen2.5-3B": "#888", "Phi-4-mini": "#009E73"}[name], label=name)
xs = [r for r, _, _, _ in rows]
ys = [k / n for _, k, n, _ in rows]
lo = [wilson(k, n)[1] for _, k, n, _ in rows]
hi = [wilson(k, n)[2] for _, k, n, _ in rows]
ax.plot(xs, ys, "o-", ms=7, lw=2.2, color="#0072B2", label="Mamba-130m (control)")
ax.fill_between(xs, lo, hi, color="#0072B2", alpha=0.2)
ax.set_xscale("log")
ax.set_xlabel(r"prompt perturbation radius $r$")
ax.set_ylabel("tie-corrected flip fraction (300 tokens, greedy)")
ax.set_title("Flip fraction vs radius, 8 prompts x 192 directions (master.md protocol):\nthe control spans the same range as the LLMs - thresholds, not chaos")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "c8_flipfrac.png"), dpi=190); plt.close(fig)

json.dump([dict(r=r, k=k, n=n, per_prompt=pp) for r, k, n, pp in rows],
          open(os.path.join(ROOT, "results", "ctrl_local", "flipfrac_summary.json"), "w"), indent=1)
print("wrote", OUT)
