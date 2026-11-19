"""Flip fraction vs radius with binomial CIs and a 50% threshold per prompt."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open(r"C:\Users\hjaca\research\chaos_llm_2026\results\figures_big\chaos_rows.json"))
OUT = r"C:\Users\hjaca\research\chaos_llm_2026\notebook\figures"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


prompts = sorted(set(r["prompt"] for r in rows))
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for p in prompts:
    rr = sorted(set(r["r"] for r in rows if r["prompt"] == p))
    fr, lo, hi = [], [], []
    for r in rr:
        obs = [o for x in rows if x["prompt"] == p and x["r"] == r for o in x["obs"]]
        k = sum(1 for o in obs if o >= 0)
        n = len(obs)
        fr.append(k / n)
        a, b = wilson(k, n)
        lo.append(k / n - a); hi.append(b - k / n)
    ax.errorbar(rr, fr, yerr=[lo, hi], fmt="o-", ms=4, capsize=2, label=p)
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"perturbation radius $r$")
ax.set_ylabel(r"fraction of directions that change any token")
ax.set_title("Stable-chaos threshold curves (20 directions per point, 95% Wilson CI)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT + r"\chaos_flip_fraction.png", dpi=190)
print("saved")
for p in prompts:
    rr = sorted(set(r["r"] for r in rows if r["prompt"] == p))
    fr = []
    for r in rr:
        obs = [o for x in rows if x["prompt"] == p and x["r"] == r for o in x["obs"]]
        fr.append(sum(1 for o in obs if o >= 0) / len(obs))
    print(p, "max", max(fr), "first r with >50%:", rr[int(np.argmax(np.array(fr) > 0.5))] if max(fr) > 0.5 else None)
