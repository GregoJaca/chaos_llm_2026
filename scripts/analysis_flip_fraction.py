"""Flip probability vs radius with 512 directions (DeepSeek), Wilson CIs."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "hazard")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


rows = []
for r in [0.1, 0.3, 0.6, 1.0]:
    f = os.path.join(B, f"deepseek_r{r}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    obs = z["obs"].ravel()
    n = len(obs)
    k = int((obs >= 0).sum())
    lo, hi = wilson(k, n)
    flips = obs[obs >= 0]
    rows.append(dict(r=r, n=n, k=k, frac=k / n, lo=lo, hi=hi,
                     t_med=float(np.median(flips)) if len(flips) else None,
                     t_q25=float(np.percentile(flips, 25)) if len(flips) else None,
                     t_q75=float(np.percentile(flips, 75)) if len(flips) else None))
print(json.dumps(rows, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axs[0]
rs = [d["r"] for d in rows]
fr = [d["frac"] for d in rows]
ax.errorbar(rs, fr, yerr=[[d["frac"] - d["lo"] for d in rows],
                          [d["hi"] - d["frac"] for d in rows]],
            fmt="o-", color="tab:blue", capsize=4, label="512 directions")
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("radius $r$"); ax.set_ylabel("fraction of directions that flip")
ax.set_title("(a) Flip probability vs radius, n=512")
ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(alpha=0.25, which="both")

ax = axs[1]
for d in rows:
    z = np.load(os.path.join(B, f"deepseek_r{d['r']}.npz"), allow_pickle=True)
    obs = z["obs"].ravel()
    t = np.arange(0, 301)
    S = np.array([np.mean((obs < 0) | (obs >= tt)) for tt in t])
    ax.step(t, S, where="post", lw=1.3, label=f"$r={d['r']}$")
ax.set_xlabel("generated token $t$"); ax.set_ylabel("$S(t)$ = fraction not yet flipped")
ax.set_title("(b) Survival curves")
ax.legend(); ax.grid(alpha=0.25)
ax.set_ylim(-0.02, 1.02)

ax = axs[2]
for d in rows:
    z = np.load(os.path.join(B, f"deepseek_r{d['r']}.npz"), allow_pickle=True)
    obs = z["obs"].ravel()
    fl = obs[obs >= 0]
    if len(fl) < 5:
        continue
    ax.hist(fl, bins=np.arange(0, 310, 10), histtype="step", lw=1.3,
            label=f"$r={d['r']}$ (n={len(fl)})")
ax.set_xlabel("first flip token"); ax.set_ylabel("directions")
ax.set_title("(c) When the flips happen")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "flip_fraction_512.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
json.dump(rows, open(os.path.join(ROOT, "results", "flip_fraction_512.json"), "w"), indent=1)
