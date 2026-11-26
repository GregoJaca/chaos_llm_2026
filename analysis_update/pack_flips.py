import os, sys, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

# small models: results/chaos_models/results/models/<m>/chaos/p*_r*/summary.json
SMALL = {"qwen3_4b": "Qwen3-4B", "qwen25_3b": "Qwen2.5-3B",
         "phi4mini": "Phi-4-mini", "qwen3_5": "Qwen3.5-2B"}
rows = {}
for m, name in SMALL.items():
    byr = {}
    for p in glob.glob(os.path.join(HERE, f"results/chaos_models/results/models/{m}/chaos/p*_r*/summary.json")):
        d = json.load(open(p))
        r = float(d["radii"][0])
        obs = np.array(d["obs"]).ravel()
        byr.setdefault(r, []).extend((obs >= 0).astype(int).tolist())
    rows[name] = {r: (np.sum(v), len(v), wilson(np.sum(v), len(v))) for r, v in byr.items()}

# DeepSeek: hazard + large runs
for p in glob.glob(os.path.join(HERE, "results/hazard/deepseek_r*/summary.json")):
    d = json.load(open(p)); r = float(d["radii"][0])
    obs = np.array(d["obs"]).ravel()
    rows.setdefault("DeepSeek-1.5B", {})[r] = (int(np.sum(obs >= 0)), len(obs),
                                               wilson(int(np.sum(obs >= 0)), len(obs)))

# big models from the fetched summaries
for m, name in [("qwen314", "Qwen3-14B"), ("phi4", "Phi-4"), ("ds14", "DeepSeek-14B")]:
    d = json.load(open(os.path.join(HERE, f"results/chaos_models/big_{m}.json")))
    for r, ff, n in zip(d["radii"], d["flip_frac"], [d["n_dirs"]] * len(d["radii"])):
        k = int(round(ff * n))
        rows.setdefault(name, {})[float(r)] = (k, n, wilson(k, n))

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax = axes[0]
for n, byr in rows.items():
    rs = sorted(byr)
    y = [byr[r][0] / byr[r][1] for r in rs]
    lo = [byr[r][2][0] for r in rs]; hi = [byr[r][2][1] for r in rs]
    ax.plot(rs, y, "o-", ms=4, color=COL[n], label=n)
    ax.fill_between(rs, lo, hi, color=COL[n], alpha=0.15, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"perturbation norm $r$")
ax.set_ylabel("fraction of directions that flip at least one token")
ax.set_title("(a) Free-run flip fraction (Wilson 95% band)", loc="left")
ax.legend(fontsize=6.8, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n, byr in rows.items():
    rs = sorted(byr)
    y = [byr[r][0] / byr[r][1] for r in rs]
    ax.plot(rs, y, "o-", ms=4, color=COL[n], label=n)
ax.set_xscale("log")
ax.set_xlabel(r"perturbation norm $r$")
ax.set_ylabel("flip fraction (linear scale)")
ax.set_title("(b) Same, linear y: DeepSeek rises; Phi saturates", loc="left")
ax.legend(fontsize=6.8, ncol=2); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "pack_flip_fraction_all.png")
json.dump({n: {str(r): (int(v[0]), int(v[1])) for r, v in byr.items()} for n, byr in rows.items()},
          open(os.path.join(HERE, "analysis_update", "pack_flips.json"), "w"), indent=1)
print("done")
