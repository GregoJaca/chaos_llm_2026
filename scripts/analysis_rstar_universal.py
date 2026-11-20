"""Is the r* distribution universal? 4 models x 5 prompts x 512 directions."""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from chaos_llm.curvefit import fit_curve

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "rstar")
OUT = os.path.join(ROOT, "notebook", "figures", "fsle")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})
COL = {"deepseek": "tab:blue", "qwen3_4b": "tab:orange", "phi4mini": "tab:green",
       "qwen3_5": "tab:purple"}

rows = []
curves = {}
for tag in COL:
    for d in sorted(glob.glob(os.path.join(B, tag, "p*"))):
        f = os.path.join(d, "a1.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        rc = z["r_c"][0].astype(float)
        m = z["margins"].astype(float)
        # a margin below the resolvable gap is a quantisation tie, not a decision gate
        rc[:, m <= 0.02] = np.nan
        rs = np.sort(np.nanmin(rc, axis=1))
        rs = rs[np.isfinite(rs) & (rs > 0)]
        ccdf = 1 - np.arange(len(rs)) / len(rs)
        r = fit_curve(rs, ccdf, families=["loglogistic", "loglogistic0", "power",
                                          "stretched_decay", "exp+offset"],
                      log_mode="native")
        p = dict(zip(r.best.pnames, r.best.params))
        rows.append(dict(model=tag, prompt=os.path.basename(d), n=len(rs),
                         best=r.best.name, k=p.get("k", np.nan), b=p.get("b", np.nan),
                         r2=r.best.r2, median=float(np.median(rs))))
        curves[(tag, os.path.basename(d))] = (rs, ccdf, p.get("k", np.nan), p.get("b", np.nan))
print("fits:", len(rows))
for r in rows:
    print(f"{r['model']:10s} {r['prompt']:3s} n={r['n']:3d} {r['best']:12s} "
          f"k={r['k']:.2f} b={r['b']:.2f} R2={r['r2']:.4f} med={r['median']:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axs[0]
for (tag, pr), (rs, ccdf, k, b) in curves.items():
    if tag != "deepseek":
        continue
    ax.loglog(rs, ccdf, ".", ms=2, color=COL[tag], alpha=0.55)
xx = np.geomspace(0.05, 20, 300)
ax.set_xlabel(r"$r_*$"); ax.set_ylabel(r"$P(r_*>x)$")
ax.set_title("(a) Five prompts, DeepSeek, 512 directions each")
ax.grid(alpha=0.25, which="both")
ax.plot([], [], ".", color=COL["deepseek"], label="prompts p1–p5")
ax.legend()

ax = axs[1]
for (tag, pr), (rs, ccdf, k, b) in curves.items():
    ax.semilogx(rs / k, ccdf, ".", ms=2, color=COL[tag], alpha=0.45)
xx = np.geomspace(0.05, 20, 300)
ax.semilogx(xx, 1 / (1 + xx ** 1.67), "k--", lw=1.2, label=r"$1/(1+x^{1.67})$")
ax.set_xlabel(r"$r_*/k$"); ax.set_ylabel("CCDF")
ax.set_title("(b) Collapse after scaling by the fitted median $k$")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rstar_universal.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# parameters per prompt
fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.4))
tags = list(COL)
for ax, key, lab in [(axs[0], "k", r"median $k$"), (axs[1], "b", r"tail exponent $b$")]:
    for i, tag in enumerate(tags):
        vals = [r[key] for r in rows if r["model"] == tag]
        ax.scatter(np.full(len(vals), i), vals, s=40, color=COL[tag])
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=COL[tag], lw=2)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=15, fontsize=8)
    ax.set_ylabel(lab); ax.grid(alpha=0.25)
axs[0].set_title("(a) Median flip radius per prompt")
axs[1].set_title("(b) Tail exponent per prompt")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rstar_params.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

json.dump(rows, open(os.path.join(ROOT, "results", "rstar_universal.json"), "w"), indent=1)
print("saved")
