"""r* CCDF per model with fits (512 directions, 5 prompts each)."""
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
NAME = {"deepseek": "DeepSeek-1.5B", "qwen3_4b": "Qwen3-4B", "phi4mini": "Phi-4-mini",
        "qwen3_5": "Qwen3.5-2B"}

fig, axs = plt.subplots(1, 4, figsize=(20, 4.4), sharey=True)
out = {}
for ax, tag in zip(axs, COL):
    rows = []
    for d in sorted(glob.glob(os.path.join(B, tag, "p*"))):
        f = os.path.join(d, "a1.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        rc = z["r_c"][0].astype(float)
        m = z["margins"].astype(float)
        rc[:, m <= 0.02] = np.nan
        rs = np.sort(np.nanmin(rc, axis=1))
        rs = rs[np.isfinite(rs) & (rs > 0)]
        ccdf = 1 - np.arange(len(rs)) / len(rs)
        ax.loglog(rs, ccdf, ".", ms=2.5, color=COL[tag], alpha=0.5)
        r = fit_curve(rs, ccdf, families=["loglogistic", "loglogistic0", "stretched_decay",
                                          "power", "exp+offset"], log_mode="native")
        p = dict(zip(r.best.pnames, r.best.params))
        rows.append(dict(prompt=os.path.basename(d), median=float(np.median(rs)),
                         k=p.get("k", np.nan), b=p.get("b", np.nan), r2=r.best.r2,
                         best=r.best.name))
    if not rows:
        continue
    med = np.median([q["median"] for q in rows])
    ax.axvline(med, color="k", ls=":", lw=0.9)
    ax.set_title(f"{NAME[tag]}\nmedian $r_*$ = {med:.2f}, 5 prompts", fontsize=9.5)
    ax.set_xlabel(r"$r_*$")
    ax.grid(alpha=0.25, which="both")
    out[tag] = rows
axs[0].set_ylabel(r"$P(r_*>x)$  (CCDF)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rstar_per_model.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
json.dump(out, open(os.path.join(ROOT, "results", "rstar_per_model.json"), "w"), indent=1)
for t, rows in out.items():
    meds = [q["median"] for q in rows]
    bs = [q["b"] for q in rows if np.isfinite(q["b"])]
    print(t, "median range", round(min(meds), 3), "-", round(max(meds), 3),
          "| b range", (round(min(bs), 2), round(max(bs), 2)) if bs else None)
