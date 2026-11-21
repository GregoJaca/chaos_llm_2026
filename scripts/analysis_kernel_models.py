"""The transport kernel across models: background decay and front amplitude."""
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
B = os.path.join(ROOT, "results", "transport")
OUT = os.path.join(ROOT, "notebook", "figures", "transport")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})
COL = {"deepseek": "tab:blue", "qwen3_4b": "tab:orange", "qwen25_3b": "tab:red",
       "phi4mini": "tab:green", "qwen3_5": "tab:purple"}
NAME = {"deepseek": "DeepSeek-1.5B", "qwen3_4b": "Qwen3-4B", "qwen25_3b": "Qwen2.5-3B",
        "phi4mini": "Phi-4-mini", "qwen3_5": "Qwen3.5-2B"}

rows = {}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
for tag, c in COL.items():
    files = sorted(glob.glob(os.path.join(B, f"{tag}_p*", "transport.npz")))
    if not files:
        continue
    Bs, As, taus = [], [], []
    for f in files:
        z = np.load(f)
        dh = z["dh"]                      # [pos, dir, layer, T]
        r = float(z["r"])
        K = np.median(dh[:, :, -1, :], axis=1) / r     # [pos, T]
        pos = z["positions"]
        bg = np.median(K[pos < 5], axis=0)            # background
        Bs.append(bg)
        amp = K.max(axis=1)
        pg = np.linspace(0, 1, 25)
        As.append(np.interp(pg, pos / pos[-1], amp))
    Bm = np.median(np.array(Bs), axis=0)
    Am = np.median(np.array(As), axis=0)
    pos = np.linspace(0, 1, 25)
    T = len(Bm)
    axs[0].semilogy(np.arange(T), np.maximum(Bm, 1e-4), lw=1.3, color=c, label=NAME[tag])
    axs[1].plot(pos, Am, "o-", ms=3, lw=1.2, color=c, label=NAME[tag])
    rows[tag] = dict(bg_start=float(Bm[0]), bg_end=float(Bm[-1]),
                     bg_ratio=float(Bm[-1] / max(Bm[0], 1e-9)),
                     amp_min=float(Am.min()), amp_max=float(Am.max()),
                     amp_argmax=float(pos[int(np.argmax(Am))]))
axs[0].set_xlabel("generated token $t$")
axs[0].set_ylabel(r"background $K(p\leq5,t)$")
axs[0].set_title("(a) Persistent background from the first prompt tokens")
axs[0].legend(fontsize=7.5); axs[0].grid(alpha=0.25, which="both")
axs[1].set_xlabel("relative prompt position $p/P$")
axs[1].set_ylabel(r"front amplitude $\max_t K(p,t)$")
axs[1].set_title("(b) Peak response of each prompt position")
axs[1].legend(fontsize=7.5); axs[1].grid(alpha=0.25)

# (c) the front position: argmax_t K(p,t) vs p
ax = axs[2]
for tag, c in COL.items():
    files = sorted(glob.glob(os.path.join(B, f"{tag}_p*", "transport.npz")))
    if not files:
        continue
    z = np.load(files[0])
    K = np.median(z["dh"][:, :, -1, :], axis=1)
    pos = z["positions"]
    ax.plot(pos / pos[-1], np.argmax(K, axis=1), lw=1.2, color=c, label=NAME[tag])
ax.set_xlabel("relative prompt position $p/P$")
ax.set_ylabel(r"$t$ of the peak")
ax.set_title("(c) The peak moves with the position (slope ≈1)")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "kernel_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# fits of the background decay per model
res = {}
for tag in COL:
    files = sorted(glob.glob(os.path.join(B, f"{tag}_p*", "transport.npz")))
    if not files:
        continue
    Bs = []
    for f in files:
        z = np.load(f)
        r = float(z["r"])
        K = np.median(z["dh"][:, :, -1, :], axis=1) / r
        Bs.append(np.median(K[z["positions"] < 5], axis=0))
    y = np.median(np.array(Bs), axis=0)
    t = np.arange(len(y))
    ok = y > 0
    r = fit_curve(t[ok], y[ok], families=["power", "exponential", "stretched_decay", "exp+offset"],
                  log_mode="native")
    res[tag] = dict(best=r.best.name, params=dict(zip(r.best.pnames, r.best.params.tolist())),
                    r2=r.best.r2, second=r.fits[1].name,
                    dbic=float(r.fits[1].bic - r.best.bic))
    print(tag, res[tag])
json.dump(dict(models=rows, fits=res),
          open(os.path.join(ROOT, "results", "kernel_models.json"), "w"), indent=1)
print("saved kernel_models.png")
