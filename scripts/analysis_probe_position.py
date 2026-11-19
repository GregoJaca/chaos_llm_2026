"""Propagation of a single-position kick: amplification vs position, all models, plus fits."""
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
B = os.path.join(ROOT, "results", "local_data", "probe2")
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})

MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Qwen2.5-3B", "qwen25_3b", "tab:red"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:purple")]

amp = {}
for name, tag, c in MODELS:
    d = os.path.join(B, tag)
    if not os.path.isdir(d):
        continue
    pos, val = [], []
    for sub in sorted(os.listdir(d)):
        f = os.path.join(d, sub, "layer_probe.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dh = z["dh"]; p = int(z["pos"]); r = float(z["r"])
        # injection at block 0, last layer, max over the next 50 tokens
        v = dh[0][:, -1, p + 1:p + 51]
        pos.append(p); val.append(float(np.median(v)) / r)
    if pos:
        o = np.argsort(pos)
        amp[name] = (np.array(pos)[o], np.array(val)[o], c)
print({k: (v[0].tolist(), np.round(v[1], 2).tolist()) for k, v in amp.items()})

fig, axs = plt.subplots(1, 2, figsize=(13, 4.7))
ax = axs[0]
fits = {}
for name, (x, y, c) in amp.items():
    ax.semilogy(x, y, "o-", ms=4, lw=1.2, color=c, label=name)
ax.set_xlabel("kick position $s$ (generated token)")
ax.set_ylabel(r"$\max_{\tau\leq50}\|\delta h_{s+\tau,L}\|\,/\,r$")
ax.set_title("(a) Amplification of a one-token kick vs position")
ax.grid(alpha=0.25, which="both"); ax.legend()

ax = axs[1]
for name, (x, y, c) in amp.items():
    r = fit_curve(x, y, families=["power+offset", "power+offset+shift", "exp+offset",
                                  "stretched_decay", "stretched_rise", "saturating",
                                  "gaussian", "linear", "log"])
    b = r.best
    fits[name] = dict(best=b.name, params=dict(zip(b.pnames, b.params.tolist())),
                      r2=b.r2, second=r.fits[1].name, dbic=float(r.fits[1].bic - b.bic))
    ax.semilogy(x, y / np.median(y), "o-", ms=4, lw=1.2, color=c,
                label=f"{name}  (max/min = {y.max()/y.min():.1f})")
ax.axhline(1, color="k", ls=":", lw=0.9)
ax.set_xlabel("kick position $s$")
ax.set_ylabel(r"amplification / its median over $s$")
ax.set_title("(b) Flat or structured? (deviation from the model's own mean)")
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "probe_position_law.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

json.dump(fits, open(os.path.join(ROOT, "results", "probe_position_fits.json"), "w"), indent=1)
print(json.dumps({k: (v["best"], round(v["r2"], 4), v["params"]) for k, v in fits.items()}, indent=1))
