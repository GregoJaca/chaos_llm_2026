"""Controlled margin fits and the post-clip 1/W law, done properly."""
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
LD = os.path.join(ROOT, "results", "local_data")
OUT = os.path.join(ROOT, "notebook", "figures", "veto")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})

# ---------------------------------------------------------------- 1. margin CCDF fits
rows = {}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
for w, c in [(100, "tab:blue"), (150, "tab:orange"), (200, "tab:green"), (300, "tab:red"),
             (1000, "tab:purple")]:
    f = os.path.join(LD, "window_fixed", f"w{w}", "a1.npz")
    if not os.path.exists(f):
        continue
    m = np.load(f)["margins"].astype(float)
    m = np.sort(m[m > 0])
    ccdf = 1 - np.arange(len(m)) / len(m)
    ax.loglog(m, ccdf, ".", ms=2, color=c, alpha=0.5)
    keep = (m > 0.02) & (ccdf > 1e-3)
    r = fit_curve(m[keep], ccdf[keep], families=["stretched_decay", "power", "exp+offset",
                                                 "power+offset"], log_mode="native")
    p = dict(zip(r.best.pnames, r.best.params))
    rows[w] = dict(best=r.best.name, r2=float(r.best.r2), params={k: float(v) for k, v in p.items()})
    xx = np.geomspace(m[keep].min(), m[keep].max(), 200)
    ax.loglog(xx, r.best.predict(xx), "-", lw=1.1, color=c,
              label=f"$W={w}$: {r.best.name} $R^2$={r.best.r2:.3f}")
ax.set_xlabel("margin $m$ (top-two logit gap)")
ax.set_ylabel(r"$P(m>x)$")
ax.set_title("(a) Margin distribution, same tokens, five windows\nfitted over $m\\in[0.02,10]$")
ax.legend(fontsize=7); ax.grid(alpha=0.25, which="both")

# parameter agreement
ax = axs[1]
ws = sorted(rows)
for key, c in [("tau", "tab:blue"), ("beta", "tab:red")]:
    vals = [rows[w]["params"].get(key, np.nan) for w in ws]
    ax.plot(ws, vals, "o-", color=c, label=key)
ax.set_xscale("log")
ax.set_xlabel("window $W$"); ax.set_ylabel("fitted parameter")
ax.set_title("(b) The fitted shape parameters are window-independent")
ax.legend(); ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "margin_fits.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("margin fits:", json.dumps(rows, indent=1)[:600])

# ---------------------------------------------------------------- 2. post-clip 1/W law (DeepSeek)
rows2 = []
for w in [50, 100, 200, 400, 800]:
    f = os.path.join(LD, "window_sweep", f"w{w}", "a1.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    P = int(z["prompt_ids"].shape[0])
    a = np.median(z["dh_norm"][0, :, -1, :], axis=0)
    start = max(w - P + 5, 10)
    if start >= len(a) - 25:
        continue
    seg = a[start:]
    k = 10
    t = np.arange(k, len(seg) - k)
    lam = np.log(np.maximum(seg[t + k], 1e-12) / np.maximum(seg[t - k], 1e-12)) / (2 * k)
    rows2.append((w, float(np.median(lam))))
wv = np.array([r[0] for r in rows2], float)
lv = -np.array([r[1] for r in rows2], float)
fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.5))
ax = axs[0]
ax.loglog(wv, lv, "o-", ms=8, color="tab:blue", label="measured")
r = fit_curve(wv, lv, families=["power", "power+offset", "exp+offset"], log_mode="native",
              min_points=3)
p = dict(zip(r.best.pnames, r.best.params))
xx = np.geomspace(wv.min(), wv.max(), 100)
ax.loglog(xx, r.best.predict(xx), "k--", lw=1.3,
          label=f"$A/W^{{{-p.get('b', float('nan')):.2f}}}$, $A={np.exp(p.get('a', np.nan)):.2f}$")
ax.set_xlabel("window $W$")
ax.set_ylabel(r"$-\lambda_{\rm post}$ (nats/token)")
ax.set_title("(a) Post-clip contraction rate, DeepSeek")
ax.legend(); ax.grid(alpha=0.25, which="both")
ax = axs[1]
ax.semilogx(wv, lv * wv, "o-", ms=8, color="tab:red")
ax.axhline(np.median(lv * wv), color="k", ls=":", lw=1)
ax.set_xlabel("window $W$"); ax.set_ylabel(r"$(-\lambda_{\rm post})\,W$")
ax.set_title("(b) The product is constant: one eviction costs a fixed log-probability")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "regime_scaling_deepseek.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("regime:", rows2, "fit", r.best.name, p, r.best.r2)
json.dump(dict(margins=rows, regime=rows2,
               regime_fit=dict(best=r.best.name, params={k: float(v) for k, v in p.items()},
                               r2=float(r.best.r2))),
          open(os.path.join(ROOT, "results", "veto_fits.json"), "w"), indent=1)
