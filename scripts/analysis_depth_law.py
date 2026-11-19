"""The depth law: what is lambda_1(layer), and is the per-layer gain constant?"""
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
OUT = os.path.join(ROOT, "notebook", "figures", "fits")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8.5})

d = np.load(os.path.join(ROOT, "results", "depth_w100", "depth.npz"))
lam, tokens, layers = d["lam"], d["tokens"], d["layers"]
med = np.median(lam[:, :, 0], axis=0)
res = fit_curve(layers, med)
b = res.best
b2 = res.fits[1]
NUM = {"best": b.name, "formula": b.formula,
       "params": dict(zip(b.pnames, b.params.tolist())),
       "se": dict(zip(b.pnames, b.se.tolist())), "r2": b.r2,
       "second": b2.name, "dbic": float(b2.bic - b.bic)}

# ---------------- 4 axis combinations
fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.4))
xx = np.linspace(layers.min(), layers.max(), 500)
for ax, (sx, sy, ttl) in zip(axs.ravel(), [("linear", "linear", "(a) linear-linear"),
                                           ("linear", "log", "(b) semilog-y (log response)"),
                                           ("log", "linear", "(c) semilog-x (log layer)"),
                                           ("log", "log", "(d) log-log")]):
    if sx == "log":
        ax.set_xscale("log")
    if sy == "log":
        ax.set_yscale("log")
    ax.plot(layers[1:], med[1:], "o", ms=4, color="0.25", label="data (median over 8 tokens)")
    x0 = 1 if sx == "log" else 0
    ax.plot(xx[x0:], b.predict(xx)[x0:], "-", lw=1.8, color="tab:red",
            label=f"{b.name}  $R^2$={b.r2:.5f}")
    ax.plot(xx[x0:], b2.predict(xx)[x0:], "--", lw=1.1, color="tab:blue",
            label=f"{b2.name}  $\\Delta$BIC={b2.bic-b.bic:.1f}")
    ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"$\lambda_1(\ell)$")
    ax.set_title(ttl); ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
fig.suptitle("DeepSeek depth exponent: the same data in four axis systems", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_law_4axes.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# ---------------- the kinematic decomposition: log sigma_1 = lambda_1 * (l+1)
logsig = med * (layers + 1)
fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
ax = axs[0]
ax.plot(layers, logsig, "o-", color="tab:green")
ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"$\log\sigma_1(\ell)$ (nats)")
ax.set_title(r"(a) Cumulative log-singular-value is flat: $\sigma_1\approx e^{3}$")
ax.grid(alpha=0.25)
r2 = fit_curve(layers[1:], logsig[1:], families=["linear"], log_mode="native")
lin = r2.best
ax.plot(layers, lin.predict(layers), "--", color="k",
        label=f"slope {lin.params[1]:+.4f} nats/layer")
ax.legend()
ax = axs[1]
slopes = []
for ti in range(lam.shape[0]):
    rr = fit_curve(layers[1:], (lam[ti, 1:, 0] * (layers[1:] + 1)), families=["linear"],
                   log_mode="native")
    slopes.append(rr.best.params[1])
ax.hist(slopes, bins=10, color="tab:green", alpha=0.8)
ax.axvline(0, color="k", lw=0.9)
ax.set_xlabel("slope of $\\log\\sigma_1$ per layer (nats)")
ax.set_ylabel("tokens")
ax.set_title("(b) Per-token slopes cluster at 0")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_logsigma.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
NUM["logsigma_slope"] = float(lin.params[1])
NUM["logsigma_slope_se"] = float(lin.se[1])
NUM["logsigma_mean"] = float(np.mean(logsig[1:]))

# ---------------- per-token exponent of a/(l+1)^b
bs = []
for ti in range(lam.shape[0]):
    rr = fit_curve(layers[1:], lam[ti, 1:, 0], families=["power+offset"], log_mode="native")
    v = dict(zip(rr.best.pnames, rr.best.params))
    bs.append(v.get("b", np.nan))
NUM["b_per_token"] = [float(x) for x in bs]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(tokens, bs, "o-", color="tab:purple")
ax.axhline(-1, color="r", ls="--", label=r"$-1$ (pure $1/\ell$)")
ax.set_xlabel("generated token $t$"); ax.set_ylabel("exponent $b$")
ax.set_title(r"$\lambda_1\propto \ell^{b}$: the exponent is $-1$ at every token")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_exponent_per_token.png"), dpi=185)
plt.close(fig)

json.dump(NUM, open(os.path.join(ROOT, "results", "depth_law_numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1)[:1200])
