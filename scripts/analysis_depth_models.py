"""lambda_1(layer) for every model: same analysis as depth_aggregate, plus fits."""
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

MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Qwen2.5-3B", "qwen25_3b", "tab:red"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:purple")]

data = {}
for name, tag, c in MODELS:
    f = os.path.join(ROOT, "results", "depth2", tag, "depth.npz")
    if not os.path.exists(f):
        print("missing", tag)
        continue
    z = np.load(f)
    data[name] = (z, c)
print("models:", list(data))

NUM = {}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
ax = axs[0]
params = {}
for name, (z, c) in data.items():
    lam, L = z["lam"], z["layers"]
    med = np.median(lam[:, :, 0], axis=0)
    n = len(L)
    rel = L / (n - 1)
    ax.semilogy(rel[1:], med[1:], "o-", ms=3, lw=1.2, color=c, label=name)
    r = fit_curve(L[1:], med[1:], families=["power+offset+shift", "power+offset", "stretched_decay",
                                            "hill", "loglogistic", "exponential", "power"])
    b = r.best
    p = dict(zip(b.pnames, b.params))
    params[name] = dict(model=b.name, formula=b.formula, params=p,
                        se=dict(zip(b.pnames, b.se.tolist())), r2=b.r2,
                        second=r.fits[1].name, dbic=float(r.fits[1].bic - b.bic))
    # fitted curve on the relative axis
    xx = np.linspace(L[1], L[-1], 300)
    ax.semilogy(xx / (n - 1), np.maximum(b.predict(xx), 1e-4), "--", lw=1.0, color=c, alpha=0.8)
ax.set_xlabel("relative depth  $\\ell/(L-1)$")
ax.set_ylabel(r"top depth exponent $\lambda_1(\ell)$")
ax.set_title("(a) Depth exponent, every model (dashed = BIC fit)")
ax.legend(); ax.grid(alpha=0.25, which="both")

ax = axs[1]
for name, (z, c) in data.items():
    lam, L = z["lam"], z["layers"]
    logsig = np.median(lam[:, :, 0], axis=0) * (L + 1)
    ax.plot(L / (len(L) - 1), logsig, "o-", ms=2.5, lw=1.1, color=c, label=name)
ax.set_xlabel("relative depth"); ax.set_ylabel(r"$\log\sigma_1(\ell)=\lambda_1(\ell)(\ell+1)$")
ax.set_title(r"(b) Cumulative log-singular-value")
ax.legend(); ax.grid(alpha=0.25)

ax = axs[2]
bs, bse = [], []
names = []
for name, d in params.items():
    if "b" in d["params"]:
        bs.append(d["params"]["b"]); bse.append(d["se"]["b"]); names.append(name)
ax.barh(np.arange(len(names)), bs, xerr=bse, color=[data[n][1] for n in names], alpha=0.85)
ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.axvline(-1, color="r", ls="--", label=r"$-1$ (pure $1/\ell$)")
ax.set_xlabel(r"fitted exponent $b$ in $c+a(\ell-x_0)^b$")
ax.set_title("(c) Exponent per model")
ax.legend(); ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_law_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# per-model fit panels
n = len(data)
fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7.2), squeeze=False)
for k, (name, (z, c)) in enumerate(data.items()):
    ax = axs[k // ((n + 1) // 2)][k % ((n + 1) // 2)]
    lam, L = z["lam"], z["layers"]
    med = np.median(lam[:, :, 0], axis=0)
    ax.semilogy(L[1:], med[1:], "o", ms=3, color="0.3")
    d = params[name]
    r = fit_curve(L[1:], med[1:], families=[d["model"]], log_mode="native")
    xx = np.linspace(L[1], L[-1], 300)
    ax.semilogy(xx, np.maximum(r.best.predict(xx), 1e-4), "-", lw=1.6, color=c)
    ax.set_title(f"{name}\n{d['model']}  $R^2$={d['r2']:.4f}", fontsize=9)
    ax.set_xlabel(r"layer $\ell$"); ax.set_ylabel(r"$\lambda_1(\ell)$")
    ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_law_models_panels.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# per-token exponent
fig, ax = plt.subplots(figsize=(8, 4.4))
for name, (z, c) in data.items():
    lam, L, toks = z["lam"], z["layers"], z["tokens"]
    bs = []
    for ti in range(lam.shape[0]):
        r = fit_curve(L[1:], lam[ti, 1:, 0], families=["power+offset+shift", "power+offset"],
                      log_mode="native")
        bs.append(dict(zip(r.best.pnames, r.best.params)).get("b", np.nan))
    ax.plot(toks, bs, "o-", ms=3, lw=1.0, color=c, label=name)
ax.axhline(-1, color="r", ls="--", lw=1)
ax.set_xlabel("generated token $t$"); ax.set_ylabel(r"$b$")
ax.set_title(r"Exponent of $\lambda_1\propto\ell^{b}$ at each token")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_exponent_tokens_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

json.dump(params, open(os.path.join(ROOT, "results", "depth_models_fits.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("model", "r2", "params")}
                  for k, v in params.items()}, indent=1))
