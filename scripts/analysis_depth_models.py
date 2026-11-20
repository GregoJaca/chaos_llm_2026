"""Depth gain sigma_1(l) for every model: where the gain happens."""
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
        continue
    data[name] = (np.load(f), c)
print("models:", list(data))

fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
late = {}
for name, (z, c) in data.items():
    lam, L = z["lam"], z["layers"]
    s1 = np.exp(np.median(lam[:, :, 0], axis=0) * (L + 1))
    axs[0].semilogy(L, s1, "o-", ms=3, lw=1.2, color=c, label=name)
    axs[1].plot(L / L[-1], s1 / s1[0], "o-", ms=3, lw=1.2, color=c, label=name)
    n4 = max(int(0.75 * len(L)), 1)
    late[name] = float(np.log(s1[-1] / s1[n4]))
axs[0].set_xlabel(r"block index $\ell$")
axs[0].set_ylabel(r"$\sigma_1(\ell)$  (stretch after $\ell$ blocks)")
axs[0].set_title("(a) The gain itself, all models"); axs[0].legend(); axs[0].grid(alpha=0.25, which="both")
axs[1].axhline(1, color="k", ls=":", lw=0.8)
axs[1].set_xlabel("relative depth  $\\ell/(L-1)$")
axs[1].set_ylabel(r"$\sigma_1(\ell)/\sigma_1(0)$")
axs[1].set_title("(b) Normalised: plateau, late rise, or both"); axs[1].legend(); axs[1].grid(alpha=0.25)

names = list(late)
axs[2].barh(np.arange(len(names)), [late[n] for n in names],
            color=[data[n][1] for n in names], alpha=0.85)
axs[2].set_yticks(np.arange(len(names))); axs[2].set_yticklabels(names, fontsize=9)
axs[2].set_xlabel(r"$\log[\sigma_1(L-1)/\sigma_1(\mathrm{at}\ 0.75L)]$")
axs[2].set_title("(c) How much of the gain is in the last quarter")
axs[2].grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_sigma_models.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

# per-model absolute sigma in blocks, with fits
fig, axs = plt.subplots(2, 3, figsize=(16, 7.4), squeeze=False)
for k, (name, (z, c)) in enumerate(data.items()):
    ax = axs[k // 3][k % 3]
    lam, L = z["lam"], z["layers"]
    s1 = np.exp(np.median(lam[:, :, 0], axis=0) * (L + 1))
    ax.semilogy(L, s1, "o", ms=3, color="0.3")
    r = fit_curve(L, s1, families=["linear", "quadratic", "power+offset", "exp+offset",
                                   "stretched_rise", "hill", "saturating"])
    xx = np.linspace(0, L[-1], 300)
    ax.semilogy(xx, np.maximum(r.best.predict(xx), 1e-3), "-", lw=1.5, color=c)
    ax.set_title(f"{name}: {r.best.name}  $R^2$={r.best.r2:.3f}", fontsize=9)
    ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\sigma_1$"); ax.grid(alpha=0.25, which="both")
for k in range(len(data), 6):
    axs[k // 3][k % 3].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "depth_sigma_models_panels.png"), dpi=185, bbox_inches="tight")
plt.close(fig)

json.dump({n: dict(late_quarter_log_gain=late[n],
                   total=float(np.exp(np.median(data[n][0]["lam"][:, -1, 0]) * len(data[n][0]["layers"]))))
           for n in late},
          open(os.path.join(ROOT, "results", "depth_sigma_models.json"), "w"), indent=1)
print(json.dumps(late, indent=1))
