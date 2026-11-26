import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MOD = ["Qwen3-4B", "Phi-4-mini", "randQwen"]
D = {}
for n in MOD:
    perT = {}
    for rr in softloop_runs(n, "loop"):
        z = rr["data"]
        if "sv_read" not in z:
            continue
        lam = np.clip(z["sv_read"].astype(float), 0, None)   # (16, steps)
        ss = lam.sum(0)
        er = (ss ** 2) / np.maximum((lam ** 2).sum(0), 1e-30)
        top = lam[0] / np.maximum(ss, 1e-30)
        perT[rr["T"]] = dict(er=er, top=top, s=z["sig_read"].astype(float), T=rr["T"])
    D[n] = perT
    if perT:
        e = np.concatenate([v["er"] for v in perT.values()])
        print(n, "median eff rank", np.median(e), "p90", np.percentile(e, 90))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
ax = axes[0]
for n in MOD:
    if not D[n]:
        continue
    e = np.concatenate([v["er"] for v in D[n].values()])
    x = np.sort(e)
    ax.plot(x, 1 - np.arange(len(x)) / len(x), color=COL[n], lw=1.8,
            label=f"{n} (median {np.median(e):.2f})")
ax.set_xscale("log")
ax.set_xlabel(r"effective rank of $\Sigma_p$ over its top-16 singular values, $(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$")
ax.set_ylabel(r"$P(\mathrm{rank}>x)$")
ax.set_title("(a) Trained models: one readout direction; random weights: many", loc="left")
ax.legend(fontsize=7.6)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n in MOD:
    if not D[n]:
        continue
    Ts = sorted(D[n].keys())
    med = [np.median(D[n][T]["er"]) for T in Ts]
    ax.plot(Ts, med, "o-", color=COL[n], ms=4, label=n)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel("median effective rank")
ax.set_title("(b) Rank is set by concentration, not by temperature", loc="left")
ax.legend(fontsize=7.6)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Is the readout covariance rank one? Measured on its top-16 singular values", fontsize=11.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig_readout_rank.png")
