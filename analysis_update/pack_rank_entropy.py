import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["Qwen3-4B", "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]
RANK, ENT = {}, {}
for n in MODELS:
    er, ent, live = [], [], []
    for T, z, m in soft_runs(n, "loop"):
        if "sv_read" in z:
            lam = np.clip(z["sv_read"].astype(float), 0, None)
            ss = lam.sum(0)
            r = (ss ** 2) / np.maximum((lam ** 2).sum(0), 1e-30)
            s = z["sig_read"].astype(float)
            er.append(r[live] if False else r)
            live.append(np.mean(s > T))
        if "ent_ref" in z:
            ent.append(z["ent_ref"].astype(float))
    RANK[n] = np.concatenate(er) if er else None
    ENT[n] = np.concatenate(ent) if ent else None
    print(n, "rank n", None if RANK[n] is None else len(RANK[n]), "ent n", None if ENT[n] is None else len(ENT[n]))

fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
ax = axes[0, 0]
for n in MODELS:
    if RANK[n] is None:
        continue
    x, F = ecdf(RANK[n])
    ax.plot(x, F, color=COL[n], lw=1.6, label=n)
ax.set_xscale("log")
ax.set_xlabel(r"effective rank $(\sum_i\lambda_i)^2/\sum_i\lambda_i^2$ of $\Sigma_p$")
ax.set_ylabel("cumulative fraction")
ax.set_title("(a) Readout rank: cumulative", loc="left")
ax.legend(fontsize=7.0); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
data = [RANK[n] for n in MODELS if RANK[n] is not None]
labels = [n for n in MODELS if RANK[n] is not None]
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.55)
for i, n in enumerate(labels):
    ax.plot(i + 1, np.median(data[i]), "o", color="white", mec=COL[n], mew=2, ms=6)
ax.set_yscale("log")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.6)
ax.set_ylabel("effective rank")
ax.set_title("(b) Readout rank: density", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="y")

ax = axes[1, 0]
for n in MODELS:
    if ENT[n] is None:
        continue
    x, F = ecdf(ENT[n])
    ax.plot(x, F, color=COL[n], lw=1.6, label=n)
ax.axvline(np.log(151936), color="k", ls=":", lw=1.0, label=r"$\log V=11.93$")
ax.set_xscale("log")
ax.set_xlabel(r"entropy $H$ of $\mathrm{softmax}(z)$ at $T=1$  (nats)")
ax.set_ylabel("cumulative fraction")
ax.set_title("(c) Entropy: cumulative", loc="left")
ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 1]
data = [ENT[n] for n in MODELS if ENT[n] is not None]
labels = [n for n in MODELS if ENT[n] is not None]
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.55)
for i, n in enumerate(labels):
    ax.plot(i + 1, np.median(data[i]), "o", color="white", mec=COL[n], mew=2, ms=6)
ax.set_yscale("log")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.6)
ax.set_ylabel("entropy $H$ (nats)")
ax.set_title("(d) Entropy: density", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_rank_entropy_all.png")

json.dump({n: dict(rank_median=float(np.median(RANK[n])) if RANK[n] is not None else None,
                   ent_median=float(np.median(ENT[n])) if ENT[n] is not None else None)
           for n in MODELS},
          open(os.path.join(HERE, "analysis_update", "pack_rank_entropy.json"), "w"), indent=1)
print("done")
