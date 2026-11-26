import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B", "Qwen3.5-9B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]
RANK, ENT = {}, {}
for n in MODELS:
    er, ent = [], []
    for T, z, m in soft_runs(n):
        if "sv_read" in z:
            lam = np.clip(z["sv_read"].astype(float), 0, None)
            ss = lam.sum(0)
            er.append((ss ** 2) / np.maximum((lam ** 2).sum(0), 1e-30))
        if "ent_ref" in z:
            ent.append(z["ent_ref"].astype(float))
    if er:
        RANK[n] = np.concatenate(er)
    if ent:
        ENT[n] = np.concatenate(ent)

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
for ax, D, lab, title in [(axes[0], RANK, "effective rank", "(a) Readout effective rank"),
                          (axes[1], ENT, "entropy H (nats)", "(b) Entropy of the logits (T=1)")]:
    labels = list(D)
    data = [D[n] for n in labels]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(COL.get(labels[i], "gray")); b.set_alpha(0.55)
    for i, n in enumerate(labels):
        v = D[n]
        idx = np.random.default_rng(0).choice(len(v), min(400, len(v)), replace=False)
        ax.scatter(np.full(len(idx), i + 1) + np.random.default_rng(1).normal(0, 0.05, len(idx)),
                   v[idx], s=2, color="k", alpha=0.25)
        med = np.median(v)
        ax.plot(i + 1, med, "o", color="white", mec=COL.get(n, "k"), mew=2, ms=5)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.2)
    ax.set_ylabel(lab)
    ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.grid(True, which="both", alpha=0.3, axis="y")
fig.suptitle("Readout rank and entropy: box = IQR, white dot = median, dots = individual steps", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_rank_entropy_box.png")
json.dump({n: dict(rank=float(np.median(RANK[n])) if n in RANK else None,
                   ent=float(np.median(ENT[n])) if n in ENT else None) for n in MODELS},
          open(os.path.join(HERE, "analysis_update", "pack_rank_entropy_box.json"), "w"), indent=1)
print("done")
