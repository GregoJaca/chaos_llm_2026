"""Top-k singular subspaces: rotation between blocks, and perturbation alignment.

Two questions the PI asked:
  (1) is the "top singular vector rotation" an artefact of near-degenerate singular
      values?  -> compare the largest principal angle between consecutive layers'
      top-k subspaces for k=1,3,10 (k>1 is invariant to within-subspace mixing);
  (2) what matters is not whether the *state* is aligned with the top singular
      vector but whether the *perturbation* is.  -> report the fraction of the
      perturbation captured by the top-k subspace of the next block.

Data: results/update/topk_<tag>/topk.npz from chaos_llm.run_topk.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE, all_names
from style import plt, save

EXTRA = ["Mamba-130m", "Pythia-410m", "randQwen"]
ORDER = all_names() + EXTRA
TAG = {"DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25", "Qwen3-4B": "qwen34",
       "Qwen3.5-2B": "qwen35", "Phi-4-mini": "phi", "Mamba-130m": "mamba",
       "Pythia-410m": "pythia", "randQwen": "randqwen", "Qwen3-14B": "qwen314",
       "DeepSeek-14B": "ds14", "Phi-4": "phi4", "Qwen3.5-9B": "qwen35_9b"}

D = {}
for n in ORDER:
    p = os.path.join(HERE, "results/update", f"topk_{TAG.get(n, n)}", "topk.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    D[n] = dict(theta=z["theta"].astype(float), align=z["align"].astype(float),
                sig=z["sig"].astype(float), layers=z["layers"].astype(int))
print("models with top-k data:", list(D))
if not D:
    raise SystemExit("no topk data yet")

fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.9))
ax = axes[0]
for n, d in D.items():
    th = np.median(d["theta"], axis=0)          # (layer, 3)
    x = np.arange(th.shape[0]) / max(th.shape[0] - 1, 1)
    ax.plot(x, th[:, 0], "-", color=COL.get(n, "gray"), lw=1.2, label=f"{n}" if n in D else None)
ax.set_xlabel("normalized depth")
ax.set_ylabel(r"largest principal angle between $\mathrm{span}(U_\ell)$ and $\mathrm{span}(U_{\ell+1})$ (deg)")
ax.set_title("(a) Top-1 singular-vector rotation", loc="left")
ax.axhline(90, color="k", ls=":", lw=0.8)
ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
for j, (kk, ls) in enumerate(zip([1, 3, 10], ["-", "--", ":"])):
    vals = []
    for n, d in D.items():
        th = np.median(d["theta"], axis=0)
        if th.shape[1] <= j:
            continue
        vals.append(np.median(th[:, j]))
    if vals:
        ax.scatter([kk] * len(vals), vals, s=40)
        ax.plot([kk], [np.median(vals)], "k_", ms=18)
ax.set_xlabel("subspace dimension $k$")
ax.set_ylabel("median rotation angle (deg)")
ax.set_xticks([1, 3, 10])
ax.set_title("(b) Rotation shrinks with $k$ (degeneracy check)", loc="left")
ax.grid(alpha=0.3)

ax = axes[2]
for n, d in D.items():
    a = d["align"]
    kk1 = np.nanmedian(a[:, :, :, 0], axis=(0, 2))
    kk10 = np.nanmedian(a[:, :, :, 2], axis=(0, 2))
    x = np.arange(len(kk1)) / max(len(kk1) - 1, 1)
    ax.plot(x, kk1, "-", color=COL.get(n, "gray"), lw=1.1)
    ax.plot(x, kk10, ":", color=COL.get(n, "gray"), lw=1.1)
ax.set_xlabel("normalized depth")
ax.set_ylabel(r"fraction of $\delta h_\ell$ inside the top-$k$ subspace")
ax.set_title("(c) Perturbation alignment ($k$=1 solid, $k$=10 dotted)", loc="left")
ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_topk.png")

json.dump({n: dict(theta_med=np.median(d["theta"], axis=0).tolist(),
                   align_k1=np.nanmedian(d["align"][:, :, :, 0], axis=(0, 2)).tolist(),
                   align_k10=np.nanmedian(d["align"][:, :, :, 2], axis=(0, 2)).tolist(),
                   sig1=np.median(d["sig"][:, :, 0], axis=0).tolist())
           for n, d in D.items()},
          open(os.path.join(HERE, "analysis_update", "pack_topk.json"), "w"), indent=1)
print("done")
