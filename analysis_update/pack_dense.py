import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

PAIRS = {
    "DeepSeek-1.5B": ("results/local_data/w100_layers/a1.npz", "results/update/dense_ds15/a1.npz"),
    "Qwen2.5-3B": ("results/update/fp32_qwen25_a1.npz", "results/update/dense_qwen25/a1.npz"),
    "Qwen3.5-2B": ("results/update/fp32_qwen35_a1.npz", "results/update/dense_qwen35/a1.npz"),
}

def rr(path):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    r = z["radii"].astype(float)
    y = np.median(z["dh_norm"][:, :, -1, 50], axis=1)
    m = z["margins"].astype(float)
    rs = []
    for i in range(len(r)):
        rc = z["r_c"][i].copy(); rc[:, m <= 0.02] = np.inf
        v = np.min(rc, axis=1); rs.append(np.median(v[np.isfinite(v)]))
    return r, y, np.array(rs)

fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))
for ax, (n, (ps, pd)) in zip(axes, PAIRS.items()):
    rs, ys, rss = rr(ps)
    rd, yd, rsd = rr(pd)
    ax.plot(rs, ys, "o--", ms=4, color=COL[n], alpha=0.5, label="sparse simplex")
    ax.plot(rd, yd, "o-", ms=5, color=COL[n], label="dense Gaussian")
    ax.plot(rd, rd, "k:", lw=1.0, label=r"$y=r$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\|\delta h_{L}\|$")
    ax.set_title(n, fontsize=9)
    ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.3)
fig.suptitle("Sparse simplex directions saturate; dense Gaussian directions stay linear", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_sparse_vs_dense.png")

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
ax = axes[0]
for n, (ps, pd) in PAIRS.items():
    rs, ys, rss = rr(ps)
    rd, yd, rsd = rr(pd)
    ax.plot(rs, rss, "o--", ms=4, color=COL[n], alpha=0.5)
    ax.plot(rd, rsd, "o-", ms=5, color=COL[n], label=n + " dense")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"probe radius $r$")
ax.set_ylabel(r"median $r_*$")
ax.set_title("(a) Dense $r_*$ is radius-independent", loc="left")
ax.legend(fontsize=7.2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n, (ps, pd) in PAIRS.items():
    rd, yd, rsd = rr(pd)
    ax.plot(rd, rsd, "o-", ms=5, color=COL[n], label=n)
ax.set_xscale("log")
ax.set_xlabel(r"probe radius $r$")
ax.set_ylabel(r"median $r_*$")
ax.set_title("(b) Comparable cross-model fragility (dense)", loc="left")
ax.legend(fontsize=7.2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_rstar_dense.png")

json.dump({n: dict(r_sparse=rr(ps)[0].tolist(), rstar_sparse=rr(ps)[2].tolist(),
                   r_dense=rr(pd)[0].tolist(), rstar_dense=rr(pd)[2].tolist())
           for n, (ps, pd) in PAIRS.items()},
          open(os.path.join(HERE, "analysis_update", "pack_dense.json"), "w"), indent=1)
print("done")
