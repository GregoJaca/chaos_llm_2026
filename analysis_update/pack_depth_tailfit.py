import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

DEP = {
    "DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
    "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
    "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
    "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
    "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
    "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz",
}
rows = []
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for ax, (n, p) in zip(axes.flat, DEP.items()):
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    lam = z["lam"].astype(float); L = lam.shape[1]
    x = np.arange(L, dtype=float)
    cum = lam * (x + 1)[None, :, None]
    per_tok = np.median(cum, axis=2)          # (token, L)
    med = np.median(per_tok, axis=0)
    lo, hi = np.percentile(per_tok, [25, 75], axis=0)
    sel = x >= 1
    c = np.polyfit(x[sel], med[sel], 3)
    pred = np.polyval(c, x[sel])
    r2 = 1 - ((med[sel] - pred) ** 2).sum() / ((med[sel] - med[sel].mean()) ** 2).sum()
    rows.append(dict(model=n, c3=float(c[0]), c2=float(c[1]), c1=float(c[2]), c0=float(c[3]), r2=float(r2)))
    ax.plot(x, med, "k.", ms=3, label="median")
    ax.fill_between(x, lo, hi, color=COL[n], alpha=0.2, lw=0)
    ax.plot(x[sel], pred, "-", color=COL[n], lw=1.6, label=f"cubic, $R^2$={r2:.2f}")
    ax.set_xlabel("block $\\ell$", fontsize=7.6)
    ax.set_ylabel(r"$\mathrm{cum}_\ell$", fontsize=7.6)
    ax.set_title(n, fontsize=8.4)
    ax.legend(fontsize=6.0); ax.grid(alpha=0.25)
fig.suptitle("Depth gain after block 0: median, IQR over tokens, and a cubic fit (layers 1+)", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_depth_tailfit.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_depth_tailfit.json"), "w"), indent=1)
for r in rows:
    print(r)
