import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
ax = axes[0]
for tag, W, c in [("ds15", 50, "#08306b"), ("ds15", 100, "#1f77b4"),
                  ("ds15", 200, "#ff7f0e"), ("ds15", 4096, "#d62728")]:
    p = os.path.join(HERE, f"results/update/attn4_{tag}_w{W}/attn_frac.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True); fh = z["frac_heads"][0]   # (heads,T)
    t = np.arange(fh.shape[1])
    med = np.median(fh, axis=0); lo, hi = np.percentile(fh, [25, 75], axis=0)
    ax.plot(t, np.maximum(med, 1e-6), color=c, lw=1.3, label=f"W={W}")
    ax.fill_between(t, np.maximum(lo, 1e-6), np.maximum(hi, 1e-6), color=c, alpha=0.12, lw=0)
ax.set_yscale("log")
ax.set_xlim(0, 200); ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("generated token $t$")
ax.set_ylabel("attention mass on prompt tokens (median over heads)")
ax.set_title("(a) DeepSeek, four windows; band = IQR over heads", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for tag, lab, c in [("ds15", "DeepSeek-1.5B", COL["DeepSeek-1.5B"]),
                    ("qwen25", "Qwen2.5-3B", COL["Qwen2.5-3B"]),
                    ("phi", "Phi-4-mini", COL["Phi-4-mini"]),
                    ("randqwen", "randQwen", COL["randQwen"])]:
    p = os.path.join(HERE, f"results/update/attn4_{tag}_w100/attn_frac.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True); f = z["frac"][0]
    ax.plot(np.arange(len(f)), np.maximum(f, 1e-6), color=c, lw=1.3, label=lab)
ax.set_yscale("log")
ax.set_xlim(0, 200); ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("generated token $t$")
ax.set_ylabel("attention mass on prompt tokens (mean over heads)")
ax.set_title("(b) W=100, four models", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_attention_models.png")

# fraction of heads with any prompt attention
fig, ax = plt.subplots(figsize=(9.8, 4.8))
for tag, W, c in [("ds15", 50, "#08306b"), ("ds15", 100, "#1f77b4"),
                  ("ds15", 200, "#ff7f0e"), ("ds15", 4096, "#d62728")]:
    p = os.path.join(HERE, f"results/update/attn4_{tag}_w{W}/attn_frac.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True); fh = z["frac_heads"][0]
    frac = np.mean(fh > 1e-4, axis=0)
    ax.plot(np.arange(len(frac)), frac, color=c, lw=1.3, label=f"W={W}")
ax.set_xlim(0, 200); ax.set_ylim(0, 1.02)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"fraction of heads with prompt attention $>10^{-4}$")
ax.set_title("Any prompt attention survives longer than the mean would suggest", loc="left")
ax.legend(fontsize=7.4); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_attention_heads.png")
print("done")

