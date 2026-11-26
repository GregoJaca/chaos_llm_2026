import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

ORDER = all_names()
DIM = {"DeepSeek-1.5B": 1536, "Qwen2.5-3B": 2048, "Qwen3-4B": 2560,
       "Qwen3.5-2B": 2048, "Phi-4-mini": 3072, "DeepSeek-14B": 5120,
       "Qwen3-14B": 5120, "Phi-4": 5120}
A = {}
for n in ORDER:
    a = angle(n)
    if a is None:
        continue
    A[n] = a

# ---------------------------------------------------------------- angle distributions
fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))
ax = axes[0]
for n in A:
    ang = A[n]["angle"]
    if ang.ndim == 4:
        v = ang[0, :, -1, :].ravel()   # r=0.03, last layer
    else:
        v = ang.ravel()
    x, F = ecdf(v)
    ax.plot(x, F, color=COL[n], lw=1.6, label=f"{n} (n={len(v)})")
ax.axvline(90, color="k", ls="--", lw=1.0)
ax.set_xlim(50, 130)
ax.set_xlabel(r"angle $\angle(\delta h_\ell,h_\ell)$  (degrees)")
ax.set_ylabel("cumulative fraction")
ax.set_title("(a) Response-vs-state angle, all models", loc="left")
ax.legend(fontsize=6.2); ax.grid(alpha=0.3)

ax = axes[1]
data, labels = [], []
for n in A:
    ang = A[n]["angle"]
    v = (ang[0, :, -1, :] if ang.ndim == 4 else ang).ravel()
    v = v[np.isfinite(v)]
    data.append(v); labels.append(n)
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.55)
for i, n in enumerate(labels):
    med = np.median(data[i])
    ax.plot(i + 1, med, "o", color="white", mec=COL[n], mew=2, ms=6)
ax.axhline(90, color="k", ls="--", lw=1.0)
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.4)
ax.set_ylabel(r"angle (degrees)")
ax.set_title("(b) Median is 90$^\\circ$; spread is model-dependent", loc="left")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
for n in A:
    ang = A[n]["angle"]
    if ang.ndim != 4:
        continue
    v = np.abs(ang[0, :, -1, :].ravel())
    x, F = ecdf(v)
    ax.plot(x, F, color=COL[n], lw=1.5, label=n)
ax.set_xlabel(r"radial fraction $|\cos\angle(\delta h_\ell,h_\ell)|$")
ax.set_ylabel("cumulative fraction")
ax.set_title("(c) The radial part is tiny; the tail is not", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_angles_all.png")

# ---------------------------------------------------------------- angle vs depth
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
for ax, n in zip(axes.flat, ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini"]):
    if n not in A:
        ax.axis("off"); continue
    a = A[n]; ang = a["angle"][0]          # (K, L, T)
    med = np.median(ang, axis=(0, 2))
    lo, hi = np.percentile(ang, [25, 75], axis=(0, 2))
    x = np.arange(len(med))
    ax.plot(x, med, "o-", ms=3, color=COL[n])
    ax.fill_between(x, lo, hi, color=COL[n], alpha=0.2, lw=0)
    ax.axhline(90, color="k", ls="--", lw=0.8)
    ax.set_xlabel("saved layer index  $\\ell$")
    ax.set_ylabel(r"angle $\angle(\delta h_\ell,h_\ell)$ (deg)")
    ax.set_title(n, fontsize=9)
    ax.grid(alpha=0.3)
fig.suptitle("Response angle vs depth (median, band = IQR over directions and tokens)", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_angle_vs_depth.png")

# save numbers
out = {}
for n, a in A.items():
    ang = a["angle"]
    v = (ang[0, :, -1, :] if ang.ndim == 4 else ang).ravel()
    out[n] = dict(n=int(len(v)), median=float(np.nanmedian(v)),
                  p25=float(np.nanpercentile(v, 25)),
                  p75=float(np.nanpercentile(v, 75)),
                  frac_gt90=float(np.nanmean(v > 90)))
json.dump(out, open(os.path.join(HERE, "analysis_update", "pack_angles.json"), "w"), indent=1)
print(out)
