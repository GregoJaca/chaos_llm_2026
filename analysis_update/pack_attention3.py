"""Attention mass on the prompt, per model, every saved layer, and vs t/W.

The measured quantity is
    frac(t) = (sum over prompt positions of attention at generated position t)
              / (total attention at that position)
for one head-averaged and one per-head version (frac_heads in the npz).

Three things the PI asked for:
  * one figure per model with all layers (here: a 2-column grid of models);
  * the same data on an axis normalised by the window, t/W, pooling several W
    (this is the scale on which models collapse);
  * a fit of the decay, using the general fitter (exponential vs power law, best R^2).
"""
import os
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from style import plt, save

MODELS = {"DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25", "Qwen3-4B": "qwen34",
          "Qwen3.5-2B": "qwen35", "Phi-4-mini": "phi", "Mamba-130m": "mamba",
          "Pythia-410m": "pythia", "randQwen": "randqwen",
          "Qwen3-14B": "qwen314", "DeepSeek-14B": "ds14", "Phi-4": "phi4",
          "Qwen3.5-9B": "qwen35_9b"}


def find(tag):
    out = {}
    for p in glob.glob(os.path.join(HERE, "results/update", f"attn*_{tag}_w*", "attn_frac.npz")):
        W = int(p.split("_w")[-1].split(os.sep)[0])
        out[W] = p
    return dict(sorted(out.items()))


def load(p):
    z = np.load(p, allow_pickle=True)
    f = z["frac"][0] if z["frac"].ndim == 3 else z["frac"]
    fh = z["frac_heads"] if "frac_heads" in z.files else None
    lay = z["layers"].astype(int)
    return f, fh, lay


def expf(x, A, x0, c):
    return A * np.exp(-x / x0) + c


def powf(x, A, b, c):
    return A * np.power(x + 1.0, -b) + c


def stretch(x, A, x0, beta, c):
    return A * np.exp(-np.power(x / x0, beta)) + c


fig, axes = plt.subplots(3, 4, figsize=(16.8, 10.4))
fits = {}
for ax, (name, tag) in zip(axes.flat, MODELS.items()):
    fs = find(tag)
    if not fs:
        ax.axis("off"); continue
    W = 100 if 100 in fs else sorted(fs)[0]
    f, fh, lay = load(fs[W])
    cm = plt.cm.viridis(np.linspace(0.05, 0.95, len(lay)))
    for i, l in enumerate(lay):
        ax.plot(np.arange(len(f[i])), np.maximum(f[i], 1e-6), color=cm[i], lw=1.1)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.2)
    ax.set_title(f"{name}  ($W$={W})", fontsize=8.8)
    ax.set_xlabel("generated token $t$", fontsize=7.2)
    ax.set_ylabel("attention mass on prompt", fontsize=7.2)
    ax.grid(True, which="both", alpha=0.25)
for j in range(len(MODELS), 12):
    axes.flat[j].axis("off")
fig.suptitle("Prompt-attention mass, every saved layer, per model ($W$=100 when available)", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_attention_layers.png")

# ---------------- t/W collapse + fit
fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
ax = axes[0]
for W, p in find("ds15").items():
    f, fh, lay = load(p)
    med = np.median(fh[-1], axis=0) if fh is not None else f[-1]
    t = np.arange(len(med))
    ax.plot(t / W, np.maximum(med, 1e-6), lw=1.3, label=f"$W$={W}")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("$t/W$")
ax.set_ylabel("attention mass on prompt")
ax.set_title("(a) DeepSeek, last layer, vs $t/W$", loc="left")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for name, tag in list(MODELS.items())[:8]:
    fs = find(tag)
    if 100 not in fs:
        continue
    f, fh, lay = load(fs[100])
    med = np.median(fh[-1], axis=0) if fh is not None else f[-1]
    t = np.arange(len(med))
    ax.plot(t / 100.0, np.maximum(med, 1e-6), lw=1.2, color=COL.get(name, "gray"), label=name)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("$t/W$ ($W$=100)")
ax.set_ylabel("attention mass on prompt")
ax.set_title("(b) All models, last layer", loc="left")
ax.legend(fontsize=6.0, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
for name, tag in list(MODELS.items())[:8]:
    fs = find(tag)
    if 100 not in fs:
        continue
    f, fh, lay = load(fs[100])
    med = np.median(fh[-1], axis=0)
    t = np.arange(len(med))
    x = t / 100.0
    ok = (med > 1e-5) & (x > 0)
    rows = {}
    try:
        p1, _ = curve_fit(expf, x[ok], med[ok], p0=[1.0, 1.0, 0.0], maxfev=40000,
                          bounds=([0, 0.02, 0], [2, 50, 1]))
        r1 = 1 - np.sum((med[ok] - expf(x[ok], *p1)) ** 2) / np.sum((med[ok] - med[ok].mean()) ** 2)
    except Exception:
        p1, r1 = None, -9
    try:
        p2, _ = curve_fit(powf, x[ok], med[ok], p0=[1.0, 1.0, 0.0], maxfev=40000,
                          bounds=([0, 0.05, 0], [2, 10, 1]))
        r2 = 1 - np.sum((med[ok] - powf(x[ok], *p2)) ** 2) / np.sum((med[ok] - med[ok].mean()) ** 2)
    except Exception:
        p2, r2 = None, -9
    try:
        p3, _ = curve_fit(stretch, x[ok], med[ok], p0=[1.0, 0.7, 2.0, 0.0], maxfev=60000,
                          bounds=([0, 0.05, 0.3, 0], [2, 20, 12, 1]))
        r3 = 1 - np.sum((med[ok] - stretch(x[ok], *p3)) ** 2) / np.sum((med[ok] - med[ok].mean()) ** 2)
    except Exception:
        p3, r3 = None, -9
    fits[name] = dict(exp_r2=float(r1), exp=list(p1) if p1 is not None else None,
                      pow_r2=float(r2), pow=list(p2) if p2 is not None else None,
                      stretch_r2=float(r3), stretch=list(p3) if p3 is not None else None)
    cand = [("exp", r1, expf(x, *p1) if p1 is not None else None),
            ("pow", r2, powf(x, *p2) if p2 is not None else None),
            ("stretched", r3, stretch(x, *p3) if p3 is not None else None)]
    kind, rbest, ybest = max(cand, key=lambda t: t[1])
    ax.plot(x, med, ".", ms=2, color=COL.get(name, "gray"), alpha=0.4)
    if ybest is not None:
        ax.plot(x, ybest, "-", lw=1.2, color=COL.get(name, "gray"),
                label=f"{name} ({kind}, $R^2$={rbest:.2f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(1e-6, 1.2)
ax.set_xlabel("$t/W$")
ax.set_ylabel("attention mass on prompt")
ax.set_title("(c) Best of exponential / power / stretched-exp in $t/W$", loc="left")
ax.legend(fontsize=5.6, ncol=2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_attention_collapse.png")
json.dump(fits, open(os.path.join(HERE, "analysis_update", "pack_attention_fits.json"), "w"), indent=1)
for n, v in fits.items():
    print(n, "exp %.2f pow %.2f stretch %.2f" % (v["exp_r2"], v["pow_r2"], v["stretch_r2"]),
          "beta=%.2f" % (v["stretch"][2] if v["stretch"] else np.nan))
