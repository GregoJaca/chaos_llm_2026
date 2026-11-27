"""The 'drop every ~100 tokens' pattern: it is the sliding-window law, model by model.

The A1 runs were made with W=100.  At generated token t the perturbation can only
be carried by positions that are both *perturbed* and *inside the window*.  The
last perturbed prompt position is p_last, so layer l loses the perturbation at

    t*_l = W - P + p_last + (l-1)(W-1)

(W-P+p_last for l=1, then one window per extra layer).  For W=100 that is a
staircase with steps ~99 tokens apart, first layer first - exactly the pattern
visible in DeepSeek-1.5B and Phi-4-mini.  This script draws the predicted
staircase on top of the measured response for every small model, plus a
W=4096 control where the staircase must disappear.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

D_HID = {"deepseek": 1536, "qwen25_3b": 2048, "qwen3_4b": 2560, "qwen3_5": 2048,
         "phi4mini": 3072}
NAMES = {"deepseek": "DeepSeek-1.5B", "qwen25_3b": "Qwen2.5-3B", "qwen3_4b": "Qwen3-4B",
         "qwen3_5": "Qwen3.5-2B", "phi4mini": "Phi-4-mini"}


def run(tag):
    p = os.path.join(HERE, "results/rstar", tag, "p1")
    z = np.load(os.path.join(p, "a1.npz"), allow_pickle=True)
    m = json.load(open(os.path.join(p, "meta.json")))
    P = int(m["prompt_len"])
    W = int(m["window"])
    pl = int(np.max(z["subspace"]) // D_HID[tag])
    dh = z["dh_norm"].astype(float)      # (r, dir, layer, T)
    lay = z["layers"].astype(int)
    return dh, lay, P, W, pl


fig, axes = plt.subplots(2, 5, figsize=(18.4, 7.2))
for ax, tag in zip(axes.flat, D_HID):
    dh, lay, P, W, pl = run(tag)
    a = np.median(dh[0], axis=0)          # (layer, T), r=0.1
    Z = np.log10(np.maximum(a / max(np.median(a[:, 5:20]), 1e-30), 1e-7))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="magma",
                   extent=(0, a.shape[1], lay[0], lay[-1]),
                   vmin=-4, vmax=0.5)
    tt = np.arange(0, a.shape[1])
    for l in lay[1:4]:
        tstar = W - P + pl + (l - 1) * (W - 1)
        if 0 <= tstar < a.shape[1]:
            ax.axvline(tstar, color="#2ca02c", lw=1.2, ls="--")
            ax.text(tstar + 2, lay[-1] - 4, f"$t^*_{{{l}}}$", color="#2ca02c", fontsize=7)
    ax.set_title(f"{NAMES[tag]}  ($W$={W}, $P$={P}, $p_{{\\rm last}}$={pl})", fontsize=8.6)
    ax.set_xlabel("token $t$", fontsize=7.4)
    ax.set_ylabel("layer $\\ell$", fontsize=7.4)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(\|\delta h\|/\|\delta h\|_{\rm med})$", fontsize=6.6)

# control: no window cut (DeepSeek, 4000 tokens, W=4096)
ax = axes.flat[5]
z = np.load(os.path.join(HERE, "results/local_data/a1_4k_test.npz"), allow_pickle=True)
a = np.median(z["dh_norm"][0], axis=0)
lay = z["layers"].astype(int)
Z = np.log10(np.maximum(a / max(np.median(a[:, 5:20]), 1e-30), 1e-7))
im = ax.imshow(Z, aspect="auto", origin="lower", cmap="magma",
               extent=(0, a.shape[1], 0, len(lay) - 1), vmin=-4, vmax=0.5)
ax.set_yticks(range(len(lay))); ax.set_yticklabels(lay, fontsize=7)
ax.set_title("DeepSeek-1.5B control: $W$=4096", fontsize=8.6)
ax.set_xlabel("token $t$", fontsize=7.4); ax.set_ylabel("layer $\\ell$", fontsize=7.4)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r"$\log_{10}(\|\delta h\|/\|\delta h\|_{\rm med})$", fontsize=6.6)
for j in range(6, 10):
    axes.flat[j].axis("off")
fig.suptitle("The 100-token staircase is the window law $t^*_\\ell=W-P+p_{\\rm last}+(\\ell-1)(W-1)$ (green), not a new phenomenon", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_window_staircase.png")
print("done")
