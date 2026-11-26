"""Layer-resolved closed loop vs replay (needs the long runs, which save 4 layers).

For each model: closed loop and replay at every saved layer, and the saturation
level L_end(ell) = log(a_closed/a_replay) against normalized depth.  The question
is where in the stack the feedback amplification is generated.
"""
import os
import sys
import glob
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAGS = {
    "DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b",
    "Qwen3.5-2B": "qwen3_5", "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130",
    "Pythia-410m": "pythia410", "randQwen": "randqwen", "Qwen3.5-9B": "qwen35_9b",
    "Qwen3-14B": "qwen314", "DeepSeek-14B": "ds14", "Phi-4": "phi4",
}


def find(tag, mode):
    for root in ("results/longloop",):
        p = os.path.join(HERE, root, tag, "T1.0", mode, "softloop.npz")
        if os.path.exists(p):
            return p
    return None


def layers_curves(path):
    z = np.load(path, allow_pickle=True)
    L = z["layers"].astype(float)
    a = z["dh"].astype(float)                 # (K, NL, T)
    a = a / np.maximum(a[:, :, [1]], 1e-30)
    med = np.median(a, axis=0)                # (NL, T)
    return L, med


def roll(x, w=9):
    if len(x) <= w:
        return x
    p = w // 2
    return np.convolve(np.pad(x, p, mode="edge"), np.ones(w) / w, mode="valid")


rows, summary = {}, {}
for name, tag in TAGS.items():
    pl, pr = find(tag, "loop"), find(tag, "replay")
    if not (pl and pr):
        continue
    Ll, ml = layers_curves(pl)
    Lr, mr = layers_curves(pr)
    if len(Ll) < 2:
        continue
    rows[name] = (Ll, ml, Lr, mr)
print("models with multi-layer long runs:", list(rows))
if not rows:
    print("no multi-layer longloop data yet")
    raise SystemExit(0)

fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for ax, (name, (Ll, ml, Lr, mr)) in zip(axes.flat, rows.items()):
    cm = plt.cm.viridis(np.linspace(0.05, 0.95, len(Ll)))
    for i, l in enumerate(Ll):
        ax.plot(roll(ml[i]), color=cm[i], lw=1.4)
        ax.plot(roll(mr[i]), color=cm[i], lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_title(name, fontsize=8.8)
    ax.set_xlabel("step $t$", fontsize=7.4)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.4)
    ax.grid(True, which="both", alpha=0.25)
    sm = ScalarMappable(norm=Normalize(Ll.min(), Ll.max()), cmap="viridis")
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("layer $\\ell$", fontsize=7)
for j in range(len(rows), 8):
    axes.flat[j].axis("off")
fig.suptitle("Closed loop (solid) vs replay (dashed) per saved layer, $T=1$", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_loop_layers.png")

# L_end vs normalized depth
fig, ax = plt.subplots(figsize=(9.6, 5.0))
for name, (Ll, ml, Lr, mr) in rows.items():
    T = min(ml.shape[-1], mr.shape[-1])
    k0 = int(0.8 * (T - 1))
    Lend = np.array([np.median(np.log(np.maximum(ml[i, k0:], 1e-30))
                              - np.log(np.maximum(mr[i, k0:], 1e-30))) for i in range(len(Ll))])
    x = (Ll - Ll[0]) / (Ll[-1] - Ll[0])
    ax.plot(x, Lend, "o-", color=COL.get(name, "gray"), lw=1.4, ms=5, label=name)
    summary[name] = dict(layers=Ll.tolist(), L_end=Lend.tolist())
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("normalized depth")
ax.set_ylabel(r"$L_{\rm end}(\ell)=\log(a_{\rm closed}/a_{\rm replay})$  (nats)")
ax.set_title("Where in the stack does the feedback amplification appear?", loc="left")
ax.legend(fontsize=6.8, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_layers_Lend.png")
json.dump(summary, open(os.path.join(HERE, "analysis_update", "pack_loop_layers.json"), "w"), indent=1)
print("done", len(rows))
