"""Layer-resolved loop amplification: where in the stack does the feedback act?

Uses the *layer numbers* recorded in meta.json, so runs that saved different layer
sets are never mixed.  Layer 0 (the token embedding) is excluded: in replay and
teacher forcing its perturbation is identically zero.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from looputil import TAGS, pairs, gather, meta_layers_of
from style import plt, save

MODELS = list(TAGS)


def layer_numbers(name, T="T1.0"):
    """union of saved layers (excluding 0) over all pairs, plus the pair count"""
    seen, keep = set(), []
    ps = pairs(name, T)
    for pl, pr in ps:
        lay = meta_layers_of(pl) or []
        for l in lay:
            if l > 0 and l not in seen:
                seen.add(l); keep.append(l)
    return sorted(keep), len(ps)


fig, axes = plt.subplots(3, 4, figsize=(16.6, 10.6))
rows = {}
for ax, name in zip(axes.flat, MODELS):
    lay, nrun = layer_numbers(name)
    if len(lay) < 2:
        ax.axis("off"); continue
    cm = plt.cm.viridis(np.linspace(0.05, 0.95, len(lay)))
    Lends = []
    for i, l in enumerate(lay):
        ac, ar, used = gather(name, layer_num=l)
        if ac is None or len(used) == 0:
            Lends.append(np.nan); continue
        t = np.arange(ac.shape[1])
        Ld = np.log(np.maximum(ac, 1e-30)) - np.log(np.maximum(ar, 1e-30))
        m = np.nanmedian(Ld, axis=0)
        ax.plot(t[1:], m[1:], color=cm[i], lw=1.3, label=f"layer {l}")
        k0 = int(0.8 * len(t))
        Lends.append(float(np.nanmedian(Ld[:, k0:])))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("step $t$", fontsize=7.2)
    ax.set_ylabel(r"$\log(a_{\rm closed}/a_{\rm replay})$", fontsize=7.2)
    ax.set_title(f"{name} ({nrun} pair{'s' if nrun > 1 else ''})", fontsize=8.8)
    ax.legend(fontsize=5.6); ax.grid(alpha=0.25)
    rows[name] = dict(layers=lay, L_end=Lends)
for j in range(len(MODELS), 12):
    axes.flat[j].axis("off")
fig.suptitle("Feedback gain by layer: $\\log(a_{\\rm closed}/a_{\\rm replay})$, $T=1$", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_layers.png")

fig, ax = plt.subplots(figsize=(9.4, 5.0))
for name, r in rows.items():
    lay = np.array(r["layers"], float)
    v = np.array(r["L_end"], float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        continue
    x = (lay - lay[0]) / max(lay[-1] - lay[0], 1)
    ax.plot(x[ok], v[ok], "o-", ms=4, lw=1.3, color=COL.get(name, "gray"), label=name)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("normalized depth")
ax.set_ylabel(r"$L_{\rm end}(\ell)$  (nats)")
ax.set_title("How much amplification each layer carries ($T=1$, end of run)", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_layers_Lend.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_loop_layers.json"), "w"), indent=1)
print(json.dumps({k: [round(v, 2) if np.isfinite(v) else None for v in r["L_end"]]
                  for k, r in rows.items()}, indent=0))

# ---------------------------------------------------------------- closed loop per layer (D3b analogue)
sel = ["DeepSeek-1.5B", "Qwen3-4B", "Phi-4-mini", "randQwen"]
fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.5))
for ax, name in zip(axes, sel):
    lay, nrun = layer_numbers(name)
    if len(lay) < 2:
        ax.axis("off"); continue
    picks = sorted(set([lay[0], lay[len(lay) // 2], lay[-1]]))
    cols = ["#08306b", "#ff7f0e", "#d62728"]
    for j, l in enumerate(picks):
        ac, ar, used = gather(name, layer_num=l)
        if ac is None or len(used) == 0:
            continue
        t = np.arange(ac.shape[1])
        mc = np.nanmedian(ac, axis=0); mr = np.nanmedian(ar, axis=0)
        ax.plot(t[1:], mc[1:], color=cols[j % 3], lw=1.3, label=f"closed, L{l}")
        ax.plot(t[1:], mr[1:], color=cols[j % 3], lw=0.9, ls="--", alpha=0.7)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(name, fontsize=8.8)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
    ax.plot([], [], color="k", ls="--", lw=1.0, label="replay (dashed, same colour)")
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Closed loop (solid) vs replay (dashed) at the first, middle and last saved layer, $T=1$, $W$=4096", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_loop_time_layers.png")
