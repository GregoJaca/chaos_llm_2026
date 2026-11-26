import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MODELS = ["qwen3_4b", "phi4mini"]
LABEL = {"qwen3_4b": "Qwen3-4B", "phi4mini": "Phi-4-mini"}
WS = [60, 100, 200, 4096]

res = {}
for m in MODELS:
    res[m] = {}
    ref = np.load(os.path.join(HERE, f"results/local_data/window4d/{m}/w4096/a1.npz"), allow_pickle=True)
    a0 = ref["dh_norm"][0].max(axis=0)
    for W in WS:
        z = np.load(os.path.join(HERE, f"results/local_data/window4d/{m}/w{W}/a1.npz"), allow_pickle=True)
        a = z["dh_norm"][0].max(axis=0)          # (layer, token), max over directions
        P = len(z["prompt_ids"])
        first = []
        for l in range(min(6, a.shape[0])):
            nz = np.where(np.abs(a[l] - a0[l]) > 1e-6)[0]
            first.append(int(nz.min()) if len(nz) else -1)
        flush = []
        for l in range(a.shape[0]):
            nz = np.where(a[l] > 1e-7)[0]
            flush.append(int(nz.max() + 1) if len(nz) else 0)
        res[m][W] = dict(P=P, flush=flush, first=first, layers=z["layers"].astype(int))
        print(m, "W", W, "P", P, "flush[:6]", flush[:6], "first", first)

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))

# (a) first deviation
ax = axes[0, 0]
for m in MODELS:
    xs, ys = [], []
    for W in WS[:-1]:
        d = res[m][W]
        xs.append(W - d["P"]); ys.append(np.median([x for x in d["first"] if x >= 0]))
    ax.plot(xs, ys, "o-", color=COL[LABEL[m]], ms=7, label=LABEL[m])
lim = [-10, 150]
ax.plot(lim, lim, "k--", lw=1.0, label=r"$y=x=W-P$")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"window minus prompt length  $W-P$  (tokens)")
ax.set_ylabel(r"first step where the response differs from $W=4096$")
ax.set_title("(a) The mask starts to matter exactly at $t=W-P$", loc="left")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) flush time vs layer
ax = axes[0, 1]
for W, mk, col in zip([60, 100, 200], ["o", "s", "^"], ["#440154", "#21918c", "#fde725"]):
    d = res["qwen3_4b"][W]
    lay = np.arange(1, len(d["flush"]) + 1)
    y = np.array(d["flush"], float)
    y[y >= 300] = np.nan
    ax.plot(lay, y, mk, color=col, ms=6, label=f"W={W} measured")
    t1 = d["flush"][0]
    ax.plot(lay, t1 + (lay - 1) * (W - 1), "-", color=col, lw=1.0, alpha=0.8,
            label=f"W={W}: $t^*_1+(\\ell-1)(W-1)$")
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0, 320)
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel(r"first token with zero response  $t^*_\ell$")
ax.set_title(r"(b) Qwen3-4B: flush time grows by $W-1$ per layer (Phi-4-mini identical)", loc="left")
ax.legend(fontsize=6.6, ncol=2)
ax.grid(alpha=0.3)

# (c) heatmap layer x W
ax = axes[1, 0]
xs = [60, 100, 200]
Z = np.full((len(res["qwen3_4b"][60]["flush"]), len(xs)), np.nan)
for j, W in enumerate(xs):
    v = np.array(res["qwen3_4b"][W]["flush"], float)
    v[v >= 300] = np.nan
    Z[:, j] = v
im = ax.imshow(Z, aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(w) for w in xs])
ax.set_yticks(range(Z.shape[0])); ax.set_yticklabels(range(1, Z.shape[0] + 1))
ax.set_xlabel("window size $W$")
ax.set_ylabel("layer index $\\ell$")
ax.set_title("(c) Qwen3-4B: flush time (white = beyond 300 tokens)", loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label(r"$t^*_\ell$  (tokens)")

# (d) flush vs W for each layer
ax = axes[1, 1]
for l, mk, col in zip([1, 2, 3], ["o", "s", "^"], ["#440154", "#21918c", "#fde725"]):
    xs, ys = [], []
    for W in [60, 100, 200]:
        v = res["qwen3_4b"][W]["flush"][l - 1]
        if v < 300:
            xs.append(W); ys.append(v)
    ax.plot(xs, ys, mk + "-", color=col, ms=6, label=f"layer {l}")
ax.set_xlabel("window size $W$")
ax.set_ylabel(r"$t^*_\ell$")
ax.set_title(r"(d) For fixed layer, the flush time rises linearly with W", loc="left")
ax.legend(fontsize=7.4)
ax.grid(alpha=0.3)

fig.suptitle("Sliding window: two different times, and why the perturbation does not vanish at $W-P$", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_window_departure.png")

json.dump(res, open(os.path.join(HERE, "analysis_update", "window_numbers.json"), "w"),
          indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("saved window_numbers.json")

