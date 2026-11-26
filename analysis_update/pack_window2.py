import os, sys, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

# ---------------------------------------------------------------- dense wspec
ps = sorted(glob.glob(os.path.join(HERE, "results/local_data/wspec/w*/a1.npz")),
            key=lambda p: int(re.search(r"w(\d+)", p).group(1)))
Ws = np.array([int(re.search(r"w(\d+)", p).group(1)) for p in ps])
ref = np.load(ps[-1], allow_pickle=True)
P = len(ref["prompt_ids"])
a0 = ref["dh_norm"][0].max(axis=0)      # (layer, token)
FD, FL = [], []
for p in ps:
    z = np.load(p, allow_pickle=True)
    a = z["dh_norm"][0].max(axis=0)
    fd, fl = [], []
    for l in range(a.shape[0]):
        d = np.abs(a[l] - a0[l])
        nz = np.where(d > 1e-6)[0]; fd.append(int(nz.min()) if len(nz) else -1)
        nz = np.where(a[l] > 1e-7)[0]; fl.append(int(nz.max() + 1) if len(nz) else 0)
    FD.append(fd); FL.append(fl)
FD = np.array(FD); FL = np.array(FL)
layers = ref["layers"].astype(int)

fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
ax = axes[0, 0]
res = []
for li in range(FD.shape[1]):
    ok = FD[:, li] >= 0
    res.append(FD[ok, li] - (Ws[ok] - P))
res = np.concatenate(res)
ax.hist(res, bins=np.arange(-8, 9) - 0.5, color="#1f77b4", alpha=0.85)
ax.set_xlabel(r"$t_{\rm first}-(W-P)$  (tokens)")
ax.set_ylabel("count")
ax.set_title(f"(a) First deviation: {len(Ws)} windows x 7 layers, residual", loc="left")
ax.grid(alpha=0.3, axis="y")

ax = axes[0, 1]
li = 0
ok = (FL[:, li] > 0) & (FL[:, li] < 199)
A = np.polyfit(Ws[ok], FL[ok, li], 1)
pred = np.polyval(A, Ws[ok])
r2 = 1 - ((FL[ok, li] - pred) ** 2).sum() / ((FL[ok, li] - FL[ok, li].mean()) ** 2).sum()
ax.plot(Ws[ok], FL[ok, li], ".", ms=5, color=COL["DeepSeek-1.5B"])
ax.plot(Ws[ok], pred, "-", color="k", lw=1.2,
        label=r"$t^*=%.3f\,W%+.1f$ ($R^2=%.4f$)" % (A[0], A[1], r2))
ax.set_xlabel("window size $W$")
ax.set_ylabel(r"flush step $t^*_1$")
ax.set_title("(b) Layer 1: 97 windows, slope exactly 1", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
for W in [60, 80, 100, 120, 150, 200]:
    i = int(np.argmin(np.abs(Ws - W)))
    fl = FL[i].astype(float)
    ok2 = (fl > 0) & (fl < 199)
    ax.plot(layers[ok2], fl[ok2], "o", ms=4, label=f"W={Ws[i]}")
    ax.plot(layers, layers * (Ws[i] - 1) - 21, "-", lw=0.9, alpha=0.7)
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel(r"$t^*_\ell$")
ax.set_title(r"(c) $t^*_\ell=\ell(W-1)-21$ for every layer", loc="left")
ax.set_ylim(0, 250)
ax.legend(fontsize=6.8); ax.grid(alpha=0.3)

ax = axes[1, 1]
for li, l in enumerate(layers):
    ok = (FL[:, li] > 0) & (FL[:, li] < 199)
    if ok.sum() < 3:
        continue
    A = np.polyfit(Ws[ok], FL[ok, li], 1)
    ax.plot(Ws[ok], FL[ok, li] - (l * (Ws[ok] - 1) - 21), ".", ms=4, label=f"layer {l}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("window size $W$")
ax.set_ylabel("residual from the line $\\ell(W-1)-21$")
ax.set_title("(d) Residuals are zero at the token level", loc="left")
ax.legend(fontsize=6.6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_window_dense.png")

# ---------------------------------------------------------------- window4d (Qwen/Phi)
MOD = ["qwen3_4b", "phi4mini"]
LAB = {"qwen3_4b": "Qwen3-4B", "phi4mini": "Phi-4-mini"}
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
ax = axes[0]
for m in MOD:
    z = np.load(os.path.join(HERE, f"results/local_data/window4d/{m}/w100/a1.npz"), allow_pickle=True)
    a = z["dh_norm"][0].max(axis=0)
    fl = []
    for l in range(a.shape[0]):
        nz = np.where(a[l] > 1e-7)[0]; fl.append(int(nz.max() + 1) if len(nz) else 300)
    ax.plot(z["layers"], fl, "o-", color=COL[LAB[m]], label=LAB[m])
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel(r"$t^*_\ell$")
ax.set_title("(a) Flush vs layer, W=100", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for m in MOD:
    for W in [60, 100, 200]:
        z = np.load(os.path.join(HERE, f"results/local_data/window4d/{m}/w{W}/a1.npz"), allow_pickle=True)
        a = z["dh_norm"][0].max(axis=0)
        fl = []
        for l in range(a.shape[0]):
            nz = np.where(a[l] > 1e-7)[0]; fl.append(int(nz.max() + 1) if len(nz) else 300)
        fl = np.array(fl, float); fl[fl >= 300] = np.nan
        ax.plot(z["layers"], fl, "o", ms=4, color=COL[LAB[m]], alpha=0.8,
                label=f"{LAB[m]} W={W}" if m == "qwen3_4b" else None)
ax.set_xlabel("layer index $\\ell$")
ax.set_ylabel(r"$t^*_\ell$")
ax.set_title("(b) Both models, three windows", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)

ax = axes[2]
for m in MOD:
    Z = []
    for W in [60, 100, 200]:
        z = np.load(os.path.join(HERE, f"results/local_data/window4d/{m}/w{W}/a1.npz"), allow_pickle=True)
        a = z["dh_norm"][0].max(axis=0)
        fl = []
        for l in range(a.shape[0]):
            nz = np.where(a[l] > 1e-7)[0]; fl.append(int(nz.max() + 1) if len(nz) else 300)
        Z.append(fl)
    Z = np.array(Z, float).T
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"flush step $t^*_\ell$ (tokens, capped at 300)", fontsize=7.4)
    break
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["60", "100", "200"])
ax.set_yticks(range(Z.shape[0])); ax.set_yticklabels(range(1, Z.shape[0] + 1), fontsize=7)
ax.set_xlabel("window $W$"); ax.set_ylabel("layer $\\ell$")
ax.set_title("(c) Qwen3-4B: $(\\ell,W)$ map", loc="left")
fig.tight_layout()
save(fig, "pack_window_models.png")

# ---------------------------------------------------------------- token agreement
z = np.load(os.path.join(HERE, "results/window_tokens/tokens.npz"), allow_pickle=True)
meta = json.load(open(os.path.join(HERE, "results/window_tokens/meta.json")))
P = int(meta["prompt_len"])
ref = z["w4096"]
Ws = [60, 100, 150, 200, 300, 500, 1000, 4096]
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
ax = axes[0, 0]
for W in Ws:
    if W == 4096:
        continue
    same = (z[f"w{W}"] == ref)
    ax.plot(np.arange(300), np.cumsum(same) / np.arange(1, 301), label=f"W={W}")
ax.set_xlabel("generated token $t$")
ax.set_ylabel("fraction identical to $W=4096$")
ax.set_title("(a) Running agreement", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3)

ax = axes[0, 1]
first = []
for W in Ws:
    if W == 4096:
        first.append(np.nan); continue
    d = np.where(z[f"w{W}"] != ref)[0]
    first.append(int(d[0]) if len(d) else np.nan)
ax.plot(Ws, first, "o", color=COL["DeepSeek-1.5B"], label="measured")
xx = np.array(Ws, float)
ax.plot(xx, xx - P, "k--", lw=1.0, label=r"$W-P$")
ax.set_xlabel("window $W$"); ax.set_ylabel("first differing token")
ax.set_title("(b) First difference at $W-P$", loc="left")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
res = np.array(first) - (np.array(Ws) - P)
ax.bar(np.arange(len(Ws)), res)
ax.set_xticks(np.arange(len(Ws))); ax.set_xticklabels([str(w) for w in Ws], rotation=25, fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("first token $- (W-P)$")
ax.set_title("(c) Residual latency (2-7 tokens)", loc="left")
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
frac = [np.mean(z[f"w{W}"] == ref) for W in Ws]
ax.plot(Ws, frac, "o-", color="#900C3F")
ax.set_xlabel("window $W$"); ax.set_ylabel("fraction of 300 tokens identical")
ax.set_title("(d) Text is identical once $W\\gtrsim500$", loc="left")
ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_window_tokens.png")

json.dump(dict(P=P, Ws=list(Ws), first=[None if np.isnan(x) else x for x in first]),
          open(os.path.join(HERE, "analysis_update", "pack_window.json"), "w"), indent=1)
print("done")

