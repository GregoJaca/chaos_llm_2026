import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

ORDER = all_names()

# ---------------------------------------------------------------- response vs time
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for ax, n in zip(axes.flat, ORDER):
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        a = z["med"][ri, -1, :]
    else:
        z = big(n); a = z["dh_med"][-1, :]
    a = a / max(a[1], 1e-30)
    ax.plot(np.arange(len(a)), a, color=COL[n])
    ax.set_yscale("log"); ax.set_xlim(0, 300)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("generated token $t$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_{t,L}\|/\|\delta h_{1,L}\|$", fontsize=7.6)
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Teacher-forced prompt perturbation: response vs token, every model", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_time_all_models.png")

# ---------------------------------------------------------------- controls replay overlay
names = ["DeepSeek-1.5B", "Qwen2.5-3B", "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]
fig, ax = plt.subplots(figsize=(10.8, 5.0))
for n in names:
    runs = [ (T, z, m) for T, z, m in soft_runs(n, "replay") if T < 1e-6 ]
    if not runs:
        continue
    T, z, m = runs[0]
    r = float(m["r"])
    a = np.median(z["dh"][:, -1, :].astype(float) / r, axis=0)
    a = a / max(a[1], 1e-30)
    ax.plot(np.arange(len(a)), a, color=COL[n], label=n)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_yscale("log"); ax.set_xlim(0, 300)
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$  (replay, last layer)")
ax.set_title("Network-only replay: LLMs vs the three controls", loc="left")
ax.legend(fontsize=7.4, ncol=2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_time_controls.png")

# ---------------------------------------------------------------- first/middle/last layer
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
for ax, n in zip(axes.flat, ORDER):
    if n in SMALL:
        z = a1(n); ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
        med = z["med"][ri]                  # (layer, T)
        lay = z["layers"]
    else:
        z = big(n); med = z["dh_med"]; lay = z["layers"]
    for frac, c in [(0.0, "#1f77b4"), (0.5, "#ff7f0e"), (1.0, "#d62728")]:
        li = int(round(frac * (med.shape[0] - 1)))
        a = med[li]; a = a / max(a[1], 1e-30)
        ax.plot(np.arange(len(a)), a, color=c, lw=1.2, label=f"layer {lay[li]}")
    ax.set_yscale("log"); ax.set_xlim(0, 300)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("$t$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_{t,\ell}\|/\|\delta h_{1,\ell}\|$", fontsize=7.6)
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Response vs time at the first, middle and last saved layer", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_time_by_layer.png")

# ---------------------------------------------------------------- transport kernel
ps = sorted(glob.glob(os.path.join(HERE, "results/transport/deepseek_p*/transport.npz")))
if not ps:
    ps = [os.path.join(HERE, "results/transport/deepseek/transport.npz")]
Ks = []
for p in ps:
    z = np.load(p, allow_pickle=True)
    K = np.median(z["dh"][:, :, -1, :], axis=1)   # (position, token)
    Ks.append(K)
from collections import Counter
shapes=[k.shape for k in Ks]
sh=Counter(shapes).most_common(1)[0][0]
Ks=[k for k in Ks if k.shape==sh]
K = np.median(np.stack(Ks), axis=0)
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax = axes[0]
im = ax.imshow(np.log10(K + 1e-12), aspect="auto", origin="lower", cmap="magma")
ax.set_xlabel("generated token $t$")
ax.set_ylabel("prompt position $p$")
ax.set_title(f"(a) Transport kernel $K(p,t)$, median over {len(Ks)} prompts", loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(r"$\log_{10}K(p,t)$")
ax = axes[1]
for p0 in [0, 5, 20, 40, len(K) - 1]:
    ax.plot(K[p0], label=f"p={p0}")
ax.set_yscale("log")
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$K(p,t)=\|\delta h_{t,L}\|/r$")
ax.set_title("(b) Rows of the kernel", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_transport_kernel.png")

# ---------------------------------------------------------------- superposition
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
pts = []
for root in ["deepseek_r0.003", "deepseek_r0.01", "deepseek_r0.03", "deepseek",
             "deepseek_p2", "deepseek_p3", "deepseek_r003",
             "qwen3_5_r003", "qwen25_3b_r003", "qwen3_4b_r003", "phi4mini_r003"]:
    p = os.path.join(HERE, "results/superposition", root, "superposition.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    if "err_L27" not in z.files:
        continue
    r = float(z["r"]); P = int(z["prompt_len"])
    e = np.nanmedian(np.abs(z["err_L27"]))
    c = np.nanmedian(z["cos_L27"])
    pts.append((r * np.sqrt(P), e, c, root))
pts.sort()
ax = axes[0]
ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#1f77b4")
ax.set_xscale("log")
ax.set_xlabel(r"total perturbation norm $r\sqrt{P}$")
ax.set_ylabel(r"$\|A-S\|/\|A\|$")
ax.set_title("(a) Superposition error vs total norm", loc="left")
for x, y, c, s in pts:
    ax.annotate(f"{s}\ncos={c:.3f}", (x, y), fontsize=5.8, xytext=(3, 4), textcoords="offset points")
ax.grid(True, which="both", alpha=0.3)
ax = axes[1]
ax.plot([p[0] for p in pts], [p[2] for p in pts], "o-", color="#d62728")
ax.set_xscale("log")
ax.set_xlabel(r"$r\sqrt{P}$")
ax.set_ylabel(r"$\cos(A,S)$")
ax.set_title("(b) The summed response points the same way", loc="left")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_superposition.png")
print("done")
