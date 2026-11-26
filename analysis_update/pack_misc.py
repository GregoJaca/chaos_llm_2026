import os, sys, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

# ---------------------------------------------------------------- sampling: soft / CRN / ind
MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m", "Pythia-410m"]
fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6))
for ax, n in zip(axes.flat, MODELS):
    tag = {"DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b",
           "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130", "Pythia-410m": "pythia410"}[n]
    base = os.path.join(HERE, "results/softloop_sample", tag)
    # soft loop T=1
    soft = [z for T, z, m in soft_runs(n) if abs(T - 1.0) < 1e-6]
    if soft:
        z = soft[0]; a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
        ax.plot(np.arange(len(a)), a, color="#d62728", lw=1.6, label="soft")
    for mode, c, lab in [("crn", "#1f77b4", "CRN sample"), ("ind", "#2ca02c", "independent")]:
        p = os.path.join(base, f"{mode}_T1.0", "softloop.npz")
        if os.path.exists(p):
            z = dict(np.load(p, allow_pickle=True))
            a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
            ax.plot(np.arange(len(a)), a, color=c, lw=1.3, ls="--", label=lab)
    ax.set_yscale("log"); ax.set_xlim(0, 200)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("step $t$", fontsize=7.6)
    ax.set_ylabel(r"$\|\delta h_t\|$  (median)", fontsize=7.6)
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Soft feedback grows; sampled feedback with common random numbers stays flat", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_sampling.png")

# ---------------------------------------------------------------- gate geometry G_t
G = {}
for p in glob.glob(os.path.join(HERE, "results/softloop_gates/*/p*/gates.npz")) + \
         glob.glob(os.path.join(HERE, "results/softloop_gates/*/loop0/gates.npz")):
    parts = p.split(os.sep)
    tag = parts[-3] if parts[-2].startswith("p") else parts[-2]
    z = dict(np.load(p, allow_pickle=True))
    G.setdefault(tag, []).append(dict(G=z["G"].astype(float), m=z["m"].astype(float)))
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
ax = axes[0]
labels, data = [], []
for tag, arrs in G.items():
    v = np.concatenate([x["G"] for x in arrs])
    labels.append(tag); data.append(v)
bp = ax.violinplot(data, showextrema=False, widths=0.85)
for i, b in enumerate(bp["bodies"]):
    b.set_facecolor(plt.cm.tab10(i % 10)); b.set_alpha(0.55)
for i, v in enumerate(data):
    ax.plot(i + 1, np.median(v), "o", color="white", mec="k", mew=1.2, ms=5)
ax.set_yscale("log")
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7.2)
ax.set_ylabel(r"$G_t=\|E_i-E_j\|\,\|W_i-W_j\|$")
ax.set_title("(a) Pair geometry per model", loc="left")
ax.grid(True, which="both", alpha=0.3, axis="y")
ax = axes[1]
for tag, arrs in G.items():
    for z in arrs[:1]:
        ax.scatter(z["m"].astype(float), z["G"].astype(float), s=3, alpha=0.3, label=tag)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"margin $m_t$")
ax.set_ylabel(r"$G_t$")
ax.set_title("(b) $G_t$ is not a function of the margin", loc="left")
ax.legend(fontsize=5.8, ncol=2); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_gate_geometry.png")

# ---------------------------------------------------------------- survival
MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m", "randQwen"]
fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6))
for ax, n in zip(axes.flat, MODELS):
    runs = soft_runs(n)
    for T, z, meta in runs:
        if T < 1e-3:
            continue
        r = float(meta["r"])
        dh = z["dh"][:, -1, :].astype(float) / r
        hn = float(np.median(z["h0n"][-1].astype(float))); sat = 0.25 * hn
        ts = []
        for i in range(dh.shape[0]):
            idx = np.where(dh[i] > sat)[0]
            ts.append(int(idx[0]) if len(idx) else dh.shape[-1])
        ts = np.array(ts, float)
        t = np.arange(dh.shape[-1])
        surv = np.array([np.mean(ts > tt) for tt in t])
        ax.plot(t, surv, lw=1.2, label=f"T={T:g}")
    ax.set_xlim(0, 200); ax.set_ylim(0, 1.02)
    ax.set_title(n, fontsize=8.6)
    ax.set_xlabel("step $t$", fontsize=7.6)
    ax.set_ylabel("fraction not separated", fontsize=7.6)
    ax.legend(fontsize=5.8); ax.grid(alpha=0.25)
fig.suptitle("Survival: fraction of directions still in the same basin", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_survival.png")
print("done")
