import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m", "Qwen3.5-9B"]
fig, ax = plt.subplots(figsize=(9.8, 5.0))
for n in MODELS:
    Ts, dens = [], []
    for T, z, m in soft_runs(n):
        mm = z["m_ref"].astype(float)
        Ts.append(T); dens.append(np.mean(mm < T))
    if not Ts:
        continue
    o = np.argsort(Ts)
    Ts = np.array(Ts)[o]; dens = np.array(dens)[o]
    ax.plot(Ts, dens, "o-", color=COL[n], label=n)
ax.set_xscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"live-gate density $P(m_t<T)$")
ax.set_title("Live-gate density vs temperature (curves sorted by $T$)", loc="left")
ax.legend(fontsize=7.0); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_live_density_sorted.png")

# rates by model: box + strip
fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True)
labels, net, sep = [], [], []
for n in ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen", "Qwen3.5-9B"]:
    lr, ls = [], []
    for T, z, m in soft_runs(n):
        if "m_ref" in z:
            a = z["dh"][:, -1, :].astype(float)
            hn = float(np.median(z["h0n"][-1].astype(float))); sat = 0.25 * hn
            ts = []
            for i in range(a.shape[0]):
                idx = np.where(a[i] > sat)[0]
                ts.append(int(idx[0]) if len(idx) else a.shape[-1])
            tmax = np.nanmedian(ts)
            if np.isfinite(tmax) and tmax >= 8:
                t = np.arange(3, int(min(tmax, a.shape[-1])))
                ls.append(np.polyfit(t, np.log(np.maximum(np.median(a[:, t], axis=0), 1e-30)), 1)[0])
    for T, z, m in soft_runs(n, "replay"):
        a = z["dh"][:, -1, :].astype(float)
        t = np.arange(3, a.shape[-1])
        lr.append(np.polyfit(t, np.log(np.maximum(np.median(a[:, t], axis=0), 1e-30)), 1)[0])
    if lr or ls:
        labels.append(n); net.append(lr); sep.append(ls)
ax = axes[0]
bp = ax.boxplot(net, patch_artist=True, showfliers=False)
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.6)
for i, v in enumerate(net):
    ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(0).normal(0, 0.04, len(v)), v,
               s=8, color="k", alpha=0.4)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel(r"$\lambda_{\rm net}$  (nats/step)")
ax.set_title("(a) Network-only rate per model: all at zero", loc="left")
ax.grid(alpha=0.3, axis="y")
ax = axes[1]
bp = ax.boxplot(sep, patch_artist=True, showfliers=False)
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(COL[labels[i]]); b.set_alpha(0.6)
for i, v in enumerate(sep):
    ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(0).normal(0, 0.04, len(v)), v,
               s=8, color="k", alpha=0.4)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.6)
ax.set_ylabel(r"$\lambda_{\rm sep}$  (nats/step)")
ax.set_title("(b) Closed-loop rate per model: broad and often positive", loc="left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_loop_rates_by_model.png")
print("done")
