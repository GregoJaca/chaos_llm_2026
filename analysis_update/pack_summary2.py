import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

# ---------------------------------------------------------------- master summary
fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
ax = axes[0, 0]
for T, c in [(0.05, "#08306b"), (0.1, "#1f77b4"), (0.5, "#ff7f0e"), (1.0, "#d62728"), (2.0, "#9467bd")]:
    runs = [z for TT, z, m in soft_runs("DeepSeek-1.5B") if abs(TT - T) < 1e-9]
    if not runs:
        continue
    z = runs[0]; a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
    a = a / max(a[1], 1e-30)
    ax.plot(np.arange(len(a)), a, color=c, lw=1.3, label=f"$T$={T:g}")
ax.set_yscale("log"); ax.set_xlim(0, 200)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
ax.set_title("(a) Soft loop: temperature changes the growth", loc="left")
ax.legend(fontsize=7.0); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
loop = [z for T, z, m in soft_runs("DeepSeek-1.5B") if abs(T - 1.0) < 1e-9]
rep = [z for T, z, m in soft_runs("DeepSeek-1.5B", "replay") if abs(T - 1.0) < 1e-9]
if loop:
    a = np.median(loop[0]["dh"][:, -1, :].astype(float), axis=0)
    ax.plot(np.arange(len(a)), a / max(a[1], 1e-30), color="#d62728", label="closed loop")
if rep:
    a = np.median(rep[0]["dh"][:, -1, :].astype(float), axis=0)
    ax.plot(np.arange(len(a)), a / max(a[1], 1e-30), color="#1f77b4", ls="--", label="replay")
ax.set_yscale("log"); ax.set_xlim(0, 300)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$")
ax.set_title("(b) Same orbit, feedback on/off", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 0]
xs, ys = [], []
for T, z, m in soft_runs("DeepSeek-1.5B"):
    mm = z["m_ref"].astype(float); s = z["sig_read"].astype(float)
    ok = np.isfinite(mm) & np.isfinite(s) & (s > 0)
    xs.append(mm[ok] / T); ys.append(s[ok])
x = np.concatenate(xs); y = np.concatenate(ys)
idx = np.arange(0, len(x), max(1, len(x) // 3000))
ax.scatter(x[idx], y[idx], s=3, alpha=0.3, color="#1f77b4")
xx = np.logspace(-1.3, 2, 300)
ax.plot(xx, 2.0 / 4 / np.cosh(xx / 2) ** 2, "r-", lw=1.2, label="identity $G/4\\cosh^{-2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$m_t/T$")
ax.set_ylabel(r"$\sigma_{\max}(\Sigma_p)$")
ax.set_title("(c) Gate law, DeepSeek all temperatures", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 1]
soft = [z for T, z, m in soft_runs("DeepSeek-1.5B") if abs(T - 1.0) < 1e-9]
crn = os.path.join(HERE, "results/softloop_sample/ds15/crn_T1.0/softloop.npz")
ind = os.path.join(HERE, "results/softloop_sample/ds15/ind_T1.0/softloop.npz")
if soft:
    a = np.median(soft[0]["dh"][:, -1, :].astype(float), axis=0)
    ax.plot(np.arange(len(a)), a, color="#d62728", label="soft")
for p, c, lab in [(crn, "#1f77b4", "CRN"), (ind, "#2ca02c", "independent")]:
    if os.path.exists(p):
        z = dict(np.load(p, allow_pickle=True))
        a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
        ax.plot(np.arange(len(a)), a, c, ls="--", label=lab)
ax.set_yscale("log"); ax.set_xlim(0, 200)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$\|\delta h_t\|$")
ax.set_title("(d) Feedback rule: soft vs sampled", loc="left")
ax.legend(fontsize=7.4); ax.grid(True, which="both", alpha=0.3)
fig.suptitle("The four central results on one page (DeepSeek-1.5B)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_master_summary.png")

# ---------------------------------------------------------------- live density
MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Mamba-130m", "Qwen3.5-9B"]
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
ax = axes[0]
for n in MODELS:
    Ts, dens = [], []
    for T, z, m in soft_runs(n):
        mm = z["m_ref"].astype(float)
        Ts.append(T); dens.append(np.mean(mm < T))
    if Ts:
        ax.plot(Ts, dens, "o-", color=COL[n], label=n)
ax.set_xscale("log")
ax.set_xlabel("temperature $T$")
ax.set_ylabel(r"live-gate density  $P(m_t<T)$")
ax.set_title("(a) Density grows with $T$, but not at the same rate", loc="left")
ax.legend(fontsize=6.6); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n in MODELS:
    Ts, dens, ts = [], [], []
    for T, z, m in soft_runs(n):
        mm = z["m_ref"].astype(float)
        r = float(m["r"])
        dh = z["dh"][:, -1, :].astype(float) / r
        hn = float(np.median(z["h0n"][-1].astype(float)))
        sat = 0.25 * hn
        tt = []
        for i in range(dh.shape[0]):
            idx = np.where(dh[i] > sat)[0]
            tt.append(int(idx[0]) if len(idx) else dh.shape[-1])
        Ts.append(T); dens.append(max(np.mean(mm < T), 1e-6)); ts.append(np.median(tt))
    if Ts:
        ax.scatter(dens, ts, color=COL[n], s=34, label=n, alpha=0.9, edgecolor="white", lw=0.4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"live-gate density  $P(m_t<T)$")
ax.set_ylabel(r"median $t_{\rm sat}$")
ax.set_title("(b) Same density, different horizon: strength matters", loc="left")
ax.legend(fontsize=6.6); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_live_density.png")
print("done")
