import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini",
          "Mamba-130m", "Pythia-410m", "randQwen", "Qwen3.5-9B"]

def get(n, mode):
    for T, z, m in soft_runs(n, mode):
        if abs(T - 1.0) < 1e-9:
            return z
    return None

def finitetime(a, taus):
    out = []
    for tau in taus:
        if len(a) <= tau + 1:
            out.append(np.nan); continue
        lam = np.log(np.maximum(a[tau:], 1e-30)) - np.log(np.maximum(a[:-tau], 1e-30))
        out.append(np.mean(lam) / tau)
    return np.array(out)

taus = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120])
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2))
vals = {}
for ax, n in zip(axes.flat, MODELS):
    for mode, c, lab in [("loop", "#d62728", "closed"), ("replay", "#1f77b4", "replay")]:
        z = get(n, mode)
        if z is None:
            continue
        a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
        a = a / max(a[1], 1e-30)
        v = finitetime(a, taus)
        ax.plot(taus, v, "o-", color=c, label=lab)
        vals[(n, mode)] = v
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_title(n, fontsize=8.4)
    ax.set_xlabel(r"$\tau$", fontsize=7.4); ax.set_ylabel(r"$\Lambda(\tau)$", fontsize=7.4)
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle(r"Finite-time exponent $\Lambda(\tau)=\frac{1}{\tau}\log(a_{t+\tau}/a_t)$: closed vs replay", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_finitetime.png")

# PSD of log a
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2))
for ax, n in zip(axes.flat, MODELS):
    for mode, c, lab in [("loop", "#d62728", "closed"), ("replay", "#1f77b4", "replay")]:
        z = get(n, mode)
        if z is None:
            continue
        a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
        y = np.log(np.maximum(a, 1e-30)); y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 2), np.arange(len(y)))
        f = np.fft.rfftfreq(len(y), 1.0); P = np.abs(np.fft.rfft(y)) ** 2
        ok = (f > 0) & (P > 0)
        ax.loglog(f[ok], P[ok], color=c, lw=1.0, label=lab)
    ax.set_title(n, fontsize=8.4)
    ax.set_xlabel("frequency", fontsize=7.4); ax.set_ylabel("power", fontsize=7.4)
    ax.legend(fontsize=6.0); ax.grid(True, which="both", alpha=0.25)
fig.suptitle("Power spectrum of the log response: closed vs replay", fontsize=11.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "pack_loop_spectrum.png")
json.dump({f"{n}|{m}": v.tolist() for (n, m), v in vals.items()},
          open(os.path.join(HERE, "analysis_update", "pack_loop_finitetime.json"), "w"), indent=1)
print("done")

