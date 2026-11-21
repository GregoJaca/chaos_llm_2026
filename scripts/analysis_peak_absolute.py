"""Is the amplification a property of the position or of the absolute generated token?"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
B = os.path.join(ROOT, "results", "local_data", "probe2")
OUT = os.path.join(ROOT, "notebook", "figures", "positions")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "legend.fontsize": 8})
MODELS = [("DeepSeek-1.5B", "deepseek", "tab:blue"),
          ("Qwen3-4B", "qwen3_4b", "tab:orange"),
          ("Qwen2.5-3B", "qwen25_3b", "tab:red"),
          ("Phi-4-mini", "phi4mini", "tab:green"),
          ("Qwen3.5-2B", "qwen3_5", "tab:purple")]

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for name, tag, c in MODELS:
    d0 = os.path.join(B, tag)
    if not os.path.isdir(d0):
        continue
    S, tau, tpk = [], [], []
    for sub in sorted(os.listdir(d0)):
        f = os.path.join(d0, sub, "layer_probe.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); p = int(z["pos"]); r = float(z["r"])
        curve = np.median(z["dh"][0][:, -1, p + 1:p + 51], axis=0) / r
        k = int(np.argmax(curve))
        S.append(p); tau.append(k + 1); tpk.append(p + k + 1)
    o = np.argsort(S)
    S = np.array(S)[o]; tau = np.array(tau)[o]; tpk = np.array(tpk)[o]
    axs[0].plot(S, tau, "o-", ms=4, lw=1.2, color=c, label=name)
    axs[1].plot(S, tpk, "o-", ms=4, lw=1.2, color=c, label=name)
    if len(S) > 3:
        sl, ic = np.polyfit(S, tau, 1)
        axs[0].text(S[0], tau[0], f"slope {sl:.2f}", fontsize=7, color=c)
for ax, ttl, ylab in [(axs[0], "(a) Lag of the peak vs position", r"peak lag $\tau^*$ (tokens)"),
                      (axs[1], "(b) Absolute token of the peak", r"$s+\tau^*$ (generated token)")]:
    ax.set_title(ttl); ax.set_xlabel("position $s$ of the perturbation")
    ax.set_ylabel(ylab); ax.grid(alpha=0.25); ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "peak_is_absolute.png"), dpi=185, bbox_inches="tight")
plt.close(fig)
print("saved")
