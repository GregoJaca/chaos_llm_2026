import os, sys, json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

DEPTH = {
    "DeepSeek-1.5B": "results/depth2/deepseek/depth.npz",
    "Qwen2.5-3B": "results/depth2/qwen25_3b/depth.npz",
    "Qwen3-4B": "results/depth2/qwen3_4b/depth.npz",
    "Qwen3.5-2B": "results/depth2/qwen3_5/depth.npz",
    "Phi-4-mini": "results/depth2/phi4mini/depth.npz",
    "DeepSeek-14B": "results_big_small/results_big/ds14/depth_w100/depth.npz",
    "Qwen3-14B": "results_big_small/results_big/qwen314/depth_w0/depth.npz",
    "Phi-4": "results_big_small/results_big/phi4/depth_w0/depth.npz",
}

def model(x, A, t1, B, t2, L):
    return 1 + A * np.exp(-x / t1) + B * np.exp((x - L) / t2)

rows = []
fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
for i, (n, p) in enumerate(DEPTH.items()):
    ax = axes.flat[i]
    z = np.load(os.path.join(HERE, p), allow_pickle=True)
    g = np.median(z["local_gain"].astype(float), axis=0)
    L = len(g); x = np.arange(L, dtype=float)
    try:
        popt, _ = curve_fit(lambda xx, A, t1, B, t2: model(xx, A, t1, B, t2, L),
                            x, g, p0=[5, 1, 0.1, 10],
                            bounds=([0, 0.05, 0, 1], [100, 20, 50, 100]), maxfev=30000)
        pred = model(x, *popt, L)
        r2 = 1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum()
        ax.plot(x, g, "k.", ms=3, label="measured")
        ax.plot(x, pred, "-", color=COL[n], lw=1.6, label="fit")
        ax.set_title(f"{n}\n" + r"$A$=%.1f, $\tau_1$=%.2f, $B$=%.2f, $\tau_2$=%.1f, $R^2$=%.2f" % (*popt, r2), fontsize=7.6)
        rows.append(dict(model=n, A=float(popt[0]), tau1=float(popt[1]),
                         B=float(popt[2]), tau2=float(popt[3]), r2=float(r2)))
    except Exception as e:
        ax.plot(x, g, "k.", ms=3)
        ax.set_title(f"{n}: fit failed", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("block $\\ell$", fontsize=7.8)
    ax.set_ylabel(r"$g_\ell=\langle\|J_\ell v_i\|\rangle_i$", fontsize=7.8)
    ax.grid(True, which="both", alpha=0.25)
fig.suptitle("A block-0 spike plus a late-depth rise: $g_\\ell=1+A e^{-\\ell/\\tau_1}+B e^{(\\ell-L)/\\tau_2}$",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_depth_local_fit.png")

json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_depth_localfit.json"), "w"), indent=1)
for r in rows:
    print(r)
