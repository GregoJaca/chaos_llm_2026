import os, sys, json
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

z = np.load(os.path.join(HERE, "results/window_tokens/tokens.npz"), allow_pickle=True)
meta = json.load(open(os.path.join(HERE, "results/window_tokens/meta.json")))
P = int(meta["prompt_len"])
ref = z["w4096"]
Ws = [60, 100, 150, 200, 300, 500, 1000, 4096]

def frac(W):
    same = (z[f"w{W}"] == ref)
    return np.cumsum(same) / np.arange(1, len(same) + 1)

def fdecay(t, finf, tau, beta):
    return finf + (1 - finf) * np.exp(-(t / tau) ** beta)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
ax = axes[0]
fits = {}
for W in Ws[:-1]:
    y = frac(W); t = np.arange(1, len(y) + 1)
    ax.plot(t, y, lw=1.2, label=f"W={W}")
    try:
        popt, _ = curve_fit(fdecay, t, y, p0=[0.4, 30, 1], bounds=([0, 1, 0.2], [1, 500, 4]), maxfev=20000)
        fits[W] = popt
        ax.plot(t, fdecay(t, *popt), "k--", lw=0.8, alpha=0.6)
    except Exception:
        pass
ax.set_xlabel("generated token $t$")
ax.set_ylabel("fraction identical to $W=4096$")
ax.set_title("(a) Agreement and stretched-exponential fits (dashed)", loc="left")
ax.legend(fontsize=6.4, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
for W in [60, 100, 150, 200, 300]:
    y = frac(W); t = np.arange(1, len(y) + 1)
    ax.semilogy(t, np.maximum(1 - y, 1e-4), lw=1.2, label=f"W={W}")
ax.set_xlabel("generated token $t$")
ax.set_ylabel(r"$1-\mathrm{agreement}(t)$  (log)")
ax.set_title("(b) The decay is not a single exponential", loc="left")
ax.legend(fontsize=7.0); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save(fig, "pack_window_agreement_fit.png")

json.dump({str(W): dict(finf=float(v[0]), tau=float(v[1]), beta=float(v[2])) for W, v in fits.items()},
          open(os.path.join(HERE, "analysis_update", "pack_window_fit.json"), "w"), indent=1)
print(fits)
