import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))

# (a) relative response vs depth, small models with IQR
ax = axes[0, 0]
small = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Phi-4-mini", "Qwen3.5-2B"]
for n in small:
    d = load_a1(n)
    radii = d["radii"].astype(float)
    ri = int(np.argmin(np.abs(radii - 0.1)))
    x = d["dh_norm"][ri].astype(float) / d["norm_h0"].astype(float)[None, :, :]  # dir,layer,t
    t = min(50, x.shape[2] - 1)
    med = np.median(x[:, :, t], axis=0)
    lo, hi = np.percentile(x[:, :, t], [25, 75], axis=0)
    med, lo, hi = med[1:], lo[1:], hi[1:]
    base = max(med[0], 1e-30)
    lay = np.arange(len(med)) / max(len(med) - 1, 1)
    ax.plot(lay, med / base, color=COL[n], label=n)
    ax.fill_between(lay, lo / base, hi / base, color=COL[n], alpha=0.15, lw=0)
ax.set_yscale("log")
ax.set_xlabel("normalized depth  $\\ell/(L-1)$")
ax.set_ylabel(r"$\|\delta h_\ell\|/\|h_\ell\|$  normalized at layer 1  ($t=50$)")
ax.set_title("(a) Relative perturbation vs depth: flat or contracting", loc="left")
ax.legend(fontsize=7.2)
ax.grid(True, which="both", alpha=0.3)

# (b) all models including 14B, from extracted summaries
ax = axes[0, 1]
big = json.load(open(os.path.join(HERE, "results_big_small", "big_a1_summary.json")))
for n, v in big.items():
    y = np.array(v["rel_depth"])[1:]
    lay = np.arange(len(y)) / max(len(y) - 1, 1)
    ax.plot(lay, y, color=COL[n], lw=1.8, label=f"{n} (last {y[-1]:.2f})")
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("normalized depth  $\\ell/(L-1)$  (saved layers only)")
ax.set_ylabel(r"relative response, normalized at layer 1")
ax.set_title("(b) The same for the 14B models", loc="left")
ax.legend(fontsize=7.4)
ax.grid(True, which="both", alpha=0.3)

# (c) absolute vs relative, DeepSeek
ax = axes[1, 0]
d = load_a1("DeepSeek-1.5B")
radii = d["radii"].astype(float); ri = int(np.argmin(np.abs(radii - 0.1)))
dh = d["dh_norm"][ri].astype(float)
hn = d["norm_h0"].astype(float)
t = 50
absg = np.median(dh[:, :, t], axis=0); absg /= max(absg[1], 1e-30)
rel = absg / (hn[:, t] / hn[1, t])
ax.plot(np.arange(len(hn)) / (len(hn) - 1), hn[:, t] / hn[1, t], color="#2ca02c",
        label=r"state norm $\|h_\ell\|/\|h_1\|$")
ax.plot(np.arange(len(absg)) / (len(absg) - 1), absg, color="#d62728",
        label=r"absolute response $\|\delta h_\ell\|/\|\delta h_1\|$")
ax.plot(np.arange(len(rel)) / (len(rel) - 1), rel, color="#1f77b4",
        label=r"relative response")
ax.set_yscale("log")
ax.set_xlabel("normalized depth")
ax.set_ylabel("ratio to layer 1")
ax.set_title("(c) DeepSeek: state grows, absolute response grows, relative response does not", loc="left")
ax.legend(fontsize=7.6)
ax.grid(True, which="both", alpha=0.3)

# (d) last-layer relative response vs model size
ax = axes[1, 1]
for n in small:
    d = load_a1(n)
    radii = d["radii"].astype(float); ri = int(np.argmin(np.abs(radii - 0.1)))
    x = d["dh_norm"][ri].astype(float) / d["norm_h0"].astype(float)[None, :, :]
    t = min(50, x.shape[2] - 1)
    med = np.median(x[:, :, t], axis=0)
    val = med[-1] / max(med[1], 1e-30)
    ax.scatter(MODEL_SIZE[n], val, color=COL[n], s=55, zorder=3)
    ax.annotate(n, (MODEL_SIZE[n], val), fontsize=7.2, xytext=(4, 4), textcoords="offset points")
for n, v in big.items():
    val = v["rel_depth"][-1]
    ax.scatter(MODEL_SIZE[n], val, color=COL[n], s=55, zorder=3)
    ax.annotate(n, (MODEL_SIZE[n], val), fontsize=7.2, xytext=(4, 4), textcoords="offset points")
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("model parameters (billions, log scale)")
ax.set_ylabel("relative response at last saved layer / layer 1")
ax.set_title("(d) Only Phi-4-mini amplifies the relative perturbation", loc="left")
ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Does the perturbation become more important with depth? (relative response)", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_relative_response_depth.png")
