import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

def curve(path):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    r = z["radii"].astype(float)
    y = np.median(z["dh_norm"][:, :, -1, 50], axis=1)
    return r, y, z

fp32 = {
    "DeepSeek-1.5B": ("results/local_data/w100_layers/a1.npz", COL["DeepSeek-1.5B"]),
    "Qwen2.5-3B": ("results/update/fp32_qwen25_a1.npz", COL["Qwen2.5-3B"]),
    "Qwen3.5-2B": ("results/update/fp32_qwen35_a1.npz", COL["Qwen3.5-2B"]),
}
fp16 = {
    "Qwen2.5-3B (fp16 run)": ("results/update/a1r_qwen25/a1.npz", "#999999"),
    "Qwen3-4B (fp16 run)": ("results/update/a1r_qwen34/a1.npz", "#bbbbbb"),
    "Phi-4-mini (fp16 run)": ("results/update/a1r_phi/a1.npz", "#dddddd"),
}

fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))
ax = axes[0]
for n, (p, c) in fp32.items():
    r, y, _ = curve(p)
    ax.plot(r, y, "o-", ms=4, color=c, label=n + " (fp32)")
for n, (p, c) in fp16.items():
    r, y, _ = curve(p)
    ax.plot(r, y, "s--", ms=3, color=c, label=n)
rr = np.logspace(-4, -0.5, 50)
ax.plot(rr, rr * 1.0, "k:", lw=1.0, label=r"$\|\delta h\|=r$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"perturbation norm $r$")
ax.set_ylabel(r"$\|\delta h_{t=50,L}\|$  (absolute response, last layer)")
ax.set_title("(a) Same model, two precisions: fp16 has a floor", loc="left")
ax.legend(fontsize=6.2, ncol=2); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for n, (p, c) in fp32.items():
    r, y, _ = curve(p)
    beta = np.diff(np.log(y)) / np.diff(np.log(r))
    ax.plot(np.sqrt(r[:-1] * r[1:]), beta, "o-", ms=4, color=c, label=n)
ax.axhline(1, color="k", ls=":", lw=1.0, label=r"$\beta=1$ (linear)")
ax.axhline(0, color="#888888", ls="--", lw=1.0, label=r"$\beta=0$ (saturated)")
ax.set_xscale("log")
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"local exponent $\beta=\mathrm{d}\log\|\delta h\|/\mathrm{d}\log r$")
ax.set_title("(b) The linear regime is $r\\lesssim10^{-2}$", loc="left")
ax.set_ylim(-0.2, 1.4)
ax.legend(fontsize=6.8); ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
for n, (p, c) in fp32.items():
    r, y, z = curve(p)
    m = z["margins"].astype(float)
    med = []
    for i in range(len(r)):
        rc = z["r_c"][i].astype(float).copy()
        rc[:, m <= 0.02] = np.inf
        rs = np.min(rc, axis=1)
        med.append(np.median(rs[np.isfinite(rs)]) if np.isfinite(rs).any() else np.nan)
    ax.plot(r, med, "o-", ms=4, color=c, label=n)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"probe radius $r$ used to measure $\delta m$")
ax.set_ylabel(r"median $r_*$ estimated at that probe")
ax.set_title(r"(c) $r_*$ is only stable at small probe radius", loc="left")
ax.legend(fontsize=7.0); ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Precision and probe radius: why the old absolute response and $r_*$ were not comparable", fontsize=11.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "pack_linearity_precision.png")
print("done")
