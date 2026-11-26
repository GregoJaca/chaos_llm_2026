"""Self-regulation of the loop: growth stops when the orbit becomes confident.

Top panel: the response a_t (log) of the closed loop.
Bottom panel: the readout margin m_t of the *reference* row, with the live-gate
region (m_t < T) shaded.  Growth happens inside the shaded region; once the
orbit drifts into confident repetition the gates close and a_t saturates.
"""
import os
import sys
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAGS = {"DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b",
        "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130", "Pythia-410m": "pythia410",
        "randQwen": "randqwen", "Qwen3.5-9B": "qwen35_9b"}
ORDER = list(TAGS)


def best(tag):
    for root in ("results/longloop", "results/softloop"):
        for sub in sorted(glob.glob(os.path.join(HERE, root, tag, "T1.0*")),
                          key=len, reverse=True):
            p = os.path.join(sub, "loop", "softloop.npz") if root.endswith("longloop") \
                else os.path.join(sub, "softloop.npz")
            if os.path.exists(p):
                return p
    return None


def roll(x, w):
    if len(x) <= w:
        return x
    p = w // 2
    return np.convolve(np.pad(x, p, mode="edge"), np.ones(w) / w, mode="valid")


D = {}
for n in ORDER:
    p = best(TAGS[n])
    if p:
        D[n] = dict(np.load(p, allow_pickle=True))

fig, axes = plt.subplots(len(D), 1, figsize=(13.4, 2.25 * len(D)), sharex=False)
for ax, (n, z) in zip(np.atleast_1d(axes), D.items()):
    a = np.median(z["dh"][:, -1, :].astype(float), axis=0)
    a = a / max(a[1], 1e-30)
    m = z["m_ref"].astype(float)
    t = np.arange(len(a))
    ax.plot(t, np.maximum(a, 1e-6), color="#d62728", lw=1.2, label="response $a_t$ (closed)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$a_t$", fontsize=7.6, color="#d62728")
    ax.tick_params(axis="y", labelsize=7, colors="#d62728")
    ax.grid(True, which="both", alpha=0.2)
    ax2 = ax.twinx()
    mm = roll(m, 21)
    ax2.plot(t, mm, color="#1f77b4", lw=1.0, label="margin $m_t$ (ref)")
    ax2.fill_between(t, 0, 1.0, where=(mm < 1.0), color="#2ca02c", alpha=0.18, lw=0,
                     transform=ax2.get_xaxis_transform(), step=None)
    ax2.axhline(1.0, color="k", ls=":", lw=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$m_t$", fontsize=7.6, color="#1f77b4")
    ax2.tick_params(axis="y", labelsize=7, colors="#1f77b4")
    frac = float(np.mean(m < 1.0))
    ax.set_title(f"{n}   —   live-gate fraction {100*frac:.0f}%   (green = $m_t<1$, growth on)",
                 loc="left", fontsize=8.6)
    ax.set_xlim(0, len(a) - 1)
fig.tight_layout()
save(fig, "pack_loop_selfreg.png")
print("done", list(D))
