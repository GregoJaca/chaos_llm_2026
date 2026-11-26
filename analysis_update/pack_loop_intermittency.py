"""How bursty is the closed-loop growth?

(a) windowed live-gate density (window = 50 steps) at T=1: the gates open in
    bursts, not uniformly;
(b) survival of "quiet" stretches (consecutive steps with m_t >= T);
(c) concentration of the log growth: fraction of total Delta log a contributed
    by the largest x% of steps (a staircase gives a very skewed curve, a
    constant-rate exponential gives the diagonal).
"""
import os
import sys
import glob
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import *
from style import plt, save

TAGS = {
    "DeepSeek-1.5B": "ds15", "Qwen2.5-3B": "qwen25_3b", "Qwen3-4B": "qwen3_4b",
    "Qwen3.5-2B": "qwen3_5", "Phi-4-mini": "phi4mini", "Mamba-130m": "mamba130",
    "Pythia-410m": "pythia410", "randQwen": "randqwen", "Qwen3.5-9B": "qwen35_9b",
    "Qwen3-14B": "qwen314", "DeepSeek-14B": "ds14", "Phi-4": "phi4",
}


def best(tag):
    for root in ("results/longloop", "results/softloop"):
        for sub in sorted(glob.glob(os.path.join(HERE, root, tag, "T1.0*"))):
            p = os.path.join(sub, "loop", "softloop.npz") if root.endswith("longloop") \
                else os.path.join(sub, "softloop.npz")
            if os.path.exists(p):
                return p
    return None


D = {}
for name, tag in TAGS.items():
    p = best(tag)
    if p:
        z = dict(np.load(p, allow_pickle=True))
        D[name] = (z, p)

fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9))
ax = axes[0]
conc = {}
for name, (z, p) in D.items():
    m = z["m_ref"].astype(float)
    w = 50
    k = np.ones(w) / w
    dens = np.convolve((m < 1.0).astype(float), k, mode="valid")
    ax.plot(np.arange(len(dens)) + w // 2, dens, color=COL.get(name, "gray"), lw=1.1,
            ls="--" if name == "randQwen" else "-",
            label=f"{name} ({len(m)})")
ax.set_xlabel("step $t$")
ax.set_ylabel(r"windowed density $P(m_t<1)$  ($w=50$)")
ax.set_title("(a) The gates open in bursts", loc="left")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=5.8, ncol=2); ax.grid(alpha=0.3)

ax = axes[1]
rows = {}
names = []
for name, (z, p) in D.items():
    m = z["m_ref"].astype(float)
    a = z["dh"][:, -1, :].astype(float)
    med = np.median(a, axis=0)
    d = np.diff(np.log(np.maximum(med, 1e-30)))
    T = min(len(d), len(m) - 1)
    d, mm = d[:T], m[:T]
    live = mm < 1.0
    if live.sum() < 10 or (~live).sum() < 10:
        continue
    rows[name] = (float(np.mean(d[live])), float(np.mean(d[~live])), float(live.mean()))
    names.append(name)
y = np.arange(len(names))
ax.barh(y - 0.2, [rows[n][0] for n in names], height=0.38, color="#d62728", label="gate open ($m_t<T$)")
ax.barh(y + 0.2, [rows[n][1] for n in names], height=0.38, color="#1f77b4", label="gate closed ($m_t\\geq T$)")
for i, n in enumerate(names):
    ax.text(max(rows[n][0], rows[n][1]) * 1.05, i, f"{100*rows[n][2]:.0f}% open",
            va="center", fontsize=6.2)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.2)
ax.set_xlabel(r"mean $\Delta\log a_t$ (nats/step)")
ax.set_title("(b) Growth happens while the gate is open", loc="left")
ax.legend(fontsize=6.6); ax.grid(alpha=0.3, axis="x")

ax = axes[2]
for name, (z, p) in D.items():
    a = z["dh"][:, -1, :].astype(float)
    med = np.median(a, axis=0)
    inc = np.diff(np.log(np.maximum(med, 1e-30)))
    inc = inc[np.isfinite(inc)]
    tot = inc[inc > 0].sum()
    if tot <= 0:
        continue
    s = np.sort(inc)[::-1]
    frac_steps = np.arange(1, len(s) + 1) / len(s)
    cum = np.cumsum(np.maximum(s, 0)) / tot
    ax.plot(frac_steps * 100, cum * 100, color=COL.get(name, "gray"), lw=1.4, label=name)
    # share of the growth produced by the top 5% of steps
    k = max(1, int(0.05 * len(s)))
    conc[name] = float(cum[k - 1])
ax.plot([0, 100], [0, 100], "k:", lw=1.0, label="constant rate (exponential)")
ax.set_xlabel("top % of steps (by positive $\\Delta\\log a$)")
ax.set_ylabel("% of total log growth")
ax.set_title("(c) Growth is concentrated in few steps", loc="left")
ax.legend(fontsize=6.0, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "pack_loop_intermittency.png")

json.dump(conc, open(os.path.join(HERE, "analysis_update", "pack_loop_intermittency.json"), "w"), indent=1)
print({k: round(v, 2) for k, v in conc.items()})
print("done")
