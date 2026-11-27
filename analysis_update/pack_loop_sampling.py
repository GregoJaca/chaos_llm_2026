"""How much sampling does the loop need?  Closed (mean) vs CRN vs independent.

At each step the model produces a distribution p_t.
  soft : feed E[e]   = sum_v p_t(v) E_v                       (mean feedback)
  crn  : sample one token from the REFERENCE row's p_t and feed the same token
         embedding to every row  -> the sampling noise is identical for all rows,
         so any spread between rows is pure state dynamics (control for noise)
  ind  : each row samples its own token from its own p_t -> ordinary sampled
         generation; rows differ because both the noise and the state differ.

All three use the same reference orbit only for CRN (it needs p_ref); soft and ind
are self-contained.  a_t = ||dh_t||/||dh_1||, median over directions, IQR band.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from looputil import TAGS, pairs, gather
from style import plt, save

MODELS = ["DeepSeek-1.5B", "Qwen2.5-3B", "Qwen3-4B", "Qwen3.5-2B",
          "Phi-4-mini", "Mamba-130m", "Pythia-410m", "randQwen"]


def mode_curve(tag, mode, layer=-1):
    """pooled per-direction curve for a feedback mode stored under longloop/<tag>/T1.0/<mode>"""
    disk = {"loop": "loop", "crn": "crn", "ind": "ind"}[mode]
    for root in ("results/longloop_v3", "results/longloop"):
        d = os.path.join(HERE, root, tag, "T1.0", disk)
        if not os.path.isdir(d):
            d2 = os.path.join(HERE, root, tag, "T1.0")
            if mode == "loop" and os.path.exists(os.path.join(d2, "loop", "softloop.npz")):
                d = os.path.join(d2, "loop")
            else:
                continue
        p = os.path.join(d, "softloop.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        a = z["dh"][:, layer, :].astype(float)
        a = a / np.maximum(a[:, [1]], 1e-30)
        return a
    return None


rows = {}
fig, axes = plt.subplots(2, 4, figsize=(16.4, 7.4))
for ax, name in zip(axes.flat, MODELS):
    tag = TAGS[name]
    drew = False
    for mode, c, ls, lab in [("loop", "#d62728", "-", "soft (mean)"),
                             ("crn", "#1f77b4", "--", "CRN (same draw)"),
                             ("ind", "#2ca02c", ":", "independent")]:
        a = mode_curve(tag, mode)
        if a is None:
            continue
        t = np.arange(a.shape[1])
        m = np.median(a, axis=0)
        lo, hi = np.percentile(a, [25, 75], axis=0)
        ax.fill_between(t[1:], np.maximum(lo[1:], 1e-8), hi[1:], color=c, alpha=0.10, lw=0)
        ax.plot(t[1:], m[1:], color=c, ls=ls, lw=1.3, label=f"{lab} ({a.shape[0]} dirs)")
        drew = True
        rows.setdefault(name, {})[mode] = dict(growth=float(m[-1]), n=int(a.shape[0]))
    if not drew:
        ax.axis("off"); continue
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(name, fontsize=8.8)
    ax.set_xlabel("step $t$", fontsize=7.2)
    ax.set_ylabel(r"$\|\delta h_t\|/\|\delta h_1\|$", fontsize=7.2)
    ax.legend(fontsize=5.8); ax.grid(True, which="both", alpha=0.25)
for j in range(len(MODELS), 8):
    axes.flat[j].axis("off")
fig.suptitle("Mean feedback vs sampled feedback at $T=1$: only the mean destabilises", fontsize=11.4)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "pack_loop_sampling_modes.png")

ax_n = len(rows)
fig, ax = plt.subplots(figsize=(11.0, 4.6))
names = list(rows)
x = np.arange(len(names))
series = [("loop", "#d62728", "soft (mean)"), ("crn", "#1f77b4", "CRN"),
          ("ind", "#2ca02c", "independent")]
for j, (mode, c, lab) in enumerate(series):
    vals = [rows[n].get(mode, {}).get("growth", np.nan) for n in names]
    ax.bar(x + (j - 1) * 0.26, vals, width=0.25, color=c, label=lab)
ax.axhline(1, color="k", lw=0.9)
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7.4)
ax.set_ylabel(r"$\|\delta h_{T}\|/\|\delta h_1\|$ at the end of the run")
ax.set_title("Final response by feedback rule ($T=1$, 3000 steps)", loc="left")
ax.legend(fontsize=7.4); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
save(fig, "pack_loop_sampling_bars.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_loop_sampling.json"), "w"), indent=1)
for n in names:
    print(n, {k: round(v["growth"], 2) for k, v in rows[n].items()})
