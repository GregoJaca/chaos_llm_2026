"""What does the loop actually do at the end?  Repetition, period, and whether the
orbit converges to a fixed point (Cauchy) or to a limit cycle.

Data used (all already saved by run_softloop):
  top1_ref[t]  argmax token of the reference row at step t
  e_ref[t]     the embedding that was fed at step t (D-dim)
  m_ref[t]     top1-top2 margin at step t

We report, for every model at T=1:
  * the shortest period p<=Pmax that reproduces the token stream over the last
    W_test steps (exact match of top1_ref),
  * the embedding distance ||e_t - e_{t-p}|| / ||e||_typ, compared with the
    step-to-step distance ||e_t - e_{t-1}||,
  * how many distinct tokens appear in the final window (vocabulary of the loop).
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import COL, HERE
from looputil import TAGS, pairs
from style import plt, save

PMAX = 24
TEST = 300


def period_of(seq, pmax=PMAX, test=TEST):
    """smallest p such that seq[t] == seq[t-p] for all t in the last `test` steps"""
    n = len(seq)
    if n < 4 * test:
        test = n // 4
    tail = seq[n - test:]
    for p in range(1, min(pmax, n // test) + 1):
        if np.all(tail[p:] == tail[:-p]):
            return p
    return 0


rows = {}
fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9))
ax = axes[0]
name_list = []
for name in TAGS:
    ps = pairs(name)
    if not ps:
        continue
    pl = ps[0][0]
    z = np.load(pl, allow_pickle=True)
    if "top1_ref" not in z.files:
        continue
    top1 = z["top1_ref"].astype(np.int64)
    e = z["e_ref"].astype(float)
    m = z["m_ref"].astype(float)
    n = min(len(top1), e.shape[0])
    top1, e, m = top1[:n], e[:n], m[:n]
    p = period_of(top1)
    tail = slice(max(n - TEST, 0), n)
    e_typ = float(np.median(np.linalg.norm(e, axis=1)))
    n0 = max(n - TEST, 1)
    d1 = float(np.linalg.norm(e[n0:] - e[n0 - 1:n - 1], axis=1).mean()) / e_typ
    dp = (float(np.linalg.norm(e[n0:] - e[n0 - p:n - p], axis=1).mean()) / e_typ) if p else np.nan
    uniq = int(len(np.unique(top1[tail])))
    rows[name] = dict(steps=int(n), period=int(p), d1=d1, d_period=dp,
                      uniq_tail=uniq, m_med=float(np.median(m[tail])),
                      top_tokens=[int(x) for x in np.bincount(top1[tail]).argsort()[::-1][:5]],
                      top_counts=[int(x) for x in np.sort(np.bincount(top1[tail]))[::-1][:5]])
    print("%-14s n=%-5d period=%-3d |e_t-e_{t-1}|/|e|=%.3f |e_t-e_{t-p}|/|e|=%.3f uniq=%d  m=%.2f"
          % (name, n, p, d1, dp, uniq, rows[name]["m_med"]))
    name_list.append(name)

names = name_list
y = np.arange(len(names))
per = np.array([rows[n]["period"] for n in names])
ax.barh(y, per, color=[COL[n] for n in names], alpha=0.85)
for i, n in enumerate(names):
    txt = f"p={per[i]}" if per[i] else "none ≤ 24"
    ax.text(max(per[i], 0.2) + 0.2, i, txt, va="center", fontsize=6.4)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.2)
ax.set_xlabel("detected token period of the reference orbit (last 300 steps)")
ax.set_title("(a) Exact periodicities: fixed point ($p$=1) to none", loc="left")
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
for name in names:
    ps = pairs(name)
    z = np.load(ps[0][0], allow_pickle=True)
    e = z["e_ref"].astype(float)
    n = e.shape[0]
    e_typ = float(np.median(np.linalg.norm(e, axis=1)))
    n0 = max(n - min(TEST, n // 4), 41)
    lags = np.arange(1, 41)
    dd = [float(np.linalg.norm(e[n0:n] - e[n0 - l:n - l], axis=1).mean()) / e_typ for l in lags]
    ax.plot(lags, dd, "-o", ms=2.5, lw=1.1, color=COL[name], label=name)
ax.set_xlabel("lag $\\Delta$ (steps)")
ax.set_ylabel(r"$\|e_t-e_{t-\Delta}\|/\|e\|$")
ax.set_title("(b) Similarity vs lag: a dip at $\\Delta=p$ means a cycle", loc="left")
ax.legend(fontsize=5.6, ncol=2); ax.grid(alpha=0.3)

ax = axes[2]
vals = [rows[n]["uniq_tail"] for n in names]
ax.barh(y, vals, color=[COL[n] for n in names], alpha=0.85)
for i, n in enumerate(names):
    ax.text(vals[i] + 1.5, i, f"{vals[i]}  ($m$={rows[n]['m_med']:.1f})", va="center", fontsize=6.2)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.2)
ax.set_xlabel("distinct tokens in the last 300 steps (with median margin)")
ax.set_title("(c) Locked repetition vs wandering", loc="left")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
save(fig, "pack_loop_repetition.png")
json.dump(rows, open(os.path.join(HERE, "analysis_update", "pack_loop_repetition.json"), "w"), indent=1)
