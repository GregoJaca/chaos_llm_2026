"""Pooled survival statistics over the frozen prompt set (dense10).

For each temperature: survival curves per prompt (thin) + the pooled band
(median across prompts, IQR band across orbits), plus the cross-orbit
horizon distribution. Data: results/softloop_dense10/ds15/p*/T*/.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'chaos_llm_2026/results/softloop_dense10/ds15'
OUT = 'chaos_llm_2026/results/figures/softloop'
L = -1

COLOR = {0.1: '#5a9bd4', 0.2: '#1a6faa', 1.0: '#08306b'}

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
stats = {}
for T in [0.1, 0.2, 1.0]:
    survs, meds = [], []
    for d in sorted(glob.glob(os.path.join(SRC, 'p*', f'T{T}'))):
        f = os.path.join(d, 'softloop.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        dh = z['dh'][:, L]
        h0 = float(np.median(z['h0n'][L]))
        S = dh.shape[1]
        ts = np.array([int(np.argmax(dh[k] > 0.25 * h0))
                       if (dh[k] > 0.25 * h0).any() else S
                       for k in range(dh.shape[0])])
        surv = [float(np.mean(ts > t)) for t in range(0, S, 10)]
        survs.append(surv)
        meds.append((os.path.basename(os.path.dirname(d)), float(np.median(ts)),
                     int((ts == S).sum()), dh.shape[0]))
    if not survs:
        continue
    S = len(survs[0])
    xs_ = np.arange(S) * 10
    ax = axes[0]
    ax.plot(xs_, np.median(survs, axis=0), lw=2.2,
            color=COLOR[T], label=f'T={T:g} ({len(survs)} prompts)')
    ax.fill_between(xs_,
                    np.percentile(survs, 25, axis=0),
                    np.percentile(survs, 75, axis=0), alpha=0.2, color=COLOR[T])
    stats[T] = meds

ax = axes[0]
ax.set_xlabel('step $t$')
ax.set_ylabel('fraction of directions not yet separated')
ax.set_title('(a) pooled survival: median across prompts,\n'
             'band = IQR across prompt-orbits; K=64 directions each', fontsize=9)
ax.legend(fontsize=8)

ax = axes[1]
labels, vals, cens = [], [], []
for T in [0.1, 0.2, 1.0]:
    if T not in stats:
        continue
    for name, med, c, K in stats[T]:
        labels.append(f'{name[:14]}\nT={T:g}')
        vals.append(med)
        cens.append(c)
x = np.arange(len(labels))
ax.bar(x, vals, color=[COLOR[0.1] if 'T=0.1' in l else
                      (COLOR[0.2] if 'T=0.2' in l else COLOR[1.0]) for l in labels])
ax.axhline(600, color='gray', ls=':')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=5)
ax.set_ylabel('median separation step')
ax.set_title('(b) horizon per prompt-orbit: the orbit heterogeneity\n'
             'that the pooled band hides', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_survival_dense10.png'), dpi=150)
print('saved sl2_survival_dense10.png')
for T, ms in stats.items():
    print(f'T={T:g}:', [(m[0][:16], round(m[1]), f'{m[2]}/{m[3]}') for m in ms])
