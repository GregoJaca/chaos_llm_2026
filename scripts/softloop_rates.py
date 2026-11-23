"""Rates vs time for the loop, unsmoothed and 20-token smoothed
(the same presentation as unified.md Figure 3d, applied to the soft loop).

lambda_1(t)  = log(||dh_{t+1}|| / ||dh_t||)          (raw, per token)
lambda_20(t) = log(||dh_{t+20}|| / ||dh_{t-20}||)/40 (20-token symmetric)
median over directions, last layer, per temperature.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'chaos_llm_2026/results/softloop/ds15v2'
OUT = 'chaos_llm_2026/results/figures/softloop'
TS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0]
L = -1

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
cmap = plt.get_cmap('viridis')
for i, T in enumerate(TS):
    f = os.path.join(SRC, f'T{T}', 'softloop.npz')
    if not os.path.exists(f):
        continue
    z = np.load(f)
    a = np.median(z['dh'][:, L], axis=0).astype(np.float64)  # median response
    S = len(a)
    color = cmap(i / max(len(TS) - 1, 1))
    lam1 = np.log(np.clip(a[1:], 1e-12, None) / np.clip(a[:-1], 1e-12, None))
    axes[0].plot(np.arange(S - 1), lam1, lw=0.7, color=color, alpha=0.8,
                 label=f'T={T:g}')
    k = 20
    lam20 = np.full(S, np.nan)
    for t in range(k, S - k):
        lam20[t] = np.log(a[t + k] / max(a[t - k], 1e-12)) / (2 * k)
    axes[1].plot(np.arange(S), lam20, lw=1.4, color=color)
axes[0].axhline(0, color='k', lw=0.8)
axes[1].axhline(0, color='k', lw=0.8)
axes[0].set_ylabel(r'$\lambda_1(t)$  (raw, nats/token)')
axes[1].set_ylabel(r'$\lambda_{20}(t)$  (smoothed, nats/token)')
axes[1].set_xlabel('step $t$')
axes[0].set_ylim(-2, 2)
axes[1].set_ylim(-0.15, 0.6)
axes[0].legend(fontsize=7, ncol=2)
axes[0].set_title('raw 1-token rates of the soft loop, per temperature\n'
                  'bursts appear as spikes exactly at gates; no sustained positive level '
                  'outside the band', fontsize=9)
axes[1].set_title('20-token symmetric-difference smoothing of the same data\n'
                  'the warm band (T=0.7-1.4) holds a positive plateau; cold T stays at/below 0',
                  fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_rates_vs_t.png'), dpi=150)
print('saved sl2_rates_vs_t.png')
