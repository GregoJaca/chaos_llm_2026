"""Master phase-summary figure for the ds15 soft-loop grid.

Panel (a): predictability horizon vs temperature (per-direction saturation
steps; censored runs at the top edge) — the phase map of the growing loop.
Panel (b): separation rate (linear regime, r<=0.03 where measured) and the
network-only replay rate lambda_net — the decomposition.
Panel (c): live-gate fraction from the orbit's own margins (m_t < 3T),
the quantity that mediates both the cold (threshold) and warm (smooth)
regimes.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'chaos_llm_2026/results/softloop'
rows = json.load(open('chaos_llm_2026/results/figures/softloop/sl_numbers.json'))['ds15']
L_LAST = -1


DIRS = {}
for d in glob.glob(os.path.join(SRC, 'ds15', 'T*')):
    if os.path.isdir(d):
        try:
            DIRS[float(os.path.basename(d)[1:])] = d
        except ValueError:
            pass


def tsat_all(d):
    z = np.load(os.path.join(d, 'softloop.npz'))
    dh = z['dh'][:, L_LAST]
    S = dh.shape[1]
    h0 = float(np.median(z['h0n'][L_LAST]))
    out = []
    for k in range(dh.shape[0]):
        sat = dh[k] > 0.25 * h0
        out.append(int(np.argmax(sat)) if sat.any() else S)
    m = z['m_ref']
    return np.array(out), m


fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

# (a) horizon
ax = axes[0]
for r in rows:
    d = DIRS[r['T']]
    ts, m = tsat_all(d)
    jitter = np.linspace(-0.02, 0.02, len(ts))
    x = r['T'] * (1 + jitter)
    ax.scatter(x, np.clip(ts, 0, 620), s=14,
               color='crimson' if r['T'] > 0.1 else 'steelblue')
Ts = [r['T'] for r in rows]
med = [min(r['tsat_med'], 620) for r in rows]
ax.plot(Ts, med, 'k-', lw=1.2, zorder=0)
ax.axhline(600, color='gray', ls=':', lw=0.8)
ax.text(1.3e-9, 560, 'censored (no separation)', fontsize=7, color='gray')
ax.set_xscale('log')
ax.set_xlabel('temperature T')
ax.set_ylabel('separation step $t_{sat}$ (600 = never)')
ax.set_title('(a) predictability horizon of the growing soft loop\n'
             'frozen (T<0.1) / gate-schedule / smooth band (T~1) / high-T weakening', fontsize=9)

# (b) rates
ax = axes[1]
lam = np.array([r['lam_sep'] for r in rows])
net = np.array([r.get('lam_net', np.nan) for r in rows])
ax.errorbar(Ts, lam, yerr=[lam - np.array([r['lam_sep_iqr'][0] for r in rows]),
                           np.array([r['lam_sep_iqr'][1] for r in rows]) - lam],
            fmt='o-', color='#1a6faa', label=r'$\lambda_{sep}$ (loop, r=0.03)')
ax.plot(Ts, net, 's-', color='#c8102e', label=r'$\lambda_{net}$ (replay: no feedback)')
ax.axhline(0, color='k', lw=0.8)
# linear-regime points at T=1.0 from the r-grid
ax.scatter([1.0], [0.056], marker='*', s=120, color='#2e8b57', zorder=5,
           label=r'$\lambda_{sep}$ at r=0.003 (linear regime)')
ax.set_xscale('log')
ax.set_xlabel('temperature T')
ax.set_ylabel('nats / step')
ax.legend(fontsize=7)
ax.set_title('(b) separation vs network-only rate\nthe network is neutral at every T; all instability is feedback', fontsize=9)

# (c) live-gate fraction from orbit margins
ax = axes[2]
frac3, frac1 = [], []
for r in rows:
    d = DIRS[r['T']]
    z = np.load(os.path.join(d, 'softloop.npz'))
    m = z['m_ref']
    frac3.append(float((m < 3 * r['T']).mean()))
    frac1.append(float((m < r['T']).mean()))
ax.plot(Ts, frac3, 'o-', color='#b8860b', label=r'fraction of steps with $m_t < 3T$')
ax.plot(Ts, frac1, 's-', color='#556b2f', label=r'fraction with $m_t < T$')
ax.set_xscale('log')
ax.set_xlabel('temperature T')
ax.set_ylabel('fraction of orbit steps')
ax.legend(fontsize=7)
ax.set_title('(c) live-gate density from the orbit\'s own margins\n'
             'the mediator of both regimes', fontsize=9)

fig.tight_layout()
out = 'chaos_llm_2026/results/figures/softloop/sl_phase_ds15.png'
fig.savefig(out, dpi=150)
print('saved', out)
