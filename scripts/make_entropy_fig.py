"""Entropy and Margin Analysis: The mediator of orbit stability across architectures.
Produces:
  sl2_entropy_horizon.png
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = 'results/softloop'
OUT = 'results/figures/softloop'
NOTEBOOK_FIGS = 'notebook/newfigures'
os.makedirs(OUT, exist_ok=True)
os.makedirs(NOTEBOOK_FIGS, exist_ok=True)

MCOL = {'ds15v2': '#1a6faa', 'mamba130': '#c8102e', 'qwen25_3b': '#2e8b57',
        'qwen3_4b': '#b8860b', 'phi4mini': '#7a4faa', 'pythia410': '#444444',
        'randqwen': '#888888'}
MLAB = {'ds15v2': 'DeepSeek-1.5B', 'mamba130': 'Mamba-130m',
        'qwen25_3b': 'Qwen2.5-3B', 'qwen3_4b': 'Qwen3-4B',
        'phi4mini': 'Phi-4-mini', 'pythia410': 'Pythia-410m',
        'randqwen': 'randQwen'}

def make_entropy_fig():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (a) Orbit entropy vs Temperature across models
    ax = axes[0]
    for g, col in MCOL.items():
        Ts, ents = [], []
        for d in sorted(glob.glob(os.path.join(SRC, g, 'T*'))):
            f = os.path.join(d, 'softloop.npz')
            if not os.path.exists(f): continue
            try:
                T = float(os.path.basename(d)[1:])
            except ValueError: continue
            if T < 1e-6: continue
            z = np.load(f)
            Ts.append(T)
            ents.append(float(np.median(z['ent_ref'])))
        if Ts:
            ax.plot(Ts, ents, 'o-', color=col, lw=1.8, ms=5, label=MLAB.get(g, g))
    ax.set_xscale('log')
    ax.set_xlabel('temperature $T$')
    ax.set_ylabel('median orbit entropy $H(p_t)$ (nats)')
    ax.set_title('(a) orbit entropy vs temperature\ntrained models sharp at low T; randQwen flat', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left')

    # (b) Predictability horizon vs Orbit Median Margin
    ax = axes[1]
    for g, col in MCOL.items():
        margins, horizons, Ts = [], [], []
        for d in sorted(glob.glob(os.path.join(SRC, g, 'T*'))):
            f = os.path.join(d, 'softloop.npz')
            if not os.path.exists(f): continue
            try:
                T = float(os.path.basename(d)[1:])
            except ValueError: continue
            if T < 1e-6: continue
            z = np.load(f)
            dh = z['dh'][:, -1]
            h0 = float(np.median(z['h0n'][-1]))
            S = dh.shape[1]
            ts = np.array([int(np.argmax(dh[k] > 0.25 * h0)) if (dh[k] > 0.25 * h0).any() else S for k in range(dh.shape[0])])
            margins.append(float(np.median(z['m_ref'])) / T)
            horizons.append(float(np.median(ts)))
            Ts.append(T)
        if margins:
            ax.scatter(margins, horizons, s=45, color=col, alpha=0.8,
                       marker={'ds15v2': 'o', 'mamba130': 's', 'qwen25_3b': '^', 'phi4mini': 'v'}.get(g, 'o'),
                       label=MLAB.get(g, g))
    ax.axhline(600, color='gray', ls=':', lw=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('normalized margin: median $(m_t / T)$')
    ax.set_ylabel('median separation step $t_{sat}$')
    ax.set_title('(b) horizon vs normalized margin $(m/T)$\nlarge margin orbits survive; low margin separates rapidly', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left')

    # (c) Trajectory: Step-by-step gate occurrences vs entropy (DeepSeek-1.5B, T=0.2)
    ax = axes[2]
    f = os.path.join(SRC, 'ds15v2', 'T0.2', 'softloop.npz')
    if os.path.exists(f):
        z = np.load(f)
        steps = min(200, len(z['ent_ref']))
        t_arr = np.arange(steps)
        ax.plot(t_arr, z['ent_ref'][:steps], color='#1a6faa', lw=1.5, label='entropy $H_t$ (nats)')
        ax.plot(t_arr, z['m_ref'][:steps], color='#e66101', lw=1.5, label='margin $m_t$ (nats)')
        # highlight live gates where m < 3T (for T=0.2, 3T=0.6)
        gates = z['m_ref'][:steps] < 0.6
        if gates.any():
            ax.scatter(t_arr[gates], z['m_ref'][:steps][gates], color='crimson', s=25, zorder=5, label='live gate ($m_t < 3T$)')
    ax.set_xlabel('generation step $t$')
    ax.set_ylabel('nats')
    ax.set_title('(c) orbit trajectory: entropy spikes align with gates\nDeepSeek-1.5B at $T=0.2$', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Entropy and Margin Dynamics: The Quantitative Mechanism of Soft-Loop Stability',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sl2_entropy_horizon.png'), dpi=160)
    fig.savefig(os.path.join(NOTEBOOK_FIGS, 'sl2_entropy_horizon.png'), dpi=160)
    plt.close(fig)
    print('saved sl2_entropy_horizon.png')

if __name__ == '__main__':
    make_entropy_fig()
