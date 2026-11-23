"""Cross-model gate analysis from the a1/rstar corpora (no GPU needed).

Inputs: results/softloop_gates/<model>/<prompt>/gates.npz  (G_t, m_t, top ids)
        for deepseek/phi4mini/qwen25_3b/qwen3_4b (5 prompts each),
        mamba130/randqwen(/pythia410) (loop0), plus the old a1_* gates.

Outputs:
  sl2_gate_geometry_models.png   pair-geometry G_t distributions per model
  sl2_live_density_models.png    predicted live-gate density vs T per model
                                  (the cross-model T_c ordering), with the
                                  margin landscape insets
  sl2_gates_table.json           all numbers
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results')
OUT = os.path.join(ROOT, 'figures', 'softloop')

SETS = {
    'deepseek': ('softloop_gates/deepseek', 'DeepSeek-1.5B'),
    'phi4mini': ('softloop_gates/phi4mini', 'Phi-4-mini'),
    'qwen25_3b': ('softloop_gates/qwen25_3b', 'Qwen2.5-3B'),
    'qwen3_4b': ('softloop_gates/qwen3_4b', 'Qwen3-4B'),
    'mamba130': ('softloop_gates/mamba130', 'Mamba-130m'),
    'randqwen': ('softloop_gates/randqwen', 'randQwen g=1'),
    'pythia410': ('softloop_gates/pythia410', 'Pythia-410m'),
    'qwen3_5': ('softloop_gates/qwen3_5', 'Qwen3.5-2B'),
}
COL = {'deepseek': '#1a6faa', 'phi4mini': '#7a4faa', 'qwen25_3b': '#2e8b57',
       'qwen3_4b': '#b8860b', 'mamba130': '#c8102e', 'randqwen': '#888888',
       'pythia410': '#444444', 'qwen3_5': '#d55e00'}

data = {}
for key, (path, label) in SETS.items():
    fs = sorted(glob.glob(os.path.join(ROOT, path, '*', 'gates.npz')))
    if not fs:
        continue
    G = [np.load(f)['G'] for f in fs]
    M = [np.load(f)['m'] for f in fs]
    data[key] = dict(label=label, G=np.concatenate(G), m=np.concatenate(M),
                     n_runs=len(fs))

# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
order = sorted(data, key=lambda k: -np.median(data[k]['G']))
bp = ax.boxplot([data[k]['G'] for k in order], vert=True, patch_artist=True,
                showfliers=False, tick_labels=[data[k]['label'] for k in order])
for i, k in enumerate(order):
    bp['boxes'][i].set_facecolor(COL[k])
    bp['boxes'][i].set_alpha(0.6)
    bp['medians'][i].set_color('black')
ax.set_ylabel(r'$G_t = \|e_i - e_j\|\cdot\|w_i - w_j\|$  (top-2 pair geometry)')
ax.set_title('(a) gate strength geometry per model\n'
             'pooled over prompts; boxes = IQR, whiskers = 1.5 IQR', fontsize=9)
ax.tick_params(axis='x', rotation=45, labelsize=8)
for i, k in enumerate(order):
    ax.text(i + 1, np.median(data[k]['G']) * 1.15,
            f"{np.median(data[k]['G']):.2f}", ha='center', fontsize=7)

ax = axes[1]
for k in order:
    G = data[k]['G']
    ax.hist(np.log10(G), bins=40, density=True, histtype='step', lw=1.6,
            color=COL[k], label=data[k]['label'])
ax.set_xlabel(r'$\log_{10} G_t$')
ax.set_ylabel('density')
ax.set_title('(b) the same distributions on a log scale\n'
             'spread of $G_t$ sets the width of the gate-law band', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_gate_geometry_models.png'), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ figure 2
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
Tgrid = np.logspace(-2.3, 0.6, 40)
for k in sorted(data):
    m = data[k]['m']
    frac = np.array([(m < 3 * T).mean() for T in Tgrid])
    ax.plot(Tgrid, frac, lw=1.8, color=COL[k], label=data[k]['label'])
ax.set_xscale('log')
ax.set_xlabel('temperature $T$')
ax.set_ylabel(r'fraction of greedy-orbit steps with $m_t < 3T$')
ax.set_title('(a) predicted live-gate density vs temperature\n'
             'computed from each model\'s measured greedy margins\n'
             '(teacher-forced pass, no loop run needed)', fontsize=9)
ax.legend(fontsize=7)

ax = axes[1]
for k in sorted(data):
    m = data[k]['m']
    ecdf = np.sort(m)
    ax.plot(ecdf, np.linspace(0, 1, len(ecdf)), lw=1.8, color=COL[k],
            label=data[k]['label'])
ax.set_xscale('log')
ax.set_xlim(1e-4, 30)
ax.set_xlabel('margin $m_t$ (nats)')
ax.set_ylabel('ECDF over all greedy-orbit steps')
ax.set_title('(b) the margin landscapes that generate panel (a)\n'
             'the lower the tail, the earlier the loop destabilises', fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_live_density_models.png'), dpi=150)
plt.close(fig)

table = {}
for k, v in data.items():
    G, m = v['G'], v['m']
    table[k] = dict(label=v['label'], n_runs=v['n_runs'], n_steps=int(len(m)),
                    G_median=float(np.median(G)), G_iqr=[float(np.percentile(G, 25)),
                                                          float(np.percentile(G, 75))],
                    G_p05=float(np.percentile(G, 5)), G_p95=float(np.percentile(G, 95)),
                    m_median=float(np.median(m)), m_min=float(m.min()),
                    m_p01=float(np.percentile(m, 1)),
                    frac_m_lt_01=float((m < 0.1).mean()),
                    frac_m_lt_03T_at_T03=float((m < 0.09).mean()))
with open(os.path.join(OUT, 'sl2_gates_table.json'), 'w') as f:
    json.dump(table, f, indent=1)
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items() if kk != 'G'}
                  for k, v in table.items()}, indent=1))
print('saved sl2_gate_geometry_models.png, sl2_live_density_models.png, json')
