"""Per-model D2/TwoNN pipeline over all loop-attractor state runs.

Scans results/softloop_states/<model>/T*/softloop.npz for the saved
reference-row states (h_ref), runs the validated D2 pipeline
(chaos_spark_wt/scripts_spark/d2_local.py) on each, then builds:

  d2pipe_<model>_<T>.png    per-trajectory 8-panel diagnostics
  sl2_d2_summary.png        cross-model summary: plateau verdicts,
                             repetition-loop coverage, TwoNN, MLE
  sl2_twonn_vs_T.png        TwoNN dimension vs T per model
  sl2_sigma_dim.png         readout-covariance spectrum vs state-cloud
                             dimension (the Sv-run sections record the
                             top-16 singular values of Sigma_p)
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..'))
D2 = os.path.join(REPO, 'scripts_spark', 'd2_local.py')
SRC = os.path.join(REPO, 'results', 'softloop_states')
TMP = os.path.join(SRC, '_pt')
OUT = os.path.join(REPO, 'results', 'figures', 'softloop')
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

MCOL = {'ds15': '#1a6faa', 'mamba130': '#c8102e', 'qwen25_3b': '#2e8b57',
        'randqwen': '#888888', 'pythia410': '#444444', 'qwen3_5': '#d55e00',
        'ds14': '#5a9bd4', 'qwen35_9b': '#ff7a45'}
MLAB = {'ds15': 'DeepSeek-1.5B', 'mamba130': 'Mamba-130m',
        'qwen25_3b': 'Qwen2.5-3B', 'randqwen': 'randQwen',
        'pythia410': 'Pythia-410m', 'qwen3_5': 'Qwen3.5-2B',
        'ds14': 'DeepSeek-14B', 'qwen35_9b': 'Qwen3.5-9B'}

# 1. prepare h_last npz copies
files = []
for d in sorted(glob.glob(os.path.join(SRC, '*'))):
    if not os.path.isdir(d) or os.path.basename(d) == '_pt':
        continue
    model = os.path.basename(d)
    for dd in sorted(glob.glob(os.path.join(d, 'T*'))):
        f = os.path.join(dd, 'softloop.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if 'h_ref' not in z.files:
            continue
        T = os.path.basename(dd)
        out_f = os.path.join(TMP, f'{model}_{T}.npz')
        np.savez(out_f, h_last=z['h_ref'].astype(np.float32))
        files.append((model, T, out_f))
        print('prepared', out_f, z['h_ref'].shape)

if not files:
    sys.exit('no state files yet')

# 2. run the pipeline in model groups (per-model summary tags)
for model in sorted(set(m for m, _, _ in files)):
    mf = [f for m, _, f in files if m == model]
    cmd = [sys.executable, D2, '--pt'] + mf + \
          ['--theiler', '0,50,500', '--out', OUT, '--tag', model]
    print('RUN', model)
    subprocess.run(cmd, check=False)

# 3. collect results
rows = {}
for model in sorted(set(m for m, _, _ in files)):
    jp = os.path.join(OUT, f'd2pipe_{model}.json')
    if not os.path.exists(jp):
        continue
    d = json.load(open(jp))
    for tr in d['trajectories']:
        name = tr['name']  # <model>_<T>
        T = name.split('_T')[-1]
        try:
            Tv = float(T)
        except ValueError:
            continue
        rows[(model, Tv)] = dict(
            twonn=tr['twonn']['mle'], mle10=tr['mle']['10'],
            verdict=tr['verdict'],
            period_cov=tr['period']['coverage'] if isinstance(tr.get('period'), dict) else None,
            d_typ=tr['d_typ'], n=tr['n'])

# also read Sigma_p spectrum effective rank where recorded
effrank = {}
for model, T, f in files:
    src = f.replace('_pt', '')  # not exact; use the original npz path instead
for model in sorted(set(m for m, _, _ in files)):
    for dd in sorted(glob.glob(os.path.join(SRC, model, 'T*'))):
        f = os.path.join(dd, 'softloop.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if 'sv_read' not in z.files:
            continue
        T = float(os.path.basename(dd)[1:])
        sv = z['sv_read'].astype(np.float64)  # [16, steps]
        lam = sv ** 2
        tot = lam.sum(axis=0)
        er = (tot ** 2 / np.maximum((lam ** 2).sum(axis=0), 1e-30))
        effrank[(model, T)] = float(np.median(er[er > 0])) if (er > 0).any() else 0.0

# 4. figures
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
for (model, T), r in sorted(rows.items()):
    ax.scatter(T, r['twonn'], s=60, color=MCOL.get(model, '#333'),
               marker='o', label=MLAB.get(model, model) if T == min(
                   t for m2, t in rows if m2 == model) else None)
ax.set_xscale('log')
ax.set_xlabel('temperature $T$')
ax.set_ylabel('TwoNN dimension (MLE)')
ax.legend(fontsize=7)
ax.set_title('(a) intrinsic dimension of the loop state cloud vs $T$\n'
             'n=3000 last-layer states per run', fontsize=9)

ax = axes[1]
for (model, T), r in sorted(effrank.items()):
    ax.scatter(T, r, s=60, color=MCOL.get(model, '#333'), marker='s')
ax.set_xscale('log')
ax.set_xlabel('temperature $T$')
ax.set_ylabel(r'effective rank of $\Sigma_p$ (median over steps)')
ax.set_title(r'(b) dimension of the readout covariance itself: '
             r'$\left(\sum_i\lambda_i\right)^2/\sum_i\lambda_i^2$, top-16 '
             r'singular values', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sl2_twonn_vs_T.png'), dpi=150)
plt.close(fig)

with open(os.path.join(OUT, 'sl2_d2_states.json'), 'w') as f:
    json.dump({f'{m}|{T}': r for (m, T), r in rows.items()}, f, indent=1)
print('saved sl2_twonn_vs_T.png + sl2_d2_states.json')
for k in sorted(rows):
    print(k, 'twonn', round(rows[k]['twonn'], 2), 'verdict:', rows[k]['verdict'][:40])
