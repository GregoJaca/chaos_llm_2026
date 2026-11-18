"""Exp0: repetition-loop detection across all trajectories.

For each trajectory: state-distance vs lag L, best period L*, repeat fidelity,
drift across cycles, pairwise-distance histogram (bimodality), cluster mass.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from common import _block_dist_sq

OUT = os.path.dirname(os.path.abspath(__file__))
TR = ['cyber', 'future', 'great', 'interstellar', 'launch', 'first']
res = {}

def state_dist_lag_curve(X, lags):
    """median_t ||x_t - x_{t+L}|| / cloud scale for each L."""
    Xd = X.astype(np.float32)
    n = len(Xd)
    out = np.full(len(lags), np.nan)
    p10 = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        D = np.linalg.norm(Xd[L:] - Xd[:-L], axis=1)
        out[i] = np.median(D)
        p10[i] = np.quantile(D, 0.1)
    return out, p10

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
for ti, name in enumerate(TR):
    t0 = time.time()
    X = load_traj(name)
    n = len(X)
    scale = median_pairwise_dist(X)
    step = float(np.median(np.linalg.norm(np.diff(X.astype(np.float64), axis=0), axis=1)))
    lags = np.arange(2, min(600, n // 4))
    med, p10 = state_dist_lag_curve(X, lags)
    # best lag by 10th percentile (best repeats), require depth
    i_best = int(np.argmin(p10))
    Lstar = int(lags[i_best])
    d_at = np.linalg.norm(X[Lstar:].astype(np.float64) - X[:-Lstar].astype(np.float64), axis=1)
    fidelity = float(np.median(d_at) / scale)
    frac5 = float((d_at < 0.05 * scale).mean())
    # drift across cycles at L*
    kmax = int((n - 1) / max(Lstar, 1))
    drift = []
    for k in range(1, min(kmax, 60)):
        D = np.linalg.norm(X[k * Lstar:].astype(np.float64) - X[:-k * Lstar].astype(np.float64), axis=1)
        drift.append(float(np.median(D) / scale))
    # pairwise-distance histogram on a subsample (block)
    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=min(1500, n), replace=False)
    D = _block_dist_sq(X, idx, X[idx])
    iu = np.triu_indices(len(idx), 1)
    dh = np.sqrt(D[iu])
    # cluster mass: C(eps) at fractions of scale
    fr = {}
    for f in (0.02, 0.05, 0.1, 0.2):
        cnt, npair = d2_counts(X, np.array([f * scale]))
        fr[f] = float(cnt[0] / npair)
    res[name] = {'n': int(n), 'scale': float(scale), 'step_median': step,
                 'step_over_scale': float(step / scale),
                 'L_star': Lstar, 'p10_dist_over_scale': float(p10[i_best] / scale),
                 'fidelity_median': fidelity, 'frac_within_5pct_at_L': frac5,
                 'drift_k': drift[:40],
                 'C_at_frac_scale': fr,
                 'lag_curve_med_over_scale': (med / scale)[:: max(1, len(med) // 200)].tolist(),
                 'lag_p10_over_scale': (p10 / scale)[:: max(1, len(p10) // 200)].tolist(),
                 'lag_sub': lags[:: max(1, len(lags) // 200)].tolist()}
    print(f'{name}: n={n} scale={scale:.1f} step={step:.1f} ({step/scale:.2f} scale) '
          f'L*={Lstar} p10={p10[i_best]/scale:.3f} med={fidelity:.3f} frac5={frac5:.3f} '
          f'C(0.05s)={fr[0.05]:.4f} ({time.time()-t0:.0f}s)')

    ax = axes.flat[ti]
    ax.plot(lags[:: max(1, len(lags) // 200)], (med / scale)[:: max(1, len(med) // 200)],
            lw=1.0, label='median')
    ax.plot(lags[:: max(1, len(lags) // 200)], (p10 / scale)[:: max(1, len(p10) // 200)],
            lw=1.0, label='10th pct')
    ax.set_xscale('log')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_title(f'{name}  L*={Lstar}, fidelity={fidelity:.2f}')
    ax.set_xlabel('lag L (tokens)'); ax.set_ylabel(r'$\|x_t-x_{t+L}\|$ / cloud scale')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_loops_scan.png'), dpi=130)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ti, name in enumerate(TR):
    r = res[name]
    ax = axes.flat[ti]
    ax.plot(np.arange(1, len(r['drift_k']) + 1), r['drift_k'], 'o-', ms=3)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_title(f"{name}: drift at L*={r['L_star']}")
    ax.set_xlabel('cycle k'); ax.set_ylabel(r'median $\|x_t-x_{t+kL}\|$/scale')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_loops_drift.png'), dpi=130)
plt.close()

with open(os.path.join(OUT, 'exp0_loops.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved exp0_loops.json')
