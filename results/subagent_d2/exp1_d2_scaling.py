"""Exp1: clean D2 scaling -- C(eps) curves, local slopes, N-dependence,
Theiler sweep (loop-aware), window-dependent D2 fits, shuffle identity.

Figure 1a: C(eps) for N=1024/2048/full (Theiler=0), shared normalized eps.
Figure 1b: local slopes, full N, Theiler variants.
JSON: exp1_d2_scaling.json
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
TR = ['cyber', 'future', 'great', 'interstellar', 'launch', 'first']
LOOPY = {'cyber': 53, 'future': 74, 'great': 69}

def theiler_list(name, n):
    if name in LOOPY:
        L = LOOPY[name]
        return [0, L, 3 * L, min(1000, n // 4)]
    if name == 'first':
        return [0, 32, 128]
    return [0, 150, 500]

res = {}
fig1, axes1 = plt.subplots(2, 3, figsize=(17, 9))
fig2, axes2 = plt.subplots(2, 3, figsize=(17, 9))

for name in TR:
    t0 = time.time()
    X = load_traj(name)
    n = len(X)
    scale = median_pairwise_dist(X)
    eps = np.geomspace(0.002 * scale, 2.2 * scale, 80)
    r = {'n': int(n), 'scale': float(scale), 'Ns': {}, 'theiler': {},
         'fits_full': {}, 'shuffle_identity': None}

    ax = axes1.flat[TR.index(name)]
    ax2 = axes2.flat[TR.index(name)]
    for N in sorted({1024, 2048, min(n, 4095)}):
        if N > n:
            continue
        idx = np.sort(RNG.choice(n, size=N, replace=False))
        C, npair = c_curve(X, eps, theiler=0, idx=idx)
        r['Ns'][N] = {'C': C.tolist(), 'npairs': int(npair)}
        ls = '-o' if N == min(n, 4095) else '-'
        ax.plot(eps / scale, C, ls, lw=1.4, ms=2.5, label=f'N={N}')
        if N == min(n, 4095):
            Cfull = C
            epsfull = eps

    # Theiler sweep at full N
    for w in theiler_list(name, n):
        C, npair = c_curve(X, eps, theiler=w)
        r['theiler'][w] = {'C': C.tolist(), 'npairs': int(npair)}
        lbl = f'w={w}'
        if w == 0:
            Cw0 = C
        ax2.plot((eps / scale)[:-1], local_slopes(eps, C), label=lbl, lw=1.2)

    # window-dependent fits (full N, w=0)
    for lo, hi in [(0.02, 0.1), (0.1, 0.3), (0.3, 0.9), (0.1, 0.9)]:
        d2, r2 = d2_fit(epsfull, Cfull, lo, hi)
        r['fits_full'][f'C{lo}-{hi}'] = {'d2': d2, 'r2': r2}

    # shuffle identity check (full N) -- overlay on fig1a
    Xs = time_shuffle(X)
    Cs, _ = c_curve(Xs, eps, theiler=0)
    ok = (Cfull > 0) & (Cs > 0)
    okbig = ok & (Cfull > 1e-4)
    r['shuffle_identity'] = {
        'max_abs_dlogC': float(np.max(np.abs(np.log(Cfull[ok]) - np.log(Cs[ok])))),
        'max_abs_dlogC_C_gt_1e-4': float(np.max(np.abs(np.log(Cfull[okbig]) - np.log(Cs[okbig]))))}
    ax.plot(eps / scale, Cs, 'k--', lw=1.0, alpha=0.7, label='time-shuffled')

    # slope-at-C crossings
    r['slope_at_C_w0'] = {str(k): v for k, v in slope_at_C(epsfull, Cfull).items()}

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ / cloud scale'); ax.set_ylabel(r'$C(\epsilon)$')
    ttl = name + (f'  (loop L={LOOPY[name]})' if name in LOOPY else '')
    ax.set_title(ttl)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\epsilon$ / cloud scale'); ax2.set_ylabel('local slope')
    ax2.set_title(ttl); ax2.legend(fontsize=7); ax2.grid(alpha=0.25)
    ax2.set_ylim(-0.5, 8)
    res[name] = r
    print(f'{name}: done {time.time()-t0:.0f}s  fits:',
          {k: round(v['d2'], 2) for k, v in r['fits_full'].items()},
          'shuffle max dlogC:', round(r['shuffle_identity']['max_abs_dlogC'], 6))

fig1.tight_layout(); fig1.savefig(os.path.join(OUT, 'fig1a_d2_scaling.png'), dpi=130); plt.close(fig1)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, 'fig1b_d2_slopes.png'), dpi=130); plt.close(fig2)
with open(os.path.join(OUT, 'exp1_d2_scaling.json'), 'w') as f:
    json.dump(jsonable(res), f)
print('saved')
