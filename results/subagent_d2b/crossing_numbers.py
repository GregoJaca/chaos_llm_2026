"""Numbers for the report: exact eps where C crosses key levels, local slope
there (finite-difference between bracketing grid points), rise width in
decades, and the ER-legal window width. Computed on the reference variant
(centered Euclid, f64) for the four dynamics trajectories + first layer.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cshape import (load_traj, median_dist, c_curve, slope_curve, fit_d2,
                    plateau_scan, jsonable)

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
TARGETS = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

for name in ['interstellar', 'launch', 'cyber', 'first']:
    X = load_traj(name)
    n = len(X)
    med = median_dist(X, 'center')
    eps = np.geomspace(0.001 * med, 5.0 * med, 96)
    C, npair, _ = c_curve(X, eps, arith='f64', mode='center')
    em, sl = slope_curve(eps, C, win=7)
    r = {'n': int(n), 'scale': float(med), 'npairs': int(npair), 'cross': {}, 'fits': {}}
    for tg in TARGETS:
        k = np.argmax(C >= tg)
        if C[k] < tg:
            r['cross'][str(tg)] = None
            continue
        if k == 0:
            sl_at = np.nan
        else:
            # local slope between grid points bracketing the crossing
            sl_at = (np.log(C[k]) - np.log(C[k - 1])) / (np.log(eps[k]) - np.log(eps[k - 1]))
        r['cross'][str(tg)] = {'eps_over_med': float(eps[k] / med),
                               'slope': None if not np.isfinite(sl_at) else float(sl_at)}
    # width of the rise C in [0.1, 0.9] in decades
    k1 = np.argmax(C >= 0.1); k9 = np.argmax(C >= 0.9)
    r['rise_width_dec'] = float(np.log10(eps[k9] / eps[k1])) if C[k9] >= 0.9 else None
    # ER-legal window: slope <= 2 log10 N and C in [10/N, 0.5]
    Cmid = np.interp(np.log(em), np.log(eps), C)
    good = np.isfinite(sl) & (sl <= 2 * np.log10(n)) & (Cmid >= 10.0 / n) & (Cmid <= 0.5)
    if good.any():
        idx = np.where(good)[0]
        w = np.log10(em[idx[-1]] / em[idx[0]])
        r['er_legal_window'] = {'width_dec': float(w),
                                'eps0': float(em[idx[0]] / med), 'eps1': float(em[idx[-1]] / med),
                                'slope_min': float(sl[idx].min()), 'slope_max': float(sl[idx].max()),
                                'npoints': int(len(idx))}
    else:
        r['er_legal_window'] = None
    # best r2 power-law fit in ANY contiguous 1-decade eps window (favorable!)
    best = None
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            if np.log10(eps[j] / eps[i]) < 1.0:
                continue
            m = (eps >= eps[i]) & (eps <= eps[j]) & (C > 0)
            if m.sum() < 8:
                continue
            if C[m].max() > 0.9 or C[m].min() < 10.0 / n:
                continue
            d2, r2, cnt = fit_d2(eps, C, C[m].min(), C[m].max())
            if r2 is not None and np.isfinite(r2) and (best is None or r2 > best[1]):
                best = (float(d2), float(r2), float(eps[i] / med), float(eps[j] / med), int(cnt))
    r['best_1dec_fit'] = best
    res[name] = r
    print('==', name, 'n', n, 'rise(0.1->0.9)=%.2f dec' % (r['rise_width_dec'] or float('nan')))
    for tg in TARGETS:
        cc = r['cross'][str(tg)]
        if cc:
            print('   C=%s at eps=%.3f med, slope there = %.2f' % (tg, cc['eps_over_med'], cc['slope'] if cc['slope'] else -1))
    print('   ER-legal window:', r['er_legal_window'])
    print('   best any-window 1-decade r2 fit:', best)

with open(os.path.join(OUT, 'numbers_shape.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved')
