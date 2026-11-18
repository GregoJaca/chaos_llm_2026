"""check_data: quantify every candidate driver of C(eps) shape.

- anisotropy ||mu|| vs cloud scale  (does centering/cosine matter?)
- distance-distribution spread (high-d concentration -> steep sigmoid)
- exact duplicates (first layer / loops)
- fp32-Gram-on-uncentered vs fp64-Gram-on-centered discrepancy
- autocorrelation of the scalar observables a notebook would embed
- cloud scales in ABSOLUTE units (fixed eps grids across prompts?)
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cshape import (load_traj, transform, median_dist, dist_stats, observables,
                    acf, pair_counts, RNG, LOOPS, jsonable)

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}

for name in ['interstellar', 'launch', 'cyber', 'first']:
    t0 = time.time()
    X = load_traj(name)
    n, d = X.shape
    X64 = X.astype(np.float64)
    mu = X64.mean(0)
    munorm = float(np.linalg.norm(mu))
    row_norms = np.linalg.norm(X64, axis=1)
    med_raw = median_dist(X, 'raw')
    med_ctr = median_dist(X, 'center')
    ds = dist_stats(X, 'center')

    # exact duplicates
    Xv = X64.view([('', np.float32)] * d).ravel()  # structured compare
    try:
        _, cnt = np.unique(Xv, return_counts=True)
        ndup_rows = int((cnt > 1).sum())
        dup_extra_pairs = int(((cnt * (cnt - 1)) // 2).sum())
    except Exception:
        ndup_rows, dup_extra_pairs = -1, -1

    # fp32-uncentered-Gram vs fp64-centered: distance discrepancy on a subsample
    idx = np.sort(RNG.choice(n, size=min(1500, n), replace=False))
    A32 = X[idx].astype(np.float32)
    G32 = A32 @ A32.T
    sq32 = (A32.astype(np.float64) ** 2).sum(1)
    D32 = np.maximum(sq32[:, None] + sq32[None, :] - 2 * G32.astype(np.float64), 0)
    Zc = transform(X, 'center')[idx]
    G64 = Zc @ Zc.T
    sq64 = (Zc ** 2).sum(1)
    D64 = np.maximum(sq64[:, None] + sq64[None, :] - 2 * G64, 0)
    iu = np.triu_indices(len(idx), 1)
    d32, d64 = np.sqrt(D32[iu]), np.sqrt(D64[iu])
    absdiff = np.abs(d32 - d64)
    relerr = absdiff / np.maximum(d64, 1e-12)
    # where does fp32 error become comparable to the distance itself?
    m = d64 > 0
    order = np.argsort(d64)
    thr = None
    for frac in [0.001, 0.005, 0.01, 0.02, 0.05]:
        k = max(1, int(frac * m.sum()))
        med_smallest = np.median(d64[order[:k]])
        res_smallest = np.median(relerr[order[:k]])
        res.setdefault(name, {}).setdefault('fp32_floor', {})[str(frac)] = {
            'med_dist_of_smallest_frac': float(med_smallest),
            'med_relerr_there': float(res_smallest),
            'med_dist_over_scale': float(med_smallest / med_ctr)}

    r = {
        'n': int(n), 'd': int(d), 'T': int(n),
        'row_norm_med': float(np.median(row_norms)),
        'row_norm_iqr': float(np.quantile(row_norms, .75) - np.quantile(row_norms, .25)),
        'mu_norm': munorm,
        'med_pairwise_raw': float(med_raw), 'med_pairwise_centered': float(med_ctr),
        'mu_over_scale': munorm / med_ctr,
        'dist_stats_centered': ds,
        'dup_rows': ndup_rows, 'dup_extra_pairs': dup_extra_pairs,
        'fp32_vs_f64_max_abs': float(absdiff.max()),
        'fp32_vs_f64_max_rel': float(relerr.max()),
        'fp32_vs_f64_rel_at_median': float(np.median(relerr)),
    }
    if name in LOOPS:
        P = LOOPS[name]
        dd = np.linalg.norm(X64[P:] - X64[:-P], axis=1)
        r['loop_P'] = P
        r['lagP_dist_over_scale'] = float(np.median(dd) / med_ctr)
    # observables
    obs = observables(X)
    r['acf'] = {}
    for oname, s in obs.items():
        r['acf'][oname] = acf(s, 10)[1:]
        r.setdefault('obs_std', {})[oname] = float(np.std(s))
    res.setdefault(name, {}).update(r)
    print(name, 'n', n, '||mu||/scale', round(r['mu_over_scale'], 2),
          'iqr/med', round(ds['iqr_over_med'], 3), 'duprows', ndup_rows,
          'fp32rel@med', f"{r['fp32_vs_f64_rel_at_median']:.2e}",
          f"{time.time()-t0:.0f}s")
    print('   acf pc1[:5]', [round(v, 3) for v in r['acf']['pc1'][:5]],
          'norm[:5]', [round(v, 3) for v in r['acf']['norm'][:5]],
          'step[:5]', [round(v, 3) for v in r['acf']['stepdist'][:5]])

with open(os.path.join(OUT, 'check_data.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved check_data.json')
