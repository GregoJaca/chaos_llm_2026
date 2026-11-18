"""Exact duplicate audit for the first layer (and all trajectories)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cshape import load_traj, median_dist, c_curve

for name in ['first', 'interstellar', 'launch', 'cyber']:
    X = load_traj(name)
    X64 = X.astype(np.float64)
    _, cnt = np.unique(X64, axis=0, return_counts=True)
    mult = cnt[cnt > 1]
    zero_pairs = int((mult * (mult - 1) // 2).sum())
    npairs = len(X) * (len(X) - 1) // 2
    med = median_dist(X, 'center')
    eps = np.array([1e-9 * med, 1e-6 * med, 1e-4 * med])
    C, npair, cnts = c_curve(X, eps, arith='f64', mode='center')
    print('%s: N=%d dup-rows=%d top-multiplicities=%s zero-pairs=%d (%.4f of pairs) '
          'C(floor) measured=%.4f' % (
              name, len(X), int((cnt > 1).sum()),
              sorted(mult.tolist(), reverse=True)[:5], zero_pairs,
              zero_pairs / npairs, C[0]))
