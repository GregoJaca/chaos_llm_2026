import time
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2')
from common import *

X = load_traj('cyber')
print('loaded', X.shape)
t0 = time.time()
med = median_pairwise_dist(X)
print(f'median pairwise dist: {med:.2f}  ({time.time()-t0:.1f}s)')

t0 = time.time()
eps = eps_grid_for(X)
print('eps grid', eps[0], eps[-1], f'({time.time()-t0:.1f}s)')

t0 = time.time()
C, npairs = c_curve(X, eps, theiler=0)
print(f'C curve theiler=0: {time.time()-t0:.1f}s, npairs={npairs}')
print('C at grid:', np.round(C[::6], 4))
lc = local_slopes(eps, C)
print('slopes at C=.05,.1,.3,.5,.7,.9:', slope_at_C(eps, C))
d2, r2 = d2_fit(eps, C)
print('fit d2 (C in [.1,.9]):', d2, r2)

t0 = time.time()
C2, _ = c_curve(X, eps, theiler=100)
print(f'theiler=100: {time.time()-t0:.1f}s  slopes:', slope_at_C(eps, C2))
d2b, _ = d2_fit(eps, C2)
print('fit d2 theiler=100:', d2b)

t0 = time.time()
print('twonn:', twonn(X, theiler=0), f'({time.time()-t0:.1f}s)')
t0 = time.time()
print('twonn theiler=100:', twonn(X, theiler=100), f'({time.time()-t0:.1f}s)')
t0 = time.time()
print('mle:', mle_dim(X, ks=(4, 8, 16)), f'({time.time()-t0:.1f}s)')
t0 = time.time()
print('pca:', {k: v for k, v in pca_stats(X).items() if k != 'eig'}, f'({time.time()-t0:.1f}s)')

# shuffled must give IDENTICAL statics
Xs = time_shuffle(X)
t0 = time.time()
Cs, _ = c_curve(Xs, eps, theiler=0)
print('shuffled C == raw C ?', np.allclose(C, Cs, atol=1e-12), f'({time.time()-t0:.1f}s)')
t0 = time.time()
print('shuffled twonn:', twonn(Xs), f'({time.time()-t0:.1f}s)')

x = X[:, 0]
t0 = time.time()
print('PE coord0:', perm_entropy_fast(x, 5, 1), f'({time.time()-t0:.2f}s)')
t0 = time.time()
print('PE shuffled:', perm_entropy_fast(time_shuffle(X)[:, 0], 5, 1), f'({time.time()-t0:.2f}s)')
t0 = time.time()
y = iaaft(x, niter=30)
print('iaaft one coord ok', y.shape, f'({time.time()-t0:.1f}s)')
print('PE iaaft:', perm_entropy_fast(y, 5, 1))

t0 = time.time()
sk = knn_predict_skill(X, horizons=[1, 8], k=12)
print('knn skill:', sk, f'({time.time()-t0:.1f}s)')

t0 = time.time()
pr = phase_rand(X)
print('phase_rand ok', pr.shape, f'({time.time()-t0:.1f}s)')
print('phase_rand knn:', knn_predict_skill(pr, horizons=[8], k=12))
