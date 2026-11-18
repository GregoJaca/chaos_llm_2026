import time
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2')
from common import *

X = load_traj('cyber')

x = X[:, 0]
print('PE coord0 m5:', perm_entropy_fast(x, 5, 1))
print('PE shuffled  m5:', perm_entropy_fast(time_shuffle(X)[:, 0], 5, 1))
print('PE coord0 m6 tau3:', perm_entropy_fast(x, 6, 3))
t0 = time.time()
y = iaaft(x, niter=30)
print('iaaft ok', f'({time.time()-t0:.1f}s)', 'PE iaaft:', perm_entropy_fast(y, 5, 1))

t0 = time.time()
sk = knn_predict_skill(X, horizons=[1, 4, 16], k=12)
print('knn skill raw:', sk, f'({time.time()-t0:.1f}s)')

t0 = time.time()
pr = phase_rand(X)
print('phase_rand ok', f'({time.time()-t0:.1f}s)')
print('knn skill phase:', knn_predict_skill(pr, horizons=[1, 4, 16], k=12))

t0 = time.time()
ou = ou_null(X)
print('ou ok', f'({time.time()-t0:.1f}s)')
print('knn skill OU:', knn_predict_skill(ou, horizons=[1, 4, 16], k=12))
print('knn skill shuffled:', knn_predict_skill(time_shuffle(X), horizons=[1, 4, 16], k=12))

cg = cov_gauss(X)
print('knn skill covGauss:', knn_predict_skill(cg, horizons=[4], k=12))

# median dist of surrogates for reference
for nm, Z in [('raw', X), ('phase', pr), ('ou', ou), ('covGauss', cg)]:
    print(nm, 'median pairwise dist:', round(median_pairwise_dist(Z), 1))
