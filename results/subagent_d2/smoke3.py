import time, numpy as np, sys
sys.path.insert(0, r'C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2')
from common import *

X = load_traj('cyber')
n = len(X)
print('n =', n)

t0 = time.time()
cg = cov_gauss(X)
print('covGauss median dist:', round(median_pairwise_dist(cg), 1), '(raw 90)', f'{time.time()-t0:.1f}s')

t0 = time.time()
pr = phase_rand(X)
print('phase ok', f'{time.time()-t0:.1f}s')

t0 = time.time()
ms, q = miaaft(X, niter=10)
print('miaaft ok', f'{time.time()-t0:.1f}s', q)
print('miaaft median dist:', round(median_pairwise_dist(ms), 1))
# marginal check
from scipy import stats
ks_p = [stats.ks_2samp(X[:, c], ms[:, c]).pvalue for c in range(0, 1536, 100)]
print('marginal KS p (min, med):', round(min(ks_p), 3), round(float(np.median(ks_p)), 3))

H = [1, 2, 4, 8, 16, 32]
W = 2 * max(H) + 8
t0 = time.time()
nb_i, nb_d = knn_neighbors(X, k=20, W=W, test_slice=(n // 2, n - max(H) - 1))
print('neighbors cached', nb_i.shape, f'{time.time()-t0:.1f}s')

sk = predict_from_neighbors(X, nb_i, nb_d, H, k=12)
for h in H:
    print('h=%2d' % h, {k: round(v, 3) for k, v in sk[h].items()})

# repetition diagnosis: NN distance distribution vs cloud scale
r1 = nb_d[:, 0]
med = median_pairwise_dist(X)
print('cloud median dist:', round(med, 1))
print('NN dist: min/1%/5%/25%/50%:', [round(float(np.quantile(r1, q)), 2) for q in (0, .01, .05, .25, .5)])
print('frac NN < 1% of cloud scale:', float((r1 < 0.01 * med).mean()))

# temporal gaps of nearest neighbours
gap = np.abs(np.arange(len(nb_i))[:, None] + n // 2 - nb_i)  # test t is n//2 + row
g1 = gap[:, 0]
print('NN temporal gap: 1%/5%/25%/50%:', [int(np.quantile(g1, q)) for q in (.01, .05, .25, .5)])

t0 = time.time()
cc = continuation_curves(X, nb_i, nb_d, [1, 8, 16])
print('continuation h=8: r bins', np.round(np.array(cc[8]['bin_r']) / med, 4))
print('continuation h=8: rc bins', np.round(np.array(cc[8]['bin_rc']) / med, 4))
print('counts', cc[8]['counts'], f'{time.time()-t0:.1f}s')
for h in (1, 8, 16):
    ds, thr, cnt = determinism_score(X, nb_i, nb_d, h)
    print(f'determinism h={h}: median rc/r (closest 5%) = {ds:.3f}, thr={thr:.2f}, n={cnt}')

t0 = time.time()
rp = ridge_pcskill(X, H)
print('ridge:', {h: {k: round(v, 3) for k, v in rp[h].items()} for h in H}, f'{time.time()-t0:.1f}s')

# phase/miaaft prediction for comparison
nb_p, nb_pd = knn_neighbors(pr, k=20, W=W, test_slice=(n // 2, n - max(H) - 1))
skp = predict_from_neighbors(pr, nb_p, nb_pd, [8], k=12)
print('phase kNN h=8:', skp[8])
ds_p, _, _ = determinism_score(pr, nb_p, nb_pd, 8)
print('phase determinism h=8:', round(ds_p, 3))
nb_m, nb_md = knn_neighbors(ms, k=20, W=W, test_slice=(n // 2, n - max(H) - 1))
skm = predict_from_neighbors(ms, nb_m, nb_md, [8], k=12)
print('miaaft kNN h=8:', skm[8])
ds_m, _, _ = determinism_score(ms, nb_m, nb_md, 8)
print('miaaft determinism h=8:', round(ds_m, 3))

# lag autocorrelation of vector norm -> loop period check
xn = np.linalg.norm(X - X.mean(0), axis=1)
ac = np.correlate(xn - xn.mean(), xn - xn.mean(), 'full')[len(xn) - 1:] / (np.var(xn) * len(xn))
top = np.argsort(ac[10:400])[::-1][:8] + 10
print('norm autocorr top lags:', top.tolist(), 'vals:', np.round(ac[top], 3).tolist())
