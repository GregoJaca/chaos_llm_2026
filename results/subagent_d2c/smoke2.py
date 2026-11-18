import numpy as np
from common_d2c import *

X, ids = load("interstellar_travel")
Xc = X - X.mean(0)
steps = np.linalg.norm(np.diff(Xc, axis=0), axis=1)
D = distm(Xc)
iu = np.triu_indices(1000, 1)
dv = D[iu]
med = np.median(dv)
print("step:  med %.2f  q10 %.2f  q90 %.2f  max %.2f   (med pair dist %.2f)" %
      (np.median(steps), np.quantile(steps, .1), np.quantile(steps, .9), steps.max(), med))
print("step/med pair dist:", round(float(np.median(steps) / med), 4))
# is the smallest distance a consecutive pair?
ii, jj = np.unravel_index(np.argmin(D + np.eye(1000) * 1e9), D.shape)
print("closest pair: (%d,%d) lag %d dist %.3f" % (ii, jj, abs(jj - ii), D[ii, jj]))
lags = np.abs(iu[0] - iu[1])
for laglo, laghi in [(1, 1), (2, 10), (11, 50), (51, 200), (201, 999)]:
    m = (lags >= laglo) & (lags <= laghi)
    print("lag %4d-%4d: median dist %.2f (%.3f x med)" % (laglo, laghi, np.median(dv[m]), np.median(dv[m]) / med))
# norm autocorrelation
n = np.linalg.norm(Xc, axis=1)
ac = np.correlate(n - n.mean(), n - n.mean(), "full")[999:]
ac /= ac[999]
print("norm ACF lags 1,2,5,10,50,100:", np.round(ac[[1, 2, 5, 10, 50, 100]], 3))
# PC1 ACF
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]
ac1 = np.correlate(pc1 - pc1.mean(), pc1 - pc1.mean(), "full")[999:] / (np.linalg.norm(pc1 - pc1.mean()) ** 2)
print("PC1 ACF lags 1,2,5,10,50,100:", np.round(ac1[[1, 2, 5, 10, 50, 100]], 3))
print("PC1 var frac:", round(float(sv[0] ** 2 / (sv ** 2).sum()), 4))
print("norm variability CV:", round(float(np.std(np.linalg.norm(Xc, axis=1)) / np.mean(np.linalg.norm(Xc, axis=1))), 4))
print("tokens: unique", len(np.unique(ids)), "/1000")
