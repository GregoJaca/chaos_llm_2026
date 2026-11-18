import numpy as np, time
from common_d2c import *

t0 = time.time()
X, ids = load("interstellar_travel")
Xc = X - X.mean(0)
D = distm(Xc)
iu = np.triu_indices(1000, 1)
dv = D[iu]
med = float(np.median(dv))
print("med dist", round(med, 2), "min", round(float(dv.min()), 4), "IQR/med",
      round((np.quantile(dv, .75) - np.quantile(dv, .25)) / med, 3))

eps = eps_grid_from_med(med)
C = c_eps_curve(D, eps, 0)
s = log_slope(eps, C); ss = smooth3(s)
run = widest_run(eps, C, ss)
print("run:", run)
print("er_tot, er_best:", erlegal_width(eps, C, ss, 1000))
nbf = naive_best_fit(eps, C)
print("naive fit:", nbf)
print("slope at C=0.1:", ss[np.searchsorted(C, 0.1)], "at C=0.5:", ss[np.searchsorted(C, 0.5)])

# shuffle identity in integer counts
rng = np.random.default_rng(0)
cnt = np.searchsorted(np.sort(dv), eps, side="right")
for _ in range(3):
    perm = rng.permutation(1000)
    Ds = distm(Xc[perm])
    cnts = np.searchsorted(np.sort(pair_values(Ds, 0)), eps, side="right")
    print("shuffle int dev:", int(np.max(np.abs(cnt - cnts))))

# RQA + run-decomposition assertion
t1 = time.time()
rqa = rqa_from_D(D)
print("rqa:", rqa, f"({time.time()-t1:.2f}s)")

# surrogates in PC space (smoke: 50 PCs)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
Y = U[:, :50] * sv[:50]
Zp = phase_surrogate(Y, np.random.default_rng(1))
C1 = (Y[:-1].T @ Y[1:]) / 999
C1p = (Zp[:-1].T @ Zp[1:]) / 999
print("phase crosscov rel err:", float(np.linalg.norm(C1p - C1) / np.linalg.norm(C1)))
t1 = time.time()
Zi = miiaft(Y, np.random.default_rng(2), n_iter=20)
C1i = (Zi[:-1].T @ Zi[1:]) / 999
print("miiaft(20) crosscov rel err:", float(np.linalg.norm(C1i - C1) / np.linalg.norm(C1)))
print("miiaft marginal max dev:", float(np.max(np.abs(np.sort(Zi, 0) - np.sort(Y, 0)))))
# per-column spectrum check
Yf = np.fft.rfft(Y, axis=0); Zf = np.fft.rfft(Zi, axis=0)
print("miiaft spectrum rel err (med col):", float(np.median(np.abs(np.abs(Zf) - np.abs(Yf)).sum(0) / np.abs(Yf).sum(0))))
y = Y[:, 0]
zi = iaaft_scalar(y, np.random.default_rng(3))
print("scalar iaaft marginal exact:", np.allclose(np.sort(zi), np.sort(y)),
      "spectrum rel err:", float(np.abs(np.abs(np.fft.rfft(zi)) - np.abs(np.fft.rfft(y))).sum() / np.abs(np.fft.rfft(y)).sum()))
print(f"total {time.time()-t0:.1f}s")
