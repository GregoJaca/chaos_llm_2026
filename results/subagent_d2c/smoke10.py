import numpy as np
from common_d2c import load, phase_surrogate, rng_for, _rank_match
from scipy.fft import rfft, irfft

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
Y = U[:, :50] * sv[:50]
pc1 = Y[:, 0] - Y[:, 0].mean()

def _iaaft_once(y, rng, n_iter):
    ysorted = np.sort(y); T = y.size
    yf = rfft(y); amp = np.abs(yf)
    Z = phase_surrogate(y[:, None], rng)[:, 0]
    for _ in range(n_iter):
        order = np.argsort(Z, kind="stable")
        Z2 = np.empty_like(Z); Z2[order] = ysorted
        Zf = rfft(Z2)
        Zf = amp * np.exp(1j * np.angle(Zf))
        Zf[0] = yf[0]
        Z = irfft(Zf, n=T)
    order = np.argsort(Z, kind="stable")
    out = np.empty_like(Z); out[order] = ysorted
    return out

print("=== scalar IAAFT on PC1: corr with data vs n_iter ===")
for ni in (0, 5, 15, 40, 100):
    cs, serrs = [], []
    for s in range(5):
        z = _iaaft_once(pc1, rng_for("interstellar_travel", 500 + s), ni)
        cs.append(abs(float(np.corrcoef(z, pc1)[0, 1])))
        serrs.append(float(np.abs(np.abs(rfft(z)) - np.abs(rfft(pc1))).sum() / np.abs(rfft(pc1)).sum()))
    print(f"n_iter={ni:3d}  |corr| med {np.median(cs):.3f}  max {np.max(cs):.3f}   spectrum rel err med {np.median(serrs):.4f}")

print("=== multivariate: 40-iter MIAAFT vs one-shot phase+rankmatch (MAAFT) ===")
C1 = (Y[:-1].T @ Y[1:]) / 999
def crosserr(Z):
    return float(np.linalg.norm((Z[:-1].T @ Z[1:]) / 999 - C1) / np.linalg.norm(C1))
def colspecerr(Z):
    Yf = rfft(Y, axis=0); Zf = rfft(Z, axis=0)
    return float(np.median(np.abs(np.abs(Zf) - np.abs(Yf)).sum(0) / np.abs(Yf).sum(0)))
def maxcolcorr(Z):
    return float(np.max(np.abs([np.corrcoef(Z[:, k], Y[:, k])[0, 1] for k in range(Y.shape[1])])))

from common_d2c import miiaft
for s in range(3):
    Zi = miiaft(Y, rng_for("interstellar_travel", 600 + s), n_iter=40)
    Z1 = _rank_match(phase_surrogate(Y, rng_for("interstellar_travel", 700 + s)), np.sort(Y, axis=0))
    print(f"iter40: crosserr {crosserr(Zi):.3f} specerr {colspecerr(Zi):.4f} maxcolcorr {maxcolcorr(Zi):.3f} | "
          f"oneshot: crosserr {crosserr(Z1):.3f} specerr {colspecerr(Z1):.4f} maxcolcorr {maxcolcorr(Z1):.3f}")
