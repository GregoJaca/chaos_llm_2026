import numpy as np
from common_d2c import load, phase_surrogate, rng_for
from scipy.fft import rfft, irfft

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]; pc1 = pc1 - pc1.mean()

Yf = np.abs(rfft(pc1))
tot = (Yf ** 2).sum()
print("power in freqs 0-3:", round(float((Yf[:4] ** 2).sum() / tot), 3),
      " 0-10:", round(float((Yf[:11] ** 2).sum() / tot), 3),
      " max freq share:", round(float((Yf ** 2).max() / tot), 4),
      " argmax freq:", int(np.argmax(Yf)))

for s in range(4):
    z = phase_surrogate(pc1[:, None], rng_for("interstellar_travel", 900 + s))[:, 0]
    print(f"seed {s}: corr(phase_surrogate, data) = {float(np.corrcoef(z, pc1)[0, 1]):+.4f} "
          f"||z||/||y|| = {np.linalg.norm(z)/np.linalg.norm(pc1):.4f}")
    zc = np.corrcoef(z, pc1)[0, 1]
# reference: AR(1)
rng = np.random.default_rng(1)
ar = np.empty(1000); ar[0] = 0
for t in range(1, 1000):
    ar[t] = 0.3 * ar[t - 1] + rng.standard_normal()
ar -= ar.mean()
for s in range(3):
    z = phase_surrogate(ar[:, None], np.random.default_rng(s))[:, 0]
    print(f"AR(1) ref: corr = {float(np.corrcoef(z, ar)[0, 1]):+.4f}")
# manual phase randomization for comparison
ph = np.random.default_rng(5).uniform(0, 2 * np.pi, 501)
ph[0] = 0; ph[-1] = 0
zf = rfft(pc1) * np.exp(1j * ph)
zm = irfft(zf, n=1000)
print("manual phase-rand corr:", round(float(np.corrcoef(zm, pc1)[0, 1]), 4))
