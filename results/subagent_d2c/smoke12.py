import numpy as np
from common_d2c import load, rng_for, _common_phases
from scipy.fft import rfft, irfft

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]; pc1 = pc1 - pc1.mean()

rng = rng_for("interstellar_travel", 900)
ph = _common_phases(1000, rng)
print("ph[0:6] =", np.round(ph[:6], 3), " unique count:", len(np.unique(np.round(ph, 6))))
Y = pc1[:, None]
Yf = np.fft.rfft(Y, axis=0)
print("Yf shape:", Yf.shape, "dtype:", Yf.dtype)
rot = np.exp(1j * _common_phases(1000, rng_for("interstellar_travel", 901)))
print("rot[1:4]:", np.round(rot[1:4], 3))
Zf = Yf * rot[:, None]
print("Zf shape:", Zf.shape)
z = np.fft.irfft(Zf, n=1000, axis=0)[:, 0]
print("corr:", round(float(np.corrcoef(z, pc1)[0, 1]), 4))
print("z==pc1?", np.allclose(z, pc1), " max|diff|:", float(np.abs(z - pc1).max()))

# what does np.fft.rfft do with axis=0 on (1000,1)?
a = np.fft.rfft(Y, axis=0)[:, 0]
b = np.fft.rfft(pc1)
print("rfft(axis=0)[:,0] == rfft(1d)?", np.allclose(a, b))

# irfft of the Yf itself (no rotation)
z0 = np.fft.irfft(Yf, n=1000, axis=0)[:, 0]
print("irfft(rfft) identity:", np.allclose(z0, pc1))
