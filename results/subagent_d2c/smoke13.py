import numpy as np
from common_d2c import load, phase_surrogate, rng_for, _common_phases

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, Vh = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]; pc1 = pc1 - pc1.mean()

for seed in (900, 901, 902, 903):
    rng = rng_for("interstellar_travel", seed)
    ph = _common_phases(1000, rng)
    rng2 = rng_for("interstellar_travel", seed)
    z1 = phase_surrogate(pc1[:, None], rng2)[:, 0]
    # inline, same phases consumed in same order
    rng3 = rng_for("interstellar_travel", seed)
    Yf = np.fft.rfft(pc1[:, None], axis=0)
    rot = np.exp(1j * _common_phases(1000, rng3))
    Zf = Yf * rot[:, None]
    Zf[:, -1] = Yf[:, -1] * rng3.choice([-1.0, 1.0])
    z2 = np.fft.irfft(Zf, n=1000, axis=0)[:, 0]
    print(f"seed {seed}: common corr {float(np.corrcoef(z1, pc1)[0, 1]):+.4f} | "
          f"inline corr {float(np.corrcoef(z2, pc1)[0, 1]):+.4f} | "
          f"same result? {np.allclose(z1, z2)} | z1==pc1? {np.allclose(z1, pc1)}")
