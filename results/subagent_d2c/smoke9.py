import numpy as np
from common_d2c import load, iaaft_scalar, rng_for, phase_surrogate
from exp5_delay import embed, d2_of

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]; pc1 = pc1 - pc1.mean()

for s in range(5):
    z = iaaft_scalar(pc1, rng_for("interstellar_travel", 70 + s))
    r = float(np.corrcoef(z, pc1)[0, 1])
    d = float(np.linalg.norm(z - pc1))
    ed = d2_of(embed(pc1, 2)); ez = d2_of(embed(z, 2))
    print(f"seed {70+s}: corr(z,data)={r:+.4f} ||z-data||={d:.1f}  D2data={ed['d2']:.3f} D2surr={ez['d2']:.3f}")

# control: time-shuffle (white noise, same marginal)
zsh = pc1[rng_for("interstellar_travel", 99).permutation(pc1.size)]
print("shuffle: D2 =", round(d2_of(embed(zsh, 2))["d2"], 3),
      " corr:", round(float(np.corrcoef(zsh, pc1)[0, 1]), 4))
# control: pure gaussian noise
zg = rng_for("interstellar_travel", 98).standard_normal(pc1.size)
print("gauss:   D2 =", round(d2_of(embed(zg, 2))["d2"], 3))
# check the ACF of a surrogate
z = iaaft_scalar(pc1, rng_for("interstellar_travel", 70))
acf = np.correlate(z, z, "full")[999:] / (z @ z)
acfd = np.correlate(pc1, pc1, "full")[999:] / (pc1 @ pc1)
print("ACF lag1 data vs surr:", round(float(acfd[1]), 4), round(float(acf[1]), 4))
print("ACF lag10 data vs surr:", round(float(acfd[10]), 4), round(float(acf[10]), 4))
