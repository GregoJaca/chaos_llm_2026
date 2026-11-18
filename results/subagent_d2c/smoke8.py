import numpy as np, time
from common_d2c import load, iaaft_scalar, rng_for
from exp5_delay import embed, d2_of

t0 = time.time()
X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
pc1 = U[:, 0] * sv[0]; pc1 = pc1 - pc1.mean()
acf = np.correlate(pc1, pc1, "full")[999:] / (pc1 @ pc1)
print("PC1 ACF lag1:", round(float(acf[1]), 3))
ia = [iaaft_scalar(pc1, rng_for("interstellar_travel", 70 + s)) for s in range(5)]
for m in range(2, 11):
    ed = d2_of(embed(pc1, m))
    ei = [d2_of(embed(z, m))["d2"] for z in ia]
    print(f"m={m:2d}  D2={ed['d2']:5.2f}  IAAFT={np.round(ei,2)}  run={ed['run']['width_dec'] if ed['run'] else None}"
          f" slope={ed['run']['slope'] if ed['run'] else None}")
print(f"{time.time()-t0:.1f}s")
