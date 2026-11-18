import numpy as np
import common2 as C

for nm in ["w100", "w200", "w500", "cyber"]:
    c = C.load_case(nm)
    rcS = c.rc_signed()[0]; rcA = c.rc_adv()[0]
    rcS[~np.isfinite(rcS)] = 1e30; rcA[~np.isfinite(rcA)] = 1e30
    rS, rA = rcS.min(axis=1), rcA.min(axis=1)
    wolf = int((rA < 0.97 * rS).sum())
    print(f"{nm}: sign-blind cries wolf on {wolf}/{c.K} directions "
          f"(adv r_* below signed r_*); median adv/signed ratio "
          f"{np.median(rA / rS):.2f}, max gap {np.max(rS / rA):.1f}x")
