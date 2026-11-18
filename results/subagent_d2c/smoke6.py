import numpy as np
from common_d2c import *

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
D = distm(Xc)
med = float(np.median(pair_values(D, 0)))
eps = eps_grid_from_med(med)
C = c_eps_curve(D, eps, 0)
s = smooth3(log_slope(eps, C))
print("N=1000, w=0; ceiling = 2*log10(N) =", round(2 * np.log10(1000), 3))
print("   eps/med      C        slope   legal(C in [.01,.5] & slope<=6)")
for k in range(len(eps)):
    if C[k] >= 3e-3:
        legal = (C[k] >= 10 / 1000) and (C[k] <= 0.5) and np.isfinite(s[k]) and (s[k] <= 2 * np.log10(1000))
        print(f"  {eps[k]/med:8.4f}  {C[k]:9.5f}  {s[k]:7.3f}   {legal}")
