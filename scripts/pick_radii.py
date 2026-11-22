"""Pick flip-zone radii from an a1i run: prints 0.5, 1, 3 x r* (one decimal-cap)."""
import sys
import numpy as np

z = np.load(sys.argv[1])
r_c = z["r_c"]  # [nr, K, T]
radii = z["radii"]
rstar = float(np.min(r_c))
lo = max(float(radii[0]), 1e-5)
out = []
for m in (0.5, 1.0, 3.0):
    r = max(lo, min(30.0, m * rstar))
    out.append(("%.4g" % r))
print(" ".join(sorted(set(out), key=float)))
