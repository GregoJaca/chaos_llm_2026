import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

D = C.load("w100")
dhm, radii, layers = D["dhm"], D["radii"], D["layers"]
large_idx = np.where(D["large"])[0]
lr = np.log(radii[large_idx])

print("alpha(l,t) over r in [0.5, 2]:")
ts = [0, 2, 5, 12, 30, 50, 80, 120, 200, 250, 299]
hdr = "layer | " + " | ".join(f"t={t:>3d}" for t in ts)
print(hdr)
for li, l in enumerate(layers):
    if l == 0:
        continue
    fl = D["flush"].get(int(l))
    vals = []
    for t in ts:
        if fl is not None and t >= fl:
            vals.append("  -- ")
            continue
        vals.append(f"{np.polyfit(lr, np.log(dhm[large_idx, li, t]), 1)[0]:.3f}")
    print(f"L{l:2d}   | " + " | ".join(vals))

# alpha over the small-r set (should be 1 everywhere -> confirms no time-dep there)
small_idx = np.where(D["small"])[0]
ls = np.log(radii[small_idx])
print("\nalpha(l,t) over r in [1e-3, 0.3] (linear regime):")
print(hdr)
for li, l in enumerate(layers):
    if l == 0:
        continue
    fl = D["flush"].get(int(l))
    vals = []
    for t in ts:
        if fl is not None and t >= fl:
            vals.append("  -- ")
            continue
        vals.append(f"{np.polyfit(ls, np.log(dhm[small_idx, li, t]), 1)[0]:.3f}")
    print(f"L{l:2d}   | " + " | ".join(vals))
