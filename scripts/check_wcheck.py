import numpy as np
a = np.load("results/big4/wcheck/native/a1.npz")
b = np.load("results/big4/wcheck/mask4d/a1.npz")
for k in ["dh_norm", "dm", "margins", "gen_ids"]:
    x = a[k].astype(np.float64); y = b[k].astype(np.float64)
    print(k, "equal", bool(np.array_equal(a[k], b[k])), "maxdiff", float(np.abs(x - y).max()))
