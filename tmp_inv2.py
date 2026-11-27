import numpy as np, glob, os, sys
sys.path.insert(0, "analysis_update")
from cache import BIG, load, HERE
for n in BIG:
    for tag in ("inter", "childhood", "cyber"):
        p = os.path.join(HERE, BIG[n][0] + "_" + tag + ".npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            print(n, tag, sorted(z.files))
            for k in ("dh_med", "rel_med", "dh_lo", "dh_hi", "norm_h0", "layers"):
                if k in z.files:
                    print("   ", k, z[k].shape)
            break
