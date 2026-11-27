import numpy as np, os, sys
sys.path.insert(0, "analysis_update")
from looputil import pairs
ps = pairs("DeepSeek-1.5B")
for pl, pr in ps:
    zc = np.load(pl, allow_pickle=True); zr = np.load(pr, allow_pickle=True)
    print(pl.replace(os.getcwd(), ""), zc["dh"].shape, "|", pr.replace(os.getcwd(), ""), zr["dh"].shape)
    a = zc["dh"][:, 0, :5].astype(float)
    b = zr["dh"][:, 0, :5].astype(float)
    print("  loop layer0 t=0..4:\n", np.round(a, 6))
    print("  replay layer0 t=0..4:\n", np.round(b, 6))
