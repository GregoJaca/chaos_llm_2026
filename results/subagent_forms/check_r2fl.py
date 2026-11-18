import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
from fitlib import fit_segments, find_peak
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
ri = int(np.argmin(np.abs(radii - 0.1)))
for L in [7, 27]:
    li = int(np.where(layers == L)[0][0])
    r2f, r2e = [], []
    for di in range(20):
        y = dh[ri, di, li, :]
        t = np.arange(300); m = (t >= 1) & (y > 0)
        tpk = find_peak(t[m], y[m])
        seg = fit_segments(t[m], y[m], tpk)
        if "decay-expfloor" in seg:
            r2f.append(seg["decay-expfloor"]["r2"]); r2e.append(seg["decay-exp"]["r2"])
    print(f"L={L}: decay R2: expfloor med={np.median(r2f):.3f}, exp med={np.median(r2e):.3f}, n={len(r2f)}")
