import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
OUT  = os.path.dirname(os.path.abspath(__file__))
from fitlib import fit_segments, find_peak, fit_exp
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
for L in [7, 14, 27]:
    li = int(np.where(layers == L)[0][0])
    for r in [0.01, 0.1, 0.3]:
        ri = int(np.argmin(np.abs(radii - r)))
        t = np.arange(300)
        rr, dr = [], []
        for di in range(20):
            y = dh[ri, di, li, :]; m = (t >= 1) & (y > 0)
            tpk = find_peak(t[m], y[m])
            seg = fit_segments(t[m], y[m], tpk)
            if "rise" in seg: rr.append(seg["rise"]["r2"])
            if "decay-exp" in seg: dr.append(seg["decay-exp"]["r2"])
        print(f"L={L} r={r}: rise R2 med={np.median(rr):.3f} (q25={np.quantile(rr,.25):.3f})  "
              f"decay-exp R2 med={np.median(dr):.3f} (q25={np.quantile(dr,.25):.3f})")
