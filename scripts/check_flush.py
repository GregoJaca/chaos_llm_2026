import numpy as np, os
B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(B, allow_pickle=True)
L = list(d["layers"]); radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.1)))
for l in [1, 2, 3, 7, 14, 21, 27]:
    li = L.index(l)
    a = np.median(d["dh_norm"][ri, :, li, :], axis=0)
    zeros = np.where(a <= 0)[0]
    print(f"L{l:2d}: first exact zero at t={int(zeros[0]) if len(zeros) else -1}, "
          f"n zeros={len(zeros)}, a(90)={a[90]:.3e}, a(95)={a[95]:.3e}, a(299)={a[299]:.3e}")
