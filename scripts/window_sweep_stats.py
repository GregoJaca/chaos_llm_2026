import os
import numpy as np
B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"
T = 300
for w in [50, 100, 200, 400, 800, 4096]:
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27)
    ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    a = np.median(d["dh_norm"][ri, :, li, :], axis=0)
    a = np.maximum(a, 1e-12)
    lam = np.log(a[1:] / a[:-1])          # per-token log growth rate
    print(f"W={w:5d}  a(t=0)={a[0]:.3f} a(20)={a[20]:.3f} a(min)={a.min():.4f} a(299)={a[-1]:.4f}")
    print(f"          mean dlog a/dt: first 20={lam[:20].mean():+.4f}  20-50={lam[20:50].mean():+.4f}  "
          f"50-100={lam[50:100].mean():+.4f}  100-200={lam[100:200].mean():+.4f}  200-299={lam[200:].mean():+.4f}")
    # finite-time rate
    for t in [10, 30, 60, 100, 200, 299]:
        print(f"          lambda({t:3d})={np.log(a[t]/a[1])/t:+.4f}", end="")
    print()
