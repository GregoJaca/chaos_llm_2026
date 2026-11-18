import os
import numpy as np

# 20-direction reference (W=100)
d = np.load(r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz", allow_pickle=True)
L = list(d["layers"]); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
print("W=100, 20 directions: first exact zero per layer (median / max over directions)")
for l in [1, 2, 3]:
    A = d["dh_norm"][ri, :, L.index(l), :]
    med, mx = [], []
    for di in range(A.shape[0]):
        z = np.where(A[di] <= 0)[0]
        if len(z):
            med.append(int(z[0])); mx.append(int(z[0]))
    print(f"  L{l}: median={int(np.median(med)) if med else -1} max={max(mx) if mx else -1}")

print("\nflush_sweep (4 directions), median over dirs")
for w in [100, 150, 200, 300]:
    p = rf"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\flush_sweep\w{w}\a1.npz"
    if not os.path.exists(p):
        continue
    dd = np.load(p, allow_pickle=True); LL = list(dd["layers"]); rr = int(np.argmin(np.abs(dd["radii"] - 0.1)))
    ts = []
    for l in LL:
        A = dd["dh_norm"][rr, :, LL.index(l), :]
        zs = []
        for di in range(A.shape[0]):
            z = np.where(A[di] <= 0)[0]
            if len(z):
                zs.append(int(z[0]))
        ts.append((l, int(np.median(zs)) if zs else -1, max(zs) if zs else -1))
    print(f"  W={w}: " + "  ".join(f"L{l}:{m}/{mx}" for l, m, mx in ts))
