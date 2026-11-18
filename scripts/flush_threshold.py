import os
import numpy as np

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\flush_sweep"
print("threshold-based flush time: first t with a(t) < f * a(1)")
for f in [0.01, 0.05]:
    print(f"\n--- f = {f}")
    for w in [100, 150, 200, 250, 300, 400, 500]:
        p = os.path.join(B, f"w{w}", "a1.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        L = list(d["layers"]); radii = d["radii"]; ri = int(np.argmin(np.abs(radii - 0.1)))
        out = []
        for l in L:
            li = L.index(l)
            A = d["dh_norm"][ri, :, li, :]
            ts = []
            for di in range(A.shape[0]):
                a = A[di]
                below = np.where(a < f * a[1])[0]
                if len(below):
                    ts.append(int(below[0]))
            out.append((l, int(np.median(ts)) if ts else -1, max(ts) if ts else -1))
        print(f"W={w:4d}: " + "  ".join(f"L{l}:{m}/{mx}" for l, m, mx in out))
