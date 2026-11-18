import os
import numpy as np

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\flush_sweep"
print("W  layer  first-zero t*   a(t*-1)")
for w in [100, 150, 200, 250, 300, 400, 500]:
    p = os.path.join(B, f"w{w}", "a1.npz")
    if not os.path.exists(p):
        continue
    d = np.load(p, allow_pickle=True)
    L = list(d["layers"]); radii = d["radii"]; ri = int(np.argmin(np.abs(radii - 0.1)))
    row = []
    for l in L:
        li = L.index(l)
        a = np.median(d["dh_norm"][ri, :, li, :], axis=0)
        z = np.where(a <= 0)[0]
        t = int(z[0]) if len(z) else -1
        row.append((l, t))
    print(f"W={w:4d} P={int(d['prompt_ids'].shape[0])}: " + "  ".join(f"L{l}:{t}" for l, t in row))
