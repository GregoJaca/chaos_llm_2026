import glob, os
import numpy as np

files = sorted(glob.glob("results/big4/a1_4k/*/a1.npz"))
print("runs", len(files))
for f in files[:3]:
    d = np.load(f)
    m = d["margins"]
    T = len(m)
    blocks = [(0, 300), (300, 600), (600, 1200), (1200, 2000), (2000, 3000), (3000, T)]
    print(os.path.basename(os.path.dirname(f)), "T=", T)
    for a, b in blocks:
        if a >= T:
            continue
        seg = m[a:min(b, T)]
        print(f"   t={a:4d}-{min(b,T):4d}: n<0.05 {int((seg<0.05).sum()):3d}  n<0.01 {int((seg<0.01).sum()):3d}  min {seg.min():.4f}")
    # response decay: fraction of the first-300 median at later blocks
    a = np.median(d["dh_norm"][0, :, -1, :], axis=0)
    base = np.median(a[:300])
    tail = [np.median(a[x:min(y, T)]) / base for x, y in blocks[1:]]
    print("   response relative to first 300:", [round(float(x), 4) for x in tail])
