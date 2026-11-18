import numpy as np
p = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(p, allow_pickle=True)
m = d["margins"]; radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.3)))
dm = d["dm"][ri] / radii[ri]          # margin change per unit radius
ties = np.argsort(m)[:8]
print("near-tie tokens (t, margin):", [(int(t), round(float(m[t]), 4)) for t in ties])
print()
print("per-tie spread of the response across directions (dm/r):")
for t in ties:
    v = dm[:, t]
    print(f"  t={t:3d} m={m[t]:.4f}  min={v.min():+.4f} max={v.max():+.4f} "
          f"neg={int((v<0).sum())}/20  std={v.std():.4f}")
print()
# correlation between ties across directions
C = np.corrcoef(dm[:, ties])
print("correlation between ties (over the 20 directions):")
print(np.round(C, 2))
print()
# per-direction consistency across ties
print("per-direction: number of ties with negative dm, and mean dm/r")
for di in range(20):
    v = dm[di, ties]
    print(f"  dir {di:2d}: neg={int((v<0).sum())}/8  mean={v.mean():+.4f}  vals={np.round(v,3)}")
