import numpy as np
base = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
d = np.load(base + r"\w100_layers\a1.npz", allow_pickle=True)
dh = d["dh_norm"]; layers = d["layers"]; cos = d["cos"]; radii = d["radii"]
prof = dh.mean(axis=1)  # [12, 8, 300]
np.set_printoptions(precision=4, suppress=True, linewidth=180)
print("layers:", layers)
i = 4  # r=0.1
print("r=0.1, dh_norm mean-over-dirs at tracked layers:")
for j in range(8):
    p = prof[i, j]
    print(f"  L{layers[j]:3d}: t0={p[0]:.4f} t1={p[1]:.4f} t5={p[5]:.4f} t10={p[10]:.4f} t25={p[25]:.4f} t50={p[50]:.4f} t100={p[100]:.4f} t150={p[150]:.4f} t200={p[200]:.4f} t299={p[299]:.4f}")
i = 9  # r=1.0
print("r=1.0:")
for j in [0, 3, 7]:
    p = prof[i, j]
    print(f"  L{layers[j]:3d}: t0={p[0]:.4f} t1={p[1]:.4f} t5={p[5]:.4f} t10={p[10]:.4f} t25={p[25]:.4f} t50={p[50]:.4f} t100={p[100]:.4f} t150={p[150]:.4f} t200={p[200]:.4f} t299={p[299]:.4f}")
cp = cos.mean(axis=1)
print("cos[i=9,L27] t0,t10,t50,t100,t200:", np.round(cp[9, 7][[0, 10, 50, 100, 200]], 4))
print("dh_norm[i=9, L27, t=100] over dirs:", np.round(dh[9, :, 7, 100], 4))
print("dh_norm[i=9, L27, t=0] over dirs:", np.round(dh[9, :, 7, 0], 4))
print("dh_norm[i=9, L27, t=1] over dirs:", np.round(dh[9, :, 7, 1], 4))
