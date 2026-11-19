import numpy as np
s = np.load(r"results\a1_ds_w100\a1.npz")
l = np.load(r"results\a1_ds_w100_large\a1.npz")
radii = np.concatenate([s["radii"], l["radii"]])
d = np.concatenate([s["dh_norm"], l["dh_norm"]], 0)[:, :, -1, :]
_, idx = np.unique(np.round(radii, 8), return_index=True)
radii, d = radii[idx], d[idx]
print("r          x0(med)     x0/r     tau2   tau4   n2 n4")
for i, r in enumerate(radii):
    x = d[i]
    x0 = np.median(x[:, 0])
    t2, t4 = [], []
    for di in range(x.shape[0]):
        if x[di, 0] <= 0:
            continue
        h2 = np.where(x[di] >= 2 * x[di, 0])[0]
        h4 = np.where(x[di] >= 4 * x[di, 0])[0]
        if len(h2):
            t2.append(h2[0])
        if len(h4):
            t4.append(h4[0])
    m2 = np.median(t2) if t2 else None
    m4 = np.median(t4) if t4 else None
    print(f"{r:8.1e} {x0:10.3e} {x0/r:8.3f}  {str(m2):>5} {str(m4):>5}  {len(t2):2d} {len(t4):2d}")
# shape collapse test
for r in [1e-3, 1e-2, 0.1, 0.3]:
    i = int(np.argmin(np.abs(radii - r)))
    x = d[i]
    xn = x / np.maximum(x[:, :1], 1e-30)
    print("r", r, "median x(t)/x(0) t=1,2,4,8,16,32,64,128,299:",
          np.round(np.median(xn[:, [1, 2, 4, 8, 16, 32, 64, 128, 299]], 0), 2)[:9])
