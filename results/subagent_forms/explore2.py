import numpy as np, os
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"]

# where are the zeros?
for li, l in enumerate(layers):
    x = dh[:, :, li, :]
    print(f"layer {l}: zeros={np.mean(x<=0)*100:6.2f}%  median={np.median(x):.3g}  t=0 med={np.median(x[:,:,0]):.3g}")

# shape stats on log curves (r<=0.3, layer 27)
print("\n=== shape stats, layer 27 ===")
for r in radii[:7]:
    i = int(np.argmin(np.abs(radii - r)))
    y = np.maximum(dh[i, :, -1, :], 1e-12).astype(np.float64)
    ly = np.log10(y)  # [20, 300]
    med = np.median(ly, axis=0)
    pk = int(np.argmax(med))
    # slope of median log-curve pre/post peak via polyfit
    t = np.arange(300)
    pre = (t >= 1) & (t <= max(pk, 3))
    post = (t >= pk) & (t <= 299)
    s_pre = np.polyfit(t[pre], med[pre], 1)[0] if pre.sum() > 3 else np.nan
    s_post = np.polyfit(t[post], med[post], 1)[0] if post.sum() > 3 else np.nan
    # noise: median over t of IQR across dirs of log10
    noise = np.median(np.percentile(ly, 75, axis=0) - np.percentile(ly, 25, axis=0))
    print(f"r={r:6g}: pk_t={pk:3d} pk={10**med[pk]:9.3g} a(1)={10**med[1]:9.3g} end={10**med[-1]:9.3g} "
          f"slope_pre={s_pre:+.4f} slope_post={s_post:+.4f} log10IQR={noise:.3f}")

# same for layer 7 and layer 14
for L in [7, 14]:
    li = list(layers).index(L)
    print(f"\n=== layer {L} ===")
    for r in radii[:7]:
        i = int(np.argmin(np.abs(radii - r)))
        y = np.maximum(dh[i, :, li, :], 1e-12).astype(np.float64)
        med = np.median(np.log10(y), axis=0)
        pk = int(np.argmax(med)); t = np.arange(300)
        pre = (t >= 1) & (t <= max(pk, 3)); post = (t >= pk) & (t <= 299)
        s_pre = np.polyfit(t[pre], med[pre], 1)[0] if pre.sum() > 3 else np.nan
        s_post = np.polyfit(t[post], med[post], 1)[0] if post.sum() > 3 else np.nan
        print(f"r={r:6g}: pk_t={pk:3d} pk={10**med[pk]:9.3g} a(1)={10**med[1]:9.3g} end={10**med[-1]:9.3g} "
              f"slope_pre={s_pre:+.4f} slope_post={s_post:+.4f}")

# cross-direction spread at fixed t (log10) for r=0.1
i = int(np.argmin(np.abs(radii - 0.1)))
y = np.maximum(dh[i, :, -1, :], 1e-12).astype(np.float64)
ly = np.log10(y)
for tt in [1, 2, 5, 10, 30, 100, 299]:
    print(f"t={tt:3d}: log10 a: min={ly[:,tt].min():+.2f} q25={np.percentile(ly[:,tt],25):+.2f} med={np.median(ly[:,tt]):+.2f} q75={np.percentile(ly[:,tt],75):+.2f} max={ly[:,tt].max():+.2f}")
