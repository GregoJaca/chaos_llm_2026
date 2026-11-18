import os, glob
import numpy as np

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
rows = []
for folder, label in [("rate_prompts", "W=100"), ("rate_w50", "W=50")]:
    for p in sorted(glob.glob(os.path.join(B, folder, "*", "a1.npz"))):
        d = np.load(p, allow_pickle=True)
        L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
        a = np.maximum(np.median(d["dh_norm"][ri, :, li, :], axis=0), 1e-12)
        k = 10; t = np.arange(k, 300 - k)
        loc = np.log(a[t + k] / a[t - k]) / (2 * k)
        rows.append((label, os.path.basename(os.path.dirname(p)), float(loc.max()), int(t[loc.argmax()]),
                     float((loc > 0).mean()), float(a[1])))
for r in rows:
    print(f"{r[0]:6s} {r[1]:4s} max_rate={r[2]:+.4f} at t={r[3]:3d} frac_pos={r[4]:.2f} a(1)={r[5]:.3f}")
