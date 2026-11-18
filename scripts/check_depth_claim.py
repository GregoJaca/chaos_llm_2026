import numpy as np
p = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(p, allow_pickle=True)
layers = list(d["layers"]); radii = d["radii"]
print("layers", layers)
for r in [0.01, 0.1]:
    ri = int(np.argmin(np.abs(radii - r)))
    print("r =", radii[ri])
    prev = None
    for li, l in enumerate(layers):
        dh = float(np.median(d["dh_norm"][ri, :, li, :]))
        nh = float(np.median(d["norm_h0"][li, :]))
        rel = dh / nh
        g = "" if (prev is None or prev <= 0) else f"  x{dh/prev:.2f}"
        print(f"  L{l:>2}: |h|={nh:8.2f}  |dh|={dh:.3e}  |dh|/|h|={rel:.2e}{g}")
        prev = dh
