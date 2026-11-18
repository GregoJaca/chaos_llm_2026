import numpy as np
p = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(p, allow_pickle=True)
layers = list(d["layers"]); radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.1)))
print("layer  |h|      |dh|     cos        angle(deg)  perp/|dh|   perp growth")
prev = None
for li, l in enumerate(layers):
    dh = float(np.median(d["dh_norm"][ri, :, li, :]))
    nh = float(np.median(d["norm_h0"][li, :]))
    c = float(np.median(d["cos"][ri, :, li, :]))
    c = min(max(c, -1.0), 1.0)
    ang = np.degrees(np.arccos(c))
    perp = np.sqrt(max(0.0, 1 - c * c))
    plen = dh * perp
    g = "" if (prev is None or prev <= 0) else f"  x{plen/prev:.2f}"
    print(f"L{l:>2}  {nh:8.2f} {dh:.3e} {c:.8f} {ang:9.4f}  {perp:.2e}   {plen:.3e}{g}")
    prev = plen
