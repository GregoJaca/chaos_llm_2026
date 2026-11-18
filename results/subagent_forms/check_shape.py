import numpy as np, os
BASE = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data"
d = np.load(os.path.join(BASE, "w100_layers", "a1.npz"))
radii = d["radii"]; layers = d["layers"]; dh = d["dh_norm"].astype(np.float64)
ri = int(np.argmin(np.abs(radii - 0.1))); li = int(np.where(layers == 27)[0][0])
y = dh[ri, :, li, :]
ly = np.log10(np.maximum(y, 1e-300))
print("t:      ", "  ".join(f"{t:5d}" for t in [1,10,20,30,40,50,60,70,75,80,85,90,95,100,110,120,130,150,200,299]))
for di in range(6):
    print(f"d={di:2d}:  ", "  ".join(f"{ly[di,t]:+5.2f}" for t in [1,10,20,30,40,50,60,70,75,80,85,90,95,100,110,120,130,150,200,299]))
med = np.median(ly, axis=0)
print("median: ", "  ".join(f"{med[t]:+5.2f}" for t in [1,10,20,30,40,50,60,70,75,80,85,90,95,100,110,120,130,150,200,299]))
# where are the top values per direction?
for di in range(6):
    o = np.argsort(ly[di])[::-1][:8]
    print(f"d={di} top-8 t:", sorted(o.tolist()), "vals:", np.round(np.sort(ly[di])[::-1][:8],2))
# smoothed versions
import numpy as np
k = np.ones(7)/7
sm = np.convolve(np.pad(med, (3,3), mode="edge"), k, mode="valid")
print("smoothed median at t:", {t: round(float(sm[t]),3) for t in range(0, 300, 10)})
# margins
m = d["margins"]
print("margin minima at:", np.argsort(m)[:12], "values:", np.round(np.sort(m)[:12], 4))
print("margin at 70-100:", np.round(m[70:100:3], 4))
