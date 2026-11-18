import numpy as np
p = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(p, allow_pickle=True)
radii = d["radii"]; dm = d["dm"]
g = np.array([np.median(np.abs(dm[i]) / radii[i]) for i in range(len(radii))])
print("r, median |dm|/r, ratio to the 0.1 value")
ref = g[int(np.argmin(np.abs(radii - 0.1)))]
for r, gi in zip(radii, g):
    print(f"{r:8.4g}  {gi:.5f}  {gi/ref:.3f}")
# power-law fit above 0.3
sel = radii >= 0.3
b, a = np.polyfit(np.log(radii[sel]), np.log(g[sel]), 1)
print("power law exponent above 0.3:", round(float(b), 3), "prefactor", round(float(np.exp(a)), 4))
# ratio r=2 / r=0.1
print("g(2)/g(0.1)", round(float(g[-1] / ref), 3))
