import json
import numpy as np

with open("delay_results.json") as f:
    res = json.load(f)
for m in range(2, 11):
    mi = m - 2
    below = inside = 0
    for n, r in res.items():
        ia = np.array(r["d2_iaaft"][mi])
        d = r["d2_data"][mi]
        if d < ia.min():
            below += 1
        elif d <= ia.max():
            inside += 1
    print(f"m={m:2d}: below-band {below:2d}/16, inside {inside:2d}/16, above {16-below-inside:2d}/16")
print()
print("m=2 data vs white-noise(same marginal):")
for n in ["interstellar_travel", "renewable_energy_economics", "sleep_and_memory"]:
    r = res[n]
    print(f"  {n[:34]:36s} data {r['d2_data'][0]:.2f}  wn {r['d2_wn'][0]:.2f}  "
          f"iaaft {np.mean(r['d2_iaaft'][0]):.2f}")
print()
print("saturators (D2(m=10) < D2(m=4)):")
for n, r in res.items():
    if r["d2_data"][8] < r["d2_data"][2]:
        print(f"  {n[:36]:38s} D2(4)={r['d2_data'][2]:.2f} D2(10)={r['d2_data'][8]:.2f}")
print()
print("run widths at m=2 (fill-dim trap) and legal flags:")
for n in res:
    r = res[n]
    print(f"  {n[:36]:38s} w={r['run_width'][0]:.2f} slope={r['run_slope'][0]} legal={r['run_legal'][0]}")
