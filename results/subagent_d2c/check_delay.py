import json
import numpy as np

with open(r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_d2c\delay_results.json") as f:
    res = json.load(f)

names = ["renewable_energy_economics", "sleep_and_memory", "ocean_currents",
         "interstellar_travel", "urban_transportation", "history_of_music"]
print(f"{'traj':34s} {'m':>2s}  {'data':>5s} {'iaaft-mean':>10s} {'below-band':>10s}")
for n in names:
    r = res[n]
    ia = np.array(r["d2_iaaft"])
    for mi, m in enumerate(range(2, 11)):
        dm = r["d2_data"][mi]
        lo, hi = ia[mi].min(), ia[mi].mean()
        below = "YES" if dm < ia[mi].min() else "no"
        if m in (2, 4, 6, 8, 10):
            print(f"{n[:34]:34s} {m:2d}  {dm:5.2f} {ia[mi].mean():10.2f} {below:>10s}")
    print()

# summary: fraction of (traj, m>=4) with data inside IAAFT band
inside = 0; below = 0; tot = 0
for n, r in res.items():
    ia = np.array(r["d2_iaaft"])
    for mi, m in enumerate(range(2, 11)):
        if m < 4: continue
        tot += 1
        dm = r["d2_data"][mi]
        if ia[mi].min() <= dm <= ia[mi].max(): inside += 1
        if dm < ia[mi].min(): below += 1
print(f"m>=4: data inside IAAFT band {inside}/{tot}, below band {below}/{tot}")
