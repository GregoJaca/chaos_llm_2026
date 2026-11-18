import json
import numpy as np

with open("d2_results.json") as f:
    d2 = json.load(f)
print(f"{'trajectory':36s} {'stable-run slope@width':>22s} {'inside ER band?':>15s}")
for n in d2:
    for cfg in ("N1000_w0", "N1000_w10", "N1000_w50", "N500_w0"):
        r = d2[n]["cfgs"][cfg]
        run = r["run"]
        if run:
            print(f"{n[:34]:36s} {cfg:9s} {run['slope']:6.2f}@{run['width_dec']:.2f}dec "
                  f"C[{run['C_lo']:.4f},{run['C_hi']:.3f}] legal={r['erlegal_run']}")
print()
w = [d2[n]["cfgs"]["N1000_w0"]["run"]["width_dec"] if d2[n]["cfgs"]["N1000_w0"]["run"] else 0
     for n in d2]
s = [d2[n]["cfgs"]["N1000_w0"]["run"]["slope"] if d2[n]["cfgs"]["N1000_w0"]["run"] else np.nan
     for n in d2]
print(f"N1000_w0: run widths med {np.median(w):.2f} dec (max {max(w):.2f}); slopes med {np.median(s):.2f}")
allw = []
for n in d2:
    for cfg, r in d2[n]["cfgs"].items():
        if r["run"]:
            allw.append((cfg, r["run"]["width_dec"]))
print("any config with a legal stable run:", sum(1 for n in d2 for c, r in d2[n]["cfgs"].items() if r["erlegal_run"]))
