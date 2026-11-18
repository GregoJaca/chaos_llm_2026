import numpy as np
import common2 as C

for src, r in [("free_w200.json", 0.1), ("free_w200.json", 0.3),
               ("free_w500.json", 0.1), ("free_w500.json", 0.3)]:
    e = [x for x in C.load_free("all") if x["source"] == src and x["r"] == r][0]
    obs = np.asarray(e["obs"])
    fl = {int(d): int(obs[d]) for d in np.where(obs >= 0)[0]}
    print(src, "r =", r, "->", fl)
