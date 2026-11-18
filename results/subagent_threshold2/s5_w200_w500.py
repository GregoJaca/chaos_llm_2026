"""Verify that the w200 and w500 free runs are literally the same experiment
(same directions, same flip steps) by comparing their raw records."""
import numpy as np

import common2 as C

for src, r in [("a1_ds_w200_large/free_run", 0.1),
               ("a1_ds_w200_large/free_run", 0.3),
               ("a1_ds_w500_large/free_run", 0.1),
               ("a1_ds_w500_large/free_run", 0.3)]:
    e = [x for x in C.load_free("all")
         if x["source"] == src and x["r"] == r][0]
    obs = np.asarray(e["obs"])
    fl = {int(d): int(obs[d]) for d in np.where(obs >= 0)[0]}
    print(src, "r =", r, "->", fl)

a = [x for x in C.load_free("all") if x["source"] == "a1_ds_w200_large/free_run"]
b = [x for x in C.load_free("all") if x["source"] == "a1_ds_w500_large/free_run"]
for ea, eb in zip(a, b):
    print(f"r={ea['r']:g}: w200 obs == w500 obs: "
          f"{(np.asarray(ea['obs']) == np.asarray(eb['obs'])).all()}")
