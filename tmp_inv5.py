import numpy as np, sys, os
sys.path.insert(0, "analysis_update")
from cache import a1, SMALL, HERE

z = a1("DeepSeek-1.5B")
dh = z["dh"]        # (r, dir, layer, T)
lay = z["layers"]
print("layers", lay)
ri = int(np.argmin(np.abs(z["radii"] - 0.1)))
for li, l in enumerate(lay[:4]):
    row = dh[ri, :, li, :]
    print("layer %2d: median over dirs at t=0,1,2,5,20,50,150: %s" % (
        l, np.array2string(np.median(row, axis=0)[[0, 1, 2, 5, 20, 50, 150]], precision=6)))
print()
print("fraction of entries exactly zero at layer 0:", float(np.mean(dh[ri, :, 0, :] == 0.0)))
print("layer0 max:", float(dh[ri, :, 0, :].max()), " layer1 max:", float(dh[ri, :, 1, :].max()))
# how many tokens of the prompt?
import json
m = json.load(open(os.path.join(HERE, "results/rstar/deepseek/p1/meta.json")))
print("prompt_len", m["prompt_len"], "window", m["window"])
