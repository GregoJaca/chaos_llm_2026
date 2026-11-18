"""Regenerate fig3/fig4 from pred_results.json (no recompute)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import exp3_predict as e3
from common_d2c import OUT

with open(OUT / "pred_results.json") as f:
    res = json.load(f)
for n in e3.PROMPTS:
    res[n]["kg"] = {int(k): v for k, v in res[n]["kg"].items()}
e3.make_fig3(res)
e3.make_fig4(res)
print("fig3, fig4 regenerated")
