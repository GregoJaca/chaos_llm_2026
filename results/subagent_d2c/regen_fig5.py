"""Regenerate fig5 from delay_results.json (no recompute of the m-sweep)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import exp5_delay as e5
from common_d2c import OUT

with open(OUT / "delay_results.json") as f:
    res = json.load(f)
e5.make_fig(res)
print("fig5 regenerated")
