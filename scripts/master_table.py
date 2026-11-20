"""One consistent table of model-level numbers for master.md."""
import json
import os

import numpy as np

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026"
LD = os.path.join(ROOT, "results", "local_data")

MODELS = [
    ("DeepSeek-1.5B", "deepseek", os.path.join(LD, "big", "all_layers", "a1.npz"), 0.3),
    ("Qwen2.5-3B", "qwen25_3b", os.path.join(LD, "models", "qwen25_3b", "a1.npz"), 0.3),
    ("Qwen3-4B", "qwen3_4b", os.path.join(LD, "models", "qwen3_4b", "a1.npz"), 0.3),
    ("Phi-4-mini", "phi4mini", os.path.join(LD, "models", "phi", "a1.npz"), 0.3),
    ("Qwen3.5-2B", "qwen3_5", os.path.join(LD, "models", "qwen35", "a1.npz"), 0.3),
]

rows = {}
for name, tag, path, r0 in MODELS:
    if not os.path.exists(path):
        continue
    d = np.load(path, allow_pickle=True)
    A = d["dh_norm"].astype(float)
    h = d["norm_h0"].astype(float)
    radii = d["radii"]
    ri = int(np.argmin(np.abs(radii - r0)))
    a = np.median(A[ri, :, -1, :], axis=0)          # last layer
    k = 10
    t = np.arange(k, len(a) - k)
    lam = np.log(np.maximum(a[t + k], 1e-12) / np.maximum(a[t - k], 1e-12)) / (2 * k)
    late = float(np.mean(lam[t > 150]))
    early = float(np.mean(lam[t < 60]))
    # direction spread at r=0.1
    ri1 = int(np.argmin(np.abs(radii - 0.1)))
    g = np.median(A[ri1, :, -1, :50], axis=1) / 0.1
    g = g[g > 0]
    spread = float(np.percentile(g, 95) / np.percentile(g, 5)) if len(g) > 3 else np.nan
    rel = float(np.median(A[ri1, :, -1, 50]) / h[-1, 50])
    # kick gain at s=16 from probe2
    f = os.path.join(LD, "probe2", tag, "pos16", "layer_probe.npz")
    kick = np.nan
    if os.path.exists(f):
        z = np.load(f)
        dh = z["dh"]; rr = float(z["r"]); p = int(z["pos"])
        kick = float(np.median(dh[0][:, -1, p + 1:p + 51])) / rr
    # block sigma at block 0
    fb = os.path.join(ROOT, "results", "block_sigma", tag, "block_sigma.npz")
    b0 = np.nan
    if os.path.exists(fb):
        b0 = float(np.median(np.load(fb)["smax"][:, 0]))
    rows[name] = dict(early_lam=early, late_lam=late, spread=spread, rel_last=rel,
                      kick16=kick, block0_sigma=b0, layers=int(A.shape[2]))

print(json.dumps(rows, indent=1))
json.dump(rows, open(os.path.join(ROOT, "results", "master_table.json"), "w"), indent=1)
