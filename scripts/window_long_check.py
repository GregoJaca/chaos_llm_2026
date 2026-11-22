#!/usr/bin/env python3
"""First departure at W=300,500 (same fixed 900-token continuation)."""
import json
import os

import numpy as np

R = os.path.expanduser("~/LLM/chaos_llm_2026/results/window_long")
res = {}
for m in ["deepseek", "qwen3_4b", "phi4mini"]:
    f3 = os.path.join(R, f"{m}_w300", "a1.npz")
    f5 = os.path.join(R, f"{m}_w500", "a1.npz")
    if not (os.path.exists(f3) and os.path.exists(f5)):
        continue
    a = np.load(f3); b = np.load(f5)
    P = int(a["prompt_ids"].shape[0])
    A = np.median(a["dh_norm"][0], axis=0)     # [layer, T]
    B = np.median(b["dh_norm"][0], axis=0)
    d = np.abs(A - B) / np.maximum(B, 1e-12)
    first = []
    for li in range(A.shape[0]):
        idx = np.where(d[li] > 1e-3)[0]
        first.append(int(idx[0]) if len(idx) else -1)
    res[m] = dict(P=P, first=first, layers=[int(x) for x in a["layers"]],
                  pred_w300=int(300 - P), pred_w500=int(500 - P))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(R), "window_long_departure.json"), "w"), indent=1)
