import sys, os
sys.path.insert(0, "analysis_update")
from looputil import TAGS, pairs, find_dirs, model_tags
import numpy as np
for name in ["DeepSeek-1.5B", "Qwen3-14B", "Phi-4-mini", "randQwen"]:
    d = find_dirs(model_tags(name))
    p = pairs(name)
    print(name, "dirs", len(d), "pairs", len(p))
    if p:
        z = np.load(p[0][0], allow_pickle=True)
        print("   files", "layers" in z.files, z["layers"] if "layers" in z.files else None,
              "dh", z["dh"].shape)
