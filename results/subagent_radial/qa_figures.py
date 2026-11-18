"""Quick QA on the generated figures: dimensions + ink coverage per panel third."""
from PIL import Image
import numpy as np
import os

OUT = r"C:\Users\hjaca\research\chaos_llm_2026\results\subagent_radial"
for f in sorted(os.listdir(OUT)):
    if not f.endswith(".png"):
        continue
    im = Image.open(os.path.join(OUT, f)).convert("L")
    a = np.asarray(im)
    h, w = a.shape
    thirds = [round(float((a[:, i * w // 3:(i + 1) * w // 3] < 235).mean()), 3) for i in range(3)]
    print(f"{f:34s} {w}x{h}  ink per third (L|C|R): {thirds}  total {float((a < 235).mean()):.3f}")
