"""Light QA on rendered figures: dimensions + ink coverage (catches blank/clipped panels)."""
from PIL import Image
import numpy as np, glob, os
for f in sorted(glob.glob('chaos_llm_2026/results/subagent_nonnorm/fig*.png')):
    im = Image.open(f)
    a = np.asarray(im.convert('L'))
    ink = (a < 200).mean()
    # ink per quadrant/strip to catch one blank panel
    h, w = a.shape
    strips = []
    for i in range(2):
        for j in range(2):
            s = a[i * h // 2:(i + 1) * h // 2, j * w // 2:(j + 1) * w // 2]
            strips.append((s < 200).mean())
    print('{:38s} {}x{}  ink={:.3f}  quadrant inks {}'.format(
        os.path.basename(f), w, h, ink, [round(x, 3) for x in strips]))
