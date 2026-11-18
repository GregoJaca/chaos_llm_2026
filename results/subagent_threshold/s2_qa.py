"""QA: verify each figure PNG is non-blank, reasonably sized, and rich; plus
numeric cross-checks used in REPORT.md."""
import glob
import os

import numpy as np
from PIL import Image

from common import Case, OUT

ok = True
for p in sorted(glob.glob(OUT + r"\fig*.png")):
    im = np.asarray(Image.open(p).convert("L"), dtype=float)
    frac_nonwhite = (im < 250).mean()
    print(f"{os.path.basename(p):36s} {im.shape[1]}x{im.shape[0]}  "
          f"non-white px: {frac_nonwhite:6.1%}  std: {im.std():6.1f}")
    if im.std() < 10 or frac_nonwhite < 0.02:
        print("  !! looks blank/degenerate")
        ok = False
print("QA:", "PASS" if ok else "FAIL")

# ---- numeric cross-checks for the report ----
print("\ncross-checks:")
c200 = Case("w200")
rc_neg = np.where(c200.dm[c200.ir_ref] < 0, c200.rc_ref(), np.inf)
rstar = rc_neg.min(axis=1)
tstar = rc_neg.argmin(axis=1)
obs03 = np.array(c200.free()["obs"][1])
fl = obs03 >= 0
# flipping dirs have low signed r*?
print("w200 r=0.3: flipping dirs' signed r_* range:",
      round(rstar[fl].max(), 3), "| non-flipping min:", round(rstar[~fl].min(), 3))
print("  -> overlap band:", round(rstar[~fl].min(), 3), "to", round(rstar[fl].max(), 3),
      " (borderline zone ~ +-30% of r)")
n44 = (tstar == 44).sum()
print(f"w200: {n44}/20 directions controlled by the t=44 tie")
c100 = Case("w100")
dm44 = c200.dm[c200.ir_ref][:, 44]
print(f"sign split at w200 tie t=44: destabilizing {(dm44 < 0).sum()}/20")
dm123 = c100.dm[c100.ir_ref][:, 123]
print(f"sign split at w100 tie t=123: destabilizing {(dm123 < 0).sum()}/20")
