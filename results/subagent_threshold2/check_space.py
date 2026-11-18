import numpy as np
import common2 as C

c = C.load_case("w100")
ir = int(np.argmin(np.abs(c.radii - 0.5)))
m = c.m
mr = c.m + c.dm[ir, 12]
mg = c.m + c.dm[ir, 18]
print("window  min(m)   median(m)  min(mr)  min(mg)")
for s in range(0, 300, 20):
    sl = slice(s, s + 20)
    print(f"{s:3d}-{s+20:3d}  {m[sl].min():8.3f}  {np.median(m[sl]):8.3f}  "
          f"{mr[sl].min():8.4f}  {mg[sl].min():8.4f}")
print("\npoints where m < 0.3:", np.where(m < 0.3)[0].tolist())
print("points where mr < 0.3:", np.where(mr < 0.3)[0].tolist())
print("points where mg < 0.3:", np.where(mg < 0.3)[0].tolist())
print("\nm at some key spots:", {t: round(float(m[t]), 3) for t in
                                 [0, 30, 60, 90, 150, 180, 210, 240, 270, 299]})
print("mr at key spots:", {t: round(float(mr[t]), 3) for t in
                           [0, 30, 60, 90, 150, 180, 210, 240, 270, 299]})
print("mg at key spots:", {t: round(float(mg[t]), 3) for t in
                           [0, 30, 60, 90, 150, 180, 210, 240, 270, 299]})
