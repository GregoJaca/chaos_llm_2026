"""Signed vs absolute margin-response thresholds.

A flip needs dm_t <= -m_t (margin destroyed). The stored r_c / pred_T use |dm|
(adversarial). Here we build the *signed* threshold rc- = r*m/max(-dm,0) and test
which better predicts observed first flips in the free runs.
"""
import json

import numpy as np

from common import BASE, Case, FREE, OUT

lines = []


def P(*a):
    print(*a)
    lines.append(" ".join(str(x) for x in a))


def first_cross(rc_row, r):
    idx = np.where(rc_row <= r)[0]
    return int(idx[0]) if len(idx) else -1


P("== signed (destabilizing) vs absolute threshold prediction of observed flips ==")
P(f"{'config':>10} {'r':>5} | {'obs flips':>9} {'pred|dm|':>9} {'predsigned':>10} | "
  f"{'exact |dm|':>10} {'exact sign':>10} | {'flip-call |dm|':>14} {'flip-call sign':>15}")

rows = []
for name in ["w100", "w200", "w500"]:
    c = Case(name)
    dm = c.dm[c.ir_ref]
    rc_abs = c.rc_ref()                      # from |dm|
    rc_neg = np.where(dm < 0, rc_abs, np.inf)  # signed: only margin-destroying
    j = c.free()
    for i, r in enumerate(j["radii"]):
        obs = np.array(j["obs"][i])
        pred_abs = np.array(j["pred"][i])
        pred_neg = np.array([first_cross(rc_neg[d], r) for d in range(c.n_dirs)])
        ea = int((obs == pred_abs).sum())
        en = int((obs == pred_neg).sum())
        fa = int(((obs >= 0) == (pred_abs >= 0)).sum())
        fn = int(((obs >= 0) == (pred_neg >= 0)).sum())
        rows.append((name, r, obs, pred_abs, pred_neg, ea, en, fa, fn))
        P(f"{name:>10} {r:>5} | {int((obs>=0).sum()):>9} {int((pred_abs>=0).sum()):>9} "
          f"{int((pred_neg>=0).sum()):>10} | {ea:>10} {en:>10} | {fa:>14} {fn:>15}")

# large-r case (w100_layers)
c = Case("w100")
dm = c.dm[c.ir_ref]
rc_abs = c.rc_ref()
rc_neg = np.where(dm < 0, rc_abs, np.inf)
j = json.load(open(FREE["w100L"]))
for i, r in enumerate(j["radii"]):
    obs = np.array(j["obs"][i])
    pred_abs = np.array(j["pred"][i])
    pred_neg = np.array([first_cross(rc_neg[d], r) for d in range(c.n_dirs)])
    ea, en = int((obs == pred_abs).sum()), int((obs == pred_neg).sum())
    fa, fn = int(((obs >= 0) == (pred_abs >= 0)).sum()), int(((obs >= 0) == (pred_neg >= 0)).sum())
    rows.append(("w100L", r, obs, pred_abs, pred_neg, ea, en, fa, fn))
    P(f"{'w100L':>10} {r:>5} | {int((obs>=0).sum()):>9} {int((pred_abs>=0).sum()):>9} "
      f"{int((pred_neg>=0).sum()):>10} | {ea:>10} {en:>10} | {fa:>14} {fn:>15}")

P("")
P(f"TOTAL over {len(rows)} configs x 20 dirs = {sum(len(r[2]) for r in rows)}:")
P(f"  flip-call correct: |dm| {sum(r[7] for r in rows)}/{sum(len(r[2]) for r in rows)}, "
  f"signed {sum(r[8] for r in rows)}/{sum(len(r[2]) for r in rows)}")
P(f"  exact-step correct: |dm| {sum(r[5] for r in rows)}, signed {sum(r[6] for r in rows)}")
P("  (both predictors tie overall; the difference is WHERE: signed fixes flip-step")
P("   attribution at r<=0.5 and flip-calls at r=0.5; both break down at r>=1.)")

# per-step detail for the borderline band, w200 r=0.3
c = Case("w200")
dm = c.dm[c.ir_ref]
rc_abs = c.rc_ref()
rc_neg = np.where(dm < 0, rc_abs, np.inf)
obs = np.array(c.free()["obs"][1])  # r=0.3
P("")
P("w200 r=0.3 detail: dir, r*|dm|, r*signed, obs, pred|dm|, pred_signed")
for d in range(20):
    P(f"  d{d:>2}: {rc_abs[d].min():7.3f}  {rc_neg[d].min():7.3f}  "
      f"obs={obs[d]:>3}  pred|dm|={first_cross(rc_abs[d], 0.3):>3}  "
      f"predsign={first_cross(rc_neg[d], 0.3):>3}")

# sign split at the controlling tie t=44
P("")
for name in ["w200", "w500"]:
    c = Case(name)
    dm44 = c.dm[c.ir_ref][:, 44]
    P(f"{name} t=44: destabilizing (dm<0): {int((dm44<0).sum())}/20, "
      f"stabilizing (dm>0): {int((dm44>0).sum())}/20")
c = Case("w100")
dm123 = c.dm[c.ir_ref][:, 123]
P(f"w100 t=123: destabilizing: {int((dm123<0).sum())}/20, stabilizing: {int((dm123>0).sum())}/20")

with open(OUT + r"\signed_check.txt", "w") as f:
    f.write("\n".join(lines))
print("\nwrote signed_check.txt")
