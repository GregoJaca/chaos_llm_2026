"""QA: re-verify every quantitative claim that appears as text in the figures."""
import json

import numpy as np

import common2 as C

c = C.load_case("w100")
free = {e["r"]: e for e in C.load_free("all") if e["geom"] == "w100"}
ir_of = {float(r): i for i, r in enumerate(c.radii)}
rc0 = c.rc_signed()[0]
rc0[~np.isfinite(rc0)] = 1e30

# fig5 claim: at r=2, extrapolated prediction worst step error 129 =
# predicted word 14, observed word 143?
obs2 = np.asarray(free[2.0]["obs"])
bad = []
for d in range(20):
    hit = np.where(rc0[d] <= 2.0)[0]
    if len(hit) and obs2[d] >= 0:
        pred = int(hit[0])
        if pred != obs2[d]:
            bad.append((d, pred, int(obs2[d]), abs(pred - int(obs2[d]))))
print("r=2 extrapolation step errors (d, predicted, observed, |err|):")
for b in sorted(bad, key=lambda x: -x[3]):
    print("   ", b)

# fig5: only 3/16 exact steps at r=2 under extrapolation
ok = 0
for d in range(20):
    hit = np.where(rc0[d] <= 2.0)[0]
    if len(hit) and obs2[d] >= 0 and int(hit[0]) == obs2[d]:
        ok += 1
print(f"r=2 extrapolation: exact steps {ok}/16")

# fig5: 'predicted 8 vs observed 8 at r=1' for delivered rule
obs1 = np.asarray(free[1.0]["obs"])
ship1 = np.asarray(free[1.0]["pred"])
print(f"r=1 delivered: predicted flips {int((ship1>=0).sum())}, "
      f"observed {int((obs1>=0).sum())}")

# fig2: pooled 70 borderline all correct
pts = json.load(open("eval_points.json"))
nb = nok = 0
for row in pts:
    if row["rho"] is None:
        continue
    rho, obs = np.asarray(row["rho"]), np.asarray(row["obs"])
    pred = np.asarray(row["pred_sgn"])
    m = np.abs(np.log2(rho)) <= 1
    nb += int(m.sum()); nok += int((m & (pred == obs)).sum())
print(f"borderline calls: {nb}, correct: {nok}")

# fig3: numbers
nflip_seq = sum(int((np.asarray(e["obs"]) >= 0).sum()) for e in C.load_free("all")
               if e["npz"] is not None)
nflip_all = sum(int((np.asarray(e["obs"]) >= 0).sum()) for e in C.load_free("all"))
print(f"flips: {nflip_all} total, {nflip_seq} with sequences; "
      f"no-flip runs {320 - nflip_all}")

# fig1: ratio typical/min gap
print(f"typical gap / min gap = {np.median(c.m) / c.m.min():.0f}x")

# fig4: d14 never flipped at any radius
for r in sorted(free):
    print(f"  d14 obs at r={r:g}: {int(free[r]['obs'][14])}")
