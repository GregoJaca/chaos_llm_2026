"""s2_eval.py -- full prediction scorecard across every local free-run record.

For each (source, radius, direction):
  obs            observed first-flip step from the free run (-1 = never)
  pred_ship      shipped prediction (adversarial |dm| rule, as delivered)
  pred_sgn       recomputed signed rule at the exact same radius
  rho            r / r_*-(r): how many times bigger the perturbation is than
                 the direction's smallest signed threshold at that radius
                 (rho < 1 -> safe by a margin; rho ~ 1 -> borderline call)
Writes eval_points.json + eval_summary.txt.
"""
import json
import os

import numpy as np

import common2 as C

np.set_printoptions(linewidth=200, threshold=10000)

records = C.load_free("all")
cases = {}


def geom(nm):
    if nm not in cases:
        cases[nm] = C.load_case(nm) if nm else None
    return cases[nm]


points = []
for e in records:
    src, r = e["source"], e["r"]
    gname = e["geom"]
    g = geom(gname)
    obs = np.asarray(e["obs"])
    pred_ship = np.asarray(e["pred"])
    K = len(obs)

    row = dict(source=src, r=r, prompt=e["prompt"], geom=gname, K=int(K),
               obs=obs.tolist(), pred_ship=pred_ship.tolist())
    if g is not None:
        ir = int(np.argmin(np.abs(g.radii - r)))
        dr = float(abs(g.radii[ir] - r))
        assert dr < 1e-12, (src, r, g.radii[ir])
        pred_sgn = g.pred("signed")[ir]
        pred_adv = g.pred("adv")[ir]
        rc = g.rc_signed()[ir]                       # (K, T)
        rstar = rc.min(axis=1)
        tstar = rc.argmin(axis=1)
        rstar = np.where(np.isfinite(rstar), rstar, 1e30)
        rho = r / rstar
        row.update(ir=ir, pred_sgn=pred_sgn.tolist(), pred_adv=pred_adv.tolist(),
                   rstar=rstar.tolist(), tstar=tstar.tolist(),
                   rho=rho.tolist(),
                   m_star=[float(g.m[t]) for t in tstar],
                   dm_star=[float(g.dm[ir, d, tstar[d]]) for d in range(K)])
    else:
        row.update(ir=None, pred_sgn=None, pred_adv=None, rstar=None,
                   tstar=None, rho=None)
    # divergence stats when sequences exist
    if e["npz"] is not None:
        f = e["npz"]
        ri = int(np.argmin(np.abs(f.radii - r)))
        div = []
        for d in range(K):
            T = int(obs[d])
            if T < 0:
                div.append(None)
                continue
            ref, rr = f.rows(ri, d)
            st = C.after_flip_stats(ref, rr, T, block=10, tail=50)
            div.append(dict(frac_after=st["frac_identical_after"],
                            frac_all=st["frac_identical_all"],
                            resynced=st["resynced"],
                            n_diff=int((np.asarray(ref) != np.asarray(rr)).sum())))
        row["div"] = div
    points.append(row)

with open(os.path.join(C.OUT, "eval_points.json"), "w") as fo:
    json.dump(points, fo, indent=1)

# ------------------------------------------------------------------ summary
lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


p("SCORECARD per (source, radius, rule). Rules: ship = delivered adversarial "
  "|dm| prediction; sgn = signed rule recomputed at the exact radius.")
p("exact = predicted step == observed step (both flips); noflip_ok = both -1; "
  "false = predicted flip, none observed; missed = flip observed, none predicted.")
p("")
grand = dict(exact=0, noflip=0, false=0, missed=0, wrongstep=0, n=0)
for row in points:
    obs = np.asarray(row["obs"])
    K = len(obs)
    tags = []
    for rule, pred in [("ship", np.asarray(row["pred_ship"])),
                       ("sgn", np.asarray(row["pred_sgn"]) if row["pred_sgn"] else None)]:
        if pred is None:
            continue
        both_flip = (obs >= 0) & (pred >= 0)
        exact = int((pred[both_flip] == obs[both_flip]).sum())
        noflip = int(((obs < 0) & (pred < 0)).sum())
        false_ = int(((obs < 0) & (pred >= 0)).sum())
        missed = int(((obs >= 0) & (pred < 0)).sum())
        wrong = int((pred[both_flip] != obs[both_flip]).sum())
        offs = np.abs(pred[both_flip] - obs[both_flip]) if both_flip.any() else np.array([])
        tags.append((rule, exact, noflip, false_, missed, wrong, offs))
        if rule == "sgn":
            grand["exact"] += exact
            grand["noflip"] += noflip
            grand["false"] += false_
            grand["missed"] += missed
            grand["wrongstep"] += wrong
            grand["n"] += K
    rho = np.asarray(row["rho"]) if row["rho"] else None
    line = f"{row['source'][:-4] if row['source'].endswith('.json') else row['source'][:-4]:>20s} r={row['r']:<5g}"
    for (rule, exact, noflip, false_, missed, wrong, offs) in tags:
        line += (f" | {rule}: exact {exact:2d} noflip {noflip:2d} false {false_:2d} "
                 f"missed {missed:2d} steperr {wrong:2d}")
        if len(offs):
            line += f" |off| med {np.median(offs):4.0f} max {offs.max():3.0f}"
    p(line)
    # borderline analysis for the signed rule
    if row["pred_sgn"] is not None and rho is not None:
        pred = np.asarray(row["pred_sgn"])
        wrong_mask = pred != obs
        if wrong_mask.any():
            bd = np.mean(np.abs(np.log2(rho[wrong_mask])) <= 1.0)
            p(f"    -> {int(wrong_mask.sum()):2d} signed-rule misses; "
              f"{bd*100:4.0f}% of them have r within a factor 2 of the direction's "
              f"threshold (rho in [0.5,2])")

p("")
p(f"GRAND TOTAL (signed rule, correct radius, all {grand['n']} direction-runs): "
  f"exact {grand['exact']}, no-flip correct {grand['noflip']}, false alarms "
  f"{grand['false']}, missed flips {grand['missed']}, wrong-step {grand['wrongstep']} "
  f"=> {(grand['exact'] + grand['noflip']) / grand['n'] * 100:.1f}% fully correct")

# per-radius flip counts: observed vs predicted (both rules)
p("")
p("FLIP COUNTS observed vs predicted per (source, r):")
for row in points:
    obs = np.asarray(row["obs"])
    nobs = int((obs >= 0).sum())
    line = f"{row['source'][:-4]:>20s} r={row['r']:<5g} obs {nobs:2d}/20"
    line += f" | ship {int((np.asarray(row['pred_ship'])>=0).sum()):2d}"
    if row["pred_sgn"] is not None:
        line += f" | sgn {int((np.asarray(row['pred_sgn'])>=0).sum()):2d}"
        line += f" | adv {int((np.asarray(row['pred_adv'])>=0).sum()):2d}"
    p(line)

# step errors vs rho among flipping directions (signed rule, pooled)
p("")
p("POOLED step errors (signed rule, only runs where both predicted and "
  "observed flip):")
allrho, alloff = [], []
for row in points:
    if row["pred_sgn"] is None:
        continue
    obs = np.asarray(row["obs"])
    pred = np.asarray(row["pred_sgn"])
    rho = np.asarray(row["rho"])
    bf = (obs >= 0) & (pred >= 0)
    allrho += list(rho[bf])
    alloff += list(np.abs(pred[bf] - obs[bf]))
allrho, alloff = np.array(allrho), np.array(alloff)
p(f"  n = {len(alloff)}; |pred-obs| median {np.median(alloff):.0f}, "
  f"90th pct {np.percentile(alloff, 90):.0f}, max {alloff.max():.0f} steps")
for thr in [1, 5, 20]:
    m = alloff <= thr
    p(f"  within {thr:2d} step(s): {m.mean()*100:5.1f}%")
big = alloff > 5
if big.any():
    p(f"  among the {int(big.sum())} misses >5 steps, rho median "
      f"{np.median(allrho[big]):.2f}, share in [0.5,2]: "
      f"{np.mean(np.abs(np.log2(allrho[big]))<=1)*100:.0f}%")
ok = alloff == 0
p(f"  exact hits: rho median {np.median(allrho[ok]):.2f} "
  f"(deep in flip territory), share rho>2: {(allrho[ok]>2).mean()*100:.0f}%")

with open(os.path.join(C.OUT, "eval_summary.txt"), "w") as fo:
    fo.write("\n".join(lines) + "\n")
print("\nwrote eval_points.json, eval_summary.txt")
