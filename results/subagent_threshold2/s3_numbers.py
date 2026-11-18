"""s3_numbers.py -- remaining numbers for figures 1-6.

 1. identity of the 2 signed-rule misses; sign of dm at the t=123 tie per
    direction (to choose the safe/green example for fig1)
 2. gain |dm|/r vs r merged over w100tiny (1e-6..0.01) + w100_layers (1e-3..2),
    incl. consistency of the two twin runs at r=1e-3, per-tie gain curves
 3. correct linear extrapolation (threshold field from small-r gain) vs
    same-radius rule vs observed, per radius
 4. borderline sharpness: among runs with rho in [0.5, 2], how many fully
    correct (signed rule)?
Writes numbers4figs.json.
"""
import json
import os

import numpy as np

import common2 as C

out = {}

# ---------------------------------------------------------------- 1. miss identities + fig1 choices
pts = json.load(open(os.path.join(C.OUT, "eval_points.json")))
misses = []
for row in pts:
    if row["pred_sgn"] is None:
        continue
    obs, pred = np.asarray(row["obs"]), np.asarray(row["pred_sgn"])
    for d in np.argwhere(pred != obs).flatten():
        misses.append(dict(source=row["source"], r=row["r"], d=int(d),
                           obs=int(obs[d]), pred=int(pred[d]),
                           ship=int(row["pred_ship"][d]), rho=row["rho"][d]))
out["signed_misses"] = misses

c100 = C.load_case("w100")
ir5 = int(np.argmin(np.abs(c100.radii - 0.5)))
dm123 = c100.dm[ir5, :, 123]
pred5 = c100.pred("signed")[ir5]
safe = np.where(pred5 < 0)[0]
green = int(safe[np.argmax(dm123[safe])])
out["fig1"] = dict(
    red_dir=12, red_r=0.5,          # d12 flips at t=123 at r=0.5 (verified)
    green_dir=green, green_r=0.5,
    dm123_all=dm123.tolist(),
    n_destab_123=int((dm123 < 0).sum()),
    m123=float(c100.m[123]),
    dm_red_123=float(c100.dm[ir5, 12, 123]),
    rc_red_123=float(c100.m[123] * 0.5 / max(-c100.dm[ir5, 12, 123], 1e-30)),
    m_min=float(c100.m.min()), m_median=float(np.median(c100.m)),
)

# ---------------------------------------------------------------- 2. gain vs r
ct = C.load_case("w100tiny")
ir_t = int(np.argmin(np.abs(ct.radii - 1e-3)))
ir_L = int(np.argmin(np.abs(c100.radii - 1e-3)))
dm_t, dm_L = ct.dm[ir_t], c100.dm[ir_L]
rel = np.abs(dm_t - dm_L) / np.maximum(np.abs(dm_L), 1e-30)
out["twin_dm_agreement_r1e-3"] = dict(
    median_rel_diff=float(np.median(rel)), p99_rel_diff=float(np.percentile(rel, 99)),
    abs_max=float(np.abs(dm_t - dm_L).max()))

gain_rows = []
for case, tag in [(ct, "w100tiny"), (c100, "w100_layers")]:
    for i, r in enumerate(case.radii):
        g = np.abs(case.dm[i]) / r
        gain_rows.append(dict(tag=tag, r=float(r), med=float(np.median(g)),
                               p25=float(np.percentile(g, 25)),
                               p75=float(np.percentile(g, 75))))
out["gain_vs_r"] = gain_rows

# per-tie gain curves (median over directions with negative dm at that tie)
ties = [2, 14, 16, 123, 143]
tie_curves = {}
for t in ties:
    rows = []
    for i, r in enumerate(c100.radii):
        v = c100.dm[i, :, t]
        neg = v[v < 0]
        rows.append(float(np.median(np.abs(neg)) / r) if len(neg) else np.nan)
    tie_curves[t] = rows
out["tie_gain_curves_w100"] = dict(radii=c100.radii.tolist(), curves=tie_curves)
out["tie_gain_curves_w100tiny"] = dict(
    radii=ct.radii.tolist(),
    curves={t: [float(np.median(np.abs(ct.dm[i, :, t][ct.dm[i, :, t] < 0])) / r)
                if (ct.dm[i, :, t] < 0).any() else np.nan
                for i, r in enumerate(ct.radii)] for t in ties})

# ---------------------------------------------------------------- 3. extrapolated rule
def extrap_pred(case, ir_from, r):
    rc = case.rc_signed()[ir_from]
    return np.array([int(np.argmax(rc[d] <= r)) if (rc[d] <= r).any() else -1
                     for d in range(case.K)])

extrap_rows = []
for row in pts:
    if row["pred_sgn"] is None:
        continue
    g = C.load_case(row["geom"])
    ir = row["ir"]
    obs = np.asarray(row["obs"])
    pred_same = np.asarray(row["pred_sgn"])
    pred_ext = extrap_pred(g, 0, row["r"])        # gain from smallest radius
    both = (obs >= 0) & (pred_ext >= 0)
    extrap_rows.append(dict(
        source=row["source"], r=row["r"],
        obs_flips=int((obs >= 0).sum()),
        same_flips=int((pred_same >= 0).sum()),
        ext_flips=int((pred_ext >= 0).sum()),
        same_exact=int(((obs >= 0) & (pred_same == obs)).sum() + ((obs < 0) & (pred_same < 0)).sum()),
        ext_exact=int(((obs >= 0) & (pred_ext == obs)).sum() + ((obs < 0) & (pred_ext < 0)).sum()),
        ext_steperr=int((pred_ext[both] != obs[both]).sum()),
        ext_steperr_max=int(np.abs(pred_ext[both] - obs[both]).max()) if both.any() else 0))
out["extrap_rows"] = extrap_rows

# ---------------------------------------------------------------- 4. borderline sharpness
tot_bd = ok_bd = 0
rho_all, ok_all = [], []
for row in pts:
    if row["rho"] is None:
        continue
    obs, pred = np.asarray(row["obs"]), np.asarray(row["pred_sgn"])
    rho = np.asarray(row["rho"])
    ok = pred == obs
    bd = np.abs(np.log2(rho)) <= 1.0
    tot_bd += int(bd.sum())
    ok_bd += int((bd & ok).sum())
    rho_all += list(rho)
    ok_all += list(ok)
out["borderline"] = dict(n_borderline=tot_bd, n_correct=ok_bd,
                          frac_correct=ok_bd / max(tot_bd, 1))

with open(os.path.join(C.OUT, "numbers4figs.json"), "w") as fo:
    json.dump(out, fo, indent=1)

print(json.dumps({k: v for k, v in out.items() if k != "gain_vs_r"}, indent=1)[:3000])
print("gain_vs_r rows:", len(gain_rows))
for gr in gain_rows:
    print(f"  r={gr['r']:<9g} med={gr['med']:.4f} [{gr['p25']:.4f},{gr['p75']:.4f}] {gr['tag']}")
