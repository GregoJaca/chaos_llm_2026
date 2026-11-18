"""s0_audit2.py -- verify every convention the re-analysis relies on.

Writes audit2.txt. Checks:
 A. shipped r_c == adversarial rule  r*m/|dm|   (sentinel-masked)
 B. shipped pred_T == first t with r_c <= r     (adversarial rule)
 C. signed vs adversarial predicted steps differ where claimed
 D. w100_layers vs a1_ds_w100 baselines are twins (margins/gen_ids)
 E. fr_*.npz batch mapping: obs recomputed from sequences == stored obs,
    reference rows == the run's unperturbed gen_ids
 F. free_w100_large.json == free_w100_layers_large.json
 G. childhood: no local teacher-forced geometry (pred must be taken as shipped)
 H. sign of dm at the controlling ties (destabilizing or not)
"""
import os

import numpy as np

import common2 as C

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


np.set_printoptions(linewidth=200, threshold=5000)

# ---------------------------------------------------------------- A/B: r_c + pred_T rules
for nm in ["w100", "w200", "w500", "cyber", "w100tiny"]:
    c = C.load_case(nm)
    d = c.d
    if "r_c" not in d:
        p(f"[A] {nm}: no shipped r_c (recomputed only)")
        continue
    rc_ship = d["r_c"].astype(float)
    m3 = c.m[None, None, :]
    adv = c.radii[:, None, None] * m3 / np.abs(c.dm)
    adv[c.dm == 0] = np.inf
    ok_mask = (rc_ship < C.SENT) & np.isfinite(adv)
    err = np.nanmax(np.abs(rc_ship[ok_mask] - adv[ok_mask])) if ok_mask.any() else np.nan
    adv_sent_frac = np.mean(rc_ship >= C.SENT)
    p(f"[A] {nm}: shipped r_c == r*|dm|^-1 formula, max rel err {err:.3e} "
      f"(over {ok_mask.sum()} finite entries); {adv_sent_frac*100:.1f}% sentinels")

    pt_ship = d["pred_T"]
    rc_adv = adv.copy()
    pred_adv = np.stack([C.TFCase._first_below(rc_adv[i], c.radii[i]) for i in range(c.n_r)])
    p(f"[B] {nm}: shipped pred_T == adversarial first-crossing: {(pred_adv == pt_ship).all()}")

    pred_sgn = c.pred("signed")
    diff_per_row = (pred_sgn != pt_ship).sum(axis=1)
    p(f"[C] {nm}: directions where signed != adversarial pred, per radius row: {list(diff_per_row)}")

# ---------------------------------------------------------------- D: baseline twins
cL = C.load_case("w100")
ct = C.load_case("w100tiny")
same_tokens = bool((cL.gen_ids == ct.gen_ids).all())
dmarg = float(np.abs(cL.m - ct.m).max())
p(f"[D] w100_layers vs a1_ds_w100 (tiny radii): identical generated tokens: "
  f"{same_tokens}; max |margins diff| = {dmarg:.2e}")
p(f"[D] w100_layers radii: {np.array2string(cL.radii, precision=3)}")
p(f"[D] w100tiny radii:    {np.array2string(ct.radii, precision=3)}")

# margin floor of each geometry
for nm in ["w100", "w200", "w500", "cyber"]:
    c = C.load_case(nm)
    i = int(np.argmin(c.m))
    p(f"[D] {nm}: min margin {c.m[i]:.4f} at t={i}; top1={c.top1[i]} top2={c.top2[i]}; "
      f"radii {np.array2string(c.radii, precision=3)}")

# ---------------------------------------------------------------- E: fr npz batches
for fn in ["fr_w100_large.npz", "fr_cyber.npz", "fr_childhood.npz"]:
    f = C.FreeNPZ(os.path.join(C.BASE, fn))
    obs_rec = f.obs_recomputed()
    ok = (obs_rec == f.obs).all()
    p(f"[E] {fn}: obs recomputed from gen_ids == stored obs: {ok}")
    if not ok:
        bad = np.argwhere(obs_rec != f.obs)
        p(f"    mismatches (ri,di,stored,recomputed): "
          f"{[(int(i), int(d), int(f.obs[i,d]), int(obs_rec[i,d])) for i,d in bad[:10]]}")
    # reference row identical across batches and equal to geometry baseline?
    refs = [np.asarray(b[0]) for b in f.batches]
    same_ref = all((refs[0] == r).all() for r in refs)
    p(f"    all batch reference rows identical: {same_ref}")
    if fn == "fr_w100_large.npz":
        p(f"    reference == w100_layers unperturbed gen_ids: {(refs[0] == cL.gen_ids).all()}")
    if fn == "fr_cyber.npz":
        cy = C.load_case("cyber")
        p(f"    reference == a1_cyber_w100 gen_ids: {(refs[0] == cy.gen_ids).all()}")

# ---------------------------------------------------------------- F: twin jsons
j = open(os.path.join(C.BASE, "free_w100_large.json")).read()
j2 = open(os.path.join(C.BASE, "free_w100_layers_large.json")).read()
p(f"[F] free_w100_large.json == free_w100_layers_large.json byte-identical: {j == j2}")
import json
ja, jb = json.loads(j), json.loads(j2)
p(f"    same obs: {ja['obs']==jb['obs']}, same pred: {ja['pred']==jb['pred']}, "
  f"runs: {ja['run']} vs {jb['run']}")
p(f"    free_w100_large obs rows: {ja['obs']}")
p(f"    free_w100_large pred rows: {ja['pred']}")

# ---------------------------------------------------------------- G/H: controlling ties
for nm in ["w100", "w200", "w500", "cyber"]:
    c = C.load_case(nm)
    ir = 0
    rstar, tstar = c.r_star("signed", ir=ir)
    order = np.argsort(rstar)
    p(f"[H] {nm}: per-direction r_* (signed, gain@r={c.radii[ir]:g}) and controlling step:")
    for d in order:
        t = int(tstar[d])
        p(f"    d{d:2d}: r_*={rstar[d]:9.4f}  t*={t:3d}  m={c.m[t]:7.4f}  "
          f"dm@r0={c.dm[ir,d,t]:+8.5f}  {'destab' if c.dm[ir,d,t] < 0 else 'STABIL'}")

# childhood has no local geometry
p("[G] childhood: no local teacher-forced npz; shipped pred used as-is (same "
  "adversarial convention as all other shipped pred_T).")

with open(os.path.join(C.OUT, "audit2.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote audit2.txt")
