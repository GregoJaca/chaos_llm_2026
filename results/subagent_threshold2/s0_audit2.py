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
GEOMS = {}
for nm in ["w100", "w200", "w500", "cyber", "w100tiny"]:
    GEOMS[nm] = C.load_case(nm)

GEOM_OF_SOURCE = {
    "a1_ds_w100/free_run_large": "w100", "a1_ds_w100/free_run": "w100tiny",
    "a1_ds_w200_large/free_run": "w200", "a1_ds_w500_large/free_run": "w500",
    "fr_w100_large.npz": "w100", "fr_cyber.npz": "cyber",
}

import json as _json
for e in C.load_free("all"):
    if e["npz"] is None:
        continue
    f, src = e["npz"], e["source"]
    ri = int(np.argmin(np.abs(f.radii - e["r"])))
    ok = (f.first_mismatch(ri, 0) == e["obs"][0])  # spot check d0
    obs_row = np.array([f.first_mismatch(ri, d) for d in range(f.K)])
    p(f"[E] {src} r={e['r']:g}: obs recomputed from gen_ids == stored obs: "
      f"{(obs_row == np.asarray(e['obs'])).all()}")
    g = GEOM_OF_SOURCE.get(src)
    if g:
        ref = np.asarray(f.batches[0][0])
        p(f"    reference row == {g} geometry gen_ids: "
          f"{(ref == GEOMS[g].gen_ids).all()}")

# cross-check: npz supersets match their summary jsons exactly
for src, fn in [("a1_ds_w100/free_run_large", "free_w100_large.json"),
                ("a1_ds_w200_large/free_run", "free_w200.json"),
                ("a1_ds_w500_large/free_run", "free_w500.json")]:
    recs = [e for e in C.load_free("all") if e["source"] == src]
    j = _json.load(open(os.path.join(C.BASE, fn)))
    same = all((np.asarray(recs[i]["obs"]).tolist() == j["obs"][i]) and
               (np.asarray(recs[i]["pred"]).tolist() == j["pred"][i])
               for i in range(len(recs)))
    p(f"[E] {src} == summary {fn} (obs and pred): {same}")

# ---------------------------------------------------------------- F: twin jsons
j = open(os.path.join(C.BASE, "free_w100_large.json")).read()
j2 = open(os.path.join(C.BASE, "free_w100_layers_large.json")).read()
p(f"[F] free_w100_large.json == free_w100_layers_large.json byte-identical: {j == j2}")
import json
ja, jb = json.loads(j), json.loads(j2)
p(f"    same obs: {ja['obs']==jb['obs']}, same pred: {ja['pred']==jb['pred']}, "
  f"runs: {ja['run']} vs {jb['run']}")
p(f"[F] note: those jsons summarize {ja['run']}/free_run_large, whose raw npz "
  f"(with sequences) is now used directly instead")

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
