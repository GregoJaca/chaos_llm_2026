"""Shared data model for the threshold-prediction re-analysis (subagent_threshold2).

Conventions (verified in s0_audit2.py against the shipped arrays):

- margins m_t = z_top1 - z_top2 at every greedy step t of the unperturbed run.
- dm[i,d,t] = change of that margin when the prompt embedding is perturbed by
  radius radii[i] along direction d, evaluated teacher-forced on the
  *unperturbed* token sequence.
- A flip at step t requires the runner-up to overtake: m_t + dm <= 0,
  i.e. dm_t <= -m_t (only NEGATIVE dm can flip; positive dm is stabilizing).
- Local threshold radius (signed rule, the physically correct one):
      r_c-(t,d;i) = radii[i] * m_t / max(-dm[i,d,t], 0)      (+inf if dm >= 0)
  The shipped r_c uses |dm| instead (adversarial rule) -> over-predicts flips.
- Predicted first flip T(r,d) = first t with r_c-(t,d) <= r, else -1 (no flip).
- Free runs: obs[ri,d] = first generated step where the perturbed autoregressive
  run differs from the unperturbed reference run, -1 if never within 300 steps.
  In fr_*.npz, gen_ids[ri*nb + b] is an array whose row 0 is the reference and
  rows 1.. are directions 6b.. (nb=4 batches of chunk=6 per radius).
"""
import json
import os

import numpy as np

from manual_npz import read_npz

ROOT = r"C:\Users\hjaca\research\chaos_llm_2026\results"
BASE = os.path.join(ROOT, "local_data")
OUT = os.path.join(ROOT, "subagent_threshold2")

SENT = 1e29  # r_c sentinel for "cannot flip"


# ---------------------------------------------------------------- teacher-forced cases
class TFCase:
    """Teacher-forced geometry: baseline margins + dm(r, direction, step)."""

    def __init__(self, name, path):
        d = read_npz(path)
        self.name = name
        self.d = d
        self.m = d["margins"].astype(float)          # (T,)
        self.T = len(self.m)
        self.top1 = d["top1"]
        self.top2 = d["top2"]
        self.gen_ids = d["gen_ids"]
        self.radii = np.asarray(d["radii"], float)
        self.dm = d["dm"].astype(float)              # (n_r, K, T)
        self.n_r, self.K, _ = self.dm.shape
        self.m3 = self.m[None, None, :]             # broadcast view

    # signed / adversarial local threshold fields, shape (n_r, K, T)
    def rc_signed(self):
        den = np.maximum(-self.dm, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = self.radii[:, None, None] * self.m3 / den
        rc[den <= 0] = np.inf
        return rc

    def rc_adv(self):
        den = np.abs(self.dm)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = self.radii[:, None, None] * self.m3 / den
        rc[den <= 0] = np.inf
        return rc

    @staticmethod
    def _first_below(rc, r):
        """first t index where rc[t] <= r, else -1. rc: (K, T), r scalar."""
        hit = rc <= r
        any_ = hit.any(axis=1)
        out = np.full(rc.shape[0], -1, dtype=int)
        out[any_] = hit.argmax(axis=1)[any_]
        return out

    def pred(self, rule="signed"):
        """(n_r, K) predicted first-flip step, -1 = no flip."""
        rc = self.rc_signed() if rule == "signed" else self.rc_adv()
        return np.stack([self._first_below(rc[i], self.radii[i])
                         for i in range(self.n_r)])

    def r_star(self, rule="signed", ir=None):
        """Per-direction smallest threshold r_* and its step t_*.
        ir=None uses the linear gain from the smallest radius row."""
        if ir is None:
            ir = 0
        rc = self.rc_signed() if rule == "signed" else self.rc_adv()
        rc0 = rc[ir]                                  # (K, T)
        rstar = rc0.min(axis=1)
        tstar = rc0.argmin(axis=1)
        rstar[~np.isfinite(rstar)] = np.inf
        return rstar, tstar


def load_case(name):
    """The five teacher-forced geometries available locally."""
    if name == "w100":
        return TFCase("w100", os.path.join(BASE, "w100_layers", "a1.npz"))
    if name == "w200":
        return TFCase("w200", os.path.join(BASE, "w200", "a1.npz"))
    if name == "w500":
        return TFCase("w500", os.path.join(BASE, "w500", "a1.npz"))
    if name == "cyber":
        return TFCase("cyber", os.path.join(ROOT, "a1_cyber_w100", "a1.npz"))
    if name == "w100tiny":
        return TFCase("w100tiny", os.path.join(ROOT, "a1_ds_w100", "a1.npz"))
    raise KeyError(name)


# ---------------------------------------------------------------- free-run records
class FreeJSON:
    """obs/pred summary json (no sequences)."""

    def __init__(self, path):
        j = json.load(open(path))
        self.path = path
        self.run = j.get("run")
        self.radii = np.asarray(j["radii"], float)
        self.obs = np.asarray(j["obs"], int)          # (n_r, K)
        self.pred = np.asarray(j["pred"], int)
        self.K = self.obs.shape[1]


class FreeNPZ:
    """free_run npz with obs/pred AND raw generated sequences."""

    def __init__(self, path, chunk=6):
        z = np.load(path, allow_pickle=True)
        self.path = path
        self.radii = np.asarray(z["radii"], float)
        self.obs = np.asarray(z["obs"], int)           # (n_r, K)
        self.pred = np.asarray(z["pred"], int)
        self.n_r, self.K = self.obs.shape
        self.chunk = chunk
        self.nb = int(np.ceil(self.K / chunk))
        batches = list(z["gen_ids"])
        assert len(batches) == self.n_r * self.nb, (len(batches), self.n_r, self.nb)
        self.batches = batches

    def rows(self, ri, di):
        """(reference row, perturbed row) for radius index ri, direction di."""
        b = di // self.chunk
        arr = self.batches[ri * self.nb + b]
        ref, row = arr[0], arr[1 + di % self.chunk]
        return np.asarray(ref), np.asarray(row)

    def first_mismatch(self, ri, di):
        ref, row = self.rows(ri, di)
        n = min(len(ref), len(row))
        mm = np.where(row[:n] != ref[:n])[0]
        return int(mm[0]) if len(mm) else -1

    def obs_recomputed(self):
        return np.array([[self.first_mismatch(ri, di) for di in range(self.K)]
                         for ri in range(self.n_r)])


def load_free(name):
    """Free-run records: returns list of (radius, obs[K], pred[K], FreeNPZ|None,
    geometry_name|None) tuples, geometry_name = teacher-forced case to score
    against (None when only the shipped pred is usable)."""
    out = []
    # w100 geometry, small radii (summary jsons; both jsons are twins -> keep one)
    j = json.load(open(os.path.join(BASE, "free_w100_large.json")))
    j2 = json.load(open(os.path.join(BASE, "free_w100_layers_large.json")))
    same = (j["obs"] == j2["obs"]) and (j["radii"] == j2["radii"])
    for i, r in enumerate(j["radii"]):
        out.append(dict(source="free_w100_large.json", twin_json_identical=same,
                        r=float(r), obs=np.asarray(j["obs"][i]),
                        pred=np.asarray(j["pred"][i]), npz=None, geom="w100"))
    # w100 geometry, large radii with sequences
    fr = FreeNPZ(os.path.join(BASE, "fr_w100_large.npz"))
    for i, r in enumerate(fr.radii):
        out.append(dict(source="fr_w100_large.npz", r=float(r), obs=fr.obs[i],
                        pred=fr.pred[i], npz=fr, geom="w100"))
    # w200 / w500 (summary jsons)
    for nm, fn in [("w200", "free_w200.json"), ("w500", "free_w500.json")]:
        jw = json.load(open(os.path.join(BASE, fn)))
        for i, r in enumerate(jw["radii"]):
            out.append(dict(source=fn, r=float(r), obs=np.asarray(jw["obs"][i]),
                            pred=np.asarray(jw["pred"][i]), npz=None, geom=nm))
    # cyber / childhood with sequences
    for nm, fn, geom in [("cyber", "fr_cyber.npz", "cyber"),
                         ("childhood", "fr_childhood.npz", None)]:
        f = FreeNPZ(os.path.join(BASE, fn))
        for i, r in enumerate(f.radii):
            out.append(dict(source=fn, r=float(r), obs=f.obs[i], pred=f.pred[i],
                            npz=f, geom=geom, prompt=nm))
    for e in out:
        e.setdefault("prompt", "propulsion")
        e.setdefault("twin_json_identical", None)
    return out


# ---------------------------------------------------------------- divergence metrics
def agreement_curve(ref, row):
    """boolean array: row[t] == ref[t]."""
    n = min(len(ref), len(row))
    a = np.zeros(n, bool)
    a[:n] = row[:n] == ref[:n]
    return a


def after_flip_stats(ref, row, T, block=10, tail=50):
    """Divergence statistics for one perturbed continuation whose first
    difference from the reference is at step T (>=0)."""
    a = agreement_curve(ref, row)
    n = len(a)
    res = {}
    res["frac_identical_all"] = float(a.mean())
    aft = a[T:]                                   # agreement from the flip on
    res["n_after"] = len(aft)
    res["frac_identical_after"] = float(aft.mean()) if len(aft) else np.nan
    # windowed agreement in blocks of `block` steps after the flip
    wins = []
    for s in range(0, len(aft), block):
        wins.append(float(aft[s:s + block].mean()))
    res["block_agree"] = wins
    # does the continuation re-sync and stay synced for the final `tail` steps?
    synced = False
    resync_step = -1
    if len(aft):
        # last index where row differs from ref
        last_diff = np.where(~aft)[0]
        if len(last_diff) == 0:
            synced = True
            resync_step = 0
        elif n - 1 - (T + last_diff[-1]) >= tail:
            synced = True
            resync_step = int(T + last_diff[-1] + 1)
    res["resynced"] = synced
    res["resync_step"] = resync_step
    # growth: agreement in first block vs last half
    if len(aft) >= 2 * block:
        res["agree_first_block"] = float(aft[:block].mean())
        res["agree_last_half"] = float(aft[len(aft) // 2:].mean())
    else:
        res["agree_first_block"] = float(aft[:block].mean()) if len(aft) else np.nan
        res["agree_last_half"] = res["agree_first_block"]
    return res
