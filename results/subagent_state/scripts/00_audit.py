"""00_audit: verify data provenance, floors, and the attention-window question.

Checks (all read-only):
 1. Shapes, dtypes, dh(t<s)=0 exactly, response floors (no numerical-floor issue).
 2. Which baseline trajectory the state runs used (margin fingerprinting).
 3. local_data/a1.npz identity (md5) vs candidate baselines.
 4. Does dh persist beyond lag 99? (A hard 100-token sliding window in the
    teacher-forced forward would force dh(t>=s+100) == 0 exactly.)
 5. Logit-level linearity: |dm| vs the smallest baseline margin (would any
    near-tie flip at r=1e-3?).
"""
import hashlib, json, os
import numpy as np
from common import (REPO, load_state, load_state_mid, load_baseline, save_json,
                    A1_LOCAL_NPZ, MARGINS_W100)

rep = {}

st = load_state()
mid = load_state_mid()
base = load_baseline()
dh, steps = st["dh"], st["steps"]
T = dh.shape[-1]

rep["state"] = dict(dh_shape=list(dh.shape), steps=[int(s) for s in steps], r=st["r"],
                    layer=st["layer"], dm_shape=list(np.load(os.path.join(REPO, "results", "local_data", "state.npz"))["dm"].shape))
rep["state_mid"] = dict(dh_shape=list(mid["dh"].shape), steps=[int(s) for s in mid["steps"]],
                        r=mid["r"], inject_layer=mid["inject_layer"], measure_layer=mid["measure_layer"])

# 1. causality + floors
for si, s in enumerate(steps):
    rep.setdefault("causality", {})[int(s)] = dict(
        max_dh_before_s=float(dh[si, :, :s].max()),
        dh_at_s_median=float(np.median(dh[si, :, s])),
        min_median_curve=float(np.median(dh[si], axis=0)[s:].min()))
d2 = mid["dh"]
for si, s in enumerate(mid["steps"]):
    rep.setdefault("causality_mid", {})[int(s)] = dict(
        max_dh_before_s=float(d2[si, :, :s].max()),
        dh_at_s_median=float(np.median(d2[si, :, s])))

# 2. baseline fingerprint: a1_ds_w100 margins vs margins/w100.npz[:300]
w100 = np.load(MARGINS_W100)["margins"]
m = base["margins"]
rep["baseline_fingerprint"] = dict(
    corr_a1w100_margins_vs_marginsw100_first300=float(np.corrcoef(m, w100[:300])[0, 1]),
    argmin_a1w100=int(m.argmin()), min_a1w100=float(m.min()),
    near_ties_margin_lt_0p2=[int(t) for t in np.where(m < 0.2)[0]],
    near_tie_margins=[float(m[t]) for t in np.where(m < 0.2)[0]])

# 3. md5 of local_data/a1.npz vs candidates
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
cands = {}
for d in ["a1_ds_w100", "a1_ds_w200_large", "a1_ds_w500_large"]:
    p = os.path.join(REPO, "results", d, "a1.npz")
    if os.path.exists(p):
        cands[d] = md5(p)
rep["a1_npz_md5"] = dict(local_data=md5(A1_LOCAL_NPZ), candidates=cands,
                         local_data_equals=[d for d, h in cands.items() if h == md5(A1_LOCAL_NPZ)])

# 4. window question: median dh at lags beyond 99 (should be EXACTLY 0 under a hard
#    100-token sliding window applied to the teacher-forced forward)
win = {}
for si, s in enumerate(steps):
    med = np.median(dh[si], axis=0)
    l50 = med[s + 40:s + 60]
    l100 = med[s + 100:s + 120] if s + 120 <= T else med[s + 100:]
    win[int(s)] = dict(median_lag_40_60=float(np.median(l50)),
                       median_lag_100_120=float(np.median(l100)),
                       max_lag_ge_100=float(dh[si, :, s + 100:].max()))
rep["window_check"] = dict(
    note="under a hard 100-token sliding window, dh(t>=s+100) would be exactly 0",
    by_s=win)

# 5. logit-level linearity: |dm| vs margins
dm = np.load(os.path.join(REPO, "results", "local_data", "state.npz"))["dm"].astype(np.float64)
rep["dm_scale"] = dict(
    abs_dm_max=float(np.abs(dm).max()),
    abs_dm_median=float(np.median(np.abs(dm))),
    min_margin=float(m.min()),
    ratio_max_absdm_to_min_margin=float(np.abs(dm).max() / m.min()),
    any_flip=bool((np.abs(dm) > m[None, None, :]).any()))

save_json(rep, "audit.json")
print(json.dumps(rep, indent=1)[:3000])
