import numpy as np
from common_d2c import *

rng = np.random.default_rng(0)
G = rng.standard_normal((300, 50))
D = distm(G)
thr = float(np.quantile(pair_values(D, 10), 0.9))
band = np.abs(np.subtract.outer(np.arange(300), np.arange(300))) <= 10
R = (D < thr) & (~band)

# brute force diagonal runs (offset by offset)
from collections import Counter
cnt = Counter()
for delta in range(-299, 300):
    if abs(delta) <= 10:
        continue
    if delta >= 0:
        a = np.arange(0, 300 - delta); b = a + delta
    else:
        b = np.arange(0, 300 + delta); a = b - delta
    run = 0
    for t in range(len(a)):
        if R[a[t], b[t]]:
            run += 1
        else:
            if run: cnt[run] += 1
            run = 0
    if run: cnt[run] += 1
tot_bf = sum(k * v for k, v in cnt.items())
det_bf = 1 - cnt[1] / tot_bf
print("brute force: tot pts", tot_bf, "R.sum", int(R.sum()), "DET", round(det_bf, 4))

# DP version on same matrix
cnt_dp, tot_dp = _run_length_hist(R)
idx = np.arange(len(cnt_dp))
print("DP: tot", tot_dp, "DET", round(1 - cnt_dp[1] / tot_dp, 4))
print("bf line-length hist:", dict(sorted(cnt.items())[:8]))

# direct shift invariance for iid gauss
A = D[:-1, :-1] < thr
B = D[1:, 1:] < thr
m = np.abs(np.subtract.outer(np.arange(299), np.arange(299))) > 10
print("iid: P(shift|rec) =", float((A & B & m).sum() / max((A & m).sum(), 1)),
      "RR =", float((D < thr).sum() / D.size))

# sanity: distance stats
dv = pair_values(D, 10)
print("med", round(float(np.median(dv)), 2), "thr(10pct)", round(thr, 2),
      "q05 q50 q95:", np.round(np.quantile(dv, [.05, .5, .95]), 2))
