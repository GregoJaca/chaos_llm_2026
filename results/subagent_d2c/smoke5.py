import numpy as np
from common_d2c import *

rng = np.random.default_rng(0)
G = rng.standard_normal((1000, 50))
print("iid Gauss 50-D:", [(d["rr"], d["det"], d["lmean"], d["lam"]) for d in rqa_from_D(distm(G))])
G2 = rng.standard_normal((1000, 1536))
print("iid Gauss 1536-D:", [(d["rr"], round(d["det"], 3)) for d in rqa_from_D(distm(G2))])

X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
D = distm(Xc)
print("data:", [(d["rr"], d["det"], d["lmean"], d["lam"]) for d in rqa_from_D(D)])
for _ in range(2):
    perm = rng.permutation(1000)
    print("shuffled:", [(d["rr"], d["det"]) for d in rqa_from_D(distm(Xc[perm]))])
# shift invariance at the correct threshold
thr = float(np.quantile(pair_values(D, 10), 0.10))
A = D[:-1, :-1] < thr
B = D[1:, 1:] < thr
m = np.abs(np.subtract.outer(np.arange(999), np.arange(999))) > 10
print("data: P(shift|rec) =", float((A & B & m).sum() / max((A & m).sum(), 1)),
      " RR =", round(float((D < thr).sum() / D.size), 4))
A2 = D[:-1, :-1] < thr; m2 = m
Pind = float((D[10:, 10:] < thr).mean())
print("iid-independent baseline P =", Pind)
