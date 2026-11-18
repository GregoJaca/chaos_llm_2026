import numpy as np, time, json
from common_d2c import load, distm, pair_values, PROMPTS

name = "cybersecurity"
X, ids = load(name)
t0 = time.time()
n = ids.size
vals, cnts = np.unique(ids, return_counts=True)
p = cnts / n
H = float(-(p * np.log2(p)).sum())
bi = {}
for t in range(n - 1):
    k = (int(ids[t]), int(ids[t + 1]))
    bi[k] = bi.get(k, 0) + 1
print(f"token unigram/bigram: {time.time()-t0:.2f}s  unique {len(vals)} H {H:.2f} bi {len(bi)}")

t0 = time.time()
longest = 0
for k in range(2, 60):
    seen = set()
    found = False
    for t in range(n - k + 1):
        key = tuple(int(x) for x in ids[t:t + k])
        if key in seen:
            found = True
            break
        seen.add(key)
    if not found:
        break
    longest = k
print(f"longest n-gram loop: {time.time()-t0:.2f}s  longest={longest}")

t0 = time.time()
Xc = X - X.mean(0)
D = distm(Xc)
iu = np.triu_indices(n, 1)
dv = D[iu]
lags = np.abs(iu[0] - iu[1])
print(f"distm+pair setup: {time.time()-t0:.2f}s")
t0 = time.time()
best = (np.inf, 0)
for P in range(2, 251):
    m = lags == P
    if m.sum() < 50:
        continue
    v = float(np.median(dv[m]) / np.median(dv))
    if v < best[0]:
        best = (v, P)
print(f"loop screen: {time.time()-t0:.2f}s best={best}")
t0 = time.time()
_, sv, _ = np.linalg.svd(Xc, full_matrices=False)
print(f"svd: {time.time()-t0:.2f}s")
