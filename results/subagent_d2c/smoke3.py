import numpy as np
from common_d2c import *

# 1) pipeline sanity: iid Gaussian cloud -> DET should be ~ RR
rng = np.random.default_rng(0)
G = rng.standard_normal((1000, 50))
print("iid Gauss DET:", [d["det"] for d in rqa_from_D(distm(G))])

# 2) real data vs its time-shuffle
X, _ = load("interstellar_travel")
Xc = X - X.mean(0)
D = distm(Xc)
print("data DET:", [d["det"] for d in rqa_from_D(D)])
perm = rng.permutation(1000)
print("shuffled DET:", [d["det"] for d in rqa_from_D(distm(Xc[perm]))])
perm = rng.permutation(1000)
print("shuffled DET:", [d["det"] for d in rqa_from_D(distm(Xc[perm]))])

# 3) is D invariant under the diagonal shift? P(D[i+1,j+1]<thr | D[i,j]<thr)
thr = float(np.quantile(pair_values(D, 10), 0.9))
A = D[:-1, :-1] < thr
B = D[1:, 1:] < thr
m = np.abs(np.subtract.outer(np.arange(999), np.arange(999))) > 10
print("thr", round(thr, 1), " P(shift|rec) =", float((A & B & m).sum() / max((A & m).sum(), 1)))

# 4) step coherence: global mean step vs residuals; local step correlation
delta = np.diff(Xc, axis=0)
mu = delta.mean(0)
print("||mean step||", round(float(np.linalg.norm(mu)), 2),
      " mean||step||", round(float(np.mean(np.linalg.norm(delta, axis=1))), 2),
      " mean||step-meanstep||", round(float(np.mean(np.linalg.norm(delta - mu, axis=1))), 2))
# local coherence: corr of consecutive step vectors
cd = []
for t in range(0, 999 - 1):
    a, b = delta[t], delta[t + 1]
    cd.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("consecutive step-vector cosine: med %.3f  q10 %.3f q90 %.3f" %
      (np.median(cd), np.quantile(cd, .1), np.quantile(cd, .9)))
# windowed coherence: avg cosine of steps within +-25 of each other
cw = []
for t in range(200, 800):
    for s in range(t - 25, t + 25):
        if s == t: continue
        a, b = delta[t], delta[s]
        cw.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print("within-window(+-25) step cosine: med %.3f" % np.median(cw))

# 5) norm ACF (fixed)
n = np.linalg.norm(Xc, axis=1)
n = n - n.mean()
acf = np.correlate(n, n, "full")[999:] / (n @ n)
print("norm ACF lags 1,2,5,10,50,100:", np.round(acf[[1, 2, 5, 10, 50, 100]], 3))
