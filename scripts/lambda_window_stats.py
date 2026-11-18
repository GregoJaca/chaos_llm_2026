import os
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

B = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\window_sweep"


def mean_rate(w, t0=10, t1=100):
    d = np.load(os.path.join(B, f"w{w}", "a1.npz"), allow_pickle=True)
    L = list(d["layers"]); li = L.index(27); ri = int(np.argmin(np.abs(d["radii"] - 0.1)))
    A = np.maximum(d["dh_norm"][ri, :, li, :], 1e-12)
    k = 10
    lams = []
    for di in range(A.shape[0]):
        a = A[di]
        t = np.arange(k, 300 - k)
        lam = np.log(a[t + k] / a[t - k]) / (2 * k)
        sel = (t >= t0) & (t <= t1)
        lams.append(lam[sel].mean())
    return np.array(lams)


r50 = mean_rate(50)
r100 = mean_rate(100)
r200 = mean_rate(200)
print("mean rate over t=10..100 per direction")
for name, v in [("W=50", r50), ("W=100", r100), ("W=200", r200)]:
    print(f"  {name}: mean={v.mean():+.4f} median={np.median(v):+.4f} sd={v.std():.4f} n={len(v)}")
print("Mann-Whitney W50 vs W100 p =", mannwhitneyu(r50, r100, alternative="two-sided").pvalue)
print("Mann-Whitney W100 vs W200 p =", mannwhitneyu(r100, r200, alternative="two-sided").pvalue)
print("paired directions (same dir index) Wilcoxon W50 vs W100 p =",
      wilcoxon(r50[:10], r100[:10]).pvalue)
