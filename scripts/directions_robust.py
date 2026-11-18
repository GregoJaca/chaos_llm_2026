import numpy as np
from scipy.stats import spearmanr
p = r"C:\Users\hjaca\research\chaos_llm_2026\results\local_data\w100_layers\a1.npz"
d = np.load(p, allow_pickle=True)
m = d["margins"]; radii = d["radii"]
ri = int(np.argmin(np.abs(radii - 0.3)))
dm = d["dm"][ri] / radii[ri]
ties = np.argsort(m)[:8]
C = spearmanr(dm[:, ties]).correlation
print("Spearman between ties, all 20 directions:")
print(np.round(C, 2))
keep = [i for i in range(20) if i not in (10, 11, 12)]
C2 = spearmanr(dm[keep][:, ties]).correlation
print("\nSpearman excluding the hot trio 10,11,12:")
print(np.round(C2, 2))
print("\nmean off-diagonal (all / no hot):", round(float((C.sum() - 8) / 56), 2), round(float((C2.sum() - 8) / 56), 2))
# variance decomposition: how much of dm variance is direction vs tie
X = dm[:, ties]
print("\nstd across directions per tie:", np.round(X.std(0), 4))
print("std across ties per direction:", np.round(X.std(1), 4))
print("total std:", round(float(X.std()), 4))
