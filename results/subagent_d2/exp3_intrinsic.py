"""Exp3: intrinsic dimension estimators (TwoNN, Levina-Bickel MLE, PCA)
for all trajectories vs static nulls (cov-Gauss, iid-Gauss) + shuffle identity.

TwoNN/MLE/PCA are functions of the point cloud only -> time-shuffled must
give identical values (verified numerically). cov-Gauss matches the data's
PCA spectrum exactly -> any TwoNN/MLE difference is due to cloud shape
beyond second moments; iid-Gauss matches only per-coordinate variances.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
TR = ['cyber', 'future', 'great', 'interstellar', 'launch', 'first']
KS = (2, 4, 8, 16, 32)

res = {}
for name in TR:
    t0 = time.time()
    X = load_traj(name)
    r = {'n': int(len(X))}
    r['pca'] = pca_stats(X)
    variants = {'raw': X, 'shuffled': time_shuffle(X), 'covGauss': cov_gauss(X), 'iidGauss': iid_gauss(X)}
    for tag, Z in variants.items():
        d2nn, nok = twonn(Z, theiler=0)
        mle = mle_dim(Z, ks=KS)
        r[tag] = {'twonn': d2nn, 'twonn_n': nok, 'mle': {str(k): v for k, v in mle.items()}}
    # TwoNN with a big Theiler window (removes trivial temporal neighbours)
    r['raw_twonn_theiler500'] = twonn(X, theiler=500)[0]
    res[name] = r
    print(f'{name}: PR={r["pca"]["pr"]:.1f} n90={r["pca"]["n90"]} '
          f'TwoNN raw={r["raw"]["twonn"]:.2f} shuf={r["shuffled"]["twonn"]:.2f} '
          f'covG={r["covGauss"]["twonn"]:.2f} iidG={r["iidGauss"]["twonn"]:.2f} '
          f'MLE(k=8): raw={r["raw"]["mle"]["8"]:.2f} covG={r["covGauss"]["mle"]["8"]:.2f} '
          f'({time.time()-t0:.0f}s)')

# ---- figure: bars for TwoNN + MLE(k) curves + PCA spectrum
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
names = TR
x = np.arange(len(names))
w = 0.2
for j, tag in enumerate(['raw', 'shuffled', 'covGauss', 'iidGauss']):
    vals = [res[nm][tag]['twonn'] for nm in names]
    axes[0].bar(x + (j - 1.5) * w, vals, w, label=tag)
axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, fontsize=8)
axes[0].set_ylabel('TwoNN dim'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25, axis='y')
axes[0].set_title('TwoNN: data vs static nulls')

for nm, col in zip(names, plt.cm.viridis(np.linspace(0, 0.95, len(names)))):
    mle = [res[nm]['raw']['mle'][str(k)] for k in KS]
    axes[1].plot(KS, mle, 'o-', color=col, label=nm, ms=4)
    mleG = [res[nm]['covGauss']['mle'][str(k)] for k in KS]
    axes[1].plot(KS, mleG, '--', color=col, alpha=0.6, ms=3)
axes[1].set_xscale('log'); axes[1].set_xlabel('k (neighbours)'); axes[1].set_ylabel('MLE dim (raw solid, covGauss dashed)')
axes[1].legend(fontsize=7); axes[1].grid(alpha=0.25); axes[1].set_title('Levina-Bickel MLE')

for nm, col in zip(names, plt.cm.viridis(np.linspace(0, 0.95, len(names)))):
    ev = res[nm]['pca']['eig']
    fr = ev / ev.sum()
    axes[2].semilogy(np.arange(1, len(fr) + 1), np.cumsum(fr), color=col, label=f"{nm} (PR={res[nm]['pca']['pr']:.0f})")
axes[2].set_xlabel('PC index'); axes[2].set_ylabel('cumulative variance fraction')
axes[2].legend(fontsize=7); axes[2].grid(alpha=0.25); axes[2].set_title('PCA spectra')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_intrinsic_dim.png'), dpi=130)
plt.close()
with open(os.path.join(OUT, 'exp3_intrinsic.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved exp3_intrinsic.json')
