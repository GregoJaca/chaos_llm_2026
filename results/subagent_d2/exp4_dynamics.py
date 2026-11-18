"""Exp4: honest dynamics tests on the NON-loop trajectories (interstellar, launch).

(0) NN-distance distribution (clustery-ness diagnostic).
(1) kNN prediction skill x horizon: data vs surrogate hierarchy:
      time-shuffled (statics only), phase-randomized x7 (2nd-order only),
      MIAAFT x3 (2nd-order + marginals), cov-Gauss, OU, plus persistence
      and reduced-rank ridge (linear ceiling).
(2) Kaplan-Glass continuation curves + determinism score, data vs surrogates.
Also: per-coordinate lag-1 autocorr (what the phase null preserves).
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.dirname(os.path.abspath(__file__))
H = [1, 2, 4, 8, 16, 32]
res = {}

for name in ['interstellar', 'launch']:
    t0 = time.time()
    X = load_traj(name)
    n = len(X)
    scale = median_pairwise_dist(X)
    W = 600 if name == 'launch' else 150  # launch: avoid partial recurrence at lag ~483
    half = n // 2
    r = {'n': n, 'scale': scale, 'W': W}

    # (0) NN-distance distribution over the full set
    nb_i, nb_d = knn_neighbors(X, k=3, W=0, test_slice=(0, n))
    r1 = nb_d[:, 0] / scale
    r['nn_dist_quantiles'] = {str(q): float(np.quantile(r1, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    r['lag1_autocorr_median'] = float(np.median([
        np.corrcoef(X[:-1, c].astype(np.float64), X[1:, c].astype(np.float64))[0, 1]
        for c in range(0, 1536, 17)]))
    print(f'{name}: NN r1/scale q50={r["nn_dist_quantiles"]["0.5"]:.3f} '
          f'lag1 acorr med={r["lag1_autocorr_median"]:.3f}')

    # (1) prediction: data + surrogate hierarchy
    variants = {'data': X, 'shuffled': time_shuffle(X)}
    phase = [phase_rand(X, rng=np.random.default_rng(1000 + i)) for i in range(7)]
    variants['phase'] = phase
    mia = []
    for i in range(3):
        m, q = miaaft(X, niter=10, rng=np.random.default_rng(2000 + i))
        mia.append(m)
    variants['miaaft'] = mia
    variants['covGauss'] = cov_gauss(X)
    variants['OU'] = ou_null(X)
    r['pred'] = {}
    for tag, Z in variants.items():
        Zs = Z if isinstance(Z, list) else [Z]
        sk_list, det_list = [], []
        for Zv in Zs:
            nb_i, nb_d = knn_neighbors(Zv, k=20, W=W, test_slice=(half, n - max(H) - 1))
            sk_list.append(predict_from_neighbors(Zv, nb_i, nb_d, H, k=12, test_offset=half))
            det_list.append({h: determinism_score(Zv, nb_i, nb_d, h, test_offset=half)
                             for h in (1, 8, 16)})
        r['pred'][tag] = {
            'nse_knn': {str(h): {'mean': float(np.mean([s[h]['nse_knn'] for s in sk_list])),
                                 'sd': float(np.std([s[h]['nse_knn'] for s in sk_list]))} for h in H},
            'nse_persist': {str(h): float(np.mean([s[h]['nse_persist'] for s in sk_list])) for h in H},
            'det': {str(h): {'mean': float(np.mean([d[h][0] for d in det_list])),
                             'sd': float(np.std([d[h][0] for d in det_list]))} for h in (1, 8, 16)}}
        print(f'  {name} pred[{tag}]: h=8 NSE={r["pred"][tag]["nse_knn"]["8"]["mean"]:.3f}'
              f'+-{r["pred"][tag]["nse_knn"]["8"]["sd"]:.3f}'
              f'  det(h=8)={r["pred"][tag]["det"]["8"]["mean"]:.2f}')

    # linear ceiling: ridge on data and on one phase surrogate
    r['ridge_data'] = {str(h): v for h, v in ridge_pcskill(X, H).items()}
    r['ridge_phase'] = {str(h): v for h, v in ridge_pcskill(phase[0], H).items()}

    # (2) continuation curves for the data (figure material)
    nb_i, nb_d = knn_neighbors(X, k=20, W=W, test_slice=(half, n - max(H) - 1))
    r['cont_data'] = {str(h): continuation_curves(X, nb_i, nb_d, [h], test_offset=half)[h] for h in (1, 8, 16)}
    nbs, nbds = knn_neighbors(phase[0], k=20, W=W, test_slice=(half, n - max(H) - 1))
    r['cont_phase'] = {str(h): continuation_curves(phase[0], nbs, nbds, [h], test_offset=half)[h] for h in (1, 8, 16)}
    res[name] = r
    print(f'  {name} done ({time.time()-t0:.0f}s)  ridge data h=8: '
          f'{r["ridge_data"]["8"]["nse_ridge"]:.3f}  ridge phase h=8: {r["ridge_phase"]["8"]["nse_ridge"]:.3f}')

# ---- figure 4a: NSE vs horizon
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, name in zip(axes, ['interstellar', 'launch']):
    r = res[name]
    for tag, col, ls in [('data', 'k', '-'), ('phase', 'tab:blue', '-'), ('miaaft', 'tab:cyan', '-.'),
                         ('shuffled', 'tab:red', '--'), ('covGauss', 'tab:orange', ':'), ('OU', 'tab:green', ':')]:
        m = [r['pred'][tag]['nse_knn'][str(h)]['mean'] for h in H]
        s = [r['pred'][tag]['nse_knn'][str(h)]['sd'] for h in H]
        ax.plot(H, m, ls, color=col, marker='o', ms=4, label=tag)
        if np.sum(s) > 0:
            ax.fill_between(H, np.array(m) - np.array(s), np.array(m) + np.array(s), color=col, alpha=0.15)
    mp = [r['pred']['data']['nse_persist'][str(h)] for h in H]
    ax.plot(H, mp, '--', color='gray', marker='s', ms=3, label='persistence (data)')
    mr = [r['ridge_data'][str(h)]['nse_ridge'] for h in H]
    ax.plot(H, mr, '-', color='purple', marker='^', ms=4, label='ridge-PC50 (data)')
    mrp = [r['ridge_phase'][str(h)]['nse_ridge'] for h in H]
    ax.plot(H, mrp, '--', color='purple', marker='^', ms=4, alpha=0.6, label='ridge-PC50 (phase surr)')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('prediction horizon h (tokens)'); ax.set_ylabel('NSE')
    ax.set_title(f'{name} (n={r["n"]}, W={r["W"]})'); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    ax.set_ylim(-1.3, 1.05)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4a_prediction.png'), dpi=130); plt.close(fig)

# ---- figure 4b: continuation curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, name in zip(axes, ['interstellar', 'launch']):
    r = res[name]
    for h, col in [(1, 'tab:blue'), (8, 'tab:orange'), (16, 'tab:green')]:
        cd = r['cont_data'][str(h)]
        cp = r['cont_phase'][str(h)]
        ax.plot(np.array(cd['bin_r']) / r['scale'], np.array(cd['bin_rc']) / r['scale'], 'o-', color=col,
                label=f'data h={h}')
        ax.plot(np.array(cp['bin_r']) / r['scale'], np.array(cp['bin_rc']) / r['scale'], 'x--', color=col,
                alpha=0.6, label=f'phase surr h={h}')
    ax.plot([1e-3, 1], [1e-3, 1], 'k:', lw=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'neighbour distance $r$/scale'); ax.set_ylabel(r'continuation distance $r_c$/scale')
    ax.set_title(f'{name}: Kaplan-Glass continuation'); ax.legend(fontsize=7); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig4b_continuation.png'), dpi=130); plt.close(fig)

with open(os.path.join(OUT, 'exp4_dynamics.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved exp4_dynamics.json')
