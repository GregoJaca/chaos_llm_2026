"""Where exactly do data and surrogate C(eps) differ at m=8? And does
interstellar show the same deviation? Signs and eps locations matter for
the honest write-up."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cshape import (load_traj, observables, embed, acf, c_curve, slope_curve,
                    phase_rand_1d, RNG, jsonable)
from common import iaaft as iaaft1d


def med_of(vecs):
    A = vecs[RNG.choice(len(vecs), size=min(1500, len(vecs)), replace=False)]
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(len(A), 1)
    dd = np.sqrt(D[iu])
    return float(np.median(dd[dd > 0]))


def cvec(vecs, times, eps, theiler=0):
    Vc = vecs - vecs.mean(0, keepdims=True)
    C, npair, cnt = c_curve(Vc, eps, theiler=theiler, times=times,
                            arith='f64', mode='raw')
    return C, cnt, npair


for name in ['launch', 'interstellar']:
    X = load_traj(name)
    obs = observables(X)
    s = obs['pc1']
    m, tau = 8, 1
    vecs, times = embed(s, m, tau)
    med = med_of(vecs - vecs.mean(0))
    eps = np.geomspace(0.002 * med, 3.0 * med, 64)
    Cd, cntd, npair = cvec(vecs, times, eps)
    print('====', name, 'PC1 m=8 tau=1; N=%d pairs=%d' % (len(vecs), npair))
    sur = {'phase_rand(5)': [phase_rand_1d(s, rng=np.random.default_rng(4000 + i)) for i in range(5)],
           'iaaft(3)': [iaaft1d(np.asarray(s, dtype=np.float64), niter=40,
                                rng=np.random.default_rng(5000 + i)) for i in range(3)],
           'time_shuffle(1)': [s[np.random.default_rng(6000).permutation(len(s))]]}
    for kind, lst in sur.items():
        Cs = []
        for sr in lst:
            v2, t2 = embed(sr, m, tau)
            C2, _, _ = cvec(v2, t2, eps)
            Cs.append(C2)
        Cs = np.array(Cs)
        for tgt in [0.05, 0.1, 0.2, 0.5]:
            kd = np.argmax(Cd >= tgt)
            if Cd[kd] < tgt:
                continue
            e_at = eps[kd] / med
            ratios = Cs[:, kd] / Cd[kd]
            print('   %-16s at C_data=%.2f (eps=%.3f med): C_surr/C_data = %s'
                  % (kind, tgt, e_at, ['%.3g' % r for r in ratios]))
        # overall: excess close pairs below eps=0.2*med
        k02 = np.searchsorted(eps, 0.2 * med) - 1
        print('   %-16s C(0.2 med): data=%.3e  surr=%s   ratio=%s'
              % (kind, Cd[k02], ['%.3e' % c for c in Cs[:, k02]],
                 ['%.2f' % (c / Cd[k02]) for c in Cs[:, k02]]))
    # slope at C=0.1 data
    k1 = np.argmax(Cd >= 0.1)
    print('   data slope at C=0.1: %.2f ; eps0.1=%.3f med' % (
        (np.log(Cd[k1]) - np.log(Cd[k1 - 1])) / (np.log(eps[k1]) - np.log(eps[k1 - 1])),
        eps[k1] / med))
