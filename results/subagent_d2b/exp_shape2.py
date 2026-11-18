"""exp_shape2: the delay-embedding (scalar observable) route + reference shapes.

Scan: observables {norm, pc1, stepdist} x m in {2..10} x tau in {1,2,3,5,10},
      w=0 (notebook style), for launch and interstellar.
Fig 5: (a) launch PC1 C(eps), m-sweep, tau=1;  (b) local slopes + ER ceiling;
       (c) interstellar PC1 slopes;  (d) best candidate vs surrogates
       (time-shuffle / phase-rand x5 / IAAFT x3).
Fig 6: reference shapes (what "good" looks like): Henon, Lorenz delay
       embedding, AR(1) matched to PC1, and the launch PC1 embedding side
       by side: top row C(eps), bottom row local slope with plateau band.
Also: Theiler detail for cyber raw (where does the w-effect live in C).
JSON: shape2.json
"""
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cshape import (load_traj, transform, median_dist, c_curve, slope_curve,
                    fit_d2, plateau_scan, observables, embed, acf, henon,
                    lorenz, ar1, phase_rand_1d, RNG, jsonable)
from common import iaaft as iaaft1d  # copied from subagent_d2

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
t0 = time.time()


def curve_for_embed(vecs, times, theiler=0):
    Vc = vecs - vecs.mean(0, keepdims=True)
    A = Vc[RNG.choice(len(Vc), size=min(1500, len(Vc)), replace=False)]
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(len(A), 1)
    med = float(np.median(np.sqrt(D[iu])))
    eps = np.geomspace(0.002 * med, 3.0 * med, 64)
    C, npair, _ = c_curve(Vc, eps, theiler=theiler, times=times,
                          arith='f64', mode='raw')
    em, sl = slope_curve(eps, C, win=7)
    return eps, C, em, sl, med, npair


def scan_one(name, obs_name, s, ms, taus):
    out = {}
    for m in ms:
        for tau in taus:
            vecs, times = embed(s, m, tau)
            if len(vecs) < 100:
                continue
            eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
            Cmid = np.interp(np.log(em), np.log(eps), C)
            N = len(vecs)
            pl = plateau_scan(em, sl, Cmid, N, c_lo=10.0 / N)
            best_flat = pl[0] if pl else None
            # best 1-decade fit avoiding depletion/saturation
            best_fit = None
            for i in range(len(eps)):
                for j in range(i + 1, len(eps)):
                    if np.log10(eps[j] / eps[i]) < 1.0:
                        continue
                    msk = (eps >= eps[i]) & (eps <= eps[j])
                    if C[msk].max() > 0.9 or C[msk].min() < 10.0 / N:
                        continue
                    d2, r2, cnt = fit_d2(eps, C, C[msk].min(), C[msk].max())
                    if r2 is not None and np.isfinite(r2) and (best_fit is None or r2 > best_fit[1]):
                        best_fit = (float(d2), float(r2), float(eps[i] / med), float(eps[j] / med))
            k1 = np.argmax(C >= 0.1)
            k5 = np.argmax(C >= 0.5)
            sl1 = float((np.log(C[k1]) - np.log(C[k1 - 1])) / (np.log(eps[k1]) - np.log(eps[k1 - 1]))) if 0 < k1 < len(C) else None
            sl5 = float((np.log(C[k5]) - np.log(C[k5 - 1])) / (np.log(eps[k5]) - np.log(eps[k5 - 1]))) if 0 < k5 < len(C) else None
            out[f'm{m}_tau{tau}'] = {
                'N': int(N), 'med': float(med),
                'slope_at_C.1': sl1, 'slope_at_C.5': sl5,
                'widest_flat_dec': None if best_flat is None else best_flat['width_dec'],
                'widest_flat_slope': None if best_flat is None else best_flat['slope'],
                'best_1dec_fit_r2': None if best_fit is None else best_fit,
            }
    return out


MS = [2, 3, 4, 5, 6, 8, 10]
TAUS = [1, 2, 3, 5, 10]
for name in ['launch', 'interstellar']:
    X = load_traj(name)
    obs = observables(X)
    res[name] = {'scan': {}}
    for oname in ['pc1', 'norm', 'stepdist']:
        res[name]['scan'][oname] = scan_one(name, oname, obs[oname], MS, TAUS)
        top = sorted(((k, v) for k, v in res[name]['scan'][oname].items()),
                     key=lambda kv: -(kv[1]['widest_flat_dec'] or 0))[:3]
        print(f'scan {name}/{oname}: ' + '; '.join(
            f'{k}: flat={kv["widest_flat_dec"]}dec slope={kv["widest_flat_slope"]} '
            f'sl@C.1={kv["slope_at_C.1"]}' for k, kv in top))
    print(f'  ({time.time()-t0:.0f}s)')

# ------------------------------------------------------------------ FIG 5
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
X = load_traj('launch')
obs = observables(X)
ER = 2 * np.log10(3031)

ax = axs[0, 0]
for m in [2, 3, 4, 5, 6, 8, 10]:
    vecs, times = embed(obs['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(eps / med, np.maximum(C, 1e-9), lw=1.4, label=f'm={m}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'$C(\epsilon)$')
ax.axhline(1.0, color='k', ls=':', lw=0.6)
ax.set_title('(a) launch, PC1 delay-embedded (tau=1), m=2..10: sigmoid family,\n'
             'small-eps slope tracks the embedding dimension m, no plateau', fontsize=9)
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.25)

ax = axs[0, 1]
for m in [2, 3, 4, 5, 6, 8, 10]:
    vecs, times = embed(obs['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(em / med, sl, lw=1.4, label=f'm={m}')
ax.axhline(ER, color='red', ls='--', lw=1.2)
ax.text(2.5e-3, ER + 0.2, '2 log10(N) = %.2f' % ER, color='red', fontsize=8)
ax.set_xscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'd log C / d log $\epsilon$')
ax.set_ylim(-0.5, 12)
ax.set_title('(b) launch PC1 local slopes: small-eps slope = m (the embedding dimension),\n'
             'then decline -- slope is legal (< ceiling) but NEVER flat over >= 1 decade', fontsize=9)
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

X2 = load_traj('interstellar')
obs2 = observables(X2)
ax = axs[1, 0]
ER2 = 2 * np.log10(1026 - 10)
for m in [2, 3, 4, 5, 6, 8, 10]:
    vecs, times = embed(obs2['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(em / med, sl, lw=1.4, label=f'm={m}')
ax.axhline(ER2, color='red', ls='--', lw=1.2)
ax.set_xscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'd log C / d log $\epsilon$')
ax.set_ylim(-0.5, 12)
ax.set_title('(c) interstellar PC1 local slopes: same story at T=1026\n'
             '(slope ~ m at small eps, monotone decline, no flat decade)', fontsize=9)
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

# (d) best candidate + surrogates. Pick by widest flat across scan.
best_key, best_val = None, -1
for oname in ['pc1', 'norm', 'stepdist']:
    for k, v in res['launch']['scan'][oname].items():
        if (v['widest_flat_dec'] or 0) > best_val:
            best_val = v['widest_flat_dec'] or 0
            best_key = (oname, k)
oname, k = best_key
m = int(k.split('_')[0][1:]); tau = int(k.split('_tau')[1])
s = obs[oname]
vecs, times = embed(s, m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
ax = axs[1, 1]
ax.plot(eps / med, np.maximum(C, 1e-9), 'b-', lw=2.0,
        label=f'data: {oname}, m={m}, tau={tau} (widest flat = {best_val:.2f} dec)')
for lbl, series in [('time-shuffled', s[RNG.permutation(len(s))]),
                    ('phase-rand #1', phase_rand_1d(s)),
                    ('IAAFT #1', iaaft1d(np.asarray(s, dtype=np.float64), niter=40))]:
    v2, t2 = embed(series, m, tau)
    e2, C2, em2, sl2, med2, np2 = curve_for_embed(v2, t2, theiler=0)
    ax.plot(e2 / med2, np.maximum(C2, 1e-9), lw=1.2, ls='--', label=lbl)
res['launch']['best_candidate'] = {'obs': oname, 'm': m, 'tau': tau,
                                   'widest_flat_dec': best_val}
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_title('(d) best candidate vs surrogates: data == phase-randomized == IAAFT\n'
             '(any low-slope region is the power spectrum, not an attractor);\n'
             'time-shuffle DOES change this curve (unlike the raw cloud) because\n'
             'shuffling destroys the delay-embedding geometry itself', fontsize=9)
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.25)
fig.suptitle('FIG 5 -- the scalar-observable delay-embedding route (what a notebook usually does)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig5_delay_embedding.png'), dpi=125)
plt.close(fig)
print('fig5 done %.0fs; best candidate:' % (time.time() - t0), res['launch']['best_candidate'])

# ------------------------------------------------------------------ FIG 6
fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))

def panel_C(ax, eps, C, med, title, color='b'):
    ax.plot(eps / med, np.maximum(C, 1e-12), color=color, lw=1.8)
    ax.axhline(1.0, color='k', ls=':', lw=0.6)
    ax.axhline(1.0 / 4000, color='gray', ls=':', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)

def panel_S(ax, em, sl, med, N, title, d2ref=None, color='b'):
    ax.plot(em / med, sl, color=color, lw=1.8)
    ax.axhline(2 * np.log10(N), color='red', ls='--', lw=1.0)
    ax.text(2e-3, 2 * np.log10(N) + 0.25, '2 log10(N)', color='red', fontsize=7)
    if d2ref is not None:
        ax.axhspan(d2ref[0], d2ref[1], color='green', alpha=0.18,
                   label='plateau D2 in [%.2f, %.2f]' % d2ref)
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel('local slope')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)

# Henon
H = henon(4096)
eps, C, em, sl, med, npair = curve_for_embed(H, np.arange(len(H)), theiler=0)
# plateau: fit region C in [1e-4, 0.5]
d2, r2, cnt = fit_d2(eps, C, 1e-4, 0.5)
panel_C(axs[0, 0], eps, C, med,
        '(i) Henon map, N=4096, true D2~1.26:\nC(eps) is a straight line over ~3 decades')
panel_S(axs[1, 0], em, sl, med, 4096,
        'Henon local slope: FLAT at D2=%.2f over >2 decades (r2=%.4f)\nTHIS is what "good" looks like' % (d2, r2),
        d2ref=(d2 - 0.25, d2 + 0.25))
res['henon'] = {'d2_fit': d2, 'r2': r2, 'med': med}

# Lorenz delay embedding
L = lorenz(60000, dt=0.01)
sub = L[::20][:3031, 0]  # x-component, dt_eff=0.2, N=3031 (matches launch)
a = acf(sub, 30)
tau = next((k for k in range(1, 31) if a[k] <= 0), 15)
m = 8
vecs, times = embed(sub, m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=(m - 1) * tau)
d2L, r2L, cntL = fit_d2(eps, C, 1e-3, 0.3)
panel_C(axs[0, 1], eps, C, med,
        '(ii) Lorenz x(t), delay-embedded m=8, tau=%d, N=3031:\n'
        'textbook chaotic attractor (true D2~2.05)' % tau)
panel_S(axs[1, 1], em, sl, med, len(vecs),
        'Lorenz local slope: flat-ish D2=%.2f over ~1 decade (r2=%.3f)\n'
        'a REAL scaling region, reachable from a scalar at this N' % (d2L, r2L),
        d2ref=(d2L - 0.3, d2L + 0.3))
res['lorenz'] = {'tau': int(tau), 'm': m, 'd2_fit': d2L, 'r2': r2L}

# AR(1) matched to launch PC1 lag-1
rho = acf(obs['pc1'], 1)[1]
ar = ar1(3031, rho)
vecs, times = embed(ar, m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=(m - 1) * tau)
d2A, r2A, cntA = fit_d2(eps, C, 1e-3, 0.3)
panel_C(axs[0, 2], eps, C, med,
        '(iii) AR(1) with launch-PC1 lag-1 (rho=%.2f), same embedding:\n'
        'linear red noise makes the same kind of curve' % rho)
panel_S(axs[1, 2], em, sl, med, len(vecs),
        'AR(1) local slope: m at small eps, monotone decline, no flat decade\n'
        'same SHAPE class as our data -- not an attractor', d2ref=None)
res['ar1'] = {'rho': float(rho), 'm': m, 'tau': tau, 'd2_fit': d2A, 'r2': r2A}

# launch PC1 same embedding
vecs, times = embed(obs['pc1'], m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=(m - 1) * tau)
d2P, r2P, cntP = fit_d2(eps, C, 1e-3, 0.3)
panel_C(axs[0, 3], eps, C, med,
        '(iv) launch PC1, same embedding (m=%d, tau=%d): shape == AR(1)\n'
        'steep sigmoid, no scaling region' % (m, tau), color='purple')
panel_S(axs[1, 3], em, sl, med, len(vecs),
        'launch PC1 slope: no plateau; the delay route gives NO D2 either\n'
        'data is indistinguishable from linear red noise', color='purple', d2ref=None)
res['launch_pc1_ref'] = {'m': m, 'tau': tau, 'd2_fit': d2P, 'r2': r2P}

fig.suptitle('FIG 6 -- reference shapes: what a trustworthy C(eps) looks like, and where ours sits', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig6_reference_shapes.png'), dpi=125)
plt.close(fig)
print('fig6 done %.0fs' % (time.time() - t0))
print('henon D2=%.3f r2=%.4f | lorenz D2=%.3f r2=%.3f tau=%d | ar1 D2=%.2f r2=%.2f | launchPC1 D2=%.2f r2=%.2f'
      % (d2, r2, d2L, r2L, tau, d2A, r2A, d2P, r2P))

# ------------------------------------------------- Theiler detail for cyber raw
Xc = load_traj('cyber')
n = len(Xc)
medc = median_dist(Xc, 'center')
epsc = np.geomspace(0.001 * medc, 5 * medc, 96)
C0, _, _ = c_curve(Xc, epsc, theiler=0, arith='f64', mode='center')
det = {}
for w in [10, 50, 100]:
    Cw, _, _ = c_curve(Xc, epsc, theiler=w, arith='f64', mode='center')
    ok = (C0 > 0) & (Cw > 0)
    dl = np.abs(np.log(C0[ok]) - np.log(Cw[ok]))
    imax = np.argmax(dl)
    e_at = (epsc[ok])[imax]
    det[w] = {'max_dlogC': float(dl.max()), 'eps_at_max': float(e_at),
              'C0_at_max': float(C0[ok][imax]),
              'max_dlogC_for_C_gt_1e-2': float(np.max(dl[(C0[ok]) > 1e-2]))}
    print('cyber raw theiler w=%d: max dlogC=%.2f at eps=%.4f med (C0=%.2e); for C>1e-2: %.4f'
          % (w, det[w]['max_dlogC'], e_at / medc, det[w]['C0_at_max'], det[w]['max_dlogC_for_C_gt_1e-2']))
res['cyber_raw_theiler_detail'] = det

with open(os.path.join(OUT, 'shape2.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved shape2.json; total %.0fs' % (time.time() - t0))
