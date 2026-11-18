"""exp_shape3: the decisive follow-ups for the delay-embedding route.

1. Fixed Lorenz reference (dt_eff=0.04, tau from ACF first zero, m=8):
   a REAL D2 plateau must appear with the same detector used on our data.
2. FIG 7: D2(m) vs m (the saturation test). References saturate (Lorenz->~2);
   white noise / AR(1) ride the y=m diagonal until the ER ceiling clamps them.
3. Surrogate verdicts for the two "good-looking" cases:
   (a) launch norm, m=2, tau=5  (0.96-decade slope-1.95 'plateau'):
       phase-rand x5, IAAFT x3, time-shuffle, D2(N) at 768/1536/full,
       Theiler w=0 vs w=(m-1)tau.
   (b) launch pc1, m=8, tau=1  (apparent 'saturated' slope 2.25):
       same surrogates -- phase-rand must reproduce it (spectrum artifact),
       time-shuffle must destroy it (fill-dim ~m=8, above ER ceiling).
4. Rewritten FIG 6 with the fixed Lorenz and honest AR(1) comparison.

JSON: shape3.json
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
from cshape import (load_traj, transform, c_curve, slope_curve, fit_d2,
                    plateau_scan, observables, embed, acf, henon, lorenz,
                    ar1, phase_rand_1d, RNG, jsonable)
from common import iaaft as iaaft1d

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}
t0 = time.time()


def curve_for_embed(vecs, times, theiler=0, ne=64):
    Vc = vecs - vecs.mean(0, keepdims=True)
    A = Vc[RNG.choice(len(Vc), size=min(1500, len(Vc)), replace=False)]
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(len(A), 1)
    dd = np.sqrt(D[iu])
    med = float(np.median(dd[dd > 0])) if (dd > 0).any() else float(np.median(dd))
    eps = np.geomspace(0.002 * med, 3.0 * med, ne)
    C, npair, _ = c_curve(Vc, eps, theiler=theiler, times=times,
                          arith='f64', mode='raw')
    em, sl = slope_curve(eps, C, win=7)
    return eps, C, em, sl, med, npair


def slope_at(eps, C, tg):
    k = np.argmax(C >= tg)
    if k == 0 or C[k] < tg:
        return None
    return float((np.log(C[k]) - np.log(C[k - 1])) / (np.log(eps[k]) - np.log(eps[k - 1])))


def widest_flat(eps, C, em, sl, N, c_lo=None):
    if c_lo is None:
        c_lo = 10.0 / N
    Cmid = np.interp(np.log(em), np.log(eps), C)
    pl = plateau_scan(em, sl, Cmid, N, c_lo=c_lo)
    return pl[0] if pl else None


# ---------------------------------------------------------------- references
NREF = 3031
L = lorenz(60000, dt=0.01)
lx = L[::4][:NREF, 0]  # dt_eff = 0.04
a = acf(lx, 25)
tauL = next((k for k in range(1, 26) if a[k] <= 0), 4)
mL = 8
vecsL, timesL = embed(lx, mL, tauL)
epsL, CL, emL, slL, medL, npL = curve_for_embed(vecsL, timesL, theiler=(mL - 1) * tauL)
plL = widest_flat(epsL, CL, emL, slL, len(vecsL))
d2L, r2L = None, None
if plL:
    msk = (epsL >= plL['eps0'] * 0.9) & (epsL <= plL['eps1'] * 1.1)
    d2L, r2L, cntL = fit_d2(epsL, CL, CL[msk].min(), CL[msk].max())
res['lorenz'] = {'dt_eff': 0.04, 'tau': int(tauL), 'm': mL, 'N': int(len(vecsL)),
                 'plateau': plL, 'd2_fit': d2L, 'r2': r2L,
                 'slope_at_C.1': slope_at(epsL, CL, 0.1)}
print('LORENZ fixed: tau=%d m=%d plateau=%s D2=%.3f r2=%.3f' % (
    tauL, mL, plL and ('%.2f dec @ slope %.2f' % (plL['width_dec'], plL['slope'])), d2L or -1, r2L or -1))

# Henon reference (direct 2-D states)
H = henon(NREF)
epsH, CH, emH, slH, medH, npH = curve_for_embed(H, np.arange(len(H)), ne=96)
plH = widest_flat(epsH, CH, emH, slH, len(H), c_lo=1e-4)
d2H, r2H, _ = fit_d2(epsH, CH, 1e-4, 0.5)
res['henon'] = {'N': int(len(H)), 'plateau': plH, 'd2_fit': d2H, 'r2': r2H}
print('HENON: plateau=%.2f dec @ %.2f, fit D2=%.3f r2=%.4f' % (
    plH['width_dec'], plH['slope'], d2H, r2H))

# white noise + AR(1) m-sweeps (tau=1)
MS = [2, 3, 4, 5, 6, 8, 10]
def msweep(series, taus=(1,), theiler_from_overlap=False):
    out = {}
    for m in MS:
        vecs, times = embed(series, m, 1)
        w = (m - 1) if theiler_from_overlap else 0
        eps, C, em, sl, med, np_ = curve_for_embed(vecs, times, theiler=w)
        pl = widest_flat(eps, C, em, sl, len(vecs))
        out[m] = {'slope_at_C.1': slope_at(eps, C, 0.1),
                  'flat_dec': None if pl is None else pl['width_dec'],
                  'flat_slope': None if pl is None else pl['slope']}
    return out

rng = np.random.default_rng(12345)
white = rng.standard_normal(NREF)
Xl = load_traj('launch')
obs = observables(Xl)
rho = acf(obs['pc1'], 1)[1]
res['white'] = msweep(white)
res['ar1_tau1'] = msweep(ar1(NREF, rho))
res['lorenz_msweep'] = msweep(lx, theiler_from_overlap=False)
print('m-sweeps done %.0fs' % (time.time() - t0))

# our data (tau=1) -- reuse scan from shape2.json
old = json.load(open(os.path.join(OUT, 'shape2.json')))
for name in ['launch', 'interstellar']:
    for oname in ['pc1', 'norm', 'stepdist']:
        row = {}
        for m in MS:
            k = 'm%d_tau1' % m
            if k in old[name]['scan'][oname]:
                v = old[name]['scan'][oname][k]
                row[m] = {'slope_at_C.1': v['slope_at_C.1'],
                          'flat_dec': v['widest_flat_dec'],
                          'flat_slope': v['widest_flat_slope']}
        res.setdefault(name, {}).setdefault('msweep_tau1', {})[oname] = row

# ---------------------------------------------------------------- FIG 7
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ER = 2 * np.log10(NREF)
for a_ in ax:
    a_.plot(MS, MS, 'k:', lw=1.0, label='y = m (fill-dimension diagonal)')
    a_.axhline(ER, color='red', ls='--', lw=1.2)
    a_.axhspan(ER, 13, color='red', alpha=0.08)
    a_.text(9.6, ER + 0.3, '2 log10(N)=%.2f: unmeasurable' % ER, color='red',
            fontsize=8, ha='right')
    a_.set_xlabel('embedding dimension m')
    a_.set_ylabel('local slope  d log C / d log eps   at C=0.1')
    a_.set_xticks(MS)
    a_.grid(alpha=0.25)

def gety(row):
    return [row[m]['slope_at_C.1'] if row.get(m) and row[m]['slope_at_C.1'] else np.nan for m in MS]

ax[0].plot(MS, gety(res['lorenz_msweep']), 'g-o', lw=1.8, ms=5,
           label='Lorenz x(t), tau=%d (chaotic attractor)' % tauL)
ax[0].plot(MS, gety(res['white']), 'k--s', lw=1.5, ms=4, label='white Gaussian noise')
ax[0].plot(MS, gety(res['ar1_tau1']), 'm-^', lw=1.5, ms=4,
           label='AR(1), rho=%.2f (= launch PC1 lag-1)' % rho)
ax[0].plot(2, d2H, 'b*', ms=14, label='Henon map (direct 2-D states): true D2~1.26')
ax[0].set_ylim(0, 12)
ax[0].set_title('(a) REFERENCES: Lorenz saturates at D2~%.1f (a real attractor);\n'
                'white noise and AR(1) ride the y=m diagonal to the ceiling' % (plL['slope'] if plL else 2.0), fontsize=9.5)
ax[0].legend(fontsize=8, loc='upper left')

for oname, lbl, mk in [('pc1', 'launch PC1', 'o'), ('norm', 'launch ||x_t||', 's'),
                       ('stepdist', 'launch ||x_t - x_{t-1}||', '^')]:
    ax[1].plot(MS, gety(res['launch']['msweep_tau1'][oname]), '-' + mk, lw=1.6, ms=4,
               label=lbl)
ax[1].plot(MS, gety(res['interstellar']['msweep_tau1']['pc1']), '--o', lw=1.4, ms=4,
           color='gray', label='interstellar PC1')
ax[1].plot(MS, gety(res['lorenz_msweep']), 'g-o', lw=1.2, ms=3, alpha=0.5,
           label='Lorenz (reference)')
ax[1].set_ylim(0, 12)
ax[1].set_title('(b) OUR DATA: all three observables of launch + interstellar ride the\n'
                'noise diagonal (with mild red-shift), NO saturation below the ceiling\n'
                '-> the scalar/delay route yields no attractor dimension on this data', fontsize=9.5)
ax[1].legend(fontsize=8, loc='upper left')
fig.suptitle('FIG 7 -- the D2(m) saturation test: references saturate, our trajectories do not', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, 'fig7_saturation_test.png'), dpi=125)
plt.close(fig)
print('fig7 done %.0fs' % (time.time() - t0))

# ------------------------------------------------ surrogate verdicts (a) m=2
s = obs['norm']
m, tau = 2, 5
vecs, times = embed(s, m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
pl_data = widest_flat(eps, C, em, sl, len(vecs))
Cw, _, _ = c_curve(vecs - vecs.mean(0, keepdims=True), eps, theiler=(m - 1) * tau,
                   times=times, arith='f64', mode='raw')
okw = (C > 0) & (Cw > 0)
theiler_delta = float(np.max(np.abs(np.log(C[okw]) - np.log(Cw[okw]))))
res['case_a'] = {'obs': 'norm', 'm': m, 'tau': tau, 'N': int(len(vecs)),
                 'plateau': pl_data, 'theiler_w_eq_m1tau_max_dlogC': theiler_delta}
print('CASE A (norm m=2 tau=5): plateau %.2f dec @ slope %.2f ; theiler d=%.4f' % (
    pl_data['width_dec'], pl_data['slope'], theiler_delta))

surs = {'phase_rand': [phase_rand_1d(s, rng=np.random.default_rng(1000 + i)) for i in range(5)],
        'iaaft': [iaaft1d(np.asarray(s, dtype=np.float64), niter=40,
                          rng=np.random.default_rng(2000 + i)) for i in range(3)],
        'time_shuffle': [s[np.random.default_rng(3000 + i).permutation(len(s))] for i in range(1)]}
res['case_a']['surrogates'] = {}
for kind, lst in surs.items():
    rows = []
    for sr in lst:
        v2, t2 = embed(sr, m, tau)
        e2, C2, em2, sl2, med2, _ = curve_for_embed(v2, t2, theiler=0)
        pl = widest_flat(e2, C2, em2, sl2, len(v2))
        ok = (C > 0) & (C2 > 0) & (C > 1e-4)
        dl = float(np.max(np.abs(np.log(C[ok]) - np.log(C2[ok])))) if ok.any() else np.nan
        rows.append({'flat_dec': None if pl is None else pl['width_dec'],
                     'flat_slope': None if pl is None else pl['slope'],
                     'max_dlogC_vs_data': dl})
    res['case_a']['surrogates'][kind] = rows
    print('   %s: flat %s, slope %s, maxdlogC vs data %s' % (
        kind,
        ['%.2f' % r['flat_dec'] for r in rows],
        ['%.2f' % r['flat_slope'] for r in rows],
        ['%.3f' % r['max_dlogC_vs_data'] for r in rows]))

# D2(N) stability for case A
res['case_a']['d2_vs_N'] = {}
for N in [768, 1536, 3036]:
    vN = vecs[:N]; tN = times[:N]
    eN, CN, emN, slN, medN, _ = curve_for_embed(vN, tN, theiler=0)
    pl = widest_flat(eN, CN, emN, slN, N)
    res['case_a']['d2_vs_N'][N] = None if pl is None else pl
    print('   N=%d: flat %.2f dec @ slope %.2f' % (N, pl['width_dec'], pl['slope']) if pl else '')

# ------------------------------------------------ surrogate verdicts (b) m=8
s = obs['pc1']
m, tau = 8, 1
vecs, times = embed(s, m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
res['case_b'] = {'obs': 'pc1', 'm': m, 'tau': tau, 'N': int(len(vecs)),
                 'slope_at_C.1_data': slope_at(eps, C, 0.1),
                 'flat_dec': (widest_flat(eps, C, em, sl, len(vecs)) or {}).get('width_dec')}
print('CASE B (pc1 m=8 tau=1): slope@C.1=%.2f' % res['case_b']['slope_at_C.1_data'])
res['case_b']['surrogates'] = {}
for kind, lst in [('phase_rand', [phase_rand_1d(s, rng=np.random.default_rng(4000 + i)) for i in range(5)]),
                  ('iaaft', [iaaft1d(np.asarray(s, dtype=np.float64), niter=40,
                                     rng=np.random.default_rng(5000 + i)) for i in range(3)]),
                  ('time_shuffle', [s[np.random.default_rng(6000).permutation(len(s))]])]:
    rows = []
    for sr in lst:
        v2, t2 = embed(sr, m, tau)
        e2, C2, em2, sl2, med2, _ = curve_for_embed(v2, t2, theiler=0)
        ok = (C > 0) & (C2 > 0) & (C > 1e-4)
        dl = float(np.max(np.abs(np.log(C[ok]) - np.log(C2[ok])))) if ok.any() else np.nan
        rows.append({'slope_at_C.1': slope_at(e2, C2, 0.1),
                     'flat_dec': (widest_flat(e2, C2, em2, sl2, len(v2)) or {}).get('width_dec'),
                     'max_dlogC_vs_data': dl})
    res['case_b']['surrogates'][kind] = rows
    print('   %s: slope@C.1 %s flat %s maxdlogC %s' % (
        kind, ['%.2f' % r['slope_at_C.1'] for r in rows],
        ['%.2f' % r['flat_dec'] for r in rows],
        ['%.3f' % r['max_dlogC_vs_data'] for r in rows]))

# ---------------------------------------------------------------- FIG 6 (rewrite)
fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))

def panel_C(ax, eps, C, med, title, color='b', label=None, ls='-'):
    ax.plot(eps / med, np.maximum(C, 1e-12), color=color, lw=1.8, ls=ls, label=label)
    ax.axhline(1.0, color='k', ls=':', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.25)

def panel_S(ax, em, sl, med, N, title, d2ref=None, color='b', label=None, ls='-', ylim=None):
    ax.plot(em / med, sl, color=color, lw=1.8, ls=ls, label=label)
    ax.axhline(2 * np.log10(N), color='red', ls='--', lw=1.0)
    if d2ref is not None:
        ax.axhspan(d2ref[0], d2ref[1], color='green', alpha=0.18,
                   label='plateau D2 in [%.2f, %.2f]' % d2ref)
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel('local slope')
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(ylim if ylim else (-0.5, 12))
    ax.grid(alpha=0.25)

panel_C(axs[0, 0], epsH, CH, medH, '(i) Henon map N=%d (true D2~1.26):\nstraight line over ~3 decades' % NREF)
panel_S(axs[1, 0], emH, slH, medH, NREF,
        'Henon: flat local slope at %.2f over >2 decades (r2=%.4f)\nTHIS is "good"' % (d2H, r2H),
        d2ref=(d2H - 0.25, d2H + 0.25), ylim=(-0.5, 3))
panel_C(axs[0, 1], epsL, CL, medL,
        '(ii) Lorenz x(t), m=8, tau=%d, N=%d:\ndelay-embedded chaotic attractor (true D2~2.05)' % (tauL, len(vecsL)))
panel_S(axs[1, 1], emL, slL, medL, len(vecsL),
        'Lorenz: flat D2=%.2f over %.2f decades (r2=%.3f)\nreal scaling region at this N' % (
            plL['slope'], plL['width_dec'], r2L),
        d2ref=(plL['slope'] - 0.3, plL['slope'] + 0.3), ylim=(-0.5, 5))

sA = ar1(NREF, rho)
vA, tA = embed(sA, 8, 1)
eA, CA, emA, slA, medA, _ = curve_for_embed(vA, tA, theiler=0)
vP, tP = embed(obs['pc1'], 8, 1)
eP, CP, emP, slP, medP, _ = curve_for_embed(vP, tP, theiler=0)
panel_C(axs[0, 2], eA, CA, medA,
        '(iii) AR(1) rho=%.2f (launch-PC1 lag-1), m=8, tau=1:\nsame "saturated" slope as our PC1' % rho,
        color='m', label='AR(1)')
axs[0, 2].plot(eP / medP, np.maximum(CP, 1e-12), color='purple', lw=1.6, label='launch PC1')
axs[0, 2].legend(fontsize=7, loc='lower right')
panel_S(axs[1, 2], emA, slA, medA, len(vA),
        'AR(1) slope@C.1=%.2f vs launch PC1 slope@C.1=%.2f:\nthe "D2" is the effective rank of red-noise\n'
        'embedding covariance, not an attractor' % (slope_at(eA, CA, 0.1), slope_at(eP, CP, 0.1)),
        color='m')
axs[1, 2].plot(emP / medP, slP, color='purple', lw=1.6)

sr = phase_rand_1d(s, rng=np.random.default_rng(4999))
vR, tR = embed(sr, 8, 1)
eR, CR, emR, slR, medR, _ = curve_for_embed(vR, tR, theiler=0)
sh = s[np.random.default_rng(6999).permutation(len(s))]
vS, tS = embed(sh, 8, 1)
eS, CS, emS, slS, medS, _ = curve_for_embed(vS, tS, theiler=0)
panel_C(axs[0, 3], eP, CP, medP,
        '(iv) launch PC1 m=8 tau=1 vs surrogates:\ndata == phase-randomized; time-shuffle is a different cloud',
        color='purple', label='data')
axs[0, 3].plot(eR / medR, np.maximum(CR, 1e-12), 'g--', lw=1.3, label='phase-rand')
axs[0, 3].plot(eS / medS, np.maximum(CS, 1e-12), 'r:', lw=1.3, label='time-shuffled')
axs[0, 3].legend(fontsize=7, loc='lower right')
panel_S(axs[1, 3], emP, slP, medP, len(vP),
        'slopes: data %.2f = phase-rand %.2f (spectrum artifact);\n'
        'time-shuffle %.2f ~ m=8 fill-dim, ABOVE the ceiling\n(2 log10 N=%.2f): unmeasurable' % (
            slope_at(eP, CP, 0.1), slope_at(eR, CR, 0.1), slope_at(eS, CS, 0.1), 2 * np.log10(len(vP))),
        color='purple')
axs[1, 3].plot(emR / medR, slR, 'g--', lw=1.3)
axs[1, 3].plot(emS / medS, slS, 'r:', lw=1.3)

fig.suptitle('FIG 6 -- reference shapes: a real scaling region (Henon, Lorenz) vs the red-noise shape class of our data', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig6_reference_shapes.png'), dpi=125)
plt.close(fig)
print('fig6 rewritten %.0fs' % (time.time() - t0))

with open(os.path.join(OUT, 'shape3.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved shape3.json; total %.0fs' % (time.time() - t0))
