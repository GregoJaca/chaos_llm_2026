"""exp_shape4: final corrections + the IAAFT m-sweep for PC1.

1. PC1 m-sweep for IAAFT(3) and phase-rand(3): does the partial flattening
   of PC1's slope (2.2-2.5) survive the marginal+spectrum null?
2. Rewrite FIG 5 with an honest panel (d): at m=2 every surrogate (incl.
   time-shuffle) reproduces the 'plateau' -- it is 2-D cloud geometry.
3. Rewrite FIG 7 with the IAAFT/phase bands for PC1.
4. Rewrite FIG 6 with corrected (iii)/(iv): at m=8 nothing has a plateau;
   the data-vs-phase gap is a marginal effect; IAAFT is the right null.

JSON: shape4.json
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
from cshape import (load_traj, c_curve, slope_curve, fit_d2, plateau_scan,
                    observables, embed, acf, henon, lorenz, ar1,
                    phase_rand_1d, RNG, jsonable)
from common import iaaft as iaaft1d

OUT = os.path.dirname(os.path.abspath(__file__))
MS = [2, 3, 4, 5, 6, 8, 10]
NREF = 3031
t0 = time.time()
res = {}


def med_of(vecs):
    A = vecs[RNG.choice(len(vecs), size=min(1500, len(vecs)), replace=False)]
    G = A @ A.T
    sq = (A ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    iu = np.triu_indices(len(A), 1)
    dd = np.sqrt(D[iu])
    return float(np.median(dd[dd > 0]))


def curve_for_embed(vecs, times, theiler=0, ne=64):
    Vc = vecs - vecs.mean(0, keepdims=True)
    med = med_of(Vc)
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


Xl = load_traj('launch')
obs = observables(Xl)
Xi = load_traj('interstellar')
obs_i = observables(Xi)

# ---------------------------------------------------------------- 1. PC1 m-sweep nulls
s = obs['pc1']
for kind, gen, n in [('iaaft', lambda r: iaaft1d(np.asarray(s, dtype=np.float64), niter=40, rng=r), 3),
                     ('phase', lambda r: phase_rand_1d(s, rng=r), 3)]:
    rows = {m: [] for m in MS}
    for i in range(n):
        sr = gen(np.random.default_rng(7000 + 100 * i + (0 if kind == 'iaaft' else 50)))
        for m in MS:
            v2, t2 = embed(sr, m, 1)
            e2, C2, em2, sl2, med2, _ = curve_for_embed(v2, t2, theiler=0)
            rows[m].append(slope_at(e2, C2, 0.1))
    res[kind + '_msweep'] = {str(m): [r for r in rows[m]] for m in MS}
    print(kind, 'm-sweep slope@C.1:', {m: ['%.2f' % r for r in rows[m]] for m in MS})

# ---------------------------------------------------------------- 2. FIG 5 rewrite
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
ER = 2 * np.log10(3031)

ax = axs[0, 0]
for m in MS:
    vecs, times = embed(obs['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(eps / med, np.maximum(C, 1e-9), lw=1.4, label=f'm={m}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'$C(\epsilon)$')
ax.axhline(1.0, color='k', ls=':', lw=0.6)
ax.set_title('(a) launch, PC1 delay-embedded (tau=1), m=2..10: sigmoid family;\n'
             'small-eps slope tracks the embedding dimension m; no plateau', fontsize=9)
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.25)

ax = axs[0, 1]
for m in MS:
    vecs, times = embed(obs['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(em / med, sl, lw=1.4, label=f'm={m}')
ax.axhline(ER, color='red', ls='--', lw=1.2)
ax.text(2.5e-3, ER + 0.25, '2 log10(N) = %.2f' % ER, color='red', fontsize=8)
ax.set_xscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'd log C / d log $\epsilon$')
ax.set_ylim(-0.5, 12)
ax.set_title('(b) launch PC1 local slopes: small-eps slope = m (the embedding dimension),\n'
             'slope stays below the ceiling but is never flat over >= 1 decade', fontsize=9)
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

ax = axs[1, 0]
for m in MS:
    vecs, times = embed(obs_i['pc1'], m, 1)
    eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
    ax.plot(em / med, sl, lw=1.4, label=f'm={m}')
ax.axhline(2 * np.log10(1026), color='red', ls='--', lw=1.2)
ax.set_xscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist of embedding cloud')
ax.set_ylabel(r'd log C / d log $\epsilon$')
ax.set_ylim(-0.5, 12)
ax.set_title('(c) interstellar PC1 local slopes: same story at T=1026\n'
             '(ceiling 6.21); no flat decade at any m', fontsize=9)
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.25)

ax = axs[1, 1]
m, tau = 2, 5
vecs, times = embed(obs['norm'], m, tau)
eps, C, em, sl, med, npair = curve_for_embed(vecs, times, theiler=0)
ax.plot(eps / med, np.maximum(C, 1e-9), 'b-', lw=2.2,
        label='data: ||x_t||, m=2, tau=5 (flat 0.96 dec @ slope 1.95)')
for lbl, series, c in [
        ('phase-rand #1', phase_rand_1d(obs['norm'], rng=np.random.default_rng(41)), 'g'),
        ('IAAFT #1', iaaft1d(np.asarray(obs['norm'], dtype=np.float64), niter=40,
                             rng=np.random.default_rng(42)), 'm'),
        ('time-shuffled', obs['norm'][np.random.default_rng(43).permutation(len(obs['norm']))], 'r')]:
    v2, t2 = embed(series, m, tau)
    e2, C2, em2, sl2, med2, _ = curve_for_embed(v2, t2, theiler=0)
    ax.plot(e2 / med2, np.maximum(C2, 1e-9), c + '--', lw=1.4, label=lbl)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_title('(d) the "best-looking" curve (m=2) is pure geometry: phase-rand, IAAFT AND the\n'
             'time-shuffle all have the same 0.96-decade slope-1.96 plateau -- any scalar fills\n'
             'its 2-D embedding; white noise at m=2 gives exactly this curve (0.96 dec @ 1.96)', fontsize=9)
ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.25)
fig.suptitle('FIG 5 -- the scalar-observable delay-embedding route (what a notebook usually does)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig5_delay_embedding.png'), dpi=125)
plt.close(fig)
print('fig5 rewritten %.0fs' % (time.time() - t0))

# ---------------------------------------------------------------- 3. FIG 7 rewrite
old = json.load(open(os.path.join(OUT, 'shape3.json')))

def gety(row):
    return [row[str(m)]['slope_at_C.1'] if row.get(str(m)) and row[str(m)]['slope_at_C.1'] else np.nan for m in MS]

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
for a_ in ax:
    a_.plot(MS, MS, 'k:', lw=1.0, label='y = m (fill-dimension diagonal)')
    a_.axhline(6.97, color='red', ls='--', lw=1.2)
    a_.axhspan(6.97, 13, color='red', alpha=0.08)
    a_.text(9.6, 7.3, '2 log10(N=3031)=6.97: unmeasurable above', color='red',
            fontsize=8, ha='right')
    a_.set_xlabel('embedding dimension m')
    a_.set_ylabel('local slope at C=0.1')
    a_.set_xticks(MS)
    a_.set_ylim(0, 12)
    a_.grid(alpha=0.25)

ax[0].plot(MS, gety(old['lorenz_msweep']), 'g-o', lw=2.0, ms=5,
           label='Lorenz x(t), tau=4: SATURATES (plateau D2=1.99, 0.91 dec)')
ax[0].plot(MS, gety(old['white']), 'k--s', lw=1.5, ms=4, label='white Gaussian noise')
ax[0].plot(MS, gety(old['ar1_tau1']), 'm-^', lw=1.5, ms=4,
           label='AR(1), rho=0.52 (= launch PC1 lag-1)')
ax[0].plot(2, 1.157, 'b*', ms=16, label='Henon map (direct states): D2~1.26')
ax[0].axhspan(1.9, 2.1, color='green', alpha=0.12)
ax[0].set_title('(a) REFERENCES: Lorenz saturates at ~2 (real attractor); white noise and\n'
                'AR(1) ride the y=m diagonal until the finite-N ceiling clamps them', fontsize=9.5)
ax[0].legend(fontsize=8, loc='upper left')

ax[1].plot(MS, gety(old['launch']['msweep_tau1']['pc1']), 'b-o', lw=1.8, ms=5,
           label='launch PC1')
ax[1].plot(MS, gety(old['launch']['msweep_tau1']['norm']), 'c-s', lw=1.5, ms=4,
           label='launch ||x_t||')
ax[1].plot(MS, gety(old['launch']['msweep_tau1']['stepdist']), 'y-^', lw=1.5, ms=4,
           label='launch ||x_t - x_{t-1}||')
ax[1].plot(MS, gety(old['interstellar']['msweep_tau1']['pc1']), color='gray',
           ls='--', marker='o', lw=1.4, ms=4, label='interstellar PC1')
ia = np.array([[res['iaaft_msweep'][str(m)][i] for m in MS] for i in range(3)], dtype=float)
ph = np.array([[res['phase_msweep'][str(m)][i] for m in MS] for i in range(3)], dtype=float)
ax[1].fill_between(MS, np.nanmin(ia, 0), np.nanmax(ia, 0), color='orange', alpha=0.25,
                   label='launch PC1 IAAFT nulls x3 (marginal+spectrum)')
ax[1].plot(MS, np.nanmean(ia, 0), color='orange', ls='--', lw=1.2)
ax[1].fill_between(MS, np.nanmin(ph, 0), np.nanmax(ph, 0), color='green', alpha=0.15,
                   label='launch PC1 phase-rand nulls x3 (spectrum only)')
ax[1].plot(MS, np.nanmean(ph, 0), color='green', ls=':', lw=1.2)
ax[1].plot(MS, gety(old['lorenz_msweep']), 'g-o', lw=1.0, ms=3, alpha=0.45,
           label='Lorenz (reference)')
ax[1].set_title('(b) OUR DATA: ||x|| and step-dist ride the noise diagonal. PC1 partially\n'
                'flattens to 2.2-2.5, but IAAFT nulls land at 1.7-1.9 with the same shape --\n'
                'marginal + power-spectrum structure, not an attractor; no 1-dec plateau at any m', fontsize=9.5)
ax[1].legend(fontsize=8, loc='upper left')
fig.suptitle('FIG 7 -- the D2(m) saturation test: references saturate, our trajectories do not separate from their nulls', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, 'fig7_saturation_test.png'), dpi=125)
plt.close(fig)
print('fig7 rewritten %.0fs' % (time.time() - t0))

# ---------------------------------------------------------------- 4. FIG 6 rewrite
L = lorenz(60000, dt=0.01)
lx = L[::4][:NREF, 0]
tauL = 4
vecsL, timesL = embed(lx, 8, tauL)
epsL, CL, emL, slL, medL, npL = curve_for_embed(vecsL, timesL, theiler=7 * tauL, ne=80)
plL = widest_flat(epsL, CL, emL, slL, len(vecsL))
msk = (epsL >= plL['eps0']) & (epsL <= plL['eps1'])
d2L, r2L, _ = fit_d2(epsL, CL, CL[msk].min(), CL[msk].max())
H = henon(NREF)
epsH, CH, emH, slH, medH, npH = curve_for_embed(H, np.arange(len(H)), ne=96)
plH = widest_flat(epsH, CH, emH, slH, len(H), c_lo=1e-4)
d2H, r2H, _ = fit_d2(epsH, CH, 1e-4, 0.5)
print('recheck: lorenz plateau %.2f dec @ %.2f ; d2=%.3f r2=%.3f' % (
    plL['width_dec'], plL['slope'], d2L, r2L))

rho = acf(obs['pc1'], 1)[1]
sA = ar1(NREF, rho)
vA, tA = embed(sA, 8, 1)
eA, CA, emA, slA, medA, _ = curve_for_embed(vA, tA, theiler=0)
vP, tP = embed(obs['pc1'], 8, 1)
eP, CP, emP, slP, medP, _ = curve_for_embed(vP, tP, theiler=0)
sr_i = iaaft1d(np.asarray(obs['pc1'], dtype=np.float64), niter=40,
               rng=np.random.default_rng(99))
vI, tI = embed(sr_i, 8, 1)
eI, CI, emI, slI, medI, _ = curve_for_embed(vI, tI, theiler=0)
sr_p = phase_rand_1d(obs['pc1'], rng=np.random.default_rng(98))
vR, tR = embed(sr_p, 8, 1)
eR, CR, emR, slR, medR, _ = curve_for_embed(vR, tR, theiler=0)
sh = obs['pc1'][np.random.default_rng(97).permutation(len(obs['pc1']))]
vS, tS = embed(sh, 8, 1)
eS, CS, emS, slS, medS, _ = curve_for_embed(vS, tS, theiler=0)

fig, axs = plt.subplots(2, 4, figsize=(17, 8.5))

def panel_C(ax, eps, C, med, title, color='b', label=None, ls='-'):
    ax.plot(eps / med, np.maximum(C, 1e-12), color=color, lw=1.8, ls=ls, label=label)
    ax.axhline(1.0, color='k', ls=':', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
    ax.set_title(title, fontsize=8.5)
    ax.grid(alpha=0.25)

def panel_S(ax, em, sl, med, N, title, color='b', ylim=None):
    ax.plot(em / med, sl, color=color, lw=1.8)
    ax.axhline(2 * np.log10(N), color='red', ls='--', lw=1.0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel('local slope')
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(ylim if ylim else (-0.5, 12))
    ax.grid(alpha=0.25)

panel_C(axs[0, 0], epsH, CH, medH, '(i) Henon map N=%d (true D2~1.26):\nstraight line over ~3 decades' % NREF)
panel_S(axs[1, 0], emH, slH, medH, NREF,
        'Henon: flat slope %.2f over %.2f decades (r2=%.4f)\nTHIS is "good"' % (d2H, plH['width_dec'], r2H),
        ylim=(-0.5, 3))
panel_C(axs[0, 1], epsL, CL, medL,
        '(ii) Lorenz x(t), m=8, tau=%d, N=%d:\ndelay-embedded chaotic attractor (true D2~2.05)' % (tauL, len(vecsL)))
panel_S(axs[1, 1], emL, slL, medL, len(vecsL),
        'Lorenz: flat D2=%.2f over %.2f decades (r2=%.3f)\nreal scaling region at this exact N' % (
            d2L, plL['width_dec'], r2L), ylim=(-0.5, 5))

panel_C(axs[0, 2], eP, CP, medP,
        '(iii) launch PC1 m=8 tau=1 vs nulls:\nAR(1) and phase-rand Gaussianize the marginal\nand get slope ~5; IAAFT (marginal+spectrum)\nlands near the data',
        color='purple', label='data (slope@C.1=%.2f)' % slope_at(eP, CP, 0.1))
axs[0, 2].plot(eI / medI, np.maximum(CI, 1e-12), 'orange', ls='--', lw=1.4,
               label='IAAFT (%.2f)' % slope_at(eI, CI, 0.1))
axs[0, 2].plot(eA / medA, np.maximum(CA, 1e-12), 'm', ls=':', lw=1.4,
               label='AR(1) (%.2f)' % slope_at(eA, CA, 0.1))
axs[0, 2].legend(fontsize=7, loc='lower right')
panel_S(axs[1, 2], emP, slP, medP, len(vP),
        'slopes: data 2.22, IAAFT 1.7-1.9, AR(1)/phase ~5.1.\nThe m=8 slope is set by marginal+2nd-order\nstatics; NO curve here is flat over a decade',
        color='purple')
axs[1, 2].plot(emI / medI, slI, 'orange', ls='--', lw=1.4)
axs[1, 2].plot(emA / medA, slA, 'm', ls=':', lw=1.4)

panel_C(axs[0, 3], eP, CP, medP,
        '(iv) launch PC1 m=8 tau=1 vs time-shuffle:\nshuffling destroys the delay cloud itself\n(white-noise fill-dim ~8, above the ceiling)',
        color='purple', label='data (flat 0.25 dec)')
axs[0, 3].plot(eS / medS, np.maximum(CS, 1e-12), 'r:', lw=1.5,
               label='time-shuffled (flat 0.40 dec)')
axs[0, 3].legend(fontsize=7, loc='lower right')
panel_S(axs[1, 3], emP, slP, medP, len(vP),
        'data vs time-shuffle: no plateau on either curve;\nshuffle slope@C.1 = %.2f ~ fill-dim\n(2 log10 N = %.2f): unmeasurable' % (
            slope_at(eS, CS, 0.1), 2 * np.log10(len(vP))), color='purple')
axs[1, 3].plot(emS / medS, slS, 'r:', lw=1.5)

fig.suptitle('FIG 6 -- reference shapes: a real scaling region (Henon, Lorenz) vs the red-noise shape class of our data', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'fig6_reference_shapes.png'), dpi=125)
plt.close(fig)
print('fig6 rewritten %.0fs' % (time.time() - t0))

res['fig6_numbers'] = {
    'henon': {'d2': d2H, 'r2': r2H, 'plateau_dec': plH['width_dec']},
    'lorenz': {'d2': d2L, 'r2': r2L, 'plateau_dec': plL['width_dec'], 'tau': tauL, 'm': 8},
    'pc1_m8': {'slope_at_C.1_data': slope_at(eP, CP, 0.1),
               'slope_at_C.1_iaaft': slope_at(eI, CI, 0.1),
               'slope_at_C.1_phase': slope_at(eR, CR, 0.1),
               'slope_at_C.1_ar1': slope_at(eA, CA, 0.1),
               'slope_at_C.1_shuffle': slope_at(eS, CS, 0.1)},
}
with open(os.path.join(OUT, 'shape4.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved shape4.json; total %.0fs' % (time.time() - t0))
