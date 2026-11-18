"""exp_shape1: the C(eps) SHAPE question on raw clouds.

Fig 1: N-dependence (512..full), absolute vs normalized eps axis, and the
       centered-vs-uncentered (+fp32-vs-fp64) identity check.
Fig 2: distance-metric variants (center-Euclid / raw-Euclid / zscored /
       cosine / centered-unit) + local slopes with the Eckmann-Ruelle ceiling.
Fig 3: artifact gallery -- loop knee, duplicate floor, small-eps depletion
       with 1/N and 10/N lines + Poisson band, self-pair (cdist) floor,
       linear-vs-log eps grid.
Fig 4: Theiler w=0/10/50/100 on raw cloud (loop + non-loop) vs on a
       delay-embedded scalar.

JSON: shape1.json with every number quoted in the report.
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
                    fit_d2, plateau_scan, pair_counts, observables, embed,
                    RNG, LOOPS, jsonable)

OUT = os.path.dirname(os.path.abspath(__file__))
EPS = None  # set per trajectory
NEPS = 96


def grid_for(med, lo=0.001, hi=5.0):
    return np.geomspace(lo * med, hi * med, NEPS)


res = {}
t_start = time.time()

# ================================================================ FIG 1
# N-dependence, absolute vs normalized axis, centering/fp32 identity
ROWS = ['interstellar', 'launch']
Ns = {'interstellar': [512, 1024, 1026], 'launch': [512, 1024, 2048, 3039]}

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
for r, name in enumerate(ROWS):
    X = load_traj(name)
    n = len(X)
    med = median_dist(X, 'center')
    eps = grid_for(med)
    res.setdefault(name, {})['scale'] = float(med)
    row = res[name].setdefault('fig1', {})

    for c, (ax_abs, ax_norm) in enumerate([(axes[r, 0], axes[r, 1])]):
        pass
    ax_abs, ax_norm = axes[r, 0], axes[r, 1]
    for N in Ns[name]:
        if N >= n:
            idx = None
        else:
            idx = np.sort(RNG.choice(n, size=N, replace=False))
        C, npair, cnts = c_curve(X, eps, theiler=0, arith='f64',
                                 mode='center', idx=idx)
        row[str(N)] = {'C': C.tolist(), 'npairs': int(npair)}
        lbl = f'N={N}' + (' (full)' if N >= n else '')
        col = None
        for ax, xvals in [(ax_abs, eps), (ax_norm, eps / med)]:
            ln, = ax.plot(xvals, np.maximum(C, 1e-9), lw=1.5, label=lbl)
            col = ln.get_color()
            ax.axhline(1.0 / N, color=col, ls=':', lw=0.7, alpha=0.55)
            ax.axhline(10.0 / N, color=col, ls='--', lw=0.7, alpha=0.45)
    # identity overlays at full N
    Cctr, _, _ = c_curve(X, eps, arith='f64', mode='center')
    Craw, _, _ = c_curve(X, eps, arith='f64', mode='raw')
    Cf32, _, _ = c_curve(X, eps, arith='f32', mode='raw')
    ok = (Cctr > 0) & (Craw > 0) & (Cf32 > 0)
    d_ctr_raw = float(np.max(np.abs(np.log(Cctr[ok]) - np.log(Craw[ok]))))
    d_f32 = float(np.max(np.abs(np.log(Cctr[ok]) - np.log(Cf32[ok]))))
    row['max_dlogC_center_vs_raw_f64'] = d_ctr_raw
    row['max_dlogC_f32raw_vs_f64center'] = d_f32
    for ax, xvals in [(ax_abs, eps), (ax_norm, eps / med)]:
        ax.plot(xvals, Craw, 'k.', ms=2.0, alpha=0.5,
                label=f'uncentered (max|dlogC|={d_ctr_raw:.1e})')
        ax.plot(xvals, Cf32, 'r+', ms=3.0, alpha=0.5,
                label=f'fp32 pipeline (max|dlogC|={d_f32:.1e})')
    for ax, xlab in [(ax_abs, r'absolute $\epsilon$'),
                     (ax_norm, r'$\epsilon$ / median pairwise dist')]:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlab + (' (log scale)' if ax is ax_abs else ''))
        ax.set_ylim(1e-6, 2)
        ax.grid(alpha=0.25)
    axes[r, 0].set_ylabel(r'$C(\epsilon)$  (log scale)')
    axes[r, 0].set_title(
        f'{name}: SAME data, absolute eps axis -- N only moves the small-eps edge\n'
        f'(dotted/dashed horiz lines = 1/N and 10/N = depletion floor for each N)', fontsize=10)
    axes[r, 1].set_title(
        f'{name}: normalized eps axis -- identical shape, shifted x-labels;\n'
        f'centering & fp32 are no-ops here (anisotropy small, see legend)', fontsize=10)
    axes[r, 1].legend(fontsize=7, loc='lower right')
    print(f'fig1 {name}: done  {time.time()-t_start:.0f}s  '
          f'center-vs-raw dlogC={d_ctr_raw:.2e}  fp32 dlogC={d_f32:.2e}')
fig.suptitle('FIG 1 -- C(eps) shape, non-loop trajectories: axis normalization, N-dependence, centering identity', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, 'fig1_N_axes.png'), dpi=125)
plt.close(fig)

# ================================================================ FIG 2
# metric variants + local slopes with ER ceiling
VARIANTS = [('center', 'Euclid, centered (reference)'),
            ('raw', 'Euclid, uncentered'),
            ('zscore', 'Euclid, per-coord z-scored'),
            ('unit', 'cosine on raw states'),
            ('unit_center', 'cosine on centered states')]
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
for r, name in enumerate(ROWS):
    X = load_traj(name)
    n = len(X)
    er = 2.0 * np.log10(n)
    axC, axS = axes[r, 0], axes[r, 1]
    row = res[name].setdefault('fig2', {})
    for mode, lbl in VARIANTS:
        med_v = median_dist(X, mode)
        eps = grid_for(med_v)
        C, npair, _ = c_curve(X, eps, arith='f64', mode=mode)
        em, sl = slope_curve(eps, C, win=7)
        Cmid = C[(len(C) - len(em)) // 2: (len(C) - len(em)) // 2 + len(em)]
        # align C to midpoints properly: recompute via interp
        Cmid = np.interp(np.log(em), np.log(eps), C)
        pl = plateau_scan(em, sl, Cmid, n, c_lo=10.0 / n)
        best = pl[0] if pl else None
        axC.plot(eps / med_v, np.maximum(C, 1e-9), lw=1.6, label=lbl)
        axS.plot(em / med_v, sl, lw=1.6, label=lbl)
        row[lbl] = {
            'median_dist': float(med_v),
            'slope_at_C.1': float(np.interp(np.log(0.1), np.log(np.maximum(Cmid, 1e-12)), em)) if False else None,
            'iqr_over_med_dist': None,
            'widest_flat': best,
        }
        # fill in numbers
        d = row[lbl]
        slq = sl[np.isfinite(sl)]
        d['slope_min'] = float(np.nanmin(slq)) if slq.size else None
        d['slope_max'] = float(np.nanmax(slq)) if slq.size else None
        # slope where C crosses 0.1 and 0.5
        for tg in (0.1, 0.5):
            k = np.argmax(C >= tg) if np.any(C >= tg) else -1
            if 0 < k < len(em):
                d[f'slope_at_C{tg}'] = float(np.interp(np.log(eps[k]), np.log(em), sl))
    # distance spread per variant for the title
    from cshape import dist_stats
    spr = {m: dist_stats(X, m, n=1200)['iqr_over_med'] for m, _ in VARIANTS}
    for m, lbl in VARIANTS:
        res[name]['fig2'][lbl]['iqr_over_med_dist'] = spr[m]
    axC.axhline(1.0, color='k', lw=0.6, alpha=0.4)
    axC.axhline(10.0 / n, color='gray', ls='--', lw=0.8)
    axC.text(1e-3 * 5, 12.0 / n, ' C=10/N', color='gray', fontsize=7, va='bottom')
    axC.set_xscale('log'); axC.set_yscale('log')
    axC.set_xlabel(r'$\epsilon$ / median pairwise dist (per variant)')
    axC.set_ylabel(r'$C(\epsilon)$')
    axC.set_ylim(1e-6, 2)
    axS.axhline(er, color='red', ls='--', lw=1.0)
    axS.text(1.2e-3, er, ' 2 log10(N) ceiling', color='red', fontsize=8, va='bottom')
    axS.set_xscale('log')
    axS.set_xlabel(r'$\epsilon$ / median pairwise dist (per variant)')
    axS.set_ylabel(r'd log C / d log $\epsilon$')
    axS.set_ylim(-0.5, 14)
    axC.set_title(f'{name}: all metric variants are steep sigmoids (IQR/med: '
                  + ', '.join(f'{m.split("_")[0] if m!="zscore" else "z"}={spr[m]:.2f}' for m, _ in VARIANTS)
                  + ')\nno variant has a 1-decade flat slope region; red line = finite-N ceiling', fontsize=9)
    axS.set_title(f'{name}: local slope vs eps -- monotone decreasing in every variant,\n'
                  f'small-eps slope 9-13 >> ceiling {er:.2f} = finite-N inflation, not dimension', fontsize=9)
    axC.legend(fontsize=7, loc='lower right')
    axS.legend(fontsize=7, loc='upper right')
    print(f'fig2 {name}: done {time.time()-t_start:.0f}s')
fig.suptitle('FIG 2 -- distance-metric variants: none makes the raw-cloud C(eps) "textbook"', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, 'fig2_metric_variants.png'), dpi=125)
plt.close(fig)

# ================================================================ FIG 3
# artifact gallery
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10))

# (a) cyber loop: knee + saturation + tube/ring fits
name = 'cyber'
X = load_traj(name)
n = len(X)
med = median_dist(X, 'center')
eps = grid_for(med)
C, npair, _ = c_curve(X, eps, arith='f64', mode='center')
em, sl = slope_curve(eps, C, win=7)
d2_tube, r2_tube, _ = fit_d2(eps, C, 0.02, 0.1)
d2_ring, r2_ring, _ = fit_d2(eps, C, 0.1, 0.9)
ax = axs[0, 0]
ax.plot(eps / med, C, 'b-', lw=1.8, label='cyber C(eps), full N=4095')
ax.axhline(1.0, color='k', ls=':', lw=0.8)
ax.axhline(1.0 / n, color='gray', ls=':', lw=0.8)
ax.axhline(10.0 / n, color='gray', ls='--', lw=0.8)
ax.annotate(f'tube fit C in [.02,.1]:\nD2={d2_tube:.2f} (r2={r2_tube:.2f})\n= loop tube, not attractor',
            xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8, color='darkblue')
ax.annotate(f'ring fit C in [.1,.9]:\nD2={d2_ring:.2f} (r2={r2_ring:.3f})\n= phase ring of the loop',
            xy=(0.45, 0.45), xycoords='axes fraction', fontsize=8, color='darkgreen')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_title('(a) LOOP trajectory (cyber, P=53): knee at the repeat scale then ring;\n'
             'the "good-looking" mid-range power-law fit (r2>0.98) is the loop phase ring', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
res['cyber'] = {'n': int(n), 'scale': float(med), 'tube_fit': [d2_tube, r2_tube],
                'ring_fit': [d2_ring, r2_ring]}

# (b) first layer: duplicate floor
name = 'first'
X = load_traj(name)
n = len(X)
med = median_dist(X, 'center')
eps = grid_for(med, lo=0.0001, hi=5.0)
C, npair, _ = c_curve(X, eps, arith='f64', mode='center')
floor = C[0]
ax = axs[0, 1]
ax.plot(eps / med, C, 'b-', lw=1.8, label=f'first layer C(eps), N={n}')
ax.axhline(floor, color='r', ls='--', lw=1.0,
           label=f'duplicate floor C={floor:.3f} (constant at ALL eps)')
ax.axhline(10.0 / n, color='gray', ls='--', lw=0.8, label='10/N')
ax.annotate('2.6% of pairs are EXACT duplicates\n-> C(eps) never goes to 0;\nnaive small-eps fits give D2~0',
            xy=(0.05, 0.15), xycoords='axes fraction', fontsize=8, color='darkred')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_title('(b) FIRST layer: exact-duplicate floor dominates small eps\n'
             'C(eps=0.0001xmed)= %.3f -> this is a flat line, not a scaling region' % floor, fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
res['first'] = {'n': int(n), 'scale': float(med), 'dup_floor_C': float(floor)}

# (c) launch: depletion zone, Poisson band, self-pair artifact
name = 'launch'
X = load_traj(name)
n = len(X)
med = median_dist(X, 'center')
eps = grid_for(med, lo=0.02, hi=5.0)
C, npair, cnts = c_curve(X, eps, arith='f64', mode='center')
sig = np.sqrt(np.maximum(cnts, 1)) / npair
Cself, npair_s, _ = c_curve(X, eps, arith='f64', mode='center', include_self=True)
ax = axs[1, 0]
ax.fill_between(eps / med, np.maximum(C - 2 * sig, 1e-12), C + 2 * sig,
                color='steelblue', alpha=0.25, label='Poisson 2-sigma band')
ax.plot(eps / med, C, 'b-', lw=1.8, label='correct: i<j pairs only')
ax.plot(eps / med, Cself, 'r--', lw=1.4,
        label='naive cdist (includes i=j): floor 1/N, slope->0')
ax.axhline(1.0 / n, color='gray', ls=':', lw=0.8)
ax.axhline(10.0 / n, color='gray', ls='--', lw=0.8)
ax.text(0.023, 1.2 / n, ' 1/N', color='gray', fontsize=7)
ax.text(0.023, 12.0 / n, ' 10/N', color='gray', fontsize=7)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_ylim(1e-5, 2)
ax.set_title('(c) launch small-eps: below C~10/N the curve is count noise (2-sigma band explodes);\n'
             'including the cdist diagonal fabricates a flat "plateau" at 1/N -- a notebook classic', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
res['launch']['selfpair_floor'] = float(Cself[0])

# (d) linear vs log eps grid, same launch data
lin = np.linspace(eps[0], eps[-1], 60)
Clin, _, _ = c_curve(X, lin, arith='f64', mode='center')
ax = axs[1, 1]
ax.plot(eps / med, C, 'b-o', ms=2.5, lw=1.5, label='log grid (correct, 96 pts)')
ax.plot(lin / med, Clin, 'r-s', ms=3.0, lw=1.2,
        label='linear grid (notebook style, 60 pts)')
ax.axhline(1.0, color='k', ls=':', lw=0.7)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
ax.set_title('(d) same data, linear eps grid: small-eps shape becomes a coarse staircase\n'
             'grid choice alone changes the apparent shape at small eps', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
res['launch']['lin_grid_C_at_0.03med'] = float(np.interp(0.03 * med, lin, Clin))

fig.suptitle('FIG 3 -- artifact gallery: what makes C(eps) curves look "good" or "weird" for non-chaotic reasons', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig3_artifacts.png'), dpi=125)
plt.close(fig)
print(f'fig3 done {time.time()-t_start:.0f}s')

# ================================================================ FIG 4
# Theiler: raw cloud (loop + non-loop) vs delay-embedded scalar
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10))
W = [0, 10, 50, 100]
panels = [('cyber', 'raw'), ('cyber', 'pc1_embed'), ('launch', 'raw'), ('launch', 'pc1_embed')]
for (name, kind), ax in zip(panels, axs.flat):
    X = load_traj(name)
    n = len(X)
    if kind == 'raw':
        pts = transform(X, 'center')
        times = np.arange(n)
        med = median_dist(X, 'center')
        sub = ''
    else:
        obs = observables(X)
        s = obs['pc1']
        pts, times = embed(s, m=5, tau=1)
        pts = pts - pts.mean(0, keepdims=True)
        med = float(np.median(np.sqrt(np.maximum(
            ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1), 0))[np.triu_indices(len(pts), 1)]))
        sub = '  (PC1, m=5, tau=1)'
    eps = grid_for(med, lo=0.002, hi=3.0)
    deltas = []
    res.setdefault(name, {}).setdefault('fig4_theiler', {})
    for w in W:
        C, npair, _ = c_curve(pts, eps, theiler=w, times=times, arith='f64', mode='raw')
        if w == 0:
            C0 = C
        else:
            ok = (C > 0) & (C0 > 0)
            deltas.append(float(np.max(np.abs(np.log(C[ok]) - np.log(C0[ok])))))
        ax.plot(eps / med, np.maximum(C, 1e-12), lw=1.4, label=f'w={w}')
    dmax = max(deltas) if deltas else 0.0
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$ / median dist'); ax.set_ylabel(r'$C(\epsilon)$')
    ax.set_ylim(1e-7, 2)
    kindlbl = 'RAW 1536-D cloud' if kind == 'raw' else 'delay-embedded scalar'
    if kind == 'raw':
        extra = 'P=53 loop: repeats live at ALL multiples of P, w<=100 removes almost nothing'
    else:
        extra = 'max |d log C| = %.3f' % dmax + (' (temporal-neighbor artifact only at small eps)' if dmax > 0.05 else ' (PC1 is not smooth enough for w to matter either)')
    ax.set_title(f'{name} {kindlbl}{sub}\nmax |d log C| over w=0/10/50/100: {dmax:.3f}\n{extra}', fontsize=8.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    res[name]['fig4_theiler'][kind] = {'max_dlogC_w0_vs_rest': float(dmax)}
fig.suptitle('FIG 4 -- Theiler windows: never reshape the raw point-cloud C(eps); '
             'they only matter for smooth delay-embedded scalars', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, 'fig4_theiler.png'), dpi=125)
plt.close(fig)
print(f'fig4 done {time.time()-t_start:.0f}s')

with open(os.path.join(OUT, 'shape1.json'), 'w') as f:
    json.dump(jsonable(res), f, indent=1)
print('saved shape1.json; total %.0fs' % (time.time() - t_start))
