"""Task 1: decay-law fits of the hidden-state response vs lag tau = t - s.

Fits four models to the mean-over-24-directions response at r=0.1, for every
injection position s and layer L, on the uniform lag range tau in [0, 99]:
  1E   A*exp(-tau/T)                       (2 params)
  2E   A*exp(-tau/Tf) + B*exp(-tau/Ts)     (4 params)
  PL   A*(1+tau)^(-p)                      (2 params, finite at tau=0)
  E+C  A*exp(-tau/T) + C                   (3 params, C = persistent floor)
Fits minimize squared error on log(dh) (i.e. relative error); model choice by AIC/BIC.
"""
import numpy as np, csv
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from common import load_emb, style, OUT

D = load_emb()
steps, layers = D['steps'], D['layers']
TAU_MAX = 99

MODELS = {
    '1E': lambda p, t: p[0] * np.exp(-t / p[1]),
    '2E': lambda p, t: p[0] * np.exp(-t / p[1]) + p[2] * np.exp(-t / p[3]),
    'PL': lambda p, t: p[0] * (1 + t) ** (-p[1]),
    'E+C': lambda p, t: p[0] * np.exp(-t / p[1]) + p[2],
}
NPAR = {'1E': 2, '2E': 4, 'PL': 2, 'E+C': 3}


def fit(name, fn, tau, y):
    def resid(p):
        out = np.log(y) - np.log(fn(p, tau))
        return np.where(np.isfinite(out), out, 1e6)

    lo = {'1E': [1e-12, 0.3], 'PL': [1e-12, -1.0],
          'E+C': [1e-12, 0.3, 0.0],
          '2E': [1e-12, 0.3, 1e-12, 3.0]}[name]
    hi = {'1E': [np.inf, 3000], 'PL': [np.inf, 12.0],
          'E+C': [np.inf, 3000, np.inf],
          '2E': [np.inf, 30, np.inf, 3000]}[name]
    starts = []
    for T0 in [2, 5, 15, 40, 120, 400]:
        if name == '1E':
            starts.append([y[0], T0])
        elif name == 'PL':
            starts.append([y[0], p0pl]) if False else None
        elif name == 'E+C':
            starts.append([y[0] * 0.7, T0, max(y[-1] * 0.7, 1e-6)])
        elif name == '2E':
            starts.append([y[0] * 0.6, min(T0, 25), y[0] * 0.5, max(T0 * 8, 40)])
    if name == 'PL':
        starts = [[y[0], p] for p in [0.1, 0.3, 0.7, 1.5, 3.0]]
    if name == '2E':
        for Af in [0.3, 0.7, 0.9]:
            for Tf0, Ts0 in [(2, 30), (3, 100), (8, 300), (1.5, 15)]:
                starts.append([y[0] * Af, Tf0, y[0] * (1 - Af) + 1e-6, Ts0])
    best = None
    for p0 in starts:
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    if best is None:
        return None
    p = best.x
    if name == '2E' and p[1] > p[3]:        # keep fast/slow label order
        p = [p[2], p[3], p[0], p[1]]
    rss = float(np.sum(resid(p) ** 2))
    n = len(tau)
    k = NPAR[name]
    return p, rss, n * np.log(rss / n) + 2 * k, n * np.log(rss / n) + k * np.log(n)


results = {}
for s in steps:
    for L in layers:
        i_s, i_l = list(steps).index(s), list(layers).index(L)
        y = D['M'][3, i_s, i_l, int(s):int(s) + TAU_MAX + 1].copy()
        tau = np.arange(len(y), dtype=float)
        ok = np.isfinite(y) & (y > 0)
        y, tau = y[ok], tau[ok]
        results[(int(s), int(L))] = dict(
            y=y, tau=tau,
            fits={nm: fit(nm, fn, tau, y) for nm, fn in MODELS.items()})

# ---- CSV dump ----
with open(OUT + r'\decay_fits.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['s', 'layer', 'y0', 'model', 'p1', 'p2', 'p3', 'p4',
               'RSS_log', 'AIC', 'BIC', 'dAIC'])
    for (s, L), r in sorted(results.items()):
        aicmin = min(v[2] for v in r['fits'].values())
        for name, v in r['fits'].items():
            p, rss, aic, bic = v
            w.writerow([s, L, f'{r["y"][0]:.4f}', name] +
                       [f'{x:.4g}' for x in p] +
                       [f'{rss:.4f}', f'{aic:.2f}', f'{bic:.2f}', f'{aic - aicmin:.2f}'])

# ---- summary print ----
cntA, cntB = {}, {}
for (s, L), r in results.items():
    winA = min(r['fits'], key=lambda m: r['fits'][m][2])
    winB = min(r['fits'], key=lambda m: r['fits'][m][3])
    cntA[winA] = cntA.get(winA, 0) + 1
    cntB[winB] = cntB.get(winB, 0) + 1
print('AIC winner counts (32 fits):', cntA)
print('BIC winner counts          :', cntB)

print('\nper (s, L=27): winner, params, floor:')
for s in steps:
    r = results[(int(s), 27)]
    win = min(r['fits'], key=lambda m: r['fits'][m][2])
    p, rss, aic, bic = r['fits'][win]
    y0 = r['y'][0]
    if win == 'E+C':
        desc = f'T={p[1]:6.1f}   floor={100*p[2]/y0:5.1f}% of head'
    elif win == '1E':
        desc = f'T={p[1]:6.1f}   (pure exp, no floor)'
    elif win == '2E':
        desc = f'Tf={p[1]:5.1f} Ts={p[3]:7.1f}   slow share={100*p[2]/(p[0]+p[2]):5.1f}%'
    else:
        desc = f'p={p[1]:6.3f}   dh(99)/dh(0)={(1+99)**(-p[1]):.3f}'
    d2 = sorted((r['fits'][m][2] - aic, m) for m in r['fits'] if m != win)[0]
    print(f'  s={s:3d}: {win:4s} {desc}   | next {d2[1]} dAIC={d2[0]:6.1f}')

print('\nwinner by (s, layer):')
for L in layers:
    row = []
    for s in steps:
        r = results[(int(s), int(L))]
        row.append(f's={s:3d}:{min(r["fits"], key=lambda m: r["fits"][m][2]):>4s}')
    print(f'  L{L}: ' + '  '.join(row))

# ---- figure ----
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
MC = {'1E': '#08519c', '2E': '#238b45', 'PL': '#d94801', 'E+C': '#6a51a3'}
LS = {'1E': ':', '2E': '-.', 'PL': '--', 'E+C': '-'}

def example_panel(ax, s, L):
    r = results[(s, L)]
    tau, y = r['tau'], r['y']
    aicmin = min(v[2] for v in r['fits'].values())
    ax.plot(tau, y, color='0.35', lw=1.7, label='data (mean of 24 dirs)', zorder=5)
    for name, fn in MODELS.items():
        p, rss, aic, bic = r['fits'][name]
        ax.plot(tau, fn(p, tau), color=MC[name], ls=LS[name], lw=1.0,
                label=f'{name}   ΔAIC={aic - aicmin:6.1f}')
    win = min(r['fits'], key=lambda m: r['fits'][m][2])
    p = r['fits'][win][0]
    if win == 'E+C':
        extra = f'T={p[1]:.0f}, floor={100*p[2]/y[0]:.0f}% of head'
    elif win == '2E':
        extra = f'Tf={p[1]:.0f}, Ts={p[3]:.0f}'
    elif win == '1E':
        extra = f'T={p[1]:.0f}'
    else:
        extra = f'p={p[1]:.2f}'
    ax.set_yscale('log')
    ax.set_xlabel('lag τ = t − s  (generated tokens after the perturbed one)')
    ax.set_ylabel('hidden-state response ‖Δh‖  (r = 0.1)')
    ax.set_title(f's={s} (margin={D["margins"][s]:.2f}), layer {L}: best = {win}  ({extra})',
                 fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.4)

example_panel(axes[0, 0], 16, 27)
example_panel(axes[0, 1], 75, 27)
example_panel(axes[1, 0], 150, 27)

ax = axes[1, 1]
BG = {'1E': '#c6dbef', '2E': '#c7e9c0', 'PL': '#fdd0a2', 'E+C': '#dadaeb'}
for i_s, s in enumerate(steps):
    for i_l, L in enumerate(layers):
        r = results[(int(s), int(L))]
        win = min(r['fits'], key=lambda m: r['fits'][m][2])
        p = r['fits'][win][0]
        y0 = r['y'][0]
        if win == 'E+C':
            txt = f'{win}\nT={p[1]:.0f} C={100*p[2]/y0:.0f}%'
        elif win == '2E':
            txt = f'{win}\n{p[1]:.0f}/{p[3]:.0f}'
        elif win == '1E':
            txt = f'{win}\nT={p[1]:.0f}'
        else:
            txt = f'{win}\np={p[1]:.2f}'
        ax.add_patch(plt.Rectangle((i_l - 0.5, i_s - 0.5), 1, 1, color=BG[win]))
        ax.text(i_l, i_s, txt, ha='center', va='center', fontsize=6.4)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(7.5, -0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([f'layer {L}' for L in layers])
ax.set_yticks(range(8))
ax.set_yticklabels([f's={s}' for s in steps])
ax.set_title('AIC winner per (s, layer); T in tokens (C = floor as % of head)',
             fontweight='bold')
for sp in ax.spines.values():
    sp.set_visible(False)

fig.suptitle('Decay of the hidden-state response vs lag: never a single clean exponential — '
             'a fast drop plus a persistent floor, s-dependent', fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig1_decay.png')
plt.close(fig)
print('\nsaved fig1_decay.png')
