"""Task 8: the response field dh(s, t) — one layer (27), one radius (0.1),
mean over 24 directions — with the causal boundary t = s drawn correctly,
plus three example lag curves and the aggregate decay.

(a) the (s, t) field: log color; causal region (t < s) masked white;
    the boundary t = s drawn as a line; small-margin positions marked.
(b) the margin-response field |dm(s, t)|: same geometry — its structure is
    vertical stripes at the small-margin positions, not at the echoes.
(c) three lag curves (s = 16, 75, 200).
(d) aggregate: median over the 8 s of the retained fraction, with an
    exp + floor fit.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import least_squares
from common import load_emb, style, OUT, SCOL

D = load_emb()
steps, layers, margins = D['steps'], D['layers'], D['margins']
M = D['M'][3]          # [s, layer, t], r=0.1
MD = D['MD'][3]        # mean |dm|, [s, t]
FIELD = M[:, 3, :]     # L27
NR, NT = FIELD.shape

# masks (causal): t < s invalid
mask = np.zeros((NR, NT), dtype=bool)
for i_s, s in enumerate(steps):
    mask[i_s, :int(s)] = True

print('=== field stats (L27, r=0.1) ===')
print('valid dh range: %.3g .. %.3g' % (np.nanmin(FIELD[~mask]), np.nanmax(FIELD[~mask])))
heads = [FIELD[i, int(s)] for i, s in enumerate(steps)]
print('head values dh(t=s):', np.round(heads, 2), ' max/min =', round(max(heads) / min(heads), 1))

frag = np.where(margins < 0.3)[0]
print('small-margin positions (<0.3):', frag.tolist())

print('\n=== |dm| field: is t=123/143 a vertical stripe? ===')
for t in [123, 143, 116, 147, 113]:
    col = []
    for i_s, s in enumerate(steps):
        if s < t:
            neigh = np.median(MD[i_s, max(int(s) + 1, t - 8):t + 8])
            col.append(MD[i_s, t] / neigh)
    print(f'  t={t}: |dm| / local-median across s<t: median = {np.median(col):5.2f} '
          f'(n={len(col)})')

print('\n=== three lag curves (numbers for annotation) ===')
for s in [16, 75, 200]:
    i = int(np.where(steps == s)[0][0])
    y = FIELD[i, int(s):]
    print(f'  s={s}: dh(0)={y[0]:.2f}  dh(10)={y[10]:.3f}  dh(50)={y[50]:.3f}  '
          f'dh(99)={y[99]:.3f}  dh(last)={y[-1]:.3f}')

# aggregate fit on median retained curve
ret = np.zeros((NR, 100))
for i_s, s in enumerate(steps):
    y = FIELD[i_s, int(s):int(s) + 100]
    ret[i_s] = y / y[0]
med = np.median(ret, axis=0)
tau = np.arange(100, dtype=float)

def fit_ec(yv, tv):
    def resid(p):
        return np.log(yv) - np.log(p[0] * np.exp(-tv / p[1]) + p[2])
    best = None
    for T0 in [2, 5, 15, 40, 120]:
        try:
            r = least_squares(resid, [yv[0] * 0.7, T0, max(yv[-1] * 0.8, 1e-6)],
                              bounds=([1e-12, 0.3, 0.0], [np.inf, 3000, np.inf]), max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best.x

pmed = fit_ec(med, tau)
print(f'\naggregate (median over 8 s) exp+floor fit: T={pmed[1]:.1f} tokens, '
      f'floor={pmed[2]:.3f} ({100*pmed[2]:.0f}% of head)')

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6),
                         gridspec_kw={'width_ratios': [1.35, 1]})

# --- (a) field heatmap ---
ax = axes[0, 0]
Fm = np.ma.masked_array(FIELD, mask)
vmin = max(np.nanpercentile(FIELD[~mask], 2), 1e-2)
im = ax.pcolormesh(np.arange(NT + 1) - 0.5, np.arange(NR + 1) - 0.5, Fm,
                   norm=mcolors.LogNorm(vmin=vmin, vmax=np.nanmax(FIELD)),
                   cmap='magma', rasterized=True)
# causal boundary t = s
bx = np.array([int(s) for s in steps], float)
by = np.array([np.where(steps == s)[0][0] for s in steps], float)
ax.plot(bx, by, color='#00ffff', lw=1.8, ls='-', label='causal boundary t = s')
ax.set_yticks(range(NR))
ax.set_yticklabels([f's={int(s)}' for s in steps])
ax.set_xlabel('generated token position t')
ax.set_ylabel('injection position s')
for t in [123, 143]:
    ax.axvline(t, color='#4deeea', ls='--', lw=1.0)
ax.text(145.5, 7.55, 't=123, 143 (smallest margins)', color='#4deeea', fontsize=7)
ax.legend(loc='lower right', fontsize=7.2)
plt.colorbar(im, ax=ax, label='‖Δh‖ (r=0.1, layer 27, mean of 24 dirs)', shrink=0.85)
ax.set_title('(a) the response field ‖Δh(s,t)‖: everything below/right of the cyan line is\n'
             'causally impossible; note the vertical echo stripes (t≈63, 86, 105–149)',
             fontweight='bold')

# --- (b) flip-probability field: fraction of directions whose gap flips at (s,t) ---
ax = axes[0, 1]
fl = D['dm'][3]                     # [s, dir, t]
flipfrac = np.zeros((NR, NT))
for i_s, s in enumerate(steps):
    for t in range(int(s) + 1, NT):
        flipfrac[i_s, t] = np.mean(fl[i_s, :, t] < -margins[t])
FFm = np.ma.masked_array(flipfrac, mask)
cmap = mcolors.ListedColormap(['#fdd0a2', '#f16913', '#7f0000'])
norm = mcolors.BoundaryNorm([1e-9, 0.05, 0.1, 1.0], cmap.N)
imb = ax.pcolormesh(np.arange(NT + 1) - 0.5, np.arange(NR + 1) - 0.5, FFm,
                    cmap=cmap, norm=norm, rasterized=True)
ax.plot(bx, by, color='#404040', lw=1.6)
ax.set_yticks(range(NR))
ax.set_yticklabels([f's={int(s)}' for s in steps])
ax.set_xlabel('generated token position t')
ax.text(123, 1, '4/24', ha='center', va='center', fontsize=8, color='w', fontweight='bold')
ax.text(143, 3, '1/24', ha='center', va='center', fontsize=8, color='w', fontweight='bold')
cb = plt.colorbar(imb, ax=ax, ticks=[0.02, 0.075, 0.55], shrink=0.85)
cb.ax.set_yticklabels(['0', '1/24', '>4/24'])
cb.set_label('fraction of the 24 directions whose top1–top2 gap flips')
ax.set_title('(b) where the gap actually flips: only s=16→t=123 (4/24 dirs) and\n'
             's=50→t=143 (1/24) — the two smallest margins of the whole sequence',
             fontweight='bold')

# --- (c) three lag curves ---
ax = axes[1, 0]
for s in [16, 75, 200]:
    i = int(np.where(steps == s)[0][0])
    y = FIELD[i, int(s):]
    ax.plot(np.arange(len(y)), y, color=SCOL[s], lw=1.6, label=f's={s}')
i16 = 1
y16 = FIELD[i16, 16:]
ax.annotate('echo (t=113)', xy=(113 - 16, y16[113 - 16]), xytext=(130, y16[0] * 0.4),
            arrowprops=dict(arrowstyle='->', lw=0.8), fontsize=7.5, color=SCOL[16])
ax.axhline(margins.min(), color='0.5', ls=':', lw=0.8)
ax.text(210, margins.min() * 1.3, 'smallest baseline margin (0.02)', fontsize=7)
ax.set_yscale('log')
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('‖Δh‖ (r=0.1, layer 27)')
ax.legend()
ax.grid(alpha=0.25, lw=0.4)
ax.set_title('(c) three regimes: near-tie s=16 (huge head, slow tail), ordinary s=75\n'
             '(clean decay), late s=200 (slow decay from a moderate head)',
             fontweight='bold')

# --- (d) aggregate ---
ax = axes[1, 1]
ax.fill_between(tau, np.percentile(ret, 25, axis=0), np.percentile(ret, 75, axis=0),
                color='#9ecae1', alpha=0.45, label='interquartile range over 8 s')
ax.plot(tau, med, color='#08306b', lw=1.8, label='median over 8 injection positions')
ax.plot(tau, pmed[0] * np.exp(-tau / pmed[1]) + pmed[2], color='#d94801', ls='--',
        lw=1.4, label=f'exp+floor fit: T={pmed[1]:.0f}, floor={100*pmed[2]:.0f}%')
ax.set_yscale('log')
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('fraction of immediate response remaining')
ax.legend(loc='upper right')
ax.grid(alpha=0.25, lw=0.4)
ax.set_title(f'(d) aggregate decay: median drops to {100*med[5]:.0f}% by τ=5, then a '
             f'floor near {100*pmed[2]:.0f}% persists to τ=99',
             fontweight='bold')

fig.suptitle('Response field at r=0.1, layer 27: strong causal response everywhere after s; '
             'echoes keep it alive, small margins convert it into token flips',
             fontsize=11, fontweight='bold')
fig.savefig(OUT + r'\fig7_response_field.png')
plt.close(fig)
print('\nsaved fig7_response_field.png')
