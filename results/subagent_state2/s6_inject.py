"""Task 6: injection depth — input-embedding (layer 0) vs residual stream at
layer 14, at matched radii r in {0.01, 0.03} and matched positions/steps
s in {16, 50, 100, 150}.

Key facts established by the data:
  * dh(tau=0, L14) of an L14 injection is exactly r (unit-norm directions),
    while an input-embedding injection has already been amplified 9-32x by
    blocks 0-13 when it reaches L14.
  * later injection is much weaker at every later layer but decays less
    (higher floor fraction, similar tail), and its margin response is ~10x
    too small to ever flip a token.
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from common import load_emb, load_l14, style, OUT, SCOL

D = load_emb()
E = load_l14()
steps_c = [16, 50, 100, 150]                 # common steps
r_idx = 2                                    # emb r=0.03
rL_idx = 1                                   # L14 r=0.03
iL = {14: 0, 21: 1, 27: 2}                   # L14 file layer indices
iE = {7: 0, 14: 1, 21: 2, 27: 3}
i_sE = {int(s): i for i, s in enumerate(D['steps'])}
i_sL = {int(s): i for i, s in enumerate(E['steps'])}

print('=== head (tau=0) at r=0.03: dh at the injection-relevant layers ===')
print('       emb dh(L14)  emb dh(L27)  L14inj dh(L14)  L14inj dh(L27)  ratio@L27')
for s in steps_c:
    a = D['M'][r_idx, i_sE[s], iE[14], s]
    b = D['M'][r_idx, i_sE[s], iE[27], s]
    c = E['M'][rL_idx, i_sL[s], iL[14], s]
    d = E['M'][rL_idx, i_sL[s], iL[27], s]
    print(f'  s={s:3d} {a:9.3f} {b:11.3f} {c:14.4f} {d:15.4f} {d/b:10.4f}')
print('  (L14inj dh(L14) = r = 0.03 exactly -> directions are unit norm, no early-block gain)')

print('\n=== ratio L14inj/emb vs tau (L27, r=0.03) ===')
for s in steps_c:
    ye = D['M'][r_idx, i_sE[s], iE[27], s:s + 100]
    yl = E['M'][rL_idx, i_sL[s], iL[27], s:s + 100]
    head = yl[0] / ye[0]
    tail = np.mean(yl[80:100] / ye[80:100])
    print(f'  s={s:3d}: head ratio {head:.4f}   tail (tau 80-99) ratio {tail:.4f}   '
          f'tail/head = {tail/head:.1f}x')

print('\n=== does the L14 injection produce echoes? (s=16, echo t=113) ===')
for L in [14, 21, 27]:
    v = E['M'][rL_idx, i_sL[16], iL[L], 113]
    v8 = E['M'][rL_idx, i_sL[16], iL[L], 116:120].mean()
    print(f'  L{L}: dh(t=113)={v:.4f}  vs local baseline (t=116-119) {v8:.4f}  -> echo? {v > 2*v8}')

print('\n=== margin response of L14 injection (r=0.03) vs emb ===')
mx = np.nanmax(np.abs(E['dm'][rL_idx]))
print(f'  L14-inject max |dm| over all s,dirs,t = {mx:.4f}  (smallest margin in sequence = 0.0202)')
for s in steps_c:
    me = np.nanmax(np.abs(D['dm'][r_idx, i_sE[s]]))
    print(f'  s={s:3d}: emb max |dm| = {me:.4f}')

# --- fits: same E+C model on both injections, L27, r=0.03, tau in [0,99] ---
def fit_ec(y, tau):
    def resid(p):
        f = p[0] * np.exp(-tau / p[1]) + p[2]
        return np.log(y) - np.log(f)
    best = None
    for T0 in [2, 5, 15, 40, 120, 400]:
        try:
            r = least_squares(resid, [y[0] * 0.7, T0, max(y[-1] * 0.7, 1e-9)],
                              bounds=([1e-12, 0.3, 0.0], [np.inf, 3000, np.inf]), max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    return best.x, float(np.sum(resid(best.x) ** 2))

print('\n=== E+C fits (log-space), L27, r=0.03: emb floors >= L14 floors ===')
fitres = {}
for s in steps_c:
    ye = D['M'][r_idx, i_sE[s], iE[27], s:s + 100]
    yl = E['M'][rL_idx, i_sL[s], iL[27], s:s + 100]
    tau = np.arange(100, dtype=float)
    pe, rse = fit_ec(ye, tau)
    pl, rsl = fit_ec(yl, tau)
    fitres[s] = dict(e=pe, l=pl, ye=ye, yl=yl)
    print(f'  s={s:3d}: emb      T={pe[1]:6.1f}  floor={100*pe[2]/ye[0]:5.1f}%  (logRSS={rse:.2f})')
    print(f'         L14inj    T={pl[1]:6.1f}  floor={100*pl[2]/yl[0]:5.1f}%  (logRSS={rsl:.2f})')

# ---------------- figure ----------------
style()
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

ax = axes[0, 0]
for s in steps_c:
    ye = D['M'][r_idx, i_sE[s], iE[27], s:s + 100]
    yl = E['M'][rL_idx, i_sL[s], iL[27], s:s + 100]
    ax.plot(np.arange(100), ye, color=SCOL[s], lw=1.4, label=f's={s} input-emb')
    ax.plot(np.arange(100), yl, color=SCOL[s], lw=1.4, ls='--')
ax.set_yscale('log')
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('‖Δh‖ at layer 27 (r = 0.03)')
ax.plot([], [], 'k-', label='input-embedding injection')
ax.plot([], [], 'k--', label='layer-14 injection')
ax.legend(ncol=2, fontsize=6.8)
ax.grid(alpha=0.25, lw=0.4)
ax.set_title('(a) layer-14 injection (dashed) is 11–38× weaker at layer 27 than\n'
             'input-embedding injection (solid) at the same r', fontweight='bold')

ax = axes[0, 1]
for s in steps_c:
    ye = D['M'][r_idx, i_sE[s], iE[27], s:s + 100]
    yl = E['M'][rL_idx, i_sL[s], iL[27], s:s + 100]
    ax.plot(np.arange(100), yl / ye, color=SCOL[s], lw=1.4, label=f's={s}')
ratios_h = [E['M'][rL_idx, i_sL[s], iL[27], s] / D['M'][r_idx, i_sE[s], iE[27], s] for s in steps_c]
ratios_t = [np.mean(E['M'][rL_idx, i_sL[s], iL[27], s + 80:s + 100] /
                    D['M'][r_idx, i_sE[s], iE[27], s + 80:s + 100]) for s in steps_c]
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('‖Δh(L14-inj)‖ / ‖Δh(emb-inj)‖  at layer 27')
ax.legend(fontsize=7)
ax.grid(alpha=0.25, lw=0.4)
ax.set_title('(b) the ratio does NOT grow with lag (tail/head = 0.5–1.0×):\n'
             'later injection is weaker but NOT flatter', fontweight='bold')

ax = axes[1, 0]
xs = np.arange(len(steps_c))
fl_e = [100 * fitres[s]['e'][2] / fitres[s]['ye'][0] for s in steps_c]
fl_l = [100 * fitres[s]['l'][2] / fitres[s]['yl'][0] for s in steps_c]
ax.bar(xs - 0.18, fl_e, width=0.34, color='#4292c4', label='input-embedding inj.')
ax.bar(xs + 0.18, fl_l, width=0.34, color='#fdae6b', label='layer-14 inj.')
for i, s in enumerate(steps_c):
    ax.text(i, max(fl_e[i], fl_l[i]) + 1, f'L14: {fl_l[i]:.0f}%\nemb: {fl_e[i]:.0f}%',
            ha='center', fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels([f's={s}' for s in steps_c])
ax.set_ylabel('persistent floor as % of the immediate response')
ax.legend()
ax.set_title('(c) exp+floor fits: input-emb injection leaves a LARGER floor fraction\n'
             '(8–19% vs 4–16%) — blocks 0–13 create most of the persistent tail',
             fontweight='bold')
ax.grid(alpha=0.3, lw=0.4, axis='y')

ax = axes[1, 1]
for s in steps_c:
    ye = fitres[s]['ye'] / fitres[s]['ye'][0]
    yl = fitres[s]['yl'] / fitres[s]['yl'][0]
    ax.plot(np.arange(100), ye, color=SCOL[s], lw=1.0, alpha=0.5)
    ax.plot(np.arange(100), yl, color=SCOL[s], lw=1.4, ls='--')
ax.set_yscale('log')
# aggregate labels
mean_e = np.mean([fitres[s]['ye'] / fitres[s]['ye'][0] for s in steps_c], axis=0)
mean_l = np.mean([fitres[s]['yl'] / fitres[s]['yl'][0] for s in steps_c], axis=0)
ax.plot(np.arange(100), mean_e, 'k-', lw=2.2, label='emb inj., mean of 4 s')
ax.plot(np.arange(100), mean_l, 'k--', lw=2.2, label='L14 inj., mean of 4 s')
ax.text(60, mean_e[60] * 1.35, f'emb: {mean_e[99]*100:.1f}% left', fontsize=7.5)
ax.text(60, mean_l[60] * 1.35, f'L14: {mean_l[99]*100:.1f}% left', fontsize=7.5)
ax.set_xlabel('lag τ = t − s (tokens)')
ax.set_ylabel('fraction of immediate response remaining')
ax.legend(loc='upper right')
ax.grid(alpha=0.25, lw=0.4)
ax.set_title('(d) normalized shapes: L14 injection retains LESS of its head than\n'
             'input-emb injection — “weaker but flatter” is half right (weaker only)',
             fontweight='bold')

fig.suptitle('Injection depth at matched r: layer-14 injection is 11–38× weaker and NOT flatter — '
             'blocks 0–13 create most of the amplification AND the persistent floor; '
             'its margin response (|Δm|≤0.002) can never flip a token',
             fontsize=10.5, fontweight='bold')
fig.savefig(OUT + r'\fig6_injectdepth.png')
plt.close(fig)
print('\nsaved fig6_injectdepth.png')
