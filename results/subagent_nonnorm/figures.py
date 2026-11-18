"""Figures for subagent_nonnorm. CPU-only; reads derived.npz + summary.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = 'chaos_llm_2026/results/subagent_nonnorm'
D = np.load(f'{OUT}/derived.npz')
RES = json.load(open(f'{OUT}/summary.json'))

S = D['S']; g_mean = D['g_mean']; lam = D['lam']; lam_rel = D['lam_rel']
local = D['local']; dtok = D['dtok']; gs_tok = D['gs_tok']; log_s_tok = D['log_s_tok']
c_abs = D['c_abs']; c_rel = D['c_rel']; h_curve = D['h_curve']; hd = D['hd']
mlp_s = D['mlp_s']; mlp_r = D['mlp_r']; ful_s = D['ful_s']; ful_r = D['ful_r']; ang = D['ang']
ntok = D['ntok']; sel_layers = D['sel_layers']; full_layers = D['full_layers']
log_s_agg = D['log_s_agg']; h_rel_maxexp = D['h_rel_maxexp']
fsle_curves = {float(k): np.asarray(v) for k, v in RES['fsle_curves'].items()}

nt, nL, k = lam.shape
lay = np.arange(nL)                    # block index (product through block l)
TOK3 = [0, 100, 143]                   # individual tokens for line plots
C3 = ['#1f77b4', '#d62728', '#2ca02c']
COL = {t: c for t, c in zip(TOK3, C3)}

plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.grid': True,
                     'grid.alpha': 0.25, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})


def band(x, axis=0):
    return np.median(x, axis=axis), *np.quantile(x, (.25, .75), axis=axis)


# ================================================================ fig 01: depth growth
fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
ax = axs[0, 0]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, np.exp(S[i, :, 0]), lw=1.4, color=COL[t], label=f'token {t}')
m, lo, hi = band(np.exp(S[:, :, 0]))
ax.plot(lay, m, color='k', lw=1.2, label='median (8 tokens)')
ax.fill_between(lay, lo, hi, color='k', alpha=.12, lw=0)
ax.set_yscale('log')
ax.set_xlabel('product through block $\\ell$')
ax.set_ylabel('$\\sigma_1$ (top singular value)')
ax.set_title('(a) product top singular value, absolute')
ax.legend(fontsize=7, frameon=False)
ax.annotate('block 0: $\\times$10-25 in one step', xy=(0.3, 12), fontsize=7.5,
            xycoords='data', color='0.35')

ax = axs[0, 1]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, lam[i, :, 0], lw=1.1, color=COL[t])
    A = np.stack([1 / (lay + 1), np.ones_like(lay)], 1)
    coef, *_ = np.linalg.lstsq(A, lam[i, :, 0], rcond=None)
    ax.plot(lay, A @ coef, ls='--', lw=1, color=COL[t], alpha=.65,
            label='token {} (tail $\\lambda_1$={:+.3f})'.format(
                t, (S[i, -1, 0] - S[i, 0, 0]) / (nL - 1)))
m, lo, hi = band(lam[:, :, 0])
ax.plot(lay, m, color='k', lw=1.1, label='median')
ax.fill_between(lay, lo, hi, color='k', alpha=.12, lw=0)
ax.set_xlabel('product through block $\\ell$')
ax.set_ylabel('$\\lambda_1 = \\log\\sigma_1/(\\ell{+}1)$  [nats/layer]')
ax.set_title('(b) finite-time exponent: transient + sustained fit')
ax.legend(fontsize=6.6, frameon=False, loc='upper right')
ax.text(.03, .04, 'dashed: $A/(\\ell{+}1)+c$ fit; label quotes the direct tail\n'
        '= mean log gain over blocks 1-27',
        transform=ax.transAxes, fontsize=7.5, color='0.35')

ax = axs[1, 0]
for i, t in enumerate(dtok):
    emphasize = t in TOK3
    ax.plot(np.arange(1, k + 1), lam[i, -1], marker='o', ms=3,
            lw=1.2 if emphasize else .7,
            color=COL.get(int(t), '0.75'), alpha=1 if emphasize else .55,
            label=f't={t}' if emphasize else None, zorder=3 if emphasize else 1)
ax.set_xlabel('exponent index $i$ (top 8)')
ax.set_ylabel('$\\lambda_i$ at depth 28')
ax.set_title('(c) final depth spectrum: near-degenerate band')
ax.legend(fontsize=7, frameon=False)

ax = axs[1, 1]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, g_mean[i], lw=1.1, color=COL[t], label=f'token {t}')
m, lo, hi = band(g_mean)
ax.plot(lay, m, color='k', lw=1.2, label='median')
ax.fill_between(lay, lo, hi, color='k', alpha=.12, lw=0)
ax.axhline(0, color='0.5', lw=.8)
ax.set_ylim(-.16, .24)
ax.set_xlabel('block $\\ell$')
ax.set_ylabel('log gain along propagated basis\n[mean top-8, nats/layer]')
ax.set_title('(d) per-layer aligned log gain (block 0 off-scale)')
ax.legend(fontsize=7, frameon=False)
ax.annotate('block 0: +{:.1f} to +{:.1f} nats (off scale)'.format(
    g_mean[:, 0].min(), g_mean[:, 0].max()), xy=(1, .20), fontsize=7.5, color='0.35')
ax.annotate('late rise', xy=(24, .10), fontsize=7.5, color='0.35')
fig.suptitle('Depth spectrum (diagonal-block tangent product, DeepSeek-1.5B, 28 blocks, k=8)',
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig01_depth_growth.png'); plt.close(fig)

# ================================================================ fig 02: scale normalization
fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
ax = axs[0, 0]
m_al, lo_al, hi_al = band(np.exp(g_mean))
m_sc, lo_sc, hi_sc = band(gs_tok)
ax.plot(lay, m_sc, color='#8c564b', lw=1.6, label='state-scale ratio $\\|h_{\\ell+1}\\|/\\|h_\\ell\\|$')
ax.fill_between(lay, lo_sc, hi_sc, color='#8c564b', alpha=.12, lw=0)
ax.plot(lay, m_al, color='#1f77b4', lw=1.6, label='aligned gain (propagated basis)')
ax.fill_between(lay, lo_al, hi_al, color='#1f77b4', alpha=.12, lw=0)
ax.set_yscale('log')
ax.set_ylim(.8, 52)
ax.set_xlabel('block $\\ell$')
ax.set_title('(a) absolute amplification vs. state-scale growth')
ax.legend(fontsize=7, frameon=False)
ax.annotate('block 0: scale $\\times$30, gain $\\times$17', xy=(1.2, 42), fontsize=7.5,
            color='0.35')

ax = axs[0, 1]
for i, t in enumerate(dtok):
    ax.plot(lay, np.exp(g_mean[i]) / gs_tok[i], lw=.8, color='0.8', alpha=.8)
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, np.exp(g_mean[i]) / gs_tok[i], lw=1.4, color=COL[t], label=f'token {t}')
m = np.median(np.exp(g_mean) / gs_tok, axis=0)
ax.plot(lay, m, color='k', lw=1.6, label='median')
ax.axhline(1, color='0.4', ls='--', lw=.9)
ax.set_yscale('log'); ax.set_ylim(.35, 1.3)
ax.set_xlabel('block $\\ell$')
ax.set_ylabel('relative gain $\\gamma_\\ell$')
ax.set_title('(b) scale-normalized gain: below 1 at every block')
ax.legend(fontsize=7, frameon=False)

ax = axs[1, 0]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, np.exp(S[i, :, 0]), lw=1.2, color=COL[t])
    rel = np.exp(S[i, :, 0] + log_s_tok[i, 0] - log_s_tok[i, 1:nL + 1])
    ax.plot(lay, rel, lw=1.2, ls='--', color=COL[t])
m, lo, hi = band(np.exp(S[:, :, 0]))
ax.plot(lay, m, color='k', lw=1.1)
ax.fill_between(lay, lo, hi, color='k', alpha=.1, lw=0)
m, lo, hi = band(np.exp(S[:, :, 0] + log_s_tok[:, 0:1] - log_s_tok[:, 1:nL + 1]))
ax.plot(lay, m, color='0.35', lw=1.1, ls='--')
ax.fill_between(lay, lo, hi, color='0.35', alpha=.1, lw=0)
ax.set_yscale('log')
ax.set_xlabel('product through block $\\ell$')
ax.set_ylabel('$\\sigma_1$')
ax.set_title('(c) product: absolute (solid) vs scale-normalized (dashed)')
ax.text(.03, .05, 'absolute final: $\\times$10-166\nrelative final: $\\times$0.02-0.33',
        transform=ax.transAxes, fontsize=8, color='0.35')

ax = axs[1, 1]
scale_ratio = np.exp(log_s_agg[:nL] - log_s_agg[0])
for r, c in fsle_curves.items():
    ax.plot(lay, c / c[0], lw=.9, alpha=.8, label=f'$r_0$={r:g}')
mfs = np.median([c / c[0] for c in fsle_curves.values()], axis=0)
ax.plot(lay, mfs, color='k', lw=1.6, label='median')
rel = mfs / scale_ratio
ax.plot(lay, rel, color='0.35', lw=1.6, ls='--', label='scale-normalized')
ax.set_yscale('log')
ax.set_xlabel('recorded layer $\\ell$ (OLD runs)')
ax.set_ylabel('$d(\\ell)/d(0)$')
ax.set_title('(d) depth FSLE: absolute vs scale-normalized')
ax.legend(fontsize=7, frameon=False, ncol=2)
ax.annotate('$\\times$206 absolute $\\to$ $\\times$0.46 relative', xy=(6, 1.6),
            fontsize=7.5, color='0.35')
fig.suptitle('The expansion is the residual-stream scale drift '
             '($\\|h\\|$: 1.2 $\\to$ 540 over depth)', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig02_scale_normalization.png'); plt.close(fig)

# ================================================================ fig 03: heatmaps
fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.2))
vmax = float(np.quantile(np.abs(g_mean[:, 1:]), .97))
im = axs[0].imshow(g_mean[:, 1:], aspect='auto', cmap='RdBu_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                   interpolation='nearest')
axs[0].set_yticks(range(nt)); axs[0].set_yticklabels([f't={int(t)}' for t in dtok])
xt = np.arange(0, 27, 4)
axs[0].set_xticks(xt); axs[0].set_xticklabels([str(x + 1) for x in xt])
axs[0].set_xlabel('block $\\ell$ (1..27)')
axs[0].set_title('(a) aligned log gain [nats/layer], block 0 excluded')
axs[0].grid(False)
cb = fig.colorbar(im, ax=axs[0], shrink=.85); cb.set_label('nats/layer')
axs[0].text(.02, .97, 'block 0: +{:.1f} to +{:.1f} nats (off scale);\nextremes $\\pm${:.2f}'.format(
    g_mean[:, 0].min(), g_mean[:, 0].max(), np.abs(g_mean[:, 1:]).max()),
    transform=axs[0].transAxes, fontsize=6.6, color='0.30', va='top')

g_rel_mean = g_mean - np.log(gs_tok)
vminr = float(np.quantile(g_rel_mean[:, 1:], .03))
im = axs[1].imshow(g_rel_mean[:, 1:], aspect='auto', cmap='Blues_r',
                   vmin=vminr, vmax=0, interpolation='nearest')
axs[1].set_yticks(range(nt)); axs[1].set_yticklabels([f't={int(t)}' for t in dtok])
axs[1].set_xticks(xt); axs[1].set_xticklabels([str(x + 1) for x in xt])
axs[1].set_xlabel('block $\\ell$ (1..27)')
axs[1].set_title('(b) scale-normalized (relative) log gain')
axs[1].grid(False)
cb = fig.colorbar(im, ax=axs[1], shrink=.85); cb.set_label('nats/layer')
axs[1].text(.02, .97, 'block 0 relative: {:.2f} to {:.2f} nats'.format(
    g_rel_mean[:, 0].min(), g_rel_mean[:, 0].max()),
    transform=axs[1].transAxes, fontsize=6.6, color='0.30', va='top')
fig.suptitle('Per-block gain structure: mild late-depth rise (a), uniform relative '
             'contraction (b)', fontsize=10)
fig.tight_layout()
fig.savefig(f'{OUT}/fig03_gain_heatmaps.png'); plt.close(fig)

# ================================================================ fig 04: non-normality
fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9))
ax = axs[0]
cmap = plt.cm.viridis
for l in sel_layers:
    rows = mlp_s[:, l] > 0
    ax.scatter(mlp_r[rows, l], mlp_s[rows, l], s=26, marker='o', facecolor='none',
               edgecolor=cmap(l / 27), lw=1.3,
               label='MLP' if l == 0 else None, zorder=3)
for l in full_layers:
    rows = ful_s[:, l] == ful_s[:, l]
    ax.scatter(ful_r[rows, l], ful_s[rows, l], s=30, marker='s',
               color=cmap(l / 27), label='full block' if l == 0 else None, zorder=4)
i143 = list(ntok).index(143)
ax.scatter([ful_r[i143, 14]], [ful_s[i143, 14]], s=120, facecolor='none',
           edgecolor='red', lw=1.6, zorder=5)
ax.annotate('t=143, L14', xy=(ful_r[i143, 14], ful_s[i143, 14]),
            xytext=(2.0, 15), fontsize=7.5, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=.8))
for kk, ls in [(1, ':'), (4, '--'), (16, '-.')]:
    xx = np.logspace(0, np.log10(60), 40)
    ax.plot(xx, kk * xx, ls, color='0.6', lw=.8, zorder=1)
    ax.annotate(f'$\\kappa$={kk}', xy=(26, kk * 26), fontsize=6.5, color='0.5')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1, 60); ax.set_ylim(2, 500)
ax.set_xlabel('spectral radius $\\rho$')
ax.set_ylabel('$\\sigma_{\\max}$')
ax.set_title('(a) components amplify; full block tames MLP')
ax.legend(fontsize=7.5, frameon=False, loc='upper left')

ax = axs[1]
for i, t in enumerate(ntok):
    ax.plot(np.arange(27), ang[i], lw=.7, color='0.8')
m, lo, hi = band(ang)
ax.plot(np.arange(27), m, color='k', lw=1.6, label='median')
ax.fill_between(np.arange(27), lo, hi, color='k', alpha=.15, lw=0)
ax.axhline(90, color='0.5', ls=':', lw=1)
ax.annotate('90° (orthogonal)', xy=(1, 90.4), fontsize=7, color='0.5')
ax.axhline(90 - np.degrees(1 / np.sqrt(1536)), color='#d62728', ls='--', lw=.9)
ax.annotate('random pair in $\\mathbb{R}^{1536}$ ($88.5°$)', xy=(9.5, 86.3),
            fontsize=7, color='#d62728')
ax.annotate('min 54.5° (t=16, L25)', xy=(25, 54.5), xytext=(15.5, 58),
            fontsize=7, color='0.4', arrowprops=dict(arrowstyle='->', lw=.7, color='0.4'))
ax.set_xlabel('block $\\ell$')
ax.set_ylabel('angle$(v_\\ell, v_{\\ell+1})$ [deg]')
ax.set_title('(b) top amplification directions rotate ~84°/block')
ax.set_ylim(50, 93)
ax.legend(fontsize=7, frameon=False)

ax = axs[2]
ax.hist(ang.ravel(), bins=np.arange(52, 92, 2), color='#1f77b4', alpha=.75)
ax.axvline(np.median(ang), color='k', lw=1.4)
ax.annotate('median 83.6°', xy=(83.6, 30), fontsize=7.5, ha='right')
ax.axvline(90, color='0.5', ls=':', lw=1)
ax.set_xlabel('angle$(v_\\ell, v_{\\ell+1})$ [deg]')
ax.set_ylabel('count (6 tokens $\\times$ 27 pairs)')
ax.set_title('(c) angle distribution: near-orthogonal, wider than random')
fig.suptitle('Non-normality: $\\sigma_{\\max}\\gg\\rho$, expanding directions '
             'rotate every block', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig04_nonnorm.png'); plt.close(fig)

# ================================================================ fig 05: gain budget
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
ax = axs[0]
groups = ['anchor L14\n(blocks 14-20)', 'anchor L27\n(blocks 21-27)']
pred = [RES['nonnorm']['misalign_pred_gain_L14'], RES['nonnorm']['misalign_pred_gain_L27']]
meas = [RES['nonnorm']['measured_gain_blocks14_21'], RES['nonnorm']['measured_gain_blocks21_27']]
comp = [RES['nonnorm']['budget']['14']['log_smax'], RES['nonnorm']['budget']['27']['log_smax']]
scal = [RES['nonnorm']['budget']['14']['log_scale'], RES['nonnorm']['budget']['27']['log_scale']]
x = np.arange(2); w = .2
ax.bar(x - 1.5 * w, comp, w, color='#d62728', label='component $\\log\\sigma_{\\max}$')
ax.bar(x - .5 * w, pred, w, color='#ff9896', label='misalignment model')
ax.bar(x + .5 * w, meas, w, color='#1f77b4', label='measured aligned gain')
ax.bar(x + 1.5 * w, scal, w, color='#8c564b', label='state-scale growth')
for xi, (a, b, c_, d) in zip(x, zip(comp, pred, meas, scal)):
    for off, v in zip([-1.5, -.5, .5, 1.5], [a, b, c_, d]):
        ax.annotate('{:.3f}'.format(v), xy=(xi + off * w, v + .012), fontsize=6.6,
                    ha='center', rotation=90)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel('nats / layer')
ax.set_title('(a) per-layer gain budget at measured full-block layers')
ax.legend(fontsize=6.8, frameon=False)
ax.set_ylim(0, 2.5)

ax = axs[1]
nn = RES['nonnorm']
names = ['naive chain\n$\\sigma_{\\max}$ (27 blocks)', '$\\rho$-chain\n(eigenvalues)',
         'measured\naligned product', 'FSLE\n(random pert.)', 'state-scale\ngrowth']
vals = [nn['naive_chain_full'], nn['naive_chain_rho'],
        nn['measured_product_top1_median'], nn['fsle_growth_median'],
        nn['scale_total_median_dup']]
ax.bar(np.arange(5), vals, color=['#d62728', '#ff9896', '#1f77b4', '#9467bd', '#8c564b'])
for i, v in enumerate(vals):
    lbl = '{:.1e}'.format(v) if v > 1e4 else '$\\times${:.1f}'.format(v)
    ax.annotate(lbl, xy=(i, v * 1.3), fontsize=7.5, ha='center')
ax.set_yscale('log'); ax.set_ylim(1, 1e17)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('total growth over 28 blocks')
ax.set_title('(b) composition at depth 28: measured $\\sim$ scale $\\ll$ bounds')
ax.text(.03, .04, 'MLP alone would give $\\sigma_{\\max}^{27}\\sim10^{50}$ at L27',
        transform=ax.transAxes, fontsize=7.5, color='0.35')
fig.suptitle('Components amplify, composition does not: quantified budget', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig05_gain_budget.png'); plt.close(fig)

# ================================================================ fig 06: entropy
fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.9))
ax = axs[0]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, h_curve[i], lw=1.2, color=COL[t], label=f'token {t}')
m, lo, hi = band(h_curve)
ax.plot(lay, m, color='k', lw=1.4, label='median')
ax.fill_between(lay, lo, hi, color='k', alpha=.12, lw=0)
hA = float(np.median(h_curve[:, 0]))
ax.plot(lay, hA / (lay + 1), ls='--', color='0.55', lw=1)
ax.annotate('pure one-block transient\n$\\propto 1/(\\ell{+}1)$', xy=(14, hA / 15),
            xytext=(9, 8), fontsize=7.5, color='0.4',
            arrowprops=dict(arrowstyle='->', lw=.7, color='0.4'))
ax.set_yscale('log')
ax.set_xlabel('product through block $\\ell$')
ax.set_ylabel('$h(\\ell)=\\sum_i\\max(\\lambda_i,0)$  [top-8 LB]')
ax.set_title('(a) expansion entropy decays $\\sim A/\\ell$')
ax.legend(fontsize=7, frameon=False)

ax = axs[1]
for t in TOK3:
    i = list(dtok).index(t)
    ax.plot(lay, h_rel_maxexp[i], lw=1.2, color=COL[t], label=f'token {t}')
m, lo, hi = band(h_rel_maxexp)
ax.plot(lay, m, color='k', lw=1.4, label='median')
ax.fill_between(lay, lo, hi, color='k', alpha=.12, lw=0)
ax.axhline(0, color='0.4', lw=.9)
ax.set_xlabel('product through block $\\ell$')
ax.set_ylabel('$\\max_i\\,\\lambda_i^{\\mathrm{rel}}$  [nats/layer]')
ax.set_title('(b) largest scale-normalized exponent: always negative')
ax.legend(fontsize=7, frameon=False)
ax.annotate('no top-8 direction expands\nrelative to the stream scale',
            xy=(8, -.05), fontsize=7.5, color='0.35')

ax = axs[2]
x = np.arange(nt); w = .38
ax.bar(x - w / 2, hd, w, color='#1f77b4', label='$h_{\\mathrm{depth}}$ (absolute, top-8 LB)')
ax.bar(x + w / 2, np.minimum(lam_rel[:, -1, :], 0).sum(axis=1), w,
       color='#8c564b', label='$\\sum_i\\lambda_i^{\\mathrm{rel}}$ (scale-normalized)')
ax.axhline(0, color='0.4', lw=.9)
ax.set_xticks(x); ax.set_xticklabels([f'{int(t)}' for t in dtok], fontsize=7)
ax.set_xlabel('token position')
ax.set_ylabel('nats/layer')
ax.set_title('(c) per token: positive absolute, negative relative')
ax.legend(fontsize=7, frameon=False)
fig.suptitle('Expansion entropy of the depth map is a scale artifact', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig06_entropy.png'); plt.close(fig)

# ================================================================ fig 07: toy mechanism
def toy_chain(sigma, theta_deg, L=27, d=64, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=d); a /= np.linalg.norm(a)
    x = rng.normal(size=d); x /= np.linalg.norm(x)   # generic start
    th = np.radians(theta_deg)
    out = [1.0]
    for _ in range(L):
        w = rng.normal(size=d); w -= (w @ a) * a; w /= np.linalg.norm(w)
        a = np.cos(th) * a + np.sin(th) * w
        x = x + (sigma - 1) * a * (a @ x)
        out.append(np.linalg.norm(x))
    return np.asarray(out)


fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
ax = axs[0]
Ltoy = 27
for th, c_, lab in [(0, '#d62728', '0° (aligned)'), (45, '#ff7f0e', '45°'),
                    (70, '#9467bd', '70°'), (84, '#1f77b4', '84° (measured median)'),
                    (88.6, '#7f7f7f', '88.6° (random, $\\mathbb{R}^{1536}$)')]:
    g = toy_chain(3.5, th, L=Ltoy)
    ax.plot(np.arange(Ltoy + 1), g, lw=1.3, color=c_, label=lab)
m_post = np.exp(S[:, :, 0] - S[:, 0:1, 0])
m, lo, hi = band(m_post)
ax.plot(np.arange(nL), m, color='k', lw=1.9, label='measured (median, post block 0)')
ax.fill_between(np.arange(nL), lo, hi, color='k', alpha=.12, lw=0)
ax.set_yscale('log'); ax.set_ylim(.5, 1e7)
ax.set_xlabel('block $\\ell$ (after block 0)')
ax.set_ylabel('growth relative to end of block 0')
ax.set_title('(a) rank-one amplifier chain, $\\sigma$=3.5')
ax.legend(fontsize=6.8, frameon=False, loc='upper left')
ax.annotate('aligned: $3.5^{27}\\sim10^{15}$ (off scale)', xy=(10, 2e5), fontsize=7,
            color='#d62728')

ax = axs[1]
cos2 = np.cos(np.radians(ang)) ** 2
pred_hist = np.log1p((3.5 - 1) * cos2.ravel())
meas_hist = g_mean[:, 1:].ravel()
bins = np.linspace(-.13, .35, 33)
ax.hist(pred_hist, bins=bins, density=True, alpha=.65, color='#ff9896',
        label='misalignment model\n$\\ln(1+(\\sigma{-}1)\\cos^2\\theta_{\\mathrm{meas}})$')
ax.hist(meas_hist, bins=bins, density=True, alpha=.65, color='#1f77b4',
        label='measured aligned log gain\n(blocks 1-27, 8 tokens)')
ax.axvline(0, color='0.5', lw=.8)
ax.set_xlabel('per-layer log gain [nats]')
ax.set_ylabel('density')
ax.set_title('(b) the angle distribution predicts the gain distribution')
ax.legend(fontsize=7, frameon=False)
ax.annotate('model tail: 0.61 at the\nsingle 54.5° pair (t=16, L25)', xy=(.36, 1.2),
            fontsize=6.8, color='0.35', ha='right')
fig.suptitle('Cancellation mechanism: amplifying directions rotate ~84° per block, '
             'so $(\\sigma{-}1)\\cos^2\\theta$ is all that chains', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig07_toy_mechanism.png'); plt.close(fig)

# ================================================================ fig 08: cross-position context
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.0))
la = np.array([0, 1, 2, 3, 7, 14, 21, 27])
ctx = RES['context_response_r0.03']
for t in ['0', '16', '50', '100', '143']:
    a = np.asarray(ctx[t]['abs'])
    axs[0].plot(la, a, marker='o', ms=3.5, lw=1.2, label=f't={t}')
    r_ = np.asarray(ctx[t]['rel'])
    axs[1].plot(la, r_, marker='o', ms=3.5, lw=1.2, label=f't={t}')
axs[0].set_yscale('log')
axs[0].set_xlabel('layer $\\ell$ (sampled)')
axs[0].set_ylabel('$\\|\\delta h^{(\\ell)}\\|$ at fixed position')
axs[0].set_title('(a) prompt perturbation $\\to$ downstream position ($r$=0.03)')
axs[0].annotate('grows to $O(r)$ by L27\n(attention cone, not diagonal scale)',
               xy=(2.2, 1.5e-3), fontsize=7.5, color='0.35')
axs[0].legend(fontsize=7, frameon=False, ncol=2)
axs[1].set_yscale('log')
axs[1].set_xlabel('layer $\\ell$ (sampled)')
axs[1].set_ylabel('$\\|\\delta h\\|/\\|h\\|$')
axs[1].set_title('(b) scale-normalized: marginal, token-dependent')
axs[1].legend(fontsize=7, frameon=False, ncol=2)
fig.suptitle('Context (A1 run, teacher-forced): the output-relevant channel is '
             'cross-position, and it is marginal', fontsize=10.5)
fig.tight_layout()
fig.savefig(f'{OUT}/fig08_context_crossposition.png'); plt.close(fig)

print('figures written to', OUT)
