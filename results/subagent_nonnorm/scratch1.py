import numpy as np
from scipy import stats
np.set_printoptions(precision=3, suppress=True, linewidth=200)

ROOT = r'chaos_llm_2026/results/local_data'
dz = np.load(f'{ROOT}/depth/depth.npz')
lam, local, hd = dz['lam'], dz['local_gain'], dz['h_depth']
toks, lay = dz['tokens'], dz['layers']
nt, nL, k = lam.shape

da = np.load(f'{ROOT}/w100_layers/a1.npz')  # context only
layers_a = np.asarray(da['layers']); norm_h0 = da['norm_h0']; margins = da['margins']

# cumulative log SVs
S = lam * (np.arange(nL) + 1)[None, :, None]   # [nt, nL, k]
g_inc = np.diff(np.concatenate([np.zeros((nt, 1, k)), S], axis=1), axis=1)  # [nt,nL,k]
g_mean = g_inc.mean(axis=2)
g_top1 = g_inc[:, :, 0]

# state norm profile per depth token position (interp in log space)
def log_interp_norm(t):
    y = np.log(norm_h0[:, t].astype(np.float64))
    return np.interp(np.arange(29), layers_a, y)

log_s = np.stack([log_interp_norm(int(t)) for t in toks])   # [nt, 29]
gs = np.exp(np.diff(log_s, axis=1))                         # per-block scale ratio [nt,28]

print('=== scale ratios (token 0):')
print('block0 x{:.2f}, blocks1-27 mean x{:.3f}'.format(gs[0, 0], gs[0, 1:].mean()))
print('total scale growth x{:.1f}'.format(np.exp(log_s[0, -1] - log_s[0, 0])))

log_gs_cum = np.concatenate([np.zeros((nt, 1)), np.cumsum(np.log(gs), axis=1)], axis=1)
scale_rate = (log_gs_cum[:, 1:nL + 1] / (np.arange(nL) + 1)[None, :])[:, :, None]  # [nt,nL,1]
lam_rel = lam - scale_rate
print('\n=== absolute final top-8 exponents (token 0):', lam[0, -1])
print('=== relative  final top-8 exponents (token 0):', lam_rel[0, -1])
print('max rel exponent over all:', lam_rel.max())
print('per-token max rel exponent:', lam_rel.reshape(nt, -1).max(axis=1))

c = (S[:, -1, :] - S[:, 0, :]) / (nL - 1)
print('\n=== sustained tail exponents (mean top-8 per token):')
print('abs:', c.mean(axis=1))
c_rel = c - np.log(gs[:, 1:]).mean(axis=1)[:, None]
print('rel:', c_rel.mean(axis=1))
print('transient share S(0)/S(27) top-1 per token:', (S[:, 0, 0] / S[:, -1, 0]))

# radial/tangential split of the a1 response at large r
radii = np.asarray(da['radii'])
ri = list(radii).index(2.0)
dhn = da['dh_norm'][ri]
csl = da['cos'][ri]
s_l = norm_h0.astype(np.float64)
d_perp = s_l[None] * np.sqrt(2 * np.maximum(0, 1 - csl))
frac_tan2 = d_perp ** 2 / np.maximum(dhn ** 2, 1e-30)
print('\n=== tangential fraction per layer (r=2, median):')
print(np.sqrt(np.median(frac_tan2, axis=(0, 2))))
print('||dh|| median per layer (r=2):', np.median(dhn, axis=(0, 2)))
print('state angle deviation per layer (r=2):', np.median(np.sqrt(2 * np.maximum(0, 1 - csl)), axis=(0, 2)))

dn = np.load(f'{ROOT}/nonnorm.npz')
ang = dn['angles']
print('\n=== angles: median {:.1f}, IQR {:.1f}-{:.1f}, min {:.1f}, max {:.1f}'.format(
    np.median(ang), *np.percentile(ang, [25, 75]), ang.min(), ang.max()))
med = np.median(ang, axis=0)
sl_, icpt, rval, pval, _ = stats.linregress(np.arange(27), med)
print('per-layer median angle trend: slope {:.3f} deg/layer, p={:.4f}'.format(sl_, pval))
print('random-vector angle sd in R^1536: {:.2f} deg'.format(np.degrees(1 / np.sqrt(1536))))

shared = [16, 50, 123, 143, 200]
rows = []
for t in shared:
    ti_nn = list(np.asarray(dn['tokens'])).index(t)
    ti_d = list(np.asarray(toks)).index(t)
    for l in range(27):
        rows.append((l, ang[ti_nn, l], local[ti_d, l + 1]))
rows = np.array(rows)
r_, p_ = stats.spearmanr(np.cos(np.radians(rows[:, 1])) ** 2, rows[:, 2])
print('spearman cos^2(angle) vs next aligned gain: rho={:.3f} p={:.2g} (n={})'.format(r_, p_, len(rows)))

m_at = margins[toks]
print('\nmargins at depth tokens:', m_at)
r_, p_ = stats.spearmanr(hd, m_at)
print('spearman h_depth vs margin: {:.3f} p={:.3f}'.format(r_, p_))
r_, p_ = stats.spearmanr(hd, np.exp(log_s[:, -1] - log_s[:, 0]))
print('spearman h_depth vs total scale growth: {:.3f} p={:.3f}'.format(r_, p_))

print('\nmean aligned gain by layer band:')
for a, b in [(1, 7), (7, 14), (14, 21), (21, 27)]:
    print('  blocks {}-{}: {:.3f}'.format(a, b, local[:, a:b].mean()))
print('local_gain block0 per token:', local[:, 0])
print('\nfinal spectra lam[t,-1]:')
for i, t in enumerate(toks):
    print(' t={}: {}'.format(t, lam[i, -1].round(3)))
