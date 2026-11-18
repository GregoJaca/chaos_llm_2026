import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

ROOT = r'chaos_llm_2026/results/local_data'
da = np.load(f'{ROOT}/w100_layers/a1.npz')
layers_a = np.asarray(da['layers']); norm_h0 = da['norm_h0'].astype(np.float64)
radii = np.asarray(da['radii']); dh = da['dh_norm']

s_med = np.median(norm_h0, axis=1)   # median state norm per sampled layer
print('median state norm per layer:', s_med)

for ri in [0, 3]:  # r=0.001, 0.1
    r = radii[ri]
    prof = dh[ri]  # [20, 8, 300]
    med_t = np.median(prof, axis=(0, 2))
    print('\n=== r={} : median dh over dirs,t per layer: {}'.format(r, med_t))
    for t in [0, 16, 50, 100, 143, 250]:
        p = np.median(prof[:, :, t], axis=0)
        print('   t={:3d}: {}'.format(t, p), ' rel:', (p / s_med).round(6))

# growth factors L2 -> L27 at fixed t, linear regime r=1e-3
print('\ngrowth factor dh(L27)/dh(L2) per t (r=1e-3), median over dirs:')
p = dh[0]
gf = np.median(p[:, 7, :] / np.maximum(p[:, 2, :], 1e-30), axis=0)
print('  t=0: {:.1f}  t=16: {:.1f}  t=50: {:.1f}  t=100: {:.1f}  t=143: {:.1f}  t=250: {:.1f}'.format(
    *[gf[x] for x in [0, 16, 50, 100, 143, 250]]))
print('  median over all t: {:.1f}'.format(np.median(gf)))
print('  scale growth L2->L27: x{:.1f}'.format(s_med[7] / s_med[2]))

# per-token band gains and relative block-0 gains
dz = np.load(f'{ROOT}/depth/depth.npz')
local = dz['local_gain']; toks = dz['tokens']
def log_interp_norm(t):
    return np.interp(np.arange(29), layers_a, np.log(norm_h0[:, t]))
log_s = np.stack([log_interp_norm(int(t)) for t in toks])
gs = np.exp(np.diff(log_s, axis=1))
print('\nblock-0: aligned gain / scale ratio per token:', (local[:, 0] / gs[:, 0]).round(3))
for a, b in [(1, 7), (7, 14), (14, 21), (21, 27)]:
    ag = local[:, a:b].mean(axis=1); sc = gs[:, a:b].mean(axis=1)
    print('blocks {}-{}: aligned {:.3f}  scale {:.3f}  rel {:.3f}'.format(
        a, b, np.median(ag), np.median(sc), np.median(ag / sc)))
