import numpy as np, os
base = r'C:\Users\hjaca\research\chaos_llm_2026\results\local_data'
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def show(name, path):
    print('=' * 78)
    print(name, path)
    d = np.load(path, allow_pickle=True)
    for k in d.files:
        a = d[k]
        if a.ndim == 0:
            print(f'  {k}: scalar = {a}')
        else:
            print(f'  {k}: shape {a.shape} dtype {a.dtype} min {np.nanmin(a):.4g} max {np.nanmax(a):.4g}')
    return d

d1 = show('EMB', base + r'\state_deep_emb\state2.npz')
dh, cosv, dm = d1['dh'], d1['cos'], d1['dm']
steps, radii, layers, margins = d1['steps'], d1['radii'], d1['layers'], d1['margins']
print('steps', steps, 'radii', radii, 'layers', layers, 'inject_layer', d1['inject_layer'])
print('margins[16] =', margins[16])
sm = np.argsort(margins)[:12]
print('12 smallest margins at t =', sm)
print('                 values  =', margins[sm])

print('\n-- causality: max dh at t < s (should be ~0) --')
for i_s, s in enumerate(steps):
    print(f'  s={s:3d}: max dh(t<s) = {dh[:, i_s, :, :, :s].max():.3e}')

print('\n-- immediate response dh(t=s), r=0.1, per layer, mean over 24 dirs --')
for i_s, s in enumerate(steps):
    row = [f'L{layers[i_l]}: {dh[3, i_s, :, i_l, s].mean():.4f}' for i_l in range(4)]
    v27 = dh[3, i_s, :, 3, s]
    print(f'  s={s:3d} ' + '  '.join(row) + f'   cv(dirs,L27)={v27.std()/v27.mean():.3f}')

print('\n-- linearity in r: ratio dh(0.1)/dh(0.03) over t>=s, s=32, L27 --')
for i_s, s in enumerate(steps):
    a = dh[3, i_s, :, 3, s:]
    b = dh[2, i_s, :, 3, s:]
    rat = a / np.maximum(b, 1e-15)
    print(f'  s={s:3d}: ratio pct[1,50,99] = {np.percentile(rat,[1,50,99])}  (expect 3.33)')

print('\n-- tail magnitude dh(t=299), r=0.1, L27, mean over dirs --')
for i_s, s in enumerate(steps):
    print(f'  s={s:3d}: dh(t=299) = {dh[3, i_s, :, 3, 299].mean():.3e}')

print('\n-- smallest-radius tail (r=0.003, t=299) -> noise floor check --')
for i_s, s in enumerate(steps[:4]):
    print(f'  s={s:3d}: dh(t=299,r=0.003) mean = {dh[0, i_s, :, 3, 299].mean():.3e}')

print('\n-- cos sanity: 1-cos at t=s and t=299, r=0.1, L27 --')
for i_s, s in enumerate(steps):
    print(f'  s={s:3d}: 1-cos(tau=0) = {1 - cosv[3, i_s, :, 3, s].mean():.3e}   1-cos(t=299) = {1 - cosv[3, i_s, :, 3, 299].mean():.3e}')

print('\n-- dm: change of top1-top2 gap; how many dirs cross the baseline margin --')
cross_total = {}
for i_s, s in enumerate(steps):
    adm = np.abs(dm[3, i_s])          # [24, 300]
    crossed = adm[:, s + 1:] > margins[None, s + 1:]
    n_dirs = crossed.any(axis=1)
    if n_dirs.any():
        firsts = [s + 1 + np.argmax(crossed[i]) for i in np.where(n_dirs)[0]]
        print(f'  s={s:3d}: {n_dirs.sum()}/24 dirs flip gap somewhere; first-cross t = {firsts}')
    else:
        print(f'  s={s:3d}: 0/24 dirs flip gap')

d2 = show('FREERUN', base + r'\state_freerun\sf.npz')
pred, ref = d2['pred'], d2['ref']
print('pred==-1 count total:', (pred == -1).sum(), ' per step:', (pred == -1).sum(axis=(1, 2)))
for i_s in range(2):
    print(f'  freerun block {i_s}: pred[:, :, :16] == ref[:16] for all? ',
          bool((pred[i_s, :, :16] == ref[None, :16]).all()))
for i_s in range(2):
    for idir in range(24):
        p = pred[i_s, idir]
        gen = p != -1
        diff = (p != ref) & gen
        if diff.any():
            t0 = int(np.argmax(diff))
            print(f'  s={[16,50][i_s]} dir={idir:2d}: FIRST CHANGE t={t0} (latency {t0-[16,50][i_s]}), pred={p[t0]} ref={ref[t0]}, gen tokens={gen.sum()}')
n_nochange = 0
for i_s in range(2):
    for idir in range(24):
        p = pred[i_s, idir]
        diff = (p != ref) & (p != -1)
        if not diff.any():
            n_nochange += 1
print('  runs with no token change anywhere:', n_nochange, '/48')

d3 = show('L14-INJECT', base + r'\state_deep_L14\state2.npz')
dh3 = d3['dh']
print('L14 steps', d3['steps'], 'radii', d3['radii'], 'layers', d3['layers'], 'inject_layer', d3['inject_layer'])
for i_s, s in enumerate(d3['steps']):
    print(f'  s={s:3d}: max dh(t<s) = {dh3[:, i_s, :, :, :s].max():.3e}')
for i_s, s in enumerate(d3['steps']):
    row = [f'L{d3["layers"][i_l]}: {dh3[1, i_s, :, i_l, s].mean():.4f}' for i_l in range(dh3.shape[3])]
    print(f'  s={s:3d} r=0.03 ' + '  '.join(row))

try:
    d4 = show('OLD state.npz', base + r'\state.npz')
except Exception as e:
    print('old state.npz skipped:', e)
